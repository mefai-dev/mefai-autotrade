"""
Order Manager - Complete order lifecycle management for Mefai Autotrade.
Handles order creation, validation, submission, modification, cancellation,
partial fills, timeouts, and thread-safe order book operations.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("mefai.autotrade.order_manager")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderStatus(Enum):
    """All possible states an order can be in."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    def is_active(self) -> bool:
        return self in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )


class OrderType(Enum):
    """Supported order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    OCO = "OCO"


class OrderSide(Enum):
    """Buy or Sell."""
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "OrderSide":
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class TimeInForce(Enum):
    """Time-in-force policies."""
    GTC = "GTC"      # Good til cancelled
    IOC = "IOC"      # Immediate or cancel
    FOK = "FOK"      # Fill or kill
    GTD = "GTD"      # Good til date
    POST_ONLY = "POST_ONLY"  # Maker only


class OrderEvent(Enum):
    """Events that can be fired during order lifecycle."""
    SUBMITTED = auto()
    FILL = auto()
    PARTIAL_FILL = auto()
    CANCEL = auto()
    REJECT = auto()
    EXPIRE = auto()
    MODIFY = auto()
    TIMEOUT = auto()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OrderValidationError(Exception):
    """Raised when order validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class OrderNotFoundError(Exception):
    """Raised when an order cannot be found."""
    pass


class OrderModificationError(Exception):
    """Raised when an order cannot be modified."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SymbolInfo:
    """Exchange symbol trading rules used for order validation."""
    symbol: str
    base_asset: str = ""
    quote_asset: str = ""
    min_qty: float = 0.0
    max_qty: float = 999999999.0
    qty_step: float = 0.001
    min_price: float = 0.0
    max_price: float = 999999999.0
    price_step: float = 0.01
    min_notional: float = 5.0
    max_notional: float = 999999999.0
    max_open_orders: int = 200
    base_precision: int = 8
    quote_precision: int = 8

    def round_qty(self, qty: float) -> float:
        if self.qty_step <= 0:
            return qty
        precision = len(str(self.qty_step).rstrip("0").split(".")[-1])
        return round(
            int(qty / self.qty_step) * self.qty_step, precision
        )

    def round_price(self, price: float) -> float:
        if self.price_step <= 0:
            return price
        precision = len(str(self.price_step).rstrip("0").split(".")[-1])
        return round(
            int(price / self.price_step) * self.price_step, precision
        )


@dataclass
class FillEvent:
    """Represents a single fill (partial or full) on an order."""
    fill_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0
    quantity: float = 0.0
    commission: float = 0.0
    commission_asset: str = "USDT"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_maker: bool = False

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class Order:
    """
    Full order representation with all fields needed for lifecycle tracking.
    """
    # Identity
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    client_order_id: str = ""
    exchange_order_id: str = ""

    # Core fields
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    take_profit: float = 0.0
    trailing_delta: float = 0.0  # For trailing stop (percentage)
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    close_position: bool = False
    leverage: int = 1

    # Status
    status: OrderStatus = OrderStatus.PENDING

    # Timestamps
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Fill tracking
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = "USDT"
    fills: List[FillEvent] = field(default_factory=list)

    # Slippage
    slippage: float = 0.0  # In percentage relative to expected price

    # OCO fields
    oco_order_id: Optional[str] = None
    linked_order_ids: List[str] = field(default_factory=list)

    # Strategy reference
    strategy_id: str = ""
    signal_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    # Error info
    reject_reason: str = ""
    error_code: int = 0

    # Timeout
    timeout_seconds: int = 0  # 0 means no timeout

    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = f"mefai_{self.id}"

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.quantity - self.filled_qty)

    @property
    def fill_ratio(self) -> float:
        if self.quantity <= 0:
            return 0.0
        return self.filled_qty / self.quantity

    @property
    def notional(self) -> float:
        price = self.avg_fill_price if self.avg_fill_price > 0 else self.price
        return price * self.quantity

    @property
    def filled_notional(self) -> float:
        return self.avg_fill_price * self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.status.is_active()

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal()

    @property
    def age_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds()

    @property
    def is_timed_out(self) -> bool:
        if self.timeout_seconds <= 0:
            return False
        return self.age_seconds > self.timeout_seconds

    def apply_fill(self, fill: FillEvent) -> None:
        """Apply a fill event, updating average price and quantities."""
        if self.is_terminal:
            raise OrderModificationError(
                f"Cannot fill terminal order {self.id} (status={self.status})"
            )

        old_total = self.avg_fill_price * self.filled_qty
        new_total = fill.price * fill.quantity
        self.filled_qty += fill.quantity
        if self.filled_qty > 0:
            self.avg_fill_price = (old_total + new_total) / self.filled_qty

        self.commission += fill.commission
        self.fills.append(fill)
        self.updated_at = datetime.now(timezone.utc)

        if self.filled_qty >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now(timezone.utc)
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

        # Calculate slippage against expected price
        if self.price > 0 and self.avg_fill_price > 0:
            if self.side == OrderSide.BUY:
                self.slippage = (
                    (self.avg_fill_price - self.price) / self.price
                ) * 100
            else:
                self.slippage = (
                    (self.price - self.avg_fill_price) / self.price
                ) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "take_profit": self.take_profit,
            "trailing_delta": self.trailing_delta,
            "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only,
            "leverage": self.leverage,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "remaining_qty": self.remaining_qty,
            "fill_ratio": self.fill_ratio,
            "notional": self.notional,
            "strategy_id": self.strategy_id,
            "signal_id": self.signal_id,
            "reject_reason": self.reject_reason,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        order = cls()
        for key, value in data.items():
            if key == "side":
                order.side = OrderSide(value)
            elif key == "order_type":
                order.order_type = OrderType(value)
            elif key == "status":
                order.status = OrderStatus(value)
            elif key == "time_in_force":
                order.time_in_force = TimeInForce(value)
            elif key in ("created_at", "updated_at", "submitted_at",
                         "filled_at", "cancelled_at"):
                if value and isinstance(value, str):
                    setattr(order, key,
                            datetime.fromisoformat(value))
            elif hasattr(order, key):
                setattr(order, key, value)
        return order


# ---------------------------------------------------------------------------
# Order Validator
# ---------------------------------------------------------------------------

class OrderValidator:
    """
    Validates orders against exchange symbol rules before submission.
    """

    def __init__(self):
        self._symbol_info: Dict[str, SymbolInfo] = {}
        self._lock = threading.Lock()

    def update_symbol_info(self, info: SymbolInfo) -> None:
        with self._lock:
            self._symbol_info[info.symbol] = info

    def update_symbol_info_batch(self, infos: List[SymbolInfo]) -> None:
        with self._lock:
            for info in infos:
                self._symbol_info[info.symbol] = info

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        with self._lock:
            return self._symbol_info.get(symbol)

    def validate(
        self,
        order: Order,
        open_order_count: int = 0,
        available_balance: float = 0.0,
    ) -> List[str]:
        """
        Validate an order. Returns list of error messages.
        Empty list means order is valid.
        """
        errors: List[str] = []

        # Basic field checks
        if not order.symbol:
            errors.append("Symbol is required")

        if order.quantity <= 0:
            errors.append("Quantity must be positive")

        if order.order_type in (
            OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT
        ):
            if order.price <= 0:
                errors.append(
                    f"Price is required for {order.order_type.value} orders"
                )

        if order.order_type in (
            OrderType.STOP_MARKET, OrderType.STOP_LIMIT
        ):
            if order.stop_price <= 0:
                errors.append(
                    f"Stop price is required for {order.order_type.value} orders"
                )

        if order.order_type == OrderType.TRAILING_STOP:
            if order.trailing_delta <= 0:
                errors.append("Trailing delta must be positive")
            if order.trailing_delta > 10.0:
                errors.append("Trailing delta exceeds 10% maximum")

        if order.order_type == OrderType.OCO:
            if order.price <= 0:
                errors.append("Limit price is required for OCO orders")
            if order.stop_price <= 0:
                errors.append("Stop price is required for OCO orders")

        # Symbol-specific validation
        info = self.get_symbol_info(order.symbol)
        if info is not None:
            # Lot size filter
            if order.quantity < info.min_qty:
                errors.append(
                    f"Quantity {order.quantity} below minimum {info.min_qty}"
                )
            if order.quantity > info.max_qty:
                errors.append(
                    f"Quantity {order.quantity} exceeds maximum {info.max_qty}"
                )

            # Check quantity step
            remainder = order.quantity % info.qty_step
            if remainder > 1e-10:
                rounded = info.round_qty(order.quantity)
                errors.append(
                    f"Quantity {order.quantity} not aligned to step "
                    f"{info.qty_step} (use {rounded})"
                )

            # Price filter (for orders that use price)
            if order.price > 0:
                if order.price < info.min_price:
                    errors.append(
                        f"Price {order.price} below minimum {info.min_price}"
                    )
                if order.price > info.max_price:
                    errors.append(
                        f"Price {order.price} exceeds maximum {info.max_price}"
                    )
                remainder = order.price % info.price_step
                if remainder > 1e-10:
                    rounded = info.round_price(order.price)
                    errors.append(
                        f"Price {order.price} not aligned to step "
                        f"{info.price_step} (use {rounded})"
                    )

            # Min notional
            effective_price = order.price if order.price > 0 else order.stop_price
            if effective_price > 0:
                notional = order.quantity * effective_price
                if notional < info.min_notional:
                    errors.append(
                        f"Notional {notional:.4f} below minimum "
                        f"{info.min_notional}"
                    )

            # Max open orders
            if open_order_count >= info.max_open_orders:
                errors.append(
                    f"Max open orders ({info.max_open_orders}) reached"
                )

        # Balance check for non-reduce-only orders
        if not order.reduce_only and available_balance > 0:
            required_margin = self._estimate_margin(order)
            if required_margin > available_balance:
                errors.append(
                    f"Insufficient balance. Required: {required_margin:.4f}, "
                    f"Available: {available_balance:.4f}"
                )

        # Stop price validation relative to current price
        if order.stop_price > 0 and order.price > 0:
            if order.order_type == OrderType.STOP_LIMIT:
                if order.side == OrderSide.BUY and order.stop_price < order.price:
                    pass  # Valid - stop above triggers limit below
                elif order.side == OrderSide.SELL and order.stop_price > order.price:
                    pass  # Valid - stop below triggers limit above

        return errors

    def _estimate_margin(self, order: Order) -> float:
        """Estimate required margin for this order."""
        effective_price = order.price if order.price > 0 else order.stop_price
        if effective_price <= 0:
            return 0.0
        notional = order.quantity * effective_price
        leverage = max(1, order.leverage)
        return notional / leverage


# ---------------------------------------------------------------------------
# Order callback types
# ---------------------------------------------------------------------------

OrderCallback = Callable[[Order, Optional[FillEvent]], None]


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Thread-safe order lifecycle manager. Handles order creation, validation,
    submission, modification, cancellation, fills, timeouts, and event callbacks.
    """

    def __init__(
        self,
        validator: Optional[OrderValidator] = None,
        max_open_orders_global: int = 500,
        max_open_orders_per_symbol: int = 50,
        default_timeout_seconds: int = 0,
    ):
        self._validator = validator or OrderValidator()

        # Thread-safe order storage
        self._lock = threading.RLock()
        self._orders: Dict[str, Order] = {}
        self._active_orders: Dict[str, Order] = {}
        self._orders_by_symbol: Dict[str, Dict[str, Order]] = defaultdict(dict)
        self._orders_by_strategy: Dict[str, Dict[str, Order]] = defaultdict(dict)
        self._client_id_map: Dict[str, str] = {}  # client_order_id -> id
        self._exchange_id_map: Dict[str, str] = {}  # exchange_order_id -> id

        # Limits
        self._max_open_global = max_open_orders_global
        self._max_open_per_symbol = max_open_orders_per_symbol
        self._default_timeout = default_timeout_seconds

        # Callbacks
        self._on_fill_callbacks: List[OrderCallback] = []
        self._on_partial_fill_callbacks: List[OrderCallback] = []
        self._on_cancel_callbacks: List[OrderCallback] = []
        self._on_reject_callbacks: List[OrderCallback] = []
        self._on_submit_callbacks: List[OrderCallback] = []
        self._on_expire_callbacks: List[OrderCallback] = []
        self._on_modify_callbacks: List[OrderCallback] = []

        # Stats
        self._total_submitted = 0
        self._total_filled = 0
        self._total_cancelled = 0
        self._total_rejected = 0

        # Timeout checker
        self._timeout_check_running = False
        self._timeout_task: Optional[asyncio.Task] = None

        logger.info(
            "OrderManager initialized (max_global=%d, max_per_symbol=%d)",
            max_open_orders_global,
            max_open_orders_per_symbol,
        )

    # -- Callback registration -----------------------------------------------

    def on_fill(self, callback: OrderCallback) -> None:
        self._on_fill_callbacks.append(callback)

    def on_partial_fill(self, callback: OrderCallback) -> None:
        self._on_partial_fill_callbacks.append(callback)

    def on_cancel(self, callback: OrderCallback) -> None:
        self._on_cancel_callbacks.append(callback)

    def on_reject(self, callback: OrderCallback) -> None:
        self._on_reject_callbacks.append(callback)

    def on_submit(self, callback: OrderCallback) -> None:
        self._on_submit_callbacks.append(callback)

    def on_expire(self, callback: OrderCallback) -> None:
        self._on_expire_callbacks.append(callback)

    def on_modify(self, callback: OrderCallback) -> None:
        self._on_modify_callbacks.append(callback)

    def _fire_callbacks(
        self,
        callbacks: List[OrderCallback],
        order: Order,
        fill: Optional[FillEvent] = None,
    ) -> None:
        for cb in callbacks:
            try:
                cb(order, fill)
            except Exception:
                logger.exception(
                    "Error in order callback for order %s", order.id
                )

    # -- Order creation and submission ----------------------------------------

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float = 0.0,
        stop_price: float = 0.0,
        take_profit: float = 0.0,
        trailing_delta: float = 0.0,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        close_position: bool = False,
        leverage: int = 1,
        strategy_id: str = "",
        signal_id: str = "",
        timeout_seconds: int = 0,
        tags: Optional[Dict[str, str]] = None,
    ) -> Order:
        """Create a new order object without submitting it."""
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            take_profit=take_profit,
            trailing_delta=trailing_delta,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            close_position=close_position,
            leverage=leverage,
            strategy_id=strategy_id,
            signal_id=signal_id,
            timeout_seconds=timeout_seconds or self._default_timeout,
            tags=tags or {},
        )

        # Round quantity and price to symbol step sizes
        info = self._validator.get_symbol_info(symbol)
        if info:
            order.quantity = info.round_qty(quantity)
            if price > 0:
                order.price = info.round_price(price)
            if stop_price > 0:
                order.stop_price = info.round_price(stop_price)
            if take_profit > 0:
                order.take_profit = info.round_price(take_profit)

        with self._lock:
            self._orders[order.id] = order
            self._client_id_map[order.client_order_id] = order.id
            self._orders_by_symbol[symbol][order.id] = order
            if strategy_id:
                self._orders_by_strategy[strategy_id][order.id] = order

        logger.info(
            "Order created: %s %s %s %.6f @ %.4f (id=%s)",
            symbol, side.value, order_type.value, order.quantity,
            order.price, order.id,
        )
        return order

    def submit_order(
        self,
        order: Order,
        available_balance: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """
        Validate and submit an order. Returns (success, errors).
        On success the order status moves to SUBMITTED.
        """
        with self._lock:
            # Check global limit
            if len(self._active_orders) >= self._max_open_global:
                return False, [
                    f"Global open order limit ({self._max_open_global}) reached"
                ]

            # Check per-symbol limit
            symbol_active = sum(
                1 for o in self._orders_by_symbol.get(order.symbol, {}).values()
                if o.is_active
            )
            if symbol_active >= self._max_open_per_symbol:
                return False, [
                    f"Per-symbol open order limit "
                    f"({self._max_open_per_symbol}) reached for {order.symbol}"
                ]

            # Validate
            errors = self._validator.validate(
                order,
                open_order_count=len(self._active_orders),
                available_balance=available_balance,
            )
            if errors:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "; ".join(errors)
                order.updated_at = datetime.now(timezone.utc)
                self._total_rejected += 1
                self._fire_callbacks(self._on_reject_callbacks, order)
                logger.warning(
                    "Order %s rejected: %s", order.id, order.reject_reason
                )
                return False, errors

            # Mark as submitted
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now(timezone.utc)
            order.updated_at = datetime.now(timezone.utc)
            self._active_orders[order.id] = order
            self._total_submitted += 1

        self._fire_callbacks(self._on_submit_callbacks, order)
        logger.info("Order %s submitted successfully", order.id)
        return True, []

    def create_and_submit(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float = 0.0,
        stop_price: float = 0.0,
        take_profit: float = 0.0,
        trailing_delta: float = 0.0,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        close_position: bool = False,
        leverage: int = 1,
        strategy_id: str = "",
        signal_id: str = "",
        timeout_seconds: int = 0,
        available_balance: float = 0.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> Tuple[Order, bool, List[str]]:
        """Create and immediately submit an order."""
        order = self.create_order(
            symbol=symbol, side=side, order_type=order_type,
            quantity=quantity, price=price, stop_price=stop_price,
            take_profit=take_profit, trailing_delta=trailing_delta,
            time_in_force=time_in_force, reduce_only=reduce_only,
            close_position=close_position, leverage=leverage,
            strategy_id=strategy_id, signal_id=signal_id,
            timeout_seconds=timeout_seconds, tags=tags,
        )
        success, errors = self.submit_order(order, available_balance)
        return order, success, errors

    # -- Order modification ---------------------------------------------------

    def set_exchange_order_id(self, order_id: str, exchange_id: str) -> None:
        """Map exchange order ID back to our internal order."""
        with self._lock:
            order = self._orders.get(order_id)
            if order:
                order.exchange_order_id = exchange_id
                self._exchange_id_map[exchange_id] = order_id
                order.updated_at = datetime.now(timezone.utc)

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_quantity: Optional[float] = None,
        new_stop_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Modify an active order's price, quantity, or stop price.
        Returns (success, message).
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return False, f"Order {order_id} not found"

            if not order.is_active:
                return False, (
                    f"Cannot modify order in {order.status.value} state"
                )

            if order.order_type == OrderType.MARKET:
                return False, "Cannot modify market orders"

            modified = False

            if new_price is not None and new_price > 0:
                info = self._validator.get_symbol_info(order.symbol)
                if info:
                    new_price = info.round_price(new_price)
                order.price = new_price
                modified = True

            if new_quantity is not None and new_quantity > 0:
                if new_quantity < order.filled_qty:
                    return False, (
                        f"New quantity {new_quantity} is less than "
                        f"already filled {order.filled_qty}"
                    )
                info = self._validator.get_symbol_info(order.symbol)
                if info:
                    new_quantity = info.round_qty(new_quantity)
                order.quantity = new_quantity
                modified = True

            if new_stop_price is not None and new_stop_price > 0:
                info = self._validator.get_symbol_info(order.symbol)
                if info:
                    new_stop_price = info.round_price(new_stop_price)
                order.stop_price = new_stop_price
                modified = True

            if modified:
                order.updated_at = datetime.now(timezone.utc)
                self._fire_callbacks(self._on_modify_callbacks, order)
                logger.info("Order %s modified", order_id)
                return True, "Order modified successfully"

            return False, "No modifications specified"

    # -- Order fill handling --------------------------------------------------

    def process_fill(
        self,
        order_id: str,
        price: float,
        quantity: float,
        commission: float = 0.0,
        commission_asset: str = "USDT",
        is_maker: bool = False,
        fill_id: Optional[str] = None,
    ) -> Optional[FillEvent]:
        """
        Process a fill event for an order.
        Returns the FillEvent or None if the order is not found/terminal.
        """
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.warning("Fill for unknown order %s", order_id)
                return None

            if order.is_terminal:
                logger.warning(
                    "Fill for terminal order %s (status=%s)",
                    order_id, order.status.value,
                )
                return None

            fill = FillEvent(
                fill_id=fill_id or uuid.uuid4().hex[:12],
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                price=price,
                quantity=quantity,
                commission=commission,
                commission_asset=commission_asset,
                is_maker=is_maker,
            )

            was_partial = order.status == OrderStatus.PARTIALLY_FILLED
            order.apply_fill(fill)

            if order.status == OrderStatus.FILLED:
                self._active_orders.pop(order_id, None)
                self._total_filled += 1
                self._fire_callbacks(self._on_fill_callbacks, order, fill)
                logger.info(
                    "Order %s FILLED: %.6f @ %.4f (commission=%.6f, "
                    "slippage=%.4f%%)",
                    order_id, order.filled_qty, order.avg_fill_price,
                    order.commission, order.slippage,
                )
            else:
                self._fire_callbacks(
                    self._on_partial_fill_callbacks, order, fill
                )
                logger.info(
                    "Order %s PARTIAL FILL: %.6f/%.6f @ %.4f",
                    order_id, order.filled_qty, order.quantity, price,
                )

            return fill

    def process_fill_by_exchange_id(
        self,
        exchange_order_id: str,
        price: float,
        quantity: float,
        commission: float = 0.0,
        commission_asset: str = "USDT",
        is_maker: bool = False,
    ) -> Optional[FillEvent]:
        """Process a fill using the exchange order ID."""
        with self._lock:
            internal_id = self._exchange_id_map.get(exchange_order_id)
        if internal_id is None:
            logger.warning(
                "Fill for unknown exchange order %s", exchange_order_id
            )
            return None
        return self.process_fill(
            internal_id, price, quantity, commission, commission_asset, is_maker
        )

    # -- Order cancellation ---------------------------------------------------

    def cancel_order(
        self, order_id: str, reason: str = "User requested"
    ) -> Tuple[bool, str]:
        """Cancel an active order. Returns (success, message)."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return False, f"Order {order_id} not found"

            if order.is_terminal:
                return False, (
                    f"Cannot cancel order in {order.status.value} state"
                )

            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.now(timezone.utc)
            order.updated_at = datetime.now(timezone.utc)
            order.reject_reason = reason
            self._active_orders.pop(order_id, None)
            self._total_cancelled += 1

        self._fire_callbacks(self._on_cancel_callbacks, order)
        logger.info("Order %s cancelled: %s", order_id, reason)
        return True, "Order cancelled"

    def cancel_all_orders(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        reason: str = "Bulk cancel",
    ) -> List[str]:
        """
        Cancel all active orders, optionally filtered by symbol or strategy.
        Returns list of cancelled order IDs.
        """
        cancelled_ids = []
        with self._lock:
            targets: Dict[str, Order]
            if symbol:
                targets = {
                    oid: o
                    for oid, o in self._orders_by_symbol.get(symbol, {}).items()
                    if o.is_active
                }
            elif strategy_id:
                targets = {
                    oid: o
                    for oid, o in self._orders_by_strategy.get(
                        strategy_id, {}
                    ).items()
                    if o.is_active
                }
            else:
                targets = dict(self._active_orders)

        for order_id in targets:
            success, _ = self.cancel_order(order_id, reason)
            if success:
                cancelled_ids.append(order_id)

        logger.info(
            "Bulk cancel: %d orders cancelled (symbol=%s, strategy=%s)",
            len(cancelled_ids), symbol, strategy_id,
        )
        return cancelled_ids

    def reject_order(
        self, order_id: str, reason: str, error_code: int = 0
    ) -> bool:
        """Mark an order as rejected by the exchange."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return False

            if order.is_terminal:
                return False

            order.status = OrderStatus.REJECTED
            order.reject_reason = reason
            order.error_code = error_code
            order.updated_at = datetime.now(timezone.utc)
            self._active_orders.pop(order_id, None)
            self._total_rejected += 1

        self._fire_callbacks(self._on_reject_callbacks, order)
        logger.warning(
            "Order %s rejected by exchange: %s (code=%d)",
            order_id, reason, error_code,
        )
        return True

    def expire_order(self, order_id: str) -> bool:
        """Mark an order as expired."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.is_terminal:
                return False

            order.status = OrderStatus.EXPIRED
            order.updated_at = datetime.now(timezone.utc)
            self._active_orders.pop(order_id, None)

        self._fire_callbacks(self._on_expire_callbacks, order)
        logger.info("Order %s expired", order_id)
        return True

    # -- Query methods --------------------------------------------------------

    def get_order(self, order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(order_id)

    def get_order_by_client_id(
        self, client_order_id: str
    ) -> Optional[Order]:
        with self._lock:
            internal_id = self._client_id_map.get(client_order_id)
            if internal_id:
                return self._orders.get(internal_id)
            return None

    def get_order_by_exchange_id(
        self, exchange_order_id: str
    ) -> Optional[Order]:
        with self._lock:
            internal_id = self._exchange_id_map.get(exchange_order_id)
            if internal_id:
                return self._orders.get(internal_id)
            return None

    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        side: Optional[OrderSide] = None,
    ) -> List[Order]:
        """Get all active orders, optionally filtered."""
        with self._lock:
            if symbol:
                orders = [
                    o for o in self._orders_by_symbol.get(symbol, {}).values()
                    if o.is_active
                ]
            elif strategy_id:
                orders = [
                    o for o in self._orders_by_strategy.get(
                        strategy_id, {}
                    ).values()
                    if o.is_active
                ]
            else:
                orders = list(self._active_orders.values())

            if side:
                orders = [o for o in orders if o.side == side]

            return sorted(orders, key=lambda o: o.created_at)

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Order]:
        """Get historical orders with optional filters."""
        with self._lock:
            if symbol:
                orders = list(
                    self._orders_by_symbol.get(symbol, {}).values()
                )
            elif strategy_id:
                orders = list(
                    self._orders_by_strategy.get(strategy_id, {}).values()
                )
            else:
                orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o.status == status]

        orders.sort(key=lambda o: o.created_at, reverse=True)
        return orders[offset: offset + limit]

    def get_filled_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Order]:
        """Get all filled orders, optionally filtered by symbol and time."""
        with self._lock:
            if symbol:
                orders = [
                    o for o in self._orders_by_symbol.get(symbol, {}).values()
                    if o.status == OrderStatus.FILLED
                ]
            else:
                orders = [
                    o for o in self._orders.values()
                    if o.status == OrderStatus.FILLED
                ]

        if since:
            orders = [
                o for o in orders
                if o.filled_at and o.filled_at >= since
            ]

        return sorted(orders, key=lambda o: o.filled_at or o.created_at)

    def count_open_orders(
        self, symbol: Optional[str] = None
    ) -> int:
        with self._lock:
            if symbol:
                return sum(
                    1 for o in self._orders_by_symbol.get(
                        symbol, {}
                    ).values()
                    if o.is_active
                )
            return len(self._active_orders)

    # -- Timeout handling -----------------------------------------------------

    async def start_timeout_checker(
        self, interval_seconds: float = 5.0
    ) -> None:
        """Start the background timeout checker loop."""
        if self._timeout_check_running:
            return
        self._timeout_check_running = True
        self._timeout_task = asyncio.create_task(
            self._timeout_loop(interval_seconds)
        )
        logger.info("Timeout checker started (interval=%.1fs)", interval_seconds)

    async def stop_timeout_checker(self) -> None:
        self._timeout_check_running = False
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None
        logger.info("Timeout checker stopped")

    async def _timeout_loop(self, interval: float) -> None:
        while self._timeout_check_running:
            try:
                self._check_timeouts()
            except Exception:
                logger.exception("Error in timeout check")
            await asyncio.sleep(interval)

    def _check_timeouts(self) -> None:
        """Check all active orders for timeouts."""
        timed_out: List[Order] = []
        with self._lock:
            for order in list(self._active_orders.values()):
                if order.is_timed_out:
                    timed_out.append(order)

        for order in timed_out:
            logger.info(
                "Order %s timed out after %.0fs",
                order.id, order.age_seconds,
            )
            self.cancel_order(order.id, "Timeout")
            self._fire_callbacks(self._on_expire_callbacks, order)

    # -- Linked / OCO orders --------------------------------------------------

    def create_oco_orders(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None,
        strategy_id: str = "",
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Tuple[Order, Order]:
        """
        Create a pair of linked OCO orders (take-profit limit + stop-loss).
        When one fills, the other should be cancelled.
        """
        tp_order = self.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=limit_price,
            time_in_force=time_in_force,
            reduce_only=True,
            strategy_id=strategy_id,
            tags={"oco_role": "take_profit"},
        )

        sl_type = (
            OrderType.STOP_LIMIT if stop_limit_price
            else OrderType.STOP_MARKET
        )
        sl_order = self.create_order(
            symbol=symbol,
            side=side,
            order_type=sl_type,
            quantity=quantity,
            price=stop_limit_price or 0.0,
            stop_price=stop_price,
            time_in_force=time_in_force,
            reduce_only=True,
            strategy_id=strategy_id,
            tags={"oco_role": "stop_loss"},
        )

        # Link them
        oco_id = uuid.uuid4().hex[:12]
        tp_order.oco_order_id = oco_id
        sl_order.oco_order_id = oco_id
        tp_order.linked_order_ids.append(sl_order.id)
        sl_order.linked_order_ids.append(tp_order.id)

        logger.info(
            "OCO pair created for %s: TP=%s (%.4f), SL=%s (%.4f)",
            symbol, tp_order.id, limit_price, sl_order.id, stop_price,
        )
        return tp_order, sl_order

    def cancel_linked_orders(self, order_id: str) -> List[str]:
        """Cancel all orders linked to this order (OCO handling)."""
        cancelled = []
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return cancelled
            linked = list(order.linked_order_ids)

        for linked_id in linked:
            success, _ = self.cancel_order(
                linked_id, f"Linked order {order_id} filled/cancelled"
            )
            if success:
                cancelled.append(linked_id)
        return cancelled

    # -- Statistics -----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_orders": len(self._orders),
                "active_orders": len(self._active_orders),
                "total_submitted": self._total_submitted,
                "total_filled": self._total_filled,
                "total_cancelled": self._total_cancelled,
                "total_rejected": self._total_rejected,
                "symbols_traded": len(self._orders_by_symbol),
                "strategies_active": len(self._orders_by_strategy),
                "fill_rate": (
                    self._total_filled / self._total_submitted
                    if self._total_submitted > 0 else 0.0
                ),
            }

    # -- Cleanup and persistence -----------------------------------------------

    def cleanup_old_orders(
        self, max_age_hours: int = 168
    ) -> int:
        """Remove terminal orders older than max_age_hours from memory."""
        cutoff = datetime.now(timezone.utc)
        removed = 0

        with self._lock:
            to_remove = []
            for order_id, order in self._orders.items():
                if not order.is_terminal:
                    continue
                age_hours = (cutoff - order.updated_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_remove.append(order_id)

            for order_id in to_remove:
                order = self._orders.pop(order_id)
                self._orders_by_symbol.get(order.symbol, {}).pop(
                    order_id, None
                )
                if order.strategy_id:
                    self._orders_by_strategy.get(
                        order.strategy_id, {}
                    ).pop(order_id, None)
                self._client_id_map.pop(order.client_order_id, None)
                if order.exchange_order_id:
                    self._exchange_id_map.pop(order.exchange_order_id, None)
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d old orders", removed)
        return removed

    def export_orders(
        self, include_terminal: bool = True
    ) -> List[Dict[str, Any]]:
        """Export all orders as dictionaries for persistence."""
        with self._lock:
            orders = list(self._orders.values())
        if not include_terminal:
            orders = [o for o in orders if not o.is_terminal]
        return [o.to_dict() for o in orders]

    def import_orders(self, order_dicts: List[Dict[str, Any]]) -> int:
        """Import orders from dictionaries (for recovery after restart)."""
        imported = 0
        with self._lock:
            for data in order_dicts:
                order = Order.from_dict(data)
                self._orders[order.id] = order
                self._client_id_map[order.client_order_id] = order.id
                if order.exchange_order_id:
                    self._exchange_id_map[order.exchange_order_id] = order.id
                self._orders_by_symbol[order.symbol][order.id] = order
                if order.strategy_id:
                    self._orders_by_strategy[order.strategy_id][
                        order.id
                    ] = order
                if order.is_active:
                    self._active_orders[order.id] = order
                imported += 1

        logger.info("Imported %d orders", imported)
        return imported

    @property
    def validator(self) -> OrderValidator:
        return self._validator
