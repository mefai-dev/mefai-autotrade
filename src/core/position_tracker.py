"""
Position Tracker - Real-time position tracking for Mefai Autotrade.
Manages open positions, unrealized/realized PnL, liquidation calculations,
partial closes, position averaging, and full audit trails.
"""

import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mefai.autotrade.position_tracker")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"  # Hedge mode - not commonly used


class PositionAction(Enum):
    OPEN = "OPEN"
    ADD = "ADD"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSE = "CLOSE"
    LIQUIDATION = "LIQUIDATION"
    ADL = "ADL"  # Auto-deleveraging
    MODIFY_SL = "MODIFY_SL"
    MODIFY_TP = "MODIFY_TP"
    MODIFY_LEVERAGE = "MODIFY_LEVERAGE"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PositionError(Exception):
    pass


class PositionNotFoundError(PositionError):
    pass


class InsufficientPositionError(PositionError):
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionEvent:
    """Record of a single action taken on a position."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    position_id: str = ""
    action: PositionAction = PositionAction.OPEN
    quantity: float = 0.0
    price: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    order_id: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "position_id": self.position_id,
            "action": self.action.value,
            "quantity": self.quantity,
            "price": self.price,
            "realized_pnl": self.realized_pnl,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "notes": self.notes,
        }


@dataclass
class Position:
    """
    Represents an open (or closed) trading position with full tracking.
    """
    # Identity
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    symbol: str = ""
    side: PositionSide = PositionSide.LONG

    # Core position data
    entry_price: float = 0.0
    quantity: float = 0.0
    leverage: int = 1
    margin_mode: str = "CROSS"  # CROSS or ISOLATED

    # PnL tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    funding_pnl: float = 0.0

    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_pct: float = 0.0
    trailing_stop_activation: float = 0.0
    trailing_stop_highest: float = 0.0  # Tracks highest price for trailing

    # Calculated fields
    liquidation_price: float = 0.0
    margin_used: float = 0.0
    mark_price: float = 0.0
    break_even_price: float = 0.0

    # Drawdown tracking
    peak_unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Timestamps
    opened_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    closed_at: Optional[datetime] = None

    # Status
    is_open: bool = True

    # Strategy info
    strategy_id: str = ""
    signal_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    # Audit trail
    events: List[PositionEvent] = field(default_factory=list)

    # Aggregation helpers
    total_bought_qty: float = 0.0
    total_bought_cost: float = 0.0  # sum(qty * price) for buys
    total_sold_qty: float = 0.0
    total_sold_cost: float = 0.0  # sum(qty * price) for sells

    @property
    def notional(self) -> float:
        """Current notional value based on mark price or entry price."""
        price = self.mark_price if self.mark_price > 0 else self.entry_price
        return abs(self.quantity) * price

    @property
    def entry_notional(self) -> float:
        return abs(self.quantity) * self.entry_price

    @property
    def holding_time_seconds(self) -> float:
        end = self.closed_at or datetime.now(timezone.utc)
        return (end - self.opened_at).total_seconds()

    @property
    def holding_time_hours(self) -> float:
        return self.holding_time_seconds / 3600.0

    @property
    def pnl_pct(self) -> float:
        """Unrealized PnL as percentage of entry notional."""
        if self.entry_notional <= 0:
            return 0.0
        return (self.unrealized_pnl / self.entry_notional) * 100.0

    @property
    def roe(self) -> float:
        """Return on equity (leveraged return)."""
        return self.pnl_pct * self.leverage

    @property
    def net_pnl(self) -> float:
        """Total PnL including commissions and funding."""
        return (
            self.unrealized_pnl
            + self.realized_pnl
            - self.total_commission
            + self.funding_pnl
        )

    @property
    def liquidation_distance_pct(self) -> float:
        """How far current price is from liquidation, in percent."""
        if self.liquidation_price <= 0 or self.mark_price <= 0:
            return 100.0
        if self.side == PositionSide.LONG:
            return (
                (self.mark_price - self.liquidation_price)
                / self.mark_price
            ) * 100.0
        else:
            return (
                (self.liquidation_price - self.mark_price)
                / self.mark_price
            ) * 100.0

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at a given price."""
        if self.quantity <= 0 or self.entry_price <= 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def calculate_liquidation_price(
        self,
        wallet_balance: float = 0.0,
        maintenance_margin_rate: float = 0.004,
        maintenance_amount: float = 0.0,
    ) -> float:
        """
        Calculate the liquidation price for this position.
        Uses the simplified Binance Futures formula.
        """
        if self.quantity <= 0:
            return 0.0

        # For isolated margin, wallet_balance = margin_used
        if self.margin_mode == "ISOLATED":
            wb = self.margin_used
        else:
            wb = wallet_balance

        if self.side == PositionSide.LONG:
            liq = (
                wb
                - self.entry_price * self.quantity
                + maintenance_amount
            ) / (
                self.quantity * (maintenance_margin_rate - 1)
            )
            # Liquidation is below entry for longs
            return max(0.0, abs(liq))
        else:
            liq = (
                wb
                + self.entry_price * self.quantity
                - maintenance_amount
            ) / (
                self.quantity * (maintenance_margin_rate + 1)
            )
            return max(0.0, abs(liq))

    def calculate_break_even(self) -> float:
        """
        Calculate break-even price accounting for commissions.
        """
        if self.quantity <= 0:
            return self.entry_price

        total_cost = self.total_commission
        if self.side == PositionSide.LONG:
            return self.entry_price + (total_cost / self.quantity)
        else:
            return self.entry_price - (total_cost / self.quantity)

    def update_trailing_stop(self, current_price: float) -> bool:
        """
        Update trailing stop. Returns True if stop should trigger.
        """
        if self.trailing_stop_pct <= 0:
            return False

        # Check activation price first
        if self.trailing_stop_activation > 0:
            if self.side == PositionSide.LONG:
                if current_price < self.trailing_stop_activation:
                    return False
            else:
                if current_price > self.trailing_stop_activation:
                    return False

        # Update highest/lowest tracked price
        if self.side == PositionSide.LONG:
            if current_price > self.trailing_stop_highest:
                self.trailing_stop_highest = current_price
            trigger_price = self.trailing_stop_highest * (
                1 - self.trailing_stop_pct / 100.0
            )
            return current_price <= trigger_price
        else:
            if (
                self.trailing_stop_highest <= 0
                or current_price < self.trailing_stop_highest
            ):
                self.trailing_stop_highest = current_price
            trigger_price = self.trailing_stop_highest * (
                1 + self.trailing_stop_pct / 100.0
            )
            return current_price >= trigger_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "margin_mode": self.margin_mode,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "funding_pnl": self.funding_pnl,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_pct": self.trailing_stop_pct,
            "liquidation_price": self.liquidation_price,
            "margin_used": self.margin_used,
            "mark_price": self.mark_price,
            "break_even_price": self.break_even_price,
            "peak_unrealized_pnl": self.peak_unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "opened_at": self.opened_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "is_open": self.is_open,
            "strategy_id": self.strategy_id,
            "signal_id": self.signal_id,
            "tags": self.tags,
            "notional": self.notional,
            "pnl_pct": self.pnl_pct,
            "roe": self.roe,
            "net_pnl": self.net_pnl,
            "holding_time_hours": self.holding_time_hours,
            "liquidation_distance_pct": self.liquidation_distance_pct,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        pos = cls()
        events_data = data.pop("events", [])
        for key, value in data.items():
            if key == "side":
                pos.side = PositionSide(value)
            elif key in ("opened_at", "updated_at"):
                if value and isinstance(value, str):
                    setattr(pos, key, datetime.fromisoformat(value))
            elif key == "closed_at":
                if value and isinstance(value, str):
                    pos.closed_at = datetime.fromisoformat(value)
            elif key in (
                "notional", "pnl_pct", "roe", "net_pnl",
                "holding_time_hours", "liquidation_distance_pct",
            ):
                # Computed properties - skip
                continue
            elif hasattr(pos, key):
                setattr(pos, key, value)
        # Restore events
        for ed in events_data:
            evt = PositionEvent()
            for k, v in ed.items():
                if k == "action":
                    evt.action = PositionAction(v)
                elif k == "timestamp" and isinstance(v, str):
                    evt.timestamp = datetime.fromisoformat(v)
                elif hasattr(evt, k):
                    setattr(evt, k, v)
            pos.events.append(evt)
        return pos


# ---------------------------------------------------------------------------
# Position Tracker callback type
# ---------------------------------------------------------------------------

PositionCallback = Callable[[Position, PositionEvent], None]


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """
    Thread-safe position tracker with real-time PnL, margin calculations,
    liquidation monitoring, and full audit trail.
    """

    def __init__(
        self,
        default_leverage: int = 1,
        default_margin_mode: str = "CROSS",
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
    ):
        self._lock = threading.RLock()

        # Position storage
        self._positions: Dict[str, Position] = {}  # id -> Position
        self._open_positions: Dict[str, Position] = {}
        self._positions_by_symbol: Dict[str, Dict[str, Position]] = defaultdict(dict)
        self._positions_by_strategy: Dict[str, Dict[str, Position]] = defaultdict(dict)
        self._closed_positions: List[Position] = []

        # Defaults
        self._default_leverage = default_leverage
        self._default_margin_mode = default_margin_mode
        self._maker_fee = maker_fee
        self._taker_fee = taker_fee

        # Leverage per symbol
        self._leverage_map: Dict[str, int] = {}

        # Callbacks
        self._on_open_callbacks: List[PositionCallback] = []
        self._on_close_callbacks: List[PositionCallback] = []
        self._on_update_callbacks: List[PositionCallback] = []
        self._on_liquidation_warning_callbacks: List[PositionCallback] = []

        # Totals
        self._total_realized_pnl: float = 0.0
        self._total_commission: float = 0.0
        self._total_positions_opened: int = 0
        self._total_positions_closed: int = 0

        logger.info(
            "PositionTracker initialized (leverage=%d, margin=%s, "
            "maker=%.4f%%, taker=%.4f%%)",
            default_leverage, default_margin_mode,
            maker_fee * 100, taker_fee * 100,
        )

    # -- Callback registration -----------------------------------------------

    def on_open(self, callback: PositionCallback) -> None:
        self._on_open_callbacks.append(callback)

    def on_close(self, callback: PositionCallback) -> None:
        self._on_close_callbacks.append(callback)

    def on_update(self, callback: PositionCallback) -> None:
        self._on_update_callbacks.append(callback)

    def on_liquidation_warning(self, callback: PositionCallback) -> None:
        self._on_liquidation_warning_callbacks.append(callback)

    def _fire_callbacks(
        self,
        callbacks: List[PositionCallback],
        position: Position,
        event: PositionEvent,
    ) -> None:
        for cb in callbacks:
            try:
                cb(position, event)
            except Exception:
                logger.exception(
                    "Error in position callback for %s", position.id
                )

    # -- Leverage management --------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> None:
        if leverage < 1 or leverage > 125:
            raise PositionError(
                f"Invalid leverage {leverage} (must be 1-125)"
            )
        with self._lock:
            self._leverage_map[symbol] = leverage
        logger.info("Leverage for %s set to %dx", symbol, leverage)

    def get_leverage(self, symbol: str) -> int:
        with self._lock:
            return self._leverage_map.get(symbol, self._default_leverage)

    # -- Position opening -----------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float,
        leverage: Optional[int] = None,
        margin_mode: Optional[str] = None,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        trailing_stop_pct: float = 0.0,
        trailing_stop_activation: float = 0.0,
        strategy_id: str = "",
        signal_id: str = "",
        order_id: str = "",
        commission: float = 0.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> Position:
        """
        Open a new position or add to an existing one for the same
        symbol/side/strategy.
        """
        if quantity <= 0:
            raise PositionError("Quantity must be positive")
        if entry_price <= 0:
            raise PositionError("Entry price must be positive")

        lev = leverage or self.get_leverage(symbol)
        mm = margin_mode or self._default_margin_mode

        # Check if there is already an open position for this symbol/side
        existing = self._find_open_position(symbol, side, strategy_id)
        if existing is not None:
            return self._add_to_position(
                existing, quantity, entry_price, commission, order_id
            )

        # Calculate commission if not provided
        if commission <= 0:
            commission = quantity * entry_price * self._taker_fee

        # Calculate margin
        notional = quantity * entry_price
        margin = notional / lev

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            leverage=lev,
            margin_mode=mm,
            margin_used=margin,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            trailing_stop_activation=trailing_stop_activation,
            trailing_stop_highest=(
                entry_price if trailing_stop_pct > 0 else 0.0
            ),
            mark_price=entry_price,
            total_commission=commission,
            strategy_id=strategy_id,
            signal_id=signal_id,
            tags=tags or {},
        )

        # Track buy/sell totals for averaging
        if side == PositionSide.LONG:
            position.total_bought_qty = quantity
            position.total_bought_cost = quantity * entry_price
        else:
            position.total_sold_qty = quantity
            position.total_sold_cost = quantity * entry_price

        # Calculate break-even and liquidation
        position.break_even_price = position.calculate_break_even()
        position.liquidation_price = position.calculate_liquidation_price(
            wallet_balance=margin
        )

        # Create audit event
        event = PositionEvent(
            position_id=position.id,
            action=PositionAction.OPEN,
            quantity=quantity,
            price=entry_price,
            commission=commission,
            order_id=order_id,
            notes=f"Opened {side.value} {symbol} x{lev}",
        )
        position.events.append(event)

        # Store
        with self._lock:
            self._positions[position.id] = position
            self._open_positions[position.id] = position
            self._positions_by_symbol[symbol][position.id] = position
            if strategy_id:
                self._positions_by_strategy[strategy_id][
                    position.id
                ] = position
            self._total_positions_opened += 1

        self._fire_callbacks(self._on_open_callbacks, position, event)
        logger.info(
            "Position opened: %s %s %s %.6f @ %.4f (lev=%dx, margin=%.4f)",
            position.id, side.value, symbol, quantity, entry_price,
            lev, margin,
        )
        return position

    def _find_open_position(
        self,
        symbol: str,
        side: PositionSide,
        strategy_id: str = "",
    ) -> Optional[Position]:
        """Find an existing open position for the same symbol/side."""
        with self._lock:
            for pos in self._positions_by_symbol.get(symbol, {}).values():
                if pos.is_open and pos.side == side:
                    if strategy_id and pos.strategy_id != strategy_id:
                        continue
                    return pos
        return None

    def _add_to_position(
        self,
        position: Position,
        quantity: float,
        price: float,
        commission: float = 0.0,
        order_id: str = "",
    ) -> Position:
        """Add to an existing position (average in)."""
        if commission <= 0:
            commission = quantity * price * self._taker_fee

        with self._lock:
            # Calculate new average entry price
            old_cost = position.entry_price * position.quantity
            new_cost = price * quantity
            new_total_qty = position.quantity + quantity
            position.entry_price = (old_cost + new_cost) / new_total_qty
            position.quantity = new_total_qty
            position.total_commission += commission

            # Update buy/sell tracking
            if position.side == PositionSide.LONG:
                position.total_bought_qty += quantity
                position.total_bought_cost += quantity * price
            else:
                position.total_sold_qty += quantity
                position.total_sold_cost += quantity * price

            # Recalculate margin
            notional = position.quantity * position.entry_price
            position.margin_used = notional / position.leverage

            # Recalculate break-even and liquidation
            position.break_even_price = position.calculate_break_even()
            position.liquidation_price = (
                position.calculate_liquidation_price(
                    wallet_balance=position.margin_used
                )
            )
            position.updated_at = datetime.now(timezone.utc)

        event = PositionEvent(
            position_id=position.id,
            action=PositionAction.ADD,
            quantity=quantity,
            price=price,
            commission=commission,
            order_id=order_id,
            notes=(
                f"Added {quantity:.6f} @ {price:.4f}, "
                f"new avg={position.entry_price:.4f}"
            ),
        )
        position.events.append(event)

        self._fire_callbacks(self._on_update_callbacks, position, event)
        logger.info(
            "Position %s added: +%.6f @ %.4f, avg=%.4f, total=%.6f",
            position.id, quantity, price,
            position.entry_price, position.quantity,
        )
        return position

    # -- Position closing / partial close ------------------------------------

    def close_position(
        self,
        position_id: str,
        close_price: float,
        quantity: Optional[float] = None,
        commission: float = 0.0,
        order_id: str = "",
        reason: str = "",
    ) -> Tuple[float, PositionEvent]:
        """
        Close a position fully or partially.
        Returns (realized_pnl, event).
        """
        with self._lock:
            position = self._open_positions.get(position_id)
            if position is None:
                raise PositionNotFoundError(
                    f"Open position {position_id} not found"
                )

            close_qty = quantity if quantity else position.quantity
            if close_qty > position.quantity:
                raise InsufficientPositionError(
                    f"Cannot close {close_qty:.6f}, "
                    f"only {position.quantity:.6f} available"
                )
            if close_qty <= 0:
                raise PositionError("Close quantity must be positive")

            # Calculate commission
            if commission <= 0:
                commission = close_qty * close_price * self._taker_fee

            # Calculate realized PnL for this portion
            if position.side == PositionSide.LONG:
                realized = (close_price - position.entry_price) * close_qty
            else:
                realized = (position.entry_price - close_price) * close_qty

            realized -= commission  # Net of commission
            position.realized_pnl += realized
            position.total_commission += commission
            self._total_realized_pnl += realized
            self._total_commission += commission

            is_full_close = abs(close_qty - position.quantity) < 1e-10

            if is_full_close:
                # Full close
                position.quantity = 0.0
                position.unrealized_pnl = 0.0
                position.is_open = False
                position.closed_at = datetime.now(timezone.utc)
                position.margin_used = 0.0

                self._open_positions.pop(position_id, None)
                self._closed_positions.append(position)
                self._total_positions_closed += 1

                action = PositionAction.CLOSE
                notes = (
                    f"Closed {position.side.value} {position.symbol} "
                    f"@ {close_price:.4f}, PnL={realized:.4f}"
                )
                if reason:
                    notes += f" ({reason})"
            else:
                # Partial close
                position.quantity -= close_qty
                notional = position.quantity * position.entry_price
                position.margin_used = notional / position.leverage
                position.break_even_price = position.calculate_break_even()
                position.liquidation_price = (
                    position.calculate_liquidation_price(
                        wallet_balance=position.margin_used
                    )
                )
                action = PositionAction.PARTIAL_CLOSE
                notes = (
                    f"Partial close {close_qty:.6f} @ {close_price:.4f}, "
                    f"remaining={position.quantity:.6f}, PnL={realized:.4f}"
                )

            position.updated_at = datetime.now(timezone.utc)

        event = PositionEvent(
            position_id=position_id,
            action=action,
            quantity=close_qty,
            price=close_price,
            realized_pnl=realized,
            commission=commission,
            order_id=order_id,
            notes=notes,
        )
        position.events.append(event)

        if is_full_close:
            self._fire_callbacks(self._on_close_callbacks, position, event)
            logger.info(
                "Position %s closed: %.6f @ %.4f, realized=%.4f, "
                "total_commission=%.4f",
                position_id, close_qty, close_price,
                realized, position.total_commission,
            )
        else:
            self._fire_callbacks(self._on_update_callbacks, position, event)
            logger.info(
                "Position %s partial close: %.6f @ %.4f, "
                "remaining=%.6f, realized=%.4f",
                position_id, close_qty, close_price,
                position.quantity, realized,
            )

        return realized, event

    def close_position_by_symbol(
        self,
        symbol: str,
        close_price: float,
        side: Optional[PositionSide] = None,
        strategy_id: str = "",
        reason: str = "",
    ) -> List[Tuple[str, float]]:
        """
        Close all positions for a symbol. Returns list of (position_id, pnl).
        """
        results = []
        positions = self.get_open_positions(
            symbol=symbol, side=side, strategy_id=strategy_id
        )
        for pos in positions:
            try:
                pnl, _ = self.close_position(
                    pos.id, close_price, reason=reason
                )
                results.append((pos.id, pnl))
            except PositionError as e:
                logger.error("Failed to close position %s: %s", pos.id, e)
        return results

    def close_all_positions(
        self,
        prices: Dict[str, float],
        reason: str = "Close all",
    ) -> List[Tuple[str, float]]:
        """
        Close all open positions using provided price map.
        Returns list of (position_id, realized_pnl).
        """
        results = []
        with self._lock:
            open_positions = list(self._open_positions.values())

        for pos in open_positions:
            price = prices.get(pos.symbol)
            if price is None:
                logger.warning(
                    "No price for %s, skipping position %s",
                    pos.symbol, pos.id,
                )
                continue
            try:
                pnl, _ = self.close_position(pos.id, price, reason=reason)
                results.append((pos.id, pnl))
            except PositionError as e:
                logger.error("Failed to close position %s: %s", pos.id, e)

        return results

    # -- Price updates and PnL recalculation ----------------------------------

    def update_mark_price(
        self, symbol: str, mark_price: float
    ) -> List[Position]:
        """
        Update mark price for all positions of a symbol.
        Recalculates unrealized PnL, drawdown, and checks liquidation.
        Returns list of affected positions.
        """
        if mark_price <= 0:
            return []

        affected: List[Position] = []
        with self._lock:
            positions = [
                p for p in self._positions_by_symbol.get(symbol, {}).values()
                if p.is_open
            ]

            for pos in positions:
                pos.mark_price = mark_price

                # Update unrealized PnL
                old_pnl = pos.unrealized_pnl
                pos.unrealized_pnl = pos.calculate_unrealized_pnl(mark_price)

                # Update peak and drawdown
                if pos.unrealized_pnl > pos.peak_unrealized_pnl:
                    pos.peak_unrealized_pnl = pos.unrealized_pnl

                if pos.peak_unrealized_pnl > 0:
                    current_dd = pos.peak_unrealized_pnl - pos.unrealized_pnl
                    if current_dd > pos.max_drawdown:
                        pos.max_drawdown = current_dd
                    dd_pct = (current_dd / pos.peak_unrealized_pnl) * 100
                    if dd_pct > pos.max_drawdown_pct:
                        pos.max_drawdown_pct = dd_pct

                # Update trailing stop
                pos.update_trailing_stop(mark_price)

                pos.updated_at = datetime.now(timezone.utc)
                affected.append(pos)

                # Liquidation warning check
                if pos.liquidation_price > 0:
                    dist = pos.liquidation_distance_pct
                    if 0 < dist < 5.0:  # Within 5% of liquidation
                        warning_event = PositionEvent(
                            position_id=pos.id,
                            action=PositionAction.LIQUIDATION,
                            price=mark_price,
                            notes=(
                                f"Liquidation warning: {dist:.2f}% away "
                                f"(liq={pos.liquidation_price:.4f})"
                            ),
                        )
                        self._fire_callbacks(
                            self._on_liquidation_warning_callbacks,
                            pos,
                            warning_event,
                        )

        return affected

    def update_mark_prices(
        self, prices: Dict[str, float]
    ) -> Dict[str, List[Position]]:
        """Update mark prices for multiple symbols at once."""
        results: Dict[str, List[Position]] = {}
        for symbol, price in prices.items():
            affected = self.update_mark_price(symbol, price)
            if affected:
                results[symbol] = affected
        return results

    def apply_funding_payment(
        self,
        symbol: str,
        funding_rate: float,
    ) -> List[Tuple[str, float]]:
        """
        Apply funding rate payment to all open positions of a symbol.
        Returns list of (position_id, funding_pnl).
        """
        results = []
        with self._lock:
            positions = [
                p for p in self._positions_by_symbol.get(symbol, {}).values()
                if p.is_open
            ]

            for pos in positions:
                # Funding = position_value * funding_rate
                # Positive rate: longs pay shorts
                # Negative rate: shorts pay longs
                position_value = pos.quantity * pos.mark_price
                if pos.side == PositionSide.LONG:
                    funding_pnl = -position_value * funding_rate
                else:
                    funding_pnl = position_value * funding_rate

                pos.funding_pnl += funding_pnl
                pos.updated_at = datetime.now(timezone.utc)
                results.append((pos.id, funding_pnl))

                event = PositionEvent(
                    position_id=pos.id,
                    action=PositionAction.OPEN,  # Reusing for funding
                    price=pos.mark_price,
                    realized_pnl=funding_pnl,
                    notes=(
                        f"Funding payment: rate={funding_rate:.6f}, "
                        f"pnl={funding_pnl:.4f}"
                    ),
                )
                pos.events.append(event)

        return results

    # -- Modify SL/TP --------------------------------------------------------

    def modify_stop_loss(
        self, position_id: str, stop_loss: float
    ) -> None:
        with self._lock:
            pos = self._open_positions.get(position_id)
            if pos is None:
                raise PositionNotFoundError(
                    f"Position {position_id} not found"
                )
            old_sl = pos.stop_loss
            pos.stop_loss = stop_loss
            pos.updated_at = datetime.now(timezone.utc)

        event = PositionEvent(
            position_id=position_id,
            action=PositionAction.MODIFY_SL,
            price=stop_loss,
            notes=f"SL changed from {old_sl:.4f} to {stop_loss:.4f}",
        )
        pos.events.append(event)
        self._fire_callbacks(self._on_update_callbacks, pos, event)
        logger.info(
            "Position %s SL modified: %.4f -> %.4f",
            position_id, old_sl, stop_loss,
        )

    def modify_take_profit(
        self, position_id: str, take_profit: float
    ) -> None:
        with self._lock:
            pos = self._open_positions.get(position_id)
            if pos is None:
                raise PositionNotFoundError(
                    f"Position {position_id} not found"
                )
            old_tp = pos.take_profit
            pos.take_profit = take_profit
            pos.updated_at = datetime.now(timezone.utc)

        event = PositionEvent(
            position_id=position_id,
            action=PositionAction.MODIFY_TP,
            price=take_profit,
            notes=f"TP changed from {old_tp:.4f} to {take_profit:.4f}",
        )
        pos.events.append(event)
        self._fire_callbacks(self._on_update_callbacks, pos, event)
        logger.info(
            "Position %s TP modified: %.4f -> %.4f",
            position_id, old_tp, take_profit,
        )

    # -- Query methods --------------------------------------------------------

    def get_position(self, position_id: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(position_id)

    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        side: Optional[PositionSide] = None,
        strategy_id: Optional[str] = None,
    ) -> List[Position]:
        with self._lock:
            if symbol:
                positions = [
                    p for p in self._positions_by_symbol.get(
                        symbol, {}
                    ).values()
                    if p.is_open
                ]
            elif strategy_id:
                positions = [
                    p for p in self._positions_by_strategy.get(
                        strategy_id, {}
                    ).values()
                    if p.is_open
                ]
            else:
                positions = list(self._open_positions.values())

            if side:
                positions = [p for p in positions if p.side == side]

            return sorted(positions, key=lambda p: p.opened_at)

    def get_closed_positions(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Position]:
        with self._lock:
            positions = list(self._closed_positions)
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        positions.sort(
            key=lambda p: p.closed_at or p.updated_at, reverse=True
        )
        return positions[offset: offset + limit]

    def get_position_history(
        self, position_id: str
    ) -> List[PositionEvent]:
        with self._lock:
            pos = self._positions.get(position_id)
            if pos is None:
                return []
            return list(pos.events)

    # -- Exposure and margin --------------------------------------------------

    def get_total_exposure(
        self, symbol: Optional[str] = None
    ) -> float:
        """Total notional exposure across all open positions."""
        with self._lock:
            if symbol:
                positions = [
                    p for p in self._positions_by_symbol.get(
                        symbol, {}
                    ).values()
                    if p.is_open
                ]
            else:
                positions = list(self._open_positions.values())
            return sum(p.notional for p in positions)

    def get_total_margin_used(self) -> float:
        with self._lock:
            return sum(
                p.margin_used for p in self._open_positions.values()
            )

    def get_margin_usage(
        self, total_balance: float
    ) -> float:
        """Margin usage as a percentage of total balance."""
        if total_balance <= 0:
            return 0.0
        used = self.get_total_margin_used()
        return (used / total_balance) * 100.0

    def get_net_exposure(self) -> Dict[str, float]:
        """
        Net exposure per symbol (long - short notionals).
        Positive = net long, negative = net short.
        """
        exposure: Dict[str, float] = defaultdict(float)
        with self._lock:
            for pos in self._open_positions.values():
                if pos.side == PositionSide.LONG:
                    exposure[pos.symbol] += pos.notional
                else:
                    exposure[pos.symbol] -= pos.notional
        return dict(exposure)

    def get_total_unrealized_pnl(self) -> float:
        with self._lock:
            return sum(
                p.unrealized_pnl for p in self._open_positions.values()
            )

    def get_total_realized_pnl(self) -> float:
        return self._total_realized_pnl

    # -- Statistics -----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            open_positions = list(self._open_positions.values())

        return {
            "open_positions": len(open_positions),
            "total_opened": self._total_positions_opened,
            "total_closed": self._total_positions_closed,
            "total_unrealized_pnl": sum(
                p.unrealized_pnl for p in open_positions
            ),
            "total_realized_pnl": self._total_realized_pnl,
            "total_commission": self._total_commission,
            "total_margin_used": sum(
                p.margin_used for p in open_positions
            ),
            "total_exposure": sum(p.notional for p in open_positions),
            "symbols": list(set(p.symbol for p in open_positions)),
            "long_positions": sum(
                1 for p in open_positions
                if p.side == PositionSide.LONG
            ),
            "short_positions": sum(
                1 for p in open_positions
                if p.side == PositionSide.SHORT
            ),
        }

    # -- Export / Import for persistence --------------------------------------

    def export_positions(
        self, include_closed: bool = False
    ) -> List[Dict[str, Any]]:
        with self._lock:
            positions = list(self._open_positions.values())
            if include_closed:
                positions.extend(self._closed_positions)
        return [p.to_dict() for p in positions]

    def import_positions(
        self, position_dicts: List[Dict[str, Any]]
    ) -> int:
        imported = 0
        with self._lock:
            for data in position_dicts:
                pos = Position.from_dict(data)
                self._positions[pos.id] = pos
                self._positions_by_symbol[pos.symbol][pos.id] = pos
                if pos.strategy_id:
                    self._positions_by_strategy[pos.strategy_id][
                        pos.id
                    ] = pos
                if pos.is_open:
                    self._open_positions[pos.id] = pos
                else:
                    self._closed_positions.append(pos)
                imported += 1
        logger.info("Imported %d positions", imported)
        return imported
