# Mefai Signal Engine - Core Backtesting Engine
# Event-driven simulation engine with realistic order matching, slippage,
# commission, funding rate, margin, and liquidation simulation.

import time
import logging
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Callable, Any, Tuple, Set, Union
)
from datetime import datetime, timedelta
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SlippageModel(Enum):
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLATILITY = "volatility"
    NONE = "none"


class SimulationMode(Enum):
    BAR_BY_BAR = "bar_by_bar"
    TICK_BY_TICK = "tick_by_tick"


class EventType(Enum):
    BAR = "BAR"
    TICK = "TICK"
    ORDER_FILL = "ORDER_FILL"
    ORDER_CANCEL = "ORDER_CANCEL"
    LIQUIDATION = "LIQUIDATION"
    FUNDING = "FUNDING"
    MARGIN_CALL = "MARGIN_CALL"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    initial_capital: float = 10000.0
    commission_maker: float = 0.0002
    commission_taker: float = 0.0004
    slippage_model: SlippageModel = SlippageModel.PERCENTAGE
    slippage_value: float = 0.0005
    slippage_volatility_window: int = 20
    slippage_volatility_multiplier: float = 0.5
    leverage: float = 1.0
    max_leverage: float = 125.0
    margin_mode: str = "cross"  # cross or isolated
    maintenance_margin_rate: float = 0.004
    funding_rate_interval_hours: int = 8
    enable_funding: bool = True
    enable_liquidation: bool = True
    enable_margin_call: bool = True
    margin_call_threshold: float = 0.8
    simulation_mode: SimulationMode = SimulationMode.BAR_BY_BAR
    max_open_orders: int = 100
    max_positions: int = 50
    order_expiry_bars: int = 0  # 0 = no expiry
    risk_free_rate: float = 0.0
    benchmark_symbol: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    warmup_bars: int = 0
    enable_short_selling: bool = True
    enable_fractional: bool = True
    min_order_size: float = 0.001
    latency_ms: float = 0.0
    verbose: bool = False
    progress_bar: bool = True
    tag: str = ""


@dataclass
class Order:
    """Represents an order in the simulation."""
    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    reduce_only: bool = False
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    expire_bar: int = 0
    client_order_id: str = ""
    tag: str = ""

    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())[:12]


@dataclass
class Position:
    """Represents an open position."""
    symbol: str = ""
    side: PositionSide = PositionSide.FLAT
    entry_price: float = 0.0
    quantity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin_used: float = 0.0
    liquidation_price: float = 0.0
    max_quantity: float = 0.0
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    funding_paid: float = 0.0
    commission_paid: float = 0.0
    trade_count: int = 0

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.entry_price

    @property
    def is_open(self) -> bool:
        return abs(self.quantity) > 1e-12

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        if not self.is_open:
            return 0.0
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) * abs(self.quantity)
        return 0.0

    def calculate_liquidation_price(self, maintenance_margin_rate: float,
                                     wallet_balance: float) -> float:
        if not self.is_open or abs(self.quantity) < 1e-12:
            return 0.0
        qty = abs(self.quantity)
        if self.side == PositionSide.LONG:
            liq = self.entry_price - (wallet_balance - maintenance_margin_rate * self.notional_value) / qty
            return max(0.0, liq)
        else:
            liq = self.entry_price + (wallet_balance - maintenance_margin_rate * self.notional_value) / qty
            return liq


@dataclass
class Trade:
    """Represents a completed trade (round trip)."""
    trade_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    funding_paid: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    bars_held: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    exit_reason: str = ""
    tag: str = ""


@dataclass
class SimulationEvent:
    """An event in the simulation timeline."""
    timestamp: datetime = None
    event_type: EventType = EventType.BAR
    symbol: str = ""
    data: Any = None
    priority: int = 0

    def __lt__(self, other):
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass
class AccountState:
    """Current state of the simulated account."""
    equity: float = 0.0
    balance: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_funding: float = 0.0
    total_slippage: float = 0.0
    margin_ratio: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class EquityPoint:
    """A single point on the equity curve."""
    timestamp: datetime = None
    equity: float = 0.0
    balance: float = 0.0
    drawdown: float = 0.0
    drawdown_percent: float = 0.0
    positions_count: int = 0
    open_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    config: BacktestConfig = None
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[EquityPoint] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    account_states: List[AccountState] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    total_bars: int = 0
    total_events: int = 0
    final_equity: float = 0.0
    final_balance: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    total_return: float = 0.0
    total_return_percent: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    liquidations: int = 0
    margin_calls: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class Strategy:
    """Base class for trading strategies used in backtesting."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.engine: Optional["BacktestEngine"] = None
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def on_init(self) -> None:
        """Called once before backtest starts. Override to initialize indicators."""
        pass

    def on_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        """Called on each new bar. Override to implement strategy logic."""
        pass

    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """Called on each new tick. Override for tick-level strategies."""
        pass

    def on_order_fill(self, order: Order) -> None:
        """Called when an order is filled."""
        pass

    def on_order_cancel(self, order: Order) -> None:
        """Called when an order is canceled or rejected."""
        pass

    def on_position_close(self, trade: Trade) -> None:
        """Called when a position is fully closed."""
        pass

    def on_liquidation(self, symbol: str) -> None:
        """Called when a position is liquidated."""
        pass

    def on_end(self) -> None:
        """Called after backtest completes. Override for cleanup."""
        pass

    # -- Convenience methods for placing orders --

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None,
            stop_price: Optional[float] = None, reduce_only: bool = False,
            tag: str = "") -> Optional[Order]:
        if price is None and stop_price is None:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=quantity, reduce_only=reduce_only, tag=tag
            )
        elif price is not None and stop_price is None:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.BUY, order_type=OrderType.LIMIT,
                quantity=quantity, price=price, reduce_only=reduce_only, tag=tag
            )
        elif stop_price is not None and price is None:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.BUY, order_type=OrderType.STOP_MARKET,
                quantity=quantity, stop_price=stop_price, reduce_only=reduce_only, tag=tag
            )
        else:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.BUY, order_type=OrderType.STOP_LIMIT,
                quantity=quantity, price=price, stop_price=stop_price,
                reduce_only=reduce_only, tag=tag
            )

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None,
             stop_price: Optional[float] = None, reduce_only: bool = False,
             tag: str = "") -> Optional[Order]:
        if price is None and stop_price is None:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=quantity, reduce_only=reduce_only, tag=tag
            )
        elif price is not None and stop_price is None:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.SELL, order_type=OrderType.LIMIT,
                quantity=quantity, price=price, reduce_only=reduce_only, tag=tag
            )
        elif stop_price is not None and price is None:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.SELL, order_type=OrderType.STOP_MARKET,
                quantity=quantity, stop_price=stop_price, reduce_only=reduce_only, tag=tag
            )
        else:
            return self.engine.place_order(
                symbol=symbol, side=OrderSide.SELL, order_type=OrderType.STOP_LIMIT,
                quantity=quantity, price=price, stop_price=stop_price,
                reduce_only=reduce_only, tag=tag
            )

    def close_position(self, symbol: str, tag: str = "") -> Optional[Order]:
        return self.engine.close_position(symbol, tag=tag)

    def cancel_order(self, order_id: str) -> bool:
        return self.engine.cancel_order(order_id)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        return self.engine.cancel_all_orders(symbol)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.engine.get_position(symbol)

    def get_positions(self) -> Dict[str, Position]:
        return self.engine.positions.copy()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return self.engine.get_open_orders(symbol)

    def get_equity(self) -> float:
        return self.engine.get_equity()

    def get_balance(self) -> float:
        return self.engine.balance

    def get_current_price(self, symbol: str) -> Optional[float]:
        return self.engine.get_current_price(symbol)

    def get_bar_history(self, symbol: str, lookback: int = 100) -> List[Dict[str, Any]]:
        return self.engine.get_bar_history(symbol, lookback)


# ---------------------------------------------------------------------------
# Slippage calculator
# ---------------------------------------------------------------------------

class SlippageCalculator:
    """Calculates slippage for order fills based on the configured model."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._volatility_cache: Dict[str, List[float]] = defaultdict(list)

    def calculate(self, symbol: str, side: OrderSide, price: float,
                  quantity: float, bar: Optional[Dict[str, Any]] = None) -> float:
        model = self.config.slippage_model
        if model == SlippageModel.NONE:
            return 0.0
        elif model == SlippageModel.FIXED:
            return self.config.slippage_value
        elif model == SlippageModel.PERCENTAGE:
            return price * self.config.slippage_value
        elif model == SlippageModel.VOLATILITY:
            return self._volatility_slippage(symbol, price, bar)
        return 0.0

    def _volatility_slippage(self, symbol: str, price: float,
                              bar: Optional[Dict[str, Any]]) -> float:
        if bar is None:
            return price * self.config.slippage_value
        high = bar.get("high", price)
        low = bar.get("low", price)
        bar_range = (high - low) / price if price > 0 else 0.0
        self._volatility_cache[symbol].append(bar_range)
        window = self.config.slippage_volatility_window
        if len(self._volatility_cache[symbol]) > window:
            self._volatility_cache[symbol] = self._volatility_cache[symbol][-window:]
        values = self._volatility_cache[symbol]
        avg_range = sum(values) / len(values) if values else 0.0
        return price * avg_range * self.config.slippage_volatility_multiplier

    def apply_slippage(self, price: float, slippage: float,
                        side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage

    def reset(self) -> None:
        self._volatility_cache.clear()


# ---------------------------------------------------------------------------
# Commission calculator
# ---------------------------------------------------------------------------

class CommissionCalculator:
    """Calculates trading commissions."""

    def __init__(self, config: BacktestConfig):
        self.maker_rate = config.commission_maker
        self.taker_rate = config.commission_taker

    def calculate(self, order_type: OrderType, price: float,
                  quantity: float) -> float:
        notional = abs(price * quantity)
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT,
                          OrderType.TAKE_PROFIT_LIMIT):
            return notional * self.maker_rate
        else:
            return notional * self.taker_rate


# ---------------------------------------------------------------------------
# Funding rate simulator
# ---------------------------------------------------------------------------

class FundingSimulator:
    """Simulates perpetual futures funding rate payments."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.funding_rates: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.interval = timedelta(hours=config.funding_rate_interval_hours)
        self.last_funding_time: Dict[str, datetime] = {}

    def load_funding_rates(self, symbol: str,
                            rates: List[Tuple[datetime, float]]) -> None:
        self.funding_rates[symbol] = sorted(rates, key=lambda x: x[0])

    def get_funding_rate(self, symbol: str, timestamp: datetime) -> float:
        rates = self.funding_rates.get(symbol, [])
        if not rates:
            return 0.0001  # default 0.01%
        best_rate = 0.0001
        for ts, rate in rates:
            if ts <= timestamp:
                best_rate = rate
            else:
                break
        return best_rate

    def should_apply_funding(self, symbol: str, timestamp: datetime) -> bool:
        if not self.config.enable_funding:
            return False
        last = self.last_funding_time.get(symbol)
        if last is None:
            return True
        return (timestamp - last) >= self.interval

    def apply_funding(self, symbol: str, position: Position,
                      timestamp: datetime) -> float:
        if not position.is_open:
            return 0.0
        rate = self.get_funding_rate(symbol, timestamp)
        notional = abs(position.quantity) * position.entry_price
        if position.side == PositionSide.LONG:
            payment = notional * rate
        else:
            payment = -notional * rate
        self.last_funding_time[symbol] = timestamp
        return payment


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks backtest progress and estimates completion time."""

    def __init__(self, total: int, enabled: bool = True):
        self.total = total
        self.enabled = enabled
        self.current = 0
        self.start_time = time.time()
        self._last_print = 0.0
        self._print_interval = 1.0  # seconds

    def update(self, n: int = 1) -> None:
        self.current += n
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_print < self._print_interval:
            return
        self._last_print = now
        self._print_progress()

    def _print_progress(self) -> None:
        if self.total <= 0:
            return
        elapsed = time.time() - self.start_time
        pct = self.current / self.total
        if pct > 0:
            eta = elapsed / pct - elapsed
        else:
            eta = 0
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "=" * filled + "-" * (bar_len - filled)
        eta_str = self._format_time(eta)
        elapsed_str = self._format_time(elapsed)
        print(
            f"\r  [{bar}] {pct*100:5.1f}% | "
            f"{self.current}/{self.total} bars | "
            f"Elapsed: {elapsed_str} | ETA: {eta_str}",
            end="", flush=True
        )

    def finish(self) -> None:
        if self.enabled:
            elapsed = time.time() - self.start_time
            print(
                f"\r  Backtest complete: {self.total} bars in "
                f"{self._format_time(elapsed)}                    "
            )

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Core event-driven backtesting engine.

    Supports multi-symbol, multi-strategy backtesting with realistic simulation
    of order matching, slippage, commissions, funding rates, margin, and
    liquidation.

    Usage:
        config = BacktestConfig(initial_capital=10000, leverage=10)
        engine = BacktestEngine(config)
        engine.add_data("BTCUSDT", bars)
        engine.add_strategy(MyStrategy(params={"fast": 10, "slow": 30}))
        result = engine.run()
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.strategies: List[Strategy] = []
        self.data: Dict[str, List[Dict[str, Any]]] = {}
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []
        self.account_states: List[AccountState] = []
        self.balance: float = self.config.initial_capital
        self.peak_equity: float = self.config.initial_capital
        self.max_drawdown: float = 0.0
        self.max_drawdown_percent: float = 0.0
        self.total_commission: float = 0.0
        self.total_funding: float = 0.0
        self.total_slippage: float = 0.0
        self.liquidation_count: int = 0
        self.margin_call_count: int = 0
        self.current_bar: Dict[str, Dict[str, Any]] = {}
        self.current_bar_index: Dict[str, int] = {}
        self.bar_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_timestamp: Optional[datetime] = None
        self.bar_count: int = 0
        self._slippage = SlippageCalculator(self.config)
        self._commission = CommissionCalculator(self.config)
        self._funding = FundingSimulator(self.config)
        self._is_running = False
        self._errors: List[str] = []
        self._event_count: int = 0
        self._warmup_complete = False
        self._position_entry_bars: Dict[str, int] = {}
        self._trade_max_favorable: Dict[str, float] = defaultdict(float)
        self._trade_max_adverse: Dict[str, float] = defaultdict(float)

    # -- Data management --

    def add_data(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """
        Add historical bar data for a symbol.

        Each bar must be a dict with keys:
            timestamp, open, high, low, close, volume
        Optional keys: quote_volume, trades_count
        """
        if not bars:
            raise ValueError(f"No data provided for {symbol}")
        required_keys = {"timestamp", "open", "high", "low", "close", "volume"}
        sample = bars[0]
        missing = required_keys - set(sample.keys())
        if missing:
            raise ValueError(
                f"Bar data for {symbol} missing keys: {missing}"
            )
        sorted_bars = sorted(bars, key=lambda x: x["timestamp"])
        self.data[symbol] = sorted_bars
        if symbol not in self.config.symbols:
            self.config.symbols.append(symbol)
        logger.info(
            f"Added {len(sorted_bars)} bars for {symbol} "
            f"({sorted_bars[0]['timestamp']} to {sorted_bars[-1]['timestamp']})"
        )

    def add_strategy(self, strategy: Strategy) -> None:
        """Register a strategy for backtesting."""
        strategy.engine = self
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")

    def load_funding_rates(self, symbol: str,
                           rates: List[Tuple[datetime, float]]) -> None:
        """Load historical funding rates for funding simulation."""
        self._funding.load_funding_rates(symbol, rates)

    # -- Order management --

    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    reduce_only: bool = False, tag: str = "",
                    time_in_force: str = "GTC") -> Optional[Order]:
        """Place an order in the simulation."""
        if not self._is_running:
            logger.warning("Cannot place orders outside of a running backtest")
            return None

        # Validate order
        if quantity < self.config.min_order_size:
            logger.warning(
                f"Order quantity {quantity} below minimum {self.config.min_order_size}"
            )
            return None

        if len(self.open_orders) >= self.config.max_open_orders:
            logger.warning("Maximum open orders reached")
            return None

        if symbol not in self.data:
            logger.warning(f"No data loaded for symbol {symbol}")
            return None

        if not self.config.enable_short_selling and side == OrderSide.SELL:
            pos = self.positions.get(symbol)
            if pos is None or not pos.is_open:
                logger.warning("Short selling disabled")
                return None

        # Check reduce-only validity
        if reduce_only:
            pos = self.positions.get(symbol)
            if pos is None or not pos.is_open:
                logger.warning("Reduce-only order but no open position")
                return None

        # Validate limit price
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT,
                          OrderType.TAKE_PROFIT_LIMIT):
            if price is None or price <= 0:
                logger.warning("Limit orders require a valid price")
                return None

        # Validate stop price
        if order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT,
                          OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT):
            if stop_price is None or stop_price <= 0:
                logger.warning("Stop orders require a valid stop price")
                return None

        # Check available margin for non-reduce-only orders
        if not reduce_only:
            required_margin = self._calculate_required_margin(
                symbol, side, quantity, price
            )
            available = self._get_available_margin()
            if required_margin > available * 1.001:  # small tolerance
                logger.warning(
                    f"Insufficient margin: need {required_margin:.2f}, "
                    f"available {available:.2f}"
                )
                return None

        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            time_in_force=time_in_force,
            created_at=self.current_timestamp,
            tag=tag,
        )

        if self.config.order_expiry_bars > 0:
            order.expire_bar = self.bar_count + self.config.order_expiry_bars

        # Market orders try to fill immediately on next bar
        self.open_orders[order.order_id] = order
        self._event_count += 1

        if self.config.verbose:
            logger.info(
                f"Order placed: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {order.order_type.value} "
                f"price={order.price} stop={order.stop_price}"
            )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        order = self.open_orders.pop(order_id, None)
        if order is None:
            return False
        order.status = OrderStatus.CANCELED
        for strategy in self.strategies:
            try:
                strategy.on_order_cancel(order)
            except Exception as e:
                self._errors.append(f"Strategy error on order cancel: {e}")
        return True

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by symbol."""
        to_cancel = []
        for oid, order in self.open_orders.items():
            if symbol is None or order.symbol == symbol:
                to_cancel.append(oid)
        count = 0
        for oid in to_cancel:
            if self.cancel_order(oid):
                count += 1
        return count

    def close_position(self, symbol: str, tag: str = "") -> Optional[Order]:
        """Close an open position with a market order."""
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open:
            return None
        side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        return self.place_order(
            symbol=symbol, side=side, order_type=OrderType.MARKET,
            quantity=abs(pos.quantity), reduce_only=True, tag=tag
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        orders = list(self.open_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current bar's close price for a symbol."""
        bar = self.current_bar.get(symbol)
        if bar:
            return bar["close"]
        return None

    def get_bar_history(self, symbol: str, lookback: int = 100) -> List[Dict[str, Any]]:
        """Get recent bar history for a symbol."""
        history = self.bar_history.get(symbol, [])
        return history[-lookback:] if lookback > 0 else history

    def get_equity(self) -> float:
        """Calculate current total equity (balance + unrealized PnL)."""
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_open:
                price = self.get_current_price(symbol)
                if price:
                    unrealized += pos.calculate_unrealized_pnl(price)
        return self.balance + unrealized

    # -- Internal calculations --

    def _get_available_margin(self) -> float:
        equity = self.get_equity()
        used = sum(
            pos.margin_used for pos in self.positions.values() if pos.is_open
        )
        return equity - used

    def _calculate_required_margin(self, symbol: str, side: OrderSide,
                                    quantity: float,
                                    price: Optional[float]) -> float:
        if price is None:
            cur_price = self.get_current_price(symbol)
            if cur_price is None:
                return 0.0
            price = cur_price
        notional = abs(quantity * price)
        leverage = self.config.leverage
        return notional / leverage

    def _get_fill_price_for_market_order(self, symbol: str, side: OrderSide,
                                          bar: Dict[str, Any]) -> float:
        """Market orders fill at the open of the current bar (next bar after signal)."""
        base_price = bar["open"]
        slippage = self._slippage.calculate(
            symbol, side, base_price, 0, bar
        )
        return self._slippage.apply_slippage(base_price, slippage, side)

    def _check_limit_fill(self, order: Order,
                           bar: Dict[str, Any]) -> Optional[float]:
        """Check if a limit order would fill in this bar. Returns fill price or None."""
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if bar["low"] <= order.price:
                    return min(order.price, bar["open"])
            else:
                if bar["high"] >= order.price:
                    return max(order.price, bar["open"])
        return None

    def _check_stop_trigger(self, order: Order,
                             bar: Dict[str, Any]) -> bool:
        """Check if a stop order's trigger condition is met."""
        if order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            if order.side == OrderSide.BUY:
                return bar["high"] >= order.stop_price
            else:
                return bar["low"] <= order.stop_price
        elif order.order_type in (OrderType.TAKE_PROFIT_MARKET,
                                   OrderType.TAKE_PROFIT_LIMIT):
            if order.side == OrderSide.BUY:
                return bar["low"] <= order.stop_price
            else:
                return bar["high"] >= order.stop_price
        return False

    def _fill_order(self, order: Order, fill_price: float,
                    bar: Dict[str, Any]) -> None:
        """Execute an order fill and update positions."""
        slippage_amount = self._slippage.calculate(
            order.symbol, order.side, fill_price, order.quantity, bar
        )
        actual_price = self._slippage.apply_slippage(
            fill_price, slippage_amount, order.side
        )
        commission = self._commission.calculate(
            order.order_type, actual_price, order.quantity
        )

        order.filled_price = actual_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage_amount
        order.status = OrderStatus.FILLED
        order.filled_at = self.current_timestamp

        self.total_commission += commission
        self.total_slippage += slippage_amount
        self.balance -= commission

        # Update position
        self._update_position(order)

        # Move to filled list
        self.open_orders.pop(order.order_id, None)
        self.filled_orders.append(order)

        # Notify strategies
        for strategy in self.strategies:
            try:
                strategy.on_order_fill(order)
            except Exception as e:
                self._errors.append(f"Strategy error on fill: {e}")

        if self.config.verbose:
            logger.info(
                f"Order filled: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {actual_price:.6f} "
                f"(commission={commission:.4f}, slippage={slippage_amount:.6f})"
            )

    def _update_position(self, order: Order) -> None:
        """Update position state after an order fill."""
        symbol = order.symbol
        pos = self.positions.get(symbol)

        if pos is None or not pos.is_open:
            # New position
            pos = Position(
                symbol=symbol,
                side=PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT,
                entry_price=order.filled_price,
                quantity=order.quantity if order.side == OrderSide.BUY else -order.quantity,
                leverage=self.config.leverage,
                entry_time=self.current_timestamp,
                last_update=self.current_timestamp,
                commission_paid=order.commission,
                trade_count=1,
            )
            pos.max_quantity = abs(pos.quantity)
            pos.margin_used = abs(pos.quantity * pos.entry_price) / self.config.leverage
            self.positions[symbol] = pos
            self._position_entry_bars[symbol] = self.bar_count
            self._trade_max_favorable[symbol] = 0.0
            self._trade_max_adverse[symbol] = 0.0
            return

        # Existing position - determine if adding or reducing
        old_qty = pos.quantity
        if order.side == OrderSide.BUY:
            new_qty = old_qty + order.quantity
        else:
            new_qty = old_qty - order.quantity

        # Check if position is flipping or closing
        if abs(new_qty) < 1e-12:
            # Position fully closed
            self._close_position_record(pos, order)
            return

        if (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
            # Position flipped - close old position first, open new
            self._close_position_record(pos, order, partial_qty=abs(old_qty))
            # Open new position with remaining quantity
            remaining = abs(new_qty)
            new_pos = Position(
                symbol=symbol,
                side=PositionSide.LONG if new_qty > 0 else PositionSide.SHORT,
                entry_price=order.filled_price,
                quantity=new_qty,
                leverage=self.config.leverage,
                entry_time=self.current_timestamp,
                last_update=self.current_timestamp,
                commission_paid=order.commission * (remaining / order.quantity),
                trade_count=1,
            )
            new_pos.max_quantity = abs(new_qty)
            new_pos.margin_used = abs(new_qty * order.filled_price) / self.config.leverage
            self.positions[symbol] = new_pos
            self._position_entry_bars[symbol] = self.bar_count
            self._trade_max_favorable[symbol] = 0.0
            self._trade_max_adverse[symbol] = 0.0
            return

        if (old_qty > 0 and order.side == OrderSide.BUY) or \
           (old_qty < 0 and order.side == OrderSide.SELL):
            # Adding to position - calculate new average entry
            old_notional = abs(old_qty) * pos.entry_price
            add_notional = order.quantity * order.filled_price
            total_qty = abs(old_qty) + order.quantity
            pos.entry_price = (old_notional + add_notional) / total_qty
            pos.quantity = new_qty
            pos.max_quantity = max(pos.max_quantity, abs(new_qty))
            pos.margin_used = abs(new_qty * pos.entry_price) / self.config.leverage
            pos.commission_paid += order.commission
            pos.trade_count += 1
            pos.last_update = self.current_timestamp
        else:
            # Partial close
            close_qty = min(order.quantity, abs(old_qty))
            pnl = self._calculate_trade_pnl(pos, order.filled_price, close_qty)
            self.balance += pnl
            pos.realized_pnl += pnl
            pos.quantity = new_qty
            pos.margin_used = abs(new_qty * pos.entry_price) / self.config.leverage
            pos.commission_paid += order.commission
            pos.trade_count += 1
            pos.last_update = self.current_timestamp

    def _close_position_record(self, pos: Position, order: Order,
                                partial_qty: Optional[float] = None) -> None:
        """Record a trade when a position is fully closed."""
        close_qty = partial_qty if partial_qty else abs(pos.quantity)
        pnl = self._calculate_trade_pnl(pos, order.filled_price, close_qty)
        self.balance += pnl

        entry_bar = self._position_entry_bars.get(pos.symbol, 0)
        bars_held = self.bar_count - entry_bar

        trade = Trade(
            trade_id=str(uuid.uuid4())[:12],
            symbol=pos.symbol,
            side="LONG" if pos.side == PositionSide.LONG else "SHORT",
            entry_price=pos.entry_price,
            exit_price=order.filled_price,
            quantity=close_qty,
            pnl=pnl - pos.commission_paid - order.commission - pos.funding_paid,
            pnl_percent=(pnl / (close_qty * pos.entry_price)) * 100 if pos.entry_price > 0 else 0.0,
            commission=pos.commission_paid + order.commission,
            slippage=order.slippage,
            funding_paid=pos.funding_paid,
            entry_time=pos.entry_time,
            exit_time=self.current_timestamp,
            duration=(self.current_timestamp - pos.entry_time) if pos.entry_time and self.current_timestamp else None,
            bars_held=bars_held,
            max_favorable=self._trade_max_favorable.get(pos.symbol, 0.0),
            max_adverse=self._trade_max_adverse.get(pos.symbol, 0.0),
            exit_reason=order.tag or "signal",
            tag=order.tag,
        )
        self.trades.append(trade)

        # Clear position
        self.positions[pos.symbol] = Position(symbol=pos.symbol)
        self._trade_max_favorable.pop(pos.symbol, None)
        self._trade_max_adverse.pop(pos.symbol, None)

        # Notify strategies
        for strategy in self.strategies:
            try:
                strategy.on_position_close(trade)
            except Exception as e:
                self._errors.append(f"Strategy error on position close: {e}")

    def _calculate_trade_pnl(self, pos: Position, exit_price: float,
                              quantity: float) -> float:
        if pos.side == PositionSide.LONG:
            return (exit_price - pos.entry_price) * quantity
        else:
            return (pos.entry_price - exit_price) * quantity

    def _update_position_tracking(self, symbol: str, price: float) -> None:
        """Update max favorable/adverse excursion for a position."""
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open:
            return
        pnl_pct = pos.calculate_unrealized_pnl(price) / (abs(pos.quantity) * pos.entry_price) * 100
        if pnl_pct > 0:
            self._trade_max_favorable[symbol] = max(
                self._trade_max_favorable.get(symbol, 0.0), pnl_pct
            )
        else:
            self._trade_max_adverse[symbol] = max(
                self._trade_max_adverse.get(symbol, 0.0), abs(pnl_pct)
            )

    # -- Margin and liquidation --

    def _check_liquidation(self, symbol: str, price: float) -> bool:
        """Check if a position should be liquidated at the given price."""
        if not self.config.enable_liquidation:
            return False
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open:
            return False

        unrealized = pos.calculate_unrealized_pnl(price)
        notional = abs(pos.quantity) * price
        maintenance = notional * self.config.maintenance_margin_rate

        if pos.margin_used + unrealized <= maintenance:
            return True
        return False

    def _execute_liquidation(self, symbol: str, price: float) -> None:
        """Force-liquidate a position."""
        pos = self.positions.get(symbol)
        if pos is None or not pos.is_open:
            return

        # Liquidation happens at the current price
        pnl = self._calculate_trade_pnl(pos, price, abs(pos.quantity))
        liquidation_fee = abs(pos.quantity) * price * self.config.commission_taker

        entry_bar = self._position_entry_bars.get(symbol, 0)
        bars_held = self.bar_count - entry_bar

        trade = Trade(
            trade_id=str(uuid.uuid4())[:12],
            symbol=symbol,
            side="LONG" if pos.side == PositionSide.LONG else "SHORT",
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=abs(pos.quantity),
            pnl=pnl - pos.commission_paid - liquidation_fee - pos.funding_paid,
            pnl_percent=(pnl / (abs(pos.quantity) * pos.entry_price)) * 100 if pos.entry_price > 0 else 0.0,
            commission=pos.commission_paid + liquidation_fee,
            funding_paid=pos.funding_paid,
            entry_time=pos.entry_time,
            exit_time=self.current_timestamp,
            duration=(self.current_timestamp - pos.entry_time) if pos.entry_time and self.current_timestamp else None,
            bars_held=bars_held,
            max_favorable=self._trade_max_favorable.get(symbol, 0.0),
            max_adverse=self._trade_max_adverse.get(symbol, 0.0),
            exit_reason="liquidation",
        )
        self.trades.append(trade)

        self.balance += pnl
        self.balance -= liquidation_fee
        self.total_commission += liquidation_fee
        self.liquidation_count += 1

        # Cancel all orders for this symbol
        self.cancel_all_orders(symbol)

        # Clear position
        self.positions[symbol] = Position(symbol=symbol)
        self._trade_max_favorable.pop(symbol, None)
        self._trade_max_adverse.pop(symbol, None)

        logger.warning(f"LIQUIDATION: {symbol} at {price}")

        for strategy in self.strategies:
            try:
                strategy.on_liquidation(symbol)
            except Exception as e:
                self._errors.append(f"Strategy error on liquidation: {e}")

    def _check_margin_call(self) -> None:
        """Check if total margin usage exceeds the margin call threshold."""
        if not self.config.enable_margin_call:
            return
        equity = self.get_equity()
        used_margin = sum(
            pos.margin_used for pos in self.positions.values() if pos.is_open
        )
        if used_margin <= 0 or equity <= 0:
            return
        ratio = used_margin / equity
        if ratio >= self.config.margin_call_threshold:
            self.margin_call_count += 1
            logger.warning(
                f"MARGIN CALL: ratio={ratio:.2%}, "
                f"equity={equity:.2f}, used={used_margin:.2f}"
            )

    # -- Order processing --

    def _process_orders(self, symbol: str, bar: Dict[str, Any]) -> None:
        """Process all open orders against the current bar."""
        to_process = [
            (oid, o) for oid, o in self.open_orders.items()
            if o.symbol == symbol
        ]

        for order_id, order in to_process:
            if order_id not in self.open_orders:
                continue

            # Check expiry
            if self.config.order_expiry_bars > 0 and \
               order.expire_bar > 0 and \
               self.bar_count >= order.expire_bar:
                order.status = OrderStatus.EXPIRED
                self.open_orders.pop(order_id, None)
                for strategy in self.strategies:
                    try:
                        strategy.on_order_cancel(order)
                    except Exception as e:
                        self._errors.append(f"Strategy error on expiry: {e}")
                continue

            # Process by order type
            if order.order_type == OrderType.MARKET:
                fill_price = self._get_fill_price_for_market_order(
                    symbol, order.side, bar
                )
                self._fill_order(order, fill_price, bar)

            elif order.order_type == OrderType.LIMIT:
                fill_price = self._check_limit_fill(order, bar)
                if fill_price is not None:
                    self._fill_order(order, fill_price, bar)

            elif order.order_type in (OrderType.STOP_MARKET,
                                       OrderType.TAKE_PROFIT_MARKET):
                if self._check_stop_trigger(order, bar):
                    fill_price = order.stop_price
                    self._fill_order(order, fill_price, bar)

            elif order.order_type in (OrderType.STOP_LIMIT,
                                       OrderType.TAKE_PROFIT_LIMIT):
                if self._check_stop_trigger(order, bar):
                    # Convert to limit order behavior
                    if order.side == OrderSide.BUY:
                        if bar["low"] <= order.price:
                            self._fill_order(order, order.price, bar)
                    else:
                        if bar["high"] >= order.price:
                            self._fill_order(order, order.price, bar)

    # -- Equity tracking --

    def _record_equity(self) -> None:
        """Record current equity state."""
        equity = self.get_equity()
        unrealized = equity - self.balance

        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = self.peak_equity - equity
        drawdown_pct = (drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0.0

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        if drawdown_pct > self.max_drawdown_percent:
            self.max_drawdown_percent = drawdown_pct

        positions_count = sum(
            1 for p in self.positions.values() if p.is_open
        )

        point = EquityPoint(
            timestamp=self.current_timestamp,
            equity=equity,
            balance=self.balance,
            drawdown=drawdown,
            drawdown_percent=drawdown_pct,
            positions_count=positions_count,
            open_pnl=unrealized,
        )
        self.equity_curve.append(point)

        used_margin = sum(
            pos.margin_used for pos in self.positions.values() if pos.is_open
        )
        state = AccountState(
            equity=equity,
            balance=self.balance,
            available_margin=equity - used_margin,
            used_margin=used_margin,
            unrealized_pnl=unrealized,
            realized_pnl=self.balance - self.config.initial_capital,
            total_commission=self.total_commission,
            total_funding=self.total_funding,
            total_slippage=self.total_slippage,
            margin_ratio=(used_margin / equity) if equity > 0 else 0.0,
            timestamp=self.current_timestamp,
        )
        self.account_states.append(state)

    # -- Main simulation loop --

    def run(self) -> BacktestResult:
        """
        Run the backtest simulation.

        Returns a BacktestResult containing all trades, equity curve,
        and performance metrics.
        """
        run_start = time.time()
        self._is_running = True
        self._validate_setup()

        # Initialize strategies
        for strategy in self.strategies:
            try:
                strategy.on_init()
            except Exception as e:
                self._errors.append(f"Strategy init error: {e}")
                logger.error(f"Strategy init error: {e}")

        # Build unified timeline
        timeline = self._build_timeline()
        total_bars = len(timeline)

        if total_bars == 0:
            logger.error("No bars in timeline")
            self._is_running = False
            return self._build_result(run_start)

        logger.info(
            f"Starting backtest: {total_bars} events, "
            f"{len(self.config.symbols)} symbols, "
            f"{len(self.strategies)} strategies"
        )

        progress = ProgressTracker(total_bars, self.config.progress_bar)

        warmup = self.config.warmup_bars
        self._warmup_complete = warmup <= 0

        for idx, (timestamp, symbol, bar) in enumerate(timeline):
            self.current_timestamp = timestamp
            self.current_bar[symbol] = bar
            self.current_bar_index[symbol] = idx
            self.bar_history[symbol].append(bar)
            self.bar_count = idx

            # Check warmup
            if not self._warmup_complete:
                min_history = min(
                    len(self.bar_history[s]) for s in self.config.symbols
                    if s in self.bar_history
                )
                if min_history >= warmup:
                    self._warmup_complete = True

            # 1. Process pending orders
            self._process_orders(symbol, bar)

            # 2. Check liquidation for all positions
            for pos_symbol in list(self.positions.keys()):
                pos_bar = self.current_bar.get(pos_symbol)
                if pos_bar:
                    price = pos_bar["close"]
                    # Check using high/low for intra-bar liquidation
                    if self.positions.get(pos_symbol) and \
                       self.positions[pos_symbol].is_open:
                        pos = self.positions[pos_symbol]
                        if pos.side == PositionSide.LONG:
                            check_price = pos_bar["low"]
                        else:
                            check_price = pos_bar["high"]
                        if self._check_liquidation(pos_symbol, check_price):
                            self._execute_liquidation(pos_symbol, check_price)

            # 3. Apply funding rates
            if self.config.enable_funding:
                for pos_symbol, pos in self.positions.items():
                    if pos.is_open and \
                       self._funding.should_apply_funding(pos_symbol, timestamp):
                        payment = self._funding.apply_funding(
                            pos_symbol, pos, timestamp
                        )
                        self.balance -= payment
                        pos.funding_paid += payment
                        self.total_funding += payment

            # 4. Check margin call
            self._check_margin_call()

            # 5. Update position tracking
            for pos_symbol in self.positions:
                pos_bar = self.current_bar.get(pos_symbol)
                if pos_bar:
                    self._update_position_tracking(pos_symbol, pos_bar["close"])

            # 6. Fire strategy callbacks (only after warmup)
            if self._warmup_complete:
                for strategy in self.strategies:
                    try:
                        strategy.on_bar(symbol, bar)
                    except Exception as e:
                        self._errors.append(
                            f"Strategy error on bar {idx}: {e}"
                        )
                        if self.config.verbose:
                            logger.error(f"Strategy error on bar {idx}: {e}")

            # 7. Record equity
            self._record_equity()
            self._event_count += 1
            progress.update()

        progress.finish()

        # Close all remaining positions at last known price
        self._close_all_positions_at_end()

        # Notify strategies of completion
        for strategy in self.strategies:
            try:
                strategy.on_end()
            except Exception as e:
                self._errors.append(f"Strategy end error: {e}")

        self._is_running = False
        return self._build_result(run_start)

    def _validate_setup(self) -> None:
        """Validate that the engine is properly configured before running."""
        if not self.data:
            raise ValueError("No data loaded. Call add_data() first.")
        if not self.strategies:
            raise ValueError("No strategies added. Call add_strategy() first.")
        if self.config.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.config.leverage < 1:
            raise ValueError("Leverage must be >= 1")
        if self.config.leverage > self.config.max_leverage:
            raise ValueError(
                f"Leverage {self.config.leverage} exceeds max {self.config.max_leverage}"
            )

    def _build_timeline(self) -> List[Tuple[datetime, str, Dict[str, Any]]]:
        """
        Build a unified, time-sorted timeline of all bars across all symbols.
        Applies date filters if configured.
        """
        timeline = []
        for symbol, bars in self.data.items():
            for bar in bars:
                ts = bar["timestamp"]
                if isinstance(ts, (int, float)):
                    ts = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)
                    bar["timestamp"] = ts
                if self.config.start_date and ts < self.config.start_date:
                    continue
                if self.config.end_date and ts > self.config.end_date:
                    continue
                timeline.append((ts, symbol, bar))

        timeline.sort(key=lambda x: x[0])
        return timeline

    def _close_all_positions_at_end(self) -> None:
        """Close all remaining positions at the last known price."""
        for symbol, pos in list(self.positions.items()):
            if pos.is_open:
                price = self.get_current_price(symbol)
                if price:
                    side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
                    # Directly create and fill a closing order
                    order = Order(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=abs(pos.quantity),
                        reduce_only=True,
                        tag="backtest_end",
                    )
                    order.filled_price = price
                    order.filled_quantity = abs(pos.quantity)
                    order.commission = self._commission.calculate(
                        OrderType.MARKET, price, abs(pos.quantity)
                    )
                    order.status = OrderStatus.FILLED
                    order.filled_at = self.current_timestamp
                    order.created_at = self.current_timestamp

                    self.total_commission += order.commission
                    self.balance -= order.commission
                    self._close_position_record(pos, order)
                    self.filled_orders.append(order)

    def _build_result(self, run_start: float) -> BacktestResult:
        """Compile all results into a BacktestResult object."""
        execution_time = time.time() - run_start
        final_equity = self.get_equity()

        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        start_time = self.equity_curve[0].timestamp if self.equity_curve else None
        end_time = self.equity_curve[-1].timestamp if self.equity_curve else None

        total_return = final_equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital * 100) \
            if self.config.initial_capital > 0 else 0.0

        result = BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            orders=self.filled_orders,
            account_states=self.account_states,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            total_bars=self.bar_count,
            total_events=self._event_count,
            final_equity=final_equity,
            final_balance=self.balance,
            peak_equity=self.peak_equity,
            max_drawdown=self.max_drawdown,
            max_drawdown_percent=self.max_drawdown_percent,
            total_return=total_return,
            total_return_percent=total_return_pct,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            liquidations=self.liquidation_count,
            margin_calls=self.margin_call_count,
            errors=self._errors,
            metadata={
                "symbols": self.config.symbols,
                "strategies": [s.name for s in self.strategies],
                "total_commission": self.total_commission,
                "total_funding": self.total_funding,
                "total_slippage": self.total_slippage,
            },
        )

        logger.info(
            f"Backtest complete: {len(self.trades)} trades, "
            f"return={total_return_pct:.2f}%, "
            f"max_dd={self.max_drawdown_percent:.2f}%, "
            f"time={execution_time:.1f}s"
        )

        return result

    def reset(self) -> None:
        """Reset the engine for a new backtest run (keeps data and strategies)."""
        self.positions.clear()
        self.open_orders.clear()
        self.filled_orders.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.account_states.clear()
        self.balance = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_percent = 0.0
        self.total_commission = 0.0
        self.total_funding = 0.0
        self.total_slippage = 0.0
        self.liquidation_count = 0
        self.margin_call_count = 0
        self.current_bar.clear()
        self.current_bar_index.clear()
        self.bar_history.clear()
        self.current_timestamp = None
        self.bar_count = 0
        self._slippage.reset()
        self._is_running = False
        self._errors.clear()
        self._event_count = 0
        self._warmup_complete = False
        self._position_entry_bars.clear()
        self._trade_max_favorable.clear()
        self._trade_max_adverse.clear()
