# mefai-autotrade - Abstract exchange base and unified data models
# All exchange connectors inherit from ExchangeBase

import asyncio
import hashlib
import hmac
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"
    LIMIT_MAKER = "LIMIT_MAKER"
    OCO = "OCO"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class MarginMode(str, Enum):
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good till cancel
    IOC = "IOC"  # Immediate or cancel
    FOK = "FOK"  # Fill or kill
    GTX = "GTX"  # Post only
    GTD = "GTD"  # Good till date


class KlineInterval(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MO1 = "1M"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ExchangeError(Exception):
    """Base error for all exchange operations."""

    def __init__(self, message: str, code: Optional[int] = None, exchange: str = ""):
        self.code = code
        self.exchange = exchange
        super().__init__(f"[{exchange}] {message}" if exchange else message)


class RateLimitError(ExchangeError):
    """Rate limit exceeded - caller should wait and retry."""

    def __init__(self, message: str, retry_after: float = 0.0, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class InsufficientBalanceError(ExchangeError):
    """Not enough balance to place the order."""
    pass


class OrderNotFoundError(ExchangeError):
    """Order does not exist or has already been canceled."""
    pass


class InvalidOrderError(ExchangeError):
    """Order parameters are invalid (bad quantity, price, etc)."""
    pass


class AuthenticationError(ExchangeError):
    """API key or signature is invalid."""
    pass


class NetworkError(ExchangeError):
    """Network connectivity issue."""
    pass


# ---------------------------------------------------------------------------
# Unified data models
# ---------------------------------------------------------------------------

@dataclass
class UnifiedTicker:
    """Normalized ticker across all exchanges."""
    symbol: str              # Unified symbol e.g. BTC/USDT
    exchange: str            # Exchange name
    last_price: float        # Last traded price
    bid_price: float         # Best bid
    ask_price: float         # Best ask
    bid_qty: float           # Bid quantity at best bid
    ask_qty: float           # Ask quantity at best ask
    high_24h: float          # 24h high
    low_24h: float           # 24h low
    volume_24h: float        # 24h volume in base asset
    quote_volume_24h: float  # 24h volume in quote asset
    price_change_24h: float  # Absolute change
    price_change_pct: float  # Percentage change
    timestamp: int           # UTC milliseconds
    raw: Optional[Dict[str, Any]] = None


@dataclass
class OrderbookLevel:
    """Single price level in the orderbook."""
    price: float
    quantity: float


@dataclass
class UnifiedOrderbook:
    """Normalized orderbook."""
    symbol: str
    exchange: str
    bids: List[OrderbookLevel]  # Sorted descending by price
    asks: List[OrderbookLevel]  # Sorted ascending by price
    timestamp: int              # UTC milliseconds
    last_update_id: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def spread(self) -> float:
        if self.asks and self.bids:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.asks and self.bids and self.bids[0].price > 0:
            return (self.spread / self.bids[0].price) * 100.0
        return 0.0

    @property
    def mid_price(self) -> float:
        if self.asks and self.bids:
            return (self.asks[0].price + self.bids[0].price) / 2.0
        return 0.0


@dataclass
class UnifiedKline:
    """Normalized candlestick/kline."""
    symbol: str
    exchange: str
    interval: str
    open_time: int     # UTC ms
    close_time: int    # UTC ms
    open: float
    high: float
    low: float
    close: float
    volume: float          # Base asset volume
    quote_volume: float    # Quote asset volume
    trades: int            # Number of trades
    taker_buy_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0
    is_closed: bool = True
    raw: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedTrade:
    """Normalized public trade."""
    symbol: str
    exchange: str
    trade_id: str
    price: float
    quantity: float
    quote_quantity: float
    side: OrderSide
    timestamp: int  # UTC ms
    is_maker: bool = False
    raw: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedOrder:
    """Normalized order."""
    symbol: str
    exchange: str
    order_id: str
    client_order_id: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    price: float           # Limit price (0 for market)
    stop_price: float      # Stop/trigger price
    quantity: float        # Ordered quantity
    filled_quantity: float
    remaining_quantity: float
    average_price: float   # Average fill price
    commission: float      # Total commission paid
    commission_asset: str  # Commission currency
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    post_only: bool = False
    created_at: int = 0    # UTC ms
    updated_at: int = 0    # UTC ms
    raw: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedPosition:
    """Normalized futures position."""
    symbol: str
    exchange: str
    side: PositionSide
    size: float             # Absolute position size
    entry_price: float
    mark_price: float
    liquidation_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    margin_mode: MarginMode
    margin: float           # Position margin
    notional: float         # Position notional value
    adl_quantile: float     # Auto-deleverage indicator (0-1)
    timestamp: int          # UTC ms
    raw: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedBalance:
    """Normalized balance for a single asset."""
    asset: str
    exchange: str
    free: float        # Available for trading
    locked: float      # In open orders
    total: float       # free + locked
    usd_value: float   # Approximate USD value
    raw: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedAccountInfo:
    """Normalized account information."""
    exchange: str
    account_type: str           # SPOT, FUTURES, MARGIN, etc
    can_trade: bool
    can_withdraw: bool
    can_deposit: bool
    balances: List[UnifiedBalance]
    total_usd_value: float
    # Futures specific
    total_margin: float = 0.0
    available_margin: float = 0.0
    total_unrealized_pnl: float = 0.0
    margin_ratio: float = 0.0  # Maintenance margin ratio
    positions: List[UnifiedPosition] = field(default_factory=list)
    timestamp: int = 0
    raw: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Exchange config
# ---------------------------------------------------------------------------

@dataclass
class ExchangeConfig:
    """Configuration for an exchange connection."""
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""       # OKX requires passphrase
    testnet: bool = False
    recv_window: int = 5000
    timeout: int = 10          # HTTP timeout seconds
    max_retries: int = 3
    retry_delay: float = 1.0   # Base delay between retries
    proxy: Optional[str] = None
    rate_limit_per_second: float = 10.0
    rate_limit_per_minute: float = 1200.0
    ws_ping_interval: float = 20.0
    ws_ping_timeout: float = 10.0
    enable_ws_reconnect: bool = True
    ws_reconnect_max_attempts: int = 10
    ws_reconnect_base_delay: float = 1.0
    ws_reconnect_max_delay: float = 60.0
    custom_headers: Dict[str, str] = field(default_factory=dict)
    base_url_override: Optional[str] = None
    ws_url_override: Optional[str] = None


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Token-bucket rate limiter with per-second and per-minute budgets.

    Supports both sync and async usage. Tracks request weights so that
    heavier endpoints consume more tokens from the bucket.
    """

    def __init__(
        self,
        per_second: float = 10.0,
        per_minute: float = 1200.0,
        name: str = "default",
    ):
        self.name = name
        self.per_second = per_second
        self.per_minute = per_minute
        self._second_tokens = per_second
        self._minute_tokens = per_minute
        self._second_last_refill = time.monotonic()
        self._minute_last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_requests = 0
        self._total_waits = 0
        self._total_wait_time = 0.0

    def _refill(self) -> None:
        now = time.monotonic()

        # Refill second bucket
        elapsed_s = now - self._second_last_refill
        if elapsed_s > 0:
            self._second_tokens = min(
                self.per_second,
                self._second_tokens + elapsed_s * self.per_second,
            )
            self._second_last_refill = now

        # Refill minute bucket
        elapsed_m = now - self._minute_last_refill
        if elapsed_m > 0:
            self._minute_tokens = min(
                self.per_minute,
                self._minute_tokens + elapsed_m * (self.per_minute / 60.0),
            )
            self._minute_last_refill = now

    async def acquire(self, weight: int = 1) -> float:
        """Acquire tokens. Returns the time waited (seconds).

        If the bucket is empty the coroutine sleeps until enough tokens
        are available, then deducts the weight.
        """
        total_waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._second_tokens >= weight and self._minute_tokens >= weight:
                    self._second_tokens -= weight
                    self._minute_tokens -= weight
                    self._total_requests += 1
                    return total_waited

                # Calculate how long to wait
                wait_second = 0.0
                wait_minute = 0.0
                if self._second_tokens < weight:
                    deficit = weight - self._second_tokens
                    wait_second = deficit / self.per_second
                if self._minute_tokens < weight:
                    deficit = weight - self._minute_tokens
                    wait_minute = deficit / (self.per_minute / 60.0)
                wait = max(wait_second, wait_minute, 0.01)

                self._total_waits += 1
                self._total_wait_time += wait
                total_waited += wait

                logger.debug(
                    "RateLimiter[%s] throttled for %.3fs (weight=%d, s=%.1f, m=%.1f)",
                    self.name, wait, weight,
                    self._second_tokens, self._minute_tokens,
                )
                # Release lock while sleeping so other tasks can check too
                # We re-acquire and re-check in the next iteration
                self._lock.release()
                try:
                    await asyncio.sleep(wait)
                finally:
                    await self._lock.acquire()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
            "total_wait_time": round(self._total_wait_time, 3),
            "second_tokens": round(self._second_tokens, 2),
            "minute_tokens": round(self._minute_tokens, 2),
        }

    def reset(self) -> None:
        self._second_tokens = self.per_second
        self._minute_tokens = self.per_minute
        self._second_last_refill = time.monotonic()
        self._minute_last_refill = time.monotonic()


# ---------------------------------------------------------------------------
# Request signing utilities
# ---------------------------------------------------------------------------

def hmac_sha256_sign(secret: str, message: str) -> str:
    """HMAC-SHA256 signature used by Binance and Bybit."""
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def build_query_string(params: Dict[str, Any]) -> str:
    """Build a sorted query string from parameters, filtering None values."""
    filtered = {k: v for k, v in params.items() if v is not None}
    return urlencode(filtered)


def current_timestamp_ms() -> int:
    """Current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Connection health
# ---------------------------------------------------------------------------

@dataclass
class ConnectionHealthStatus:
    """Health status for an exchange connection."""
    exchange: str
    connected: bool
    latency_ms: float           # Last measured REST latency
    avg_latency_ms: float       # Average over recent window
    ws_connected: bool
    ws_subscriptions: int
    last_heartbeat: int         # UTC ms
    errors_last_5m: int
    rate_limit_remaining: float # Percentage of rate limit remaining
    uptime_seconds: float
    status: str                 # "healthy", "degraded", "disconnected"


class HealthMonitor:
    """Tracks exchange connection health metrics."""

    def __init__(self, exchange_name: str, window_size: int = 100):
        self.exchange_name = exchange_name
        self._window_size = window_size
        self._latencies: List[float] = []
        self._errors: List[Tuple[float, str]] = []
        self._start_time = time.monotonic()
        self._last_heartbeat = 0
        self._ws_connected = False
        self._ws_subscriptions = 0
        self._rest_connected = False

    def record_latency(self, latency_ms: float) -> None:
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._window_size:
            self._latencies.pop(0)
        self._rest_connected = True

    def record_error(self, error_msg: str) -> None:
        self._errors.append((time.monotonic(), error_msg))
        # Prune old errors (older than 5 minutes)
        cutoff = time.monotonic() - 300
        self._errors = [(t, m) for t, m in self._errors if t >= cutoff]

    def record_heartbeat(self) -> None:
        self._last_heartbeat = current_timestamp_ms()

    def set_ws_status(self, connected: bool, subscriptions: int = 0) -> None:
        self._ws_connected = connected
        self._ws_subscriptions = subscriptions

    @property
    def avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def last_latency(self) -> float:
        return self._latencies[-1] if self._latencies else 0.0

    @property
    def errors_last_5m(self) -> int:
        cutoff = time.monotonic() - 300
        return sum(1 for t, _ in self._errors if t >= cutoff)

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._start_time

    def get_status(self, rate_limiter: Optional[RateLimiter] = None) -> ConnectionHealthStatus:
        rl_remaining = 100.0
        if rate_limiter:
            rate_limiter._refill()
            rl_remaining = (rate_limiter._minute_tokens / rate_limiter.per_minute) * 100

        error_count = self.errors_last_5m
        if not self._rest_connected and not self._ws_connected:
            status = "disconnected"
        elif error_count > 10 or (rate_limiter and rl_remaining < 10):
            status = "degraded"
        else:
            status = "healthy"

        return ConnectionHealthStatus(
            exchange=self.exchange_name,
            connected=self._rest_connected,
            latency_ms=self.last_latency,
            avg_latency_ms=round(self.avg_latency, 2),
            ws_connected=self._ws_connected,
            ws_subscriptions=self._ws_subscriptions,
            last_heartbeat=self._last_heartbeat,
            errors_last_5m=error_count,
            rate_limit_remaining=round(rl_remaining, 1),
            uptime_seconds=round(self.uptime, 1),
            status=status,
        )


# ---------------------------------------------------------------------------
# Retry decorator for exchange operations
# ---------------------------------------------------------------------------

async def retry_async(
    coro_factory: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: Tuple = (NetworkError, RateLimitError),
    exchange_name: str = "",
) -> Any:
    """Retry an async operation with exponential backoff.

    Args:
        coro_factory: Callable that returns a coroutine (called each attempt).
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries.
        retryable: Tuple of exception classes that trigger a retry.
        exchange_name: Name for logging purposes.

    Returns:
        The result of the coroutine.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable as exc:
            last_exc = exc
            if attempt == max_retries:
                logger.error(
                    "[%s] All %d retries exhausted: %s",
                    exchange_name, max_retries, exc,
                )
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if isinstance(exc, RateLimitError) and exc.retry_after > 0:
                delay = max(delay, exc.retry_after)

            logger.warning(
                "[%s] Attempt %d/%d failed (%s), retrying in %.1fs",
                exchange_name, attempt + 1, max_retries, exc, delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore


# ---------------------------------------------------------------------------
# Abstract exchange base
# ---------------------------------------------------------------------------

class ExchangeBase(ABC):
    """Abstract base class for all exchange connectors.

    Every exchange connector must implement all abstract methods below.
    The base class provides rate limiting, health monitoring, and common
    utilities. Subclasses should call super().__init__() and then set
    up exchange-specific HTTP sessions and WebSocket connections.
    """

    def __init__(self, config: ExchangeConfig, exchange_name: str):
        self.config = config
        self.exchange_name = exchange_name
        self._rate_limiter = RateLimiter(
            per_second=config.rate_limit_per_second,
            per_minute=config.rate_limit_per_minute,
            name=exchange_name,
        )
        self._health = HealthMonitor(exchange_name)
        self._initialized = False
        self._ws_callbacks: Dict[str, List[Callable]] = {}
        self._exchange_info_cache: Dict[str, Any] = {}
        self._exchange_info_ts: float = 0.0
        self._exchange_info_ttl: float = 300.0  # 5 min cache
        self._logger = logging.getLogger(f"exchange.{exchange_name}")

    # -- Lifecycle ----------------------------------------------------------

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the exchange connection (HTTP session, load exchange info)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Gracefully close all connections (HTTP + WebSocket)."""
        ...

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # -- Market data --------------------------------------------------------

    @abstractmethod
    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        """Get current ticker for a symbol."""
        ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> UnifiedOrderbook:
        """Get current orderbook snapshot."""
        ...

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedKline]:
        """Get historical klines/candlesticks."""
        ...

    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 500) -> List[UnifiedTrade]:
        """Get recent public trades."""
        ...

    # -- Account ------------------------------------------------------------

    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> List[UnifiedBalance]:
        """Get account balances. If asset is specified, return only that one."""
        ...

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[UnifiedPosition]:
        """Get open positions (futures only). Returns empty for spot."""
        ...

    @abstractmethod
    async def get_account_info(self) -> UnifiedAccountInfo:
        """Get full account information including balances and positions."""
        ...

    # -- Orders -------------------------------------------------------------

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        post_only: bool = False,
        client_order_id: Optional[str] = None,
        position_side: Optional[PositionSide] = None,
        callback_rate: Optional[float] = None,
        **kwargs,
    ) -> UnifiedOrder:
        """Place a new order."""
        ...

    @abstractmethod
    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> UnifiedOrder:
        """Cancel an existing order."""
        ...

    @abstractmethod
    async def modify_order(
        self,
        symbol: str,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> UnifiedOrder:
        """Modify an existing order (exchange support varies)."""
        ...

    @abstractmethod
    async def get_order(
        self, symbol: str, order_id: str
    ) -> UnifiedOrder:
        """Get order details by ID."""
        ...

    @abstractmethod
    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[UnifiedOrder]:
        """Get all open orders, optionally filtered by symbol."""
        ...

    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedOrder]:
        """Get historical orders."""
        ...

    # -- Futures specific ---------------------------------------------------

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol. Returns True on success."""
        ...

    @abstractmethod
    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        """Set margin mode (isolated/cross). Returns True on success."""
        ...

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Get current funding rate info."""
        ...

    @abstractmethod
    async def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        """Get current mark price and index price."""
        ...

    # -- WebSocket ----------------------------------------------------------

    @abstractmethod
    async def subscribe_ticker(
        self, symbol: str, callback: Callable[[UnifiedTicker], Any]
    ) -> str:
        """Subscribe to real-time ticker updates. Returns subscription ID."""
        ...

    @abstractmethod
    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[UnifiedOrderbook], Any], depth: int = 20
    ) -> str:
        """Subscribe to orderbook updates. Returns subscription ID."""
        ...

    @abstractmethod
    async def subscribe_trades(
        self, symbol: str, callback: Callable[[UnifiedTrade], Any]
    ) -> str:
        """Subscribe to real-time trade stream. Returns subscription ID."""
        ...

    @abstractmethod
    async def subscribe_klines(
        self, symbol: str, interval: str, callback: Callable[[UnifiedKline], Any]
    ) -> str:
        """Subscribe to kline/candlestick stream. Returns subscription ID."""
        ...

    @abstractmethod
    async def subscribe_user_data(
        self,
        on_order: Optional[Callable[[UnifiedOrder], Any]] = None,
        on_position: Optional[Callable[[UnifiedPosition], Any]] = None,
        on_balance: Optional[Callable[[UnifiedBalance], Any]] = None,
    ) -> str:
        """Subscribe to user data stream (orders, positions, balances).
        Returns subscription ID.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a stream. Returns True if successful."""
        ...

    # -- Health and status --------------------------------------------------

    def get_health(self) -> ConnectionHealthStatus:
        """Get current health status of this exchange connection."""
        return self._health.get_status(self._rate_limiter)

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # -- Utility methods available to all subclasses ------------------------

    def _sign_hmac_sha256(self, message: str) -> str:
        """Sign a message with the configured API secret using HMAC-SHA256."""
        return hmac_sha256_sign(self.config.api_secret, message)

    def _build_query(self, params: Dict[str, Any]) -> str:
        """Build query string from params dict, filtering out None values."""
        return build_query_string(params)

    @staticmethod
    def _ts() -> int:
        """Current UTC timestamp in milliseconds."""
        return current_timestamp_ms()

    async def _throttle(self, weight: int = 1) -> None:
        """Wait for rate limiter approval before making a request."""
        await self._rate_limiter.acquire(weight)

    def _register_callback(self, stream_key: str, callback: Callable) -> None:
        """Register a callback for a WebSocket stream."""
        if stream_key not in self._ws_callbacks:
            self._ws_callbacks[stream_key] = []
        self._ws_callbacks[stream_key].append(callback)

    def _dispatch_callback(self, stream_key: str, data: Any) -> None:
        """Dispatch data to all registered callbacks for a stream."""
        callbacks = self._ws_callbacks.get(stream_key, [])
        for cb in callbacks:
            try:
                result = cb(data)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as exc:
                self._logger.error(
                    "Callback error for stream %s: %s", stream_key, exc
                )

    def _remove_callbacks(self, stream_key: str) -> None:
        """Remove all callbacks for a stream."""
        self._ws_callbacks.pop(stream_key, None)

    async def _get_exchange_info_cached(self) -> Dict[str, Any]:
        """Return cached exchange info, refreshing if stale."""
        now = time.monotonic()
        if not self._exchange_info_cache or (now - self._exchange_info_ts) > self._exchange_info_ttl:
            self._exchange_info_cache = await self._fetch_exchange_info()
            self._exchange_info_ts = now
        return self._exchange_info_cache

    @abstractmethod
    async def _fetch_exchange_info(self) -> Dict[str, Any]:
        """Fetch raw exchange info from the API. Subclasses implement this."""
        ...

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.__class__.__name__} [{self.exchange_name}] ({status})>"
