"""
Mefai Autotrade - Smart Order Router
Production-grade order routing engine with multi-venue support,
latency tracking, fill rate analysis, and failover routing.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VenueStatus(Enum):
    """Operational status of a trading venue."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"


class RouteType(Enum):
    """How an order should be routed."""
    SINGLE = "single"          # Send entire order to one venue
    SPLIT = "split"            # Split across multiple venues
    SEQUENTIAL = "sequential"  # Try venues in order until filled
    FAILOVER = "failover"      # Use backup if primary fails


class OrderTypeSupport(Enum):
    """Whether a venue supports a given order type natively."""
    NATIVE = "native"
    SIMULATED = "simulated"
    UNSUPPORTED = "unsupported"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VenueMetrics:
    """Performance metrics for a single venue."""
    venue_id: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    fill_rate: float = 0.0         # 0.0 - 1.0
    partial_fill_rate: float = 0.0
    rejection_rate: float = 0.0
    avg_slippage_bps: float = 0.0  # basis points
    uptime_pct: float = 100.0
    total_orders: int = 0
    total_fills: int = 0
    total_rejections: int = 0
    total_timeouts: int = 0
    last_order_time: float = 0.0
    last_latency_ms: float = 0.0
    error_count_last_hour: int = 0

    @property
    def health_score(self) -> float:
        """Composite health score 0-100 combining all metrics."""
        if self.total_orders == 0:
            return 50.0  # Unknown - neutral score
        fill_score = self.fill_rate * 40.0
        latency_score = max(0, 30.0 - (self.avg_latency_ms / 100.0) * 30.0)
        slippage_score = max(0, 20.0 - abs(self.avg_slippage_bps) * 2.0)
        uptime_score = (self.uptime_pct / 100.0) * 10.0
        return min(100.0, fill_score + latency_score + slippage_score + uptime_score)


@dataclass
class Venue:
    """Configuration and state of a trading venue."""
    venue_id: str
    name: str
    exchange_type: str  # "binance_spot", "binance_futures", "bybit", etc.
    api_endpoint: str
    status: VenueStatus = VenueStatus.ACTIVE
    priority: int = 0  # Lower = higher priority
    max_order_size: float = 0.0  # 0 = no limit
    min_order_size: float = 0.0
    supported_order_types: List[str] = field(default_factory=list)
    supported_symbols: List[str] = field(default_factory=list)
    fee_maker_bps: float = 1.0
    fee_taker_bps: float = 1.0
    rate_limit_per_second: int = 10
    rate_limit_per_minute: int = 1200
    weight_allocation: float = 1.0  # For split routing
    is_primary: bool = False
    is_backup: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    _metrics: Optional[VenueMetrics] = field(default=None, repr=False)
    _latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _order_timestamps: deque = field(default_factory=lambda: deque(maxlen=100), repr=False)
    _error_timestamps: deque = field(default_factory=lambda: deque(maxlen=100), repr=False)

    def __post_init__(self):
        if self._metrics is None:
            self._metrics = VenueMetrics(venue_id=self.venue_id)

    @property
    def metrics(self) -> VenueMetrics:
        return self._metrics

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample for this venue."""
        self._latency_samples.append(latency_ms)
        self._metrics.last_latency_ms = latency_ms
        samples = list(self._latency_samples)
        if samples:
            samples_sorted = sorted(samples)
            self._metrics.avg_latency_ms = sum(samples) / len(samples)
            idx_95 = int(len(samples_sorted) * 0.95)
            idx_99 = int(len(samples_sorted) * 0.99)
            self._metrics.p95_latency_ms = samples_sorted[min(idx_95, len(samples_sorted) - 1)]
            self._metrics.p99_latency_ms = samples_sorted[min(idx_99, len(samples_sorted) - 1)]

    def record_fill(self, slippage_bps: float = 0.0, partial: bool = False) -> None:
        """Record a successful fill."""
        self._metrics.total_orders += 1
        self._metrics.total_fills += 1
        self._metrics.last_order_time = time.time()
        if partial:
            self._metrics.partial_fill_rate = (
                (self._metrics.partial_fill_rate * (self._metrics.total_fills - 1) + 1.0)
                / self._metrics.total_fills
            )
        # Rolling average slippage
        n = self._metrics.total_fills
        self._metrics.avg_slippage_bps = (
            (self._metrics.avg_slippage_bps * (n - 1) + slippage_bps) / n
        )
        self._update_fill_rate()

    def record_rejection(self) -> None:
        """Record an order rejection."""
        self._metrics.total_orders += 1
        self._metrics.total_rejections += 1
        self._metrics.last_order_time = time.time()
        self._error_timestamps.append(time.time())
        self._update_fill_rate()
        self._update_error_count()

    def record_timeout(self) -> None:
        """Record an order timeout."""
        self._metrics.total_orders += 1
        self._metrics.total_timeouts += 1
        self._metrics.last_order_time = time.time()
        self._error_timestamps.append(time.time())
        self._update_fill_rate()
        self._update_error_count()

    def _update_fill_rate(self) -> None:
        if self._metrics.total_orders > 0:
            self._metrics.fill_rate = self._metrics.total_fills / self._metrics.total_orders
            self._metrics.rejection_rate = self._metrics.total_rejections / self._metrics.total_orders

    def _update_error_count(self) -> None:
        cutoff = time.time() - 3600.0
        while self._error_timestamps and self._error_timestamps[0] < cutoff:
            self._error_timestamps.popleft()
        self._metrics.error_count_last_hour = len(self._error_timestamps)

    def check_rate_limit(self) -> bool:
        """Return True if we are within rate limits."""
        now = time.time()
        # Clean old timestamps
        while self._order_timestamps and self._order_timestamps[0] < now - 60:
            self._order_timestamps.popleft()
        if len(self._order_timestamps) >= self.rate_limit_per_minute:
            return False
        recent = sum(1 for t in self._order_timestamps if t > now - 1.0)
        if recent >= self.rate_limit_per_second:
            return False
        return True

    def record_order_sent(self) -> None:
        """Record that an order was sent for rate limiting."""
        self._order_timestamps.append(time.time())

    def supports_order_type(self, order_type: str) -> OrderTypeSupport:
        """Check if venue supports the given order type."""
        if order_type in self.supported_order_types:
            return OrderTypeSupport.NATIVE
        # Known decomposition mappings
        decomposable = {
            "OCO": ["LIMIT", "STOP_MARKET"],
            "BRACKET": ["LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"],
            "TRAILING_STOP": ["STOP_MARKET"],
        }
        if order_type in decomposable:
            required = decomposable[order_type]
            if all(r in self.supported_order_types for r in required):
                return OrderTypeSupport.SIMULATED
        return OrderTypeSupport.UNSUPPORTED

    def supports_symbol(self, symbol: str) -> bool:
        """Check if venue trades the given symbol."""
        if not self.supported_symbols:
            return True  # Empty list = all symbols supported
        return symbol in self.supported_symbols

    def is_available(self) -> bool:
        """Check if venue is available for order routing."""
        return self.status in (VenueStatus.ACTIVE, VenueStatus.DEGRADED)


@dataclass
class RoutingRule:
    """Configurable rule for order routing decisions."""
    rule_id: str
    name: str
    priority: int = 0  # Lower = evaluated first
    enabled: bool = True

    # Match conditions (all must be True for rule to apply)
    symbol_pattern: Optional[str] = None       # Glob pattern, e.g. "BTC*"
    min_order_value_usd: Optional[float] = None
    max_order_value_usd: Optional[float] = None
    order_types: Optional[List[str]] = None
    time_start_utc: Optional[str] = None       # "HH:MM"
    time_end_utc: Optional[str] = None

    # Routing action
    route_type: RouteType = RouteType.SINGLE
    preferred_venues: List[str] = field(default_factory=list)
    split_weights: Dict[str, float] = field(default_factory=dict)
    max_venues: int = 3
    min_health_score: float = 30.0

    def matches(self, symbol: str, order_value_usd: float,
                order_type: str, current_time_utc: Optional[str] = None) -> bool:
        """Check if this rule matches the given order parameters."""
        if not self.enabled:
            return False
        if self.symbol_pattern and not self._match_pattern(symbol, self.symbol_pattern):
            return False
        if self.min_order_value_usd is not None and order_value_usd < self.min_order_value_usd:
            return False
        if self.max_order_value_usd is not None and order_value_usd > self.max_order_value_usd:
            return False
        if self.order_types and order_type not in self.order_types:
            return False
        if current_time_utc and self.time_start_utc and self.time_end_utc:
            if not self._in_time_window(current_time_utc, self.time_start_utc, self.time_end_utc):
                return False
        return True

    @staticmethod
    def _match_pattern(text: str, pattern: str) -> bool:
        """Simple glob match supporting * wildcard."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return text.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return text.endswith(pattern[1:])
        return text == pattern

    @staticmethod
    def _in_time_window(current: str, start: str, end: str) -> bool:
        """Check if current HH:MM is within start-end window."""
        try:
            c_parts = current.split(":")
            s_parts = start.split(":")
            e_parts = end.split(":")
            c_min = int(c_parts[0]) * 60 + int(c_parts[1])
            s_min = int(s_parts[0]) * 60 + int(s_parts[1])
            e_min = int(e_parts[0]) * 60 + int(e_parts[1])
            if s_min <= e_min:
                return s_min <= c_min <= e_min
            else:
                # Wraps midnight
                return c_min >= s_min or c_min <= e_min
        except (ValueError, IndexError):
            return True


@dataclass
class RoutingDecision:
    """Result of the routing engine - where and how to route an order."""
    decision_id: str
    route_type: RouteType
    venues: List[Venue]
    allocations: Dict[str, float]  # venue_id -> fraction of order (0.0 - 1.0)
    rule_applied: Optional[str] = None
    reason: str = ""
    estimated_cost_bps: float = 0.0
    estimated_latency_ms: float = 0.0
    fallback_venues: List[Venue] = field(default_factory=list)
    order_type_conversions: Dict[str, List[str]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def primary_venue(self) -> Optional[Venue]:
        """Get the venue with the largest allocation."""
        if not self.venues:
            return None
        max_venue_id = max(self.allocations, key=self.allocations.get)
        for v in self.venues:
            if v.venue_id == max_venue_id:
                return v
        return self.venues[0]

    def get_venue_quantity(self, venue_id: str, total_quantity: float) -> float:
        """Get the quantity to send to a specific venue."""
        alloc = self.allocations.get(venue_id, 0.0)
        return total_quantity * alloc


# ---------------------------------------------------------------------------
# Price improvement tracker
# ---------------------------------------------------------------------------

class PriceImprovementTracker:
    """Track price improvements across venues for the same symbol."""

    def __init__(self, window_size: int = 500):
        self._window_size = window_size
        # symbol -> venue_id -> deque of (expected_price, actual_price)
        self._records: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )

    def record(self, symbol: str, venue_id: str,
               expected_price: float, actual_price: float, side: str) -> None:
        """Record an execution for price improvement analysis."""
        self._records[symbol][venue_id].append((expected_price, actual_price, side))

    def get_improvement_bps(self, symbol: str, venue_id: str) -> float:
        """Get average price improvement in basis points for a venue.
        Positive = improvement (got better price than expected).
        Negative = slippage.
        """
        records = self._records.get(symbol, {}).get(venue_id)
        if not records or len(records) == 0:
            return 0.0
        total_improvement = 0.0
        for expected, actual, side in records:
            if expected == 0:
                continue
            if side == "BUY":
                # For buys, improvement = buying cheaper than expected
                improvement = (expected - actual) / expected * 10000.0
            else:
                # For sells, improvement = selling higher than expected
                improvement = (actual - expected) / expected * 10000.0
            total_improvement += improvement
        return total_improvement / len(records)

    def get_best_venue(self, symbol: str) -> Optional[str]:
        """Get the venue_id with the best price improvement for a symbol."""
        venues = self._records.get(symbol, {})
        if not venues:
            return None
        best_venue = None
        best_improvement = float("-inf")
        for venue_id in venues:
            imp = self.get_improvement_bps(symbol, venue_id)
            if imp > best_improvement:
                best_improvement = imp
                best_venue = venue_id
        return best_venue

    def get_venue_rankings(self, symbol: str) -> List[Tuple[str, float]]:
        """Get all venues ranked by price improvement for a symbol."""
        venues = self._records.get(symbol, {})
        rankings = []
        for venue_id in venues:
            imp = self.get_improvement_bps(symbol, venue_id)
            rankings.append((venue_id, imp))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Track round-trip latencies per venue with statistical analysis."""

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._last_update: Dict[str, float] = {}

    def record(self, venue_id: str, latency_ms: float) -> None:
        """Record a latency sample."""
        self._samples[venue_id].append(latency_ms)
        self._last_update[venue_id] = time.time()

    def get_avg(self, venue_id: str) -> float:
        """Get average latency for a venue."""
        samples = self._samples.get(venue_id)
        if not samples:
            return float("inf")
        return sum(samples) / len(samples)

    def get_percentile(self, venue_id: str, pct: float) -> float:
        """Get the p-th percentile latency."""
        samples = self._samples.get(venue_id)
        if not samples:
            return float("inf")
        sorted_s = sorted(samples)
        idx = int(len(sorted_s) * pct / 100.0)
        idx = min(idx, len(sorted_s) - 1)
        return sorted_s[idx]

    def get_jitter(self, venue_id: str) -> float:
        """Get latency jitter (standard deviation)."""
        samples = self._samples.get(venue_id)
        if not samples or len(samples) < 2:
            return 0.0
        avg = sum(samples) / len(samples)
        variance = sum((s - avg) ** 2 for s in samples) / (len(samples) - 1)
        return variance ** 0.5

    def get_trend(self, venue_id: str, recent_n: int = 50) -> float:
        """Get latency trend. Positive = increasing latency (degrading).
        Returns slope in ms per sample.
        """
        samples = self._samples.get(venue_id)
        if not samples or len(samples) < 10:
            return 0.0
        recent = list(samples)[-recent_n:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def is_degraded(self, venue_id: str, threshold_ms: float = 500.0) -> bool:
        """Check if a venue's latency is degraded."""
        avg = self.get_avg(venue_id)
        return avg > threshold_ms

    def get_fastest_venue(self, venue_ids: List[str]) -> Optional[str]:
        """Get the fastest venue from a list."""
        best = None
        best_latency = float("inf")
        for vid in venue_ids:
            avg = self.get_avg(vid)
            if avg < best_latency:
                best_latency = avg
                best = vid
        return best

    def get_rankings(self, venue_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Get venues ranked by average latency (fastest first)."""
        if venue_ids is None:
            venue_ids = list(self._samples.keys())
        rankings = [(vid, self.get_avg(vid)) for vid in venue_ids]
        rankings.sort(key=lambda x: x[1])
        return rankings


# ---------------------------------------------------------------------------
# Fill rate analyzer
# ---------------------------------------------------------------------------

class FillRateAnalyzer:
    """Analyze fill rates per venue, symbol, and order type."""

    def __init__(self, window_size: int = 500):
        self._window_size = window_size
        # (venue_id, symbol, order_type) -> deque of bool (True=filled)
        self._records: Dict[Tuple[str, str, str], deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # Aggregate per venue
        self._venue_records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def record(self, venue_id: str, symbol: str, order_type: str, filled: bool) -> None:
        """Record a fill or miss."""
        key = (venue_id, symbol, order_type)
        self._records[key].append(filled)
        self._venue_records[venue_id].append(filled)

    def get_fill_rate(self, venue_id: str, symbol: Optional[str] = None,
                      order_type: Optional[str] = None) -> float:
        """Get fill rate for given filters. Returns 0.0-1.0."""
        if symbol and order_type:
            records = self._records.get((venue_id, symbol, order_type))
        elif symbol:
            records = deque()
            for (vid, sym, ot), r in self._records.items():
                if vid == venue_id and sym == symbol:
                    records.extend(r)
        elif order_type:
            records = deque()
            for (vid, sym, ot), r in self._records.items():
                if vid == venue_id and ot == order_type:
                    records.extend(r)
        else:
            records = self._venue_records.get(venue_id)
        if not records:
            return 0.5  # Unknown - assume 50%
        return sum(1 for r in records if r) / len(records)

    def get_best_venue_for_symbol(self, symbol: str, venues: List[str]) -> Optional[str]:
        """Get the venue with the highest fill rate for a specific symbol."""
        best = None
        best_rate = -1.0
        for vid in venues:
            rate = self.get_fill_rate(vid, symbol=symbol)
            if rate > best_rate:
                best_rate = rate
                best = vid
        return best


# ---------------------------------------------------------------------------
# Order type converter
# ---------------------------------------------------------------------------

class OrderTypeConverter:
    """Convert unsupported order types into supported equivalents."""

    def __init__(self):
        # Decomposition rules: complex_type -> list of simpler types
        self._decomposition_rules: Dict[str, List[Dict[str, Any]]] = {
            "OCO": [
                {"type": "LIMIT", "role": "take_profit"},
                {"type": "STOP_MARKET", "role": "stop_loss"},
            ],
            "BRACKET": [
                {"type": "LIMIT", "role": "entry"},
                {"type": "STOP_MARKET", "role": "stop_loss"},
                {"type": "TAKE_PROFIT_MARKET", "role": "take_profit"},
            ],
            "TRAILING_STOP_MARKET": [
                {"type": "STOP_MARKET", "role": "trailing", "simulated": True},
            ],
            "STOP_LIMIT": [
                {"type": "STOP_MARKET", "role": "trigger"},
                {"type": "LIMIT", "role": "limit_after_trigger"},
            ],
        }

    def can_convert(self, order_type: str, supported_types: List[str]) -> bool:
        """Check if we can decompose order_type into supported types."""
        if order_type in supported_types:
            return True
        rules = self._decomposition_rules.get(order_type)
        if not rules:
            return False
        return all(r["type"] in supported_types for r in rules)

    def decompose(self, order_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a complex order type into simpler orders.

        Args:
            order_type: The complex order type to decompose
            params: Order parameters (symbol, side, quantity, price, stop_price, etc.)

        Returns:
            List of order parameter dicts ready for submission
        """
        rules = self._decomposition_rules.get(order_type)
        if not rules:
            # Return as-is
            return [{"type": order_type, **params}]

        orders = []
        for rule in rules:
            order = {
                "type": rule["type"],
                "symbol": params.get("symbol"),
                "side": params.get("side"),
                "quantity": params.get("quantity"),
                "role": rule["role"],
            }
            if rule["role"] == "entry":
                order["price"] = params.get("price")
            elif rule["role"] == "take_profit":
                order["price"] = params.get("take_profit_price")
                # Flip side for exit
                order["side"] = "SELL" if params.get("side") == "BUY" else "BUY"
            elif rule["role"] == "stop_loss":
                order["stop_price"] = params.get("stop_price")
                order["side"] = "SELL" if params.get("side") == "BUY" else "BUY"
            elif rule["role"] == "limit_after_trigger":
                order["price"] = params.get("price")
                order["stop_price"] = params.get("stop_price")
            elif rule["role"] == "trailing":
                order["stop_price"] = params.get("stop_price")
                order["callback_rate"] = params.get("callback_rate", 1.0)
            orders.append(order)

        return orders

    def add_rule(self, order_type: str, decomposition: List[Dict[str, Any]]) -> None:
        """Add a custom decomposition rule."""
        self._decomposition_rules[order_type] = decomposition


# ---------------------------------------------------------------------------
# Smart Router
# ---------------------------------------------------------------------------

class SmartRouter:
    """Smart order routing engine.

    Routes orders to the best venue or splits across multiple venues
    based on configurable rules, real-time metrics, and market conditions.
    """

    def __init__(self):
        self._venues: Dict[str, Venue] = {}
        self._rules: List[RoutingRule] = []
        self._latency_tracker = LatencyTracker()
        self._fill_rate_analyzer = FillRateAnalyzer()
        self._price_improvement = PriceImprovementTracker()
        self._order_type_converter = OrderTypeConverter()
        self._default_route_type = RouteType.SINGLE
        self._max_split_venues = 3
        self._min_health_for_routing = 20.0
        self._failover_cooldown_sec = 60.0
        self._last_failover: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        logger.info("SmartRouter initialized")

    # -----------------------------------------------------------------------
    # Venue management
    # -----------------------------------------------------------------------

    def add_venue(self, venue: Venue) -> None:
        """Register a venue for order routing."""
        self._venues[venue.venue_id] = venue
        logger.info("Venue added: %s (%s)", venue.venue_id, venue.name)

    def remove_venue(self, venue_id: str) -> None:
        """Remove a venue from routing."""
        if venue_id in self._venues:
            del self._venues[venue_id]
            logger.info("Venue removed: %s", venue_id)

    def get_venue(self, venue_id: str) -> Optional[Venue]:
        """Get venue by ID."""
        return self._venues.get(venue_id)

    def get_active_venues(self) -> List[Venue]:
        """Get all active venues."""
        return [v for v in self._venues.values() if v.is_available()]

    def update_venue_status(self, venue_id: str, status: VenueStatus) -> None:
        """Update a venue's operational status."""
        venue = self._venues.get(venue_id)
        if venue:
            old_status = venue.status
            venue.status = status
            logger.info(
                "Venue %s status: %s -> %s", venue_id, old_status.value, status.value
            )

    # -----------------------------------------------------------------------
    # Rule management
    # -----------------------------------------------------------------------

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule. Rules are evaluated in priority order."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
        logger.info("Routing rule added: %s (priority=%d)", rule.name, rule.priority)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a routing rule."""
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def get_rules(self) -> List[RoutingRule]:
        """Get all routing rules in priority order."""
        return list(self._rules)

    # -----------------------------------------------------------------------
    # Core routing
    # -----------------------------------------------------------------------

    async def route(self, symbol: str, side: str, quantity: float,
                    order_type: str, price: Optional[float] = None,
                    order_value_usd: Optional[float] = None) -> RoutingDecision:
        """Determine the best routing for an order.

        Args:
            symbol: Trading symbol (e.g. BTCUSDT)
            side: BUY or SELL
            quantity: Order quantity
            order_type: LIMIT, MARKET, STOP_MARKET, etc.
            price: Limit price (if applicable)
            order_value_usd: Estimated order value in USD for rule matching

        Returns:
            RoutingDecision with venues, allocations, and any needed conversions
        """
        async with self._lock:
            return self._compute_route(
                symbol, side, quantity, order_type, price, order_value_usd
            )

    def _compute_route(self, symbol: str, side: str, quantity: float,
                       order_type: str, price: Optional[float],
                       order_value_usd: Optional[float]) -> RoutingDecision:
        """Internal routing computation."""
        decision_id = str(uuid.uuid4())[:12]
        est_value = order_value_usd or (quantity * price if price else 0.0)

        # Step 1: Find eligible venues
        eligible = self._get_eligible_venues(symbol, order_type, quantity)
        if not eligible:
            logger.warning(
                "No eligible venues for %s %s %s qty=%.6f",
                side, order_type, symbol, quantity
            )
            return RoutingDecision(
                decision_id=decision_id,
                route_type=RouteType.SINGLE,
                venues=[],
                allocations={},
                reason="No eligible venues found",
            )

        # Step 2: Check routing rules
        matched_rule = self._find_matching_rule(symbol, est_value, order_type)

        # Step 3: Determine route type and allocations
        if matched_rule:
            return self._apply_rule(
                decision_id, matched_rule, eligible, symbol, side,
                quantity, order_type, est_value
            )

        # Step 4: Default routing - score-based single venue selection
        return self._default_routing(
            decision_id, eligible, symbol, side, quantity, order_type, est_value
        )

    def _get_eligible_venues(self, symbol: str, order_type: str,
                             quantity: float) -> List[Venue]:
        """Get venues that can handle this order."""
        eligible = []
        for venue in self._venues.values():
            if not venue.is_available():
                continue
            if not venue.supports_symbol(symbol):
                continue
            if not venue.check_rate_limit():
                continue
            if venue.metrics.health_score < self._min_health_for_routing:
                continue
            # Check order type support (native or simulated)
            support = venue.supports_order_type(order_type)
            if support == OrderTypeSupport.UNSUPPORTED:
                if not self._order_type_converter.can_convert(
                    order_type, venue.supported_order_types
                ):
                    continue
            # Check quantity limits
            if venue.max_order_size > 0 and quantity > venue.max_order_size:
                # Could still be eligible for split routing
                pass
            if venue.min_order_size > 0 and quantity < venue.min_order_size:
                continue
            eligible.append(venue)
        return eligible

    def _find_matching_rule(self, symbol: str, order_value_usd: float,
                            order_type: str) -> Optional[RoutingRule]:
        """Find the first matching routing rule."""
        for rule in self._rules:
            if rule.matches(symbol, order_value_usd, order_type):
                return rule
        return None

    def _apply_rule(self, decision_id: str, rule: RoutingRule,
                    eligible: List[Venue], symbol: str, side: str,
                    quantity: float, order_type: str,
                    est_value: float) -> RoutingDecision:
        """Apply a matched routing rule."""
        # Filter to preferred venues if specified
        if rule.preferred_venues:
            preferred = [v for v in eligible if v.venue_id in rule.preferred_venues]
            if preferred:
                eligible = preferred

        if rule.route_type == RouteType.SPLIT:
            return self._build_split_decision(
                decision_id, rule, eligible, symbol, order_type,
                quantity, est_value
            )
        elif rule.route_type == RouteType.SEQUENTIAL:
            return self._build_sequential_decision(
                decision_id, rule, eligible, symbol, order_type, est_value
            )
        elif rule.route_type == RouteType.FAILOVER:
            return self._build_failover_decision(
                decision_id, rule, eligible, symbol, order_type, est_value
            )
        else:
            # SINGLE - pick best
            best = self._score_and_rank(eligible, symbol)
            if not best:
                return RoutingDecision(
                    decision_id=decision_id,
                    route_type=RouteType.SINGLE,
                    venues=[],
                    allocations={},
                    rule_applied=rule.rule_id,
                    reason="No venues passed scoring",
                )
            conversions = self._check_order_type_conversion(best[0], order_type)
            return RoutingDecision(
                decision_id=decision_id,
                route_type=RouteType.SINGLE,
                venues=[best[0]],
                allocations={best[0].venue_id: 1.0},
                rule_applied=rule.rule_id,
                reason=f"Rule '{rule.name}' - single venue, score={best[0].metrics.health_score:.1f}",
                estimated_cost_bps=best[0].fee_taker_bps,
                estimated_latency_ms=best[0].metrics.avg_latency_ms,
                fallback_venues=best[1:3] if len(best) > 1 else [],
                order_type_conversions=conversions,
            )

    def _build_split_decision(self, decision_id: str, rule: RoutingRule,
                              eligible: List[Venue], symbol: str,
                              order_type: str, quantity: float,
                              est_value: float) -> RoutingDecision:
        """Build a split-routing decision across multiple venues."""
        ranked = self._score_and_rank(eligible, symbol)
        selected = ranked[: rule.max_venues]
        if not selected:
            return RoutingDecision(
                decision_id=decision_id,
                route_type=RouteType.SPLIT,
                venues=[],
                allocations={},
                rule_applied=rule.rule_id,
                reason="No venues available for split",
            )

        # Compute allocations
        allocations = {}
        if rule.split_weights:
            # Use explicit weights from rule
            total_weight = 0.0
            for v in selected:
                w = rule.split_weights.get(v.venue_id, v.weight_allocation)
                allocations[v.venue_id] = w
                total_weight += w
            if total_weight > 0:
                for vid in allocations:
                    allocations[vid] /= total_weight
        else:
            # Weight by health score
            total_score = sum(v.metrics.health_score for v in selected)
            if total_score > 0:
                for v in selected:
                    allocations[v.venue_id] = v.metrics.health_score / total_score
            else:
                # Equal split
                n = len(selected)
                for v in selected:
                    allocations[v.venue_id] = 1.0 / n

        # Check minimum order sizes
        for v in selected:
            alloc_qty = quantity * allocations.get(v.venue_id, 0)
            if v.min_order_size > 0 and alloc_qty < v.min_order_size:
                # Redistribute to other venues
                removed_alloc = allocations.pop(v.venue_id, 0)
                remaining = [sv for sv in selected if sv.venue_id in allocations]
                if remaining:
                    per_venue = removed_alloc / len(remaining)
                    for rv in remaining:
                        allocations[rv.venue_id] = allocations.get(rv.venue_id, 0) + per_venue

        final_venues = [v for v in selected if v.venue_id in allocations]
        fallbacks = [v for v in ranked if v.venue_id not in allocations][:2]

        avg_cost = sum(
            v.fee_taker_bps * allocations.get(v.venue_id, 0) for v in final_venues
        )
        avg_latency = max(
            (v.metrics.avg_latency_ms for v in final_venues), default=0.0
        )

        conversions = {}
        for v in final_venues:
            c = self._check_order_type_conversion(v, order_type)
            conversions.update(c)

        return RoutingDecision(
            decision_id=decision_id,
            route_type=RouteType.SPLIT,
            venues=final_venues,
            allocations=allocations,
            rule_applied=rule.rule_id,
            reason=f"Split across {len(final_venues)} venues by rule '{rule.name}'",
            estimated_cost_bps=avg_cost,
            estimated_latency_ms=avg_latency,
            fallback_venues=fallbacks,
            order_type_conversions=conversions,
        )

    def _build_sequential_decision(self, decision_id: str, rule: RoutingRule,
                                   eligible: List[Venue], symbol: str,
                                   order_type: str,
                                   est_value: float) -> RoutingDecision:
        """Build a sequential routing decision - try venues in order."""
        ranked = self._score_and_rank(eligible, symbol)
        if not ranked:
            return RoutingDecision(
                decision_id=decision_id,
                route_type=RouteType.SEQUENTIAL,
                venues=[],
                allocations={},
                rule_applied=rule.rule_id,
                reason="No venues for sequential routing",
            )
        # Primary gets 100%, but fallback order is preserved
        allocations = {ranked[0].venue_id: 1.0}
        return RoutingDecision(
            decision_id=decision_id,
            route_type=RouteType.SEQUENTIAL,
            venues=ranked[:1],
            allocations=allocations,
            rule_applied=rule.rule_id,
            reason=f"Sequential routing via rule '{rule.name}', {len(ranked)} venues available",
            estimated_cost_bps=ranked[0].fee_taker_bps,
            estimated_latency_ms=ranked[0].metrics.avg_latency_ms,
            fallback_venues=ranked[1: rule.max_venues],
            order_type_conversions=self._check_order_type_conversion(ranked[0], order_type),
        )

    def _build_failover_decision(self, decision_id: str, rule: RoutingRule,
                                 eligible: List[Venue], symbol: str,
                                 order_type: str,
                                 est_value: float) -> RoutingDecision:
        """Build a failover routing decision."""
        primary_venues = [v for v in eligible if v.is_primary]
        backup_venues = [v for v in eligible if v.is_backup]

        if primary_venues:
            primary = self._score_and_rank(primary_venues, symbol)
            target = primary[0] if primary else None
        else:
            ranked = self._score_and_rank(eligible, symbol)
            target = ranked[0] if ranked else None

        if not target:
            return RoutingDecision(
                decision_id=decision_id,
                route_type=RouteType.FAILOVER,
                venues=[],
                allocations={},
                rule_applied=rule.rule_id,
                reason="No primary or backup venues available",
            )

        # Check if primary is in cooldown from recent failover
        if target.venue_id in self._last_failover:
            elapsed = time.time() - self._last_failover[target.venue_id]
            if elapsed < self._failover_cooldown_sec and backup_venues:
                # Use backup instead
                backup = self._score_and_rank(backup_venues, symbol)
                if backup:
                    target = backup[0]

        fallbacks = backup_venues if target.is_primary else primary_venues
        fallbacks = [v for v in fallbacks if v.venue_id != target.venue_id]

        return RoutingDecision(
            decision_id=decision_id,
            route_type=RouteType.FAILOVER,
            venues=[target],
            allocations={target.venue_id: 1.0},
            rule_applied=rule.rule_id,
            reason=f"Failover routing via rule '{rule.name}', primary={target.venue_id}",
            estimated_cost_bps=target.fee_taker_bps,
            estimated_latency_ms=target.metrics.avg_latency_ms,
            fallback_venues=fallbacks[:2],
            order_type_conversions=self._check_order_type_conversion(target, order_type),
        )

    def _default_routing(self, decision_id: str, eligible: List[Venue],
                         symbol: str, side: str, quantity: float,
                         order_type: str, est_value: float) -> RoutingDecision:
        """Default routing when no rule matches - pick best venue by score."""
        ranked = self._score_and_rank(eligible, symbol)
        if not ranked:
            return RoutingDecision(
                decision_id=decision_id,
                route_type=RouteType.SINGLE,
                venues=[],
                allocations={},
                reason="No eligible venues after scoring",
            )

        best = ranked[0]
        conversions = self._check_order_type_conversion(best, order_type)

        return RoutingDecision(
            decision_id=decision_id,
            route_type=RouteType.SINGLE,
            venues=[best],
            allocations={best.venue_id: 1.0},
            reason=f"Default routing to {best.venue_id} (score={best.metrics.health_score:.1f})",
            estimated_cost_bps=best.fee_taker_bps,
            estimated_latency_ms=best.metrics.avg_latency_ms,
            fallback_venues=ranked[1:3],
            order_type_conversions=conversions,
        )

    def _score_and_rank(self, venues: List[Venue], symbol: str) -> List[Venue]:
        """Score and rank venues for a symbol. Best first."""
        scored = []
        for v in venues:
            score = v.metrics.health_score

            # Bonus for price improvement history
            improvement = self._price_improvement.get_improvement_bps(symbol, v.venue_id)
            score += improvement * 0.5  # Each bps improvement adds 0.5 to score

            # Bonus for lower latency
            avg_latency = self._latency_tracker.get_avg(v.venue_id)
            if avg_latency < float("inf"):
                latency_bonus = max(0, 10.0 - (avg_latency / 50.0))
                score += latency_bonus

            # Bonus for higher fill rate on this symbol
            fill_rate = self._fill_rate_analyzer.get_fill_rate(v.venue_id, symbol=symbol)
            score += fill_rate * 10.0

            # Penalty for high fees
            score -= v.fee_taker_bps * 0.5

            # Priority bonus
            score += max(0, 10 - v.priority)

            scored.append((v, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in scored]

    def _check_order_type_conversion(self, venue: Venue,
                                     order_type: str) -> Dict[str, List[str]]:
        """Check if order type needs conversion for a venue."""
        support = venue.supports_order_type(order_type)
        if support == OrderTypeSupport.NATIVE:
            return {}
        if support == OrderTypeSupport.SIMULATED:
            decomposed = self._order_type_converter.decompose(order_type, {})
            types = [d.get("type", order_type) for d in decomposed]
            return {venue.venue_id: types}
        return {}

    # -----------------------------------------------------------------------
    # Feedback - call these after order execution
    # -----------------------------------------------------------------------

    def record_execution(self, venue_id: str, symbol: str, side: str,
                         order_type: str, latency_ms: float,
                         filled: bool, slippage_bps: float = 0.0,
                         expected_price: float = 0.0,
                         actual_price: float = 0.0,
                         partial: bool = False) -> None:
        """Record execution results to update venue metrics."""
        venue = self._venues.get(venue_id)
        if not venue:
            return

        venue.record_latency(latency_ms)
        self._latency_tracker.record(venue_id, latency_ms)
        self._fill_rate_analyzer.record(venue_id, symbol, order_type, filled)

        if filled:
            venue.record_fill(slippage_bps=slippage_bps, partial=partial)
            if expected_price > 0 and actual_price > 0:
                self._price_improvement.record(
                    symbol, venue_id, expected_price, actual_price, side
                )
        else:
            venue.record_rejection()

    def record_timeout(self, venue_id: str) -> None:
        """Record a venue timeout."""
        venue = self._venues.get(venue_id)
        if venue:
            venue.record_timeout()

    def trigger_failover(self, venue_id: str) -> None:
        """Mark a venue as having failed over."""
        self._last_failover[venue_id] = time.time()
        venue = self._venues.get(venue_id)
        if venue:
            venue.status = VenueStatus.DEGRADED
            logger.warning("Failover triggered for venue %s", venue_id)

    # -----------------------------------------------------------------------
    # Analytics
    # -----------------------------------------------------------------------

    def get_venue_metrics(self, venue_id: str) -> Optional[VenueMetrics]:
        """Get current metrics for a venue."""
        venue = self._venues.get(venue_id)
        if venue:
            return venue.metrics
        return None

    def get_all_metrics(self) -> Dict[str, VenueMetrics]:
        """Get metrics for all venues."""
        return {vid: v.metrics for vid, v in self._venues.items()}

    def get_latency_rankings(self) -> List[Tuple[str, float]]:
        """Get venues ranked by latency."""
        return self._latency_tracker.get_rankings(list(self._venues.keys()))

    def get_price_improvement_rankings(self, symbol: str) -> List[Tuple[str, float]]:
        """Get venues ranked by price improvement for a symbol."""
        return self._price_improvement.get_venue_rankings(symbol)

    def get_fill_rate_rankings(self, symbol: Optional[str] = None) -> List[Tuple[str, float]]:
        """Get venues ranked by fill rate."""
        rankings = []
        for vid in self._venues:
            rate = self._fill_rate_analyzer.get_fill_rate(vid, symbol=symbol)
            rankings.append((vid, rate))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_routing_summary(self) -> Dict[str, Any]:
        """Get a summary of the routing engine state."""
        active = self.get_active_venues()
        return {
            "total_venues": len(self._venues),
            "active_venues": len(active),
            "routing_rules": len(self._rules),
            "default_route_type": self._default_route_type.value,
            "venues": {
                vid: {
                    "name": v.name,
                    "status": v.status.value,
                    "health_score": v.metrics.health_score,
                    "avg_latency_ms": v.metrics.avg_latency_ms,
                    "fill_rate": v.metrics.fill_rate,
                    "total_orders": v.metrics.total_orders,
                }
                for vid, v in self._venues.items()
            },
        }
