# Mefai Signal Engine - Market Simulator
# Simulated exchange with order book, price impact, fill probability,
# spread, latency, and exchange downtime simulation.

import logging
import random
import uuid
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class OrderBookSide(Enum):
    BID = "bid"
    ASK = "ask"


@dataclass
class OrderBookLevel:
    """A single price level in the order book."""
    price: float = 0.0
    quantity: float = 0.0
    order_count: int = 1


@dataclass
class SimulatedOrderBook:
    """
    Simplified L1 order book simulation.

    Generates synthetic bid/ask levels around the mid price using
    configurable spread and depth parameters.
    """
    symbol: str = ""
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    mid_price: float = 0.0
    spread: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0

    def update(self, bar: Dict[str, Any], spread_bps: float = 1.0,
               depth_levels: int = 10, base_liquidity: float = 100.0,
               liquidity_decay: float = 0.8) -> None:
        """
        Generate synthetic order book from bar data.

        spread_bps: spread in basis points
        depth_levels: number of price levels on each side
        base_liquidity: base quantity at best bid/ask
        liquidity_decay: liquidity multiplier per level away from best
        """
        close = bar["close"]
        volume = bar.get("volume", 0)

        # Scale liquidity by volume
        vol_factor = 1.0
        if volume > 0:
            vol_factor = max(0.1, min(10.0, volume / 1000.0))

        self.mid_price = close
        half_spread = close * spread_bps / 10000.0
        self.best_bid = close - half_spread
        self.best_ask = close + half_spread
        self.spread = half_spread * 2
        self.timestamp = bar.get("timestamp")

        # Generate bid levels
        self.bids = []
        for i in range(depth_levels):
            level_price = self.best_bid - (i * half_spread * 0.5)
            level_qty = base_liquidity * vol_factor * (liquidity_decay ** i)
            level_orders = max(1, int(5 * (liquidity_decay ** i)))
            self.bids.append(OrderBookLevel(
                price=max(0.0, level_price),
                quantity=level_qty,
                order_count=level_orders,
            ))

        # Generate ask levels
        self.asks = []
        for i in range(depth_levels):
            level_price = self.best_ask + (i * half_spread * 0.5)
            level_qty = base_liquidity * vol_factor * (liquidity_decay ** i)
            level_orders = max(1, int(5 * (liquidity_decay ** i)))
            self.asks.append(OrderBookLevel(
                price=level_price,
                quantity=level_qty,
                order_count=level_orders,
            ))

    def get_fill_price(self, side: str, quantity: float) -> Tuple[float, float]:
        """
        Calculate average fill price for a given quantity.
        Returns (avg_fill_price, total_cost).
        Walks through the order book levels to simulate market impact.
        """
        if side == "BUY":
            levels = self.asks
        else:
            levels = self.bids

        if not levels:
            return (self.mid_price, self.mid_price * quantity)

        remaining = quantity
        total_cost = 0.0
        total_filled = 0.0

        for level in levels:
            if remaining <= 0:
                break
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            total_filled += fill_qty
            remaining -= fill_qty

        # If we exhausted all levels, fill remaining at last price
        if remaining > 0 and levels:
            last_price = levels[-1].price
            # Add extra slippage for exceeding book depth
            if side == "BUY":
                overshoot_price = last_price * 1.01
            else:
                overshoot_price = last_price * 0.99
            total_cost += remaining * overshoot_price
            total_filled += remaining

        avg_price = total_cost / total_filled if total_filled > 0 else self.mid_price
        return (avg_price, total_cost)

    def get_total_liquidity(self, side: str) -> float:
        """Get total available liquidity on one side of the book."""
        if side == "BUY":
            return sum(l.quantity for l in self.asks)
        else:
            return sum(l.quantity for l in self.bids)


@dataclass
class FillResult:
    """Result of an order fill attempt."""
    filled: bool = False
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    price_impact: float = 0.0
    spread_cost: float = 0.0
    latency_ms: float = 0.0
    rejection_reason: str = ""
    partial: bool = False


@dataclass
class ExchangeState:
    """Current state of the simulated exchange."""
    is_online: bool = True
    maintenance_until: Optional[datetime] = None
    circuit_breaker_active: bool = False
    circuit_breaker_until: Optional[datetime] = None
    rate_limit_remaining: int = 1200
    rate_limit_reset: Optional[datetime] = None


@dataclass
class LatencyConfig:
    """Configuration for latency simulation."""
    base_latency_ms: float = 10.0
    jitter_ms: float = 5.0
    spike_probability: float = 0.001
    spike_multiplier: float = 10.0
    market_order_latency_ms: float = 5.0
    limit_order_latency_ms: float = 3.0


@dataclass
class SpreadConfig:
    """Configuration for spread simulation."""
    base_spread_bps: float = 1.0
    volatility_multiplier: float = 2.0
    volume_impact: bool = True
    min_spread_bps: float = 0.5
    max_spread_bps: float = 50.0
    time_of_day_factor: bool = False


@dataclass
class PriceImpactConfig:
    """Configuration for price impact simulation."""
    enabled: bool = True
    linear_coefficient: float = 0.001
    square_root_model: bool = True
    temporary_impact_ratio: float = 0.5
    permanent_impact_ratio: float = 0.5
    decay_bars: int = 10
    max_impact_percent: float = 1.0


@dataclass
class FillProbabilityConfig:
    """Configuration for limit order fill probability."""
    enabled: bool = True
    base_fill_rate: float = 0.8
    depth_penalty: float = 0.1
    time_bonus: float = 0.02
    volatility_bonus: float = 0.5
    min_fill_rate: float = 0.1
    max_fill_rate: float = 1.0


@dataclass
class DowntimeEvent:
    """Represents a scheduled or random exchange downtime."""
    start: datetime = None
    end: datetime = None
    reason: str = "maintenance"
    affects_orders: bool = True
    affects_new_orders: bool = True


# ---------------------------------------------------------------------------
# Price impact model
# ---------------------------------------------------------------------------

class PriceImpactModel:
    """
    Models the price impact of large orders.

    Uses the square-root market impact model:
        impact = sigma * sqrt(Q / V)
    where sigma is volatility, Q is order size, V is average volume.
    """

    def __init__(self, config: PriceImpactConfig):
        self.config = config
        self._permanent_impact: Dict[str, float] = defaultdict(float)
        self._impact_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

    def calculate_impact(self, symbol: str, side: str, quantity: float,
                          price: float, bar: Dict[str, Any]) -> float:
        """
        Calculate price impact for an order.
        Returns the impact in price units (not percent).
        """
        if not self.config.enabled:
            return 0.0

        volume = bar.get("volume", 0)
        if volume <= 0:
            volume = 1.0

        # Track volume history
        self._volume_history[symbol].append(volume)
        avg_volume = sum(self._volume_history[symbol]) / len(self._volume_history[symbol])

        # Calculate bar volatility
        high = bar.get("high", price)
        low = bar.get("low", price)
        bar_volatility = (high - low) / price if price > 0 else 0.01

        participation = quantity / avg_volume if avg_volume > 0 else 1.0

        if self.config.square_root_model:
            impact_pct = bar_volatility * math.sqrt(abs(participation))
        else:
            impact_pct = self.config.linear_coefficient * participation

        impact_pct = impact_pct * self.config.linear_coefficient * 100
        impact_pct = min(impact_pct, self.config.max_impact_percent / 100.0)

        impact = price * impact_pct

        # Apply direction
        if side == "SELL":
            impact = -impact

        # Track permanent impact
        perm_impact = impact * self.config.permanent_impact_ratio
        self._permanent_impact[symbol] += perm_impact

        self._impact_history[symbol].append(impact)

        return impact

    def get_adjusted_price(self, symbol: str, base_price: float) -> float:
        """Get price adjusted for accumulated permanent impact."""
        return base_price + self._permanent_impact.get(symbol, 0.0)

    def decay_impact(self, symbol: str) -> None:
        """Decay accumulated permanent impact over time."""
        if symbol in self._permanent_impact:
            decay_factor = 1.0 / max(1, self.config.decay_bars)
            self._permanent_impact[symbol] *= (1.0 - decay_factor)

    def reset(self) -> None:
        self._permanent_impact.clear()
        self._impact_history.clear()
        self._volume_history.clear()


# ---------------------------------------------------------------------------
# Fill probability model
# ---------------------------------------------------------------------------

class FillProbabilityModel:
    """
    Models the probability that a limit order gets filled.

    Factors that affect fill probability:
    - Distance from current price (depth penalty)
    - Time the order has been open (time bonus)
    - Current volatility (volatility bonus)
    - Order size relative to volume
    """

    def __init__(self, config: FillProbabilityConfig):
        self.config = config

    def calculate_fill_probability(self, order_price: float,
                                    current_price: float,
                                    bars_open: int,
                                    bar: Dict[str, Any],
                                    order_quantity: float) -> float:
        """
        Calculate the probability that a limit order fills on this bar.
        Returns a value between 0 and 1.
        """
        if not self.config.enabled:
            return 1.0

        prob = self.config.base_fill_rate

        # Depth penalty - further from price = less likely to fill
        if current_price > 0:
            depth = abs(order_price - current_price) / current_price
            depth_penalty = depth * self.config.depth_penalty * 100
            prob -= depth_penalty

        # Time bonus - longer open = more likely to fill
        time_bonus = min(bars_open * self.config.time_bonus, 0.3)
        prob += time_bonus

        # Volatility bonus - higher volatility = more likely to fill
        if bar:
            high = bar.get("high", current_price)
            low = bar.get("low", current_price)
            if current_price > 0:
                bar_range = (high - low) / current_price
                vol_bonus = bar_range * self.config.volatility_bonus * 10
                prob += min(vol_bonus, 0.3)

        # Volume factor - large orders relative to volume are less likely
        volume = bar.get("volume", 0) if bar else 0
        if volume > 0 and order_quantity > 0:
            size_ratio = order_quantity / volume
            if size_ratio > 0.1:
                prob -= size_ratio * 0.5

        # Clamp
        prob = max(self.config.min_fill_rate, min(self.config.max_fill_rate, prob))
        return prob

    def should_fill(self, probability: float) -> bool:
        """Determine if the order should fill based on probability."""
        return random.random() < probability


# ---------------------------------------------------------------------------
# Spread simulator
# ---------------------------------------------------------------------------

class SpreadSimulator:
    """Simulates realistic bid-ask spreads."""

    def __init__(self, config: SpreadConfig):
        self.config = config
        self._volatility_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

    def calculate_spread(self, symbol: str, bar: Dict[str, Any]) -> float:
        """
        Calculate the bid-ask spread in price units for the current bar.
        """
        price = bar.get("close", 0)
        if price <= 0:
            return 0.0

        spread_bps = self.config.base_spread_bps

        # Volatility adjustment
        high = bar.get("high", price)
        low = bar.get("low", price)
        bar_range = (high - low) / price if price > 0 else 0.0
        self._volatility_history[symbol].append(bar_range)

        avg_vol = sum(self._volatility_history[symbol]) / \
                  len(self._volatility_history[symbol])
        spread_bps += avg_vol * self.config.volatility_multiplier * 10000

        # Volume adjustment
        if self.config.volume_impact:
            volume = bar.get("volume", 0)
            if volume > 0:
                # Lower volume = wider spread
                vol_factor = max(0.5, min(3.0, 1000.0 / max(volume, 1)))
                spread_bps *= vol_factor

        # Time of day factor (simplified)
        if self.config.time_of_day_factor:
            ts = bar.get("timestamp")
            if ts and hasattr(ts, "hour"):
                hour = ts.hour
                # Wider spreads during low-volume hours (UTC)
                if 0 <= hour <= 6:
                    spread_bps *= 1.3
                elif 12 <= hour <= 15:
                    spread_bps *= 0.8

        # Clamp
        spread_bps = max(self.config.min_spread_bps,
                         min(self.config.max_spread_bps, spread_bps))

        return price * spread_bps / 10000.0

    def get_bid_ask(self, symbol: str,
                     bar: Dict[str, Any]) -> Tuple[float, float]:
        """Get simulated bid and ask prices."""
        price = bar.get("close", 0)
        spread = self.calculate_spread(symbol, bar)
        half = spread / 2
        return (price - half, price + half)


# ---------------------------------------------------------------------------
# Latency simulator
# ---------------------------------------------------------------------------

class LatencySimulator:
    """Simulates network and exchange processing latency."""

    def __init__(self, config: LatencyConfig):
        self.config = config
        self._latency_history: deque = deque(maxlen=100)

    def get_latency(self, order_type: str = "MARKET") -> float:
        """
        Get simulated latency in milliseconds.
        """
        if order_type in ("MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"):
            base = self.config.market_order_latency_ms
        else:
            base = self.config.limit_order_latency_ms

        base += self.config.base_latency_ms

        # Add jitter
        jitter = random.uniform(-self.config.jitter_ms, self.config.jitter_ms)
        latency = base + jitter

        # Occasional spike
        if random.random() < self.config.spike_probability:
            latency *= self.config.spike_multiplier

        latency = max(0.1, latency)
        self._latency_history.append(latency)
        return latency

    def get_average_latency(self) -> float:
        if not self._latency_history:
            return self.config.base_latency_ms
        return sum(self._latency_history) / len(self._latency_history)

    def get_p99_latency(self) -> float:
        if not self._latency_history:
            return self.config.base_latency_ms * 5
        sorted_latencies = sorted(self._latency_history)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


# ---------------------------------------------------------------------------
# Exchange downtime simulator
# ---------------------------------------------------------------------------

class DowntimeSimulator:
    """Simulates exchange maintenance and downtime events."""

    def __init__(self):
        self.scheduled_downtimes: List[DowntimeEvent] = []
        self.random_downtime_probability: float = 0.0001
        self.random_downtime_duration_minutes: int = 5
        self._current_downtime: Optional[DowntimeEvent] = None

    def add_scheduled_downtime(self, start: datetime, end: datetime,
                                reason: str = "maintenance") -> None:
        self.scheduled_downtimes.append(DowntimeEvent(
            start=start, end=end, reason=reason,
        ))

    def set_random_downtime(self, probability: float = 0.0001,
                             duration_minutes: int = 5) -> None:
        self.random_downtime_probability = probability
        self.random_downtime_duration_minutes = duration_minutes

    def is_online(self, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Check if the exchange is online at the given timestamp."""
        # Check scheduled downtimes
        for dt in self.scheduled_downtimes:
            if dt.start <= timestamp <= dt.end:
                return (False, dt.reason)

        # Check current random downtime
        if self._current_downtime:
            if timestamp <= self._current_downtime.end:
                return (False, self._current_downtime.reason)
            else:
                self._current_downtime = None

        # Random downtime
        if random.random() < self.random_downtime_probability:
            end = timestamp + timedelta(
                minutes=self.random_downtime_duration_minutes
            )
            self._current_downtime = DowntimeEvent(
                start=timestamp, end=end,
                reason="unexpected_outage",
            )
            return (False, "unexpected_outage")

        return (True, None)


# ---------------------------------------------------------------------------
# Simulated exchange
# ---------------------------------------------------------------------------

class SimulatedExchange:
    """
    Complete simulated exchange environment.

    Combines order book, price impact, fill probability, spread,
    latency, and downtime simulation into a unified exchange model.

    Usage:
        exchange = SimulatedExchange(
            spread_config=SpreadConfig(base_spread_bps=2.0),
            latency_config=LatencyConfig(base_latency_ms=20.0),
        )

        # On each bar
        exchange.update(symbol, bar)
        result = exchange.attempt_fill(order, bar)

        # Check if exchange is available
        online, reason = exchange.is_online(timestamp)
    """

    def __init__(self,
                 spread_config: Optional[SpreadConfig] = None,
                 latency_config: Optional[LatencyConfig] = None,
                 impact_config: Optional[PriceImpactConfig] = None,
                 fill_config: Optional[FillProbabilityConfig] = None,
                 seed: Optional[int] = None):
        self.spread_sim = SpreadSimulator(spread_config or SpreadConfig())
        self.latency_sim = LatencySimulator(latency_config or LatencyConfig())
        self.impact_model = PriceImpactModel(impact_config or PriceImpactConfig())
        self.fill_model = FillProbabilityModel(fill_config or FillProbabilityConfig())
        self.downtime_sim = DowntimeSimulator()

        self.order_books: Dict[str, SimulatedOrderBook] = {}
        self.state = ExchangeState()
        self._bar_count: Dict[str, int] = defaultdict(int)
        self._order_creation_bar: Dict[str, int] = {}
        self._fill_stats = {
            "attempted": 0,
            "filled": 0,
            "rejected": 0,
            "partial": 0,
        }

        if seed is not None:
            random.seed(seed)

    def update(self, symbol: str, bar: Dict[str, Any]) -> None:
        """Update exchange state with a new bar."""
        # Update order book
        if symbol not in self.order_books:
            self.order_books[symbol] = SimulatedOrderBook(symbol=symbol)

        spread_bps = self.spread_sim.config.base_spread_bps
        spread = self.spread_sim.calculate_spread(symbol, bar)
        if bar.get("close", 0) > 0:
            spread_bps = (spread / bar["close"]) * 10000

        self.order_books[symbol].update(
            bar, spread_bps=spread_bps,
            depth_levels=10, base_liquidity=bar.get("volume", 100) / 50,
        )

        # Decay price impact
        self.impact_model.decay_impact(symbol)

        # Update bar counter
        self._bar_count[symbol] += 1

        # Check online status
        ts = bar.get("timestamp")
        if ts:
            online, reason = self.downtime_sim.is_online(ts)
            self.state.is_online = online

    def attempt_fill(self, order_type: str, side: str, symbol: str,
                     quantity: float, price: Optional[float],
                     stop_price: Optional[float],
                     bar: Dict[str, Any],
                     order_id: str = "",
                     created_bar: int = 0) -> FillResult:
        """
        Attempt to fill an order against the simulated exchange.

        Returns a FillResult with fill details or rejection reason.
        """
        self._fill_stats["attempted"] += 1
        result = FillResult()

        # Check if exchange is online
        if not self.state.is_online:
            result.rejection_reason = "exchange_offline"
            self._fill_stats["rejected"] += 1
            return result

        # Calculate latency
        latency = self.latency_sim.get_latency(order_type)
        result.latency_ms = latency

        # Get order book
        book = self.order_books.get(symbol)
        if book is None:
            result.rejection_reason = "no_order_book"
            self._fill_stats["rejected"] += 1
            return result

        current_bar_count = self._bar_count.get(symbol, 0)

        # Process by order type
        if order_type == "MARKET":
            result = self._fill_market_order(
                side, symbol, quantity, bar, book, latency
            )
        elif order_type == "LIMIT":
            bars_open = max(0, current_bar_count - created_bar)
            result = self._fill_limit_order(
                side, symbol, quantity, price, bar, book, latency, bars_open
            )
        elif order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            result = self._fill_stop_market_order(
                side, symbol, quantity, stop_price, bar, book, latency
            )
        elif order_type in ("STOP_LIMIT", "TAKE_PROFIT_LIMIT"):
            bars_open = max(0, current_bar_count - created_bar)
            result = self._fill_stop_limit_order(
                side, symbol, quantity, price, stop_price, bar, book,
                latency, bars_open
            )
        else:
            result.rejection_reason = f"unsupported_order_type: {order_type}"
            self._fill_stats["rejected"] += 1

        if result.filled:
            self._fill_stats["filled"] += 1
            if result.partial:
                self._fill_stats["partial"] += 1

        return result

    def _fill_market_order(self, side: str, symbol: str, quantity: float,
                            bar: Dict[str, Any], book: SimulatedOrderBook,
                            latency: float) -> FillResult:
        """Fill a market order using the order book."""
        result = FillResult(latency_ms=latency)

        # Check liquidity
        available = book.get_total_liquidity(side)
        if available <= 0:
            result.rejection_reason = "no_liquidity"
            self._fill_stats["rejected"] += 1
            return result

        fill_qty = min(quantity, available)
        avg_price, total_cost = book.get_fill_price(side, fill_qty)

        # Calculate price impact
        impact = self.impact_model.calculate_impact(
            symbol, side, fill_qty, avg_price, bar
        )
        avg_price += impact

        # Calculate spread cost
        mid = book.mid_price
        spread_cost = abs(avg_price - mid) * fill_qty

        result.filled = True
        result.fill_price = avg_price
        result.fill_quantity = fill_qty
        result.price_impact = impact
        result.spread_cost = spread_cost
        result.partial = fill_qty < quantity

        return result

    def _fill_limit_order(self, side: str, symbol: str, quantity: float,
                           price: float, bar: Dict[str, Any],
                           book: SimulatedOrderBook, latency: float,
                           bars_open: int) -> FillResult:
        """Fill a limit order with fill probability check."""
        result = FillResult(latency_ms=latency)

        # Check if price is reached
        if side == "BUY":
            if bar["low"] > price:
                return result  # price not reached
        else:
            if bar["high"] < price:
                return result  # price not reached

        # Calculate fill probability
        current_price = bar["close"]
        prob = self.fill_model.calculate_fill_probability(
            price, current_price, bars_open, bar, quantity
        )

        if not self.fill_model.should_fill(prob):
            return result  # did not fill this bar

        # Fill at the limit price (not worse)
        fill_price = price

        # Price impact for limit orders is lower
        impact = self.impact_model.calculate_impact(
            symbol, side, quantity, fill_price, bar
        ) * 0.3  # limit orders have less impact

        result.filled = True
        result.fill_price = fill_price
        result.fill_quantity = quantity
        result.price_impact = impact
        result.spread_cost = 0.0  # no spread cost for limit orders

        return result

    def _fill_stop_market_order(self, side: str, symbol: str, quantity: float,
                                 stop_price: float, bar: Dict[str, Any],
                                 book: SimulatedOrderBook,
                                 latency: float) -> FillResult:
        """Fill a stop market order."""
        result = FillResult(latency_ms=latency)

        # Check if stop price is triggered
        triggered = False
        if side == "BUY" and bar["high"] >= stop_price:
            triggered = True
        elif side == "SELL" and bar["low"] <= stop_price:
            triggered = True

        if not triggered:
            return result

        # Fill at stop price with slippage from order book
        base_price = stop_price
        avg_price, _ = book.get_fill_price(side, quantity)

        # Use the worse of stop price and book price
        if side == "BUY":
            fill_price = max(base_price, avg_price)
        else:
            fill_price = min(base_price, avg_price)

        impact = self.impact_model.calculate_impact(
            symbol, side, quantity, fill_price, bar
        )

        result.filled = True
        result.fill_price = fill_price + impact
        result.fill_quantity = quantity
        result.price_impact = impact

        return result

    def _fill_stop_limit_order(self, side: str, symbol: str, quantity: float,
                                price: float, stop_price: float,
                                bar: Dict[str, Any],
                                book: SimulatedOrderBook, latency: float,
                                bars_open: int) -> FillResult:
        """Fill a stop limit order."""
        result = FillResult(latency_ms=latency)

        # Check if stop is triggered
        triggered = False
        if side == "BUY" and bar["high"] >= stop_price:
            triggered = True
        elif side == "SELL" and bar["low"] <= stop_price:
            triggered = True

        if not triggered:
            return result

        # After trigger, behave like a limit order
        return self._fill_limit_order(
            side, symbol, quantity, price, bar, book, latency, bars_open
        )

    def validate_order(self, side: str, symbol: str, quantity: float,
                        price: Optional[float], balance: float,
                        leverage: float = 1.0,
                        existing_position_qty: float = 0.0,
                        min_order_size: float = 0.001,
                        max_position_size: float = float('inf')) -> Tuple[bool, str]:
        """
        Validate an order before submission.
        Returns (is_valid, rejection_reason).
        """
        # Check minimum order size
        if quantity < min_order_size:
            return (False, f"Below minimum order size: {quantity} < {min_order_size}")

        # Check balance
        if price and price > 0:
            required = (quantity * price) / leverage
            if required > balance * 1.001:  # small tolerance
                return (False, f"Insufficient balance: need {required:.2f}, have {balance:.2f}")

        # Check position size limit
        new_position = abs(existing_position_qty) + quantity
        if new_position > max_position_size:
            return (False, f"Exceeds max position size: {new_position} > {max_position_size}")

        # Check if exchange is online
        if not self.state.is_online:
            return (False, "Exchange is offline")

        return (True, "")

    def get_fill_stats(self) -> Dict[str, int]:
        """Get fill statistics."""
        return self._fill_stats.copy()

    def get_order_book(self, symbol: str) -> Optional[SimulatedOrderBook]:
        """Get the current order book for a symbol."""
        return self.order_books.get(symbol)

    def get_spread(self, symbol: str,
                    bar: Dict[str, Any]) -> Tuple[float, float]:
        """Get current bid-ask spread."""
        return self.spread_sim.get_bid_ask(symbol, bar)

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        return {
            "average_ms": self.latency_sim.get_average_latency(),
            "p99_ms": self.latency_sim.get_p99_latency(),
        }

    def reset(self) -> None:
        """Reset exchange state."""
        self.order_books.clear()
        self.state = ExchangeState()
        self._bar_count.clear()
        self._order_creation_bar.clear()
        self._fill_stats = {
            "attempted": 0,
            "filled": 0,
            "rejected": 0,
            "partial": 0,
        }
        self.impact_model.reset()


# ---------------------------------------------------------------------------
# Market replay simulator
# ---------------------------------------------------------------------------

class MarketReplaySimulator:
    """
    Replays historical data through the simulated exchange,
    generating synthetic tick data from bar data for higher fidelity
    simulation.
    """

    def __init__(self, exchange: Optional[SimulatedExchange] = None):
        self.exchange = exchange or SimulatedExchange()

    def generate_ticks_from_bar(self, bar: Dict[str, Any],
                                 ticks_per_bar: int = 10) -> List[Dict[str, Any]]:
        """
        Generate synthetic tick data from a single bar.
        Creates a realistic price path: open -> high/low -> close.
        """
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        ts = bar["timestamp"]
        volume = bar.get("volume", 0)
        tick_volume = volume / max(ticks_per_bar, 1)

        ticks = []
        prices = self._generate_price_path(o, h, l, c, ticks_per_bar)

        bar_duration = timedelta(minutes=1)  # default
        tick_interval = bar_duration / max(ticks_per_bar, 1)

        for i, price in enumerate(prices):
            tick_ts = ts + tick_interval * i if isinstance(ts, datetime) else ts
            ticks.append({
                "timestamp": tick_ts,
                "price": price,
                "volume": tick_volume + random.uniform(-tick_volume * 0.3,
                                                        tick_volume * 0.3),
                "side": "BUY" if random.random() > 0.5 else "SELL",
            })

        return ticks

    def _generate_price_path(self, o: float, h: float, l: float,
                              c: float, steps: int) -> List[float]:
        """
        Generate a realistic price path within a bar.
        Uses a constrained random walk that hits high and low.
        """
        if steps <= 1:
            return [o]
        if steps == 2:
            return [o, c]

        prices = [o]

        # Decide order: does price go to high first or low first?
        # Use a simple heuristic based on open-close direction
        if c >= o:
            # Bullish bar: likely goes low then high
            mid_point = steps // 3
            high_point = 2 * steps // 3
        else:
            # Bearish bar: likely goes high then low
            mid_point = steps // 3
            high_point = 2 * steps // 3

        for i in range(1, steps):
            if i <= mid_point:
                # Move toward low (or high for bearish)
                if c >= o:
                    target = l
                else:
                    target = h
                progress = i / mid_point
                base = o + (target - o) * progress
            elif i <= high_point:
                # Move toward high (or low for bearish)
                if c >= o:
                    target = h
                    prev_target = l
                else:
                    target = l
                    prev_target = h
                progress = (i - mid_point) / (high_point - mid_point)
                base = prev_target + (target - prev_target) * progress
            else:
                # Move toward close
                if c >= o:
                    prev_target = h
                else:
                    prev_target = l
                progress = (i - high_point) / (steps - high_point)
                base = prev_target + (c - prev_target) * progress

            # Add small noise
            noise = (h - l) * random.uniform(-0.05, 0.05)
            price = max(l, min(h, base + noise))
            prices.append(price)

        # Ensure last price is close
        prices[-1] = c

        return prices

    def replay(self, symbol: str, bars: List[Dict[str, Any]],
               ticks_per_bar: int = 10,
               callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Replay historical bars through the exchange, generating ticks.

        callback: function(symbol, tick, bar_index) called for each tick.
        """
        all_ticks = []
        for i, bar in enumerate(bars):
            self.exchange.update(symbol, bar)
            ticks = self.generate_ticks_from_bar(bar, ticks_per_bar)
            all_ticks.extend(ticks)
            if callback:
                for tick in ticks:
                    callback(symbol, tick, i)

        return all_ticks
