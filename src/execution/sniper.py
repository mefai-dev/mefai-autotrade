"""
Mefai Autotrade - Sniper Execution
Precision entry execution that waits for exact price levels,
validates order book depth and liquidity before striking,
and supports multi-level targeting with time limits.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SniperStatus(Enum):
    """Status of a sniper execution."""
    ARMED = "armed"          # Waiting for trigger
    TRIGGERED = "triggered"  # Price reached, executing
    FILLED = "filled"        # Order filled
    PARTIAL = "partial"      # Partially filled
    EXPIRED = "expired"      # Time limit reached
    CANCELLED = "cancelled"  # Manually cancelled
    FAILED = "failed"        # Execution failed
    DISARMED = "disarmed"    # Conditions no longer valid


class TriggerType(Enum):
    """How the sniper triggers."""
    PRICE_CROSS = "price_cross"       # Price crosses target level
    PRICE_TOUCH = "price_touch"       # Price touches target (wick)
    BID_AT = "bid_at"                 # Best bid reaches level
    ASK_AT = "ask_at"                 # Best ask reaches level
    DEPTH_AVAILABLE = "depth_available"  # Sufficient depth at level


@dataclass
class SniperTarget:
    """A price level target for the sniper."""
    target_id: str
    price: float
    quantity: float
    trigger_type: TriggerType = TriggerType.PRICE_CROSS
    direction: str = "below"    # "below" = trigger when price goes below, "above" = above
    priority: int = 0           # Lower = higher priority
    filled: bool = False
    filled_quantity: float = 0.0
    fill_price: float = 0.0
    fill_time: float = 0.0
    order_id: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    status: SniperStatus = SniperStatus.ARMED
    error_message: str = ""

    def is_triggered(self, current_price: float, best_bid: float = 0.0,
                     best_ask: float = 0.0) -> bool:
        """Check if this target should be triggered."""
        if self.filled or self.status != SniperStatus.ARMED:
            return False

        if self.trigger_type == TriggerType.PRICE_CROSS:
            if self.direction == "below":
                return current_price <= self.price
            else:
                return current_price >= self.price

        elif self.trigger_type == TriggerType.PRICE_TOUCH:
            # Within 0.01% of target
            threshold = self.price * 0.0001
            return abs(current_price - self.price) <= threshold

        elif self.trigger_type == TriggerType.BID_AT:
            return best_bid >= self.price if self.direction == "above" else best_bid <= self.price

        elif self.trigger_type == TriggerType.ASK_AT:
            return best_ask <= self.price if self.direction == "below" else best_ask >= self.price

        return False


@dataclass
class LiquidityRequirements:
    """Minimum liquidity requirements before executing."""
    min_depth_quantity: float = 0.0     # Min quantity available at target price level
    min_depth_levels: int = 0           # Min number of price levels with orders
    max_spread_bps: float = 0.0        # Max allowed spread (0 = no limit)
    min_bid_depth_usd: float = 0.0     # Min total bid depth in USD
    min_ask_depth_usd: float = 0.0     # Min total ask depth in USD
    require_both_sides: bool = False    # Require both bid and ask depth


@dataclass
class SniperConfig:
    """Configuration for sniper execution."""
    symbol: str
    side: str                            # BUY or SELL
    targets: List[SniperTarget] = field(default_factory=list)
    total_quantity: float = 0.0          # Total across all targets (0 = sum of target quantities)
    order_type: str = "LIMIT"            # LIMIT or MARKET
    time_in_force: str = "IOC"           # IOC for sniper precision
    max_wait_seconds: float = 0.0       # Max time to wait for trigger (0 = unlimited)
    poll_interval_ms: float = 100.0     # How often to check price
    liquidity_requirements: LiquidityRequirements = field(default_factory=LiquidityRequirements)
    retry_delay_seconds: float = 0.5    # Delay between retry attempts
    cancel_unfilled_after_seconds: float = 5.0  # Cancel IOC order if not filled
    allow_partial: bool = True
    execute_all_targets: bool = False    # True = execute all triggered targets, False = first only
    venue_id: Optional[str] = None

    def __post_init__(self):
        if self.total_quantity == 0 and self.targets:
            self.total_quantity = sum(t.quantity for t in self.targets)
        if not self.targets and self.total_quantity > 0:
            # Create a single target at market price
            pass  # Will be created during execution


@dataclass
class SniperResult:
    """Result of sniper execution."""
    execution_id: str
    config: SniperConfig
    status: SniperStatus
    targets_triggered: int = 0
    targets_filled: int = 0
    total_filled: float = 0.0
    total_notional: float = 0.0
    avg_fill_price: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    wait_time_seconds: float = 0.0
    execution_latency_ms: float = 0.0
    market_price_at_trigger: float = 0.0
    spread_at_trigger_bps: float = 0.0
    slippage_bps: float = 0.0
    depth_at_trigger: float = 0.0


class OrderBookAnalyzer:
    """Analyze order book depth and liquidity conditions."""

    def __init__(self):
        self._depth_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_ms: float = 500.0

    async def analyze(self, symbol: str, depth_fetcher: Callable,
                      requirements: LiquidityRequirements) -> Tuple[bool, Dict[str, Any]]:
        """Check if liquidity requirements are met.

        Args:
            symbol: Trading pair
            depth_fetcher: Async function returning order book data
            requirements: Minimum requirements

        Returns:
            (requirements_met, analysis_data)
        """
        analysis = {
            "spread_bps": 0.0,
            "bid_depth_quantity": 0.0,
            "ask_depth_quantity": 0.0,
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "bid_levels": 0,
            "ask_levels": 0,
            "best_bid": 0.0,
            "best_ask": 0.0,
            "mid_price": 0.0,
            "requirements_met": True,
            "failed_checks": [],
        }

        try:
            depth = await depth_fetcher(symbol)
            if not depth:
                analysis["requirements_met"] = False
                analysis["failed_checks"].append("No depth data")
                return False, analysis

            bids = depth.get("bids", [])
            asks = depth.get("asks", [])

            if bids:
                analysis["best_bid"] = float(bids[0][0]) if isinstance(bids[0], (list, tuple)) else float(bids[0].get("price", 0))
                analysis["bid_levels"] = len(bids)
                for level in bids:
                    price = float(level[0]) if isinstance(level, (list, tuple)) else float(level.get("price", 0))
                    qty = float(level[1]) if isinstance(level, (list, tuple)) else float(level.get("quantity", 0))
                    analysis["bid_depth_quantity"] += qty
                    analysis["bid_depth_usd"] += qty * price

            if asks:
                analysis["best_ask"] = float(asks[0][0]) if isinstance(asks[0], (list, tuple)) else float(asks[0].get("price", 0))
                analysis["ask_levels"] = len(asks)
                for level in asks:
                    price = float(level[0]) if isinstance(level, (list, tuple)) else float(level.get("price", 0))
                    qty = float(level[1]) if isinstance(level, (list, tuple)) else float(level.get("quantity", 0))
                    analysis["ask_depth_quantity"] += qty
                    analysis["ask_depth_usd"] += qty * price

            if analysis["best_bid"] > 0 and analysis["best_ask"] > 0:
                mid = (analysis["best_bid"] + analysis["best_ask"]) / 2
                analysis["mid_price"] = mid
                analysis["spread_bps"] = (
                    (analysis["best_ask"] - analysis["best_bid"]) / mid * 10000.0
                )

            # Check requirements
            met = True
            if requirements.max_spread_bps > 0:
                if analysis["spread_bps"] > requirements.max_spread_bps:
                    met = False
                    analysis["failed_checks"].append(
                        f"Spread {analysis['spread_bps']:.1f}bps > max {requirements.max_spread_bps:.1f}bps"
                    )

            if requirements.min_depth_quantity > 0:
                relevant_depth = (
                    analysis["ask_depth_quantity"] if True  # BUY side
                    else analysis["bid_depth_quantity"]
                )
                if relevant_depth < requirements.min_depth_quantity:
                    met = False
                    analysis["failed_checks"].append(
                        f"Depth {relevant_depth:.4f} < min {requirements.min_depth_quantity:.4f}"
                    )

            if requirements.min_depth_levels > 0:
                min_levels = min(analysis["bid_levels"], analysis["ask_levels"])
                if min_levels < requirements.min_depth_levels:
                    met = False
                    analysis["failed_checks"].append(
                        f"Levels {min_levels} < min {requirements.min_depth_levels}"
                    )

            if requirements.min_bid_depth_usd > 0:
                if analysis["bid_depth_usd"] < requirements.min_bid_depth_usd:
                    met = False
                    analysis["failed_checks"].append("Insufficient bid depth USD")

            if requirements.min_ask_depth_usd > 0:
                if analysis["ask_depth_usd"] < requirements.min_ask_depth_usd:
                    met = False
                    analysis["failed_checks"].append("Insufficient ask depth USD")

            analysis["requirements_met"] = met
            return met, analysis

        except Exception as e:
            analysis["requirements_met"] = False
            analysis["failed_checks"].append(str(e))
            return False, analysis


class SpreadAnalyzer:
    """Track and analyze bid-ask spreads over time."""

    def __init__(self, window_size: int = 500):
        self._spreads: Dict[str, deque] = {}
        self._window_size = window_size

    def record(self, symbol: str, spread_bps: float) -> None:
        """Record a spread observation."""
        if symbol not in self._spreads:
            self._spreads[symbol] = deque(maxlen=self._window_size)
        self._spreads[symbol].append((time.time(), spread_bps))

    def get_avg_spread(self, symbol: str, lookback_seconds: float = 60.0) -> float:
        """Get average spread over recent period."""
        spreads = self._spreads.get(symbol)
        if not spreads:
            return float("inf")
        cutoff = time.time() - lookback_seconds
        recent = [s for t, s in spreads if t > cutoff]
        if not recent:
            return float("inf")
        return sum(recent) / len(recent)

    def is_tight(self, symbol: str, threshold_bps: float = 5.0) -> bool:
        """Check if current spread is tight enough."""
        avg = self.get_avg_spread(symbol, lookback_seconds=10.0)
        return avg <= threshold_bps

    def get_percentile(self, symbol: str, pct: float = 50.0) -> float:
        """Get spread percentile."""
        spreads = self._spreads.get(symbol)
        if not spreads:
            return float("inf")
        values = sorted([s for _, s in spreads])
        idx = int(len(values) * pct / 100.0)
        return values[min(idx, len(values) - 1)]


class SniperExecutor:
    """Sniper execution engine.

    Waits for exact price levels with configurable trigger conditions,
    validates liquidity before executing, and supports multi-target
    setups with time limits.
    """

    def __init__(self):
        self._active_executions: Dict[str, SniperResult] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._order_executor: Optional[Callable] = None
        self._price_fetcher: Optional[Callable] = None
        self._depth_fetcher: Optional[Callable] = None
        self._spread_fetcher: Optional[Callable] = None
        self._order_book_analyzer = OrderBookAnalyzer()
        self._spread_analyzer = SpreadAnalyzer()
        self._lock = asyncio.Lock()
        logger.info("SniperExecutor initialized")

    def set_order_executor(self, executor: Callable) -> None:
        """Set order execution function."""
        self._order_executor = executor

    def set_price_fetcher(self, fetcher: Callable) -> None:
        """Set price fetcher: async def fetch(symbol) -> float"""
        self._price_fetcher = fetcher

    def set_depth_fetcher(self, fetcher: Callable) -> None:
        """Set depth fetcher: async def fetch(symbol) -> dict with 'bids', 'asks'"""
        self._depth_fetcher = fetcher

    def set_spread_fetcher(self, fetcher: Callable) -> None:
        """Set spread fetcher: async def fetch(symbol) -> dict with 'bid', 'ask', 'spread_bps'"""
        self._spread_fetcher = fetcher

    async def arm(self, config: SniperConfig) -> SniperResult:
        """Arm a sniper execution. Starts watching for trigger conditions."""
        execution_id = f"snp_{uuid.uuid4().hex[:10]}"

        result = SniperResult(
            execution_id=execution_id,
            config=config,
            status=SniperStatus.ARMED,
            start_time=time.time(),
        )

        async with self._lock:
            self._active_executions[execution_id] = result
            self._cancel_flags[execution_id] = False

        asyncio.create_task(self._watch_and_execute(execution_id))

        targets_desc = ", ".join(
            f"{t.price} ({t.direction})" for t in config.targets
        )
        logger.info(
            "Sniper armed: id=%s symbol=%s side=%s qty=%.6f targets=[%s]",
            execution_id, config.symbol, config.side, config.total_quantity, targets_desc,
        )
        return result

    async def _watch_and_execute(self, execution_id: str) -> None:
        """Main watching loop - polls price and triggers execution."""
        result = self._active_executions.get(execution_id)
        if not result:
            return

        config = result.config
        poll_interval = config.poll_interval_ms / 1000.0

        try:
            while True:
                # Check cancel
                if self._cancel_flags.get(execution_id, False):
                    result.status = SniperStatus.CANCELLED
                    break

                # Check time limit
                if config.max_wait_seconds > 0:
                    elapsed = time.time() - result.start_time
                    if elapsed > config.max_wait_seconds:
                        result.status = SniperStatus.EXPIRED
                        result.wait_time_seconds = elapsed
                        logger.info("Sniper %s expired after %.1fs", execution_id, elapsed)
                        break

                # Get current market data
                if not self._price_fetcher:
                    result.status = SniperStatus.FAILED
                    break

                current_price = await self._price_fetcher(config.symbol)
                best_bid = 0.0
                best_ask = 0.0
                spread_bps = 0.0

                if self._spread_fetcher:
                    spread_data = await self._spread_fetcher(config.symbol)
                    if spread_data:
                        best_bid = spread_data.get("bid", 0)
                        best_ask = spread_data.get("ask", 0)
                        spread_bps = spread_data.get("spread_bps", 0)
                        self._spread_analyzer.record(config.symbol, spread_bps)

                # Check each target
                triggered_targets = []
                for target in config.targets:
                    if target.is_triggered(current_price, best_bid, best_ask):
                        triggered_targets.append(target)

                if not triggered_targets:
                    await asyncio.sleep(poll_interval)
                    continue

                # Sort by priority
                triggered_targets.sort(key=lambda t: t.priority)

                if not config.execute_all_targets:
                    triggered_targets = triggered_targets[:1]

                result.market_price_at_trigger = current_price
                result.spread_at_trigger_bps = spread_bps
                result.wait_time_seconds = time.time() - result.start_time

                # Check liquidity requirements
                liquidity_ok = True
                if self._depth_fetcher:
                    liquidity_ok, depth_analysis = await self._order_book_analyzer.analyze(
                        config.symbol, self._depth_fetcher, config.liquidity_requirements
                    )
                    result.depth_at_trigger = depth_analysis.get("ask_depth_quantity", 0)
                    if config.side == "SELL":
                        result.depth_at_trigger = depth_analysis.get("bid_depth_quantity", 0)

                if not liquidity_ok:
                    logger.debug(
                        "Sniper %s: trigger at %.6f but liquidity insufficient",
                        execution_id, current_price,
                    )
                    await asyncio.sleep(poll_interval)
                    continue

                # Execute triggered targets
                result.status = SniperStatus.TRIGGERED
                result.targets_triggered = len(triggered_targets)
                self._notify(execution_id, "triggered", triggered_targets)

                for target in triggered_targets:
                    await self._execute_target(execution_id, target, config)

                # Compute results
                result.targets_filled = sum(
                    1 for t in config.targets if t.filled
                )
                result.total_filled = sum(
                    t.filled_quantity for t in config.targets
                )
                result.total_notional = sum(
                    t.filled_quantity * t.fill_price for t in config.targets if t.filled
                )
                if result.total_filled > 0:
                    result.avg_fill_price = result.total_notional / result.total_filled
                    # Slippage vs target price
                    target_vwap = sum(
                        t.price * t.filled_quantity for t in config.targets if t.filled
                    ) / result.total_filled
                    if config.side == "BUY":
                        result.slippage_bps = (
                            (result.avg_fill_price - target_vwap) / target_vwap * 10000.0
                        )
                    else:
                        result.slippage_bps = (
                            (target_vwap - result.avg_fill_price) / target_vwap * 10000.0
                        )

                # Check if all targets handled
                all_done = all(
                    t.filled or t.status != SniperStatus.ARMED
                    for t in config.targets
                )
                if all_done or not config.execute_all_targets:
                    if result.total_filled >= config.total_quantity * 0.99:
                        result.status = SniperStatus.FILLED
                    elif result.total_filled > 0:
                        result.status = SniperStatus.PARTIAL
                    else:
                        result.status = SniperStatus.FAILED
                    break

                # Continue watching remaining targets
                await asyncio.sleep(poll_interval)

        except Exception as e:
            result.status = SniperStatus.FAILED
            logger.error("Sniper %s error: %s", execution_id, str(e))

        result.end_time = time.time()
        if result.start_time > 0:
            result.execution_latency_ms = (result.end_time - result.start_time) * 1000.0

        self._notify(execution_id, "complete", result)

        logger.info(
            "Sniper %s finished: status=%s filled=%.6f/%.6f wait=%.1fs slippage=%.2fbps",
            execution_id, result.status.value, result.total_filled,
            config.total_quantity, result.wait_time_seconds, result.slippage_bps,
        )

    async def _execute_target(self, execution_id: str, target: SniperTarget,
                               config: SniperConfig) -> None:
        """Execute a single sniper target."""
        if not self._order_executor:
            target.status = SniperStatus.FAILED
            target.error_message = "No order executor"
            return

        for attempt in range(target.max_attempts):
            target.attempts = attempt + 1
            target.status = SniperStatus.TRIGGERED

            try:
                # Determine execution price
                price = target.price if config.order_type == "LIMIT" else None

                start = time.time()
                result = await self._order_executor(
                    symbol=config.symbol,
                    side=config.side,
                    quantity=target.quantity,
                    order_type=config.order_type,
                    price=price,
                    time_in_force=config.time_in_force,
                    venue_id=config.venue_id,
                )
                latency_ms = (time.time() - start) * 1000.0

                if result and result.get("status") in ("FILLED", "PARTIALLY_FILLED", "filled", "partial"):
                    filled_qty = float(result.get("filled_quantity", 0))
                    fill_price = float(result.get("fill_price", 0))
                    target.filled_quantity = filled_qty
                    target.fill_price = fill_price
                    target.fill_time = time.time()
                    target.order_id = result.get("order_id")

                    if filled_qty >= target.quantity * 0.99:
                        target.filled = True
                        target.status = SniperStatus.FILLED
                    elif config.allow_partial:
                        target.filled = True
                        target.status = SniperStatus.PARTIAL
                    else:
                        # Reject partial - retry
                        target.error_message = f"Partial fill {filled_qty}/{target.quantity}"
                        continue

                    logger.info(
                        "Sniper %s target %.6f filled: qty=%.6f price=%.6f latency=%.0fms",
                        execution_id, target.price, filled_qty, fill_price, latency_ms,
                    )
                    return

                else:
                    error = result.get("error", "Not filled") if result else "No result"
                    target.error_message = error
                    logger.debug(
                        "Sniper %s target %.6f attempt %d failed: %s",
                        execution_id, target.price, attempt + 1, error,
                    )

            except Exception as e:
                target.error_message = str(e)
                logger.error(
                    "Sniper %s target error (attempt %d): %s",
                    execution_id, attempt + 1, str(e),
                )

            # Wait before retry
            if attempt < target.max_attempts - 1:
                await asyncio.sleep(config.retry_delay_seconds)

        # All attempts exhausted
        target.status = SniperStatus.FAILED

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a sniper execution."""
        if execution_id not in self._active_executions:
            return False
        self._cancel_flags[execution_id] = True
        logger.info("Sniper cancel requested: %s", execution_id)
        return True

    async def disarm(self, execution_id: str) -> bool:
        """Disarm a sniper (cancel without error status)."""
        if execution_id not in self._active_executions:
            return False
        self._cancel_flags[execution_id] = True
        result = self._active_executions[execution_id]
        result.status = SniperStatus.DISARMED
        return True

    async def modify_target(self, execution_id: str, target_id: str,
                             new_price: Optional[float] = None,
                             new_quantity: Optional[float] = None) -> bool:
        """Modify a target while sniper is armed."""
        result = self._active_executions.get(execution_id)
        if not result or result.status != SniperStatus.ARMED:
            return False
        for target in result.config.targets:
            if target.target_id == target_id and not target.filled:
                if new_price is not None:
                    target.price = new_price
                if new_quantity is not None:
                    target.quantity = new_quantity
                logger.info(
                    "Sniper %s target %s modified: price=%.6f qty=%.6f",
                    execution_id, target_id,
                    target.price, target.quantity,
                )
                return True
        return False

    def add_target(self, execution_id: str, target: SniperTarget) -> bool:
        """Add a new target to an armed sniper."""
        result = self._active_executions.get(execution_id)
        if not result or result.status != SniperStatus.ARMED:
            return False
        result.config.targets.append(target)
        result.config.total_quantity += target.quantity
        return True

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def get_result(self, execution_id: str) -> Optional[SniperResult]:
        """Get sniper result."""
        return self._active_executions.get(execution_id)

    def get_active_snipers(self) -> Dict[str, SniperResult]:
        """Get all armed/running snipers."""
        return {
            eid: r for eid, r in self._active_executions.items()
            if r.status in (SniperStatus.ARMED, SniperStatus.TRIGGERED)
        }

    def get_spread_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get spread analysis for a symbol."""
        return {
            "avg_spread_10s": self._spread_analyzer.get_avg_spread(symbol, 10.0),
            "avg_spread_60s": self._spread_analyzer.get_avg_spread(symbol, 60.0),
            "median_spread": self._spread_analyzer.get_percentile(symbol, 50.0),
            "p95_spread": self._spread_analyzer.get_percentile(symbol, 95.0),
            "is_tight": self._spread_analyzer.is_tight(symbol),
        }

    # -----------------------------------------------------------------------
    # Callbacks and cleanup
    # -----------------------------------------------------------------------

    def on_event(self, execution_id: str, callback: Callable) -> None:
        """Register event callback."""
        if execution_id not in self._callbacks:
            self._callbacks[execution_id] = []
        self._callbacks[execution_id].append(callback)

    def _notify(self, execution_id: str, event_type: str, data: Any) -> None:
        """Notify callbacks."""
        for cb in self._callbacks.get(execution_id, []):
            try:
                cb(event_type, data)
            except Exception as e:
                logger.error("Sniper callback error: %s", str(e))

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove old completed executions."""
        now = time.time()
        to_remove = []
        for eid, r in self._active_executions.items():
            if r.status in (SniperStatus.FILLED, SniperStatus.PARTIAL,
                           SniperStatus.CANCELLED, SniperStatus.EXPIRED,
                           SniperStatus.FAILED, SniperStatus.DISARMED):
                if r.end_time > 0 and (now - r.end_time) > max_age_seconds:
                    to_remove.append(eid)
        for eid in to_remove:
            del self._active_executions[eid]
            self._cancel_flags.pop(eid, None)
            self._callbacks.pop(eid, None)
        return len(to_remove)
