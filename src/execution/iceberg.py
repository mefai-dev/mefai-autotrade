"""
Mefai Autotrade - Iceberg Order Execution
Shows only a fraction of total order size, refreshing the visible
quantity after each fill. Supports random size variation, passive-only
mode, and dynamic price adjustment.
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class IcebergStatus(Enum):
    """Status of an iceberg execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class IcebergConfig:
    """Configuration for iceberg order execution."""
    symbol: str
    side: str                           # BUY or SELL
    total_quantity: float               # Full hidden order size
    visible_quantity: float             # Quantity shown at a time
    visible_variation_pct: float = 0.15 # Random variation in visible qty (0-0.5)
    price: float = 0.0                 # Initial limit price
    price_adjustment_bps: float = 0.0  # Adjust price between refreshes (basis points)
    max_price_drift_pct: float = 0.5   # Max total price drift from initial price (%)
    follow_market: bool = True         # Adjust price to follow market mid
    follow_offset_bps: float = 2.0     # Offset from mid when following market
    max_execution_time_seconds: float = 3600.0  # Max total execution time (0 = unlimited)
    passive_only: bool = False         # Only post limit orders, never cross spread
    min_refresh_interval_seconds: float = 0.5   # Min time between refreshes
    max_refresh_interval_seconds: float = 5.0   # Max time between refreshes
    order_type: str = "LIMIT"
    time_in_force: str = "GTC"         # GTC for passive, IOC for aggressive
    venue_id: Optional[str] = None
    cancel_on_adverse_move_pct: float = 0.0  # Cancel if price moves X% against

    def __post_init__(self):
        if self.visible_quantity <= 0:
            self.visible_quantity = self.total_quantity * 0.1
        if self.visible_quantity > self.total_quantity:
            self.visible_quantity = self.total_quantity
        if self.visible_variation_pct < 0:
            self.visible_variation_pct = 0
        if self.visible_variation_pct > 0.5:
            self.visible_variation_pct = 0.5

    @property
    def estimated_refreshes(self) -> int:
        """Estimated number of refresh cycles needed."""
        if self.visible_quantity <= 0:
            return 0
        return max(1, int(self.total_quantity / self.visible_quantity))


@dataclass
class IcebergRefresh:
    """Record of a single iceberg refresh cycle."""
    refresh_id: str
    index: int
    timestamp: float
    visible_quantity: float
    filled_quantity: float = 0.0
    fill_price: float = 0.0
    order_id: Optional[str] = None
    status: str = "pending"     # pending, submitted, filled, partial, failed, cancelled
    error_message: str = ""
    latency_ms: float = 0.0
    market_price: float = 0.0
    market_spread_bps: float = 0.0

    @property
    def notional_value(self) -> float:
        return self.filled_quantity * self.fill_price


@dataclass
class IcebergProgress:
    """Real-time progress of iceberg execution."""
    execution_id: str
    config: IcebergConfig
    status: IcebergStatus
    refreshes: List[IcebergRefresh]
    start_time: float = 0.0
    end_time: float = 0.0
    total_filled: float = 0.0
    total_notional: float = 0.0
    avg_fill_price: float = 0.0
    remaining_quantity: float = 0.0
    refreshes_completed: int = 0
    refreshes_failed: int = 0
    current_price: float = 0.0
    current_visible_order_id: Optional[str] = None
    fill_pct: float = 0.0
    elapsed_seconds: float = 0.0

    def update(self) -> None:
        """Recalculate metrics."""
        self.total_filled = sum(r.filled_quantity for r in self.refreshes)
        self.total_notional = sum(r.notional_value for r in self.refreshes)
        if self.total_filled > 0:
            self.avg_fill_price = self.total_notional / self.total_filled
        self.remaining_quantity = max(0, self.config.total_quantity - self.total_filled)
        self.refreshes_completed = sum(
            1 for r in self.refreshes if r.status in ("filled", "partial")
        )
        self.refreshes_failed = sum(1 for r in self.refreshes if r.status == "failed")
        if self.config.total_quantity > 0:
            self.fill_pct = self.total_filled / self.config.total_quantity
        if self.start_time > 0:
            self.elapsed_seconds = time.time() - self.start_time


class IcebergExecutor:
    """Iceberg order execution engine.

    Places visible-quantity orders and automatically refreshes them
    after each fill until the total quantity is executed.
    """

    def __init__(self):
        self._active_executions: Dict[str, IcebergProgress] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._pause_flags: Dict[str, bool] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._order_executor: Optional[Callable] = None
        self._order_canceller: Optional[Callable] = None
        self._price_fetcher: Optional[Callable] = None
        self._spread_fetcher: Optional[Callable] = None
        self._lock = asyncio.Lock()
        logger.info("IcebergExecutor initialized")

    def set_order_executor(self, executor: Callable) -> None:
        """Set order execution function.
        Signature: async def execute(symbol, side, quantity, order_type,
                                     price, time_in_force, venue_id) -> dict
        """
        self._order_executor = executor

    def set_order_canceller(self, canceller: Callable) -> None:
        """Set order cancel function.
        Signature: async def cancel(symbol, order_id, venue_id) -> bool
        """
        self._order_canceller = canceller

    def set_price_fetcher(self, fetcher: Callable) -> None:
        """Set price fetcher: async def fetch(symbol) -> float"""
        self._price_fetcher = fetcher

    def set_spread_fetcher(self, fetcher: Callable) -> None:
        """Set spread fetcher: async def fetch(symbol) -> dict with 'bid', 'ask', 'spread_bps'"""
        self._spread_fetcher = fetcher

    async def execute(self, config: IcebergConfig) -> IcebergProgress:
        """Start an iceberg execution."""
        execution_id = f"ice_{uuid.uuid4().hex[:10]}"

        progress = IcebergProgress(
            execution_id=execution_id,
            config=config,
            status=IcebergStatus.PENDING,
            refreshes=[],
            remaining_quantity=config.total_quantity,
        )

        async with self._lock:
            self._active_executions[execution_id] = progress
            self._cancel_flags[execution_id] = False
            self._pause_flags[execution_id] = False

        asyncio.create_task(self._run_execution(execution_id))

        logger.info(
            "Iceberg started: id=%s symbol=%s side=%s total=%.6f visible=%.6f passive=%s",
            execution_id, config.symbol, config.side, config.total_quantity,
            config.visible_quantity, config.passive_only,
        )
        return progress

    async def _run_execution(self, execution_id: str) -> None:
        """Main iceberg execution loop."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return

        progress.status = IcebergStatus.RUNNING
        progress.start_time = time.time()
        config = progress.config
        refresh_index = 0
        current_price = config.price
        initial_price = config.price
        consecutive_failures = 0
        max_consecutive_failures = 5

        try:
            # Get initial price if not set
            if current_price <= 0 and self._price_fetcher:
                current_price = await self._price_fetcher(config.symbol)
                initial_price = current_price

            while progress.remaining_quantity > 1e-10:
                # Check cancel
                if self._cancel_flags.get(execution_id, False):
                    progress.status = IcebergStatus.CANCELLED
                    break

                # Check pause
                while self._pause_flags.get(execution_id, False):
                    await asyncio.sleep(0.5)
                    if self._cancel_flags.get(execution_id, False):
                        break

                # Check execution time limit
                if config.max_execution_time_seconds > 0:
                    elapsed = time.time() - progress.start_time
                    if elapsed > config.max_execution_time_seconds:
                        progress.status = IcebergStatus.EXPIRED
                        logger.info("Iceberg %s expired after %.0fs", execution_id, elapsed)
                        break

                # Check consecutive failure limit
                if consecutive_failures >= max_consecutive_failures:
                    progress.status = IcebergStatus.FAILED
                    logger.warning(
                        "Iceberg %s failed: %d consecutive failures",
                        execution_id, consecutive_failures,
                    )
                    break

                # Determine visible quantity with random variation
                visible_qty = self._randomize_quantity(
                    config.visible_quantity, config.visible_variation_pct
                )
                # Don't exceed remaining
                visible_qty = min(visible_qty, progress.remaining_quantity)

                # Determine price
                execution_price = await self._determine_price(
                    config, current_price, initial_price, refresh_index
                )
                if execution_price:
                    current_price = execution_price

                # Check passive-only constraint
                if config.passive_only and self._spread_fetcher:
                    spread_data = await self._spread_fetcher(config.symbol)
                    if spread_data:
                        progress.current_price = spread_data.get(
                            "mid", (spread_data.get("bid", 0) + spread_data.get("ask", 0)) / 2
                        )
                        if config.side == "BUY":
                            # Must be at or below best bid
                            best_bid = spread_data.get("bid", 0)
                            if execution_price and execution_price > best_bid:
                                execution_price = best_bid
                        else:
                            # Must be at or above best ask
                            best_ask = spread_data.get("ask", 0)
                            if execution_price and execution_price < best_ask:
                                execution_price = best_ask

                # Check adverse price move
                if config.cancel_on_adverse_move_pct > 0 and initial_price > 0:
                    if self._price_fetcher:
                        live_price = await self._price_fetcher(config.symbol)
                        move_pct = abs((live_price - initial_price) / initial_price) * 100
                        is_adverse = (
                            (config.side == "BUY" and live_price > initial_price)
                            or (config.side == "SELL" and live_price < initial_price)
                        )
                        if is_adverse and move_pct > config.cancel_on_adverse_move_pct:
                            progress.status = IcebergStatus.CANCELLED
                            logger.info(
                                "Iceberg %s cancelled: adverse move %.2f%%",
                                execution_id, move_pct,
                            )
                            break

                # Create refresh record
                refresh = IcebergRefresh(
                    refresh_id=f"{execution_id}_r{refresh_index:04d}",
                    index=refresh_index,
                    timestamp=time.time(),
                    visible_quantity=visible_qty,
                    market_price=current_price,
                )
                progress.refreshes.append(refresh)

                # Execute the visible order
                await self._execute_refresh(
                    execution_id, refresh, visible_qty, execution_price, config
                )

                if refresh.status in ("filled", "partial"):
                    consecutive_failures = 0
                    progress.update()
                    self._notify(execution_id, "refresh_filled", refresh)
                else:
                    consecutive_failures += 1
                    progress.update()
                    self._notify(execution_id, "refresh_failed", refresh)

                refresh_index += 1

                # Wait between refreshes (randomized)
                wait = random.uniform(
                    config.min_refresh_interval_seconds,
                    config.max_refresh_interval_seconds,
                )
                await asyncio.sleep(wait)

            # Final status
            if progress.status == IcebergStatus.RUNNING:
                if progress.fill_pct >= 0.99:
                    progress.status = IcebergStatus.COMPLETED
                elif progress.fill_pct > 0:
                    progress.status = IcebergStatus.COMPLETED
                else:
                    progress.status = IcebergStatus.FAILED

            progress.end_time = time.time()
            progress.update()
            self._notify(execution_id, "execution_complete", None)

            logger.info(
                "Iceberg %s finished: status=%s filled=%.6f/%.6f (%.1f%%) refreshes=%d avg=%.6f",
                execution_id, progress.status.value,
                progress.total_filled, config.total_quantity,
                progress.fill_pct * 100, refresh_index, progress.avg_fill_price,
            )

        except Exception as e:
            progress.status = IcebergStatus.FAILED
            progress.end_time = time.time()
            logger.error("Iceberg %s failed: %s", execution_id, str(e))
            self._notify(execution_id, "execution_error", str(e))

    def _randomize_quantity(self, base_qty: float, variation_pct: float) -> float:
        """Add random variation to visible quantity."""
        if variation_pct <= 0:
            return base_qty
        min_qty = base_qty * (1.0 - variation_pct)
        max_qty = base_qty * (1.0 + variation_pct)
        return random.uniform(min_qty, max_qty)

    async def _determine_price(self, config: IcebergConfig, current_price: float,
                                initial_price: float,
                                refresh_index: int) -> Optional[float]:
        """Determine the price for the next refresh."""
        if config.order_type != "LIMIT":
            return None

        price = current_price

        # Follow market
        if config.follow_market and self._price_fetcher:
            market_price = await self._price_fetcher(config.symbol)
            offset = market_price * config.follow_offset_bps / 10000.0
            if config.side == "BUY":
                price = market_price - offset  # Bid below market
            else:
                price = market_price + offset  # Ask above market

        # Apply per-refresh adjustment
        if config.price_adjustment_bps != 0 and refresh_index > 0:
            adjustment = price * config.price_adjustment_bps / 10000.0
            price += adjustment

        # Enforce max price drift
        if config.max_price_drift_pct > 0 and initial_price > 0:
            max_drift = initial_price * config.max_price_drift_pct / 100.0
            if config.side == "BUY":
                price = min(price, initial_price + max_drift)
            else:
                price = max(price, initial_price - max_drift)

        return price

    async def _execute_refresh(self, execution_id: str, refresh: IcebergRefresh,
                                quantity: float, price: Optional[float],
                                config: IcebergConfig) -> None:
        """Execute a single refresh order."""
        if not self._order_executor:
            refresh.status = "failed"
            refresh.error_message = "No order executor"
            return

        if quantity <= 0:
            refresh.status = "skipped"
            return

        refresh.status = "submitted"

        try:
            start = time.time()
            result = await self._order_executor(
                symbol=config.symbol,
                side=config.side,
                quantity=quantity,
                order_type=config.order_type,
                price=price,
                time_in_force=config.time_in_force,
                venue_id=config.venue_id,
            )
            elapsed_ms = (time.time() - start) * 1000.0
            refresh.latency_ms = elapsed_ms

            if result and result.get("status") in ("FILLED", "PARTIALLY_FILLED", "filled", "partial"):
                filled_qty = float(result.get("filled_quantity", 0))
                fill_price = float(result.get("fill_price", 0))
                refresh.filled_quantity = filled_qty
                refresh.fill_price = fill_price
                refresh.order_id = result.get("order_id")
                refresh.status = "filled" if filled_qty >= quantity * 0.99 else "partial"
            else:
                refresh.status = "failed"
                refresh.error_message = result.get("error", "Not filled") if result else "No result"

                # For GTC orders that weren't immediately filled, we may need to wait
                if config.time_in_force == "GTC" and result and result.get("order_id"):
                    refresh.order_id = result.get("order_id")
                    refresh.status = "submitted"
                    # Wait for fill (with timeout)
                    filled = await self._wait_for_fill(
                        execution_id, refresh, config, timeout_seconds=30.0
                    )
                    if not filled:
                        # Cancel unfilled order
                        if self._order_canceller and refresh.order_id:
                            await self._order_canceller(
                                config.symbol, refresh.order_id, config.venue_id
                            )
                        refresh.status = "cancelled"

        except Exception as e:
            refresh.status = "failed"
            refresh.error_message = str(e)
            logger.error("Iceberg %s refresh %d error: %s", execution_id, refresh.index, str(e))

    async def _wait_for_fill(self, execution_id: str, refresh: IcebergRefresh,
                              config: IcebergConfig,
                              timeout_seconds: float = 30.0) -> bool:
        """Wait for a GTC order to fill with timeout."""
        start = time.time()
        poll_interval = 0.5

        while time.time() - start < timeout_seconds:
            if self._cancel_flags.get(execution_id, False):
                return False
            # In production, this would check order status via API
            # For now, we assume GTC orders get handled by the exchange
            await asyncio.sleep(poll_interval)
            # If we had an order status checker, we'd check here
            # For the framework, we return False to cancel unfilled GTC orders
            break

        return False

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    async def cancel(self, execution_id: str) -> bool:
        """Cancel iceberg execution and any outstanding orders."""
        if execution_id not in self._active_executions:
            return False

        self._cancel_flags[execution_id] = True
        progress = self._active_executions[execution_id]

        # Cancel any outstanding order
        if self._order_canceller and progress.current_visible_order_id:
            try:
                await self._order_canceller(
                    progress.config.symbol,
                    progress.current_visible_order_id,
                    progress.config.venue_id,
                )
            except Exception as e:
                logger.error("Failed to cancel outstanding order: %s", str(e))

        logger.info("Iceberg cancel requested: %s", execution_id)
        return True

    async def pause(self, execution_id: str) -> bool:
        """Pause execution."""
        if execution_id not in self._active_executions:
            return False
        self._pause_flags[execution_id] = True
        self._active_executions[execution_id].status = IcebergStatus.PAUSED
        return True

    async def resume(self, execution_id: str) -> bool:
        """Resume execution."""
        if execution_id not in self._active_executions:
            return False
        self._pause_flags[execution_id] = False
        self._active_executions[execution_id].status = IcebergStatus.RUNNING
        return True

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def get_progress(self, execution_id: str) -> Optional[IcebergProgress]:
        """Get current progress."""
        progress = self._active_executions.get(execution_id)
        if progress:
            progress.update()
        return progress

    def get_active_executions(self) -> Dict[str, IcebergProgress]:
        """Get all active executions."""
        return {
            eid: p for eid, p in self._active_executions.items()
            if p.status in (IcebergStatus.RUNNING, IcebergStatus.PAUSED, IcebergStatus.PENDING)
        }

    def get_execution_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of execution."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return None
        progress.update()
        return {
            "execution_id": execution_id,
            "symbol": progress.config.symbol,
            "side": progress.config.side,
            "total_quantity": progress.config.total_quantity,
            "visible_quantity": progress.config.visible_quantity,
            "total_filled": progress.total_filled,
            "remaining": progress.remaining_quantity,
            "fill_pct": progress.fill_pct,
            "avg_fill_price": progress.avg_fill_price,
            "status": progress.status.value,
            "refreshes_completed": progress.refreshes_completed,
            "refreshes_failed": progress.refreshes_failed,
            "elapsed_seconds": progress.elapsed_seconds,
            "passive_only": progress.config.passive_only,
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
                logger.error("Iceberg callback error: %s", str(e))

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove old completed executions."""
        now = time.time()
        to_remove = []
        for eid, p in self._active_executions.items():
            if p.status in (IcebergStatus.COMPLETED, IcebergStatus.CANCELLED,
                           IcebergStatus.FAILED, IcebergStatus.EXPIRED):
                if p.end_time > 0 and (now - p.end_time) > max_age_seconds:
                    to_remove.append(eid)
        for eid in to_remove:
            del self._active_executions[eid]
            self._cancel_flags.pop(eid, None)
            self._pause_flags.pop(eid, None)
            self._callbacks.pop(eid, None)
        return len(to_remove)
