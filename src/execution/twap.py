"""
Mefai Autotrade - TWAP (Time-Weighted Average Price) Execution
Splits large orders into equal time slices with configurable jitter,
price limits, volume participation caps, and real-time monitoring.
"""

import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TWAPStatus(Enum):
    """Status of a TWAP execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class TWAPConfig:
    """Configuration for TWAP execution."""
    symbol: str
    side: str                    # BUY or SELL
    total_quantity: float
    duration_seconds: float      # Total execution window
    num_slices: int = 0          # 0 = auto-calculate based on duration
    price_limit: float = 0.0    # Max price for BUY, min price for SELL. 0 = no limit
    jitter_pct: float = 0.10    # Random timing jitter as fraction of interval (0-0.5)
    volume_participation_limit: float = 0.0  # Max % of market volume per slice (0 = unlimited)
    min_slice_quantity: float = 0.0
    max_slice_quantity: float = 0.0  # 0 = no limit
    order_type: str = "LIMIT"       # LIMIT or MARKET for each slice
    limit_offset_bps: float = 5.0   # For LIMIT orders: offset from mid in bps
    cancel_on_price_move_pct: float = 0.0  # Cancel if price moves X% against us
    time_in_force: str = "IOC"      # IOC, GTC, FOK for each slice
    allow_partial_fills: bool = True
    catch_up_on_miss: bool = True   # If a slice misses, add to next slice
    venue_id: Optional[str] = None

    def __post_init__(self):
        if self.num_slices == 0:
            # Auto-calculate: one slice per 30 seconds minimum
            self.num_slices = max(2, int(self.duration_seconds / 30))
        if self.num_slices < 2:
            self.num_slices = 2
        if self.jitter_pct < 0:
            self.jitter_pct = 0
        if self.jitter_pct > 0.5:
            self.jitter_pct = 0.5

    @property
    def slice_interval(self) -> float:
        """Time between slices in seconds."""
        return self.duration_seconds / self.num_slices

    @property
    def base_slice_quantity(self) -> float:
        """Base quantity per slice before adjustments."""
        return self.total_quantity / self.num_slices


@dataclass
class TWAPSlice:
    """Represents a single time slice in TWAP execution."""
    slice_id: str
    index: int
    scheduled_time: float
    actual_time: float = 0.0
    target_quantity: float = 0.0
    filled_quantity: float = 0.0
    fill_price: float = 0.0
    order_id: Optional[str] = None
    status: str = "pending"      # pending, submitted, filled, partial, failed, skipped
    error_message: str = ""
    latency_ms: float = 0.0
    slippage_bps: float = 0.0
    market_price_at_execution: float = 0.0
    market_volume_at_execution: float = 0.0

    @property
    def fill_pct(self) -> float:
        if self.target_quantity == 0:
            return 0.0
        return min(1.0, self.filled_quantity / self.target_quantity)

    @property
    def notional_value(self) -> float:
        return self.filled_quantity * self.fill_price


@dataclass
class TWAPProgress:
    """Real-time progress of TWAP execution."""
    execution_id: str
    config: TWAPConfig
    status: TWAPStatus
    slices: List[TWAPSlice]
    start_time: float = 0.0
    end_time: float = 0.0
    total_filled: float = 0.0
    total_notional: float = 0.0
    avg_fill_price: float = 0.0
    slices_completed: int = 0
    slices_failed: int = 0
    slices_remaining: int = 0
    elapsed_seconds: float = 0.0
    remaining_seconds: float = 0.0
    fill_pct: float = 0.0
    current_market_price: float = 0.0
    vwap_so_far: float = 0.0
    estimated_completion_time: float = 0.0

    def update(self) -> None:
        """Recalculate progress metrics from slices."""
        self.total_filled = sum(s.filled_quantity for s in self.slices)
        self.total_notional = sum(s.notional_value for s in self.slices)
        if self.total_filled > 0:
            self.avg_fill_price = self.total_notional / self.total_filled
            self.vwap_so_far = self.avg_fill_price
        self.slices_completed = sum(
            1 for s in self.slices if s.status in ("filled", "partial")
        )
        self.slices_failed = sum(1 for s in self.slices if s.status == "failed")
        self.slices_remaining = sum(
            1 for s in self.slices if s.status in ("pending", "submitted")
        )
        if self.config.total_quantity > 0:
            self.fill_pct = self.total_filled / self.config.total_quantity
        if self.start_time > 0:
            self.elapsed_seconds = time.time() - self.start_time
            self.remaining_seconds = max(
                0, self.config.duration_seconds - self.elapsed_seconds
            )
            if self.fill_pct > 0 and self.fill_pct < 1.0:
                rate = self.elapsed_seconds / self.fill_pct
                self.estimated_completion_time = self.start_time + rate


class TWAPExecutor:
    """TWAP execution engine.

    Splits a large order into time-weighted slices, executing at regular
    intervals with configurable jitter for detection avoidance.
    Supports price limits, volume participation caps, and real-time
    progress monitoring.
    """

    def __init__(self):
        self._active_executions: Dict[str, TWAPProgress] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._pause_flags: Dict[str, bool] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._order_executor: Optional[Callable] = None
        self._price_fetcher: Optional[Callable] = None
        self._volume_fetcher: Optional[Callable] = None
        self._lock = asyncio.Lock()
        logger.info("TWAPExecutor initialized")

    def set_order_executor(self, executor: Callable) -> None:
        """Set the function that submits orders to the exchange.
        Signature: async def execute(symbol, side, quantity, order_type,
                                     price, time_in_force, venue_id) -> dict
        Returns dict with 'order_id', 'filled_quantity', 'fill_price', 'status'
        """
        self._order_executor = executor

    def set_price_fetcher(self, fetcher: Callable) -> None:
        """Set function to get current market price.
        Signature: async def fetch(symbol) -> float
        """
        self._price_fetcher = fetcher

    def set_volume_fetcher(self, fetcher: Callable) -> None:
        """Set function to get recent market volume.
        Signature: async def fetch(symbol, period_seconds) -> float
        """
        self._volume_fetcher = fetcher

    async def execute(self, config: TWAPConfig) -> TWAPProgress:
        """Start a TWAP execution. Returns immediately with progress handle.
        The execution runs in background.
        """
        execution_id = f"twap_{uuid.uuid4().hex[:10]}"
        slices = self._build_slices(config, execution_id)

        progress = TWAPProgress(
            execution_id=execution_id,
            config=config,
            status=TWAPStatus.PENDING,
            slices=slices,
            slices_remaining=len(slices),
        )

        async with self._lock:
            self._active_executions[execution_id] = progress
            self._cancel_flags[execution_id] = False
            self._pause_flags[execution_id] = False

        # Start background execution
        asyncio.create_task(self._run_execution(execution_id))

        logger.info(
            "TWAP started: id=%s symbol=%s side=%s qty=%.6f slices=%d duration=%ds",
            execution_id, config.symbol, config.side, config.total_quantity,
            config.num_slices, int(config.duration_seconds),
        )
        return progress

    def _build_slices(self, config: TWAPConfig, execution_id: str) -> List[TWAPSlice]:
        """Build time slices for the execution."""
        slices = []
        base_qty = config.base_slice_quantity
        interval = config.slice_interval
        now = time.time()

        for i in range(config.num_slices):
            # Base scheduled time
            scheduled = now + (i * interval)

            # Apply jitter
            if config.jitter_pct > 0 and i > 0:  # No jitter on first slice
                max_jitter = interval * config.jitter_pct
                jitter = random.uniform(-max_jitter, max_jitter)
                scheduled += jitter

            # Adjust quantity
            qty = base_qty
            if config.min_slice_quantity > 0:
                qty = max(qty, config.min_slice_quantity)
            if config.max_slice_quantity > 0:
                qty = min(qty, config.max_slice_quantity)

            slice_obj = TWAPSlice(
                slice_id=f"{execution_id}_s{i:04d}",
                index=i,
                scheduled_time=scheduled,
                target_quantity=qty,
            )
            slices.append(slice_obj)

        # Ensure total quantity is exact
        total_allocated = sum(s.target_quantity for s in slices)
        if total_allocated > 0:
            remainder = config.total_quantity - total_allocated
            if abs(remainder) > 1e-10:
                # Add remainder to last slice
                slices[-1].target_quantity += remainder

        return slices

    async def _run_execution(self, execution_id: str) -> None:
        """Main execution loop for TWAP."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return

        progress.status = TWAPStatus.RUNNING
        progress.start_time = time.time()
        config = progress.config
        unfilled_carryover = 0.0

        try:
            for i, slice_obj in enumerate(progress.slices):
                # Check cancel
                if self._cancel_flags.get(execution_id, False):
                    slice_obj.status = "skipped"
                    progress.status = TWAPStatus.CANCELLED
                    logger.info("TWAP %s cancelled at slice %d/%d",
                                execution_id, i + 1, config.num_slices)
                    break

                # Check pause
                while self._pause_flags.get(execution_id, False):
                    await asyncio.sleep(0.5)
                    if self._cancel_flags.get(execution_id, False):
                        break

                # Wait until scheduled time
                now = time.time()
                wait_time = slice_obj.scheduled_time - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Check price limit before executing
                if config.price_limit > 0 and self._price_fetcher:
                    current_price = await self._price_fetcher(config.symbol)
                    progress.current_market_price = current_price
                    if not self._price_within_limit(config, current_price):
                        slice_obj.status = "skipped"
                        slice_obj.error_message = (
                            f"Price {current_price} beyond limit {config.price_limit}"
                        )
                        unfilled_carryover += slice_obj.target_quantity
                        logger.info(
                            "TWAP %s slice %d skipped - price limit",
                            execution_id, i + 1,
                        )
                        progress.update()
                        self._notify(execution_id, "slice_skipped", slice_obj)
                        continue

                # Check cancel-on-price-move
                if config.cancel_on_price_move_pct > 0 and self._price_fetcher:
                    if progress.avg_fill_price > 0:
                        current_price = await self._price_fetcher(config.symbol)
                        move_pct = abs(
                            (current_price - progress.avg_fill_price) / progress.avg_fill_price
                        ) * 100.0
                        if move_pct > config.cancel_on_price_move_pct:
                            is_adverse = (
                                (config.side == "BUY" and current_price > progress.avg_fill_price)
                                or (config.side == "SELL" and current_price < progress.avg_fill_price)
                            )
                            if is_adverse:
                                progress.status = TWAPStatus.CANCELLED
                                slice_obj.status = "skipped"
                                logger.warning(
                                    "TWAP %s cancelled - price moved %.2f%% against position",
                                    execution_id, move_pct,
                                )
                                break

                # Check volume participation
                actual_qty = slice_obj.target_quantity
                if config.catch_up_on_miss and unfilled_carryover > 0:
                    actual_qty += unfilled_carryover
                    unfilled_carryover = 0.0

                if config.volume_participation_limit > 0 and self._volume_fetcher:
                    market_volume = await self._volume_fetcher(
                        config.symbol, config.slice_interval
                    )
                    max_participation = market_volume * config.volume_participation_limit
                    if actual_qty > max_participation > 0:
                        carryover = actual_qty - max_participation
                        actual_qty = max_participation
                        unfilled_carryover += carryover
                        logger.debug(
                            "TWAP %s slice %d volume capped: %.6f -> %.6f",
                            execution_id, i + 1, slice_obj.target_quantity, actual_qty,
                        )

                # Determine price for limit orders
                execution_price = None
                if config.order_type == "LIMIT" and self._price_fetcher:
                    current_price = await self._price_fetcher(config.symbol)
                    progress.current_market_price = current_price
                    slice_obj.market_price_at_execution = current_price
                    offset = current_price * config.limit_offset_bps / 10000.0
                    if config.side == "BUY":
                        execution_price = current_price + offset
                    else:
                        execution_price = current_price - offset

                # Execute the slice
                await self._execute_slice(
                    execution_id, slice_obj, actual_qty, execution_price, config
                )

                progress.update()
                self._notify(execution_id, "slice_complete", slice_obj)

            # Mark remaining slices
            for s in progress.slices:
                if s.status == "pending":
                    s.status = "skipped"

            # Final status
            if progress.status == TWAPStatus.RUNNING:
                if progress.fill_pct >= 0.99:
                    progress.status = TWAPStatus.COMPLETED
                elif progress.fill_pct > 0:
                    progress.status = TWAPStatus.COMPLETED
                else:
                    progress.status = TWAPStatus.FAILED

            progress.end_time = time.time()
            progress.update()
            self._notify(execution_id, "execution_complete", None)

            logger.info(
                "TWAP %s finished: status=%s filled=%.6f/%.6f (%.1f%%) avg_price=%.6f",
                execution_id, progress.status.value,
                progress.total_filled, config.total_quantity,
                progress.fill_pct * 100, progress.avg_fill_price,
            )

        except Exception as e:
            progress.status = TWAPStatus.FAILED
            progress.end_time = time.time()
            logger.error("TWAP %s failed with error: %s", execution_id, str(e))
            self._notify(execution_id, "execution_error", str(e))

    async def _execute_slice(self, execution_id: str, slice_obj: TWAPSlice,
                             quantity: float, price: Optional[float],
                             config: TWAPConfig) -> None:
        """Execute a single TWAP slice."""
        if not self._order_executor:
            slice_obj.status = "failed"
            slice_obj.error_message = "No order executor configured"
            return

        if quantity <= 0:
            slice_obj.status = "skipped"
            return

        slice_obj.actual_time = time.time()
        slice_obj.target_quantity = quantity
        slice_obj.status = "submitted"

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
            slice_obj.latency_ms = elapsed_ms

            if result and result.get("status") in ("FILLED", "PARTIALLY_FILLED", "filled", "partial"):
                filled_qty = float(result.get("filled_quantity", 0))
                fill_price = float(result.get("fill_price", 0))
                slice_obj.filled_quantity = filled_qty
                slice_obj.fill_price = fill_price
                slice_obj.order_id = result.get("order_id")

                if filled_qty >= quantity * 0.99:
                    slice_obj.status = "filled"
                else:
                    slice_obj.status = "partial"

                # Calculate slippage
                if price and fill_price > 0:
                    if config.side == "BUY":
                        slice_obj.slippage_bps = (fill_price - price) / price * 10000.0
                    else:
                        slice_obj.slippage_bps = (price - fill_price) / price * 10000.0
            else:
                slice_obj.status = "failed"
                slice_obj.error_message = result.get("error", "Order not filled") if result else "No result"

        except Exception as e:
            slice_obj.status = "failed"
            slice_obj.error_message = str(e)
            logger.error(
                "TWAP %s slice %d failed: %s",
                execution_id, slice_obj.index + 1, str(e),
            )

    def _price_within_limit(self, config: TWAPConfig, price: float) -> bool:
        """Check if current price is within the configured limit."""
        if config.price_limit <= 0:
            return True
        if config.side == "BUY":
            return price <= config.price_limit
        else:
            return price >= config.price_limit

    # -----------------------------------------------------------------------
    # Control methods
    # -----------------------------------------------------------------------

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running TWAP execution."""
        if execution_id not in self._active_executions:
            return False
        self._cancel_flags[execution_id] = True
        logger.info("TWAP cancel requested: %s", execution_id)
        return True

    async def pause(self, execution_id: str) -> bool:
        """Pause a running TWAP execution."""
        if execution_id not in self._active_executions:
            return False
        self._pause_flags[execution_id] = True
        progress = self._active_executions[execution_id]
        progress.status = TWAPStatus.PAUSED
        logger.info("TWAP paused: %s", execution_id)
        return True

    async def resume(self, execution_id: str) -> bool:
        """Resume a paused TWAP execution."""
        if execution_id not in self._active_executions:
            return False
        self._pause_flags[execution_id] = False
        progress = self._active_executions[execution_id]
        progress.status = TWAPStatus.RUNNING
        logger.info("TWAP resumed: %s", execution_id)
        return True

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def get_progress(self, execution_id: str) -> Optional[TWAPProgress]:
        """Get current progress of a TWAP execution."""
        progress = self._active_executions.get(execution_id)
        if progress:
            progress.update()
        return progress

    def get_active_executions(self) -> Dict[str, TWAPProgress]:
        """Get all active TWAP executions."""
        return {
            eid: p for eid, p in self._active_executions.items()
            if p.status in (TWAPStatus.RUNNING, TWAPStatus.PAUSED, TWAPStatus.PENDING)
        }

    def get_execution_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary dict of an execution."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return None
        progress.update()
        return {
            "execution_id": execution_id,
            "symbol": progress.config.symbol,
            "side": progress.config.side,
            "total_quantity": progress.config.total_quantity,
            "total_filled": progress.total_filled,
            "fill_pct": progress.fill_pct,
            "avg_fill_price": progress.avg_fill_price,
            "vwap": progress.vwap_so_far,
            "status": progress.status.value,
            "slices_completed": progress.slices_completed,
            "slices_failed": progress.slices_failed,
            "slices_remaining": progress.slices_remaining,
            "elapsed_seconds": progress.elapsed_seconds,
            "remaining_seconds": progress.remaining_seconds,
            "total_notional": progress.total_notional,
        }

    def get_slice_details(self, execution_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get detailed information about each slice."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return None
        return [
            {
                "slice_id": s.slice_id,
                "index": s.index,
                "status": s.status,
                "target_qty": s.target_quantity,
                "filled_qty": s.filled_quantity,
                "fill_price": s.fill_price,
                "fill_pct": s.fill_pct,
                "slippage_bps": s.slippage_bps,
                "latency_ms": s.latency_ms,
                "market_price": s.market_price_at_execution,
                "error": s.error_message,
            }
            for s in progress.slices
        ]

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def on_event(self, execution_id: str, callback: Callable) -> None:
        """Register a callback for execution events.
        Callback signature: def cb(event_type: str, data: Any) -> None
        """
        if execution_id not in self._callbacks:
            self._callbacks[execution_id] = []
        self._callbacks[execution_id].append(callback)

    def _notify(self, execution_id: str, event_type: str, data: Any) -> None:
        """Notify registered callbacks."""
        callbacks = self._callbacks.get(execution_id, [])
        for cb in callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                logger.error(
                    "TWAP callback error for %s: %s", execution_id, str(e)
                )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove completed executions older than max_age_seconds."""
        now = time.time()
        to_remove = []
        for eid, p in self._active_executions.items():
            if p.status in (TWAPStatus.COMPLETED, TWAPStatus.CANCELLED, TWAPStatus.FAILED):
                if p.end_time > 0 and (now - p.end_time) > max_age_seconds:
                    to_remove.append(eid)
        for eid in to_remove:
            del self._active_executions[eid]
            self._cancel_flags.pop(eid, None)
            self._pause_flags.pop(eid, None)
            self._callbacks.pop(eid, None)
        return len(to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall TWAP executor statistics."""
        total = len(self._active_executions)
        by_status = {}
        total_filled_value = 0.0
        total_slippage = []
        for p in self._active_executions.values():
            status = p.status.value
            by_status[status] = by_status.get(status, 0) + 1
            total_filled_value += p.total_notional
            for s in p.slices:
                if s.status == "filled" and s.slippage_bps != 0:
                    total_slippage.append(s.slippage_bps)

        avg_slippage = sum(total_slippage) / len(total_slippage) if total_slippage else 0.0

        return {
            "total_executions": total,
            "by_status": by_status,
            "total_filled_value_usd": total_filled_value,
            "avg_slippage_bps": avg_slippage,
        }
