"""
Mefai Autotrade - VWAP (Volume-Weighted Average Price) Execution
Sizes order slices according to historical volume profile for minimal
market impact. Supports front/back loading, real-time adjustment,
and benchmark tracking.
"""

import asyncio
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VWAPStatus(Enum):
    """Status of a VWAP execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class VWAPMode(Enum):
    """Execution aggressiveness mode."""
    PASSIVE = "passive"        # Only post limit orders on near side
    NEUTRAL = "neutral"        # Standard volume-weighted
    AGGRESSIVE = "aggressive"  # Cross spread when behind schedule


@dataclass
class VolumeProfile:
    """Historical volume profile for a symbol.
    Stores volume distribution across time intervals for the trading day.
    """
    symbol: str
    interval_minutes: int = 30
    # List of (start_minute_of_day, relative_volume) tuples
    # relative_volume is fraction of daily volume in that interval (sums to 1.0)
    profile: List[Tuple[int, float]] = field(default_factory=list)
    total_daily_volume: float = 0.0
    last_updated: float = 0.0
    data_days: int = 0  # How many days of data used to build profile

    def get_volume_fraction(self, minute_of_day: int) -> float:
        """Get the expected volume fraction for a given time of day."""
        if not self.profile:
            # Flat profile if no data
            return 1.0 / (1440 / self.interval_minutes)
        for start_min, vol_frac in self.profile:
            if start_min <= minute_of_day < start_min + self.interval_minutes:
                return vol_frac
        return 0.0

    def get_volume_between(self, start_minute: int, end_minute: int) -> float:
        """Get cumulative volume fraction between two times of day."""
        if not self.profile:
            duration = end_minute - start_minute
            return duration / 1440.0
        total = 0.0
        for start_min, vol_frac in self.profile:
            interval_end = start_min + self.interval_minutes
            # Check overlap
            overlap_start = max(start_minute, start_min)
            overlap_end = min(end_minute, interval_end)
            if overlap_start < overlap_end:
                overlap_frac = (overlap_end - overlap_start) / self.interval_minutes
                total += vol_frac * overlap_frac
        return total

    def get_peak_hours(self, top_n: int = 3) -> List[Tuple[int, float]]:
        """Get the top N highest volume intervals."""
        sorted_profile = sorted(self.profile, key=lambda x: x[1], reverse=True)
        return sorted_profile[:top_n]

    @classmethod
    def from_hourly_volumes(cls, symbol: str, hourly_volumes: List[float],
                            total_daily: float = 0.0) -> "VolumeProfile":
        """Build a profile from 24 hourly volume values."""
        if not hourly_volumes or len(hourly_volumes) != 24:
            return cls(symbol=symbol)
        total = sum(hourly_volumes)
        if total == 0:
            return cls(symbol=symbol)
        profile = []
        for hour, vol in enumerate(hourly_volumes):
            start_min = hour * 60
            frac = vol / total
            # Split into 30-min intervals
            profile.append((start_min, frac / 2))
            profile.append((start_min + 30, frac / 2))
        return cls(
            symbol=symbol,
            interval_minutes=30,
            profile=profile,
            total_daily_volume=total_daily or total,
            last_updated=time.time(),
            data_days=1,
        )

    @classmethod
    def from_kline_data(cls, symbol: str, klines: List[Dict[str, Any]],
                        interval_minutes: int = 30) -> "VolumeProfile":
        """Build a volume profile from kline/candlestick data.
        Each kline should have 'open_time' (ms) and 'volume' fields.
        """
        if not klines:
            return cls(symbol=symbol, interval_minutes=interval_minutes)

        # Aggregate volume by time-of-day interval
        buckets: Dict[int, List[float]] = {}
        for kline in klines:
            open_time_ms = kline.get("open_time", 0)
            volume = float(kline.get("volume", 0))
            # Convert to minute of day (UTC)
            total_seconds = (open_time_ms / 1000) % 86400
            minute_of_day = int(total_seconds / 60)
            bucket = (minute_of_day // interval_minutes) * interval_minutes
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(volume)

        # Average across days
        avg_volumes = {}
        for bucket, volumes in buckets.items():
            avg_volumes[bucket] = sum(volumes) / len(volumes)

        total = sum(avg_volumes.values())
        if total == 0:
            return cls(symbol=symbol, interval_minutes=interval_minutes)

        profile = [(bucket, vol / total) for bucket, vol in sorted(avg_volumes.items())]
        num_days = max(1, len(klines) // (1440 // interval_minutes))

        return cls(
            symbol=symbol,
            interval_minutes=interval_minutes,
            profile=profile,
            total_daily_volume=total,
            last_updated=time.time(),
            data_days=num_days,
        )


@dataclass
class VWAPSlice:
    """A single slice in VWAP execution."""
    slice_id: str
    index: int
    scheduled_time: float
    actual_time: float = 0.0
    target_quantity: float = 0.0
    filled_quantity: float = 0.0
    fill_price: float = 0.0
    order_id: Optional[str] = None
    status: str = "pending"
    error_message: str = ""
    latency_ms: float = 0.0
    slippage_bps: float = 0.0
    market_price: float = 0.0
    market_volume: float = 0.0
    volume_fraction: float = 0.0  # What fraction of total this slice represents
    adjusted: bool = False        # Whether quantity was dynamically adjusted

    @property
    def notional_value(self) -> float:
        return self.filled_quantity * self.fill_price


@dataclass
class VWAPProgress:
    """Real-time progress of VWAP execution."""
    execution_id: str
    config: "VWAPConfig"
    status: VWAPStatus
    slices: List[VWAPSlice]
    start_time: float = 0.0
    end_time: float = 0.0
    total_filled: float = 0.0
    total_notional: float = 0.0
    actual_vwap: float = 0.0
    theoretical_vwap: float = 0.0
    vwap_deviation_bps: float = 0.0
    slices_completed: int = 0
    slices_failed: int = 0
    slices_remaining: int = 0
    fill_pct: float = 0.0
    current_market_price: float = 0.0
    elapsed_seconds: float = 0.0
    remaining_seconds: float = 0.0
    cumulative_slippage_bps: float = 0.0

    def update(self) -> None:
        """Recalculate metrics from slice data."""
        self.total_filled = sum(s.filled_quantity for s in self.slices)
        self.total_notional = sum(s.notional_value for s in self.slices)
        if self.total_filled > 0:
            self.actual_vwap = self.total_notional / self.total_filled
        self.slices_completed = sum(
            1 for s in self.slices if s.status in ("filled", "partial")
        )
        self.slices_failed = sum(1 for s in self.slices if s.status == "failed")
        self.slices_remaining = sum(
            1 for s in self.slices if s.status == "pending"
        )
        if self.config.total_quantity > 0:
            self.fill_pct = self.total_filled / self.config.total_quantity
        if self.start_time > 0:
            self.elapsed_seconds = time.time() - self.start_time
            self.remaining_seconds = max(
                0, self.config.duration_seconds - self.elapsed_seconds
            )
        # VWAP benchmark deviation
        if self.theoretical_vwap > 0 and self.actual_vwap > 0:
            self.vwap_deviation_bps = (
                (self.actual_vwap - self.theoretical_vwap) / self.theoretical_vwap * 10000.0
            )
        # Cumulative slippage
        slippage_values = [s.slippage_bps for s in self.slices if s.status == "filled"]
        if slippage_values:
            self.cumulative_slippage_bps = sum(slippage_values) / len(slippage_values)


@dataclass
class VWAPConfig:
    """Configuration for VWAP execution."""
    symbol: str
    side: str                        # BUY or SELL
    total_quantity: float
    duration_seconds: float
    num_slices: int = 0              # 0 = auto from volume profile
    volume_profile: Optional[VolumeProfile] = None
    mode: VWAPMode = VWAPMode.NEUTRAL
    front_load_factor: float = 1.0   # >1 front-loads, <1 back-loads
    price_limit: float = 0.0
    volume_participation_limit: float = 0.05  # Max 5% of market volume per interval
    min_slice_quantity: float = 0.0
    order_type: str = "LIMIT"
    limit_offset_bps: float = 3.0    # Passive offset from mid
    aggressive_offset_bps: float = -2.0  # Cross spread when aggressive
    time_in_force: str = "IOC"
    catch_up_enabled: bool = True     # Adjust future slices if behind
    max_catch_up_factor: float = 2.0  # Max multiplier for catch-up slices
    venue_id: Optional[str] = None

    def __post_init__(self):
        if self.num_slices == 0:
            self.num_slices = max(2, int(self.duration_seconds / 30))
        if self.front_load_factor <= 0:
            self.front_load_factor = 1.0

    @property
    def slice_interval(self) -> float:
        return self.duration_seconds / self.num_slices


class VWAPExecutor:
    """VWAP execution engine.

    Uses historical volume profiles to size slices proportionally
    to expected market volume, minimizing market impact.
    """

    def __init__(self):
        self._active_executions: Dict[str, VWAPProgress] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._pause_flags: Dict[str, bool] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._volume_profiles: Dict[str, VolumeProfile] = {}
        self._order_executor: Optional[Callable] = None
        self._price_fetcher: Optional[Callable] = None
        self._volume_fetcher: Optional[Callable] = None
        self._theoretical_vwap_prices: Dict[str, deque] = {}
        self._lock = asyncio.Lock()
        logger.info("VWAPExecutor initialized")

    def set_order_executor(self, executor: Callable) -> None:
        """Set order execution function."""
        self._order_executor = executor

    def set_price_fetcher(self, fetcher: Callable) -> None:
        """Set price fetcher function."""
        self._price_fetcher = fetcher

    def set_volume_fetcher(self, fetcher: Callable) -> None:
        """Set volume fetcher function."""
        self._volume_fetcher = fetcher

    def set_volume_profile(self, symbol: str, profile: VolumeProfile) -> None:
        """Set volume profile for a symbol."""
        self._volume_profiles[symbol] = profile
        logger.info(
            "Volume profile set for %s: %d intervals, %.0f daily volume",
            symbol, len(profile.profile), profile.total_daily_volume,
        )

    def get_volume_profile(self, symbol: str) -> Optional[VolumeProfile]:
        """Get stored volume profile for a symbol."""
        return self._volume_profiles.get(symbol)

    async def execute(self, config: VWAPConfig) -> VWAPProgress:
        """Start a VWAP execution."""
        execution_id = f"vwap_{uuid.uuid4().hex[:10]}"

        # Use provided profile or stored one
        profile = config.volume_profile or self._volume_profiles.get(config.symbol)
        if not profile:
            # Build flat profile as fallback
            profile = VolumeProfile(symbol=config.symbol)
            logger.warning(
                "No volume profile for %s - using flat distribution", config.symbol
            )

        slices = self._build_slices(config, profile, execution_id)

        progress = VWAPProgress(
            execution_id=execution_id,
            config=config,
            status=VWAPStatus.PENDING,
            slices=slices,
            slices_remaining=len(slices),
        )

        async with self._lock:
            self._active_executions[execution_id] = progress
            self._cancel_flags[execution_id] = False
            self._pause_flags[execution_id] = False
            self._theoretical_vwap_prices[execution_id] = deque(maxlen=10000)

        asyncio.create_task(self._run_execution(execution_id))

        logger.info(
            "VWAP started: id=%s symbol=%s side=%s qty=%.6f slices=%d duration=%ds mode=%s",
            execution_id, config.symbol, config.side, config.total_quantity,
            config.num_slices, int(config.duration_seconds), config.mode.value,
        )
        return progress

    def _build_slices(self, config: VWAPConfig, profile: VolumeProfile,
                      execution_id: str) -> List[VWAPSlice]:
        """Build volume-weighted slices."""
        slices = []
        now = time.time()
        interval = config.slice_interval

        # Get current time-of-day in minutes
        current_utc_seconds = now % 86400
        start_minute = int(current_utc_seconds / 60)

        # Calculate volume fractions for each slice interval
        volume_fractions = []
        for i in range(config.num_slices):
            slice_start_minute = (start_minute + int(i * interval / 60)) % 1440
            slice_end_minute = (slice_start_minute + int(interval / 60)) % 1440
            vol_frac = profile.get_volume_between(slice_start_minute, slice_end_minute)
            volume_fractions.append(vol_frac)

        # Normalize fractions to sum to 1.0
        total_frac = sum(volume_fractions)
        if total_frac > 0:
            volume_fractions = [f / total_frac for f in volume_fractions]
        else:
            # Flat distribution fallback
            volume_fractions = [1.0 / config.num_slices] * config.num_slices

        # Apply front/back loading factor
        if config.front_load_factor != 1.0:
            adjusted = []
            for i, frac in enumerate(volume_fractions):
                position = i / max(1, config.num_slices - 1)  # 0 to 1
                if config.front_load_factor > 1.0:
                    # Front-load: boost early slices
                    weight = 1.0 + (config.front_load_factor - 1.0) * (1.0 - position)
                else:
                    # Back-load: boost later slices
                    weight = 1.0 + (1.0 - config.front_load_factor) * position
                adjusted.append(frac * weight)
            total_adj = sum(adjusted)
            if total_adj > 0:
                volume_fractions = [a / total_adj for a in adjusted]

        # Build slice objects
        for i in range(config.num_slices):
            scheduled = now + (i * interval)
            qty = config.total_quantity * volume_fractions[i]
            if config.min_slice_quantity > 0:
                qty = max(qty, config.min_slice_quantity)

            slice_obj = VWAPSlice(
                slice_id=f"{execution_id}_s{i:04d}",
                index=i,
                scheduled_time=scheduled,
                target_quantity=qty,
                volume_fraction=volume_fractions[i],
            )
            slices.append(slice_obj)

        # Ensure total quantity is exact
        total_allocated = sum(s.target_quantity for s in slices)
        if total_allocated > 0:
            remainder = config.total_quantity - total_allocated
            if abs(remainder) > 1e-10:
                slices[-1].target_quantity += remainder

        return slices

    async def _run_execution(self, execution_id: str) -> None:
        """Main VWAP execution loop."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return

        progress.status = VWAPStatus.RUNNING
        progress.start_time = time.time()
        config = progress.config
        unfilled_carryover = 0.0

        try:
            for i, slice_obj in enumerate(progress.slices):
                if self._cancel_flags.get(execution_id, False):
                    slice_obj.status = "skipped"
                    progress.status = VWAPStatus.CANCELLED
                    break

                while self._pause_flags.get(execution_id, False):
                    await asyncio.sleep(0.5)
                    if self._cancel_flags.get(execution_id, False):
                        break

                # Wait until scheduled time
                wait_time = slice_obj.scheduled_time - time.time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Get current market data
                current_price = 0.0
                if self._price_fetcher:
                    current_price = await self._price_fetcher(config.symbol)
                    progress.current_market_price = current_price
                    slice_obj.market_price = current_price
                    # Track theoretical VWAP
                    self._theoretical_vwap_prices[execution_id].append(current_price)

                # Check price limit
                if config.price_limit > 0 and current_price > 0:
                    if config.side == "BUY" and current_price > config.price_limit:
                        slice_obj.status = "skipped"
                        slice_obj.error_message = "Price above limit"
                        unfilled_carryover += slice_obj.target_quantity
                        progress.update()
                        continue
                    elif config.side == "SELL" and current_price < config.price_limit:
                        slice_obj.status = "skipped"
                        slice_obj.error_message = "Price below limit"
                        unfilled_carryover += slice_obj.target_quantity
                        progress.update()
                        continue

                # Dynamic volume adjustment
                actual_qty = slice_obj.target_quantity
                if config.volume_participation_limit > 0 and self._volume_fetcher:
                    market_volume = await self._volume_fetcher(
                        config.symbol, config.slice_interval
                    )
                    slice_obj.market_volume = market_volume
                    max_qty = market_volume * config.volume_participation_limit
                    if max_qty > 0 and actual_qty > max_qty:
                        unfilled_carryover += (actual_qty - max_qty)
                        actual_qty = max_qty
                        slice_obj.adjusted = True

                # Catch-up logic
                if config.catch_up_enabled and unfilled_carryover > 0:
                    remaining_slices = config.num_slices - i
                    if remaining_slices > 0:
                        catch_up_per_slice = unfilled_carryover / remaining_slices
                        max_extra = actual_qty * (config.max_catch_up_factor - 1.0)
                        add_qty = min(catch_up_per_slice, max_extra)
                        actual_qty += add_qty
                        unfilled_carryover -= add_qty
                        if add_qty > 0:
                            slice_obj.adjusted = True

                # Determine execution price based on mode
                execution_price = self._determine_price(
                    config, current_price, progress
                )

                # Execute slice
                await self._execute_slice(
                    execution_id, slice_obj, actual_qty, execution_price, config
                )

                # Update theoretical VWAP
                prices = self._theoretical_vwap_prices.get(execution_id, deque())
                if prices:
                    progress.theoretical_vwap = sum(prices) / len(prices)

                progress.update()
                self._notify(execution_id, "slice_complete", slice_obj)

            # Finalize
            for s in progress.slices:
                if s.status == "pending":
                    s.status = "skipped"

            if progress.status == VWAPStatus.RUNNING:
                progress.status = VWAPStatus.COMPLETED

            progress.end_time = time.time()
            progress.update()
            self._notify(execution_id, "execution_complete", None)

            logger.info(
                "VWAP %s finished: status=%s filled=%.6f/%.6f vwap=%.6f theoretical=%.6f dev=%.2fbps",
                execution_id, progress.status.value,
                progress.total_filled, config.total_quantity,
                progress.actual_vwap, progress.theoretical_vwap,
                progress.vwap_deviation_bps,
            )

        except Exception as e:
            progress.status = VWAPStatus.FAILED
            progress.end_time = time.time()
            logger.error("VWAP %s failed: %s", execution_id, str(e))
            self._notify(execution_id, "execution_error", str(e))

    def _determine_price(self, config: VWAPConfig, current_price: float,
                         progress: VWAPProgress) -> Optional[float]:
        """Determine the limit price based on mode and progress."""
        if config.order_type != "LIMIT" or current_price <= 0:
            return None

        if config.mode == VWAPMode.PASSIVE:
            offset = current_price * config.limit_offset_bps / 10000.0
            if config.side == "BUY":
                return current_price - offset  # Bid below market
            else:
                return current_price + offset  # Ask above market

        elif config.mode == VWAPMode.AGGRESSIVE:
            # Check if behind schedule
            elapsed_frac = progress.elapsed_seconds / config.duration_seconds if config.duration_seconds > 0 else 0
            behind = progress.fill_pct < elapsed_frac * 0.8  # 80% threshold
            if behind:
                offset = current_price * abs(config.aggressive_offset_bps) / 10000.0
                if config.side == "BUY":
                    return current_price + offset  # Cross spread
                else:
                    return current_price - offset
            else:
                offset = current_price * config.limit_offset_bps / 10000.0
                if config.side == "BUY":
                    return current_price + offset
                else:
                    return current_price - offset

        else:  # NEUTRAL
            offset = current_price * config.limit_offset_bps / 10000.0
            if config.side == "BUY":
                return current_price + offset
            else:
                return current_price - offset

    async def _execute_slice(self, execution_id: str, slice_obj: VWAPSlice,
                             quantity: float, price: Optional[float],
                             config: VWAPConfig) -> None:
        """Execute a single VWAP slice."""
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
                slice_obj.status = "filled" if filled_qty >= quantity * 0.99 else "partial"

                if price and fill_price > 0:
                    if config.side == "BUY":
                        slice_obj.slippage_bps = (fill_price - price) / price * 10000.0
                    else:
                        slice_obj.slippage_bps = (price - fill_price) / price * 10000.0
            else:
                slice_obj.status = "failed"
                slice_obj.error_message = result.get("error", "Not filled") if result else "No result"

        except Exception as e:
            slice_obj.status = "failed"
            slice_obj.error_message = str(e)
            logger.error("VWAP %s slice %d error: %s", execution_id, slice_obj.index, str(e))

    # -----------------------------------------------------------------------
    # Volume profile prediction
    # -----------------------------------------------------------------------

    def predict_intraday_volume(self, symbol: str, from_minute: int,
                                to_minute: int) -> float:
        """Predict the volume that will occur between two times of day.
        Returns estimated volume in base currency.
        """
        profile = self._volume_profiles.get(symbol)
        if not profile:
            return 0.0
        frac = profile.get_volume_between(from_minute, to_minute)
        return frac * profile.total_daily_volume

    def get_optimal_slice_count(self, symbol: str, total_quantity: float,
                                duration_seconds: float,
                                max_participation: float = 0.05) -> int:
        """Calculate optimal number of slices based on volume profile."""
        profile = self._volume_profiles.get(symbol)
        if not profile or profile.total_daily_volume == 0:
            return max(2, int(duration_seconds / 30))

        duration_minutes = duration_seconds / 60
        start_minute = int((time.time() % 86400) / 60)
        end_minute = int(start_minute + duration_minutes) % 1440

        expected_volume = profile.get_volume_between(start_minute, end_minute) * profile.total_daily_volume
        if expected_volume <= 0:
            return max(2, int(duration_seconds / 30))

        # Each slice should be at most max_participation of interval volume
        min_slices = math.ceil(total_quantity / (expected_volume * max_participation))
        # But at least one slice per 30 seconds
        time_based = max(2, int(duration_seconds / 30))

        return max(min_slices, time_based)

    # -----------------------------------------------------------------------
    # Benchmark analysis
    # -----------------------------------------------------------------------

    def get_benchmark_comparison(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Compare actual execution against theoretical VWAP benchmark."""
        progress = self._active_executions.get(execution_id)
        if not progress:
            return None

        progress.update()
        prices = self._theoretical_vwap_prices.get(execution_id, deque())
        theoretical = sum(prices) / len(prices) if prices else 0.0

        return {
            "execution_id": execution_id,
            "actual_vwap": progress.actual_vwap,
            "theoretical_vwap": theoretical,
            "deviation_bps": progress.vwap_deviation_bps,
            "fill_pct": progress.fill_pct,
            "avg_slippage_bps": progress.cumulative_slippage_bps,
            "mode": progress.config.mode.value,
            "slices_adjusted": sum(1 for s in progress.slices if s.adjusted),
            "side": progress.config.side,
            "outperformed": (
                (progress.config.side == "BUY" and progress.actual_vwap < theoretical)
                or (progress.config.side == "SELL" and progress.actual_vwap > theoretical)
            ) if theoretical > 0 else False,
        }

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running VWAP execution."""
        if execution_id not in self._active_executions:
            return False
        self._cancel_flags[execution_id] = True
        return True

    async def pause(self, execution_id: str) -> bool:
        """Pause execution."""
        if execution_id not in self._active_executions:
            return False
        self._pause_flags[execution_id] = True
        self._active_executions[execution_id].status = VWAPStatus.PAUSED
        return True

    async def resume(self, execution_id: str) -> bool:
        """Resume execution."""
        if execution_id not in self._active_executions:
            return False
        self._pause_flags[execution_id] = False
        self._active_executions[execution_id].status = VWAPStatus.RUNNING
        return True

    def get_progress(self, execution_id: str) -> Optional[VWAPProgress]:
        """Get current progress."""
        progress = self._active_executions.get(execution_id)
        if progress:
            progress.update()
        return progress

    def get_active_executions(self) -> Dict[str, VWAPProgress]:
        """Get all active executions."""
        return {
            eid: p for eid, p in self._active_executions.items()
            if p.status in (VWAPStatus.RUNNING, VWAPStatus.PAUSED, VWAPStatus.PENDING)
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
            "total_filled": progress.total_filled,
            "fill_pct": progress.fill_pct,
            "actual_vwap": progress.actual_vwap,
            "theoretical_vwap": progress.theoretical_vwap,
            "deviation_bps": progress.vwap_deviation_bps,
            "status": progress.status.value,
            "mode": progress.config.mode.value,
            "slices_completed": progress.slices_completed,
            "slices_remaining": progress.slices_remaining,
            "elapsed_seconds": progress.elapsed_seconds,
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
                logger.error("VWAP callback error: %s", str(e))

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove old completed executions."""
        now = time.time()
        to_remove = []
        for eid, p in self._active_executions.items():
            if p.status in (VWAPStatus.COMPLETED, VWAPStatus.CANCELLED, VWAPStatus.FAILED):
                if p.end_time > 0 and (now - p.end_time) > max_age_seconds:
                    to_remove.append(eid)
        for eid in to_remove:
            del self._active_executions[eid]
            self._cancel_flags.pop(eid, None)
            self._pause_flags.pop(eid, None)
            self._callbacks.pop(eid, None)
            self._theoretical_vwap_prices.pop(eid, None)
        return len(to_remove)
