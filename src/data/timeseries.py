"""
Time Series Utilities - Pure numpy time series operations for Mefai Autotrade.
Provides resampling, alignment, fill methods, rolling/exponential windows,
timezone handling, and calendar utilities - all without pandas dependency.
"""

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("mefai.autotrade.timeseries")


# ---------------------------------------------------------------------------
# Interval definitions (seconds)
# ---------------------------------------------------------------------------

INTERVAL_MAP = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400, "1w": 604800,
}


def parse_interval(interval: str) -> int:
    """Parse interval string to seconds."""
    key = interval.lower().strip()
    if key in INTERVAL_MAP:
        return INTERVAL_MAP[key]
    raise ValueError(f"Unknown interval: {interval}. Valid: {list(INTERVAL_MAP.keys())}")


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------

class TimeZoneHandler:
    """UTC-based timezone utilities for trading."""

    UTC = timezone.utc

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def now_timestamp() -> float:
        return time.time()

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert any datetime to UTC."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def timestamp_to_datetime(ts: float) -> datetime:
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> float:
        return dt.timestamp()

    @staticmethod
    def format_iso(ts: float) -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def parse_iso(s: str) -> float:
        """Parse ISO format string to Unix timestamp."""
        s = s.rstrip("Z")
        if "." in s:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()

    @staticmethod
    def align_to_interval(ts: float, interval_seconds: int) -> float:
        """Align a timestamp to the start of its interval bucket."""
        return float(int(ts // interval_seconds) * interval_seconds)

    @staticmethod
    def ms_to_seconds(ms: Union[int, float]) -> float:
        return float(ms) / 1000.0

    @staticmethod
    def seconds_to_ms(s: Union[int, float]) -> int:
        return int(s * 1000)


# ---------------------------------------------------------------------------
# Calendar utilities
# ---------------------------------------------------------------------------

class CalendarUtils:
    """Calendar and session utilities for trading."""

    @staticmethod
    def is_weekend(ts: float) -> bool:
        """Check if timestamp falls on a weekend (Saturday or Sunday UTC)."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.weekday() >= 5

    @staticmethod
    def is_weekday(ts: float) -> bool:
        return not CalendarUtils.is_weekend(ts)

    @staticmethod
    def day_of_week(ts: float) -> int:
        """0=Monday, 6=Sunday."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.weekday()

    @staticmethod
    def day_of_week_name(ts: float) -> str:
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return names[CalendarUtils.day_of_week(ts)]

    @staticmethod
    def hour_of_day(ts: float) -> int:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.hour

    @staticmethod
    def start_of_day(ts: float) -> float:
        """Get the start of the UTC day containing this timestamp."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        sod = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return sod.timestamp()

    @staticmethod
    def end_of_day(ts: float) -> float:
        """Get the end of the UTC day containing this timestamp."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        eod = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return eod.timestamp()

    @staticmethod
    def start_of_week(ts: float) -> float:
        """Get start of the ISO week (Monday 00:00 UTC)."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        monday = dt - timedelta(days=dt.weekday())
        sow = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return sow.timestamp()

    @staticmethod
    def start_of_month(ts: float) -> float:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        som = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return som.timestamp()

    @staticmethod
    def is_session_open(ts: float, session_start_hour: int = 0, session_end_hour: int = 24) -> bool:
        """
        Check if timestamp is within a trading session.
        For crypto (24/7), always returns True by default.
        """
        if session_start_hour == 0 and session_end_hour == 24:
            return True
        hour = CalendarUtils.hour_of_day(ts)
        if session_start_hour <= session_end_hour:
            return session_start_hour <= hour < session_end_hour
        # Wrapping session (e.g., 22-06)
        return hour >= session_start_hour or hour < session_end_hour

    @staticmethod
    def is_low_liquidity_period(ts: float) -> bool:
        """
        Detect typical low-liquidity periods in crypto markets.
        Weekends and early UTC morning (00:00-06:00) tend to have lower liquidity.
        """
        hour = CalendarUtils.hour_of_day(ts)
        is_wknd = CalendarUtils.is_weekend(ts)
        return is_wknd or (0 <= hour < 6)

    @staticmethod
    def trading_days_between(start_ts: float, end_ts: float) -> int:
        """Count weekdays between two timestamps."""
        start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc).date()
        end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc).date()
        count = 0
        current = start_dt
        while current <= end_dt:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        return count

    @staticmethod
    def next_period_start(ts: float, period: str) -> float:
        """Get the start of the next period (day, week, month)."""
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if period == "day":
            next_day = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return next_day.timestamp()
        elif period == "week":
            days_to_monday = 7 - dt.weekday()
            next_monday = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_to_monday)
            return next_monday.timestamp()
        elif period == "month":
            if dt.month == 12:
                next_month = dt.replace(year=dt.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_month = dt.replace(month=dt.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            return next_month.timestamp()
        else:
            raise ValueError(f"Unknown period: {period}")


# ---------------------------------------------------------------------------
# Time series resampling
# ---------------------------------------------------------------------------

class TimeSeriesResampler:
    """
    Resample OHLCV time series from one interval to another.
    Pure numpy implementation.
    """

    @staticmethod
    def resample_ohlcv(
        timestamps: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        source_interval: int,
        target_interval: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample OHLCV data from source to target interval.
        Target must be a multiple of source.
        Returns (timestamps, opens, highs, lows, closes, volumes).
        """
        if target_interval < source_interval:
            raise ValueError("Cannot downsample to a smaller interval")
        if target_interval % source_interval != 0:
            raise ValueError(
                f"Target interval ({target_interval}s) must be exact "
                f"multiple of source ({source_interval}s)"
            )

        n = len(timestamps)
        if n == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, empty, empty

        # Compute bucket indices
        buckets = (timestamps // target_interval).astype(np.int64)
        unique_buckets = np.unique(buckets)
        m = len(unique_buckets)

        out_ts = np.zeros(m, dtype=np.float64)
        out_o = np.zeros(m, dtype=np.float64)
        out_h = np.zeros(m, dtype=np.float64)
        out_l = np.zeros(m, dtype=np.float64)
        out_c = np.zeros(m, dtype=np.float64)
        out_v = np.zeros(m, dtype=np.float64)

        for i, bucket in enumerate(unique_buckets):
            mask = buckets == bucket
            out_ts[i] = float(bucket * target_interval)
            out_o[i] = opens[mask][0]
            out_h[i] = np.max(highs[mask])
            out_l[i] = np.min(lows[mask])
            out_c[i] = closes[mask][-1]
            out_v[i] = np.sum(volumes[mask])

        return out_ts, out_o, out_h, out_l, out_c, out_v

    @staticmethod
    def resample_values(
        timestamps: np.ndarray,
        values: np.ndarray,
        target_interval: int,
        method: str = "last",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample a single value series.
        Methods: last, first, mean, sum, max, min, median.
        Returns (timestamps, values).
        """
        n = len(timestamps)
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        buckets = (timestamps // target_interval).astype(np.int64)
        unique_buckets = np.unique(buckets)
        m = len(unique_buckets)

        out_ts = np.zeros(m, dtype=np.float64)
        out_vals = np.zeros(m, dtype=np.float64)

        agg_fn = {
            "last": lambda x: x[-1],
            "first": lambda x: x[0],
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "min": np.min,
            "median": np.median,
        }.get(method, lambda x: x[-1])

        for i, bucket in enumerate(unique_buckets):
            mask = buckets == bucket
            out_ts[i] = float(bucket * target_interval)
            out_vals[i] = agg_fn(values[mask])

        return out_ts, out_vals


# ---------------------------------------------------------------------------
# Time series alignment
# ---------------------------------------------------------------------------

class TimeSeriesAligner:
    """Align multiple time series to a common timestamp axis."""

    @staticmethod
    def align_to_common(
        series_list: List[Tuple[np.ndarray, np.ndarray]],
        method: str = "inner",
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Align multiple (timestamps, values) pairs to common timestamps.
        Methods: inner (intersection), outer (union with NaN fill).
        Returns (common_timestamps, list_of_aligned_values).
        """
        if not series_list:
            return np.empty(0), []

        if len(series_list) == 1:
            return series_list[0][0].copy(), [series_list[0][1].copy()]

        if method == "inner":
            # Find common timestamps (intersection)
            common = set(series_list[0][0].astype(np.int64))
            for ts, _ in series_list[1:]:
                common &= set(ts.astype(np.int64))
            common_ts = np.array(sorted(common), dtype=np.float64)
        else:
            # Union of all timestamps
            all_ts = set()
            for ts, _ in series_list:
                all_ts.update(ts.astype(np.int64))
            common_ts = np.array(sorted(all_ts), dtype=np.float64)

        # Map values to common timestamps
        aligned_values: List[np.ndarray] = []
        for ts, vals in series_list:
            ts_set = {int(t): v for t, v in zip(ts, vals)}
            aligned = np.array(
                [ts_set.get(int(t), np.nan) for t in common_ts],
                dtype=np.float64,
            )
            aligned_values.append(aligned)

        return common_ts, aligned_values

    @staticmethod
    def align_to_reference(
        ref_timestamps: np.ndarray,
        timestamps: np.ndarray,
        values: np.ndarray,
        fill_method: str = "ffill",
    ) -> np.ndarray:
        """
        Align values to reference timestamps using specified fill method.
        Fill methods: ffill, bfill, nearest, nan.
        """
        n = len(ref_timestamps)
        result = np.full(n, np.nan)

        if len(timestamps) == 0:
            return result

        ts_idx = {int(t): i for i, t in enumerate(timestamps)}

        for i, ref_t in enumerate(ref_timestamps):
            key = int(ref_t)
            if key in ts_idx:
                result[i] = values[ts_idx[key]]

        if fill_method == "ffill":
            result = TimeSeriesAligner._forward_fill(result)
        elif fill_method == "bfill":
            result = TimeSeriesAligner._backward_fill(result)
        elif fill_method == "nearest":
            result = TimeSeriesAligner._nearest_fill(result, ref_timestamps, timestamps, values)

        return result

    @staticmethod
    def _forward_fill(arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        for i in range(1, len(result)):
            if np.isnan(result[i]):
                result[i] = result[i - 1]
        return result

    @staticmethod
    def _backward_fill(arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        for i in range(len(result) - 2, -1, -1):
            if np.isnan(result[i]):
                result[i] = result[i + 1]
        return result

    @staticmethod
    def _nearest_fill(
        arr: np.ndarray,
        ref_ts: np.ndarray,
        src_ts: np.ndarray,
        src_vals: np.ndarray,
    ) -> np.ndarray:
        result = arr.copy()
        for i in range(len(result)):
            if np.isnan(result[i]):
                diffs = np.abs(src_ts - ref_ts[i])
                nearest_idx = np.argmin(diffs)
                result[i] = src_vals[nearest_idx]
        return result


# ---------------------------------------------------------------------------
# Fill methods
# ---------------------------------------------------------------------------

def forward_fill(arr: np.ndarray) -> np.ndarray:
    """Forward fill NaN values in a 1D array."""
    result = arr.copy()
    for i in range(1, len(result)):
        if np.isnan(result[i]):
            result[i] = result[i - 1]
    return result


def backward_fill(arr: np.ndarray) -> np.ndarray:
    """Backward fill NaN values in a 1D array."""
    result = arr.copy()
    for i in range(len(result) - 2, -1, -1):
        if np.isnan(result[i]):
            result[i] = result[i + 1]
    return result


def interpolate_linear(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values."""
    result = arr.copy()
    valid = ~np.isnan(result)
    if not np.any(valid):
        return result

    indices = np.arange(len(result))
    valid_indices = indices[valid]
    valid_values = result[valid]

    # Interpolate
    interp = np.interp(indices, valid_indices, valid_values)

    # Only fill NaN positions between first and last valid
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    for i in range(first_valid, last_valid + 1):
        if np.isnan(result[i]):
            result[i] = interp[i]

    return result


def fill_with_value(arr: np.ndarray, value: float = 0.0) -> np.ndarray:
    """Fill NaN values with a constant."""
    result = arr.copy()
    result[np.isnan(result)] = value
    return result


# ---------------------------------------------------------------------------
# Rolling window calculations
# ---------------------------------------------------------------------------

class RollingWindow:
    """
    Efficient rolling window calculations using numpy.
    Avoids recomputation where possible.
    """

    @staticmethod
    def mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean (simple moving average)."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        cumsum = np.cumsum(arr)
        result[window - 1:] = (
            cumsum[window - 1:] - np.concatenate(([0.0], cumsum[:-window]))
        ) / window
        return result

    @staticmethod
    def std(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
        """Rolling standard deviation."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)

        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            result[i] = np.std(arr[i - window + 1: i + 1], ddof=ddof)
        return result

    @staticmethod
    def variance(arr: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
        """Rolling variance."""
        std_vals = RollingWindow.std(arr, window, ddof)
        return std_vals ** 2

    @staticmethod
    def max(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling maximum."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            result[i] = np.max(arr[i - window + 1: i + 1])
        return result

    @staticmethod
    def min(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling minimum."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            result[i] = np.min(arr[i - window + 1: i + 1])
        return result

    @staticmethod
    def sum(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling sum."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        cumsum = np.cumsum(arr)
        result[window - 1:] = cumsum[window - 1:] - np.concatenate(([0.0], cumsum[:-window]))
        return result

    @staticmethod
    def median(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling median."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            result[i] = np.median(arr[i - window + 1: i + 1])
        return result

    @staticmethod
    def zscore(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling z-score."""
        mean_vals = RollingWindow.mean(arr, window)
        std_vals = RollingWindow.std(arr, window)
        result = np.full(len(arr), np.nan)
        valid = ~np.isnan(mean_vals) & ~np.isnan(std_vals) & (std_vals > 0)
        result[valid] = (arr[valid] - mean_vals[valid]) / std_vals[valid]
        return result

    @staticmethod
    def percentile(arr: np.ndarray, window: int, pct: float) -> np.ndarray:
        """Rolling percentile."""
        n = len(arr)
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            result[i] = np.percentile(arr[i - window + 1: i + 1], pct)
        return result

    @staticmethod
    def correlation(arr1: np.ndarray, arr2: np.ndarray, window: int) -> np.ndarray:
        """Rolling Pearson correlation between two arrays."""
        n = min(len(arr1), len(arr2))
        if n < window:
            return np.full(n, np.nan)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            w1 = arr1[i - window + 1: i + 1]
            w2 = arr2[i - window + 1: i + 1]
            std1 = np.std(w1)
            std2 = np.std(w2)
            if std1 > 0 and std2 > 0:
                result[i] = np.corrcoef(w1, w2)[0, 1]
            else:
                result[i] = 0.0
        return result

    @staticmethod
    def returns(arr: np.ndarray, period: int = 1) -> np.ndarray:
        """Calculate returns over a given period."""
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(period, n):
            if arr[i - period] != 0:
                result[i] = (arr[i] - arr[i - period]) / arr[i - period]
        return result

    @staticmethod
    def log_returns(arr: np.ndarray, period: int = 1) -> np.ndarray:
        """Calculate log returns over a given period."""
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(period, n):
            if arr[i - period] > 0 and arr[i] > 0:
                result[i] = np.log(arr[i] / arr[i - period])
        return result

    @staticmethod
    def cumulative_returns(arr: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns from a price series."""
        if len(arr) == 0 or arr[0] == 0:
            return np.zeros(len(arr))
        return (arr / arr[0]) - 1.0

    @staticmethod
    def drawdown(arr: np.ndarray) -> np.ndarray:
        """Calculate drawdown series from equity/price curve."""
        n = len(arr)
        result = np.zeros(n)
        peak = arr[0] if n > 0 else 0
        for i in range(n):
            if arr[i] > peak:
                peak = arr[i]
            if peak > 0:
                result[i] = (arr[i] - peak) / peak
        return result

    @staticmethod
    def max_drawdown(arr: np.ndarray, window: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Maximum drawdown. If window is specified, returns rolling max drawdown.
        Otherwise returns single value.
        """
        dd = RollingWindow.drawdown(arr)

        if window is None:
            return float(np.min(dd)) if len(dd) > 0 else 0.0

        return RollingWindow.min(dd, window)


# ---------------------------------------------------------------------------
# Exponential weighted calculations
# ---------------------------------------------------------------------------

class ExponentialWeighted:
    """Exponential weighted calculations (EMA, EWMA variance, etc.)."""

    @staticmethod
    def mean(arr: np.ndarray, span: Optional[int] = None, alpha: Optional[float] = None) -> np.ndarray:
        """
        Exponential weighted moving average.
        Specify either span or alpha (alpha = 2 / (span + 1)).
        """
        if alpha is None:
            if span is None:
                raise ValueError("Must provide either span or alpha")
            alpha = 2.0 / (span + 1)

        n = len(arr)
        if n == 0:
            return np.empty(0)

        result = np.zeros(n)
        result[0] = arr[0]
        for i in range(1, n):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def std(arr: np.ndarray, span: Optional[int] = None, alpha: Optional[float] = None) -> np.ndarray:
        """Exponential weighted standard deviation."""
        if alpha is None:
            if span is None:
                raise ValueError("Must provide either span or alpha")
            alpha = 2.0 / (span + 1)

        n = len(arr)
        if n == 0:
            return np.empty(0)

        ewma = ExponentialWeighted.mean(arr, alpha=alpha)
        result = np.zeros(n)

        for i in range(1, n):
            diff = arr[i] - ewma[i - 1]
            result[i] = math.sqrt(
                (1 - alpha) * (result[i - 1] ** 2 + alpha * diff ** 2)
            )

        return result

    @staticmethod
    def variance(arr: np.ndarray, span: Optional[int] = None, alpha: Optional[float] = None) -> np.ndarray:
        """Exponential weighted variance."""
        std_vals = ExponentialWeighted.std(arr, span=span, alpha=alpha)
        return std_vals ** 2

    @staticmethod
    def covariance(
        arr1: np.ndarray, arr2: np.ndarray,
        span: Optional[int] = None, alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Exponential weighted covariance."""
        if alpha is None:
            if span is None:
                raise ValueError("Must provide either span or alpha")
            alpha = 2.0 / (span + 1)

        n = min(len(arr1), len(arr2))
        if n == 0:
            return np.empty(0)

        ewma1 = ExponentialWeighted.mean(arr1[:n], alpha=alpha)
        ewma2 = ExponentialWeighted.mean(arr2[:n], alpha=alpha)

        result = np.zeros(n)
        for i in range(1, n):
            diff1 = arr1[i] - ewma1[i - 1]
            diff2 = arr2[i] - ewma2[i - 1]
            result[i] = (1 - alpha) * (result[i - 1] + alpha * diff1 * diff2)

        return result

    @staticmethod
    def correlation(
        arr1: np.ndarray, arr2: np.ndarray,
        span: Optional[int] = None, alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Exponential weighted correlation."""
        cov = ExponentialWeighted.covariance(arr1, arr2, span=span, alpha=alpha)
        std1 = ExponentialWeighted.std(arr1, span=span, alpha=alpha)
        std2 = ExponentialWeighted.std(arr2, span=span, alpha=alpha)

        n = len(cov)
        result = np.zeros(n)
        for i in range(n):
            denom = std1[i] * std2[i]
            if denom > 1e-10:
                result[i] = cov[i] / denom
        return result

    @staticmethod
    def half_life_alpha(half_life: float) -> float:
        """Convert half-life to alpha for EWM calculations."""
        return 1 - math.exp(-math.log(2) / half_life)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def diff(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """First difference of an array."""
    result = np.full(len(arr), np.nan)
    result[periods:] = arr[periods:] - arr[:-periods]
    return result


def pct_change(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Percentage change."""
    result = np.full(len(arr), np.nan)
    for i in range(periods, len(arr)):
        if arr[i - periods] != 0:
            result[i] = (arr[i] - arr[i - periods]) / arr[i - periods]
    return result


def shift(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """Shift array by N periods (positive = forward, negative = backward)."""
    result = np.full(len(arr), np.nan)
    if periods > 0:
        result[periods:] = arr[:-periods]
    elif periods < 0:
        result[:periods] = arr[-periods:]
    else:
        result[:] = arr
    return result


def clip(arr: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Clip values to [lower, upper] range."""
    return np.clip(arr, lower, upper)


def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] range."""
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros(len(arr))
    return (arr - arr_min) / (arr_max - arr_min)


def standardize(arr: np.ndarray) -> np.ndarray:
    """Z-score standardization (mean=0, std=1)."""
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0:
        return np.zeros(len(arr))
    return (arr - mean) / std
