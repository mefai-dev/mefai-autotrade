"""
Indicator Cache - Cached indicator calculations for Mefai Autotrade.
Provides lazy computation, incremental updates, cache invalidation,
thread-safe access, memory monitoring, and indicator dependency tracking.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("mefai.autotrade.indicator_cache")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IndicatorConfig:
    """Configuration for an indicator computation."""
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    incremental: bool = True
    cache_ttl_seconds: float = 0.0  # 0 = no expiry
    min_data_points: int = 1

    @property
    def cache_key(self) -> str:
        """Unique key combining name and params."""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}:{param_str}" if param_str else self.name


# ---------------------------------------------------------------------------
# Cached indicator result
# ---------------------------------------------------------------------------

@dataclass
class CachedIndicator:
    """Stores a cached indicator result with metadata."""
    name: str = ""
    cache_key: str = ""
    values: Optional[np.ndarray] = None
    last_input_length: int = 0
    last_compute_time: float = 0.0
    compute_duration_ms: float = 0.0
    hit_count: int = 0
    created_at: float = 0.0
    ttl: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True

    @property
    def is_expired(self) -> bool:
        if self.ttl <= 0:
            return False
        return (time.time() - self.last_compute_time) > self.ttl

    @property
    def latest_value(self) -> Optional[float]:
        if self.values is not None and len(self.values) > 0:
            return float(self.values[-1])
        return None

    def get_last_n(self, n: int) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        return self.values[-n:] if n <= len(self.values) else self.values


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

class IndicatorDependencyGraph:
    """
    Tracks dependencies between indicators.
    Used for proper cache invalidation and computation ordering.
    """

    def __init__(self):
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()

    def add_dependency(self, indicator: str, depends_on: str) -> None:
        """Register that 'indicator' depends on 'depends_on'."""
        with self._lock:
            self._dependencies[indicator].add(depends_on)
            self._dependents[depends_on].add(indicator)

    def add_dependencies(self, indicator: str, depends_on: List[str]) -> None:
        for dep in depends_on:
            self.add_dependency(indicator, dep)

    def get_dependencies(self, indicator: str) -> Set[str]:
        """Get direct dependencies of an indicator."""
        with self._lock:
            return set(self._dependencies.get(indicator, set()))

    def get_all_dependencies(self, indicator: str) -> Set[str]:
        """Get all transitive dependencies (recursive)."""
        visited: Set[str] = set()
        stack = [indicator]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            with self._lock:
                deps = self._dependencies.get(current, set())
            stack.extend(deps - visited)

        visited.discard(indicator)
        return visited

    def get_dependents(self, indicator: str) -> Set[str]:
        """Get indicators that depend on this one."""
        with self._lock:
            return set(self._dependents.get(indicator, set()))

    def get_all_dependents(self, indicator: str) -> Set[str]:
        """Get all transitive dependents (recursive)."""
        visited: Set[str] = set()
        stack = [indicator]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            with self._lock:
                deps = self._dependents.get(current, set())
            stack.extend(deps - visited)

        visited.discard(indicator)
        return visited

    def get_computation_order(self) -> List[str]:
        """
        Topological sort - returns indicators in the order they should be computed.
        Indicators with no dependencies come first.
        """
        with self._lock:
            all_indicators = set(self._dependencies.keys()) | set(self._dependents.keys())
            in_degree: Dict[str, int] = {ind: 0 for ind in all_indicators}
            for ind, deps in self._dependencies.items():
                in_degree[ind] = len(deps)

        result: List[str] = []
        queue = [ind for ind, deg in in_degree.items() if deg == 0]

        while queue:
            current = queue.pop(0)
            result.append(current)
            with self._lock:
                for dependent in self._dependents.get(current, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(in_degree):
            missing = set(in_degree.keys()) - set(result)
            logger.warning("Circular dependency detected involving: %s", missing)
            result.extend(missing)

        return result

    def has_circular_dependency(self) -> bool:
        """Check if the dependency graph has any cycles."""
        order = self.get_computation_order()
        with self._lock:
            all_indicators = set(self._dependencies.keys()) | set(self._dependents.keys())
        return len(order) != len(all_indicators)

    def clear(self) -> None:
        with self._lock:
            self._dependencies.clear()
            self._dependents.clear()

    def to_dict(self) -> Dict[str, List[str]]:
        with self._lock:
            return {k: sorted(v) for k, v in self._dependencies.items()}


# ---------------------------------------------------------------------------
# Built-in indicator computations
# ---------------------------------------------------------------------------

class IndicatorFunctions:
    """
    Pure numpy indicator computations.
    All functions take numpy arrays and return numpy arrays.
    """

    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        cumsum = np.cumsum(data)
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))) / period
        return result

    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        if len(data) == 0:
            return np.empty(0)
        alpha = 2.0 / (period + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative strength index."""
        if len(data) < period + 1:
            return np.full(len(data), np.nan)

        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        result = np.full(len(data), np.nan)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

        return result

    @staticmethod
    def macd(
        data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD line, signal line, histogram."""
        fast_ema = IndicatorFunctions.ema(data, fast)
        slow_ema = IndicatorFunctions.ema(data, slow)
        macd_line = fast_ema - slow_ema
        signal_line = IndicatorFunctions.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        data: np.ndarray, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Upper band, middle band (SMA), lower band."""
        middle = IndicatorFunctions.sma(data, period)
        result_upper = np.full(len(data), np.nan)
        result_lower = np.full(len(data), np.nan)

        for i in range(period - 1, len(data)):
            window = data[i - period + 1: i + 1]
            std = np.std(window, ddof=0)
            result_upper[i] = middle[i] + std_dev * std
            result_lower[i] = middle[i] - std_dev * std

        return result_upper, middle, result_lower

    @staticmethod
    def atr(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average true range."""
        if len(highs) < 2:
            return np.full(len(highs), np.nan)

        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        result = np.full(len(highs), np.nan)
        if len(tr) >= period:
            result[period - 1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

        return result

    @staticmethod
    def stochastic(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        k_period: int = 14, d_period: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic %K and %D."""
        n = len(closes)
        k = np.full(n, np.nan)

        for i in range(k_period - 1, n):
            window_high = np.max(highs[i - k_period + 1: i + 1])
            window_low = np.min(lows[i - k_period + 1: i + 1])
            denom = window_high - window_low
            if denom > 0:
                k[i] = ((closes[i] - window_low) / denom) * 100.0
            else:
                k[i] = 50.0

        d = IndicatorFunctions.sma(k, d_period)
        return k, d

    @staticmethod
    def vwap(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """Volume-weighted average price (cumulative)."""
        typical = (highs + lows + closes) / 3.0
        cumulative_tp_vol = np.cumsum(typical * volumes)
        cumulative_vol = np.cumsum(volumes)
        result = np.where(cumulative_vol > 0, cumulative_tp_vol / cumulative_vol, 0.0)
        return result

    @staticmethod
    def adx(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average directional index."""
        n = len(highs)
        if n < period + 1:
            return np.full(n, np.nan)

        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

        # Smooth using Wilder's method
        atr_val = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)

        atr_val[period] = np.sum(tr[1: period + 1])
        smooth_plus = np.sum(plus_dm[1: period + 1])
        smooth_minus = np.sum(minus_dm[1: period + 1])

        for i in range(period + 1, n):
            atr_val[i] = atr_val[i - 1] - (atr_val[i - 1] / period) + tr[i]
            smooth_plus = smooth_plus - (smooth_plus / period) + plus_dm[i]
            smooth_minus = smooth_minus - (smooth_minus / period) + minus_dm[i]

            if atr_val[i] > 0:
                plus_di[i] = (smooth_plus / atr_val[i]) * 100
                minus_di[i] = (smooth_minus / atr_val[i]) * 100

        dx = np.zeros(n)
        for i in range(period, n):
            denom = plus_di[i] + minus_di[i]
            if denom > 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / denom * 100

        adx_result = np.full(n, np.nan)
        if n > 2 * period:
            adx_result[2 * period] = np.mean(dx[period + 1: 2 * period + 1])
            for i in range(2 * period + 1, n):
                adx_result[i] = (adx_result[i - 1] * (period - 1) + dx[i]) / period

        return adx_result

    @staticmethod
    def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """On-balance volume."""
        if len(closes) == 0:
            return np.empty(0)
        result = np.zeros(len(closes))
        result[0] = volumes[0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                result[i] = result[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                result[i] = result[i - 1] - volumes[i]
            else:
                result[i] = result[i - 1]
        return result

    @staticmethod
    def supertrend(
        highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 10, multiplier: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Supertrend indicator. Returns (supertrend_values, direction)."""
        atr_vals = IndicatorFunctions.atr(highs, lows, closes, period)
        n = len(closes)
        supertrend = np.zeros(n)
        direction = np.ones(n)  # 1 = uptrend, -1 = downtrend

        upper_band = np.zeros(n)
        lower_band = np.zeros(n)

        for i in range(n):
            mid = (highs[i] + lows[i]) / 2.0
            if np.isnan(atr_vals[i]):
                upper_band[i] = mid
                lower_band[i] = mid
            else:
                upper_band[i] = mid + multiplier * atr_vals[i]
                lower_band[i] = mid - multiplier * atr_vals[i]

        for i in range(1, n):
            if lower_band[i] > lower_band[i - 1] or closes[i - 1] < lower_band[i - 1]:
                pass
            else:
                lower_band[i] = lower_band[i - 1]

            if upper_band[i] < upper_band[i - 1] or closes[i - 1] > upper_band[i - 1]:
                pass
            else:
                upper_band[i] = upper_band[i - 1]

            if direction[i - 1] == 1:
                if closes[i] < lower_band[i]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                if closes[i] > upper_band[i]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]

        return supertrend, direction


# ---------------------------------------------------------------------------
# Main indicator cache
# ---------------------------------------------------------------------------

class IndicatorCache:
    """
    Thread-safe indicator cache with lazy computation,
    incremental updates, and dependency tracking.
    """

    def __init__(self, max_memory_mb: float = 256.0):
        self._cache: Dict[str, CachedIndicator] = {}
        self._configs: Dict[str, IndicatorConfig] = {}
        self._compute_fns: Dict[str, Callable] = {}
        self._dep_graph = IndicatorDependencyGraph()
        self._lock = threading.RLock()
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._total_computes = 0
        self._total_hits = 0

        # Register built-in indicators
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in indicator computation functions."""
        self.register_indicator(
            "sma", lambda data, **kw: IndicatorFunctions.sma(data, kw.get("period", 20)),
            dependencies=[],
        )
        self.register_indicator(
            "ema", lambda data, **kw: IndicatorFunctions.ema(data, kw.get("period", 20)),
            dependencies=[],
        )
        self.register_indicator(
            "rsi", lambda data, **kw: IndicatorFunctions.rsi(data, kw.get("period", 14)),
            dependencies=[],
        )
        self.register_indicator(
            "obv", lambda data, **kw: IndicatorFunctions.obv(kw["closes"], kw["volumes"]),
            dependencies=[],
        )

    def register_indicator(
        self,
        name: str,
        compute_fn: Callable,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Register a custom indicator computation function."""
        with self._lock:
            self._compute_fns[name] = compute_fn
            if dependencies:
                self._dep_graph.add_dependencies(name, dependencies)

    def register_config(self, config: IndicatorConfig) -> None:
        """Register an indicator configuration."""
        with self._lock:
            self._configs[config.cache_key] = config
            if config.dependencies:
                self._dep_graph.add_dependencies(config.name, config.dependencies)

    def compute(
        self,
        name: str,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        """
        Compute or retrieve a cached indicator.
        Returns numpy array of indicator values.
        """
        params = params or {}
        config = IndicatorConfig(name=name, params=params)
        cache_key = config.cache_key

        with self._lock:
            cached = self._cache.get(cache_key)

            # Check if cache is still valid
            if cached and not force:
                if (
                    cached.is_valid
                    and not cached.is_expired
                    and cached.last_input_length == len(data)
                ):
                    cached.hit_count += 1
                    self._total_hits += 1
                    return cached.values

                # Incremental update - only recompute if data grew
                if (
                    cached.is_valid
                    and not cached.is_expired
                    and cached.last_input_length < len(data)
                    and config.name in self._compute_fns
                ):
                    pass  # Fall through to recompute

        # Compute the indicator
        compute_fn = self._compute_fns.get(name)
        if compute_fn is None:
            logger.warning("No compute function registered for indicator: %s", name)
            return None

        start_time = time.time()
        try:
            result = compute_fn(data, **params, **kwargs)
        except Exception as exc:
            logger.error("Error computing indicator %s: %s", name, exc)
            return None
        elapsed_ms = (time.time() - start_time) * 1000

        # Handle tuple returns (e.g., MACD returns 3 arrays)
        if isinstance(result, tuple):
            result = result[0]  # Store primary value

        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.float64)

        # Cache the result
        cached_result = CachedIndicator(
            name=name,
            cache_key=cache_key,
            values=result,
            last_input_length=len(data),
            last_compute_time=time.time(),
            compute_duration_ms=elapsed_ms,
            hit_count=0,
            created_at=time.time(),
            ttl=config.cache_ttl_seconds,
            params=params,
            is_valid=True,
        )

        with self._lock:
            self._cache[cache_key] = cached_result
            self._total_computes += 1

        return result

    def compute_multi(
        self,
        name: str,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Compute an indicator that returns multiple arrays (e.g., MACD, Bollinger).
        Returns tuple of numpy arrays.
        """
        params = params or {}
        compute_fn = self._compute_fns.get(name)
        if compute_fn is None:
            logger.warning("No compute function registered for: %s", name)
            return None

        try:
            result = compute_fn(data, **params, **kwargs)
        except Exception as exc:
            logger.error("Error computing multi-indicator %s: %s", name, exc)
            return None

        if isinstance(result, tuple):
            return result
        return (result,)

    def get_cached(self, name: str, params: Optional[Dict[str, Any]] = None) -> Optional[CachedIndicator]:
        """Get a cached indicator result without recomputing."""
        config = IndicatorConfig(name=name, params=params or {})
        with self._lock:
            return self._cache.get(config.cache_key)

    def invalidate(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Invalidate a cached indicator and all its dependents."""
        config = IndicatorConfig(name=name, params=params or {})
        cache_key = config.cache_key

        with self._lock:
            if cache_key in self._cache:
                self._cache[cache_key].is_valid = False

            # Also invalidate dependents
            dependents = self._dep_graph.get_all_dependents(name)
            for dep_name in dependents:
                for key, cached in self._cache.items():
                    if cached.name == dep_name:
                        cached.is_valid = False

    def invalidate_all(self) -> None:
        """Invalidate all cached indicators."""
        with self._lock:
            for cached in self._cache.values():
                cached.is_valid = False

    def clear(self) -> None:
        """Remove all cached data."""
        with self._lock:
            self._cache.clear()

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count evicted."""
        evicted = 0
        with self._lock:
            expired_keys = [
                key for key, cached in self._cache.items()
                if cached.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                evicted += 1
        return evicted

    def evict_lru(self, target_count: Optional[int] = None) -> int:
        """Evict least-recently-used entries. Returns count evicted."""
        with self._lock:
            if target_count is None:
                target_count = max(1, len(self._cache) // 2)

            if len(self._cache) <= target_count:
                return 0

            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_compute_time,
            )

            to_evict = len(self._cache) - target_count
            evicted = 0
            for key, _ in sorted_items[:to_evict]:
                del self._cache[key]
                evicted += 1

        return evicted

    @property
    def memory_usage_bytes(self) -> int:
        total = 0
        with self._lock:
            for cached in self._cache.values():
                if cached.values is not None:
                    total += cached.values.nbytes
        return total

    @property
    def memory_usage_mb(self) -> float:
        return self.memory_usage_bytes / (1024 * 1024)

    @property
    def dependency_graph(self) -> IndicatorDependencyGraph:
        return self._dep_graph

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_entries = len(self._cache)
            valid_entries = sum(1 for c in self._cache.values() if c.is_valid)
            expired_entries = sum(1 for c in self._cache.values() if c.is_expired)

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "total_computes": self._total_computes,
            "total_hits": self._total_hits,
            "hit_rate": (
                round(self._total_hits / (self._total_hits + self._total_computes) * 100, 2)
                if (self._total_hits + self._total_computes) > 0
                else 0.0
            ),
            "memory_mb": round(self.memory_usage_mb, 2),
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
            "registered_indicators": list(self._compute_fns.keys()),
        }
