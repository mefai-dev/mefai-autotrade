# Mefai Signal Engine - Walk-Forward Analysis
# Anchored and rolling walk-forward optimization with out-of-sample validation,
# parameter stability analysis, degradation ratio, and statistical significance.

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardWindow:
    """A single walk-forward window (train + test period)."""
    window_index: int = 0
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None
    train_bars: int = 0
    test_bars: int = 0
    best_params: Dict[str, Any] = field(default_factory=dict)
    in_sample_metric: float = 0.0
    out_of_sample_metric: float = 0.0
    in_sample_return: float = 0.0
    out_of_sample_return: float = 0.0
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    in_sample_max_dd: float = 0.0
    out_of_sample_max_dd: float = 0.0
    in_sample_trades: int = 0
    out_of_sample_trades: int = 0
    in_sample_win_rate: float = 0.0
    out_of_sample_win_rate: float = 0.0
    optimization_time: float = 0.0
    evaluation_time: float = 0.0


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    mode: str = "rolling"  # "rolling" or "anchored"
    n_windows: int = 5
    train_ratio: float = 0.7  # fraction of window for training
    min_train_bars: int = 100
    min_test_bars: int = 50
    step_bars: int = 0  # bars to step forward (0 = auto)
    optimization_metric: str = "sharpe"
    max_optimization_iterations: int = 200
    verbose: bool = True


@dataclass
class ParameterStabilityResult:
    """Results of parameter stability analysis across windows."""
    parameter_name: str = ""
    values_per_window: List[Any] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    coefficient_of_variation: float = 0.0
    stability_score: float = 0.0  # 0 = unstable, 1 = perfectly stable
    trend: str = ""  # "stable", "increasing", "decreasing", "erratic"


@dataclass
class DegradationAnalysis:
    """Analysis of in-sample vs out-of-sample performance degradation."""
    degradation_ratio: float = 0.0  # (IS - OOS) / IS
    is_avg_metric: float = 0.0
    oos_avg_metric: float = 0.0
    is_avg_return: float = 0.0
    oos_avg_return: float = 0.0
    is_avg_sharpe: float = 0.0
    oos_avg_sharpe: float = 0.0
    consistency_ratio: float = 0.0  # fraction of windows where OOS > 0
    worst_oos_metric: float = 0.0
    best_oos_metric: float = 0.0
    metric_correlation: float = 0.0  # correlation between IS and OOS


@dataclass
class StatisticalSignificance:
    """Statistical significance of walk-forward results."""
    t_statistic: float = 0.0
    p_value: float = 0.0
    degrees_of_freedom: int = 0
    is_significant_5pct: bool = False
    is_significant_1pct: bool = False
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    mean_oos_return: float = 0.0
    std_oos_return: float = 0.0


@dataclass
class WalkForwardResult:
    """Complete results from a walk-forward analysis."""
    config: WalkForwardConfig = None
    windows: List[WalkForwardWindow] = field(default_factory=list)
    parameter_stability: List[ParameterStabilityResult] = field(default_factory=list)
    degradation: Optional[DegradationAnalysis] = None
    significance: Optional[StatisticalSignificance] = None
    aggregate_oos_return: float = 0.0
    aggregate_oos_sharpe: float = 0.0
    aggregate_oos_trades: int = 0
    aggregate_oos_win_rate: float = 0.0
    total_time: float = 0.0
    passed: bool = False  # overall quality assessment
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Walk-forward analyzer
# ---------------------------------------------------------------------------

class WalkForwardAnalyzer:
    """
    Walk-forward analysis engine.

    Splits historical data into sequential train/test windows, optimizes
    strategy parameters on each training window, then evaluates on the
    corresponding test window. This validates that optimized parameters
    generalize to unseen data.

    Supports:
    - Anchored walk-forward (growing training window)
    - Rolling walk-forward (fixed training window size)
    - Configurable train/test ratio
    - Parameter stability analysis across windows
    - Performance degradation analysis (IS vs OOS)
    - Statistical significance testing

    Usage:
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(
            bars=historical_data,
            optimize_fn=my_optimizer,
            evaluate_fn=my_evaluator,
        )
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def run(self,
            bars: List[Dict[str, Any]],
            optimize_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
            evaluate_fn: Callable[[List[Dict[str, Any]], Dict[str, Any]], Dict[str, float]],
            ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        bars: full historical dataset (list of OHLCV bar dicts)
        optimize_fn: function(train_bars) -> best_params
            Runs optimization on training data and returns best parameters.
        evaluate_fn: function(test_bars, params) -> metrics_dict
            Evaluates a parameter set on a data segment.
            Must return dict with keys: metric, return, sharpe, max_dd,
            trades, win_rate.
        """
        start_time = time.time()

        if not bars:
            raise ValueError("No data provided for walk-forward analysis")

        # Generate windows
        windows = self._generate_windows(bars)

        if not windows:
            raise ValueError(
                f"Cannot generate walk-forward windows. "
                f"Need at least {self.config.min_train_bars + self.config.min_test_bars} bars, "
                f"have {len(bars)}"
            )

        logger.info(
            f"Walk-forward analysis: {len(windows)} windows, "
            f"mode={self.config.mode}, train_ratio={self.config.train_ratio}"
        )

        wf_windows = []
        all_params = defaultdict(list)

        for i, (train_slice, test_slice) in enumerate(windows):
            train_data = bars[train_slice[0]:train_slice[1]]
            test_data = bars[test_slice[0]:test_slice[1]]

            wf = WalkForwardWindow(
                window_index=i,
                train_bars=len(train_data),
                test_bars=len(test_data),
            )

            # Set timestamps
            if train_data:
                wf.train_start = train_data[0].get("timestamp")
                wf.train_end = train_data[-1].get("timestamp")
            if test_data:
                wf.test_start = test_data[0].get("timestamp")
                wf.test_end = test_data[-1].get("timestamp")

            if self.config.verbose:
                logger.info(
                    f"  Window {i+1}/{len(windows)}: "
                    f"train={wf.train_start} to {wf.train_end} "
                    f"({len(train_data)} bars), "
                    f"test={wf.test_start} to {wf.test_end} "
                    f"({len(test_data)} bars)"
                )

            # Phase 1: Optimize on training data
            opt_start = time.time()
            try:
                best_params = optimize_fn(train_data)
                wf.best_params = best_params
                wf.optimization_time = time.time() - opt_start

                # Track parameter values across windows
                for param_name, param_val in best_params.items():
                    all_params[param_name].append(param_val)

            except Exception as e:
                logger.error(f"  Optimization error in window {i+1}: {e}")
                wf.optimization_time = time.time() - opt_start
                wf_windows.append(wf)
                continue

            # Phase 2: Evaluate on training data (in-sample)
            try:
                is_metrics = evaluate_fn(train_data, best_params)
                wf.in_sample_metric = is_metrics.get("metric", 0)
                wf.in_sample_return = is_metrics.get("return", 0)
                wf.in_sample_sharpe = is_metrics.get("sharpe", 0)
                wf.in_sample_max_dd = is_metrics.get("max_dd", 0)
                wf.in_sample_trades = is_metrics.get("trades", 0)
                wf.in_sample_win_rate = is_metrics.get("win_rate", 0)
            except Exception as e:
                logger.error(f"  IS evaluation error in window {i+1}: {e}")

            # Phase 3: Evaluate on test data (out-of-sample)
            eval_start = time.time()
            try:
                oos_metrics = evaluate_fn(test_data, best_params)
                wf.out_of_sample_metric = oos_metrics.get("metric", 0)
                wf.out_of_sample_return = oos_metrics.get("return", 0)
                wf.out_of_sample_sharpe = oos_metrics.get("sharpe", 0)
                wf.out_of_sample_max_dd = oos_metrics.get("max_dd", 0)
                wf.out_of_sample_trades = oos_metrics.get("trades", 0)
                wf.out_of_sample_win_rate = oos_metrics.get("win_rate", 0)
                wf.evaluation_time = time.time() - eval_start
            except Exception as e:
                logger.error(f"  OOS evaluation error in window {i+1}: {e}")
                wf.evaluation_time = time.time() - eval_start

            if self.config.verbose:
                logger.info(
                    f"    IS: metric={wf.in_sample_metric:.4f}, "
                    f"return={wf.in_sample_return:.2f}%, "
                    f"sharpe={wf.in_sample_sharpe:.3f}"
                )
                logger.info(
                    f"    OOS: metric={wf.out_of_sample_metric:.4f}, "
                    f"return={wf.out_of_sample_return:.2f}%, "
                    f"sharpe={wf.out_of_sample_sharpe:.3f}"
                )

            wf_windows.append(wf)

        # Analyze results
        stability = self._analyze_parameter_stability(all_params)
        degradation = self._analyze_degradation(wf_windows)
        significance = self._test_significance(wf_windows)

        # Aggregate OOS performance
        oos_returns = [w.out_of_sample_return for w in wf_windows]
        oos_sharpes = [w.out_of_sample_sharpe for w in wf_windows]
        oos_trades = sum(w.out_of_sample_trades for w in wf_windows)

        total_oos_return = 1.0
        for r in oos_returns:
            total_oos_return *= (1 + r / 100)
        aggregate_return = (total_oos_return - 1) * 100

        aggregate_sharpe = sum(oos_sharpes) / max(1, len(oos_sharpes))

        total_wins = sum(
            w.out_of_sample_trades * w.out_of_sample_win_rate / 100
            for w in wf_windows
        )
        aggregate_win_rate = (total_wins / oos_trades * 100) if oos_trades > 0 else 0

        # Overall quality assessment
        passed = self._assess_quality(
            degradation, significance, wf_windows
        )

        total_time = time.time() - start_time

        result = WalkForwardResult(
            config=self.config,
            windows=wf_windows,
            parameter_stability=stability,
            degradation=degradation,
            significance=significance,
            aggregate_oos_return=aggregate_return,
            aggregate_oos_sharpe=aggregate_sharpe,
            aggregate_oos_trades=oos_trades,
            aggregate_oos_win_rate=aggregate_win_rate,
            total_time=total_time,
            passed=passed,
            metadata={
                "total_bars": len(bars),
                "n_windows": len(wf_windows),
                "mode": self.config.mode,
            },
        )

        logger.info(
            f"Walk-forward complete: "
            f"OOS return={aggregate_return:.2f}%, "
            f"OOS sharpe={aggregate_sharpe:.3f}, "
            f"degradation={degradation.degradation_ratio:.3f}, "
            f"passed={passed}, "
            f"time={total_time:.1f}s"
        )

        return result

    def _generate_windows(self, bars: List[Dict[str, Any]]
                           ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Generate train/test window index pairs.

        Returns list of ((train_start, train_end), (test_start, test_end))
        where indices refer to positions in the bars list.
        """
        n = len(bars)
        n_windows = self.config.n_windows
        train_ratio = self.config.train_ratio
        min_train = self.config.min_train_bars
        min_test = self.config.min_test_bars

        if n < min_train + min_test:
            return []

        windows = []

        if self.config.mode == "rolling":
            windows = self._generate_rolling_windows(n, n_windows,
                                                       train_ratio,
                                                       min_train, min_test)
        elif self.config.mode == "anchored":
            windows = self._generate_anchored_windows(n, n_windows,
                                                        train_ratio,
                                                        min_train, min_test)
        else:
            raise ValueError(f"Unknown walk-forward mode: {self.config.mode}")

        return windows

    def _generate_rolling_windows(self, n: int, n_windows: int,
                                    train_ratio: float,
                                    min_train: int,
                                    min_test: int) -> List[Tuple]:
        """Generate rolling (fixed-size) walk-forward windows."""
        step = self.config.step_bars
        if step <= 0:
            # Auto-calculate step to create n_windows
            # Total = train_size + (n_windows * step)
            # We need the last window's test to end at n
            window_size = n // n_windows
            train_size = max(min_train, int(window_size * train_ratio))
            test_size = max(min_test, window_size - train_size)
            step = test_size
        else:
            train_size = max(min_train, int(step * train_ratio / (1 - train_ratio)))
            test_size = step

        windows = []
        pos = 0

        while pos + train_size + test_size <= n and len(windows) < n_windows:
            train_start = pos
            train_end = pos + train_size
            test_start = train_end
            test_end = min(test_start + test_size, n)

            if test_end - test_start >= min_test:
                windows.append(
                    ((train_start, train_end), (test_start, test_end))
                )

            pos += step

        # If we got fewer windows than requested, try with smaller step
        if len(windows) < n_windows and step > min_test:
            return self._generate_rolling_windows(
                n, n_windows, train_ratio, min_train, min_test
            )

        return windows

    def _generate_anchored_windows(self, n: int, n_windows: int,
                                     train_ratio: float,
                                     min_train: int,
                                     min_test: int) -> List[Tuple]:
        """Generate anchored (growing training window) walk-forward windows."""
        # Training always starts at 0, grows each window
        # Test window moves forward
        usable = n - min_train
        if usable < min_test:
            return []

        test_size = max(min_test, usable // (n_windows + 1))
        windows = []

        for i in range(n_windows):
            test_end_idx = n - (n_windows - i - 1) * test_size
            test_start_idx = test_end_idx - test_size
            train_start = 0
            train_end = test_start_idx

            if train_end - train_start < min_train:
                continue
            if test_end_idx > n:
                continue

            windows.append(
                ((train_start, train_end), (test_start_idx, test_end_idx))
            )

        return windows

    def _analyze_parameter_stability(self,
                                       all_params: Dict[str, List[Any]]
                                       ) -> List[ParameterStabilityResult]:
        """Analyze how stable each parameter is across windows."""
        results = []

        for param_name, values in all_params.items():
            psr = ParameterStabilityResult(
                parameter_name=param_name,
                values_per_window=values,
            )

            if not values:
                results.append(psr)
                continue

            # Check if values are numeric
            try:
                numeric = [float(v) for v in values]
            except (ValueError, TypeError):
                # Categorical parameter
                unique = len(set(values))
                psr.stability_score = 1.0 / max(1, unique)
                psr.trend = "stable" if unique == 1 else "erratic"
                results.append(psr)
                continue

            n = len(numeric)
            psr.mean = sum(numeric) / n
            variance = sum((v - psr.mean) ** 2 for v in numeric) / max(1, n - 1)
            psr.std = variance ** 0.5

            if abs(psr.mean) > 1e-10:
                psr.coefficient_of_variation = psr.std / abs(psr.mean)
                psr.stability_score = max(0.0, 1.0 - psr.coefficient_of_variation)
            else:
                psr.stability_score = 1.0 if psr.std < 1e-10 else 0.0

            # Detect trend
            if n >= 3:
                # Simple linear regression
                x_mean = (n - 1) / 2
                x_vals = list(range(n))
                cov = sum((x - x_mean) * (y - psr.mean) for x, y in zip(x_vals, numeric))
                var_x = sum((x - x_mean) ** 2 for x in x_vals)
                slope = cov / var_x if var_x > 0 else 0

                # Normalize slope
                if abs(psr.mean) > 1e-10:
                    norm_slope = abs(slope * n / psr.mean)
                else:
                    norm_slope = abs(slope * n)

                if norm_slope < 0.1:
                    psr.trend = "stable"
                elif slope > 0:
                    psr.trend = "increasing"
                elif slope < 0:
                    psr.trend = "decreasing"
                else:
                    psr.trend = "stable"

                # Erratic if CV is too high
                if psr.coefficient_of_variation > 0.5:
                    psr.trend = "erratic"
            else:
                psr.trend = "insufficient_data"

            results.append(psr)

        return results

    def _analyze_degradation(self,
                               windows: List[WalkForwardWindow]
                               ) -> DegradationAnalysis:
        """Analyze performance degradation from IS to OOS."""
        da = DegradationAnalysis()

        if not windows:
            return da

        n = len(windows)
        is_metrics = [w.in_sample_metric for w in windows]
        oos_metrics = [w.out_of_sample_metric for w in windows]
        is_returns = [w.in_sample_return for w in windows]
        oos_returns = [w.out_of_sample_return for w in windows]
        is_sharpes = [w.in_sample_sharpe for w in windows]
        oos_sharpes = [w.out_of_sample_sharpe for w in windows]

        da.is_avg_metric = sum(is_metrics) / n
        da.oos_avg_metric = sum(oos_metrics) / n
        da.is_avg_return = sum(is_returns) / n
        da.oos_avg_return = sum(oos_returns) / n
        da.is_avg_sharpe = sum(is_sharpes) / n
        da.oos_avg_sharpe = sum(oos_sharpes) / n

        # Degradation ratio
        if abs(da.is_avg_metric) > 1e-10:
            da.degradation_ratio = (
                (da.is_avg_metric - da.oos_avg_metric) / abs(da.is_avg_metric)
            )
        else:
            da.degradation_ratio = 0.0

        # Consistency
        positive_oos = sum(1 for m in oos_metrics if m > 0)
        da.consistency_ratio = positive_oos / n

        da.worst_oos_metric = min(oos_metrics) if oos_metrics else 0
        da.best_oos_metric = max(oos_metrics) if oos_metrics else 0

        # Correlation between IS and OOS
        if n >= 3:
            is_mean = da.is_avg_metric
            oos_mean = da.oos_avg_metric
            cov = sum(
                (is_v - is_mean) * (oos_v - oos_mean)
                for is_v, oos_v in zip(is_metrics, oos_metrics)
            ) / n
            is_var = sum((v - is_mean) ** 2 for v in is_metrics) / n
            oos_var = sum((v - oos_mean) ** 2 for v in oos_metrics) / n
            denom = (is_var * oos_var) ** 0.5
            da.metric_correlation = cov / denom if denom > 1e-10 else 0.0

        return da

    def _test_significance(self,
                             windows: List[WalkForwardWindow]
                             ) -> StatisticalSignificance:
        """Test statistical significance of out-of-sample results."""
        ss = StatisticalSignificance()

        if not windows or len(windows) < 2:
            return ss

        oos_returns = [w.out_of_sample_return for w in windows]
        n = len(oos_returns)

        mean = sum(oos_returns) / n
        variance = sum((r - mean) ** 2 for r in oos_returns) / (n - 1)
        std = variance ** 0.5

        ss.mean_oos_return = mean
        ss.std_oos_return = std
        ss.degrees_of_freedom = n - 1

        # T-test (H0: mean OOS return = 0)
        if std > 1e-10:
            ss.t_statistic = mean / (std / math.sqrt(n))
        else:
            ss.t_statistic = 0.0

        # Approximate p-value using t-distribution
        # Using Abramowitz and Stegun approximation for CDF
        ss.p_value = self._approximate_t_pvalue(
            abs(ss.t_statistic), ss.degrees_of_freedom
        ) * 2  # two-tailed

        ss.is_significant_5pct = ss.p_value < 0.05
        ss.is_significant_1pct = ss.p_value < 0.01

        # 95% confidence interval
        t_crit = self._approximate_t_critical(0.025, ss.degrees_of_freedom)
        margin = t_crit * std / math.sqrt(n)
        ss.confidence_interval_95 = (mean - margin, mean + margin)

        return ss

    def _assess_quality(self, degradation: DegradationAnalysis,
                         significance: StatisticalSignificance,
                         windows: List[WalkForwardWindow]) -> bool:
        """
        Overall quality assessment of the walk-forward analysis.

        A strategy passes if:
        1. OOS results are statistically significant (p < 0.05)
        2. Degradation ratio is reasonable (< 0.5)
        3. Majority of OOS windows are profitable
        4. OOS Sharpe > 0
        """
        if not windows:
            return False

        checks = []

        # Check 1: Statistical significance
        checks.append(significance.is_significant_5pct)

        # Check 2: Degradation ratio
        checks.append(abs(degradation.degradation_ratio) < 0.5)

        # Check 3: Consistency
        checks.append(degradation.consistency_ratio >= 0.5)

        # Check 4: Positive OOS Sharpe
        checks.append(degradation.oos_avg_sharpe > 0)

        # Need at least 3 of 4 checks to pass
        return sum(checks) >= 3

    @staticmethod
    def _approximate_t_pvalue(t: float, df: int) -> float:
        """Approximate one-tailed p-value for t-distribution."""
        # Use normal approximation for large df
        if df > 30:
            # Normal CDF approximation
            return WalkForwardAnalyzer._normal_sf(t)

        # Adjusted approximation for small df
        x = df / (df + t * t)
        if x <= 0:
            return 0.0
        if x >= 1:
            return 0.5

        # Regularized incomplete beta function approximation
        # Simple approximation using normal with adjusted t
        adjusted_t = t * (1 - 1 / (4 * max(1, df)))
        return WalkForwardAnalyzer._normal_sf(adjusted_t)

    @staticmethod
    def _approximate_t_critical(alpha: float, df: int) -> float:
        """Approximate t critical value."""
        # Normal approximation with correction
        z = WalkForwardAnalyzer._normal_ppf(1 - alpha)
        # Cornish-Fisher expansion
        g1 = (z ** 3 + z) / (4 * max(1, df))
        g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * max(1, df * df))
        return z + g1 + g2

    @staticmethod
    def _normal_sf(x: float) -> float:
        """Survival function (1 - CDF) of standard normal."""
        return 1.0 - WalkForwardAnalyzer._normal_cdf(x)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """CDF of standard normal distribution."""
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        d = 0.3989422804014327
        p = d * math.exp(-0.5 * x * x) * (
            t * (0.3193815 + t * (-0.3565638 + t * (1.781478 +
            t * (-1.821256 + t * 1.330274))))
        )
        if x > 0:
            return 1.0 - p
        return p

    @staticmethod
    def _normal_ppf(p: float) -> float:
        """Inverse CDF (percent point function) of standard normal."""
        # Rational approximation (Abramowitz and Stegun 26.2.23)
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0
        if p == 0.5:
            return 0.0

        if p > 0.5:
            sign = 1
            p_adj = 1 - p
        else:
            sign = -1
            p_adj = p

        t = math.sqrt(-2 * math.log(p_adj))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        z = t - (c0 + c1 * t + c2 * t * t) / \
            (1 + d1 * t + d2 * t * t + d3 * t * t * t)

        return sign * z

    # -- Convenience methods --

    def print_summary(self, result: WalkForwardResult) -> str:
        """Print a formatted summary of walk-forward results."""
        lines = []
        lines.append("=" * 60)
        lines.append("  WALK-FORWARD ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"  Mode:           {result.config.mode}")
        lines.append(f"  Windows:        {len(result.windows)}")
        lines.append(f"  Train Ratio:    {result.config.train_ratio:.0%}")
        lines.append(f"  Total Time:     {result.total_time:.1f}s")
        lines.append(f"  Result:         {'PASSED' if result.passed else 'FAILED'}")
        lines.append("")

        # Per-window results
        lines.append("-- Per-Window Results --")
        lines.append(
            f"  {'#':>3} | {'IS Metric':>10} | {'OOS Metric':>10} | "
            f"{'IS Return':>10} | {'OOS Return':>10} | {'Degrad':>8}"
        )
        lines.append("  " + "-" * 65)

        for w in result.windows:
            degrad = 0.0
            if abs(w.in_sample_metric) > 1e-10:
                degrad = (w.in_sample_metric - w.out_of_sample_metric) / \
                         abs(w.in_sample_metric)
            lines.append(
                f"  {w.window_index+1:>3} | "
                f"{w.in_sample_metric:>10.4f} | "
                f"{w.out_of_sample_metric:>10.4f} | "
                f"{w.in_sample_return:>9.2f}% | "
                f"{w.out_of_sample_return:>9.2f}% | "
                f"{degrad:>7.1%}"
            )

        lines.append("")

        # Aggregate
        lines.append("-- Aggregate OOS Performance --")
        lines.append(f"  Return:         {result.aggregate_oos_return:.2f}%")
        lines.append(f"  Sharpe:         {result.aggregate_oos_sharpe:.3f}")
        lines.append(f"  Trades:         {result.aggregate_oos_trades}")
        lines.append(f"  Win Rate:       {result.aggregate_oos_win_rate:.1f}%")
        lines.append("")

        # Degradation
        if result.degradation:
            d = result.degradation
            lines.append("-- Degradation Analysis --")
            lines.append(f"  Degradation:    {d.degradation_ratio:.1%}")
            lines.append(f"  Consistency:    {d.consistency_ratio:.1%}")
            lines.append(f"  IS vs OOS Corr: {d.metric_correlation:.3f}")
            lines.append("")

        # Significance
        if result.significance:
            s = result.significance
            lines.append("-- Statistical Significance --")
            lines.append(f"  T-Statistic:    {s.t_statistic:.3f}")
            lines.append(f"  P-Value:        {s.p_value:.4f}")
            lines.append(f"  Significant (5%): {'Yes' if s.is_significant_5pct else 'No'}")
            ci = s.confidence_interval_95
            lines.append(f"  95% CI:         [{ci[0]:.2f}%, {ci[1]:.2f}%]")
            lines.append("")

        # Parameter stability
        if result.parameter_stability:
            lines.append("-- Parameter Stability --")
            for ps in result.parameter_stability:
                lines.append(
                    f"  {ps.parameter_name}: "
                    f"stability={ps.stability_score:.3f}, "
                    f"trend={ps.trend}, "
                    f"CV={ps.coefficient_of_variation:.3f}"
                )
            lines.append("")

        lines.append("=" * 60)

        output = "\n".join(lines)
        print(output)
        return output
