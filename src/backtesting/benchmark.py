# Mefai Signal Engine - Benchmark Comparisons
# Buy-and-hold, equal-weight, risk-parity, random entry benchmarks,
# alpha/beta calculation, information ratio, and tracking error.

import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result from a single benchmark comparison."""
    name: str = ""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    volatility: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Comparison of strategy performance against a benchmark."""
    strategy_name: str = ""
    benchmark_name: str = ""
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    excess_return: float = 0.0
    excess_return_pct: float = 0.0
    correlation: float = 0.0
    up_capture: float = 0.0
    down_capture: float = 0.0
    strategy_sharpe: float = 0.0
    benchmark_sharpe: float = 0.0
    strategy_max_dd: float = 0.0
    benchmark_max_dd: float = 0.0
    outperformance_periods: float = 0.0  # fraction of periods where strategy > benchmark


@dataclass
class FullBenchmarkResult:
    """Complete benchmark analysis with multiple benchmarks."""
    comparisons: List[ComparisonResult] = field(default_factory=list)
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    strategy_result: Optional[BenchmarkResult] = None
    best_benchmark: str = ""
    worst_benchmark: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

class BenchmarkMetrics:
    """Static metric calculation methods for benchmarks."""

    @staticmethod
    def calculate_returns(prices: List[float]) -> List[float]:
        """Calculate periodic returns from price series."""
        if len(prices) < 2:
            return []
        return [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]

    @staticmethod
    def sharpe_ratio(returns: List[float], risk_free: float = 0.0,
                      periods: float = 252.0) -> float:
        if not returns or len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns) - risk_free / periods
        variance = sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / (len(returns) - 1)
        std = variance ** 0.5
        if std < 1e-10:
            return 0.0
        return (mean / std) * math.sqrt(periods)

    @staticmethod
    def sortino_ratio(returns: List[float], risk_free: float = 0.0,
                       periods: float = 252.0) -> float:
        if not returns or len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns) - risk_free / periods
        downside = [r for r in returns if r < 0]
        if not downside:
            return float('inf') if mean > 0 else 0.0
        dd_var = sum(r ** 2 for r in downside) / len(returns)
        dd_std = dd_var ** 0.5
        if dd_std < 1e-10:
            return 0.0
        return (mean / dd_std) * math.sqrt(periods)

    @staticmethod
    def max_drawdown(equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def volatility(returns: List[float], periods: float = 252.0) -> float:
        if not returns or len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return (variance ** 0.5) * math.sqrt(periods) * 100

    @staticmethod
    def cagr(initial: float, final: float, years: float) -> float:
        if initial <= 0 or final <= 0 or years <= 0:
            return 0.0
        return ((final / initial) ** (1.0 / years) - 1.0) * 100

    @staticmethod
    def alpha_beta(strategy_returns: List[float],
                    benchmark_returns: List[float],
                    risk_free: float = 0.0,
                    periods: float = 252.0) -> Tuple[float, float]:
        """Calculate alpha and beta using linear regression."""
        n = min(len(strategy_returns), len(benchmark_returns))
        if n < 2:
            return (0.0, 1.0)

        sr = strategy_returns[:n]
        br = benchmark_returns[:n]
        rf = risk_free / periods

        sr_excess = [s - rf for s in sr]
        br_excess = [b - rf for b in br]

        br_mean = sum(br_excess) / n
        sr_mean = sum(sr_excess) / n

        cov = sum(
            (s - sr_mean) * (b - br_mean)
            for s, b in zip(sr_excess, br_excess)
        ) / n

        br_var = sum((b - br_mean) ** 2 for b in br_excess) / n

        beta = cov / br_var if br_var > 1e-10 else 1.0
        alpha = (sr_mean - beta * br_mean) * periods  # annualized

        return (alpha, beta)

    @staticmethod
    def information_ratio(strategy_returns: List[float],
                           benchmark_returns: List[float],
                           periods: float = 252.0) -> float:
        """Calculate information ratio (excess return / tracking error)."""
        n = min(len(strategy_returns), len(benchmark_returns))
        if n < 2:
            return 0.0

        excess = [s - b for s, b in zip(
            strategy_returns[:n], benchmark_returns[:n]
        )]
        mean_excess = sum(excess) / n
        te_var = sum((e - mean_excess) ** 2 for e in excess) / (n - 1)
        te = te_var ** 0.5

        if te < 1e-10:
            return 0.0
        return (mean_excess / te) * math.sqrt(periods)

    @staticmethod
    def tracking_error(strategy_returns: List[float],
                        benchmark_returns: List[float],
                        periods: float = 252.0) -> float:
        """Calculate annualized tracking error."""
        n = min(len(strategy_returns), len(benchmark_returns))
        if n < 2:
            return 0.0

        excess = [s - b for s, b in zip(
            strategy_returns[:n], benchmark_returns[:n]
        )]
        mean_excess = sum(excess) / n
        te_var = sum((e - mean_excess) ** 2 for e in excess) / (n - 1)
        return (te_var ** 0.5) * math.sqrt(periods) * 100

    @staticmethod
    def correlation(returns_a: List[float],
                     returns_b: List[float]) -> float:
        """Calculate Pearson correlation between two return series."""
        n = min(len(returns_a), len(returns_b))
        if n < 2:
            return 0.0

        a = returns_a[:n]
        b = returns_b[:n]

        a_mean = sum(a) / n
        b_mean = sum(b) / n

        cov = sum((ai - a_mean) * (bi - b_mean) for ai, bi in zip(a, b)) / n
        a_var = sum((ai - a_mean) ** 2 for ai in a) / n
        b_var = sum((bi - b_mean) ** 2 for bi in b) / n
        denom = (a_var * b_var) ** 0.5

        if denom < 1e-10:
            return 0.0
        return cov / denom

    @staticmethod
    def capture_ratios(strategy_returns: List[float],
                        benchmark_returns: List[float]) -> Tuple[float, float]:
        """Calculate up-capture and down-capture ratios."""
        n = min(len(strategy_returns), len(benchmark_returns))
        if n < 2:
            return (1.0, 1.0)

        up_strat = []
        up_bench = []
        down_strat = []
        down_bench = []

        for i in range(n):
            if benchmark_returns[i] > 0:
                up_strat.append(strategy_returns[i])
                up_bench.append(benchmark_returns[i])
            elif benchmark_returns[i] < 0:
                down_strat.append(strategy_returns[i])
                down_bench.append(benchmark_returns[i])

        up_capture = 1.0
        if up_bench:
            up_mean_s = sum(up_strat) / len(up_strat)
            up_mean_b = sum(up_bench) / len(up_bench)
            if abs(up_mean_b) > 1e-10:
                up_capture = up_mean_s / up_mean_b

        down_capture = 1.0
        if down_bench:
            down_mean_s = sum(down_strat) / len(down_strat)
            down_mean_b = sum(down_bench) / len(down_bench)
            if abs(down_mean_b) > 1e-10:
                down_capture = down_mean_s / down_mean_b

        return (up_capture, down_capture)


# ---------------------------------------------------------------------------
# Benchmark base class
# ---------------------------------------------------------------------------

class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calculate(self, bars: List[Dict[str, Any]],
                  initial_capital: float) -> BenchmarkResult:
        pass


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------

class BuyAndHoldBenchmark(Benchmark):
    """
    Buy-and-hold benchmark.

    Buys at the first bar and holds until the last bar.
    Simulates a full allocation at the start.
    """

    def __init__(self, symbol: Optional[str] = None):
        self._symbol = symbol

    @property
    def name(self) -> str:
        suffix = f" ({self._symbol})" if self._symbol else ""
        return f"Buy & Hold{suffix}"

    def calculate(self, bars: List[Dict[str, Any]],
                  initial_capital: float) -> BenchmarkResult:
        if not bars:
            return BenchmarkResult(name=self.name)

        entry_price = bars[0]["open"]
        if entry_price <= 0:
            entry_price = bars[0]["close"]

        position_size = initial_capital / entry_price
        equity_curve = []
        timestamps = []

        for bar in bars:
            equity = position_size * bar["close"]
            equity_curve.append(equity)
            ts = bar.get("timestamp")
            if ts:
                timestamps.append(ts)

        returns = BenchmarkMetrics.calculate_returns(equity_curve)
        final = equity_curve[-1] if equity_curve else initial_capital

        # Estimate trading days
        years = len(bars) / 252 if len(bars) > 0 else 1

        return BenchmarkResult(
            name=self.name,
            total_return=final - initial_capital,
            total_return_pct=(final - initial_capital) / initial_capital * 100,
            cagr=BenchmarkMetrics.cagr(initial_capital, final, years),
            sharpe_ratio=BenchmarkMetrics.sharpe_ratio(returns),
            sortino_ratio=BenchmarkMetrics.sortino_ratio(returns),
            max_drawdown_pct=BenchmarkMetrics.max_drawdown(equity_curve),
            volatility=BenchmarkMetrics.volatility(returns),
            total_trades=1,
            win_rate=100.0 if final > initial_capital else 0.0,
            equity_curve=equity_curve,
            returns=returns,
            timestamps=timestamps,
        )


# ---------------------------------------------------------------------------
# Equal-weight portfolio benchmark
# ---------------------------------------------------------------------------

class EqualWeightBenchmark(Benchmark):
    """
    Equal-weight portfolio benchmark.

    Divides capital equally among all symbols and rebalances
    at a configurable interval.
    """

    def __init__(self, rebalance_bars: int = 0):
        self.rebalance_bars = rebalance_bars  # 0 = no rebalancing

    @property
    def name(self) -> str:
        return "Equal Weight"

    def calculate(self, bars_by_symbol: Dict[str, List[Dict[str, Any]]],
                  initial_capital: float) -> BenchmarkResult:
        if not bars_by_symbol:
            return BenchmarkResult(name=self.name)

        n_symbols = len(bars_by_symbol)
        allocation = initial_capital / n_symbols

        # Calculate per-symbol equity
        symbol_equities: Dict[str, List[float]] = {}
        min_len = min(len(b) for b in bars_by_symbol.values())

        for symbol, bars in bars_by_symbol.items():
            if not bars:
                continue
            entry_price = bars[0]["open"]
            if entry_price <= 0:
                continue
            pos_size = allocation / entry_price
            symbol_equities[symbol] = [
                pos_size * bars[i]["close"] for i in range(min_len)
            ]

        if not symbol_equities:
            return BenchmarkResult(name=self.name)

        # Sum across symbols
        portfolio_equity = []
        for i in range(min_len):
            total = sum(eq[i] for eq in symbol_equities.values())
            portfolio_equity.append(total)

        returns = BenchmarkMetrics.calculate_returns(portfolio_equity)
        final = portfolio_equity[-1] if portfolio_equity else initial_capital
        years = min_len / 252 if min_len > 0 else 1

        return BenchmarkResult(
            name=self.name,
            total_return=final - initial_capital,
            total_return_pct=(final - initial_capital) / initial_capital * 100,
            cagr=BenchmarkMetrics.cagr(initial_capital, final, years),
            sharpe_ratio=BenchmarkMetrics.sharpe_ratio(returns),
            sortino_ratio=BenchmarkMetrics.sortino_ratio(returns),
            max_drawdown_pct=BenchmarkMetrics.max_drawdown(portfolio_equity),
            volatility=BenchmarkMetrics.volatility(returns),
            total_trades=n_symbols,
            equity_curve=portfolio_equity,
            returns=returns,
        )


# ---------------------------------------------------------------------------
# Risk-parity benchmark
# ---------------------------------------------------------------------------

class RiskParityBenchmark(Benchmark):
    """
    Risk-parity benchmark.

    Allocates capital inversely proportional to each asset's volatility,
    so each asset contributes equal risk to the portfolio.
    """

    def __init__(self, lookback: int = 60, rebalance_bars: int = 20):
        self.lookback = lookback
        self.rebalance_bars = rebalance_bars

    @property
    def name(self) -> str:
        return "Risk Parity"

    def calculate(self, bars_by_symbol: Dict[str, List[Dict[str, Any]]],
                  initial_capital: float) -> BenchmarkResult:
        if not bars_by_symbol:
            return BenchmarkResult(name=self.name)

        symbols = list(bars_by_symbol.keys())
        min_len = min(len(b) for b in bars_by_symbol.values())

        if min_len < self.lookback + 1:
            return BenchmarkResult(name=self.name)

        # Build price matrix
        prices: Dict[str, List[float]] = {}
        for symbol, bars in bars_by_symbol.items():
            prices[symbol] = [bars[i]["close"] for i in range(min_len)]

        portfolio_equity = [initial_capital]
        capital = initial_capital
        positions: Dict[str, float] = {}

        for i in range(self.lookback, min_len):
            # Rebalance check
            if (i - self.lookback) % self.rebalance_bars == 0 or not positions:
                # Calculate volatilities
                vols = {}
                for symbol in symbols:
                    window = prices[symbol][max(0, i - self.lookback):i]
                    rets = BenchmarkMetrics.calculate_returns(window)
                    if rets:
                        mean_r = sum(rets) / len(rets)
                        var = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
                        vols[symbol] = max(var ** 0.5, 1e-10)
                    else:
                        vols[symbol] = 1e-10

                # Inverse volatility weights
                inv_vols = {s: 1.0 / v for s, v in vols.items()}
                total_inv = sum(inv_vols.values())
                weights = {s: iv / total_inv for s, iv in inv_vols.items()}

                # Calculate new positions
                positions = {}
                for symbol in symbols:
                    alloc = capital * weights[symbol]
                    price = prices[symbol][i]
                    if price > 0:
                        positions[symbol] = alloc / price

            # Calculate portfolio value
            value = sum(
                positions.get(s, 0) * prices[s][i] for s in symbols
            )
            portfolio_equity.append(value)
            capital = value

        returns = BenchmarkMetrics.calculate_returns(portfolio_equity)
        final = portfolio_equity[-1] if portfolio_equity else initial_capital
        years = (min_len - self.lookback) / 252

        return BenchmarkResult(
            name=self.name,
            total_return=final - initial_capital,
            total_return_pct=(final - initial_capital) / initial_capital * 100,
            cagr=BenchmarkMetrics.cagr(initial_capital, final, years),
            sharpe_ratio=BenchmarkMetrics.sharpe_ratio(returns),
            sortino_ratio=BenchmarkMetrics.sortino_ratio(returns),
            max_drawdown_pct=BenchmarkMetrics.max_drawdown(portfolio_equity),
            volatility=BenchmarkMetrics.volatility(returns),
            equity_curve=portfolio_equity,
            returns=returns,
        )


# ---------------------------------------------------------------------------
# Random entry benchmark
# ---------------------------------------------------------------------------

class RandomEntryBenchmark(Benchmark):
    """
    Random entry benchmark.

    Generates n random entry/exit pairs and compares performance.
    This tests whether the strategy has genuine edge beyond random timing.
    """

    def __init__(self, n_simulations: int = 1000,
                 avg_holding_bars: int = 20,
                 seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.avg_holding_bars = avg_holding_bars
        if seed is not None:
            random.seed(seed)

    @property
    def name(self) -> str:
        return f"Random Entry ({self.n_simulations} sims)"

    def calculate(self, bars: List[Dict[str, Any]],
                  initial_capital: float,
                  n_trades: int = 50) -> BenchmarkResult:
        """
        Run random entry benchmark.

        n_trades: number of trades per simulation (should match strategy)
        """
        if not bars or len(bars) < self.avg_holding_bars * 2:
            return BenchmarkResult(name=self.name)

        all_returns = []
        all_final_equities = []
        all_sharpes = []
        all_drawdowns = []

        for _ in range(self.n_simulations):
            sim_equity, sim_returns = self._simulate_random(
                bars, initial_capital, n_trades
            )
            if sim_equity:
                all_final_equities.append(sim_equity[-1])
                all_returns.extend(sim_returns)
                sharpe = BenchmarkMetrics.sharpe_ratio(sim_returns)
                all_sharpes.append(sharpe)
                dd = BenchmarkMetrics.max_drawdown(sim_equity)
                all_drawdowns.append(dd)

        if not all_final_equities:
            return BenchmarkResult(name=self.name)

        n = len(all_final_equities)
        avg_final = sum(all_final_equities) / n
        avg_sharpe = sum(all_sharpes) / n
        avg_dd = sum(all_drawdowns) / n
        years = len(bars) / 252

        # Median equity curve (use average for simplicity)
        median_idx = n // 2
        sorted_finals = sorted(all_final_equities)
        median_final = sorted_finals[median_idx]

        return BenchmarkResult(
            name=self.name,
            total_return=avg_final - initial_capital,
            total_return_pct=(avg_final - initial_capital) / initial_capital * 100,
            cagr=BenchmarkMetrics.cagr(initial_capital, avg_final, years),
            sharpe_ratio=avg_sharpe,
            max_drawdown_pct=avg_dd,
            total_trades=n_trades,
            win_rate=sum(1 for e in all_final_equities if e > initial_capital) / n * 100,
        )

    def _simulate_random(self, bars: List[Dict[str, Any]],
                          initial_capital: float,
                          n_trades: int) -> Tuple[List[float], List[float]]:
        """Run a single random entry simulation."""
        n = len(bars)
        capital = initial_capital
        equity = [capital]
        returns_list = []

        for _ in range(n_trades):
            # Random entry point
            max_entry = n - self.avg_holding_bars - 1
            if max_entry < 1:
                break
            entry_idx = random.randint(0, max_entry)

            # Random holding period (geometric distribution around average)
            holding = max(1, int(random.expovariate(1.0 / self.avg_holding_bars)))
            exit_idx = min(entry_idx + holding, n - 1)

            # Random direction
            is_long = random.random() > 0.5

            entry_price = bars[entry_idx]["close"]
            exit_price = bars[exit_idx]["close"]

            if entry_price <= 0:
                continue

            if is_long:
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price

            pnl = capital * 0.02 * ret  # 2% position size
            capital += pnl
            capital = max(0, capital)
            equity.append(capital)
            returns_list.append(ret * 0.02)

        return equity, returns_list

    def percentile_rank(self, strategy_metric: float,
                         benchmark_metrics: List[float]) -> float:
        """
        Calculate where the strategy metric ranks among random simulations.
        Returns percentile (0-100). Higher = strategy beats more random sims.
        """
        if not benchmark_metrics:
            return 50.0
        beats = sum(1 for m in benchmark_metrics if strategy_metric > m)
        return beats / len(benchmark_metrics) * 100


# ---------------------------------------------------------------------------
# Custom benchmark
# ---------------------------------------------------------------------------

class CustomBenchmark(Benchmark):
    """
    User-defined benchmark.

    Allows providing a custom equity curve or return series.
    """

    def __init__(self, benchmark_name: str = "Custom",
                 equity_curve: Optional[List[float]] = None,
                 returns: Optional[List[float]] = None,
                 timestamps: Optional[List[datetime]] = None):
        self._name = benchmark_name
        self._equity_curve = equity_curve or []
        self._returns = returns or []
        self._timestamps = timestamps or []

    @property
    def name(self) -> str:
        return self._name

    def calculate(self, bars: List[Dict[str, Any]] = None,
                  initial_capital: float = 10000) -> BenchmarkResult:
        equity = self._equity_curve
        returns = self._returns

        if equity and not returns:
            returns = BenchmarkMetrics.calculate_returns(equity)
        elif returns and not equity:
            equity = [initial_capital]
            for r in returns:
                equity.append(equity[-1] * (1 + r))

        if not equity:
            return BenchmarkResult(name=self.name)

        final = equity[-1]
        years = len(equity) / 252

        return BenchmarkResult(
            name=self.name,
            total_return=final - initial_capital,
            total_return_pct=(final - initial_capital) / initial_capital * 100,
            cagr=BenchmarkMetrics.cagr(initial_capital, final, years),
            sharpe_ratio=BenchmarkMetrics.sharpe_ratio(returns),
            sortino_ratio=BenchmarkMetrics.sortino_ratio(returns),
            max_drawdown_pct=BenchmarkMetrics.max_drawdown(equity),
            volatility=BenchmarkMetrics.volatility(returns),
            equity_curve=equity,
            returns=returns,
            timestamps=self._timestamps,
        )


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------

class BenchmarkEngine:
    """
    Unified benchmark comparison engine.

    Compares strategy performance against multiple benchmarks and
    calculates alpha, beta, information ratio, tracking error,
    and capture ratios.

    Usage:
        engine = BenchmarkEngine(initial_capital=10000)
        engine.add_benchmark(BuyAndHoldBenchmark())
        engine.add_benchmark(RandomEntryBenchmark(n_simulations=1000))

        result = engine.compare(
            strategy_equity=[...],
            strategy_returns=[...],
            bars=historical_bars,
        )

        for comp in result.comparisons:
            print(f"{comp.benchmark_name}: alpha={comp.alpha:.3f}")
    """

    def __init__(self, initial_capital: float = 10000.0,
                 risk_free_rate: float = 0.0):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.benchmarks: List[Benchmark] = []

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for comparison."""
        self.benchmarks.append(benchmark)

    def compare(self, strategy_equity: List[float],
                strategy_returns: Optional[List[float]] = None,
                bars: Optional[List[Dict[str, Any]]] = None,
                bars_by_symbol: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                strategy_name: str = "Strategy",
                n_strategy_trades: int = 50) -> FullBenchmarkResult:
        """
        Compare strategy against all registered benchmarks.
        """
        if not strategy_equity:
            return FullBenchmarkResult()

        # Calculate strategy returns if not provided
        if strategy_returns is None:
            strategy_returns = BenchmarkMetrics.calculate_returns(strategy_equity)

        # Strategy as BenchmarkResult
        final = strategy_equity[-1]
        years = len(strategy_equity) / 252

        strategy_result = BenchmarkResult(
            name=strategy_name,
            total_return=final - self.initial_capital,
            total_return_pct=(final - self.initial_capital) / self.initial_capital * 100,
            cagr=BenchmarkMetrics.cagr(self.initial_capital, final, years),
            sharpe_ratio=BenchmarkMetrics.sharpe_ratio(strategy_returns),
            sortino_ratio=BenchmarkMetrics.sortino_ratio(strategy_returns),
            max_drawdown_pct=BenchmarkMetrics.max_drawdown(strategy_equity),
            volatility=BenchmarkMetrics.volatility(strategy_returns),
            equity_curve=strategy_equity,
            returns=strategy_returns,
        )

        comparisons = []
        benchmark_results = []

        for benchmark in self.benchmarks:
            try:
                # Calculate benchmark performance
                if isinstance(benchmark, (EqualWeightBenchmark, RiskParityBenchmark)):
                    if bars_by_symbol:
                        br = benchmark.calculate(bars_by_symbol, self.initial_capital)
                    else:
                        continue
                elif isinstance(benchmark, RandomEntryBenchmark):
                    if bars:
                        br = benchmark.calculate(
                            bars, self.initial_capital, n_strategy_trades
                        )
                    else:
                        continue
                elif isinstance(benchmark, CustomBenchmark):
                    br = benchmark.calculate(
                        initial_capital=self.initial_capital
                    )
                else:
                    if bars:
                        br = benchmark.calculate(bars, self.initial_capital)
                    else:
                        continue

                benchmark_results.append(br)

                # Compare
                bench_returns = br.returns or BenchmarkMetrics.calculate_returns(
                    br.equity_curve
                )

                alpha, beta = BenchmarkMetrics.alpha_beta(
                    strategy_returns, bench_returns, self.risk_free_rate
                )
                ir = BenchmarkMetrics.information_ratio(
                    strategy_returns, bench_returns
                )
                te = BenchmarkMetrics.tracking_error(
                    strategy_returns, bench_returns
                )
                corr = BenchmarkMetrics.correlation(
                    strategy_returns, bench_returns
                )
                up_cap, down_cap = BenchmarkMetrics.capture_ratios(
                    strategy_returns, bench_returns
                )

                # Outperformance periods
                n_periods = min(len(strategy_returns), len(bench_returns))
                outperform = 0
                for i in range(n_periods):
                    if strategy_returns[i] > bench_returns[i]:
                        outperform += 1
                outperform_pct = outperform / max(1, n_periods)

                comp = ComparisonResult(
                    strategy_name=strategy_name,
                    benchmark_name=benchmark.name,
                    alpha=alpha,
                    beta=beta,
                    information_ratio=ir,
                    tracking_error=te,
                    excess_return=strategy_result.total_return - br.total_return,
                    excess_return_pct=strategy_result.total_return_pct - br.total_return_pct,
                    correlation=corr,
                    up_capture=up_cap,
                    down_capture=down_cap,
                    strategy_sharpe=strategy_result.sharpe_ratio,
                    benchmark_sharpe=br.sharpe_ratio,
                    strategy_max_dd=strategy_result.max_drawdown_pct,
                    benchmark_max_dd=br.max_drawdown_pct,
                    outperformance_periods=outperform_pct,
                )
                comparisons.append(comp)

            except Exception as e:
                logger.error(
                    f"Error calculating benchmark {benchmark.name}: {e}"
                )
                continue

        # Find best/worst benchmark
        best = ""
        worst = ""
        if benchmark_results:
            sorted_br = sorted(
                benchmark_results, key=lambda b: b.total_return_pct
            )
            best = sorted_br[-1].name
            worst = sorted_br[0].name

        return FullBenchmarkResult(
            comparisons=comparisons,
            benchmarks=benchmark_results,
            strategy_result=strategy_result,
            best_benchmark=best,
            worst_benchmark=worst,
            metadata={
                "n_benchmarks": len(benchmark_results),
                "initial_capital": self.initial_capital,
                "risk_free_rate": self.risk_free_rate,
            },
        )

    def print_comparison(self, result: FullBenchmarkResult) -> str:
        """Print a formatted benchmark comparison table."""
        lines = []
        lines.append("=" * 70)
        lines.append("  BENCHMARK COMPARISON")
        lines.append("=" * 70)
        lines.append("")

        if result.strategy_result:
            s = result.strategy_result
            lines.append(f"  Strategy: {s.name}")
            lines.append(f"  Return: {s.total_return_pct:+.2f}%  |  "
                         f"Sharpe: {s.sharpe_ratio:.3f}  |  "
                         f"Max DD: {s.max_drawdown_pct:.2f}%")
            lines.append("")

        lines.append(
            f"  {'Benchmark':<25} | {'Return':>8} | {'Sharpe':>7} | "
            f"{'Alpha':>7} | {'Beta':>6} | {'IR':>6} | {'TE':>6}"
        )
        lines.append("  " + "-" * 68)

        for comp in result.comparisons:
            bench_ret = 0.0
            for br in result.benchmarks:
                if br.name == comp.benchmark_name:
                    bench_ret = br.total_return_pct
                    break

            lines.append(
                f"  {comp.benchmark_name:<25} | "
                f"{bench_ret:>7.2f}% | "
                f"{comp.benchmark_sharpe:>7.3f} | "
                f"{comp.alpha:>7.3f} | "
                f"{comp.beta:>6.3f} | "
                f"{comp.information_ratio:>6.3f} | "
                f"{comp.tracking_error:>5.2f}%"
            )

        lines.append("")
        lines.append("=" * 70)

        output = "\n".join(lines)
        print(output)
        return output
