# Mefai Signal Engine - Monte Carlo Analysis
# Trade sequence resampling, bootstrap confidence intervals, probability of ruin,
# worst-case scenarios, and optimal position sizing via simulation.

import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 1000
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.90, 0.95, 0.99]
    )
    seed: Optional[int] = None
    use_replacement: bool = True  # bootstrap with replacement
    preserve_streaks: bool = False  # preserve win/loss streak patterns
    block_size: int = 5  # block bootstrap block size
    initial_capital: float = 10000.0
    max_trades_per_sim: int = 0  # 0 = use all trades
    ruin_threshold: float = 0.5  # fraction of capital lost = ruin
    position_size_range: Tuple[float, float] = (0.01, 0.20)
    position_size_steps: int = 20


@dataclass
class EquityCurveStats:
    """Statistics for a set of simulated equity curves."""
    mean_final_equity: float = 0.0
    median_final_equity: float = 0.0
    std_final_equity: float = 0.0
    min_final_equity: float = 0.0
    max_final_equity: float = 0.0
    mean_max_drawdown: float = 0.0
    median_max_drawdown: float = 0.0
    worst_max_drawdown: float = 0.0
    mean_return_pct: float = 0.0
    median_return_pct: float = 0.0
    positive_return_pct: float = 0.0
    confidence_intervals: Dict[float, Tuple[float, float]] = field(
        default_factory=dict
    )


@dataclass
class RuinAnalysis:
    """Probability of ruin analysis results."""
    probability_of_ruin: float = 0.0
    ruin_threshold_pct: float = 50.0
    average_trades_to_ruin: float = 0.0
    median_trades_to_ruin: float = 0.0
    ruin_simulations: int = 0
    total_simulations: int = 0
    survival_rate: float = 0.0
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)


@dataclass
class PositionSizeResult:
    """Result of optimal position size analysis."""
    optimal_size: float = 0.0
    optimal_metric: float = 0.0
    sizes_tested: List[float] = field(default_factory=list)
    metrics: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    ruin_rates: List[float] = field(default_factory=list)
    kelly_fraction: float = 0.0


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo analysis results."""
    config: MonteCarloConfig = None
    equity_stats: EquityCurveStats = None
    ruin_analysis: RuinAnalysis = None
    position_size: Optional[PositionSizeResult] = None
    worst_case: Dict[str, Any] = field(default_factory=dict)
    random_comparison: Dict[str, Any] = field(default_factory=dict)
    trade_distribution: Dict[str, Any] = field(default_factory=dict)
    simulated_equity_curves: List[List[float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trade sequence resampler
# ---------------------------------------------------------------------------

class TradeResampler:
    """Resample trade sequences for Monte Carlo simulation."""

    @staticmethod
    def bootstrap(trades: List[float], n_trades: int,
                  with_replacement: bool = True) -> List[float]:
        """
        Simple bootstrap resampling of trade PnL values.
        """
        if with_replacement:
            return [random.choice(trades) for _ in range(n_trades)]
        else:
            if n_trades <= len(trades):
                return random.sample(trades, n_trades)
            else:
                result = []
                while len(result) < n_trades:
                    shuffled = trades.copy()
                    random.shuffle(shuffled)
                    result.extend(shuffled)
                return result[:n_trades]

    @staticmethod
    def block_bootstrap(trades: List[float], n_trades: int,
                         block_size: int = 5) -> List[float]:
        """
        Block bootstrap - preserves local serial correlation.
        Resamples blocks of consecutive trades.
        """
        n = len(trades)
        if n < block_size:
            return TradeResampler.bootstrap(trades, n_trades)

        result = []
        while len(result) < n_trades:
            start = random.randint(0, n - block_size)
            block = trades[start:start + block_size]
            result.extend(block)

        return result[:n_trades]

    @staticmethod
    def shuffle(trades: List[float]) -> List[float]:
        """Simple random shuffle of trade order."""
        shuffled = trades.copy()
        random.shuffle(shuffled)
        return shuffled

    @staticmethod
    def stratified_resample(winning_trades: List[float],
                             losing_trades: List[float],
                             n_trades: int,
                             win_rate: float) -> List[float]:
        """
        Stratified resampling that preserves the win rate.
        """
        n_wins = int(n_trades * win_rate)
        n_losses = n_trades - n_wins

        wins = [random.choice(winning_trades) for _ in range(n_wins)] \
            if winning_trades else []
        losses = [random.choice(losing_trades) for _ in range(n_losses)] \
            if losing_trades else []

        combined = wins + losses
        random.shuffle(combined)
        return combined


# ---------------------------------------------------------------------------
# Equity curve simulator
# ---------------------------------------------------------------------------

class EquityCurveSimulator:
    """Simulate equity curves from resampled trade sequences."""

    @staticmethod
    def simulate(trades: List[float], initial_capital: float,
                 position_size_pct: float = 1.0) -> List[float]:
        """
        Simulate an equity curve from a sequence of trade PnL values.

        position_size_pct: fraction of current equity risked per trade
        Returns list of equity values (length = len(trades) + 1).
        """
        equity = [initial_capital]
        current = initial_capital

        for pnl in trades:
            scaled_pnl = pnl * position_size_pct
            current += scaled_pnl
            current = max(0.0, current)  # cannot go below zero
            equity.append(current)

        return equity

    @staticmethod
    def simulate_with_sizing(trades_pct: List[float], initial_capital: float,
                              position_size_pct: float = 0.02) -> List[float]:
        """
        Simulate with percentage-based position sizing.

        trades_pct: trade returns as percentages (e.g., 2.5 for 2.5% profit)
        position_size_pct: fraction of equity allocated per trade
        """
        equity = [initial_capital]
        current = initial_capital

        for ret_pct in trades_pct:
            risk_amount = current * position_size_pct
            pnl = risk_amount * (ret_pct / 100.0)
            current += pnl
            current = max(0.0, current)
            equity.append(current)

        return equity

    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, float]:
        """
        Calculate maximum drawdown from an equity curve.
        Returns (max_drawdown_amount, max_drawdown_percent).
        """
        if not equity_curve:
            return (0.0, 0.0)

        peak = equity_curve[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            dd_pct = (dd / peak * 100) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return (max_dd, max_dd_pct)

    @staticmethod
    def calculate_cagr(initial: float, final: float,
                        years: float) -> float:
        """Calculate compound annual growth rate."""
        if initial <= 0 or final <= 0 or years <= 0:
            return 0.0
        return ((final / initial) ** (1.0 / years) - 1.0) * 100

    @staticmethod
    def calculate_sharpe(returns: List[float],
                          risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from a series of returns."""
        if not returns or len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        excess = mean_ret - risk_free_rate
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = variance ** 0.5
        if std < 1e-10:
            return 0.0
        return excess / std


# ---------------------------------------------------------------------------
# Monte Carlo analyzer
# ---------------------------------------------------------------------------

class MonteCarloAnalyzer:
    """
    Comprehensive Monte Carlo analysis for backtested trading strategies.

    Performs:
    - Trade sequence resampling (shuffle trade order)
    - Bootstrap confidence intervals on returns
    - Probability of ruin calculation
    - Worst-case scenario analysis
    - Equity curve distribution
    - Optimal position sizing via simulation
    - Comparison with random strategy
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        self.config = config or MonteCarloConfig()
        self._resampler = TradeResampler()
        self._simulator = EquityCurveSimulator()

        if self.config.seed is not None:
            random.seed(self.config.seed)

    def analyze(self, trade_pnls: List[float],
                trade_returns_pct: Optional[List[float]] = None,
                initial_capital: Optional[float] = None) -> MonteCarloResult:
        """
        Run full Monte Carlo analysis on trade results.

        trade_pnls: list of PnL values for each trade
        trade_returns_pct: optional list of percentage returns per trade
        initial_capital: starting capital (overrides config)
        """
        capital = initial_capital or self.config.initial_capital

        if not trade_pnls:
            logger.warning("No trades provided for Monte Carlo analysis")
            return MonteCarloResult(config=self.config)

        logger.info(
            f"Running Monte Carlo analysis: {self.config.n_simulations} "
            f"simulations on {len(trade_pnls)} trades"
        )

        # 1. Simulate equity curves
        equity_curves = self._simulate_equity_curves(trade_pnls, capital)
        equity_stats = self._calculate_equity_stats(equity_curves, capital)

        # 2. Probability of ruin
        ruin = self._calculate_ruin_probability(trade_pnls, capital)

        # 3. Worst-case analysis
        worst_case = self._worst_case_analysis(equity_curves, capital)

        # 4. Trade distribution analysis
        trade_dist = self._analyze_trade_distribution(trade_pnls)

        # 5. Random strategy comparison
        random_comp = self._compare_with_random(
            trade_pnls, trade_returns_pct, capital
        )

        # 6. Optimal position sizing
        position_size = None
        if trade_returns_pct:
            position_size = self._optimize_position_size(
                trade_returns_pct, capital
            )

        # Store subset of equity curves for visualization
        max_curves = min(100, len(equity_curves))
        stored_curves = equity_curves[:max_curves]

        result = MonteCarloResult(
            config=self.config,
            equity_stats=equity_stats,
            ruin_analysis=ruin,
            position_size=position_size,
            worst_case=worst_case,
            random_comparison=random_comp,
            trade_distribution=trade_dist,
            simulated_equity_curves=stored_curves,
            metadata={
                "n_trades": len(trade_pnls),
                "n_simulations": self.config.n_simulations,
                "initial_capital": capital,
            },
        )

        logger.info(
            f"Monte Carlo complete: "
            f"mean_return={equity_stats.mean_return_pct:.2f}%, "
            f"ruin_prob={ruin.probability_of_ruin:.4f}, "
            f"median_dd={equity_stats.median_max_drawdown:.2f}%"
        )

        return result

    def _simulate_equity_curves(self, trade_pnls: List[float],
                                 capital: float) -> List[List[float]]:
        """Generate n_simulations equity curves from resampled trades."""
        curves = []
        n_trades = len(trade_pnls)
        max_trades = self.config.max_trades_per_sim or n_trades

        for _ in range(self.config.n_simulations):
            if self.config.preserve_streaks:
                resampled = self._resampler.block_bootstrap(
                    trade_pnls, max_trades, self.config.block_size
                )
            else:
                resampled = self._resampler.bootstrap(
                    trade_pnls, max_trades, self.config.use_replacement
                )
            curve = self._simulator.simulate(resampled, capital)
            curves.append(curve)

        return curves

    def _calculate_equity_stats(self, curves: List[List[float]],
                                 capital: float) -> EquityCurveStats:
        """Calculate statistics across all simulated equity curves."""
        stats = EquityCurveStats()

        if not curves:
            return stats

        final_equities = [c[-1] for c in curves]
        returns_pct = [(e - capital) / capital * 100 for e in final_equities]
        drawdowns = []
        for curve in curves:
            _, dd_pct = self._simulator.calculate_max_drawdown(curve)
            drawdowns.append(dd_pct)

        # Sort for percentile calculations
        sorted_equities = sorted(final_equities)
        sorted_returns = sorted(returns_pct)
        sorted_dd = sorted(drawdowns)

        n = len(sorted_equities)

        stats.mean_final_equity = sum(final_equities) / n
        stats.median_final_equity = sorted_equities[n // 2]
        variance = sum(
            (e - stats.mean_final_equity) ** 2 for e in final_equities
        ) / n
        stats.std_final_equity = variance ** 0.5
        stats.min_final_equity = sorted_equities[0]
        stats.max_final_equity = sorted_equities[-1]

        stats.mean_max_drawdown = sum(drawdowns) / n
        stats.median_max_drawdown = sorted_dd[n // 2]
        stats.worst_max_drawdown = sorted_dd[-1]

        stats.mean_return_pct = sum(returns_pct) / n
        stats.median_return_pct = sorted_returns[n // 2]
        stats.positive_return_pct = (
            sum(1 for r in returns_pct if r > 0) / n * 100
        )

        # Confidence intervals
        for level in self.config.confidence_levels:
            lower_idx = int((1 - level) / 2 * n)
            upper_idx = int((1 + level) / 2 * n) - 1
            lower_idx = max(0, min(lower_idx, n - 1))
            upper_idx = max(0, min(upper_idx, n - 1))
            stats.confidence_intervals[level] = (
                sorted_equities[lower_idx],
                sorted_equities[upper_idx],
            )

        return stats

    def _calculate_ruin_probability(self, trade_pnls: List[float],
                                     capital: float) -> RuinAnalysis:
        """Calculate the probability of account ruin."""
        ruin_level = capital * (1 - self.config.ruin_threshold)
        n_trades = len(trade_pnls)
        max_trades = self.config.max_trades_per_sim or n_trades

        ruin_count = 0
        trades_to_ruin = []

        for _ in range(self.config.n_simulations):
            resampled = self._resampler.bootstrap(
                trade_pnls, max_trades, self.config.use_replacement
            )
            equity = capital
            ruined = False

            for i, pnl in enumerate(resampled):
                equity += pnl
                if equity <= ruin_level:
                    ruined = True
                    trades_to_ruin.append(i + 1)
                    break

            if ruined:
                ruin_count += 1

        n_sims = self.config.n_simulations
        prob = ruin_count / n_sims

        # Wilson score confidence interval
        z = 1.96  # 95% CI
        p_hat = prob
        denominator = 1 + z * z / n_sims
        center = (p_hat + z * z / (2 * n_sims)) / denominator
        spread = z * math.sqrt(
            (p_hat * (1 - p_hat) + z * z / (4 * n_sims)) / n_sims
        ) / denominator
        ci_lower = max(0, center - spread)
        ci_upper = min(1, center + spread)

        avg_trades_to_ruin = 0.0
        median_trades_to_ruin = 0.0
        if trades_to_ruin:
            avg_trades_to_ruin = sum(trades_to_ruin) / len(trades_to_ruin)
            sorted_ttr = sorted(trades_to_ruin)
            median_trades_to_ruin = sorted_ttr[len(sorted_ttr) // 2]

        return RuinAnalysis(
            probability_of_ruin=prob,
            ruin_threshold_pct=self.config.ruin_threshold * 100,
            average_trades_to_ruin=avg_trades_to_ruin,
            median_trades_to_ruin=median_trades_to_ruin,
            ruin_simulations=ruin_count,
            total_simulations=n_sims,
            survival_rate=1.0 - prob,
            confidence_interval_95=(ci_lower, ci_upper),
        )

    def _worst_case_analysis(self, curves: List[List[float]],
                              capital: float) -> Dict[str, Any]:
        """Analyze worst-case scenarios across simulations."""
        if not curves:
            return {}

        # Find worst curves
        final_equities = [(i, c[-1]) for i, c in enumerate(curves)]
        final_equities.sort(key=lambda x: x[1])

        worst_idx = final_equities[0][0]
        worst_curve = curves[worst_idx]

        # Worst drawdown across all simulations
        all_drawdowns = []
        for curve in curves:
            _, dd_pct = self._simulator.calculate_max_drawdown(curve)
            all_drawdowns.append(dd_pct)

        sorted_dd = sorted(all_drawdowns, reverse=True)
        n = len(sorted_dd)

        # Consecutive losing streaks
        losing_streaks = []
        for curve in curves:
            streak = 0
            max_streak = 0
            for i in range(1, len(curve)):
                if curve[i] < curve[i - 1]:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            losing_streaks.append(max_streak)

        # Risk of losing X% in next Y trades
        loss_probabilities = {}
        for loss_pct in [5, 10, 20, 30, 50]:
            loss_count = sum(
                1 for c in curves
                if (capital - min(c)) / capital * 100 >= loss_pct
            )
            loss_probabilities[f"lose_{loss_pct}pct"] = loss_count / len(curves)

        return {
            "worst_final_equity": final_equities[0][1],
            "worst_return_pct": (final_equities[0][1] - capital) / capital * 100,
            "worst_1pct_equity": final_equities[max(0, int(n * 0.01))][1],
            "worst_5pct_equity": final_equities[max(0, int(n * 0.05))][1],
            "worst_drawdown_pct": sorted_dd[0],
            "p95_drawdown_pct": sorted_dd[max(0, int(n * 0.05))],
            "p99_drawdown_pct": sorted_dd[max(0, int(n * 0.01))],
            "avg_worst_streak": sum(losing_streaks) / max(1, len(losing_streaks)),
            "max_losing_streak": max(losing_streaks) if losing_streaks else 0,
            "loss_probabilities": loss_probabilities,
        }

    def _analyze_trade_distribution(self, trade_pnls: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of trade PnLs."""
        if not trade_pnls:
            return {}

        n = len(trade_pnls)
        sorted_pnls = sorted(trade_pnls)

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]

        mean = sum(trade_pnls) / n
        variance = sum((p - mean) ** 2 for p in trade_pnls) / n
        std = variance ** 0.5

        # Skewness
        skewness = 0.0
        if std > 0 and n >= 3:
            skewness = sum(((p - mean) / std) ** 3 for p in trade_pnls) * n / ((n - 1) * (n - 2))

        # Kurtosis (excess)
        kurtosis = 0.0
        if std > 0 and n >= 4:
            kurtosis = (
                sum(((p - mean) / std) ** 4 for p in trade_pnls) / n - 3
            )

        # Percentiles
        def percentile(data: List[float], pct: float) -> float:
            idx = int(len(data) * pct / 100)
            return data[max(0, min(idx, len(data) - 1))]

        return {
            "mean": mean,
            "median": sorted_pnls[n // 2],
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "min": sorted_pnls[0],
            "max": sorted_pnls[-1],
            "p1": percentile(sorted_pnls, 1),
            "p5": percentile(sorted_pnls, 5),
            "p25": percentile(sorted_pnls, 25),
            "p75": percentile(sorted_pnls, 75),
            "p95": percentile(sorted_pnls, 95),
            "p99": percentile(sorted_pnls, 99),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / n * 100 if n > 0 else 0,
            "avg_win": sum(wins) / max(1, len(wins)),
            "avg_loss": sum(losses) / max(1, len(losses)) if losses else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
            "profit_factor": (
                abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            ),
            "expectancy": mean,
        }

    def _compare_with_random(self, trade_pnls: List[float],
                              trade_returns_pct: Optional[List[float]],
                              capital: float) -> Dict[str, Any]:
        """
        Compare the strategy with a random entry strategy.

        Generates random trades with the same win rate and average
        win/loss sizes, then compares equity curves.
        """
        if not trade_pnls:
            return {}

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p <= 0]
        win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.5
        avg_win = sum(wins) / max(1, len(wins)) if wins else 0
        avg_loss = sum(losses) / max(1, len(losses)) if losses else 0
        n_trades = len(trade_pnls)

        # Simulate random strategy
        random_finals = []
        random_drawdowns = []

        for _ in range(self.config.n_simulations):
            random_trades = []
            for _ in range(n_trades):
                if random.random() < win_rate:
                    # Random win with same distribution shape
                    pnl = abs(random.gauss(avg_win, abs(avg_win) * 0.5))
                else:
                    pnl = -abs(random.gauss(abs(avg_loss), abs(avg_loss) * 0.5))
                random_trades.append(pnl)

            curve = self._simulator.simulate(random_trades, capital)
            random_finals.append(curve[-1])
            _, dd = self._simulator.calculate_max_drawdown(curve)
            random_drawdowns.append(dd)

        # Strategy metrics from resampled curves
        strategy_finals = []
        for _ in range(self.config.n_simulations):
            resampled = self._resampler.shuffle(trade_pnls)
            curve = self._simulator.simulate(resampled, capital)
            strategy_finals.append(curve[-1])

        n = len(random_finals)
        random_mean = sum(random_finals) / n
        strategy_mean = sum(strategy_finals) / n

        # What fraction of random simulations beat the strategy median
        strategy_median = sorted(strategy_finals)[n // 2]
        random_beats = sum(1 for r in random_finals if r >= strategy_median) / n

        return {
            "strategy_mean_equity": strategy_mean,
            "random_mean_equity": random_mean,
            "strategy_advantage": (
                (strategy_mean - random_mean) / random_mean * 100
                if random_mean > 0 else 0
            ),
            "random_beats_strategy_pct": random_beats * 100,
            "strategy_is_significant": random_beats < 0.05,
            "random_mean_drawdown": sum(random_drawdowns) / n,
        }

    def _optimize_position_size(self, trade_returns_pct: List[float],
                                  capital: float) -> PositionSizeResult:
        """
        Find optimal position size via Monte Carlo simulation.

        Tests multiple position sizes and finds the one that maximizes
        risk-adjusted returns.
        """
        result = PositionSizeResult()
        min_size, max_size = self.config.position_size_range
        steps = self.config.position_size_steps

        sizes = [
            min_size + (max_size - min_size) * i / max(1, steps - 1)
            for i in range(steps)
        ]

        best_metric = float('-inf')
        best_size = min_size

        for size in sizes:
            sim_returns = []
            sim_drawdowns = []
            sim_ruins = 0
            ruin_level = capital * (1 - self.config.ruin_threshold)

            n_sims = min(200, self.config.n_simulations)

            for _ in range(n_sims):
                resampled = self._resampler.bootstrap(
                    trade_returns_pct, len(trade_returns_pct),
                    self.config.use_replacement
                )
                curve = self._simulator.simulate_with_sizing(
                    resampled, capital, size
                )
                final = curve[-1]
                ret = (final - capital) / capital * 100
                _, dd = self._simulator.calculate_max_drawdown(curve)
                sim_returns.append(ret)
                sim_drawdowns.append(dd)
                if min(curve) <= ruin_level:
                    sim_ruins += 1

            avg_return = sum(sim_returns) / n_sims
            avg_dd = sum(sim_drawdowns) / n_sims
            ruin_rate = sim_ruins / n_sims

            # Risk-adjusted metric: return / drawdown, penalized by ruin
            if avg_dd > 0:
                metric = avg_return / avg_dd * (1 - ruin_rate * 2)
            else:
                metric = avg_return

            result.sizes_tested.append(size)
            result.metrics.append(metric)
            result.returns.append(avg_return)
            result.drawdowns.append(avg_dd)
            result.ruin_rates.append(ruin_rate)

            if metric > best_metric:
                best_metric = metric
                best_size = size

        result.optimal_size = best_size
        result.optimal_metric = best_metric

        # Calculate Kelly fraction
        result.kelly_fraction = self._calculate_kelly(trade_returns_pct)

        return result

    def _calculate_kelly(self, returns_pct: List[float]) -> float:
        """
        Calculate the Kelly criterion fraction.

        Kelly% = W - (1-W)/R
        where W = win probability, R = win/loss ratio
        """
        if not returns_pct:
            return 0.0

        wins = [r for r in returns_pct if r > 0]
        losses = [r for r in returns_pct if r <= 0]

        if not wins or not losses:
            return 0.0

        win_prob = len(wins) / len(returns_pct)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss <= 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly = win_prob - (1 - win_prob) / win_loss_ratio

        # Clamp to reasonable range
        return max(0.0, min(1.0, kelly))

    # -- Utility methods --

    def risk_of_loss(self, trade_pnls: List[float], capital: float,
                      loss_pct: float, n_trades: int) -> float:
        """
        Calculate probability of losing loss_pct% in the next n_trades trades.
        """
        loss_threshold = capital * loss_pct / 100
        loss_count = 0

        for _ in range(self.config.n_simulations):
            resampled = self._resampler.bootstrap(
                trade_pnls, n_trades, self.config.use_replacement
            )
            equity = capital
            min_equity = capital
            for pnl in resampled:
                equity += pnl
                min_equity = min(min_equity, equity)

            if capital - min_equity >= loss_threshold:
                loss_count += 1

        return loss_count / self.config.n_simulations

    def confidence_band(self, trade_pnls: List[float], capital: float,
                         confidence: float = 0.95,
                         n_periods: int = 0) -> List[Tuple[float, float]]:
        """
        Calculate confidence bands for the equity curve.

        Returns list of (lower, upper) tuples for each trade position.
        """
        n_trades = n_periods or len(trade_pnls)
        curves = self._simulate_equity_curves(trade_pnls, capital)

        bands = []
        max_len = max(len(c) for c in curves)

        for i in range(min(n_trades + 1, max_len)):
            values = [c[i] for c in curves if i < len(c)]
            if not values:
                break
            values.sort()
            n = len(values)
            lower_idx = int((1 - confidence) / 2 * n)
            upper_idx = int((1 + confidence) / 2 * n) - 1
            lower_idx = max(0, min(lower_idx, n - 1))
            upper_idx = max(0, min(upper_idx, n - 1))
            bands.append((values[lower_idx], values[upper_idx]))

        return bands
