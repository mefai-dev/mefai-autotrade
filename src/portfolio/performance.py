"""
Mefai Signal Engine - Portfolio Performance Analytics

Portfolio-level analytics including return calculation (time-weighted and
money-weighted), attribution analysis, factor analysis, regime analysis,
benchmark correlation, risk contribution, and diversification benefit
quantification.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("mefai.autotrade.portfolio.performance")


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class MarketRegime(str, Enum):
    """Detected market regime."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"


class ReturnMethod(str, Enum):
    """Return calculation methodology."""
    TIME_WEIGHTED = "time_weighted"      # TWRR - not affected by cash flows
    MONEY_WEIGHTED = "money_weighted"    # MWRR - IRR considering cash flows


@dataclass
class AttributionResult:
    """Result of performance attribution analysis."""
    strategy_name: str
    absolute_return: float = 0.0        # raw return from this strategy
    contribution: float = 0.0           # weighted contribution to portfolio return
    contribution_pct: float = 0.0       # percentage of total portfolio return
    allocation_effect: float = 0.0      # return from allocation decision
    selection_effect: float = 0.0       # return from strategy selection
    interaction_effect: float = 0.0     # interaction between allocation and selection
    average_weight: float = 0.0
    active_return: float = 0.0          # return above benchmark


@dataclass
class FactorExposure:
    """Factor exposure analysis result."""
    factor_name: str
    beta: float = 0.0                   # sensitivity to factor
    alpha: float = 0.0                  # excess return unexplained by factor
    r_squared: float = 0.0             # how much variance is explained
    t_statistic: float = 0.0
    p_value: float = 1.0
    correlation: float = 0.0


@dataclass
class RegimePerformance:
    """Performance metrics within a specific market regime."""
    regime: MarketRegime
    num_periods: int = 0
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0


@dataclass
class RiskContribution:
    """Risk contribution of a single strategy to the portfolio."""
    strategy_name: str
    weight: float = 0.0
    marginal_risk: float = 0.0          # marginal contribution to portfolio vol
    component_risk: float = 0.0         # absolute risk contribution
    pct_risk: float = 0.0              # percentage of total portfolio risk
    standalone_vol: float = 0.0
    correlation_with_portfolio: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state for analytics."""
    timestamp: float
    total_value: float
    cash_flow: float = 0.0              # external cash added/withdrawn
    weights: Dict[str, float] = field(default_factory=dict)
    strategy_returns: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Portfolio Performance Engine
# ---------------------------------------------------------------------------

class PortfolioPerformance:
    """
    Comprehensive portfolio analytics engine. Tracks portfolio snapshots
    over time and computes returns, attribution, factor analysis, regime
    performance, and risk decomposition.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        benchmark_returns: Optional[np.ndarray] = None,
        benchmark_name: str = "BTC",
    ):
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.benchmark_name = benchmark_name

        self._snapshots: List[PortfolioSnapshot] = []
        self._strategy_names: List[str] = []

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def add_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Add a point-in-time portfolio snapshot."""
        self._snapshots.append(snapshot)
        for name in snapshot.weights:
            if name not in self._strategy_names:
                self._strategy_names.append(name)

    def add_snapshots(self, snapshots: List[PortfolioSnapshot]) -> None:
        """Add multiple snapshots at once."""
        for s in snapshots:
            self.add_snapshot(s)

    # ------------------------------------------------------------------
    # Return calculations
    # ------------------------------------------------------------------

    def calculate_returns(
        self,
        method: ReturnMethod = ReturnMethod.TIME_WEIGHTED,
    ) -> Dict[str, Any]:
        """
        Calculate portfolio returns using the specified method.
        Returns a dict with total_return, annualized_return, and period_returns.
        """
        if len(self._snapshots) < 2:
            return {"total_return": 0.0, "annualized_return": 0.0, "period_returns": []}

        if method == ReturnMethod.TIME_WEIGHTED:
            return self._time_weighted_return()
        else:
            return self._money_weighted_return()

    def _time_weighted_return(self) -> Dict[str, Any]:
        """
        Time-Weighted Rate of Return (TWRR).
        Eliminates the effect of external cash flows by compounding
        sub-period returns between cash flow events.
        """
        sub_period_returns = []
        values = []

        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]

            # Adjust for cash flow at the start of the period
            adjusted_start = prev.total_value + curr.cash_flow
            if adjusted_start > 0:
                period_return = (curr.total_value / adjusted_start) - 1.0
            else:
                period_return = 0.0

            sub_period_returns.append(period_return)
            values.append(curr.total_value)

        # Compound sub-period returns
        total_return = 1.0
        for r in sub_period_returns:
            total_return *= (1.0 + r)
        total_return -= 1.0

        # Annualize
        if len(self._snapshots) >= 2:
            days = (self._snapshots[-1].timestamp - self._snapshots[0].timestamp) / 86400
            if days > 0:
                annualized = (1.0 + total_return) ** (365.0 / days) - 1.0
            else:
                annualized = 0.0
        else:
            annualized = 0.0

        return {
            "total_return": total_return,
            "annualized_return": annualized,
            "period_returns": sub_period_returns,
            "method": "time_weighted",
        }

    def _money_weighted_return(self) -> Dict[str, Any]:
        """
        Money-Weighted Rate of Return (MWRR / IRR).
        Accounts for the timing and size of cash flows.
        Uses Newton's method to find the rate r that satisfies:
        sum of PV(cash_flows) = 0
        """
        if len(self._snapshots) < 2:
            return {"total_return": 0.0, "annualized_return": 0.0, "period_returns": []}

        start = self._snapshots[0]
        end = self._snapshots[-1]
        total_days = (end.timestamp - start.timestamp) / 86400
        if total_days <= 0:
            return {"total_return": 0.0, "annualized_return": 0.0, "period_returns": []}

        # Cash flows: initial investment (negative), intermediate flows, final value (positive)
        cash_flows = [(-start.total_value, 0.0)]  # (amount, days_from_start)

        for snap in self._snapshots[1:-1]:
            if abs(snap.cash_flow) > 0.01:
                days = (snap.timestamp - start.timestamp) / 86400
                cash_flows.append((-snap.cash_flow, days))

        cash_flows.append((end.total_value, total_days))

        # Newton's method to find IRR (daily rate)
        r = 0.0001  # initial guess
        for iteration in range(100):
            npv = sum(cf / (1 + r) ** (d / 365) for cf, d in cash_flows)
            dnpv = sum(-cf * (d / 365) / (1 + r) ** (d / 365 + 1) for cf, d in cash_flows)

            if abs(dnpv) < 1e-12:
                break
            r_new = r - npv / dnpv
            if abs(r_new - r) < 1e-10:
                r = r_new
                break
            r = r_new

        annualized = r
        total_return = (1 + annualized) ** (total_days / 365) - 1

        return {
            "total_return": total_return,
            "annualized_return": annualized,
            "period_returns": [],
            "method": "money_weighted",
        }

    # ------------------------------------------------------------------
    # Attribution analysis
    # ------------------------------------------------------------------

    def attribution_analysis(
        self,
        benchmark_weights: Optional[Dict[str, float]] = None,
    ) -> List[AttributionResult]:
        """
        Brinson-Fachler attribution: decomposes portfolio return into
        allocation effect, selection effect, and interaction effect for
        each strategy.
        """
        if len(self._snapshots) < 2:
            return []

        # Equal-weight benchmark if none provided
        if benchmark_weights is None:
            n = len(self._strategy_names)
            benchmark_weights = {name: 1.0 / n for name in self._strategy_names}

        results = []
        for name in self._strategy_names:
            # Average weight and return across all periods
            weights_list = []
            returns_list = []
            for snap in self._snapshots:
                w = snap.weights.get(name, 0.0)
                r = snap.strategy_returns.get(name, 0.0)
                weights_list.append(w)
                returns_list.append(r)

            avg_weight = float(np.mean(weights_list)) if weights_list else 0.0
            total_strat_return = float(np.sum(returns_list)) if returns_list else 0.0
            avg_strat_return = float(np.mean(returns_list)) if returns_list else 0.0

            # Benchmark weight and return
            bench_weight = benchmark_weights.get(name, 0.0)
            # Assume benchmark return = equal-weighted average of all strategies
            all_returns = []
            for snap in self._snapshots:
                all_returns.extend(snap.strategy_returns.values())
            bench_return = float(np.mean(all_returns)) if all_returns else 0.0

            # Brinson-Fachler decomposition
            allocation_effect = (avg_weight - bench_weight) * (bench_return - bench_return)
            selection_effect = bench_weight * (avg_strat_return - bench_return)
            interaction_effect = (avg_weight - bench_weight) * (avg_strat_return - bench_return)

            contribution = avg_weight * total_strat_return

            result = AttributionResult(
                strategy_name=name,
                absolute_return=total_strat_return,
                contribution=contribution,
                allocation_effect=allocation_effect,
                selection_effect=selection_effect,
                interaction_effect=interaction_effect,
                average_weight=avg_weight,
            )

            # Contribution percentage
            total_port_return = sum(
                snap.weights.get(n, 0.0) * snap.strategy_returns.get(n, 0.0)
                for snap in self._snapshots
                for n in self._strategy_names
            )
            if abs(total_port_return) > 1e-10:
                result.contribution_pct = contribution / total_port_return * 100

            results.append(result)

        # Sort by absolute contribution (highest first)
        results.sort(key=lambda x: abs(x.contribution), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Factor analysis
    # ------------------------------------------------------------------

    def factor_analysis(
        self,
        factors: Dict[str, np.ndarray],
    ) -> List[FactorExposure]:
        """
        Multi-factor regression analysis. Regresses portfolio returns against
        provided factor returns to determine exposures and alpha.

        factors: dict of factor_name -> array of factor returns (same length as snapshots)
        """
        port_returns = self._get_portfolio_returns()
        if len(port_returns) < 10:
            logger.warning("Insufficient data for factor analysis (%d periods)", len(port_returns))
            return []

        results = []
        for factor_name, factor_returns in factors.items():
            # Align lengths
            min_len = min(len(port_returns), len(factor_returns))
            if min_len < 5:
                continue

            y = port_returns[-min_len:]
            x = factor_returns[-min_len:]

            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
            t_stat = slope / std_err if std_err > 0 else 0.0

            exposure = FactorExposure(
                factor_name=factor_name,
                beta=float(slope),
                alpha=float(intercept) * 252,  # annualized
                r_squared=float(r_value ** 2),
                t_statistic=float(t_stat),
                p_value=float(p_value),
                correlation=float(r_value),
            )
            results.append(exposure)

        # Sort by R-squared (most explanatory first)
        results.sort(key=lambda x: x.r_squared, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Regime analysis
    # ------------------------------------------------------------------

    def regime_analysis(
        self,
        market_returns: Optional[np.ndarray] = None,
    ) -> List[RegimePerformance]:
        """
        Analyze portfolio performance across different market regimes.
        Detects regimes from market returns and calculates strategy
        performance within each regime.
        """
        port_returns = self._get_portfolio_returns()
        if len(port_returns) < 20:
            return []

        if market_returns is None:
            market_returns = self.benchmark_returns
        if market_returns is None:
            market_returns = port_returns  # use portfolio as its own benchmark

        min_len = min(len(port_returns), len(market_returns))
        port_returns = port_returns[-min_len:]
        market_returns = market_returns[-min_len:]

        # Classify each period into a regime
        regimes = self._classify_regimes(market_returns)

        # Group portfolio returns by regime
        regime_results = {}
        for regime in MarketRegime:
            indices = [i for i, r in enumerate(regimes) if r == regime]
            if not indices:
                continue

            regime_returns = port_returns[indices]
            n = len(regime_returns)
            total = float(np.sum(regime_returns))
            mean_ret = float(np.mean(regime_returns))
            vol = float(np.std(regime_returns)) * np.sqrt(252) if n > 1 else 0.0
            ann_ret = mean_ret * 252

            # Max drawdown within regime
            cum = np.cumsum(regime_returns)
            peak = np.maximum.accumulate(cum)
            dd = peak - cum
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

            win_count = int(np.sum(regime_returns > 0))
            win_rate = win_count / n if n > 0 else 0.0
            sharpe = ann_ret / vol if vol > 0 else 0.0

            regime_results[regime] = RegimePerformance(
                regime=regime,
                num_periods=n,
                total_return=total,
                annualized_return=ann_ret,
                volatility=vol,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                avg_return=mean_ret,
            )

        return sorted(regime_results.values(), key=lambda x: x.num_periods, reverse=True)

    def _classify_regimes(self, returns: np.ndarray) -> List[MarketRegime]:
        """Classify each period into a market regime based on return and volatility."""
        n = len(returns)
        regimes = []
        lookback = 20

        for i in range(n):
            start = max(0, i - lookback)
            window = returns[start:i + 1]

            if len(window) < 5:
                regimes.append(MarketRegime.SIDEWAYS)
                continue

            cum_return = float(np.sum(window))
            vol = float(np.std(window))
            mean = float(np.mean(window))

            # Crash: sharp negative return
            if returns[i] < -0.05:  # > 5% drop in one period
                regimes.append(MarketRegime.CRASH)
            elif cum_return > 0.10 and mean > 0:
                if vol > np.percentile(np.abs(returns[:i + 1]), 75) if i > 5 else True:
                    regimes.append(MarketRegime.HIGH_VOLATILITY)
                else:
                    regimes.append(MarketRegime.BULL)
            elif cum_return < -0.10 and mean < 0:
                regimes.append(MarketRegime.BEAR)
            elif cum_return > 0.03 and i > 0 and np.sum(returns[max(0, i - lookback * 2):start]) < -0.05:
                regimes.append(MarketRegime.RECOVERY)
            elif vol < np.percentile(np.abs(returns[:i + 1]), 25) if i > 5 else False:
                regimes.append(MarketRegime.LOW_VOLATILITY)
            else:
                regimes.append(MarketRegime.SIDEWAYS)

        return regimes

    # ------------------------------------------------------------------
    # Benchmark correlation
    # ------------------------------------------------------------------

    def benchmark_correlation(
        self,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate correlation and tracking metrics against a benchmark.
        """
        port_returns = self._get_portfolio_returns()
        bench = benchmark_returns if benchmark_returns is not None else self.benchmark_returns

        if bench is None or len(port_returns) < 5:
            return {
                "correlation": 0.0,
                "beta": 0.0,
                "alpha": 0.0,
                "tracking_error": 0.0,
                "information_ratio": 0.0,
            }

        min_len = min(len(port_returns), len(bench))
        p = port_returns[-min_len:]
        b = bench[-min_len:]

        correlation = float(np.corrcoef(p, b)[0, 1])

        # Beta and alpha
        slope, intercept, _, _, _ = scipy_stats.linregress(b, p)
        beta = float(slope)
        alpha = float(intercept) * 252  # annualized

        # Tracking error
        excess = p - b
        tracking_error = float(np.std(excess)) * np.sqrt(252)

        # Information ratio
        info_ratio = float(np.mean(excess)) * 252 / tracking_error if tracking_error > 0 else 0.0

        return {
            "correlation": correlation,
            "beta": beta,
            "alpha": alpha,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "benchmark": self.benchmark_name,
        }

    # ------------------------------------------------------------------
    # Risk contribution
    # ------------------------------------------------------------------

    def risk_contribution(
        self,
        current_weights: Dict[str, float],
        strategy_returns: Dict[str, np.ndarray],
    ) -> List[RiskContribution]:
        """
        Decompose portfolio risk into per-strategy contributions.
        Uses the Euler decomposition: risk_i = w_i * (Cov * w)_i / port_vol
        """
        names = sorted(current_weights.keys())
        n = len(names)
        if n == 0:
            return []

        weights = np.array([current_weights[name] for name in names])

        # Build returns matrix
        min_len = min(len(strategy_returns.get(name, [])) for name in names)
        if min_len < 2:
            return []

        returns_matrix = np.zeros((n, min_len))
        for i, name in enumerate(names):
            returns_matrix[i, :] = strategy_returns[name][-min_len:]

        cov = np.cov(returns_matrix, rowvar=True)
        port_vol = np.sqrt(float(weights @ cov @ weights))

        if port_vol < 1e-10:
            return []

        marginal = cov @ weights
        component = weights * marginal
        total_component = component.sum()

        # Portfolio returns for correlation
        port_returns = returns_matrix.T @ weights

        results = []
        for i, name in enumerate(names):
            standalone_vol = float(np.std(returns_matrix[i, :])) * np.sqrt(252)
            corr = float(np.corrcoef(returns_matrix[i, :], port_returns)[0, 1])

            results.append(RiskContribution(
                strategy_name=name,
                weight=float(weights[i]),
                marginal_risk=float(marginal[i]) / port_vol,
                component_risk=float(component[i]) / port_vol,
                pct_risk=float(component[i]) / total_component * 100 if total_component > 0 else 0,
                standalone_vol=standalone_vol,
                correlation_with_portfolio=corr,
            ))

        results.sort(key=lambda x: x.pct_risk, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Diversification benefit
    # ------------------------------------------------------------------

    def diversification_benefit(
        self,
        current_weights: Dict[str, float],
        strategy_returns: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Quantify the diversification benefit of the current portfolio.
        Compares actual portfolio volatility to the weighted average of
        individual strategy volatilities.
        """
        names = sorted(current_weights.keys())
        n = len(names)
        if n < 2:
            return {"diversification_ratio": 1.0, "volatility_reduction_pct": 0.0}

        weights = np.array([current_weights[name] for name in names])

        min_len = min(len(strategy_returns.get(name, [])) for name in names)
        if min_len < 2:
            return {"diversification_ratio": 1.0, "volatility_reduction_pct": 0.0}

        returns_matrix = np.zeros((n, min_len))
        for i, name in enumerate(names):
            returns_matrix[i, :] = strategy_returns[name][-min_len:]

        # Individual volatilities
        individual_vols = np.array([float(np.std(returns_matrix[i, :])) for i in range(n)])
        weighted_vol_sum = float(weights @ individual_vols)

        # Portfolio volatility (with correlations)
        cov = np.cov(returns_matrix, rowvar=True)
        port_vol = np.sqrt(float(weights @ cov @ weights))

        div_ratio = weighted_vol_sum / port_vol if port_vol > 0 else 1.0
        vol_reduction = ((weighted_vol_sum - port_vol) / weighted_vol_sum * 100) if weighted_vol_sum > 0 else 0.0

        # Correlation matrix
        corr_matrix = np.corrcoef(returns_matrix)
        avg_corr = float((np.sum(corr_matrix) - n) / (n * (n - 1))) if n > 1 else 0.0

        # Effective number of strategies (based on PCA)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        eff_n = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)) if len(eigenvalues) > 0 else 1.0

        return {
            "diversification_ratio": div_ratio,
            "volatility_reduction_pct": vol_reduction,
            "portfolio_vol_annualized": port_vol * np.sqrt(252),
            "undiversified_vol_annualized": weighted_vol_sum * np.sqrt(252),
            "average_correlation": avg_corr,
            "effective_strategies": eff_n,
            "actual_strategies": n,
            "correlation_matrix": {
                names[i]: {names[j]: float(corr_matrix[i, j]) for j in range(n)}
                for i in range(n)
            },
        }

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Generate a comprehensive performance summary."""
        port_returns = self._get_portfolio_returns()
        if len(port_returns) == 0:
            return {"status": "insufficient_data", "snapshots": len(self._snapshots)}

        total_return_data = self.calculate_returns(ReturnMethod.TIME_WEIGHTED)

        # Risk metrics
        vol = float(np.std(port_returns)) * np.sqrt(252)
        sharpe = (total_return_data["annualized_return"] - self.risk_free_rate) / vol if vol > 0 else 0.0

        # Sortino ratio (downside deviation)
        negative_returns = port_returns[port_returns < 0]
        downside_dev = float(np.std(negative_returns)) * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        sortino = (total_return_data["annualized_return"] - self.risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

        # Calmar ratio (return / max drawdown)
        cum = np.cumsum(port_returns)
        peak = np.maximum.accumulate(cum)
        drawdowns = peak - cum
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        calmar = total_return_data["annualized_return"] / max_dd if max_dd > 0 else 0.0

        # Win rate
        win_count = int(np.sum(port_returns > 0))
        total_periods = len(port_returns)
        win_rate = win_count / total_periods if total_periods > 0 else 0.0

        # Best/worst periods
        best = float(np.max(port_returns))
        worst = float(np.min(port_returns))

        # Skewness and kurtosis
        skew = float(scipy_stats.skew(port_returns))
        kurt = float(scipy_stats.kurtosis(port_returns))

        # VaR and CVaR (95%)
        var_95 = float(np.percentile(port_returns, 5))
        cvar_95 = float(np.mean(port_returns[port_returns <= var_95])) if np.any(port_returns <= var_95) else var_95

        return {
            "total_return": total_return_data["total_return"],
            "annualized_return": total_return_data["annualized_return"],
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "best_period": best,
            "worst_period": worst,
            "skewness": skew,
            "kurtosis": kurt,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "total_periods": total_periods,
            "strategies": len(self._strategy_names),
            "snapshots": len(self._snapshots),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_portfolio_returns(self) -> np.ndarray:
        """Extract portfolio-level returns from snapshots."""
        if len(self._snapshots) < 2:
            return np.array([])

        returns = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i - 1]
            curr = self._snapshots[i]
            adjusted_start = prev.total_value + curr.cash_flow
            if adjusted_start > 0:
                ret = (curr.total_value / adjusted_start) - 1.0
            else:
                ret = 0.0
            returns.append(ret)

        return np.array(returns)

    def reset(self) -> None:
        """Clear all snapshots and cached data."""
        self._snapshots = []
        self._strategy_names = []
