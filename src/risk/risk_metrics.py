"""
Mefai Autotrade - Comprehensive Risk Metrics Calculator

Production-grade risk metrics:
- Value at Risk (Historical, Parametric, Cornish-Fisher)
- Expected Shortfall (CVaR)
- Maximum drawdown and recovery time
- Sharpe, Sortino, Calmar ratios
- Omega ratio
- Profit factor, win rate, expectancy
- RAROC, Information ratio, Tail ratio
- Ulcer index, Recovery factor
"""

import logging
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class VaRMethod(str, Enum):
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    CORNISH_FISHER = "cornish_fisher"


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculation."""
    # VaR settings
    var_confidence: float = 0.95  # 95% confidence
    var_holding_period_days: int = 1
    var_lookback_days: int = 252  # 1 year of trading days

    # Annualization
    trading_days_per_year: int = 365  # crypto trades every day
    periods_per_day: int = 1  # 1 if daily returns, 24 if hourly

    # Risk-free rate (annualized)
    risk_free_rate: float = 0.05  # 5% annual

    # Benchmark returns (for information ratio)
    benchmark_returns: List[float] = field(default_factory=list)

    # Minimum observations
    min_observations: int = 20

    # MAR (minimum acceptable return) for Sortino - annualized
    mar: float = 0.0


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    method: VaRMethod
    confidence: float
    var_amount: float
    var_pct: float
    holding_period_days: int
    observation_count: int


@dataclass
class DrawdownInfo:
    """Drawdown analysis result."""
    max_drawdown_pct: float
    max_drawdown_amount: float
    max_drawdown_duration_days: float
    recovery_time_days: float
    current_drawdown_pct: float
    drawdown_periods: List[Dict]  # start, end, depth, duration


@dataclass
class RiskReport:
    """Comprehensive risk metrics report."""
    timestamp: datetime
    observation_count: int
    total_return_pct: float
    annualized_return_pct: float

    # VaR
    var_historical: VaRResult
    var_parametric: VaRResult
    var_cornish_fisher: VaRResult
    expected_shortfall: float  # CVaR

    # Drawdown
    drawdown: DrawdownInfo

    # Ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    tail_ratio: float

    # Trade statistics
    profit_factor: float
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float
    expectancy: float
    expectancy_pct: float

    # Other
    volatility_annual: float
    downside_deviation: float
    raroc: float
    ulcer_index: float
    recovery_factor: float

    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk Metrics Calculator
# ---------------------------------------------------------------------------

class RiskMetrics:
    """
    Comprehensive risk metrics calculator.

    Calculates all standard portfolio and trading risk metrics
    from return series and trade records.
    """

    def __init__(self, config: Optional[RiskMetricsConfig] = None):
        self.config = config or RiskMetricsConfig()
        self._returns: List[float] = []
        self._equity_curve: List[float] = []
        self._trade_pnls: List[float] = []
        self._trade_pnl_pcts: List[float] = []

    # ------------------------------------------------------------------
    # Data input
    # ------------------------------------------------------------------

    def add_return(self, return_pct: float) -> None:
        """Add a period return (as decimal, e.g., 0.01 for 1%)."""
        self._returns.append(return_pct)

    def add_returns(self, returns: List[float]) -> None:
        """Add multiple returns at once."""
        self._returns.extend(returns)

    def add_equity(self, equity: float) -> None:
        """Add an equity observation and calculate return."""
        self._equity_curve.append(equity)
        if len(self._equity_curve) >= 2:
            prev = self._equity_curve[-2]
            if prev > 0:
                ret = (equity - prev) / prev
                self._returns.append(ret)

    def add_equity_curve(self, equities: List[float]) -> None:
        """Add a full equity curve and derive returns."""
        self._equity_curve = list(equities)
        self._returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                ret = (equities[i] - equities[i - 1]) / equities[i - 1]
                self._returns.append(ret)

    def add_trade_result(self, pnl: float, pnl_pct: float) -> None:
        """Add a trade PnL result."""
        self._trade_pnls.append(pnl)
        self._trade_pnl_pcts.append(pnl_pct)

    def add_trade_results(
        self,
        pnls: List[float],
        pnl_pcts: List[float],
    ) -> None:
        """Add multiple trade results at once."""
        self._trade_pnls.extend(pnls)
        self._trade_pnl_pcts.extend(pnl_pcts)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def calculate_report(
        self,
        portfolio_value: Optional[float] = None,
    ) -> RiskReport:
        """
        Calculate comprehensive risk report.

        Args:
            portfolio_value: Current portfolio value (for VaR amounts)

        Returns:
            RiskReport with all metrics
        """
        warnings = []
        n = len(self._returns)

        if n < self.config.min_observations:
            warnings.append(
                f"Insufficient data: {n} observations "
                f"(minimum {self.config.min_observations})"
            )

        portfolio_val = portfolio_value or (
            self._equity_curve[-1] if self._equity_curve else 10000.0
        )

        # Basic statistics
        total_return = self._total_return()
        annual_return = self._annualized_return()
        volatility = self._annualized_volatility()
        downside_dev = self._downside_deviation()

        # VaR
        var_hist = self.calculate_var(VaRMethod.HISTORICAL, portfolio_val)
        var_param = self.calculate_var(VaRMethod.PARAMETRIC, portfolio_val)
        var_cf = self.calculate_var(VaRMethod.CORNISH_FISHER, portfolio_val)
        cvar = self.calculate_expected_shortfall(portfolio_val)

        # Drawdown
        dd_info = self.calculate_drawdown()

        # Ratios
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        calmar = self.calculate_calmar_ratio()
        omega = self.calculate_omega_ratio()
        info_ratio = self.calculate_information_ratio()
        tail = self.calculate_tail_ratio()

        # Trade metrics
        pf = self.calculate_profit_factor()
        wr = self.calculate_win_rate()
        avg_w, avg_l = self._avg_win_loss()
        payoff = self._payoff_ratio()
        exp, exp_pct = self.calculate_expectancy()

        # Other
        raroc = self.calculate_raroc(portfolio_val)
        ulcer = self.calculate_ulcer_index()
        recovery = self.calculate_recovery_factor()

        return RiskReport(
            timestamp=datetime.now(timezone.utc),
            observation_count=n,
            total_return_pct=total_return * 100,
            annualized_return_pct=annual_return * 100,
            var_historical=var_hist,
            var_parametric=var_param,
            var_cornish_fisher=var_cf,
            expected_shortfall=cvar,
            drawdown=dd_info,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            information_ratio=info_ratio,
            tail_ratio=tail,
            profit_factor=pf,
            win_rate=wr,
            avg_win=avg_w,
            avg_loss=avg_l,
            payoff_ratio=payoff,
            expectancy=exp,
            expectancy_pct=exp_pct,
            volatility_annual=volatility * 100,
            downside_deviation=downside_dev * 100,
            raroc=raroc,
            ulcer_index=ulcer,
            recovery_factor=recovery,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # VaR calculations
    # ------------------------------------------------------------------

    def calculate_var(
        self,
        method: VaRMethod = VaRMethod.HISTORICAL,
        portfolio_value: float = 10000.0,
    ) -> VaRResult:
        """
        Calculate Value at Risk.

        VaR is the maximum expected loss at a given confidence level
        over a specified holding period.
        """
        n = len(self._returns)
        confidence = self.config.var_confidence

        if n < 2:
            return VaRResult(
                method=method,
                confidence=confidence,
                var_amount=0.0,
                var_pct=0.0,
                holding_period_days=self.config.var_holding_period_days,
                observation_count=n,
            )

        if method == VaRMethod.HISTORICAL:
            var_pct = self._var_historical(confidence)
        elif method == VaRMethod.PARAMETRIC:
            var_pct = self._var_parametric(confidence)
        elif method == VaRMethod.CORNISH_FISHER:
            var_pct = self._var_cornish_fisher(confidence)
        else:
            var_pct = self._var_historical(confidence)

        # Scale to holding period (sqrt-time rule)
        if self.config.var_holding_period_days > 1:
            var_pct *= math.sqrt(self.config.var_holding_period_days)

        var_amount = portfolio_value * abs(var_pct)

        return VaRResult(
            method=method,
            confidence=confidence,
            var_amount=var_amount,
            var_pct=abs(var_pct) * 100,
            holding_period_days=self.config.var_holding_period_days,
            observation_count=n,
        )

    def _var_historical(self, confidence: float) -> float:
        """Historical VaR - uses actual return distribution percentile."""
        sorted_returns = sorted(self._returns)
        index = int((1 - confidence) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))
        return sorted_returns[index]

    def _var_parametric(self, confidence: float) -> float:
        """Parametric VaR - assumes normal distribution."""
        mean = sum(self._returns) / len(self._returns)
        variance = sum((r - mean) ** 2 for r in self._returns) / len(self._returns)
        std = math.sqrt(variance)

        # Z-score for confidence level
        z = self._norm_ppf(1 - confidence)
        return mean + z * std

    def _var_cornish_fisher(self, confidence: float) -> float:
        """
        Cornish-Fisher VaR - adjusts for skewness and kurtosis.

        Uses the Cornish-Fisher expansion to modify the z-score
        based on the actual distribution shape, providing a more
        accurate VaR for non-normal distributions.
        """
        n = len(self._returns)
        if n < 4:
            return self._var_parametric(confidence)

        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / n
        std = math.sqrt(variance) if variance > 0 else 0.001

        # Skewness
        skew = sum((r - mean) ** 3 for r in self._returns) / (n * std ** 3)

        # Excess kurtosis
        kurt = sum((r - mean) ** 4 for r in self._returns) / (n * std ** 4) - 3.0

        # Standard normal quantile
        z = self._norm_ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z ** 2 - 1) * skew / 6
            + (z ** 3 - 3 * z) * kurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
        )

        return mean + z_cf * std

    def calculate_expected_shortfall(
        self,
        portfolio_value: float = 10000.0,
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR / CVaR).

        ES is the average of all losses beyond the VaR threshold.
        It captures tail risk better than VaR.
        """
        n = len(self._returns)
        if n < 2:
            return 0.0

        sorted_returns = sorted(self._returns)
        cutoff = int((1 - self.config.var_confidence) * n)
        cutoff = max(1, cutoff)

        tail_returns = sorted_returns[:cutoff]
        avg_tail = sum(tail_returns) / len(tail_returns)

        return abs(avg_tail) * portfolio_value

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    def calculate_drawdown(self) -> DrawdownInfo:
        """Calculate comprehensive drawdown metrics."""
        if not self._equity_curve and not self._returns:
            return DrawdownInfo(
                max_drawdown_pct=0.0,
                max_drawdown_amount=0.0,
                max_drawdown_duration_days=0.0,
                recovery_time_days=0.0,
                current_drawdown_pct=0.0,
                drawdown_periods=[],
            )

        # Build equity curve from returns if not provided
        if self._equity_curve:
            equity = list(self._equity_curve)
        else:
            equity = [10000.0]
            for r in self._returns:
                equity.append(equity[-1] * (1 + r))

        # Calculate drawdown series
        peak = equity[0]
        drawdowns = []
        dd_periods = []
        current_dd_start = None
        max_dd = 0.0
        max_dd_amount = 0.0

        periods_per_day = max(self.config.periods_per_day, 1)

        for i, eq in enumerate(equity):
            if eq > peak:
                peak = eq
                if current_dd_start is not None:
                    # Drawdown recovered
                    dd_depth = max(drawdowns[current_dd_start:i]) if current_dd_start < i else 0
                    dd_periods.append({
                        "start_index": current_dd_start,
                        "end_index": i,
                        "depth_pct": dd_depth * 100,
                        "duration_days": (i - current_dd_start) / periods_per_day,
                        "recovered": True,
                    })
                    current_dd_start = None

            dd = (peak - eq) / peak if peak > 0 else 0
            drawdowns.append(dd)

            if dd > 0 and current_dd_start is None:
                current_dd_start = i

            if dd > max_dd:
                max_dd = dd
                max_dd_amount = peak - eq

        # Current drawdown
        current_dd = drawdowns[-1] if drawdowns else 0

        # If still in drawdown
        if current_dd_start is not None:
            dd_depth = max(drawdowns[current_dd_start:])
            dd_periods.append({
                "start_index": current_dd_start,
                "end_index": len(equity) - 1,
                "depth_pct": dd_depth * 100,
                "duration_days": (len(equity) - 1 - current_dd_start) / periods_per_day,
                "recovered": False,
            })

        # Max drawdown duration and recovery time
        max_duration = 0.0
        max_recovery = 0.0
        for period in dd_periods:
            if period["duration_days"] > max_duration:
                max_duration = period["duration_days"]
            if period["recovered"]:
                max_recovery = max(max_recovery, period["duration_days"])

        return DrawdownInfo(
            max_drawdown_pct=max_dd * 100,
            max_drawdown_amount=max_dd_amount,
            max_drawdown_duration_days=max_duration,
            recovery_time_days=max_recovery,
            current_drawdown_pct=current_dd * 100,
            drawdown_periods=dd_periods[-20:],  # last 20 drawdown periods
        )

    # ------------------------------------------------------------------
    # Performance ratios
    # ------------------------------------------------------------------

    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (Return - RiskFree) / Volatility

        Measures excess return per unit of total risk.
        """
        n = len(self._returns)
        if n < 2:
            return 0.0

        annual_return = self._annualized_return()
        annual_vol = self._annualized_volatility()

        if annual_vol <= 0:
            return 0.0

        return (annual_return - self.config.risk_free_rate) / annual_vol

    def calculate_sortino_ratio(self) -> float:
        """
        Calculate annualized Sortino ratio.

        Sortino = (Return - MAR) / Downside Deviation

        Like Sharpe but only penalizes downside volatility.
        Better for asymmetric return distributions.
        """
        n = len(self._returns)
        if n < 2:
            return 0.0

        annual_return = self._annualized_return()
        dd = self._downside_deviation()

        if dd <= 0:
            return 0.0

        # Annualize downside deviation
        annualization = math.sqrt(
            self.config.trading_days_per_year * self.config.periods_per_day
        )
        annual_dd = dd * annualization

        if annual_dd <= 0:
            return 0.0

        mar_per_period = self.config.mar / (
            self.config.trading_days_per_year * self.config.periods_per_day
        )

        return (annual_return - self.config.mar) / annual_dd

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.

        Calmar = Annualized Return / Max Drawdown

        Measures return relative to worst-case drawdown.
        """
        dd_info = self.calculate_drawdown()
        if dd_info.max_drawdown_pct <= 0:
            return 0.0

        annual_return = self._annualized_return()
        return annual_return / (dd_info.max_drawdown_pct / 100)

    def calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Omega = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0))

        Measures the probability-weighted ratio of gains versus losses
        relative to a threshold return. Unlike Sharpe, considers the
        entire distribution, not just mean and variance.
        """
        n = len(self._returns)
        if n < 2:
            return 0.0

        gains = sum(max(r - threshold, 0) for r in self._returns)
        losses = sum(max(threshold - r, 0) for r in self._returns)

        if losses <= 0:
            return float("inf") if gains > 0 else 0.0

        return gains / losses

    def calculate_information_ratio(self) -> float:
        """
        Calculate Information ratio.

        IR = (Portfolio Return - Benchmark Return) / Tracking Error

        Measures risk-adjusted active return versus a benchmark.
        """
        benchmark = self.config.benchmark_returns
        if not benchmark or len(benchmark) != len(self._returns):
            return 0.0

        n = len(self._returns)
        if n < 2:
            return 0.0

        active_returns = [
            self._returns[i] - benchmark[i] for i in range(n)
        ]

        mean_active = sum(active_returns) / n
        tracking_error_sq = sum(
            (ar - mean_active) ** 2 for ar in active_returns
        ) / n

        if tracking_error_sq <= 0:
            return 0.0

        tracking_error = math.sqrt(tracking_error_sq)
        annualization = math.sqrt(
            self.config.trading_days_per_year * self.config.periods_per_day
        )

        return (mean_active * annualization) / (tracking_error * annualization)

    def calculate_tail_ratio(self) -> float:
        """
        Calculate Tail ratio.

        Tail Ratio = |95th percentile| / |5th percentile|

        Measures the relative size of the right tail to the left tail.
        Values > 1 indicate fatter right tail (more large gains than losses).
        """
        n = len(self._returns)
        if n < 20:
            return 0.0

        sorted_returns = sorted(self._returns)
        idx_5 = max(0, int(0.05 * n))
        idx_95 = min(n - 1, int(0.95 * n))

        left_tail = abs(sorted_returns[idx_5])
        right_tail = abs(sorted_returns[idx_95])

        if left_tail <= 0:
            return float("inf") if right_tail > 0 else 1.0

        return right_tail / left_tail

    # ------------------------------------------------------------------
    # Trade statistics
    # ------------------------------------------------------------------

    def calculate_profit_factor(self) -> float:
        """
        Calculate Profit Factor.

        PF = Gross Profits / Gross Losses

        PF > 1 means profitable system. PF > 2 is considered good.
        """
        if not self._trade_pnls:
            return 0.0

        gross_profit = sum(p for p in self._trade_pnls if p > 0)
        gross_loss = sum(abs(p) for p in self._trade_pnls if p < 0)

        if gross_loss <= 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_win_rate(self) -> float:
        """Calculate win rate as a decimal (0-1)."""
        if not self._trade_pnls:
            return 0.0

        wins = sum(1 for p in self._trade_pnls if p > 0)
        return wins / len(self._trade_pnls)

    def calculate_expectancy(self) -> Tuple[float, float]:
        """
        Calculate per-trade expectancy.

        E = (Win% * Avg Win) - (Loss% * Avg Loss)

        Returns (expectancy_amount, expectancy_pct)
        """
        if not self._trade_pnls:
            return 0.0, 0.0

        wins = [p for p in self._trade_pnls if p > 0]
        losses = [p for p in self._trade_pnls if p <= 0]

        n = len(self._trade_pnls)
        win_rate = len(wins) / n
        loss_rate = len(losses) / n

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(abs(l) for l in losses) / len(losses) if losses else 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Percentage-based expectancy
        if self._trade_pnl_pcts:
            wins_pct = [p for p in self._trade_pnl_pcts if p > 0]
            losses_pct = [p for p in self._trade_pnl_pcts if p <= 0]
            avg_win_pct = sum(wins_pct) / len(wins_pct) if wins_pct else 0
            avg_loss_pct = sum(abs(l) for l in losses_pct) / len(losses_pct) if losses_pct else 0
            expectancy_pct = (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct)
        else:
            expectancy_pct = 0.0

        return expectancy, expectancy_pct

    # ------------------------------------------------------------------
    # Other risk metrics
    # ------------------------------------------------------------------

    def calculate_raroc(self, portfolio_value: float = 10000.0) -> float:
        """
        Calculate Risk-Adjusted Return on Capital.

        RAROC = Expected Return / Economic Capital (VaR)

        Higher RAROC means better risk-adjusted performance.
        """
        var_result = self.calculate_var(VaRMethod.HISTORICAL, portfolio_value)
        if var_result.var_amount <= 0:
            return 0.0

        annual_return = self._annualized_return()
        expected_return_amount = portfolio_value * annual_return

        return expected_return_amount / var_result.var_amount

    def calculate_ulcer_index(self) -> float:
        """
        Calculate Ulcer Index.

        UI = sqrt(mean of squared drawdowns)

        Measures the depth and duration of drawdowns.
        Lower is better. Unlike standard deviation, only penalizes
        negative performance.
        """
        if not self._equity_curve and not self._returns:
            return 0.0

        # Build equity curve
        if self._equity_curve:
            equity = list(self._equity_curve)
        else:
            equity = [10000.0]
            for r in self._returns:
                equity.append(equity[-1] * (1 + r))

        peak = equity[0]
        squared_drawdowns = []

        for eq in equity:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak * 100) if peak > 0 else 0
            squared_drawdowns.append(dd ** 2)

        if not squared_drawdowns:
            return 0.0

        mean_sq_dd = sum(squared_drawdowns) / len(squared_drawdowns)
        return math.sqrt(mean_sq_dd)

    def calculate_recovery_factor(self) -> float:
        """
        Calculate Recovery Factor.

        RF = Total Net Profit / Max Drawdown

        Measures how well the system recovers from drawdowns.
        Higher is better. RF > 3 is considered good.
        """
        if not self._returns:
            return 0.0

        total_return = self._total_return()
        dd_info = self.calculate_drawdown()

        if dd_info.max_drawdown_pct <= 0:
            return float("inf") if total_return > 0 else 0.0

        return (total_return * 100) / dd_info.max_drawdown_pct

    # ------------------------------------------------------------------
    # Helper calculations
    # ------------------------------------------------------------------

    def _total_return(self) -> float:
        """Calculate total cumulative return (as decimal)."""
        if not self._returns:
            return 0.0

        cumulative = 1.0
        for r in self._returns:
            cumulative *= (1 + r)

        return cumulative - 1.0

    def _annualized_return(self) -> float:
        """Calculate annualized return."""
        n = len(self._returns)
        if n < 2:
            return 0.0

        total = self._total_return()
        periods_per_year = (
            self.config.trading_days_per_year * self.config.periods_per_day
        )
        years = n / periods_per_year

        if years <= 0:
            return 0.0

        # Compound annual growth rate
        if total <= -1:
            return -1.0  # total loss

        return (1 + total) ** (1 / years) - 1

    def _annualized_volatility(self) -> float:
        """Calculate annualized volatility (standard deviation of returns)."""
        n = len(self._returns)
        if n < 2:
            return 0.0

        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / (n - 1)
        std = math.sqrt(variance)

        annualization = math.sqrt(
            self.config.trading_days_per_year * self.config.periods_per_day
        )
        return std * annualization

    def _downside_deviation(self) -> float:
        """
        Calculate downside deviation (semi-deviation below MAR).

        Only considers returns below the minimum acceptable return.
        """
        n = len(self._returns)
        if n < 2:
            return 0.0

        mar_per_period = self.config.mar / (
            self.config.trading_days_per_year * self.config.periods_per_day
        )

        downside_diffs = [
            min(r - mar_per_period, 0) ** 2 for r in self._returns
        ]

        return math.sqrt(sum(downside_diffs) / n)

    def _avg_win_loss(self) -> Tuple[float, float]:
        """Calculate average win and average loss."""
        if not self._trade_pnls:
            return 0.0, 0.0

        wins = [p for p in self._trade_pnls if p > 0]
        losses = [abs(p) for p in self._trade_pnls if p < 0]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return avg_win, avg_loss

    def _payoff_ratio(self) -> float:
        """Calculate payoff ratio (avg win / avg loss)."""
        avg_win, avg_loss = self._avg_win_loss()
        if avg_loss <= 0:
            return float("inf") if avg_win > 0 else 0.0
        return avg_win / avg_loss

    def _norm_ppf(self, p: float) -> float:
        """
        Inverse normal CDF (percent point function).

        Rational approximation for the inverse of the standard normal CDF.
        Accuracy to approximately 4.5 decimal places.
        """
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0

        if p < 0.5:
            return -self._norm_ppf(1 - p)

        # Rational approximation (Abramowitz and Stegun)
        t = math.sqrt(-2 * math.log(1 - p))

        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        result = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
        return result

    # ------------------------------------------------------------------
    # Public utilities
    # ------------------------------------------------------------------

    def get_return_statistics(self) -> Dict:
        """Get basic return statistics."""
        n = len(self._returns)
        if n == 0:
            return {"count": 0}

        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / n if n > 1 else 0
        std = math.sqrt(variance)
        sorted_r = sorted(self._returns)

        return {
            "count": n,
            "mean": mean,
            "std": std,
            "min": sorted_r[0],
            "max": sorted_r[-1],
            "median": sorted_r[n // 2],
            "skewness": self._skewness(),
            "kurtosis": self._kurtosis(),
            "total_return_pct": self._total_return() * 100,
            "annualized_return_pct": self._annualized_return() * 100,
            "annualized_volatility_pct": self._annualized_volatility() * 100,
        }

    def _skewness(self) -> float:
        """Calculate return distribution skewness."""
        n = len(self._returns)
        if n < 3:
            return 0.0

        mean = sum(self._returns) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in self._returns) / n)
        if std <= 0:
            return 0.0

        return sum((r - mean) ** 3 for r in self._returns) / (n * std ** 3)

    def _kurtosis(self) -> float:
        """Calculate return distribution excess kurtosis."""
        n = len(self._returns)
        if n < 4:
            return 0.0

        mean = sum(self._returns) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in self._returns) / n)
        if std <= 0:
            return 0.0

        return sum((r - mean) ** 4 for r in self._returns) / (n * std ** 4) - 3.0

    def clear(self) -> None:
        """Clear all data."""
        self._returns.clear()
        self._equity_curve.clear()
        self._trade_pnls.clear()
        self._trade_pnl_pcts.clear()

    def to_dict(self) -> Dict:
        """Serialize metrics state."""
        return {
            "return_count": len(self._returns),
            "trade_count": len(self._trade_pnls),
            "equity_points": len(self._equity_curve),
            "statistics": self.get_return_statistics(),
            "config": {
                "var_confidence": self.config.var_confidence,
                "trading_days_per_year": self.config.trading_days_per_year,
                "risk_free_rate": self.config.risk_free_rate,
            },
        }
