# Mefai Signal Engine - Backtest Reporting
# Performance summary, trade analysis, drawdown analysis, risk metrics,
# benchmark comparison, and export to JSON/CSV.

import json
import csv
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from io import StringIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSummary:
    """Comprehensive performance summary metrics."""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown: float = 0.0
    avg_drawdown_duration_days: float = 0.0
    recovery_factor: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0

    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0

    # Benchmark
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # Time
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    bars_processed: int = 0
    execution_time: float = 0.0


@dataclass
class DrawdownPeriod:
    """A single drawdown period."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    recovery: Optional[datetime] = None
    depth: float = 0.0
    depth_pct: float = 0.0
    duration_days: float = 0.0
    recovery_days: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0


@dataclass
class MonthlyReturn:
    """Monthly return data."""
    year: int = 0
    month: int = 0
    return_pct: float = 0.0
    trades: int = 0
    pnl: float = 0.0


@dataclass
class RollingMetric:
    """Rolling metric data point."""
    timestamp: Optional[datetime] = None
    value: float = 0.0
    window: int = 0


# ---------------------------------------------------------------------------
# Metric calculators
# ---------------------------------------------------------------------------

class MetricCalculator:
    """Calculate various performance metrics from equity and trade data."""

    @staticmethod
    def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0,
                      periods_per_year: float = 252.0) -> float:
        """Annualized Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        excess = mean - risk_free_rate / periods_per_year
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = variance ** 0.5
        if std < 1e-10:
            return 0.0
        return (excess / std) * math.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(returns: List[float], risk_free_rate: float = 0.0,
                       periods_per_year: float = 252.0) -> float:
        """Annualized Sortino ratio (uses downside deviation)."""
        if not returns or len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        excess = mean - risk_free_rate / periods_per_year
        downside = [r for r in returns if r < 0]
        if not downside:
            return 0.0 if excess <= 0 else float('inf')
        dd_var = sum(r ** 2 for r in downside) / len(returns)
        dd_std = dd_var ** 0.5
        if dd_std < 1e-10:
            return 0.0
        return (excess / dd_std) * math.sqrt(periods_per_year)

    @staticmethod
    def calmar_ratio(total_return_pct: float,
                      max_drawdown_pct: float) -> float:
        """Calmar ratio (annualized return / max drawdown)."""
        if abs(max_drawdown_pct) < 1e-10:
            return 0.0
        return total_return_pct / abs(max_drawdown_pct)

    @staticmethod
    def omega_ratio(returns: List[float], threshold: float = 0.0) -> float:
        """Omega ratio (probability weighted gain/loss)."""
        if not returns:
            return 0.0
        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)
        if losses < 1e-10:
            return float('inf') if gains > 0 else 0.0
        return gains / losses

    @staticmethod
    def tail_ratio(returns: List[float]) -> float:
        """Tail ratio (95th percentile / abs(5th percentile))."""
        if len(returns) < 20:
            return 0.0
        sorted_r = sorted(returns)
        n = len(sorted_r)
        p95 = sorted_r[int(n * 0.95)]
        p5 = sorted_r[int(n * 0.05)]
        if abs(p5) < 1e-10:
            return 0.0
        return abs(p95 / p5)

    @staticmethod
    def var(returns: List[float], confidence: float = 0.95) -> float:
        """Value at Risk at given confidence level."""
        if not returns:
            return 0.0
        sorted_r = sorted(returns)
        idx = int(len(sorted_r) * (1 - confidence))
        return sorted_r[max(0, idx)]

    @staticmethod
    def cvar(returns: List[float], confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)."""
        if not returns:
            return 0.0
        sorted_r = sorted(returns)
        idx = int(len(sorted_r) * (1 - confidence))
        tail = sorted_r[:max(1, idx)]
        return sum(tail) / len(tail)

    @staticmethod
    def cagr(initial: float, final: float, years: float) -> float:
        """Compound Annual Growth Rate."""
        if initial <= 0 or final <= 0 or years <= 0:
            return 0.0
        return ((final / initial) ** (1.0 / years) - 1.0) * 100

    @staticmethod
    def max_consecutive(values: List[bool]) -> int:
        """Maximum consecutive True values."""
        max_count = 0
        current = 0
        for v in values:
            if v:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count

    @staticmethod
    def profit_factor(winning_pnl: float, losing_pnl: float) -> float:
        """Profit factor (gross profit / gross loss)."""
        if abs(losing_pnl) < 1e-10:
            return float('inf') if winning_pnl > 0 else 0.0
        return abs(winning_pnl / losing_pnl)

    @staticmethod
    def recovery_factor(total_return: float,
                         max_drawdown: float) -> float:
        """Recovery factor (net profit / max drawdown)."""
        if abs(max_drawdown) < 1e-10:
            return 0.0
        return total_return / abs(max_drawdown)


# ---------------------------------------------------------------------------
# Drawdown analyzer
# ---------------------------------------------------------------------------

class DrawdownAnalyzer:
    """Analyze drawdown periods from equity curve data."""

    @staticmethod
    def analyze(equity_curve: List[Any]) -> List[DrawdownPeriod]:
        """
        Extract all drawdown periods from an equity curve.

        equity_curve: list of EquityPoint objects or dicts with
                      timestamp and equity fields.
        """
        if not equity_curve:
            return []

        periods = []
        peak = 0.0
        peak_time = None
        in_drawdown = False
        current_trough = float('inf')
        current_trough_time = None
        drawdown_start = None

        for point in equity_curve:
            if hasattr(point, 'equity'):
                equity = point.equity
                ts = point.timestamp
            elif isinstance(point, dict):
                equity = point.get("equity", 0)
                ts = point.get("timestamp")
            else:
                continue

            if equity > peak:
                if in_drawdown:
                    # Drawdown ended - record it
                    dd = DrawdownPeriod(
                        start=drawdown_start,
                        end=current_trough_time,
                        recovery=ts,
                        depth=peak - current_trough,
                        depth_pct=(peak - current_trough) / peak * 100 if peak > 0 else 0,
                        peak_equity=peak,
                        trough_equity=current_trough,
                    )
                    if drawdown_start and ts:
                        dd.duration_days = (ts - drawdown_start).total_seconds() / 86400
                    if current_trough_time and ts:
                        dd.recovery_days = (ts - current_trough_time).total_seconds() / 86400
                    if dd.depth_pct > 0.01:  # filter trivial drawdowns
                        periods.append(dd)

                    in_drawdown = False
                    current_trough = float('inf')

                peak = equity
                peak_time = ts
            elif equity < peak:
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = peak_time
                    current_trough = equity
                    current_trough_time = ts
                elif equity < current_trough:
                    current_trough = equity
                    current_trough_time = ts

        # Handle ongoing drawdown at end
        if in_drawdown:
            dd = DrawdownPeriod(
                start=drawdown_start,
                end=current_trough_time,
                depth=peak - current_trough,
                depth_pct=(peak - current_trough) / peak * 100 if peak > 0 else 0,
                peak_equity=peak,
                trough_equity=current_trough,
            )
            if drawdown_start and current_trough_time:
                dd.duration_days = (current_trough_time - drawdown_start).total_seconds() / 86400
            if dd.depth_pct > 0.01:
                periods.append(dd)

        return sorted(periods, key=lambda d: d.depth_pct, reverse=True)


# ---------------------------------------------------------------------------
# Monthly returns
# ---------------------------------------------------------------------------

class MonthlyReturnsCalculator:
    """Calculate monthly returns from equity curve or trade data."""

    @staticmethod
    def from_equity_curve(equity_curve: List[Any]) -> List[MonthlyReturn]:
        """Calculate monthly returns from equity curve data."""
        if not equity_curve:
            return []

        monthly: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        for point in equity_curve:
            if hasattr(point, 'equity'):
                equity = point.equity
                ts = point.timestamp
            elif isinstance(point, dict):
                equity = point.get("equity", 0)
                ts = point.get("timestamp")
            else:
                continue

            if ts:
                monthly[(ts.year, ts.month)].append(equity)

        results = []
        sorted_months = sorted(monthly.keys())

        prev_end = None
        for year, month in sorted_months:
            equities = monthly[(year, month)]
            start_eq = equities[0] if prev_end is None else prev_end
            end_eq = equities[-1]
            ret = (end_eq - start_eq) / start_eq * 100 if start_eq > 0 else 0
            results.append(MonthlyReturn(
                year=year, month=month, return_pct=ret, pnl=end_eq - start_eq,
            ))
            prev_end = end_eq

        return results

    @staticmethod
    def from_trades(trades: List[Any]) -> List[MonthlyReturn]:
        """Calculate monthly returns from trade data."""
        monthly: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        monthly_count: Dict[Tuple[int, int], int] = defaultdict(int)

        for trade in trades:
            if hasattr(trade, 'exit_time') and trade.exit_time:
                ts = trade.exit_time
                pnl = trade.pnl
            elif isinstance(trade, dict):
                ts = trade.get("exit_time")
                pnl = trade.get("pnl", 0)
            else:
                continue

            if ts:
                monthly[(ts.year, ts.month)].append(pnl)
                monthly_count[(ts.year, ts.month)] += 1

        results = []
        for (year, month) in sorted(monthly.keys()):
            pnl_sum = sum(monthly[(year, month)])
            results.append(MonthlyReturn(
                year=year, month=month,
                return_pct=0,  # needs initial capital to calculate
                pnl=pnl_sum,
                trades=monthly_count[(year, month)],
            ))

        return results

    @staticmethod
    def to_table(monthly_returns: List[MonthlyReturn]) -> Dict[int, Dict[int, float]]:
        """Convert to a year -> month -> return table."""
        table: Dict[int, Dict[int, float]] = defaultdict(dict)
        for mr in monthly_returns:
            table[mr.year][mr.month] = mr.return_pct
        return dict(table)


# ---------------------------------------------------------------------------
# PnL by time analysis
# ---------------------------------------------------------------------------

class TimeAnalysis:
    """Analyze PnL by day of week, hour of day, etc."""

    @staticmethod
    def pnl_by_day_of_week(trades: List[Any]) -> Dict[str, Dict[str, float]]:
        """PnL breakdown by day of week."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
        by_day: Dict[int, List[float]] = defaultdict(list)

        for trade in trades:
            ts = getattr(trade, 'entry_time', None) or \
                 (trade.get('entry_time') if isinstance(trade, dict) else None)
            pnl = getattr(trade, 'pnl', 0) if hasattr(trade, 'pnl') else \
                  trade.get('pnl', 0) if isinstance(trade, dict) else 0

            if ts:
                by_day[ts.weekday()].append(pnl)

        result = {}
        for dow in range(7):
            pnls = by_day.get(dow, [])
            if pnls:
                result[days[dow]] = {
                    "total_pnl": sum(pnls),
                    "avg_pnl": sum(pnls) / len(pnls),
                    "trades": len(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                }
            else:
                result[days[dow]] = {
                    "total_pnl": 0, "avg_pnl": 0, "trades": 0, "win_rate": 0,
                }
        return result

    @staticmethod
    def pnl_by_hour(trades: List[Any]) -> Dict[int, Dict[str, float]]:
        """PnL breakdown by hour of day (UTC)."""
        by_hour: Dict[int, List[float]] = defaultdict(list)

        for trade in trades:
            ts = getattr(trade, 'entry_time', None) or \
                 (trade.get('entry_time') if isinstance(trade, dict) else None)
            pnl = getattr(trade, 'pnl', 0) if hasattr(trade, 'pnl') else \
                  trade.get('pnl', 0) if isinstance(trade, dict) else 0

            if ts:
                by_hour[ts.hour].append(pnl)

        result = {}
        for hour in range(24):
            pnls = by_hour.get(hour, [])
            if pnls:
                result[hour] = {
                    "total_pnl": sum(pnls),
                    "avg_pnl": sum(pnls) / len(pnls),
                    "trades": len(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                }
            else:
                result[hour] = {
                    "total_pnl": 0, "avg_pnl": 0, "trades": 0, "win_rate": 0,
                }
        return result

    @staticmethod
    def pnl_by_month(trades: List[Any]) -> Dict[int, Dict[str, float]]:
        """PnL breakdown by calendar month."""
        by_month: Dict[int, List[float]] = defaultdict(list)

        for trade in trades:
            ts = getattr(trade, 'entry_time', None) or \
                 (trade.get('entry_time') if isinstance(trade, dict) else None)
            pnl = getattr(trade, 'pnl', 0) if hasattr(trade, 'pnl') else \
                  trade.get('pnl', 0) if isinstance(trade, dict) else 0

            if ts:
                by_month[ts.month].append(pnl)

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        result = {}
        for m in range(1, 13):
            pnls = by_month.get(m, [])
            if pnls:
                result[m] = {
                    "month_name": months[m - 1],
                    "total_pnl": sum(pnls),
                    "avg_pnl": sum(pnls) / len(pnls),
                    "trades": len(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                }
            else:
                result[m] = {
                    "month_name": months[m - 1],
                    "total_pnl": 0, "avg_pnl": 0, "trades": 0, "win_rate": 0,
                }
        return result


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

class RollingMetricsCalculator:
    """Calculate rolling performance metrics."""

    @staticmethod
    def rolling_sharpe(returns: List[Tuple[datetime, float]],
                        window: int = 60,
                        periods_per_year: float = 252.0) -> List[RollingMetric]:
        """Calculate rolling Sharpe ratio."""
        if len(returns) < window:
            return []

        results = []
        for i in range(window, len(returns) + 1):
            window_returns = [r[1] for r in returns[i - window:i]]
            sharpe = MetricCalculator.sharpe_ratio(
                window_returns, periods_per_year=periods_per_year
            )
            results.append(RollingMetric(
                timestamp=returns[i - 1][0],
                value=sharpe,
                window=window,
            ))
        return results

    @staticmethod
    def rolling_win_rate(trades: List[Any], window: int = 50) -> List[RollingMetric]:
        """Calculate rolling win rate over a trade window."""
        if len(trades) < window:
            return []

        results = []
        for i in range(window, len(trades) + 1):
            window_trades = trades[i - window:i]
            wins = sum(
                1 for t in window_trades
                if (getattr(t, 'pnl', 0) if hasattr(t, 'pnl') else
                    t.get('pnl', 0) if isinstance(t, dict) else 0) > 0
            )
            ts = getattr(window_trades[-1], 'exit_time', None)
            results.append(RollingMetric(
                timestamp=ts,
                value=wins / window * 100,
                window=window,
            ))
        return results

    @staticmethod
    def rolling_return(equity_curve: List[Any],
                        window: int = 60) -> List[RollingMetric]:
        """Calculate rolling return percentage."""
        if len(equity_curve) < window:
            return []

        results = []
        for i in range(window, len(equity_curve)):
            if hasattr(equity_curve[i - window], 'equity'):
                start_eq = equity_curve[i - window].equity
                end_eq = equity_curve[i].equity
                ts = equity_curve[i].timestamp
            elif isinstance(equity_curve[i], dict):
                start_eq = equity_curve[i - window].get("equity", 0)
                end_eq = equity_curve[i].get("equity", 0)
                ts = equity_curve[i].get("timestamp")
            else:
                continue

            ret = (end_eq - start_eq) / start_eq * 100 if start_eq > 0 else 0
            results.append(RollingMetric(
                timestamp=ts, value=ret, window=window,
            ))
        return results

    @staticmethod
    def rolling_volatility(returns: List[Tuple[datetime, float]],
                            window: int = 30,
                            annualize: bool = True,
                            periods_per_year: float = 252.0) -> List[RollingMetric]:
        """Calculate rolling volatility."""
        if len(returns) < window:
            return []

        results = []
        for i in range(window, len(returns) + 1):
            window_returns = [r[1] for r in returns[i - window:i]]
            mean = sum(window_returns) / len(window_returns)
            variance = sum(
                (r - mean) ** 2 for r in window_returns
            ) / (len(window_returns) - 1)
            vol = variance ** 0.5
            if annualize:
                vol *= math.sqrt(periods_per_year)
            results.append(RollingMetric(
                timestamp=returns[i - 1][0],
                value=vol * 100,
                window=window,
            ))
        return results


# ---------------------------------------------------------------------------
# Main report class
# ---------------------------------------------------------------------------

class BacktestReport:
    """
    Comprehensive backtest reporting engine.

    Generates detailed performance analysis from backtest results including
    performance summary, monthly returns, trade analysis, drawdowns,
    risk metrics, time analysis, rolling metrics, and benchmark comparison.

    Usage:
        from src.backtesting.report import BacktestReport

        report = BacktestReport(backtest_result)
        summary = report.generate_summary()
        report.export_json("report.json")
        report.export_csv("trades.csv")
        report.print_summary()
    """

    def __init__(self, result: Any, risk_free_rate: float = 0.0):
        self.result = result
        self.risk_free_rate = risk_free_rate
        self._summary: Optional[PerformanceSummary] = None
        self._drawdowns: Optional[List[DrawdownPeriod]] = None
        self._monthly_returns: Optional[List[MonthlyReturn]] = None

    def generate_summary(self) -> PerformanceSummary:
        """Generate a comprehensive performance summary."""
        if self._summary is not None:
            return self._summary

        r = self.result
        summary = PerformanceSummary()

        # Time range
        if r.equity_curve:
            first = r.equity_curve[0]
            last = r.equity_curve[-1]
            summary.start_date = first.timestamp if hasattr(first, 'timestamp') else None
            summary.end_date = last.timestamp if hasattr(last, 'timestamp') else None
            if summary.start_date and summary.end_date:
                delta = summary.end_date - summary.start_date
                summary.trading_days = max(1, delta.days)

        summary.bars_processed = r.total_bars
        summary.execution_time = r.execution_time_seconds

        # Returns
        initial = r.config.initial_capital if r.config else 10000
        summary.total_return = r.total_return
        summary.total_return_pct = r.total_return_percent
        years = summary.trading_days / 365.25 if summary.trading_days > 0 else 1
        summary.cagr = MetricCalculator.cagr(initial, r.final_equity, years)
        summary.annualized_return = summary.cagr

        # Calculate periodic returns from equity curve
        returns = self._calculate_returns()

        # Risk metrics
        summary.sharpe_ratio = MetricCalculator.sharpe_ratio(
            returns, self.risk_free_rate
        )
        summary.sortino_ratio = MetricCalculator.sortino_ratio(
            returns, self.risk_free_rate
        )
        summary.calmar_ratio = MetricCalculator.calmar_ratio(
            summary.cagr, r.max_drawdown_percent
        )
        summary.omega_ratio = MetricCalculator.omega_ratio(returns)
        summary.tail_ratio = MetricCalculator.tail_ratio(returns)
        summary.var_95 = MetricCalculator.var(returns, 0.95)
        summary.var_99 = MetricCalculator.var(returns, 0.99)
        summary.cvar_95 = MetricCalculator.cvar(returns, 0.95)
        summary.cvar_99 = MetricCalculator.cvar(returns, 0.99)

        if returns:
            mean_r = sum(returns) / len(returns)
            variance = sum((x - mean_r) ** 2 for x in returns) / max(1, len(returns) - 1)
            summary.volatility = (variance ** 0.5) * math.sqrt(252) * 100
            downside = [r for r in returns if r < 0]
            if downside:
                dd_var = sum(r ** 2 for r in downside) / len(returns)
                summary.downside_deviation = (dd_var ** 0.5) * math.sqrt(252) * 100

        # Drawdown
        summary.max_drawdown = r.max_drawdown
        summary.max_drawdown_pct = r.max_drawdown_percent
        drawdowns = self.get_drawdowns()
        if drawdowns:
            summary.avg_drawdown = sum(d.depth_pct for d in drawdowns) / len(drawdowns)
            summary.max_drawdown_duration_days = max(
                d.duration_days for d in drawdowns
            ) if drawdowns else 0
            durations = [d.duration_days for d in drawdowns if d.duration_days > 0]
            summary.avg_drawdown_duration_days = (
                sum(durations) / len(durations) if durations else 0
            )
        summary.recovery_factor = MetricCalculator.recovery_factor(
            r.total_return, r.max_drawdown
        )

        # Trade statistics
        trades = r.trades
        summary.total_trades = len(trades)
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        summary.winning_trades = len(winners)
        summary.losing_trades = len(losers)
        summary.win_rate = (
            len(winners) / len(trades) * 100 if trades else 0
        )

        if trades:
            pnls = [t.pnl for t in trades]
            summary.avg_trade_pnl = sum(pnls) / len(pnls)
        if winners:
            win_pnls = [t.pnl for t in winners]
            summary.avg_winning_trade = sum(win_pnls) / len(win_pnls)
            summary.largest_win = max(win_pnls)
        if losers:
            loss_pnls = [t.pnl for t in losers]
            summary.avg_losing_trade = sum(loss_pnls) / len(loss_pnls)
            summary.largest_loss = min(loss_pnls)

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = sum(t.pnl for t in losers)
        summary.profit_factor = MetricCalculator.profit_factor(
            gross_profit, gross_loss
        )

        # Win/loss streaks
        if trades:
            win_loss = [t.pnl > 0 for t in trades]
            summary.max_consecutive_wins = MetricCalculator.max_consecutive(win_loss)
            summary.max_consecutive_losses = MetricCalculator.max_consecutive(
                [not w for w in win_loss]
            )

        # Holding period
        if trades:
            durations = [
                t.bars_held for t in trades if t.bars_held > 0
            ]
            summary.avg_holding_period = (
                sum(durations) / len(durations) if durations else 0
            )

        # Expectancy and payoff ratio
        if trades:
            summary.expectancy = summary.avg_trade_pnl
        if summary.avg_losing_trade != 0:
            summary.payoff_ratio = abs(
                summary.avg_winning_trade / summary.avg_losing_trade
            )

        # Costs
        summary.total_commission = r.metadata.get("total_commission", 0)
        summary.total_slippage = r.metadata.get("total_slippage", 0)
        summary.total_funding = r.metadata.get("total_funding", 0)

        self._summary = summary
        return summary

    def get_drawdowns(self, top_n: int = 0) -> List[DrawdownPeriod]:
        """Get drawdown periods sorted by depth."""
        if self._drawdowns is None:
            self._drawdowns = DrawdownAnalyzer.analyze(self.result.equity_curve)
        if top_n > 0:
            return self._drawdowns[:top_n]
        return self._drawdowns

    def get_monthly_returns(self) -> List[MonthlyReturn]:
        """Get monthly returns."""
        if self._monthly_returns is None:
            self._monthly_returns = MonthlyReturnsCalculator.from_equity_curve(
                self.result.equity_curve
            )
        return self._monthly_returns

    def get_monthly_returns_table(self) -> Dict[int, Dict[int, float]]:
        """Get monthly returns as a year -> month -> return table."""
        return MonthlyReturnsCalculator.to_table(self.get_monthly_returns())

    def get_trade_list(self) -> List[Dict[str, Any]]:
        """Get all trades as a list of dicts."""
        trades = []
        for t in self.result.trades:
            trade_dict = {
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "commission": t.commission,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "duration": str(t.duration) if t.duration else None,
                "bars_held": t.bars_held,
                "max_favorable": t.max_favorable,
                "max_adverse": t.max_adverse,
                "exit_reason": t.exit_reason,
                "tag": t.tag,
            }
            trades.append(trade_dict)
        return trades

    def get_pnl_by_day_of_week(self) -> Dict[str, Dict[str, float]]:
        """PnL breakdown by day of week."""
        return TimeAnalysis.pnl_by_day_of_week(self.result.trades)

    def get_pnl_by_hour(self) -> Dict[int, Dict[str, float]]:
        """PnL breakdown by hour of day."""
        return TimeAnalysis.pnl_by_hour(self.result.trades)

    def get_pnl_by_month(self) -> Dict[int, Dict[str, float]]:
        """PnL breakdown by calendar month."""
        return TimeAnalysis.pnl_by_month(self.result.trades)

    def get_rolling_sharpe(self, window: int = 60) -> List[RollingMetric]:
        """Get rolling Sharpe ratio."""
        returns = self._calculate_returns_with_timestamps()
        return RollingMetricsCalculator.rolling_sharpe(returns, window)

    def get_rolling_win_rate(self, window: int = 50) -> List[RollingMetric]:
        """Get rolling win rate."""
        return RollingMetricsCalculator.rolling_win_rate(
            self.result.trades, window
        )

    def get_rolling_return(self, window: int = 60) -> List[RollingMetric]:
        """Get rolling return."""
        return RollingMetricsCalculator.rolling_return(
            self.result.equity_curve, window
        )

    def get_rolling_volatility(self, window: int = 30) -> List[RollingMetric]:
        """Get rolling volatility."""
        returns = self._calculate_returns_with_timestamps()
        return RollingMetricsCalculator.rolling_volatility(returns, window)

    def get_best_worst_trades(self, n: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get the best and worst trades."""
        trades = self.get_trade_list()
        sorted_by_pnl = sorted(trades, key=lambda t: t["pnl"], reverse=True)
        return {
            "best": sorted_by_pnl[:n],
            "worst": sorted_by_pnl[-n:] if len(sorted_by_pnl) >= n else sorted_by_pnl,
        }

    def get_win_loss_streaks(self) -> Dict[str, Any]:
        """Analyze winning and losing streaks."""
        trades = self.result.trades
        if not trades:
            return {"win_streaks": [], "loss_streaks": []}

        win_streaks = []
        loss_streaks = []
        current_streak = 0
        current_type = None

        for trade in trades:
            is_win = trade.pnl > 0
            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif is_win == current_type:
                current_streak += 1
            else:
                if current_type:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                current_type = is_win
                current_streak = 1

        if current_type is not None:
            if current_type:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)

        return {
            "win_streaks": sorted(win_streaks, reverse=True),
            "loss_streaks": sorted(loss_streaks, reverse=True),
            "max_win_streak": max(win_streaks) if win_streaks else 0,
            "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_win_streak": sum(win_streaks) / max(1, len(win_streaks)),
            "avg_loss_streak": sum(loss_streaks) / max(1, len(loss_streaks)),
        }

    # -- Export methods --

    def export_json(self, filepath: str) -> None:
        """Export full report to JSON."""
        summary = self.generate_summary()
        report = {
            "summary": self._summary_to_dict(summary),
            "monthly_returns": [
                {"year": m.year, "month": m.month,
                 "return_pct": m.return_pct, "pnl": m.pnl}
                for m in self.get_monthly_returns()
            ],
            "trades": self.get_trade_list(),
            "drawdowns": [
                {
                    "start": d.start.isoformat() if d.start else None,
                    "end": d.end.isoformat() if d.end else None,
                    "depth_pct": d.depth_pct,
                    "duration_days": d.duration_days,
                }
                for d in self.get_drawdowns()
            ],
            "pnl_by_day": self.get_pnl_by_day_of_week(),
            "pnl_by_hour": {str(k): v for k, v in self.get_pnl_by_hour().items()},
            "best_worst": self.get_best_worst_trades(),
            "streaks": self.get_win_loss_streaks(),
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report exported to {filepath}")

    def export_csv(self, filepath: str) -> None:
        """Export trade list to CSV."""
        trades = self.get_trade_list()
        if not trades:
            logger.warning("No trades to export")
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)

        logger.info(f"Trades exported to {filepath} ({len(trades)} trades)")

    def export_equity_csv(self, filepath: str) -> None:
        """Export equity curve to CSV."""
        if not self.result.equity_curve:
            return

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "equity", "balance", "drawdown",
                "drawdown_pct", "positions", "open_pnl",
            ])
            for point in self.result.equity_curve:
                writer.writerow([
                    point.timestamp.isoformat() if point.timestamp else "",
                    f"{point.equity:.2f}",
                    f"{point.balance:.2f}",
                    f"{point.drawdown:.2f}",
                    f"{point.drawdown_percent:.2f}",
                    point.positions_count,
                    f"{point.open_pnl:.2f}",
                ])

        logger.info(f"Equity curve exported to {filepath}")

    def print_summary(self) -> str:
        """Generate a formatted text summary."""
        s = self.generate_summary()
        lines = []
        lines.append("=" * 60)
        lines.append("  BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Period
        lines.append("-- Period --")
        if s.start_date:
            lines.append(f"  Start:          {s.start_date.strftime('%Y-%m-%d %H:%M')}")
        if s.end_date:
            lines.append(f"  End:            {s.end_date.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  Trading Days:   {s.trading_days}")
        lines.append(f"  Bars Processed: {s.bars_processed}")
        lines.append(f"  Execution Time: {s.execution_time:.1f}s")
        lines.append("")

        # Returns
        lines.append("-- Returns --")
        lines.append(f"  Total Return:   ${s.total_return:,.2f} ({s.total_return_pct:+.2f}%)")
        lines.append(f"  CAGR:           {s.cagr:+.2f}%")
        lines.append("")

        # Risk
        lines.append("-- Risk Metrics --")
        lines.append(f"  Sharpe Ratio:   {s.sharpe_ratio:.3f}")
        lines.append(f"  Sortino Ratio:  {s.sortino_ratio:.3f}")
        lines.append(f"  Calmar Ratio:   {s.calmar_ratio:.3f}")
        lines.append(f"  Omega Ratio:    {s.omega_ratio:.3f}")
        lines.append(f"  Volatility:     {s.volatility:.2f}%")
        lines.append(f"  VaR (95%):      {s.var_95:.4f}")
        lines.append(f"  CVaR (95%):     {s.cvar_95:.4f}")
        lines.append(f"  Tail Ratio:     {s.tail_ratio:.3f}")
        lines.append("")

        # Drawdown
        lines.append("-- Drawdown --")
        lines.append(f"  Max Drawdown:   ${s.max_drawdown:,.2f} ({s.max_drawdown_pct:.2f}%)")
        lines.append(f"  Avg Drawdown:   {s.avg_drawdown:.2f}%")
        lines.append(f"  Max DD Duration: {s.max_drawdown_duration_days:.1f} days")
        lines.append(f"  Recovery Factor: {s.recovery_factor:.3f}")
        lines.append("")

        # Trades
        lines.append("-- Trade Statistics --")
        lines.append(f"  Total Trades:   {s.total_trades}")
        lines.append(f"  Winners:        {s.winning_trades} ({s.win_rate:.1f}%)")
        lines.append(f"  Losers:         {s.losing_trades}")
        lines.append(f"  Avg Trade PnL:  ${s.avg_trade_pnl:,.2f}")
        lines.append(f"  Avg Win:        ${s.avg_winning_trade:,.2f}")
        lines.append(f"  Avg Loss:       ${s.avg_losing_trade:,.2f}")
        lines.append(f"  Largest Win:    ${s.largest_win:,.2f}")
        lines.append(f"  Largest Loss:   ${s.largest_loss:,.2f}")
        lines.append(f"  Profit Factor:  {s.profit_factor:.3f}")
        lines.append(f"  Payoff Ratio:   {s.payoff_ratio:.3f}")
        lines.append(f"  Expectancy:     ${s.expectancy:,.2f}")
        lines.append(f"  Max Win Streak: {s.max_consecutive_wins}")
        lines.append(f"  Max Loss Streak:{s.max_consecutive_losses}")
        lines.append(f"  Avg Holding:    {s.avg_holding_period:.1f} bars")
        lines.append("")

        # Costs
        lines.append("-- Costs --")
        lines.append(f"  Commission:     ${s.total_commission:,.2f}")
        lines.append(f"  Slippage:       ${s.total_slippage:,.2f}")
        lines.append(f"  Funding:        ${s.total_funding:,.2f}")
        lines.append("")

        lines.append("=" * 60)

        output = "\n".join(lines)
        print(output)
        return output

    # -- Internal methods --

    def _calculate_returns(self) -> List[float]:
        """Calculate periodic returns from equity curve."""
        ec = self.result.equity_curve
        if not ec or len(ec) < 2:
            return []

        returns = []
        for i in range(1, len(ec)):
            prev = ec[i - 1].equity if hasattr(ec[i - 1], 'equity') else 0
            curr = ec[i].equity if hasattr(ec[i], 'equity') else 0
            if prev > 0:
                returns.append((curr - prev) / prev)
        return returns

    def _calculate_returns_with_timestamps(self) -> List[Tuple[datetime, float]]:
        """Calculate returns with timestamps."""
        ec = self.result.equity_curve
        if not ec or len(ec) < 2:
            return []

        returns = []
        for i in range(1, len(ec)):
            prev = ec[i - 1].equity if hasattr(ec[i - 1], 'equity') else 0
            curr = ec[i].equity if hasattr(ec[i], 'equity') else 0
            ts = ec[i].timestamp if hasattr(ec[i], 'timestamp') else None
            if prev > 0 and ts:
                returns.append((ts, (curr - prev) / prev))
        return returns

    def _summary_to_dict(self, summary: PerformanceSummary) -> Dict[str, Any]:
        """Convert PerformanceSummary to a JSON-serializable dict."""
        d = {}
        for field_name in summary.__dataclass_fields__:
            val = getattr(summary, field_name)
            if isinstance(val, datetime):
                d[field_name] = val.isoformat()
            elif isinstance(val, float):
                d[field_name] = round(val, 6)
            else:
                d[field_name] = val
        return d
