"""
Real-time Performance Analytics for Mefai Autotrade.
Tracks rolling metrics, equity curves, drawdown analysis, trade distributions,
win/loss streaks, time-based analysis, benchmark comparison, and risk-adjusted
return metrics across multiple windows and dimensions.
"""

import logging
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("mefai.autotrade.performance")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Immutable record of a completed trade."""
    trade_id: str
    symbol: str
    side: str  # LONG or SHORT
    strategy: str
    timeframe: str
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    pnl_percent: float
    fees: float
    net_pnl: float
    entry_time: datetime
    exit_time: datetime
    duration_seconds: float
    close_reason: str
    leverage: int = 1

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def is_loser(self) -> bool:
        return self.net_pnl < 0

    @property
    def r_multiple(self) -> float:
        """Return as a multiple of risk (approximation using pnl_percent)."""
        if self.pnl_percent == 0:
            return 0.0
        return self.pnl_percent / abs(self.pnl_percent)


@dataclass
class EquityCurvePoint:
    """Single point on the equity curve."""
    timestamp: datetime
    equity: float
    unrealized_pnl: float
    realized_pnl_cumulative: float
    drawdown_pct: float
    peak_equity: float
    num_positions: int


@dataclass
class DrawdownInfo:
    """Information about a drawdown period."""
    start_time: datetime
    end_time: Optional[datetime]  # None if still in drawdown
    peak_equity: float
    trough_equity: float
    max_drawdown_pct: float
    recovery_time: Optional[datetime]  # None if not yet recovered
    duration_seconds: float
    recovery_seconds: Optional[float]

    @property
    def is_recovered(self) -> bool:
        return self.recovery_time is not None


@dataclass
class StreakInfo:
    """Win/loss streak tracking."""
    current_streak: int  # positive = wins, negative = losses
    longest_win_streak: int
    longest_loss_streak: int
    current_streak_pnl: float
    longest_win_streak_pnl: float
    longest_loss_streak_pnl: float


@dataclass
class TimeAnalysis:
    """Performance breakdown by time periods."""
    best_hour: int  # 0-23
    worst_hour: int
    best_day_of_week: int  # 0=Monday, 6=Sunday
    worst_day_of_week: int
    hourly_pnl: Dict[int, float]  # hour -> cumulative PnL
    hourly_trades: Dict[int, int]  # hour -> trade count
    daily_pnl: Dict[int, float]  # day_of_week -> cumulative PnL
    daily_trades: Dict[int, int]  # day_of_week -> trade count
    hourly_win_rate: Dict[int, float]  # hour -> win rate


@dataclass
class MonthlyReturn:
    """Monthly return data for calendar heatmap."""
    year: int
    month: int
    return_pct: float
    trade_count: int
    win_count: int
    loss_count: int
    total_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float


@dataclass
class RollingMetrics:
    """Performance metrics over a rolling time window."""
    window_name: str
    window_seconds: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    average_winner: float
    average_loser: float
    largest_winner: float
    largest_loser: float
    profit_factor: float
    expectancy: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown_pct: float
    avg_trade_duration_seconds: float
    total_fees: float
    net_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "average_pnl": round(self.average_pnl, 2),
            "average_winner": round(self.average_winner, 2),
            "average_loser": round(self.average_loser, 2),
            "largest_winner": round(self.largest_winner, 2),
            "largest_loser": round(self.largest_loser, 2),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "omega_ratio": round(self.omega_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "avg_trade_duration_seconds": round(self.avg_trade_duration_seconds, 0),
            "total_fees": round(self.total_fees, 2),
            "net_pnl": round(self.net_pnl, 2),
        }


# ---------------------------------------------------------------------------
# Rolling Window
# ---------------------------------------------------------------------------

class _RollingWindow:
    """Time-based rolling window for trade records."""

    WINDOWS = {
        "1h": 3600,
        "24h": 86400,
        "7d": 604800,
        "30d": 2592000,
        "all": 0,
    }

    def __init__(self, name: str, window_seconds: int):
        self.name = name
        self.window_seconds = window_seconds
        self.trades: Deque[TradeRecord] = deque()
        self._pnl_series: Deque[float] = deque()

    def add_trade(self, trade: TradeRecord) -> None:
        self.trades.append(trade)
        self._pnl_series.append(trade.net_pnl)
        self._prune()

    def _prune(self) -> None:
        if self.window_seconds == 0:
            return  # "all" window - never prune
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
        while self.trades and self.trades[0].exit_time < cutoff:
            self.trades.popleft()
            if self._pnl_series:
                self._pnl_series.popleft()

    def get_trades(self) -> List[TradeRecord]:
        self._prune()
        return list(self.trades)

    def get_pnl_series(self) -> List[float]:
        self._prune()
        return list(self._pnl_series)


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Real-time performance analytics engine.

    Tracks rolling metrics across multiple time windows and dimensions
    (strategy, symbol, timeframe). Computes risk-adjusted returns,
    drawdown analysis, equity curves, and more.
    """

    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
    ANNUALIZATION_FACTOR = 365.25  # Days per year for crypto (24/7)

    def __init__(self, initial_equity: float = 10000.0):
        self._lock = threading.Lock()
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._peak_equity = initial_equity
        self._unrealized_pnl = 0.0
        self._cumulative_pnl = 0.0
        self._total_fees = 0.0

        # All trades ever recorded
        self._all_trades: List[TradeRecord] = []

        # Rolling windows
        self._windows: Dict[str, _RollingWindow] = {}
        for name, seconds in _RollingWindow.WINDOWS.items():
            self._windows[name] = _RollingWindow(name, seconds)

        # Per-dimension trackers
        self._strategy_trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._symbol_trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._timeframe_trades: Dict[str, List[TradeRecord]] = defaultdict(list)

        # Equity curve
        self._equity_curve: List[EquityCurvePoint] = []
        self._last_equity_snapshot = 0.0

        # Drawdown tracking
        self._drawdowns: List[DrawdownInfo] = []
        self._current_drawdown: Optional[DrawdownInfo] = None

        # Streak tracking
        self._current_streak = 0
        self._current_streak_pnl = 0.0
        self._longest_win_streak = 0
        self._longest_loss_streak = 0
        self._longest_win_streak_pnl = 0.0
        self._longest_loss_streak_pnl = 0.0

        # Time analysis
        self._hourly_pnl: Dict[int, float] = defaultdict(float)
        self._hourly_trades: Dict[int, int] = defaultdict(int)
        self._hourly_wins: Dict[int, int] = defaultdict(int)
        self._daily_pnl: Dict[int, float] = defaultdict(float)
        self._daily_trades: Dict[int, int] = defaultdict(int)

        # Monthly returns
        self._monthly_data: Dict[Tuple[int, int], Dict] = {}

        # Benchmark (BTC buy-and-hold)
        self._benchmark_start_price: Optional[float] = None
        self._benchmark_current_price: Optional[float] = None

    # -------------------------------------------------------------------
    # Trade Recording
    # -------------------------------------------------------------------

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade and update all metrics."""
        with self._lock:
            self._all_trades.append(trade)

            # Update equity
            self._cumulative_pnl += trade.net_pnl
            self._total_fees += trade.fees
            self._current_equity = self._initial_equity + self._cumulative_pnl

            # Peak tracking
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity
                self._close_drawdown(trade.exit_time)

            # Drawdown tracking
            self._update_drawdown(trade.exit_time)

            # Rolling windows
            for window in self._windows.values():
                window.add_trade(trade)

            # Per-dimension
            self._strategy_trades[trade.strategy].append(trade)
            self._symbol_trades[trade.symbol].append(trade)
            self._timeframe_trades[trade.timeframe].append(trade)

            # Streak tracking
            self._update_streak(trade)

            # Time analysis
            hour = trade.exit_time.hour
            day = trade.exit_time.weekday()
            self._hourly_pnl[hour] += trade.net_pnl
            self._hourly_trades[hour] += 1
            if trade.is_winner:
                self._hourly_wins[hour] += 1
            self._daily_pnl[day] += trade.net_pnl
            self._daily_trades[day] += 1

            # Monthly data
            key = (trade.exit_time.year, trade.exit_time.month)
            if key not in self._monthly_data:
                self._monthly_data[key] = {
                    "pnl": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "best": 0.0,
                    "worst": 0.0,
                }
            md = self._monthly_data[key]
            md["pnl"] += trade.net_pnl
            md["trades"] += 1
            if trade.is_winner:
                md["wins"] += 1
            elif trade.is_loser:
                md["losses"] += 1
            md["best"] = max(md["best"], trade.net_pnl)
            md["worst"] = min(md["worst"], trade.net_pnl)

    def update_unrealized_pnl(self, unrealized_pnl: float, num_positions: int = 0) -> None:
        """Update unrealized PnL for equity curve snapshots."""
        with self._lock:
            self._unrealized_pnl = unrealized_pnl
            now = datetime.now(timezone.utc)
            equity = self._current_equity + unrealized_pnl

            dd_pct = 0.0
            if self._peak_equity > 0:
                dd_pct = (self._peak_equity - equity) / self._peak_equity * 100

            point = EquityCurvePoint(
                timestamp=now,
                equity=equity,
                unrealized_pnl=unrealized_pnl,
                realized_pnl_cumulative=self._cumulative_pnl,
                drawdown_pct=max(0, dd_pct),
                peak_equity=self._peak_equity,
                num_positions=num_positions,
            )
            self._equity_curve.append(point)

            # Keep equity curve manageable (max 50000 points)
            if len(self._equity_curve) > 50000:
                self._equity_curve = self._equity_curve[-25000:]

    def set_benchmark_price(self, price: float) -> None:
        """Set the current BTC price for benchmark comparison."""
        with self._lock:
            if self._benchmark_start_price is None:
                self._benchmark_start_price = price
            self._benchmark_current_price = price

    # -------------------------------------------------------------------
    # Drawdown Tracking
    # -------------------------------------------------------------------

    def _update_drawdown(self, timestamp: datetime) -> None:
        if self._current_equity >= self._peak_equity:
            return

        dd_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100

        if self._current_drawdown is None:
            self._current_drawdown = DrawdownInfo(
                start_time=timestamp,
                end_time=None,
                peak_equity=self._peak_equity,
                trough_equity=self._current_equity,
                max_drawdown_pct=dd_pct,
                recovery_time=None,
                duration_seconds=0,
                recovery_seconds=None,
            )
        else:
            self._current_drawdown.end_time = timestamp
            self._current_drawdown.duration_seconds = (
                timestamp - self._current_drawdown.start_time
            ).total_seconds()
            if self._current_equity < self._current_drawdown.trough_equity:
                self._current_drawdown.trough_equity = self._current_equity
                self._current_drawdown.max_drawdown_pct = dd_pct

    def _close_drawdown(self, timestamp: datetime) -> None:
        if self._current_drawdown is not None:
            self._current_drawdown.recovery_time = timestamp
            self._current_drawdown.end_time = timestamp
            self._current_drawdown.duration_seconds = (
                timestamp - self._current_drawdown.start_time
            ).total_seconds()
            self._current_drawdown.recovery_seconds = (
                self._current_drawdown.duration_seconds
            )
            self._drawdowns.append(self._current_drawdown)
            self._current_drawdown = None

    # -------------------------------------------------------------------
    # Streak Tracking
    # -------------------------------------------------------------------

    def _update_streak(self, trade: TradeRecord) -> None:
        if trade.is_winner:
            if self._current_streak > 0:
                self._current_streak += 1
                self._current_streak_pnl += trade.net_pnl
            else:
                self._current_streak = 1
                self._current_streak_pnl = trade.net_pnl
            if self._current_streak > self._longest_win_streak:
                self._longest_win_streak = self._current_streak
                self._longest_win_streak_pnl = self._current_streak_pnl
        elif trade.is_loser:
            if self._current_streak < 0:
                self._current_streak -= 1
                self._current_streak_pnl += trade.net_pnl
            else:
                self._current_streak = -1
                self._current_streak_pnl = trade.net_pnl
            if abs(self._current_streak) > self._longest_loss_streak:
                self._longest_loss_streak = abs(self._current_streak)
                self._longest_loss_streak_pnl = self._current_streak_pnl

    # -------------------------------------------------------------------
    # Metric Calculations
    # -------------------------------------------------------------------

    def get_rolling_metrics(self, window: str = "all") -> RollingMetrics:
        """Calculate metrics for a given rolling window."""
        with self._lock:
            if window not in self._windows:
                raise ValueError(f"Unknown window: {window}. "
                                 f"Valid: {list(self._windows.keys())}")
            trades = self._windows[window].get_trades()
            return self._calculate_metrics(
                trades,
                self._windows[window].name,
                self._windows[window].window_seconds,
            )

    def get_strategy_metrics(self, strategy: str) -> RollingMetrics:
        """Get performance metrics for a specific strategy."""
        with self._lock:
            trades = self._strategy_trades.get(strategy, [])
            return self._calculate_metrics(trades, f"strategy:{strategy}", 0)

    def get_symbol_metrics(self, symbol: str) -> RollingMetrics:
        """Get performance metrics for a specific symbol."""
        with self._lock:
            trades = self._symbol_trades.get(symbol, [])
            return self._calculate_metrics(trades, f"symbol:{symbol}", 0)

    def get_timeframe_metrics(self, timeframe: str) -> RollingMetrics:
        """Get performance metrics for a specific timeframe."""
        with self._lock:
            trades = self._timeframe_trades.get(timeframe, [])
            return self._calculate_metrics(trades, f"tf:{timeframe}", 0)

    def _calculate_metrics(
        self,
        trades: List[TradeRecord],
        window_name: str,
        window_seconds: int,
    ) -> RollingMetrics:
        """Core metric calculation engine."""
        if not trades:
            return RollingMetrics(
                window_name=window_name,
                window_seconds=window_seconds,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                average_pnl=0.0,
                average_winner=0.0,
                average_loser=0.0,
                largest_winner=0.0,
                largest_loser=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                omega_ratio=0.0,
                max_drawdown_pct=0.0,
                avg_trade_duration_seconds=0.0,
                total_fees=0.0,
                net_pnl=0.0,
            )

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if t.is_loser]
        pnl_list = [t.net_pnl for t in trades]
        pnl_pct_list = [t.pnl_percent for t in trades]

        total_trades = len(trades)
        winning_count = len(winners)
        losing_count = len(losers)
        win_rate = winning_count / total_trades if total_trades > 0 else 0.0

        total_pnl = sum(pnl_list)
        avg_pnl = total_pnl / total_trades
        avg_winner = (
            sum(t.net_pnl for t in winners) / winning_count
            if winning_count > 0
            else 0.0
        )
        avg_loser = (
            sum(t.net_pnl for t in losers) / losing_count
            if losing_count > 0
            else 0.0
        )
        largest_winner = max(pnl_list) if pnl_list else 0.0
        largest_loser = min(pnl_list) if pnl_list else 0.0

        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_winner) - (loss_rate * abs(avg_loser))

        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe(pnl_pct_list)
        sortino = self._calculate_sortino(pnl_pct_list)
        max_dd = self._calculate_max_drawdown_from_pnl(pnl_list)
        calmar = self._calculate_calmar(pnl_pct_list, max_dd)
        omega = self._calculate_omega(pnl_pct_list)

        avg_duration = (
            sum(t.duration_seconds for t in trades) / total_trades
        )
        total_fees = sum(t.fees for t in trades)
        net_pnl = total_pnl

        return RollingMetrics(
            window_name=window_name,
            window_seconds=window_seconds,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_pnl=avg_pnl,
            average_winner=avg_winner,
            average_loser=avg_loser,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            profit_factor=profit_factor,
            expectancy=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            max_drawdown_pct=max_dd,
            avg_trade_duration_seconds=avg_duration,
            total_fees=total_fees,
            net_pnl=net_pnl,
        )

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Annualized Sharpe ratio from trade returns."""
        if len(returns) < 2:
            return 0.0
        mean_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns)
        if std_ret == 0:
            return 0.0
        daily_rf = self.RISK_FREE_RATE / self.ANNUALIZATION_FACTOR
        excess_return = mean_ret - daily_rf
        # Approximate annualization based on trade frequency
        trades_per_day = len(returns) / max(1, self._trading_days(returns))
        annualization = math.sqrt(trades_per_day * self.ANNUALIZATION_FACTOR)
        return (excess_return / std_ret) * annualization

    def _calculate_sortino(self, returns: List[float]) -> float:
        """Annualized Sortino ratio (penalizes only downside volatility)."""
        if len(returns) < 2:
            return 0.0
        mean_ret = statistics.mean(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if mean_ret > 0 else 0.0
        downside_dev = math.sqrt(sum(r ** 2 for r in downside) / len(returns))
        if downside_dev == 0:
            return 0.0
        daily_rf = self.RISK_FREE_RATE / self.ANNUALIZATION_FACTOR
        excess = mean_ret - daily_rf
        trades_per_day = len(returns) / max(1, self._trading_days(returns))
        annualization = math.sqrt(trades_per_day * self.ANNUALIZATION_FACTOR)
        return (excess / downside_dev) * annualization

    def _calculate_calmar(self, returns: List[float], max_dd: float) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if max_dd == 0 or len(returns) < 2:
            return 0.0
        total_return = sum(returns)
        days = max(1, self._trading_days(returns))
        annualized = (total_return / days) * self.ANNUALIZATION_FACTOR
        return annualized / max_dd

    def _calculate_omega(
        self, returns: List[float], threshold: float = 0.0
    ) -> float:
        """Omega ratio = sum of gains above threshold / sum of losses below threshold."""
        if not returns:
            return 0.0
        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def _calculate_max_drawdown_from_pnl(self, pnl_list: List[float]) -> float:
        """Calculate max drawdown percentage from a PnL series."""
        if not pnl_list:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_list:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if peak > 0:
                dd_pct = dd / (self._initial_equity + peak) * 100
                max_dd = max(max_dd, dd_pct)
        return max_dd

    def _trading_days(self, returns: List[float]) -> float:
        """Estimate number of trading days from the trade list."""
        if not self._all_trades or len(self._all_trades) < 2:
            return 1.0
        first = self._all_trades[0].entry_time
        last = self._all_trades[-1].exit_time
        days = (last - first).total_seconds() / 86400
        return max(1.0, days)

    # -------------------------------------------------------------------
    # Equity Curve
    # -------------------------------------------------------------------

    def get_equity_curve(
        self,
        resolution: str = "hourly",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Get equity curve data at the specified resolution.

        Args:
            resolution: "hourly", "daily", or "weekly"
            limit: Maximum number of data points.
        """
        with self._lock:
            if not self._equity_curve:
                return []

            # Resample based on resolution
            interval_seconds = {
                "hourly": 3600,
                "daily": 86400,
                "weekly": 604800,
            }.get(resolution, 3600)

            resampled: List[Dict[str, Any]] = []
            bucket_start = None
            bucket_point = None

            for point in self._equity_curve:
                ts = point.timestamp.timestamp()
                bucket = int(ts // interval_seconds) * interval_seconds

                if bucket != bucket_start:
                    if bucket_point is not None:
                        resampled.append(self._equity_point_to_dict(bucket_point))
                    bucket_start = bucket
                    bucket_point = point
                else:
                    bucket_point = point  # Keep last point in bucket

            if bucket_point is not None:
                resampled.append(self._equity_point_to_dict(bucket_point))

            return resampled[-limit:]

    @staticmethod
    def _equity_point_to_dict(point: EquityCurvePoint) -> Dict[str, Any]:
        return {
            "timestamp": point.timestamp.isoformat(),
            "equity": round(point.equity, 2),
            "unrealized_pnl": round(point.unrealized_pnl, 2),
            "realized_pnl_cumulative": round(point.realized_pnl_cumulative, 2),
            "drawdown_pct": round(point.drawdown_pct, 4),
            "peak_equity": round(point.peak_equity, 2),
            "num_positions": point.num_positions,
        }

    # -------------------------------------------------------------------
    # Drawdown Analysis
    # -------------------------------------------------------------------

    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """Get comprehensive drawdown analysis."""
        with self._lock:
            all_drawdowns = list(self._drawdowns)
            if self._current_drawdown:
                all_drawdowns.append(self._current_drawdown)

            if not all_drawdowns:
                return {
                    "current_drawdown_pct": 0.0,
                    "max_drawdown_pct": 0.0,
                    "avg_drawdown_pct": 0.0,
                    "avg_recovery_seconds": 0.0,
                    "total_drawdown_periods": 0,
                    "currently_in_drawdown": False,
                    "drawdowns": [],
                }

            max_dd = max(d.max_drawdown_pct for d in all_drawdowns)
            avg_dd = statistics.mean(d.max_drawdown_pct for d in all_drawdowns)

            recovered = [d for d in all_drawdowns if d.is_recovered]
            avg_recovery = (
                statistics.mean(d.recovery_seconds for d in recovered)
                if recovered
                else 0.0
            )

            current_dd = 0.0
            if self._peak_equity > 0:
                current_dd = max(
                    0,
                    (self._peak_equity - self._current_equity)
                    / self._peak_equity
                    * 100,
                )

            return {
                "current_drawdown_pct": round(current_dd, 4),
                "max_drawdown_pct": round(max_dd, 4),
                "avg_drawdown_pct": round(avg_dd, 4),
                "avg_recovery_seconds": round(avg_recovery, 0),
                "total_drawdown_periods": len(all_drawdowns),
                "currently_in_drawdown": self._current_drawdown is not None,
                "drawdowns": [
                    {
                        "start": d.start_time.isoformat(),
                        "end": d.end_time.isoformat() if d.end_time else None,
                        "max_dd_pct": round(d.max_drawdown_pct, 4),
                        "peak": round(d.peak_equity, 2),
                        "trough": round(d.trough_equity, 2),
                        "recovered": d.is_recovered,
                        "duration_seconds": round(d.duration_seconds, 0),
                    }
                    for d in all_drawdowns[-20:]  # Last 20 drawdowns
                ],
            }

    # -------------------------------------------------------------------
    # Trade Distribution
    # -------------------------------------------------------------------

    def get_trade_distribution(self, bins: int = 20) -> Dict[str, Any]:
        """PnL and duration distribution analysis."""
        with self._lock:
            if not self._all_trades:
                return {"pnl_histogram": [], "duration_histogram": []}

            pnl_values = [t.net_pnl for t in self._all_trades]
            durations = [t.duration_seconds for t in self._all_trades]

            return {
                "pnl_histogram": self._build_histogram(pnl_values, bins),
                "duration_histogram": self._build_histogram(durations, bins),
                "pnl_stats": {
                    "mean": round(statistics.mean(pnl_values), 2),
                    "median": round(statistics.median(pnl_values), 2),
                    "stdev": round(statistics.stdev(pnl_values), 2) if len(pnl_values) > 1 else 0.0,
                    "skewness": round(self._skewness(pnl_values), 4),
                    "kurtosis": round(self._kurtosis(pnl_values), 4),
                },
                "duration_stats": {
                    "mean": round(statistics.mean(durations), 0),
                    "median": round(statistics.median(durations), 0),
                    "min": round(min(durations), 0),
                    "max": round(max(durations), 0),
                },
            }

    @staticmethod
    def _build_histogram(values: List[float], bins: int) -> List[Dict[str, Any]]:
        if not values:
            return []
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return [{"bin_start": min_val, "bin_end": max_val, "count": len(values)}]
        bin_width = (max_val - min_val) / bins
        histogram = []
        for i in range(bins):
            start = min_val + i * bin_width
            end = start + bin_width
            count = sum(1 for v in values if start <= v < end or (i == bins - 1 and v == end))
            histogram.append({
                "bin_start": round(start, 2),
                "bin_end": round(end, 2),
                "count": count,
            })
        return histogram

    @staticmethod
    def _skewness(values: List[float]) -> float:
        n = len(values)
        if n < 3:
            return 0.0
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        if std == 0:
            return 0.0
        return (n / ((n - 1) * (n - 2))) * sum(((v - mean) / std) ** 3 for v in values)

    @staticmethod
    def _kurtosis(values: List[float]) -> float:
        n = len(values)
        if n < 4:
            return 0.0
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        if std == 0:
            return 0.0
        k = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        s = sum(((v - mean) / std) ** 4 for v in values)
        correction = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        return k * s - correction

    # -------------------------------------------------------------------
    # Streak Info
    # -------------------------------------------------------------------

    def get_streak_info(self) -> StreakInfo:
        with self._lock:
            return StreakInfo(
                current_streak=self._current_streak,
                longest_win_streak=self._longest_win_streak,
                longest_loss_streak=self._longest_loss_streak,
                current_streak_pnl=round(self._current_streak_pnl, 2),
                longest_win_streak_pnl=round(self._longest_win_streak_pnl, 2),
                longest_loss_streak_pnl=round(self._longest_loss_streak_pnl, 2),
            )

    # -------------------------------------------------------------------
    # Time Analysis
    # -------------------------------------------------------------------

    def get_time_analysis(self) -> TimeAnalysis:
        with self._lock:
            hourly_wr: Dict[int, float] = {}
            for h in range(24):
                total = self._hourly_trades.get(h, 0)
                wins = self._hourly_wins.get(h, 0)
                hourly_wr[h] = wins / total if total > 0 else 0.0

            best_hour = max(self._hourly_pnl, key=self._hourly_pnl.get) if self._hourly_pnl else 0
            worst_hour = min(self._hourly_pnl, key=self._hourly_pnl.get) if self._hourly_pnl else 0
            best_day = max(self._daily_pnl, key=self._daily_pnl.get) if self._daily_pnl else 0
            worst_day = min(self._daily_pnl, key=self._daily_pnl.get) if self._daily_pnl else 0

            return TimeAnalysis(
                best_hour=best_hour,
                worst_hour=worst_hour,
                best_day_of_week=best_day,
                worst_day_of_week=worst_day,
                hourly_pnl=dict(self._hourly_pnl),
                hourly_trades=dict(self._hourly_trades),
                daily_pnl=dict(self._daily_pnl),
                daily_trades=dict(self._daily_trades),
                hourly_win_rate=hourly_wr,
            )

    # -------------------------------------------------------------------
    # Monthly Returns
    # -------------------------------------------------------------------

    def get_monthly_returns(self) -> List[MonthlyReturn]:
        """Get monthly returns for calendar heatmap."""
        with self._lock:
            result = []
            for (year, month), data in sorted(self._monthly_data.items()):
                # Calculate return percentage relative to equity at start of month
                trades_before = [
                    t
                    for t in self._all_trades
                    if (t.exit_time.year, t.exit_time.month) < (year, month)
                ]
                equity_at_start = self._initial_equity + sum(
                    t.net_pnl for t in trades_before
                )
                return_pct = (
                    (data["pnl"] / equity_at_start * 100)
                    if equity_at_start > 0
                    else 0.0
                )

                result.append(
                    MonthlyReturn(
                        year=year,
                        month=month,
                        return_pct=round(return_pct, 4),
                        trade_count=data["trades"],
                        win_count=data["wins"],
                        loss_count=data["losses"],
                        total_pnl=round(data["pnl"], 2),
                        best_trade_pnl=round(data["best"], 2),
                        worst_trade_pnl=round(data["worst"], 2),
                    )
                )
            return result

    # -------------------------------------------------------------------
    # Benchmark Comparison
    # -------------------------------------------------------------------

    def get_benchmark_comparison(self) -> Dict[str, Any]:
        """Compare strategy performance vs BTC buy-and-hold."""
        with self._lock:
            strategy_return = (
                (self._cumulative_pnl / self._initial_equity * 100)
                if self._initial_equity > 0
                else 0.0
            )

            btc_return = 0.0
            if (
                self._benchmark_start_price
                and self._benchmark_current_price
                and self._benchmark_start_price > 0
            ):
                btc_return = (
                    (self._benchmark_current_price - self._benchmark_start_price)
                    / self._benchmark_start_price
                    * 100
                )

            return {
                "strategy_return_pct": round(strategy_return, 4),
                "benchmark_return_pct": round(btc_return, 4),
                "alpha": round(strategy_return - btc_return, 4),
                "benchmark": "BTCUSDT (buy-and-hold)",
                "strategy_equity": round(self._current_equity, 2),
                "benchmark_equity": round(
                    self._initial_equity * (1 + btc_return / 100), 2
                ),
            }

    # -------------------------------------------------------------------
    # Per-Dimension Summaries
    # -------------------------------------------------------------------

    def get_all_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all strategies."""
        with self._lock:
            return {
                strategy: self._calculate_metrics(
                    trades, f"strategy:{strategy}", 0
                ).to_dict()
                for strategy, trades in self._strategy_trades.items()
            }

    def get_all_symbol_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all symbols."""
        with self._lock:
            return {
                symbol: self._calculate_metrics(
                    trades, f"symbol:{symbol}", 0
                ).to_dict()
                for symbol, trades in self._symbol_trades.items()
            }

    def get_all_timeframe_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all timeframes."""
        with self._lock:
            return {
                tf: self._calculate_metrics(
                    trades, f"tf:{tf}", 0
                ).to_dict()
                for tf, trades in self._timeframe_trades.items()
            }

    # -------------------------------------------------------------------
    # Summary / Overview
    # -------------------------------------------------------------------

    def get_overview(self) -> Dict[str, Any]:
        """Get a high-level performance overview."""
        with self._lock:
            current_dd = 0.0
            if self._peak_equity > 0:
                current_dd = max(
                    0,
                    (self._peak_equity - self._current_equity)
                    / self._peak_equity
                    * 100,
                )

            all_time = self._calculate_metrics(
                self._all_trades, "all", 0
            )

            return {
                "initial_equity": self._initial_equity,
                "current_equity": round(self._current_equity, 2),
                "peak_equity": round(self._peak_equity, 2),
                "total_pnl": round(self._cumulative_pnl, 2),
                "unrealized_pnl": round(self._unrealized_pnl, 2),
                "total_return_pct": round(
                    self._cumulative_pnl / self._initial_equity * 100, 4
                )
                if self._initial_equity > 0
                else 0.0,
                "current_drawdown_pct": round(current_dd, 4),
                "max_drawdown_pct": round(all_time.max_drawdown_pct, 4),
                "total_trades": all_time.total_trades,
                "win_rate": round(all_time.win_rate, 4),
                "profit_factor": round(all_time.profit_factor, 4),
                "sharpe_ratio": round(all_time.sharpe_ratio, 4),
                "sortino_ratio": round(all_time.sortino_ratio, 4),
                "total_fees": round(self._total_fees, 2),
                "strategies_active": len(self._strategy_trades),
                "symbols_traded": len(self._symbol_trades),
            }

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent trade records."""
        with self._lock:
            trades = self._all_trades[-limit:]
            return [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "strategy": t.strategy,
                    "timeframe": t.timeframe,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "realized_pnl": round(t.realized_pnl, 2),
                    "pnl_percent": round(t.pnl_percent, 4),
                    "fees": round(t.fees, 4),
                    "net_pnl": round(t.net_pnl, 2),
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "duration_seconds": round(t.duration_seconds, 0),
                    "close_reason": t.close_reason,
                    "leverage": t.leverage,
                }
                for t in reversed(trades)
            ]
