"""
PnL Engine - Profit and loss calculation engine for Mefai Autotrade.
Tracks real-time PnL, commissions, funding rates, drawdowns, equity curves,
and computes risk-adjusted performance metrics (Sharpe, Sortino, Calmar, etc).
"""

import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mefai.autotrade.pnl_engine")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PnLPeriod(Enum):
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PnLRecord:
    """A single PnL event (trade close, funding payment, commission, etc)."""
    record_id: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    symbol: str = ""
    strategy_id: str = ""
    side: str = ""  # LONG or SHORT
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    gross_pnl: float = 0.0
    commission: float = 0.0
    funding_pnl: float = 0.0
    net_pnl: float = 0.0
    holding_time_seconds: float = 0.0
    leverage: int = 1
    is_winner: bool = False
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.net_pnl == 0.0 and (
            self.gross_pnl != 0.0 or self.commission != 0.0
        ):
            self.net_pnl = self.gross_pnl - self.commission + self.funding_pnl
        self.is_winner = self.net_pnl > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "gross_pnl": self.gross_pnl,
            "commission": self.commission,
            "funding_pnl": self.funding_pnl,
            "net_pnl": self.net_pnl,
            "holding_time_seconds": self.holding_time_seconds,
            "leverage": self.leverage,
            "is_winner": self.is_winner,
            "tags": self.tags,
        }


@dataclass
class PnLSnapshot:
    """Equity snapshot at a point in time."""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    equity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_funding: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: int = 0
    margin_used: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "total_funding": self.total_funding,
            "drawdown": self.drawdown,
            "drawdown_pct": self.drawdown_pct,
            "open_positions": self.open_positions,
            "margin_used": self.margin_used,
        }


@dataclass
class DrawdownInfo:
    """Tracks drawdown state."""
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    drawdown_start: Optional[datetime] = None
    drawdown_end: Optional[datetime] = None
    recovery_time_seconds: float = 0.0
    max_recovery_time_seconds: float = 0.0
    in_drawdown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_drawdown": self.current_drawdown,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "peak_equity": self.peak_equity,
            "trough_equity": self.trough_equity,
            "in_drawdown": self.in_drawdown,
            "recovery_time_seconds": self.recovery_time_seconds,
            "max_recovery_time_seconds": self.max_recovery_time_seconds,
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # PnL
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_commission: float = 0.0
    total_funding_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    average_trade_pnl: float = 0.0

    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    average_drawdown: float = 0.0
    max_recovery_time_seconds: float = 0.0

    # Time-based
    average_holding_time_seconds: float = 0.0
    total_trading_days: int = 0
    annualized_return: float = 0.0

    # Streaks
    max_win_streak: int = 0
    max_loss_streak: int = 0
    current_streak: int = 0
    current_streak_type: str = ""  # "WIN" or "LOSS"

    # By symbol / strategy
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)
    pnl_by_strategy: Dict[str, float] = field(default_factory=dict)
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)
    win_rate_by_symbol: Dict[str, float] = field(default_factory=dict)
    win_rate_by_strategy: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "total_gross_pnl": self.total_gross_pnl,
            "total_net_pnl": self.total_net_pnl,
            "total_commission": self.total_commission,
            "total_funding_pnl": self.total_funding_pnl,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "average_trade_pnl": self.average_trade_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "payoff_ratio": self.payoff_ratio,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "average_drawdown": self.average_drawdown,
            "max_recovery_time_seconds": self.max_recovery_time_seconds,
            "average_holding_time_seconds": self.average_holding_time_seconds,
            "total_trading_days": self.total_trading_days,
            "annualized_return": self.annualized_return,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "current_streak": self.current_streak,
            "current_streak_type": self.current_streak_type,
            "pnl_by_symbol": self.pnl_by_symbol,
            "pnl_by_strategy": self.pnl_by_strategy,
            "trades_by_symbol": self.trades_by_symbol,
            "trades_by_strategy": self.trades_by_strategy,
            "win_rate_by_symbol": self.win_rate_by_symbol,
            "win_rate_by_strategy": self.win_rate_by_strategy,
        }


# ---------------------------------------------------------------------------
# PnL Engine
# ---------------------------------------------------------------------------

class PnLEngine:
    """
    Production PnL engine. Tracks every trade, computes rolling metrics,
    equity curves, drawdown, and risk-adjusted returns.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_free_rate: float = 0.04,  # Annual, for Sharpe/Sortino
        snapshot_interval_minutes: int = 60,
        max_snapshots: int = 8760,  # ~1 year of hourly snapshots
    ):
        self._lock = threading.RLock()

        # Core state
        self._initial_balance = initial_balance
        self._current_equity = initial_balance
        self._risk_free_rate = risk_free_rate

        # Trade records
        self._records: List[PnLRecord] = []
        self._records_by_symbol: Dict[str, List[PnLRecord]] = defaultdict(list)
        self._records_by_strategy: Dict[str, List[PnLRecord]] = defaultdict(list)

        # Commission and funding tracking
        self._total_commission: float = 0.0
        self._total_funding_pnl: float = 0.0
        self._commission_by_symbol: Dict[str, float] = defaultdict(float)
        self._funding_by_symbol: Dict[str, float] = defaultdict(float)

        # Equity curve
        self._snapshots: List[PnLSnapshot] = []
        self._snapshot_interval = timedelta(minutes=snapshot_interval_minutes)
        self._max_snapshots = max_snapshots
        self._last_snapshot_time: Optional[datetime] = None

        # Drawdown tracking
        self._drawdown = DrawdownInfo(peak_equity=initial_balance)
        self._drawdown_periods: List[DrawdownInfo] = []

        # Daily returns for Sharpe/Sortino
        self._daily_returns: List[float] = []
        self._daily_pnl: Dict[str, float] = {}  # date_str -> pnl
        self._current_day: str = ""

        # Streak tracking
        self._current_streak = 0
        self._current_streak_type = ""
        self._max_win_streak = 0
        self._max_loss_streak = 0

        # Aggregated PnL by period
        self._period_pnl: Dict[str, Dict[str, float]] = {
            PnLPeriod.HOURLY.value: defaultdict(float),
            PnLPeriod.DAILY.value: defaultdict(float),
            PnLPeriod.WEEKLY.value: defaultdict(float),
            PnLPeriod.MONTHLY.value: defaultdict(float),
        }

        logger.info(
            "PnLEngine initialized (balance=%.2f, risk_free=%.2f%%)",
            initial_balance, risk_free_rate * 100,
        )

    # -- Record trades --------------------------------------------------------

    def record_trade(self, record: PnLRecord) -> None:
        """Record a completed trade and update all metrics."""
        with self._lock:
            self._records.append(record)
            self._records_by_symbol[record.symbol].append(record)
            if record.strategy_id:
                self._records_by_strategy[record.strategy_id].append(record)

            # Update totals
            self._total_commission += record.commission
            self._total_funding_pnl += record.funding_pnl
            self._commission_by_symbol[record.symbol] += record.commission
            self._funding_by_symbol[record.symbol] += record.funding_pnl

            # Update equity
            self._current_equity += record.net_pnl

            # Update drawdown
            self._update_drawdown()

            # Update period aggregation
            ts = record.timestamp
            hour_key = ts.strftime("%Y-%m-%d %H:00")
            day_key = ts.strftime("%Y-%m-%d")
            week_key = (
                f"{ts.isocalendar()[0]}-W{ts.isocalendar()[1]:02d}"
            )
            month_key = ts.strftime("%Y-%m")

            self._period_pnl[PnLPeriod.HOURLY.value][hour_key] += record.net_pnl
            self._period_pnl[PnLPeriod.DAILY.value][day_key] += record.net_pnl
            self._period_pnl[PnLPeriod.WEEKLY.value][week_key] += record.net_pnl
            self._period_pnl[PnLPeriod.MONTHLY.value][month_key] += record.net_pnl

            # Track daily PnL for returns calculation
            if day_key != self._current_day:
                if self._current_day and self._current_day in self._daily_pnl:
                    day_return = (
                        self._daily_pnl[self._current_day]
                        / max(1.0, self._current_equity - record.net_pnl)
                    )
                    self._daily_returns.append(day_return)
                self._current_day = day_key
                self._daily_pnl[day_key] = 0.0
            self._daily_pnl[day_key] += record.net_pnl

            # Streak tracking
            self._update_streaks(record.is_winner)

            # Auto-snapshot if interval passed
            self._maybe_take_snapshot()

        logger.debug(
            "Trade recorded: %s %s pnl=%.4f (equity=%.2f)",
            record.symbol, record.side, record.net_pnl,
            self._current_equity,
        )

    def record_commission(
        self, symbol: str, amount: float, strategy_id: str = ""
    ) -> None:
        """Record a standalone commission charge (e.g., funding fee)."""
        with self._lock:
            self._total_commission += amount
            self._commission_by_symbol[symbol] += amount
            self._current_equity -= amount
            self._update_drawdown()

    def record_funding(
        self, symbol: str, amount: float, strategy_id: str = ""
    ) -> None:
        """Record a funding rate payment."""
        with self._lock:
            self._total_funding_pnl += amount
            self._funding_by_symbol[symbol] += amount
            self._current_equity += amount
            self._update_drawdown()

    # -- Equity updates -------------------------------------------------------

    def update_equity(
        self,
        unrealized_pnl: float,
        realized_pnl: float = 0.0,
        open_positions: int = 0,
        margin_used: float = 0.0,
    ) -> None:
        """
        Update current equity based on external calculations (e.g., from
        position tracker). Call this periodically to keep equity curve current.
        """
        with self._lock:
            total_realized = sum(r.net_pnl for r in self._records)
            self._current_equity = (
                self._initial_balance + total_realized + unrealized_pnl
            )
            self._update_drawdown()
            self._maybe_take_snapshot(
                unrealized_pnl=unrealized_pnl,
                open_positions=open_positions,
                margin_used=margin_used,
            )

    # -- Drawdown tracking ----------------------------------------------------

    def _update_drawdown(self) -> None:
        """Update drawdown state based on current equity."""
        dd = self._drawdown
        now = datetime.now(timezone.utc)

        if self._current_equity > dd.peak_equity:
            # New peak - recovery complete
            if dd.in_drawdown:
                dd.drawdown_end = now
                recovery_time = (
                    (now - dd.drawdown_start).total_seconds()
                    if dd.drawdown_start else 0.0
                )
                dd.recovery_time_seconds = recovery_time
                if recovery_time > dd.max_recovery_time_seconds:
                    dd.max_recovery_time_seconds = recovery_time
                # Archive this drawdown period
                self._drawdown_periods.append(
                    DrawdownInfo(
                        current_drawdown=0.0,
                        max_drawdown=dd.current_drawdown,
                        max_drawdown_pct=dd.current_drawdown_pct,
                        peak_equity=dd.peak_equity,
                        trough_equity=dd.trough_equity,
                        drawdown_start=dd.drawdown_start,
                        drawdown_end=dd.drawdown_end,
                        recovery_time_seconds=recovery_time,
                    )
                )
                dd.in_drawdown = False

            dd.peak_equity = self._current_equity
            dd.current_drawdown = 0.0
            dd.current_drawdown_pct = 0.0
        else:
            # In drawdown
            dd_amount = dd.peak_equity - self._current_equity
            dd_pct = (dd_amount / dd.peak_equity) * 100 if dd.peak_equity > 0 else 0.0

            if not dd.in_drawdown:
                dd.in_drawdown = True
                dd.drawdown_start = now
                dd.trough_equity = self._current_equity

            dd.current_drawdown = dd_amount
            dd.current_drawdown_pct = dd_pct

            if self._current_equity < dd.trough_equity:
                dd.trough_equity = self._current_equity

            if dd_amount > dd.max_drawdown:
                dd.max_drawdown = dd_amount
            if dd_pct > dd.max_drawdown_pct:
                dd.max_drawdown_pct = dd_pct

    # -- Streak tracking ------------------------------------------------------

    def _update_streaks(self, is_winner: bool) -> None:
        streak_type = "WIN" if is_winner else "LOSS"

        if streak_type == self._current_streak_type:
            self._current_streak += 1
        else:
            self._current_streak_type = streak_type
            self._current_streak = 1

        if is_winner:
            if self._current_streak > self._max_win_streak:
                self._max_win_streak = self._current_streak
        else:
            if self._current_streak > self._max_loss_streak:
                self._max_loss_streak = self._current_streak

    # -- Snapshot management --------------------------------------------------

    def _maybe_take_snapshot(
        self,
        unrealized_pnl: float = 0.0,
        open_positions: int = 0,
        margin_used: float = 0.0,
    ) -> None:
        now = datetime.now(timezone.utc)
        if (
            self._last_snapshot_time is not None
            and (now - self._last_snapshot_time) < self._snapshot_interval
        ):
            return

        total_realized = sum(r.net_pnl for r in self._records)
        snapshot = PnLSnapshot(
            timestamp=now,
            equity=self._current_equity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=total_realized,
            total_commission=self._total_commission,
            total_funding=self._total_funding_pnl,
            drawdown=self._drawdown.current_drawdown,
            drawdown_pct=self._drawdown.current_drawdown_pct,
            open_positions=open_positions,
            margin_used=margin_used,
        )

        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]
        self._last_snapshot_time = now

    def take_snapshot(
        self,
        unrealized_pnl: float = 0.0,
        open_positions: int = 0,
        margin_used: float = 0.0,
    ) -> PnLSnapshot:
        """Force a snapshot now regardless of interval."""
        with self._lock:
            total_realized = sum(r.net_pnl for r in self._records)
            snapshot = PnLSnapshot(
                timestamp=datetime.now(timezone.utc),
                equity=self._current_equity,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=total_realized,
                total_commission=self._total_commission,
                total_funding=self._total_funding_pnl,
                drawdown=self._drawdown.current_drawdown,
                drawdown_pct=self._drawdown.current_drawdown_pct,
                open_positions=open_positions,
                margin_used=margin_used,
            )
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots:]
            self._last_snapshot_time = snapshot.timestamp
            return snapshot

    # -- Metric calculations --------------------------------------------------

    def calculate_metrics(
        self,
        records: Optional[List[PnLRecord]] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics from trade records.
        If records is None, uses all recorded trades.
        """
        with self._lock:
            recs = records if records is not None else list(self._records)

        if not recs:
            return PerformanceMetrics()

        metrics = PerformanceMetrics()
        metrics.total_trades = len(recs)

        wins = [r for r in recs if r.net_pnl > 0]
        losses = [r for r in recs if r.net_pnl < 0]
        breakeven = [r for r in recs if abs(r.net_pnl) < 1e-10]

        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.breakeven_trades = len(breakeven)

        # PnL totals
        metrics.total_gross_pnl = sum(r.gross_pnl for r in recs)
        metrics.total_net_pnl = sum(r.net_pnl for r in recs)
        metrics.total_commission = sum(r.commission for r in recs)
        metrics.total_funding_pnl = sum(r.funding_pnl for r in recs)

        # Extremes
        all_pnl = [r.net_pnl for r in recs]
        metrics.largest_win = max(all_pnl) if all_pnl else 0.0
        metrics.largest_loss = min(all_pnl) if all_pnl else 0.0
        metrics.average_trade_pnl = (
            metrics.total_net_pnl / metrics.total_trades
            if metrics.total_trades > 0 else 0.0
        )

        # Averages
        if wins:
            metrics.average_win = sum(r.net_pnl for r in wins) / len(wins)
        if losses:
            metrics.average_loss = sum(r.net_pnl for r in losses) / len(losses)

        # Win rate
        if metrics.total_trades > 0:
            metrics.win_rate = (
                metrics.winning_trades / metrics.total_trades
            ) * 100.0

        # Profit factor
        total_wins = sum(r.net_pnl for r in wins)
        total_losses = abs(sum(r.net_pnl for r in losses))
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float("inf")

        # Payoff ratio (avg win / avg loss)
        if metrics.average_loss != 0:
            metrics.payoff_ratio = abs(
                metrics.average_win / metrics.average_loss
            )

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        win_rate_dec = metrics.win_rate / 100.0
        loss_rate_dec = 1.0 - win_rate_dec
        metrics.expectancy = (
            (win_rate_dec * metrics.average_win)
            + (loss_rate_dec * metrics.average_loss)
        )

        # Holding time
        holding_times = [r.holding_time_seconds for r in recs if r.holding_time_seconds > 0]
        if holding_times:
            metrics.average_holding_time_seconds = (
                sum(holding_times) / len(holding_times)
            )

        # Trading days
        trade_days = set(r.timestamp.strftime("%Y-%m-%d") for r in recs)
        metrics.total_trading_days = len(trade_days)

        # Risk-adjusted metrics
        metrics.sharpe_ratio = self._calculate_sharpe(recs)
        metrics.sortino_ratio = self._calculate_sortino(recs)

        # Drawdown metrics
        metrics.max_drawdown = self._drawdown.max_drawdown
        metrics.max_drawdown_pct = self._drawdown.max_drawdown_pct
        metrics.max_recovery_time_seconds = (
            self._drawdown.max_recovery_time_seconds
        )

        if self._drawdown_periods:
            metrics.average_drawdown = (
                sum(d.max_drawdown for d in self._drawdown_periods)
                / len(self._drawdown_periods)
            )

        # Calmar ratio = annualized_return / max_drawdown
        if metrics.total_trading_days > 0 and metrics.max_drawdown_pct > 0:
            total_return = (
                (self._current_equity - self._initial_balance)
                / self._initial_balance
            )
            ann_factor = 365.0 / max(1, metrics.total_trading_days)
            metrics.annualized_return = total_return * ann_factor * 100
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct

        # Streaks
        metrics.max_win_streak = self._max_win_streak
        metrics.max_loss_streak = self._max_loss_streak
        metrics.current_streak = self._current_streak
        metrics.current_streak_type = self._current_streak_type

        # By symbol
        symbol_wins: Dict[str, int] = defaultdict(int)
        symbol_total: Dict[str, int] = defaultdict(int)
        for r in recs:
            metrics.pnl_by_symbol[r.symbol] = (
                metrics.pnl_by_symbol.get(r.symbol, 0.0) + r.net_pnl
            )
            metrics.trades_by_symbol[r.symbol] = (
                metrics.trades_by_symbol.get(r.symbol, 0) + 1
            )
            symbol_total[r.symbol] += 1
            if r.is_winner:
                symbol_wins[r.symbol] += 1

        for sym in symbol_total:
            metrics.win_rate_by_symbol[sym] = (
                (symbol_wins[sym] / symbol_total[sym]) * 100.0
                if symbol_total[sym] > 0 else 0.0
            )

        # By strategy
        strategy_wins: Dict[str, int] = defaultdict(int)
        strategy_total: Dict[str, int] = defaultdict(int)
        for r in recs:
            if not r.strategy_id:
                continue
            metrics.pnl_by_strategy[r.strategy_id] = (
                metrics.pnl_by_strategy.get(r.strategy_id, 0.0) + r.net_pnl
            )
            metrics.trades_by_strategy[r.strategy_id] = (
                metrics.trades_by_strategy.get(r.strategy_id, 0) + 1
            )
            strategy_total[r.strategy_id] += 1
            if r.is_winner:
                strategy_wins[r.strategy_id] += 1

        for strat in strategy_total:
            metrics.win_rate_by_strategy[strat] = (
                (strategy_wins[strat] / strategy_total[strat]) * 100.0
                if strategy_total[strat] > 0 else 0.0
            )

        return metrics

    def _calculate_sharpe(
        self,
        records: List[PnLRecord],
        annualization_factor: float = 252.0,
    ) -> float:
        """
        Calculate Sharpe ratio from trade returns.
        Uses daily returns where possible, otherwise per-trade returns.
        """
        if len(records) < 2:
            return 0.0

        # Use daily returns if we have enough days
        with self._lock:
            returns = list(self._daily_returns)

        if len(returns) < 2:
            # Fall back to per-trade returns
            returns = []
            for r in records:
                notional = r.quantity * r.entry_price
                if notional > 0:
                    returns.append(r.net_pnl / notional)

        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        daily_rf = self._risk_free_rate / annualization_factor

        variance = sum(
            (r - mean_return) ** 2 for r in returns
        ) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev <= 0:
            return 0.0

        sharpe = (
            (mean_return - daily_rf) / std_dev
        ) * math.sqrt(annualization_factor)

        return round(sharpe, 4)

    def _calculate_sortino(
        self,
        records: List[PnLRecord],
        annualization_factor: float = 252.0,
    ) -> float:
        """
        Calculate Sortino ratio (only counts downside volatility).
        """
        if len(records) < 2:
            return 0.0

        with self._lock:
            returns = list(self._daily_returns)

        if len(returns) < 2:
            returns = []
            for r in records:
                notional = r.quantity * r.entry_price
                if notional > 0:
                    returns.append(r.net_pnl / notional)

        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        daily_rf = self._risk_free_rate / annualization_factor

        # Downside deviation - only negative returns
        downside_returns = [r for r in returns if r < daily_rf]
        if not downside_returns:
            return float("inf") if mean_return > daily_rf else 0.0

        downside_variance = sum(
            (r - daily_rf) ** 2 for r in downside_returns
        ) / len(downside_returns)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

        if downside_dev <= 0:
            return 0.0

        sortino = (
            (mean_return - daily_rf) / downside_dev
        ) * math.sqrt(annualization_factor)

        return round(sortino, 4)

    # -- Period PnL -----------------------------------------------------------

    def get_period_pnl(
        self,
        period: PnLPeriod,
        limit: int = 30,
    ) -> List[Tuple[str, float]]:
        """Get PnL aggregated by period (daily, weekly, etc)."""
        with self._lock:
            pnl_data = dict(
                self._period_pnl.get(period.value, {})
            )
        items = sorted(pnl_data.items(), reverse=True)
        return items[:limit]

    def get_daily_pnl(self, limit: int = 30) -> List[Tuple[str, float]]:
        return self.get_period_pnl(PnLPeriod.DAILY, limit)

    def get_weekly_pnl(self, limit: int = 12) -> List[Tuple[str, float]]:
        return self.get_period_pnl(PnLPeriod.WEEKLY, limit)

    def get_monthly_pnl(self, limit: int = 12) -> List[Tuple[str, float]]:
        return self.get_period_pnl(PnLPeriod.MONTHLY, limit)

    # -- Equity curve ---------------------------------------------------------

    def get_equity_curve(
        self, limit: int = 0
    ) -> List[PnLSnapshot]:
        """Return equity curve snapshots."""
        with self._lock:
            snapshots = list(self._snapshots)
        if limit > 0:
            snapshots = snapshots[-limit:]
        return snapshots

    def get_equity_values(self) -> List[Tuple[datetime, float]]:
        """Get timestamp-equity pairs for plotting."""
        with self._lock:
            return [
                (s.timestamp, s.equity) for s in self._snapshots
            ]

    # -- Query methods --------------------------------------------------------

    def get_drawdown_info(self) -> DrawdownInfo:
        with self._lock:
            return DrawdownInfo(
                current_drawdown=self._drawdown.current_drawdown,
                current_drawdown_pct=self._drawdown.current_drawdown_pct,
                max_drawdown=self._drawdown.max_drawdown,
                max_drawdown_pct=self._drawdown.max_drawdown_pct,
                peak_equity=self._drawdown.peak_equity,
                trough_equity=self._drawdown.trough_equity,
                in_drawdown=self._drawdown.in_drawdown,
                drawdown_start=self._drawdown.drawdown_start,
                recovery_time_seconds=self._drawdown.recovery_time_seconds,
                max_recovery_time_seconds=self._drawdown.max_recovery_time_seconds,
            )

    def get_records(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[PnLRecord]:
        """Query trade records with filters."""
        with self._lock:
            if symbol:
                recs = list(self._records_by_symbol.get(symbol, []))
            elif strategy_id:
                recs = list(
                    self._records_by_strategy.get(strategy_id, [])
                )
            else:
                recs = list(self._records)

        if since:
            recs = [r for r in recs if r.timestamp >= since]
        if until:
            recs = [r for r in recs if r.timestamp <= until]

        recs.sort(key=lambda r: r.timestamp, reverse=True)
        return recs[:limit]

    def get_commission_breakdown(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._commission_by_symbol)

    def get_funding_breakdown(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._funding_by_symbol)

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def initial_balance(self) -> float:
        return self._initial_balance

    @property
    def total_return(self) -> float:
        return self._current_equity - self._initial_balance

    @property
    def total_return_pct(self) -> float:
        if self._initial_balance <= 0:
            return 0.0
        return (self.total_return / self._initial_balance) * 100.0

    @property
    def total_trades(self) -> int:
        with self._lock:
            return len(self._records)

    # -- Export / Import ------------------------------------------------------

    def export_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._records]

    def export_snapshots(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._snapshots]

    def import_records(self, record_dicts: List[Dict[str, Any]]) -> int:
        """Import historical trade records and recalculate metrics."""
        imported = 0
        for data in record_dicts:
            record = PnLRecord(**{
                k: v for k, v in data.items()
                if k != "timestamp"
            })
            if "timestamp" in data and isinstance(data["timestamp"], str):
                record.timestamp = datetime.fromisoformat(data["timestamp"])
            self.record_trade(record)
            imported += 1
        logger.info("Imported %d PnL records", imported)
        return imported

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of PnL state."""
        metrics = self.calculate_metrics()
        return {
            "equity": self._current_equity,
            "initial_balance": self._initial_balance,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "total_commission": self._total_commission,
            "total_funding_pnl": self._total_funding_pnl,
            "expectancy": metrics.expectancy,
        }
