"""
Mefai Signal Engine - Copy Trading System

Master/follower architecture for copy trading. Masters publish trade signals,
followers subscribe and automatically replicate trades with configurable size
adjustment, delay, filtering, and independent risk limits.

Supports multiple masters per follower, performance tracking per master,
and follower-side risk controls that operate independently from master positions.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("mefai.autotrade.portfolio.copy_trading")


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class CopyStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TradeSignal:
    """A trade signal published by a master trader."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    master_id: str = ""
    symbol: str = ""
    direction: TradeDirection = TradeDirection.LONG
    price: float = 0.0
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_name: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.direction in (TradeDirection.LONG, TradeDirection.SHORT)

    @property
    def is_exit(self) -> bool:
        return self.direction in (TradeDirection.CLOSE_LONG, TradeDirection.CLOSE_SHORT)


@dataclass
class CopiedTrade:
    """Record of a trade copied by a follower."""
    copy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    signal_id: str = ""
    master_id: str = ""
    follower_id: str = ""
    symbol: str = ""
    direction: TradeDirection = TradeDirection.LONG
    master_price: float = 0.0
    follower_price: float = 0.0         # actual execution price
    master_quantity: float = 0.0
    follower_quantity: float = 0.0       # adjusted quantity
    stop_loss: float = 0.0
    take_profit: float = 0.0
    delay_seconds: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    status: str = "pending"             # pending, executed, failed, skipped
    timestamp: float = field(default_factory=time.time)
    close_time: float = 0.0
    close_price: float = 0.0
    skip_reason: str = ""


@dataclass
class MasterStats:
    """Performance statistics for a master trader."""
    master_id: str
    total_signals: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_time: float = 0.0       # seconds
    best_trade: float = 0.0
    worst_trade: float = 0.0
    active_since: float = field(default_factory=time.time)
    last_signal_time: float = 0.0
    followers_count: int = 0
    equity_curve: List[float] = field(default_factory=list)

    def update(self, pnl: float, holding_time: float = 0.0) -> None:
        self.total_trades += 1
        self.total_pnl += pnl
        self.equity_curve.append(self.total_pnl)

        if pnl > 0:
            self.winning_trades += 1
            self.best_trade = max(self.best_trade, pnl)
        else:
            self.losing_trades += 1
            self.worst_trade = min(self.worst_trade, pnl)

        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        wins = [p for p in self.equity_curve if p > 0]
        losses = [p for p in self.equity_curve if p <= 0]
        self.avg_win = float(np.mean(wins)) if wins else 0.0
        self.avg_loss = float(np.mean(losses)) if losses else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        total_win = sum(wins) if wins else 0.0
        self.profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        if holding_time > 0:
            self.avg_holding_time = (
                (self.avg_holding_time * (self.total_trades - 1) + holding_time)
                / self.total_trades
            )

        # Max drawdown
        if len(self.equity_curve) >= 2:
            peak = self.equity_curve[0]
            max_dd = 0.0
            for val in self.equity_curve:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown = max_dd


@dataclass
class FollowerConfig:
    """Configuration for a copy trading follower."""
    follower_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""

    # Size adjustment
    size_multiplier: float = 1.0        # 0.5 = half size, 2.0 = double
    fixed_size: Optional[float] = None  # if set, use fixed size regardless of master
    max_position_size: float = 0.0      # 0 = no limit

    # Delay
    delay_seconds: float = 0.0          # wait N seconds before copying

    # Filters
    allowed_symbols: Optional[Set[str]] = None          # None = all allowed
    blocked_symbols: Set[str] = field(default_factory=set)
    allowed_strategies: Optional[Set[str]] = None       # None = all allowed
    allowed_directions: Optional[Set[TradeDirection]] = None  # None = all
    min_confidence: float = 0.0         # minimum signal confidence to copy

    # Risk limits (independent from master)
    max_daily_trades: int = 50
    max_daily_loss: float = 0.0         # 0 = no limit, in absolute terms
    max_drawdown: float = 0.0           # 0 = no limit
    max_open_positions: int = 10
    max_exposure_per_symbol: float = 0.0  # max dollar exposure per symbol

    # Copy control
    copy_stop_loss: bool = True         # copy master's SL
    copy_take_profit: bool = True       # copy master's TP
    override_stop_loss: Optional[float] = None   # override SL (in %)
    override_take_profit: Optional[float] = None # override TP (in %)

    # Status
    status: CopyStatus = CopyStatus.ACTIVE
    subscribed_masters: Set[str] = field(default_factory=set)

    def should_copy(self, signal: TradeSignal) -> Tuple[bool, str]:
        """Determine if this follower should copy a given signal."""
        if self.status != CopyStatus.ACTIVE:
            return False, f"follower_{self.status.value}"

        if signal.master_id not in self.subscribed_masters:
            return False, "not_subscribed"

        if self.allowed_symbols and signal.symbol not in self.allowed_symbols:
            return False, f"symbol_{signal.symbol}_not_allowed"

        if signal.symbol in self.blocked_symbols:
            return False, f"symbol_{signal.symbol}_blocked"

        if self.allowed_strategies and signal.strategy_name not in self.allowed_strategies:
            return False, f"strategy_{signal.strategy_name}_not_allowed"

        if self.allowed_directions and signal.direction not in self.allowed_directions:
            return False, f"direction_{signal.direction.value}_not_allowed"

        if signal.confidence < self.min_confidence:
            return False, f"confidence_{signal.confidence:.2f}_below_{self.min_confidence:.2f}"

        return True, "ok"

    def calculate_quantity(self, master_quantity: float) -> float:
        """Calculate follower's trade quantity based on config."""
        if self.fixed_size is not None:
            qty = self.fixed_size
        else:
            qty = master_quantity * self.size_multiplier

        if self.max_position_size > 0:
            qty = min(qty, self.max_position_size)

        return qty


@dataclass
class MasterTrader:
    """A master trader that publishes signals."""
    master_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    strategies: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    stats: MasterStats = field(default=None)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.stats is None:
            self.stats = MasterStats(master_id=self.master_id)


# ---------------------------------------------------------------------------
# Copy Trading Hub
# ---------------------------------------------------------------------------

class CopyTradingHub:
    """
    Central hub for copy trading. Manages master traders, followers,
    signal routing, trade copying, and performance tracking.
    """

    def __init__(
        self,
        trade_executor: Optional[Callable[[CopiedTrade], bool]] = None,
        on_copy: Optional[Callable[[CopiedTrade], None]] = None,
    ):
        self._masters: Dict[str, MasterTrader] = {}
        self._followers: Dict[str, FollowerConfig] = {}
        self._trade_executor = trade_executor
        self._on_copy = on_copy

        # Trade history
        self._copied_trades: List[CopiedTrade] = []
        self._pending_signals: List[Tuple[TradeSignal, float]] = []  # (signal, execute_at_time)

        # Follower state tracking
        self._follower_daily_trades: Dict[str, int] = defaultdict(int)
        self._follower_daily_pnl: Dict[str, float] = defaultdict(float)
        self._follower_open_positions: Dict[str, List[CopiedTrade]] = defaultdict(list)

        # Signal subscribers (master_id -> set of follower_ids)
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)

        logger.info("Copy Trading Hub initialized")

    # ------------------------------------------------------------------
    # Master management
    # ------------------------------------------------------------------

    def register_master(self, master: MasterTrader) -> None:
        """Register a new master trader."""
        self._masters[master.master_id] = master
        logger.info("Master trader registered: %s (%s)", master.name, master.master_id)

    def deactivate_master(self, master_id: str) -> None:
        """Deactivate a master (stops new signals, existing positions unaffected)."""
        if master_id in self._masters:
            self._masters[master_id].is_active = False
            logger.info("Master trader deactivated: %s", master_id)

    def get_master(self, master_id: str) -> Optional[MasterTrader]:
        return self._masters.get(master_id)

    def list_masters(self) -> List[Dict[str, Any]]:
        """List all masters with their stats."""
        result = []
        for m in self._masters.values():
            result.append({
                "master_id": m.master_id,
                "name": m.name,
                "is_active": m.is_active,
                "total_trades": m.stats.total_trades,
                "win_rate": m.stats.win_rate,
                "total_pnl": m.stats.total_pnl,
                "profit_factor": m.stats.profit_factor,
                "max_drawdown": m.stats.max_drawdown,
                "followers": len(self._subscriptions.get(m.master_id, set())),
                "strategies": m.strategies,
            })
        return result

    # ------------------------------------------------------------------
    # Follower management
    # ------------------------------------------------------------------

    def register_follower(self, config: FollowerConfig) -> None:
        """Register a new follower."""
        self._followers[config.follower_id] = config
        logger.info("Follower registered: %s (%s)", config.name, config.follower_id)

    def subscribe(self, follower_id: str, master_id: str) -> bool:
        """Subscribe a follower to a master's signals."""
        if follower_id not in self._followers:
            logger.error("Follower %s not found", follower_id)
            return False
        if master_id not in self._masters:
            logger.error("Master %s not found", master_id)
            return False

        self._followers[follower_id].subscribed_masters.add(master_id)
        self._subscriptions[master_id].add(follower_id)
        self._masters[master_id].stats.followers_count = len(self._subscriptions[master_id])

        logger.info("Follower %s subscribed to master %s", follower_id, master_id)
        return True

    def unsubscribe(self, follower_id: str, master_id: str) -> bool:
        """Unsubscribe a follower from a master."""
        if follower_id in self._followers:
            self._followers[follower_id].subscribed_masters.discard(master_id)
        if master_id in self._subscriptions:
            self._subscriptions[master_id].discard(follower_id)
            if master_id in self._masters:
                self._masters[master_id].stats.followers_count = len(self._subscriptions[master_id])

        logger.info("Follower %s unsubscribed from master %s", follower_id, master_id)
        return True

    def pause_follower(self, follower_id: str) -> None:
        if follower_id in self._followers:
            self._followers[follower_id].status = CopyStatus.PAUSED

    def resume_follower(self, follower_id: str) -> None:
        if follower_id in self._followers:
            self._followers[follower_id].status = CopyStatus.ACTIVE

    # ------------------------------------------------------------------
    # Signal publishing and copying
    # ------------------------------------------------------------------

    def publish_signal(self, signal: TradeSignal) -> List[CopiedTrade]:
        """
        Publish a trade signal from a master. Routes to all subscribed
        followers, applies filters and adjustments, and executes copies.
        Returns list of copied trades.
        """
        master = self._masters.get(signal.master_id)
        if master is None:
            logger.error("Signal from unknown master: %s", signal.master_id)
            return []

        if not master.is_active:
            logger.warning("Signal from inactive master: %s", signal.master_id)
            return []

        master.stats.total_signals += 1
        master.stats.last_signal_time = time.time()

        # Route to followers
        follower_ids = self._subscriptions.get(signal.master_id, set())
        copied_trades = []

        for fid in follower_ids:
            follower = self._followers.get(fid)
            if follower is None:
                continue

            # Check if follower should copy this signal
            should_copy, reason = follower.should_copy(signal)
            if not should_copy:
                logger.debug("Follower %s skipped signal: %s", fid, reason)
                skipped = CopiedTrade(
                    signal_id=signal.signal_id,
                    master_id=signal.master_id,
                    follower_id=fid,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    master_price=signal.price,
                    master_quantity=signal.quantity,
                    status="skipped",
                    skip_reason=reason,
                )
                self._copied_trades.append(skipped)
                continue

            # Check follower risk limits
            risk_ok, risk_reason = self._check_follower_risk(fid, follower, signal)
            if not risk_ok:
                logger.info("Follower %s blocked by risk: %s", fid, risk_reason)
                skipped = CopiedTrade(
                    signal_id=signal.signal_id,
                    master_id=signal.master_id,
                    follower_id=fid,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    master_price=signal.price,
                    master_quantity=signal.quantity,
                    status="skipped",
                    skip_reason=risk_reason,
                )
                self._copied_trades.append(skipped)
                continue

            # Calculate adjusted quantity
            follower_qty = follower.calculate_quantity(signal.quantity)

            # Build the copied trade
            copied = CopiedTrade(
                signal_id=signal.signal_id,
                master_id=signal.master_id,
                follower_id=fid,
                symbol=signal.symbol,
                direction=signal.direction,
                master_price=signal.price,
                follower_price=signal.price,  # will be updated on execution
                master_quantity=signal.quantity,
                follower_quantity=follower_qty,
                delay_seconds=follower.delay_seconds,
            )

            # Stop loss and take profit
            if follower.copy_stop_loss and signal.stop_loss > 0:
                copied.stop_loss = signal.stop_loss
            if follower.override_stop_loss is not None and signal.price > 0:
                sl_pct = follower.override_stop_loss / 100.0
                if signal.direction == TradeDirection.LONG:
                    copied.stop_loss = signal.price * (1 - sl_pct)
                else:
                    copied.stop_loss = signal.price * (1 + sl_pct)

            if follower.copy_take_profit and signal.take_profit > 0:
                copied.take_profit = signal.take_profit
            if follower.override_take_profit is not None and signal.price > 0:
                tp_pct = follower.override_take_profit / 100.0
                if signal.direction == TradeDirection.LONG:
                    copied.take_profit = signal.price * (1 + tp_pct)
                else:
                    copied.take_profit = signal.price * (1 - tp_pct)

            # Execute (with delay if configured)
            if follower.delay_seconds > 0:
                execute_at = time.time() + follower.delay_seconds
                self._pending_signals.append((signal, execute_at))
                copied.status = "pending"
                logger.info(
                    "Copy trade queued with %.1fs delay: %s -> follower %s",
                    follower.delay_seconds, signal.symbol, fid,
                )
            else:
                copied = self._execute_copy(copied)

            self._copied_trades.append(copied)
            if copied.status == "executed":
                copied_trades.append(copied)

                # Track follower state
                self._follower_daily_trades[fid] += 1
                if signal.is_entry:
                    self._follower_open_positions[fid].append(copied)

        return copied_trades

    def process_pending(self) -> List[CopiedTrade]:
        """Process delayed copy trades that are due for execution."""
        now = time.time()
        executed = []
        remaining = []

        for signal, execute_at in self._pending_signals:
            if now >= execute_at:
                # Find the pending copied trade and execute
                for ct in self._copied_trades:
                    if ct.signal_id == signal.signal_id and ct.status == "pending":
                        ct = self._execute_copy(ct)
                        if ct.status == "executed":
                            executed.append(ct)
                        break
            else:
                remaining.append((signal, execute_at))

        self._pending_signals = remaining
        return executed

    def close_copy(
        self,
        follower_id: str,
        symbol: str,
        close_price: float,
    ) -> Optional[CopiedTrade]:
        """Close a copied position and update PnL."""
        positions = self._follower_open_positions.get(follower_id, [])

        for i, pos in enumerate(positions):
            if pos.symbol == symbol and pos.status == "executed":
                # Calculate PnL
                if pos.direction == TradeDirection.LONG:
                    pnl = (close_price - pos.follower_price) * pos.follower_quantity
                elif pos.direction == TradeDirection.SHORT:
                    pnl = (pos.follower_price - close_price) * pos.follower_quantity
                else:
                    pnl = 0.0

                pos.pnl = pnl
                pos.close_price = close_price
                pos.close_time = time.time()
                pos.status = "closed"

                # Update follower PnL
                self._follower_daily_pnl[follower_id] += pnl

                # Update master stats
                master = self._masters.get(pos.master_id)
                if master:
                    holding_time = pos.close_time - pos.timestamp
                    master.stats.update(pnl, holding_time)

                # Remove from open positions
                positions.pop(i)

                logger.info(
                    "Copy trade closed: follower=%s symbol=%s pnl=%.4f",
                    follower_id, symbol, pnl,
                )

                if self._on_copy:
                    self._on_copy(pos)

                return pos

        return None

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _check_follower_risk(
        self,
        follower_id: str,
        config: FollowerConfig,
        signal: TradeSignal,
    ) -> Tuple[bool, str]:
        """Check follower-specific risk limits."""

        # Daily trade limit
        daily_trades = self._follower_daily_trades.get(follower_id, 0)
        if daily_trades >= config.max_daily_trades:
            return False, f"daily_trade_limit_{config.max_daily_trades}"

        # Daily loss limit
        if config.max_daily_loss > 0:
            daily_pnl = self._follower_daily_pnl.get(follower_id, 0.0)
            if daily_pnl <= -config.max_daily_loss:
                return False, f"daily_loss_limit_${config.max_daily_loss}"

        # Max open positions
        open_positions = self._follower_open_positions.get(follower_id, [])
        if len(open_positions) >= config.max_open_positions:
            return False, f"max_positions_{config.max_open_positions}"

        # Max exposure per symbol
        if config.max_exposure_per_symbol > 0:
            symbol_exposure = sum(
                pos.follower_quantity * pos.follower_price
                for pos in open_positions
                if pos.symbol == signal.symbol
            )
            new_exposure = config.calculate_quantity(signal.quantity) * signal.price
            if symbol_exposure + new_exposure > config.max_exposure_per_symbol:
                return False, f"symbol_exposure_limit_{signal.symbol}"

        return True, "ok"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_copy(self, trade: CopiedTrade) -> CopiedTrade:
        """Execute a single copy trade."""
        if self._trade_executor:
            success = self._trade_executor(trade)
            if success:
                trade.status = "executed"
                trade.slippage = abs(trade.follower_price - trade.master_price)
                logger.info(
                    "Copy trade executed: %s %s qty=%.6f price=%.4f (master=%.4f)",
                    trade.direction.value, trade.symbol,
                    trade.follower_quantity, trade.follower_price, trade.master_price,
                )
            else:
                trade.status = "failed"
                logger.warning(
                    "Copy trade execution failed: %s %s",
                    trade.direction.value, trade.symbol,
                )
        else:
            # No executor - mark as executed (simulation/dry run)
            trade.status = "executed"
            trade.follower_price = trade.master_price

        if self._on_copy and trade.status == "executed":
            self._on_copy(trade)

        return trade

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def get_master_stats(self, master_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stats for a master trader."""
        master = self._masters.get(master_id)
        if master is None:
            return None

        stats = master.stats
        return {
            "master_id": master_id,
            "name": master.name,
            "is_active": master.is_active,
            "total_signals": stats.total_signals,
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "win_rate": stats.win_rate,
            "total_pnl": stats.total_pnl,
            "profit_factor": stats.profit_factor,
            "max_drawdown": stats.max_drawdown,
            "avg_holding_time_min": stats.avg_holding_time / 60,
            "best_trade": stats.best_trade,
            "worst_trade": stats.worst_trade,
            "followers": len(self._subscriptions.get(master_id, set())),
            "active_since": stats.active_since,
            "last_signal": stats.last_signal_time,
        }

    def get_follower_stats(self, follower_id: str) -> Optional[Dict[str, Any]]:
        """Get stats for a follower."""
        config = self._followers.get(follower_id)
        if config is None:
            return None

        trades = [t for t in self._copied_trades if t.follower_id == follower_id]
        executed = [t for t in trades if t.status in ("executed", "closed")]
        closed = [t for t in trades if t.status == "closed"]

        total_pnl = sum(t.pnl for t in closed)
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]

        return {
            "follower_id": follower_id,
            "name": config.name,
            "status": config.status.value,
            "subscribed_masters": list(config.subscribed_masters),
            "total_signals_received": len(trades),
            "total_executed": len(executed),
            "total_skipped": len([t for t in trades if t.status == "skipped"]),
            "total_closed": len(closed),
            "open_positions": len(self._follower_open_positions.get(follower_id, [])),
            "total_pnl": total_pnl,
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "avg_slippage": float(np.mean([t.slippage for t in executed])) if executed else 0.0,
            "daily_trades": self._follower_daily_trades.get(follower_id, 0),
            "daily_pnl": self._follower_daily_pnl.get(follower_id, 0.0),
        }

    def get_leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get master trader leaderboard ranked by performance."""
        masters = []
        for m in self._masters.values():
            if m.stats.total_trades > 0:
                masters.append({
                    "master_id": m.master_id,
                    "name": m.name,
                    "total_pnl": m.stats.total_pnl,
                    "win_rate": m.stats.win_rate,
                    "profit_factor": m.stats.profit_factor,
                    "total_trades": m.stats.total_trades,
                    "max_drawdown": m.stats.max_drawdown,
                    "followers": len(self._subscriptions.get(m.master_id, set())),
                    "sharpe_ratio": m.stats.sharpe_ratio,
                })

        # Sort by PnL
        masters.sort(key=lambda x: x["total_pnl"], reverse=True)
        return masters[:top_n]

    # ------------------------------------------------------------------
    # Daily reset and status
    # ------------------------------------------------------------------

    def reset_daily_counters(self) -> None:
        """Reset daily trade counters and PnL for all followers."""
        self._follower_daily_trades.clear()
        self._follower_daily_pnl.clear()
        logger.info("Daily counters reset for all followers")

    def get_status(self) -> Dict[str, Any]:
        """Overall copy trading hub status."""
        return {
            "masters": len(self._masters),
            "active_masters": sum(1 for m in self._masters.values() if m.is_active),
            "followers": len(self._followers),
            "active_followers": sum(
                1 for f in self._followers.values() if f.status == CopyStatus.ACTIVE
            ),
            "total_subscriptions": sum(len(s) for s in self._subscriptions.values()),
            "total_copied_trades": len(self._copied_trades),
            "pending_signals": len(self._pending_signals),
            "total_open_positions": sum(
                len(p) for p in self._follower_open_positions.values()
            ),
        }
