"""
Mefai Signal Engine - Multi-Strategy Composite

Combines multiple strategies with configurable weights, voting system,
conflict resolution, per-strategy performance tracking, dynamic weight
adjustment based on recent performance, and strategy correlation monitoring.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import numpy as np
from collections import deque

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyVote:
    """A signal vote from one sub-strategy."""
    strategy_name: str
    signal_type: SignalType
    confidence: float
    weight: float
    price: float
    stop_loss: float
    take_profit: float
    quantity: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformanceTracker:
    """Track individual strategy performance within the composite."""
    name: str
    weight: float
    initial_weight: float
    total_signals: int = 0
    profitable_signals: int = 0
    total_pnl: float = 0.0
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_win: deque = field(default_factory=lambda: deque(maxlen=50))
    sharpe_ratio: float = 0.0
    correlation_with_others: Dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.profitable_signals / self.total_signals if self.total_signals > 0 else 0.0

    @property
    def recent_win_rate(self) -> float:
        return sum(self.recent_win) / len(self.recent_win) if self.recent_win else 0.0

    def record_signal(self, pnl: float) -> None:
        self.total_signals += 1
        self.total_pnl += pnl
        self.recent_pnl.append(pnl)
        if pnl > 0:
            self.profitable_signals += 1
            self.recent_win.append(1)
        else:
            self.recent_win.append(0)

        # Update Sharpe ratio
        if len(self.recent_pnl) >= 10:
            returns = list(self.recent_pnl)
            mean_ret = float(np.mean(returns))
            std_ret = float(np.std(returns))
            self.sharpe_ratio = mean_ret / std_ret if std_ret > 0 else 0.0


class CompositeStrategy(BaseStrategy):
    """
    Multi-Strategy Composite

    Orchestrates multiple trading strategies and combines their signals:
    1. Weighted combination of strategy signals
    2. Voting system - trade when majority of strategies agree
    3. Conflict resolution - reduce size or skip on disagreement
    4. Per-strategy performance tracking
    5. Dynamic weight adjustment based on recent performance
    6. Strategy correlation monitoring to avoid redundancy
    """

    name = "Composite"
    description = "Multi-strategy composite with weighted voting and conflict resolution"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Voting
        self.voting_threshold: float = self.params.get("voting_threshold", 0.6)
        self.min_strategies: int = self.params.get("min_strategies", 2)
        self.conflict_action: str = self.params.get("conflict_action", "reduce")  # "skip" or "reduce"
        self.conflict_size_reduction: float = self.params.get("conflict_size_reduction", 0.5)

        # Weight adjustment
        self.enable_dynamic_weights: bool = self.params.get("enable_dynamic_weights", True)
        self.weight_adjustment_interval: int = self.params.get("weight_adjustment_interval", 20)
        self.min_weight: float = self.params.get("min_weight", 0.1)
        self.max_weight: float = self.params.get("max_weight", 3.0)

        # Correlation
        self.max_correlation: float = self.params.get("max_correlation", 0.8)
        self.correlation_lookback: int = self.params.get("correlation_lookback", 30)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_mult", 2.0)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # Sub-strategies
        self._sub_strategies: List[BaseStrategy] = []
        self._strategy_trackers: Dict[str, StrategyPerformanceTracker] = {}
        self._signal_history: Dict[str, deque] = {}  # for correlation
        self._signals_since_adjustment: int = 0
        self._last_composite_signal: Optional[StrategySignal] = None

    # ------------------------------------------------------------------
    # Sub-strategy management
    # ------------------------------------------------------------------

    def add_strategy(
        self, strategy: BaseStrategy, weight: float = 1.0,
    ) -> None:
        """Add a sub-strategy with a weight."""
        self._sub_strategies.append(strategy)
        name = strategy.name
        self._strategy_trackers[name] = StrategyPerformanceTracker(
            name=name,
            weight=weight,
            initial_weight=weight,
        )
        self._signal_history[name] = deque(maxlen=self.correlation_lookback)
        self._logger.info("Added sub-strategy: %s (weight=%.2f)", name, weight)

    def remove_strategy(self, name: str) -> bool:
        """Remove a sub-strategy by name."""
        for i, strat in enumerate(self._sub_strategies):
            if strat.name == name:
                self._sub_strategies.pop(i)
                self._strategy_trackers.pop(name, None)
                self._signal_history.pop(name, None)
                return True
        return False

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def _collect_votes(self, candles: np.ndarray) -> List[StrategyVote]:
        """Collect signal votes from all sub-strategies."""
        votes: List[StrategyVote] = []

        for strategy in self._sub_strategies:
            tracker = self._strategy_trackers.get(strategy.name)
            if tracker is None:
                continue

            try:
                signal = strategy.generate_signal(candles)
            except Exception as e:
                self._logger.error("Strategy %s error: %s", strategy.name, e)
                continue

            if signal is not None and signal.signal_type != SignalType.NEUTRAL:
                votes.append(StrategyVote(
                    strategy_name=strategy.name,
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    weight=tracker.weight,
                    price=signal.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    quantity=signal.quantity,
                    reason=signal.reason,
                    metadata=signal.metadata,
                ))

                # Record vote direction for correlation tracking
                direction = 1 if signal.signal_type in (SignalType.LONG, SignalType.CLOSE_SHORT) else -1
                self._signal_history[strategy.name].append(direction)
            else:
                self._signal_history[strategy.name].append(0)

        return votes

    def _resolve_votes(
        self, votes: List[StrategyVote],
    ) -> Optional[StrategySignal]:
        """
        Resolve multiple strategy votes into a single signal.
        Uses weighted voting with conflict resolution.
        """
        if not votes:
            return None

        # Separate by direction
        long_votes = [v for v in votes if v.signal_type in (SignalType.LONG,)]
        short_votes = [v for v in votes if v.signal_type in (SignalType.SHORT,)]
        close_long_votes = [v for v in votes if v.signal_type == SignalType.CLOSE_LONG]
        close_short_votes = [v for v in votes if v.signal_type == SignalType.CLOSE_SHORT]

        # Priority: close signals over open signals
        if close_long_votes and self.has_position and self.position.side == "LONG":
            weighted_confidence = sum(v.confidence * v.weight for v in close_long_votes)
            total_weight = sum(v.weight for v in close_long_votes)
            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
            reasons = [v.reason for v in close_long_votes]
            return StrategySignal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.symbol,
                price=close_long_votes[0].price,
                confidence=avg_confidence,
                reason=f"Composite CLOSE_LONG ({len(close_long_votes)} strategies): {'; '.join(reasons[:3])}",
                metadata={"voting_strategies": [v.strategy_name for v in close_long_votes]},
            )

        if close_short_votes and self.has_position and self.position.side == "SHORT":
            weighted_confidence = sum(v.confidence * v.weight for v in close_short_votes)
            total_weight = sum(v.weight for v in close_short_votes)
            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
            reasons = [v.reason for v in close_short_votes]
            return StrategySignal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.symbol,
                price=close_short_votes[0].price,
                confidence=avg_confidence,
                reason=f"Composite CLOSE_SHORT ({len(close_short_votes)} strategies): {'; '.join(reasons[:3])}",
                metadata={"voting_strategies": [v.strategy_name for v in close_short_votes]},
            )

        # Entry signals - need voting threshold
        total_strategies = len(self._sub_strategies)
        if total_strategies == 0:
            return None

        long_weight = sum(v.weight for v in long_votes)
        short_weight = sum(v.weight for v in short_votes)
        total_entry_weight = long_weight + short_weight

        if total_entry_weight == 0:
            return None

        # Check for conflict
        has_conflict = len(long_votes) > 0 and len(short_votes) > 0

        if has_conflict:
            if self.conflict_action == "skip":
                self._logger.debug("Conflict detected - skipping trade")
                return None

        # Determine winning direction
        if long_weight > short_weight:
            winning_votes = long_votes
            signal_type = SignalType.LONG
            losing_weight = short_weight
        elif short_weight > long_weight:
            winning_votes = short_votes
            signal_type = SignalType.SHORT
            losing_weight = long_weight
        else:
            return None

        # Check voting threshold
        vote_pct = len(winning_votes) / total_strategies
        if vote_pct < self.voting_threshold:
            return None

        if len(winning_votes) < self.min_strategies:
            return None

        # Aggregate signal parameters
        total_w = sum(v.weight for v in winning_votes)
        avg_confidence = sum(v.confidence * v.weight for v in winning_votes) / total_w
        avg_price = sum(v.price * v.weight for v in winning_votes) / total_w
        avg_sl = sum(v.stop_loss * v.weight for v in winning_votes) / total_w if all(v.stop_loss > 0 for v in winning_votes) else 0
        avg_tp = sum(v.take_profit * v.weight for v in winning_votes) / total_w if all(v.take_profit > 0 for v in winning_votes) else 0
        avg_qty = sum(v.quantity * v.weight for v in winning_votes) / total_w

        # Reduce size on conflict
        if has_conflict and self.conflict_action == "reduce":
            avg_qty *= self.conflict_size_reduction
            avg_confidence *= 0.8
            self._logger.info("Conflict - reducing size by %.0f%%",
                              (1 - self.conflict_size_reduction) * 100)

        reasons = [f"{v.strategy_name}:{v.reason}" for v in winning_votes[:3]]

        signal = StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=avg_price,
            quantity=avg_qty,
            stop_loss=avg_sl,
            take_profit=avg_tp,
            confidence=min(0.95, avg_confidence),
            reason=f"Composite {signal_type.value}: {len(winning_votes)}/{total_strategies} agree - {'; '.join(reasons)}",
            metadata={
                "voting_strategies": [v.strategy_name for v in winning_votes],
                "vote_count": len(winning_votes),
                "total_strategies": total_strategies,
                "vote_pct": vote_pct,
                "had_conflict": has_conflict,
                "long_weight": long_weight,
                "short_weight": short_weight,
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Dynamic weight adjustment
    # ------------------------------------------------------------------

    def _adjust_weights(self) -> None:
        """Adjust strategy weights based on recent performance."""
        if not self.enable_dynamic_weights:
            return

        for name, tracker in self._strategy_trackers.items():
            if tracker.total_signals < 10:
                continue

            # Increase weight for high performers, decrease for low
            if tracker.recent_win_rate > 0.6 and tracker.sharpe_ratio > 0.5:
                tracker.weight = min(self.max_weight, tracker.weight * 1.1)
            elif tracker.recent_win_rate < 0.4:
                tracker.weight = max(self.min_weight, tracker.weight * 0.9)

            self._logger.debug("Weight adjusted: %s -> %.2f (win_rate=%.2f sharpe=%.2f)",
                               name, tracker.weight, tracker.recent_win_rate, tracker.sharpe_ratio)

    # ------------------------------------------------------------------
    # Correlation monitoring
    # ------------------------------------------------------------------

    def _check_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate pairwise correlation between strategy signals.
        High correlation means strategies are redundant.
        """
        correlations: Dict[str, Dict[str, float]] = {}
        names = list(self._signal_history.keys())

        for i in range(len(names)):
            correlations[names[i]] = {}
            for j in range(i + 1, len(names)):
                hist_a = list(self._signal_history[names[i]])
                hist_b = list(self._signal_history[names[j]])

                min_len = min(len(hist_a), len(hist_b))
                if min_len < 10:
                    continue

                a = np.array(hist_a[-min_len:])
                b = np.array(hist_b[-min_len:])

                if np.std(a) == 0 or np.std(b) == 0:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(a, b)[0, 1])

                correlations[names[i]][names[j]] = corr

                # Update trackers
                if names[i] in self._strategy_trackers:
                    self._strategy_trackers[names[i]].correlation_with_others[names[j]] = corr
                if names[j] in self._strategy_trackers:
                    self._strategy_trackers[names[j]].correlation_with_others[names[i]] = corr

                # Warn on high correlation
                if abs(corr) > self.max_correlation:
                    self._logger.warning(
                        "High correlation between %s and %s: %.2f",
                        names[i], names[j], corr,
                    )

        return correlations

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        if len(candles) < 50:
            return None

        if not self._sub_strategies:
            self._logger.warning("No sub-strategies configured")
            return None

        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])

        # Collect votes
        votes = self._collect_votes(candles)

        # Resolve into single signal
        signal = self._resolve_votes(votes)

        # Periodic maintenance
        self._signals_since_adjustment += 1
        if self._signals_since_adjustment >= self.weight_adjustment_interval:
            self._adjust_weights()
            self._check_correlations()
            self._signals_since_adjustment = 0

        if signal is None:
            return None

        # Apply composite risk check
        if signal.is_entry and not self.check_risk(signal):
            return None

        # Open/close position
        if signal.is_entry:
            side = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"
            self.open_position(side, current_price, signal.quantity,
                               signal.stop_loss, signal.take_profit)
        elif signal.is_exit and self.has_position:
            pnl = self.close_position(current_price, "composite_exit")
            # Update per-strategy trackers
            voting_strats = signal.metadata.get("voting_strategies", [])
            for name in voting_strats:
                if name in self._strategy_trackers:
                    per_strat_pnl = pnl / max(len(voting_strats), 1)
                    self._strategy_trackers[name].record_signal(per_strat_pnl)

        self._last_composite_signal = signal
        return signal

    def on_tick(self, price: float, volume: float, timestamp: float) -> Optional[StrategySignal]:
        if self.has_position:
            self.update_position_price(price)
        # Forward tick to sub-strategies
        for strat in self._sub_strategies:
            strat.on_tick(price, volume, timestamp)
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        if self.has_position:
            close = ohlcv.get("close", 0.0)
            if close > 0:
                self.update_position_price(close)
        for strat in self._sub_strategies:
            strat.on_bar(ohlcv)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        for strat in self._sub_strategies:
            strat.on_fill(order_info)

    def get_composite_status(self) -> Dict[str, Any]:
        return {
            "sub_strategies": len(self._sub_strategies),
            "strategy_details": {
                name: {
                    "weight": tracker.weight,
                    "initial_weight": tracker.initial_weight,
                    "total_signals": tracker.total_signals,
                    "win_rate": tracker.win_rate,
                    "recent_win_rate": tracker.recent_win_rate,
                    "total_pnl": tracker.total_pnl,
                    "sharpe": tracker.sharpe_ratio,
                    "correlations": dict(tracker.correlation_with_others),
                }
                for name, tracker in self._strategy_trackers.items()
            },
            "voting_threshold": self.voting_threshold,
            "min_strategies": self.min_strategies,
            "dynamic_weights": self.enable_dynamic_weights,
        }
