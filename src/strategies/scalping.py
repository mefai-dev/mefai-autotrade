"""
Mefai Signal Engine - High-Frequency Scalping Strategy

Order book imbalance detection, bid-ask spread analysis, micro-trend
detection on 1m/5m EMA, volume spike detection, quick entry/exit targeting
0.1-0.3%, maximum holding time, commission-aware filtering, and rate limiting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import math
import numpy as np
from collections import deque

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Simplified order book state for imbalance calculation."""
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_volume: float = 0.0      # total volume on bid side (top N levels)
    ask_volume: float = 0.0      # total volume on ask side (top N levels)
    spread: float = 0.0
    spread_pct: float = 0.0
    imbalance: float = 0.0       # -1.0 to 1.0 (positive = more bids)
    timestamp: float = 0.0

    def calculate(self) -> None:
        if self.best_ask > 0 and self.best_bid > 0:
            self.spread = self.best_ask - self.best_bid
            mid = (self.best_ask + self.best_bid) / 2.0
            self.spread_pct = (self.spread / mid) * 100.0 if mid > 0 else 0.0
        total_vol = self.bid_volume + self.ask_volume
        if total_vol > 0:
            self.imbalance = (self.bid_volume - self.ask_volume) / total_vol


@dataclass
class ScalpTrade:
    """Record of a scalp trade."""
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    entry_time: float
    exit_time: float
    pnl: float
    pnl_pct: float
    hold_seconds: float
    commission: float
    net_pnl: float


class ScalpingStrategy(BaseStrategy):
    """
    High-Frequency Scalping Strategy

    Targets small profits (0.1-0.3%) with tight risk management:
    - Order book imbalance for direction bias
    - Bid-ask spread monitoring (only trade when spread is tight)
    - Micro-trend detection on 1m/5m EMA alignment
    - Volume spike detection for momentum entries
    - Maximum holding time enforcement
    - Commission-aware filtering (expected profit > 2x commission)
    - Rate limiting (max trades per hour)
    """

    name = "Scalping"
    description = "High-frequency scalping with order book and micro-trend analysis"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Targets
        self.target_pct: float = self.params.get("target_pct", 0.2)
        self.stop_pct: float = self.params.get("stop_pct", 0.15)
        self.max_target_pct: float = self.params.get("max_target_pct", 0.5)

        # Holding time
        self.max_hold_seconds: float = self.params.get("max_hold_minutes", 15) * 60.0
        self.min_hold_seconds: float = self.params.get("min_hold_seconds", 5.0)

        # Commission
        self.maker_fee_pct: float = self.params.get("maker_fee_pct", 0.02)
        self.taker_fee_pct: float = self.params.get("taker_fee_pct", 0.04)
        self.min_profit_to_commission: float = self.params.get("min_profit_to_commission", 2.0)

        # Rate limiting
        self.max_trades_per_hour: int = self.params.get("max_trades_per_hour", 10)
        self._trade_timestamps: deque = deque(maxlen=100)

        # Order book
        self.imbalance_threshold: float = self.params.get("imbalance_threshold", 0.3)
        self.max_spread_pct: float = self.params.get("max_spread_pct", 0.05)

        # Micro-trend EMAs
        self.ema_ultra_fast: int = self.params.get("ema_ultra_fast", 3)
        self.ema_fast: int = self.params.get("ema_fast", 8)
        self.ema_slow: int = self.params.get("ema_slow", 21)

        # Volume spike
        self.volume_spike_mult: float = self.params.get("volume_spike_mult", 3.0)
        self.volume_ma_period: int = self.params.get("volume_ma_period", 20)

        # Position sizing
        self.position_size_pct: float = self.params.get("position_size_pct", 5.0)

        # State
        self._last_orderbook = OrderBookSnapshot()
        self._orderbook_history: deque = deque(maxlen=60)
        self._scalp_history: List[ScalpTrade] = []
        self._entry_time: float = 0.0

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _trades_in_last_hour(self) -> int:
        """Count trades executed in the last hour."""
        now = time.time()
        cutoff = now - 3600.0
        count = 0
        for ts in self._trade_timestamps:
            if ts > cutoff:
                count += 1
        return count

    def _can_trade(self) -> bool:
        """Check if rate limit allows another trade."""
        return self._trades_in_last_hour() < self.max_trades_per_hour

    # ------------------------------------------------------------------
    # Commission calculation
    # ------------------------------------------------------------------

    def _estimate_commission(self, price: float, quantity: float) -> float:
        """Estimate round-trip commission (entry + exit)."""
        notional = price * quantity
        entry_fee = notional * (self.taker_fee_pct / 100.0)
        exit_fee = notional * (self.maker_fee_pct / 100.0)
        return entry_fee + exit_fee

    def _is_profitable_after_commission(
        self, price: float, quantity: float,
    ) -> bool:
        """Check if expected profit exceeds minimum commission multiple."""
        expected_profit = price * quantity * (self.target_pct / 100.0)
        commission = self._estimate_commission(price, quantity)
        return expected_profit >= commission * self.min_profit_to_commission

    # ------------------------------------------------------------------
    # Order book analysis
    # ------------------------------------------------------------------

    def update_orderbook(
        self,
        best_bid: float,
        best_ask: float,
        bid_volume: float,
        ask_volume: float,
    ) -> None:
        """Update the order book snapshot. Called externally by the data feed."""
        ob = OrderBookSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            timestamp=time.time(),
        )
        ob.calculate()
        self._last_orderbook = ob
        self._orderbook_history.append(ob)

    def _get_imbalance_direction(self) -> int:
        """
        Get direction bias from order book imbalance.
        Returns 1 (bullish), -1 (bearish), 0 (neutral).
        """
        ob = self._last_orderbook
        if ob.timestamp == 0:
            return 0

        if ob.imbalance > self.imbalance_threshold:
            return 1   # more buying pressure
        elif ob.imbalance < -self.imbalance_threshold:
            return -1  # more selling pressure
        return 0

    def _spread_acceptable(self) -> bool:
        """Check if spread is tight enough for scalping."""
        if self._last_orderbook.timestamp == 0:
            return True  # no orderbook data, allow based on candles only
        return self._last_orderbook.spread_pct <= self.max_spread_pct

    # ------------------------------------------------------------------
    # Micro-trend detection
    # ------------------------------------------------------------------

    def _detect_micro_trend(self, close: np.ndarray) -> int:
        """
        Detect micro-trend from EMA alignment.
        Returns 1 (up), -1 (down), 0 (neutral).
        """
        if len(close) < self.ema_slow + 5:
            return 0

        ema_uf = self.indicators.ema(close, self.ema_ultra_fast)
        ema_f = self.indicators.ema(close, self.ema_fast)
        ema_s = self.indicators.ema(close, self.ema_slow)

        last = len(close) - 1
        uf = float(ema_uf[last]) if not np.isnan(ema_uf[last]) else 0
        f = float(ema_f[last]) if not np.isnan(ema_f[last]) else 0
        s = float(ema_s[last]) if not np.isnan(ema_s[last]) else 0

        if uf == 0 or f == 0 or s == 0:
            return 0

        # Perfect alignment = strong micro-trend
        if uf > f > s:
            return 1
        elif uf < f < s:
            return -1

        # Partial alignment
        if uf > f and f > s * 0.999:
            return 1
        elif uf < f and f < s * 1.001:
            return -1

        return 0

    # ------------------------------------------------------------------
    # Volume spike detection
    # ------------------------------------------------------------------

    def _detect_volume_spike(self, volume: np.ndarray) -> bool:
        """Detect if current volume is a spike (> 3x average)."""
        if len(volume) < self.volume_ma_period + 2:
            return False
        vol_ma = self.indicators.sma(volume, self.volume_ma_period)
        avg = float(vol_ma[-2]) if not np.isnan(vol_ma[-2]) else 1.0
        current = float(volume[-1])
        return current >= avg * self.volume_spike_mult

    # ------------------------------------------------------------------
    # Entry / Exit logic
    # ------------------------------------------------------------------

    def _evaluate_scalp_entry(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        """Evaluate all scalping entry conditions."""
        close = candles[:, 4].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)
        current_price = float(close[-1])

        # Rate limit check
        if not self._can_trade():
            return None

        # Spread check
        if not self._spread_acceptable():
            return None

        # Micro-trend
        micro_trend = self._detect_micro_trend(close)
        if micro_trend == 0:
            return None

        # Order book imbalance
        ob_direction = self._get_imbalance_direction()

        # Volume spike gives extra confidence
        vol_spike = self._detect_volume_spike(volume)

        # Alignment check: micro-trend + (orderbook or volume spike)
        confirmed = False
        if ob_direction == micro_trend:
            confirmed = True
        elif vol_spike and micro_trend != 0:
            confirmed = True

        if not confirmed:
            return None

        # Position sizing
        account_balance = self.params.get("account_balance", 10000.0)
        position_value = account_balance * (self.position_size_pct / 100.0)
        quantity = position_value / current_price

        # Commission check
        if not self._is_profitable_after_commission(current_price, quantity):
            return None

        # Calculate stops
        if micro_trend == 1:
            sl = current_price * (1.0 - self.stop_pct / 100.0)
            tp = current_price * (1.0 + self.target_pct / 100.0)
            signal_type = SignalType.LONG
            side = "LONG"
        else:
            sl = current_price * (1.0 + self.stop_pct / 100.0)
            tp = current_price * (1.0 - self.target_pct / 100.0)
            signal_type = SignalType.SHORT
            side = "SHORT"

        commission = self._estimate_commission(current_price, quantity)
        expected_profit = abs(tp - current_price) * quantity - commission

        signal = StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=current_price,
            quantity=quantity,
            stop_loss=sl,
            take_profit=tp,
            confidence=0.55 + (0.15 if vol_spike else 0.0)
            + (0.1 if ob_direction == micro_trend else 0.0),
            reason=f"Scalp {side}: micro-trend={micro_trend}"
            + (f" ob_imbalance={self._last_orderbook.imbalance:.2f}" if ob_direction != 0 else "")
            + (" vol_spike" if vol_spike else ""),
            metadata={
                "micro_trend": micro_trend,
                "ob_imbalance": self._last_orderbook.imbalance,
                "volume_spike": vol_spike,
                "spread_pct": self._last_orderbook.spread_pct,
                "expected_profit": expected_profit,
                "commission": commission,
                "max_hold_seconds": self.max_hold_seconds,
            },
        )
        return signal

    def _check_scalp_exit(
        self, current_price: float,
    ) -> Optional[StrategySignal]:
        """Check if the scalp position should be closed."""
        if not self.has_position:
            return None

        self.update_position_price(current_price)
        hold_time = time.time() - self._entry_time

        should_exit = False
        reason = ""

        # Maximum hold time
        if hold_time >= self.max_hold_seconds:
            should_exit = True
            reason = f"Max hold time reached ({hold_time:.0f}s)"

        # Stop loss
        if self.position.side == "LONG" and current_price <= self.position.stop_loss:
            should_exit = True
            reason = f"Stop loss hit at {current_price:.4f}"
        elif self.position.side == "SHORT" and current_price >= self.position.stop_loss:
            should_exit = True
            reason = f"Stop loss hit at {current_price:.4f}"

        # Take profit
        if self.position.side == "LONG" and current_price >= self.position.take_profit:
            should_exit = True
            reason = f"Take profit hit at {current_price:.4f}"
        elif self.position.side == "SHORT" and current_price <= self.position.take_profit:
            should_exit = True
            reason = f"Take profit hit at {current_price:.4f}"

        # Extended take profit - if we passed the target, use a tighter trail
        if self.position.side == "LONG":
            pnl_pct = ((current_price - self.position.entry_price) / self.position.entry_price) * 100
            if pnl_pct > self.target_pct:
                # Trail at 50% of the gain
                trail = self.position.entry_price + (current_price - self.position.entry_price) * 0.5
                if current_price < trail:
                    should_exit = True
                    reason = f"Trail stop after target: {pnl_pct:.2f}% profit"
        elif self.position.side == "SHORT":
            pnl_pct = ((self.position.entry_price - current_price) / self.position.entry_price) * 100
            if pnl_pct > self.target_pct:
                trail = self.position.entry_price - (self.position.entry_price - current_price) * 0.5
                if current_price > trail:
                    should_exit = True
                    reason = f"Trail stop after target: {pnl_pct:.2f}% profit"

        # Order book reversal during trade
        ob_dir = self._get_imbalance_direction()
        if self.position.side == "LONG" and ob_dir == -1 and hold_time > self.min_hold_seconds:
            if self.position.unrealized_pnl > 0:
                should_exit = True
                reason = "Order book reversal - taking profit"
        elif self.position.side == "SHORT" and ob_dir == 1 and hold_time > self.min_hold_seconds:
            if self.position.unrealized_pnl > 0:
                should_exit = True
                reason = "Order book reversal - taking profit"

        if should_exit:
            side = self.position.side
            entry_price = self.position.entry_price
            qty = self.position.quantity
            pnl = self.close_position(current_price, reason)
            commission = self._estimate_commission(entry_price, qty)

            scalp = ScalpTrade(
                entry_price=entry_price,
                exit_price=current_price,
                quantity=qty,
                side=side,
                entry_time=self._entry_time,
                exit_time=time.time(),
                pnl=pnl,
                pnl_pct=((current_price - entry_price) / entry_price * 100)
                if side == "LONG"
                else ((entry_price - current_price) / entry_price * 100),
                hold_seconds=hold_time,
                commission=commission,
                net_pnl=pnl - commission,
            )
            self._scalp_history.append(scalp)

            signal_type = SignalType.CLOSE_LONG if side == "LONG" else SignalType.CLOSE_SHORT
            return StrategySignal(
                signal_type=signal_type,
                symbol=self.symbol,
                price=current_price,
                confidence=0.8,
                reason=reason,
                metadata={
                    "hold_time": hold_time,
                    "pnl": pnl,
                    "commission": commission,
                    "net_pnl": pnl - commission,
                },
            )

        return None

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        if len(candles) < self.ema_slow + 5:
            return None

        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])

        # Check exit first
        exit_signal = self._check_scalp_exit(current_price)
        if exit_signal is not None:
            return exit_signal

        # Entry
        if not self.has_position:
            signal = self._evaluate_scalp_entry(candles)
            if signal is not None and self.check_risk(signal):
                self.open_position(
                    "LONG" if signal.signal_type == SignalType.LONG else "SHORT",
                    current_price, signal.quantity,
                    signal.stop_loss, signal.take_profit,
                )
                self._entry_time = time.time()
                self._trade_timestamps.append(time.time())
                return signal

        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        """Scalping checks on every tick for fast exit."""
        return self._check_scalp_exit(price)

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        close = ohlcv.get("close", 0.0)
        if close > 0:
            return self._check_scalp_exit(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        self._logger.info("Scalp fill: %s", order_info)

    def get_scalp_stats(self) -> Dict[str, Any]:
        """Return scalping performance stats."""
        if not self._scalp_history:
            return {"total_scalps": 0}

        net_pnls = [s.net_pnl for s in self._scalp_history]
        hold_times = [s.hold_seconds for s in self._scalp_history]
        wins = [s for s in self._scalp_history if s.net_pnl > 0]

        return {
            "total_scalps": len(self._scalp_history),
            "winning_scalps": len(wins),
            "win_rate": len(wins) / len(self._scalp_history),
            "total_net_pnl": sum(net_pnls),
            "avg_net_pnl": float(np.mean(net_pnls)),
            "avg_hold_seconds": float(np.mean(hold_times)),
            "max_hold_seconds": max(hold_times),
            "total_commission": sum(s.commission for s in self._scalp_history),
            "trades_last_hour": self._trades_in_last_hour(),
        }
