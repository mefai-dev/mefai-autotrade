"""
Mefai Signal Engine - Statistical Mean Reversion Strategy

Bollinger Band mean reversion, z-score based entry/exit, RSI confirmation,
Keltner Channel squeeze detection, higher-timeframe trend filter, and
statistical edge calculation per z-score level.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import numpy as np

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class ZScoreStats:
    """Track trade results at each z-score level for edge calculation."""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trades if self.trades > 0 else 0.0


class MeanReversionStrategy(BaseStrategy):
    """
    Statistical Mean Reversion Strategy

    Trades reversions to the mean using multiple confirmation layers:
    1. Bollinger Band extremes (enter at 2 SD, exit at mean)
    2. Z-score based entry/exit thresholds
    3. RSI overbought/oversold confirmation
    4. Keltner Channel squeeze as range filter
    5. Higher timeframe trend filter (only trade reversions WITH the trend)
    6. Statistical edge tracking per z-score level
    """

    name = "MeanReversion"
    description = "Statistical mean reversion with Bollinger Bands and z-score"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Bollinger Bands
        self.bb_period: int = self.params.get("bb_period", 20)
        self.bb_std: float = self.params.get("bb_std", 2.0)

        # Z-score
        self.zscore_lookback: int = self.params.get("zscore_lookback", 20)
        self.zscore_entry: float = self.params.get("zscore_entry", 2.0)
        self.zscore_exit: float = self.params.get("zscore_exit", 0.5)
        self.zscore_stop: float = self.params.get("zscore_stop", 3.5)

        # RSI confirmation
        self.rsi_period: int = self.params.get("rsi_period", 14)
        self.rsi_ob: float = self.params.get("rsi_overbought", 70.0)
        self.rsi_os: float = self.params.get("rsi_oversold", 30.0)
        self.require_rsi_confirmation: bool = self.params.get("require_rsi_confirmation", True)

        # Keltner squeeze
        self.keltner_ema: int = self.params.get("keltner_ema", 20)
        self.keltner_atr: int = self.params.get("keltner_atr", 10)
        self.keltner_mult: float = self.params.get("keltner_mult", 1.5)
        self.require_squeeze: bool = self.params.get("require_squeeze", False)

        # Higher TF trend filter
        self.trend_ema_period: int = self.params.get("trend_ema_period", 50)
        self.use_trend_filter: bool = self.params.get("use_trend_filter", True)

        # ATR stop
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_multiplier", 2.5)

        # Position sizing
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # Stats tracking per z-score bucket
        self._zscore_stats: Dict[str, ZScoreStats] = {}
        for bucket in ["1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0+"]:
            self._zscore_stats[bucket] = ZScoreStats()

        self._entry_zscore: float = 0.0

    # ------------------------------------------------------------------
    # Z-score calculation
    # ------------------------------------------------------------------

    def _calculate_zscore(self, close: np.ndarray) -> float:
        """Calculate current z-score of price relative to rolling mean."""
        if len(close) < self.zscore_lookback:
            return 0.0
        window = close[-self.zscore_lookback:]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        if std == 0:
            return 0.0
        return (float(close[-1]) - mean) / std

    def _zscore_to_bucket(self, z: float) -> str:
        """Map absolute z-score to a stats bucket."""
        az = abs(z)
        if az >= 3.0:
            return "3.0+"
        elif az >= 2.5:
            return "2.5-3.0"
        elif az >= 2.0:
            return "2.0-2.5"
        else:
            return "1.5-2.0"

    # ------------------------------------------------------------------
    # Squeeze detection
    # ------------------------------------------------------------------

    def _detect_squeeze(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ) -> bool:
        """
        Keltner Channel squeeze: BB inside Keltner = low volatility range.
        When the squeeze releases, mean reversion trades have higher win rate.
        """
        bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(
            close, self.bb_period, self.bb_std,
        )
        kc_upper, kc_mid, kc_lower = self.indicators.keltner_channel(
            high, low, close, self.keltner_ema, self.keltner_atr, self.keltner_mult,
        )

        last = len(close) - 1
        if (np.isnan(bb_upper[last]) or np.isnan(kc_upper[last])):
            return False

        # Squeeze = BB is inside KC
        bb_inside = (bb_lower[last] > kc_lower[last]) and (bb_upper[last] < kc_upper[last])
        return bb_inside

    # ------------------------------------------------------------------
    # Trend filter
    # ------------------------------------------------------------------

    def _get_trend_direction(self, close: np.ndarray) -> int:
        """
        Higher timeframe trend direction.
        Returns 1 (uptrend), -1 (downtrend), 0 (neutral).
        """
        if len(close) < self.trend_ema_period + 5:
            return 0
        ema = self.indicators.ema(close, self.trend_ema_period)
        last = len(close) - 1
        if np.isnan(ema[last]):
            return 0

        # Slope of EMA over last 5 bars
        if last >= 5 and not np.isnan(ema[last - 5]):
            slope = (ema[last] - ema[last - 5]) / 5.0
            threshold = float(close[last]) * 0.0001
            if slope > threshold:
                return 1
            elif slope < -threshold:
                return -1
        return 0

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Optional[StrategySignal]:
        """Evaluate whether conditions are met for a mean reversion entry."""
        current_price = float(close[-1])
        zscore = self._calculate_zscore(close)
        abs_zscore = abs(zscore)

        # Must exceed entry threshold
        if abs_zscore < self.zscore_entry:
            return None

        # RSI confirmation
        rsi_vals = self.indicators.rsi(close, self.rsi_period)
        rsi = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else 50.0

        if self.require_rsi_confirmation:
            if zscore > 0 and rsi < self.rsi_ob:
                return None   # z-score says overbought but RSI does not confirm
            if zscore < 0 and rsi > self.rsi_os:
                return None   # z-score says oversold but RSI does not confirm

        # Squeeze filter
        if self.require_squeeze:
            if not self._detect_squeeze(high, low, close):
                return None

        # Trend filter - only take reversions in direction of higher TF trend
        if self.use_trend_filter:
            trend = self._get_trend_direction(close)
            if zscore > self.zscore_entry and trend == 1:
                return None   # overbought in uptrend - don't short
            if zscore < -self.zscore_entry and trend == -1:
                return None   # oversold in downtrend - don't buy

        # Bollinger Band confirmation
        bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(
            close, self.bb_period, self.bb_std,
        )
        last = len(close) - 1
        bb_u = float(bb_upper[last]) if not np.isnan(bb_upper[last]) else current_price
        bb_l = float(bb_lower[last]) if not np.isnan(bb_lower[last]) else current_price
        bb_m = float(bb_mid[last]) if not np.isnan(bb_mid[last]) else current_price

        # ATR for stop
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Generate signal
        if zscore < -self.zscore_entry:
            # Oversold - go long (reversion to mean)
            sl = current_price - self.atr_stop_mult * atr_val
            tp = bb_m  # target the mean
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, self.risk_pct,
            )

            self._entry_zscore = zscore
            confidence = min(0.95, 0.5 + abs_zscore * 0.15)

            signal = StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                reason=f"Mean reversion LONG: z={zscore:.2f} RSI={rsi:.1f}",
                metadata={
                    "zscore": zscore,
                    "rsi": rsi,
                    "bb_lower": bb_l,
                    "bb_middle": bb_m,
                    "bb_upper": bb_u,
                    "atr": atr_val,
                    "bucket": self._zscore_to_bucket(zscore),
                    "bucket_win_rate": self._zscore_stats[self._zscore_to_bucket(zscore)].win_rate,
                },
            )
            return signal

        elif zscore > self.zscore_entry:
            # Overbought - go short (reversion to mean)
            sl = current_price + self.atr_stop_mult * atr_val
            tp = bb_m
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, self.risk_pct,
            )

            self._entry_zscore = zscore
            confidence = min(0.95, 0.5 + abs_zscore * 0.15)

            signal = StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                reason=f"Mean reversion SHORT: z={zscore:.2f} RSI={rsi:.1f}",
                metadata={
                    "zscore": zscore,
                    "rsi": rsi,
                    "bb_lower": bb_l,
                    "bb_middle": bb_m,
                    "bb_upper": bb_u,
                    "atr": atr_val,
                    "bucket": self._zscore_to_bucket(zscore),
                    "bucket_win_rate": self._zscore_stats[self._zscore_to_bucket(zscore)].win_rate,
                },
            )
            return signal

        return None

    def _evaluate_exit(self, close: np.ndarray) -> Optional[StrategySignal]:
        """Check if the mean reversion trade should be closed."""
        if not self.has_position:
            return None

        current_price = float(close[-1])
        zscore = self._calculate_zscore(close)
        self.update_position_price(current_price)

        should_exit = False
        reason = ""

        # Z-score reverted past exit threshold
        if self.position.side == "LONG" and zscore >= -self.zscore_exit:
            should_exit = True
            reason = f"Mean reverted: z={zscore:.2f} (exit threshold={self.zscore_exit})"
        elif self.position.side == "SHORT" and zscore <= self.zscore_exit:
            should_exit = True
            reason = f"Mean reverted: z={zscore:.2f} (exit threshold={self.zscore_exit})"

        # Stop loss - z-score moved further against us
        if self.position.side == "LONG" and zscore < -self.zscore_stop:
            should_exit = True
            reason = f"Z-score stop hit: z={zscore:.2f} (stop={self.zscore_stop})"
        elif self.position.side == "SHORT" and zscore > self.zscore_stop:
            should_exit = True
            reason = f"Z-score stop hit: z={zscore:.2f} (stop={self.zscore_stop})"

        # Price-based stop loss
        if self.position.stop_loss > 0:
            if self.position.side == "LONG" and current_price <= self.position.stop_loss:
                should_exit = True
                reason = f"Stop loss hit at {current_price:.4f}"
            elif self.position.side == "SHORT" and current_price >= self.position.stop_loss:
                should_exit = True
                reason = f"Stop loss hit at {current_price:.4f}"

        # Take profit
        if self.position.take_profit > 0:
            if self.position.side == "LONG" and current_price >= self.position.take_profit:
                should_exit = True
                reason = f"Take profit hit at {current_price:.4f}"
            elif self.position.side == "SHORT" and current_price <= self.position.take_profit:
                should_exit = True
                reason = f"Take profit hit at {current_price:.4f}"

        if should_exit:
            side = self.position.side
            pnl = self.close_position(current_price, reason)

            # Update z-score stats
            bucket = self._zscore_to_bucket(self._entry_zscore)
            stats = self._zscore_stats[bucket]
            stats.trades += 1
            stats.total_pnl += pnl
            if pnl > 0:
                stats.wins += 1

            signal_type = SignalType.CLOSE_LONG if side == "LONG" else SignalType.CLOSE_SHORT
            return StrategySignal(
                signal_type=signal_type,
                symbol=self.symbol,
                price=current_price,
                confidence=0.8,
                reason=reason,
                metadata={
                    "zscore": zscore,
                    "pnl": pnl,
                    "entry_zscore": self._entry_zscore,
                },
            )

        return None

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(self.bb_period + 5, self.zscore_lookback + 5,
                      self.trend_ema_period + 10, self.rsi_period + 5)
        if len(candles) < min_len:
            return None

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)

        # Check exit first
        exit_signal = self._evaluate_exit(close)
        if exit_signal is not None:
            return exit_signal

        # Then check entry
        if not self.has_position:
            entry_signal = self._evaluate_entry(close, high, low, volume)
            if entry_signal is not None and self.check_risk(entry_signal):
                self.open_position(
                    "LONG" if entry_signal.signal_type == SignalType.LONG else "SHORT",
                    entry_signal.price,
                    entry_signal.quantity,
                    entry_signal.stop_loss,
                    entry_signal.take_profit,
                )
                return entry_signal

        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        if self.has_position:
            self.update_position_price(price)
            # Check price-based stops
            if self.position.stop_loss > 0:
                if self.position.side == "LONG" and price <= self.position.stop_loss:
                    side = self.position.side
                    self.close_position(price, "stop_loss_tick")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol,
                        price=price,
                        confidence=0.9,
                        reason="Stop loss hit on tick",
                    )
                elif self.position.side == "SHORT" and price >= self.position.stop_loss:
                    self.close_position(price, "stop_loss_tick")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol,
                        price=price,
                        confidence=0.9,
                        reason="Stop loss hit on tick",
                    )
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        close = ohlcv.get("close", 0.0)
        if self.has_position and close > 0:
            self.update_position_price(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        price = order_info.get("price", 0.0)
        if price > 0:
            self._logger.info("Fill: %s", order_info)

    def get_edge_stats(self) -> Dict[str, Any]:
        """Return statistical edge per z-score bucket."""
        return {
            bucket: {
                "trades": stats.trades,
                "win_rate": stats.win_rate,
                "avg_pnl": stats.avg_pnl,
                "total_pnl": stats.total_pnl,
            }
            for bucket, stats in self._zscore_stats.items()
        }
