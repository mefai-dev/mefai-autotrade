"""
Mefai Signal Engine - Multi-Timeframe Momentum Strategy

RSI momentum scoring across 3 timeframes, MACD histogram momentum,
Rate of Change analysis, volume-confirmed breakouts, ADX trend strength
filter, EMA crossover confirmation, momentum divergence detection,
and ATR-based dynamic trailing stop.
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
class MomentumScore:
    """Composite momentum score from multiple sources."""
    rsi_score: float = 0.0
    macd_score: float = 0.0
    roc_score: float = 0.0
    volume_score: float = 0.0
    ema_score: float = 0.0
    adx_value: float = 0.0
    divergence_detected: bool = False
    divergence_type: str = ""   # "bullish" or "bearish"
    total_score: float = 0.0
    direction: int = 0          # 1 = bullish, -1 = bearish, 0 = neutral

    def calculate_total(self, weights: Dict[str, float]) -> None:
        raw = (
            self.rsi_score * weights.get("rsi", 0.25)
            + self.macd_score * weights.get("macd", 0.25)
            + self.roc_score * weights.get("roc", 0.15)
            + self.volume_score * weights.get("volume", 0.15)
            + self.ema_score * weights.get("ema", 0.20)
        )
        # Divergence bonus
        if self.divergence_detected:
            if self.divergence_type == "bullish":
                raw += 0.2
            elif self.divergence_type == "bearish":
                raw -= 0.2
        self.total_score = max(-1.0, min(1.0, raw))
        if self.total_score > 0.3:
            self.direction = 1
        elif self.total_score < -0.3:
            self.direction = -1
        else:
            self.direction = 0


class MomentumStrategy(BaseStrategy):
    """
    Multi-Timeframe Momentum Strategy

    Combines multiple momentum indicators into a unified score.
    Only trades when ADX confirms a trending environment (> 25).
    Uses divergence detection for early reversal signals and
    ATR-based trailing stops for position management.
    """

    name = "Momentum"
    description = "Multi-timeframe momentum with RSI, MACD, ADX confirmation"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # RSI
        self.rsi_period: int = self.params.get("rsi_period", 14)
        self.rsi_overbought: float = self.params.get("rsi_overbought", 70.0)
        self.rsi_oversold: float = self.params.get("rsi_oversold", 30.0)

        # MACD
        self.macd_fast: int = self.params.get("macd_fast", 12)
        self.macd_slow: int = self.params.get("macd_slow", 26)
        self.macd_signal: int = self.params.get("macd_signal", 9)

        # ROC
        self.roc_period: int = self.params.get("roc_period", 12)
        self.roc_threshold: float = self.params.get("roc_threshold", 2.0)

        # ADX filter
        self.adx_threshold: float = self.params.get("adx_threshold", 25.0)
        self.adx_period: int = self.params.get("adx_period", 14)

        # EMA crossover
        self.ema_fast: int = self.params.get("ema_fast", 9)
        self.ema_slow: int = self.params.get("ema_slow", 21)

        # Volume
        self.volume_ma_period: int = self.params.get("volume_ma_period", 20)
        self.volume_threshold: float = self.params.get("volume_threshold", 1.5)

        # Divergence
        self.divergence_lookback: int = self.params.get("divergence_lookback", 14)

        # Stop loss / trailing
        self.atr_stop_multiplier: float = self.params.get("atr_stop_multiplier", 2.0)
        self.atr_period: int = self.params.get("atr_period", 14)

        # Signal threshold
        self.entry_threshold: float = self.params.get("entry_threshold", 0.5)
        self.exit_threshold: float = self.params.get("exit_threshold", -0.2)

        # Score weights
        self.weights: Dict[str, float] = self.params.get("weights", {
            "rsi": 0.25,
            "macd": 0.25,
            "roc": 0.15,
            "volume": 0.15,
            "ema": 0.20,
        })

        self._prev_macd_hist: float = 0.0
        self._prev_rsi: float = 50.0
        self._price_highs: List[float] = []
        self._rsi_highs: List[float] = []
        self._price_lows: List[float] = []
        self._rsi_lows: List[float] = []

    # ------------------------------------------------------------------
    # Indicator scoring
    # ------------------------------------------------------------------

    def _score_rsi(self, rsi: float) -> float:
        """
        Score RSI momentum.
        Positive = bullish momentum, negative = bearish.
        """
        if rsi > 80:
            return -0.8   # extreme overbought
        elif rsi > self.rsi_overbought:
            return -0.4
        elif rsi > 55:
            return 0.3    # moderate bullish
        elif rsi > 50:
            return 0.1
        elif rsi > 45:
            return -0.1
        elif rsi > self.rsi_oversold:
            return -0.3
        elif rsi > 20:
            return 0.4    # oversold = potential reversal
        else:
            return 0.8    # extreme oversold

    def _score_macd(
        self, macd_line: float, signal_line: float, histogram: float,
    ) -> float:
        """Score MACD momentum based on histogram direction and crossover."""
        score = 0.0

        # Histogram direction
        if histogram > 0 and histogram > self._prev_macd_hist:
            score += 0.5  # increasing positive momentum
        elif histogram > 0:
            score += 0.2  # positive but decreasing
        elif histogram < 0 and histogram < self._prev_macd_hist:
            score -= 0.5  # increasing negative momentum
        elif histogram < 0:
            score -= 0.2

        # Crossover
        if macd_line > signal_line and self._prev_macd_hist <= 0 and histogram > 0:
            score += 0.4  # bullish crossover
        elif macd_line < signal_line and self._prev_macd_hist >= 0 and histogram < 0:
            score -= 0.4  # bearish crossover

        self._prev_macd_hist = histogram
        return max(-1.0, min(1.0, score))

    def _score_roc(self, roc_value: float) -> float:
        """Score Rate of Change."""
        if roc_value > self.roc_threshold * 2:
            return 0.8
        elif roc_value > self.roc_threshold:
            return 0.5
        elif roc_value > 0:
            return 0.2
        elif roc_value > -self.roc_threshold:
            return -0.2
        elif roc_value > -self.roc_threshold * 2:
            return -0.5
        else:
            return -0.8

    def _score_volume(
        self, current_volume: float, avg_volume: float,
    ) -> float:
        """Score volume confirmation."""
        if avg_volume == 0:
            return 0.0
        ratio = current_volume / avg_volume
        if ratio > self.volume_threshold * 2:
            return 0.8   # very high volume
        elif ratio > self.volume_threshold:
            return 0.5   # above average
        elif ratio > 1.0:
            return 0.2
        elif ratio > 0.5:
            return -0.1
        else:
            return -0.3  # very low volume

    def _score_ema(
        self, close: float, ema_fast_val: float, ema_slow_val: float,
    ) -> float:
        """Score EMA crossover."""
        if np.isnan(ema_fast_val) or np.isnan(ema_slow_val):
            return 0.0

        # Price above both EMAs = bullish
        if close > ema_fast_val > ema_slow_val:
            return 0.8
        elif ema_fast_val > ema_slow_val:
            return 0.4
        elif close < ema_fast_val < ema_slow_val:
            return -0.8
        elif ema_fast_val < ema_slow_val:
            return -0.4
        return 0.0

    # ------------------------------------------------------------------
    # Divergence detection
    # ------------------------------------------------------------------

    def _detect_divergence(
        self, close: np.ndarray, rsi: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Detect RSI divergence vs price.
        Bullish: price makes lower low, RSI makes higher low
        Bearish: price makes higher high, RSI makes lower high
        """
        lookback = self.divergence_lookback
        if len(close) < lookback + 5 or len(rsi) < lookback + 5:
            return False, ""

        recent_close = close[-lookback:]
        recent_rsi = rsi[-lookback:]

        # Filter out NaN
        valid_mask = ~np.isnan(recent_rsi)
        if np.sum(valid_mask) < lookback // 2:
            return False, ""

        # Find local lows and highs using simple method
        half = lookback // 2

        price_first_half_low = float(np.min(recent_close[:half]))
        price_second_half_low = float(np.min(recent_close[half:]))
        rsi_first_half = recent_rsi[:half]
        rsi_second_half = recent_rsi[half:]
        rsi_valid_first = rsi_first_half[~np.isnan(rsi_first_half)]
        rsi_valid_second = rsi_second_half[~np.isnan(rsi_second_half)]

        if len(rsi_valid_first) == 0 or len(rsi_valid_second) == 0:
            return False, ""

        rsi_first_half_low = float(np.min(rsi_valid_first))
        rsi_second_half_low = float(np.min(rsi_valid_second))

        # Bullish divergence: price lower low, RSI higher low
        if price_second_half_low < price_first_half_low and rsi_second_half_low > rsi_first_half_low:
            return True, "bullish"

        price_first_half_high = float(np.max(recent_close[:half]))
        price_second_half_high = float(np.max(recent_close[half:]))
        rsi_first_half_high = float(np.max(rsi_valid_first))
        rsi_second_half_high = float(np.max(rsi_valid_second))

        # Bearish divergence: price higher high, RSI lower high
        if price_second_half_high > price_first_half_high and rsi_second_half_high < rsi_first_half_high:
            return True, "bearish"

        return False, ""

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def _calculate_momentum_score(
        self, candles: np.ndarray,
    ) -> MomentumScore:
        """Calculate the full composite momentum score."""
        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)

        score = MomentumScore()

        # RSI
        rsi_vals = self.indicators.rsi(close, self.rsi_period)
        current_rsi = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else 50.0
        score.rsi_score = self._score_rsi(current_rsi)

        # MACD
        macd_line, signal_line, histogram = self.indicators.macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        last_idx = len(close) - 1
        ml = float(macd_line[last_idx]) if not np.isnan(macd_line[last_idx]) else 0.0
        sl = float(signal_line[last_idx]) if not np.isnan(signal_line[last_idx]) else 0.0
        hist = float(histogram[last_idx]) if not np.isnan(histogram[last_idx]) else 0.0
        score.macd_score = self._score_macd(ml, sl, hist)

        # ROC
        roc_vals = self.indicators.roc(close, self.roc_period)
        roc_val = float(roc_vals[-1]) if not np.isnan(roc_vals[-1]) else 0.0
        score.roc_score = self._score_roc(roc_val)

        # Volume
        vol_ma = self.indicators.sma(volume, self.volume_ma_period)
        avg_vol = float(vol_ma[-1]) if not np.isnan(vol_ma[-1]) else 1.0
        score.volume_score = self._score_volume(float(volume[-1]), avg_vol)

        # EMA
        ema_fast = self.indicators.ema(close, self.ema_fast)
        ema_slow = self.indicators.ema(close, self.ema_slow)
        ef = float(ema_fast[-1]) if not np.isnan(ema_fast[-1]) else close[-1]
        es = float(ema_slow[-1]) if not np.isnan(ema_slow[-1]) else close[-1]
        score.ema_score = self._score_ema(float(close[-1]), ef, es)

        # ADX
        adx_vals = self.indicators.adx(high, low, close, self.adx_period)
        score.adx_value = float(adx_vals[-1]) if not np.isnan(adx_vals[-1]) else 0.0

        # Divergence
        div_detected, div_type = self._detect_divergence(close, rsi_vals)
        score.divergence_detected = div_detected
        score.divergence_type = div_type

        # Calculate total
        score.calculate_total(self.weights)

        self._prev_rsi = current_rsi
        return score

    # ------------------------------------------------------------------
    # Stop loss calculation
    # ------------------------------------------------------------------

    def _calculate_stop_loss(
        self, candles: np.ndarray, side: str,
    ) -> float:
        """ATR-based stop loss."""
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else close[-1] * 0.02
        current = float(close[-1])

        if side == "LONG":
            return current - self.atr_stop_multiplier * atr_val
        else:
            return current + self.atr_stop_multiplier * atr_val

    def _calculate_take_profit(
        self, entry: float, stop_loss: float, side: str, rr_ratio: float = 2.0,
    ) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(entry - stop_loss)
        if side == "LONG":
            return entry + risk * rr_ratio
        else:
            return entry - risk * rr_ratio

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(self.macd_slow + self.macd_signal, self.adx_period * 2 + 1,
                      self.divergence_lookback + 5, 60)
        if len(candles) < min_len:
            return None

        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)

        # Calculate momentum score
        score = self._calculate_momentum_score(candles)

        # ADX filter - only trade in trending markets
        if score.adx_value < self.adx_threshold:
            # If we have a position, check trailing stop
            if self.has_position:
                atr = self.indicators.atr(high, low, close, self.atr_period)
                atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02
                self.update_position_price(current_price)
                if self.check_trailing_stop(current_price, atr_val, self.atr_stop_multiplier):
                    pnl = self.close_position(current_price, "trailing_stop")
                    signal_type = (SignalType.CLOSE_LONG if self.position is None
                                   else SignalType.CLOSE_SHORT)
                    return StrategySignal(
                        signal_type=signal_type,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.7,
                        reason=f"Trailing stop hit in non-trending market (ADX={score.adx_value:.1f})",
                    )
            return None

        # Entry signals
        if not self.has_position:
            if score.total_score >= self.entry_threshold and score.direction == 1:
                sl = self._calculate_stop_loss(candles, "LONG")
                tp = self._calculate_take_profit(current_price, sl, "LONG")
                qty = self.calculate_position_size(
                    self.params.get("account_balance", 10000.0),
                    current_price, sl, risk_pct=1.0,
                )
                signal = StrategySignal(
                    signal_type=SignalType.LONG,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=qty,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=min(0.95, 0.5 + abs(score.total_score) * 0.5),
                    reason=f"Momentum LONG: score={score.total_score:.2f} ADX={score.adx_value:.1f}",
                    metadata={
                        "rsi_score": score.rsi_score,
                        "macd_score": score.macd_score,
                        "roc_score": score.roc_score,
                        "volume_score": score.volume_score,
                        "ema_score": score.ema_score,
                        "adx": score.adx_value,
                        "divergence": score.divergence_type,
                    },
                )
                if self.check_risk(signal):
                    self.open_position("LONG", current_price, qty, sl, tp)
                    return signal

            elif score.total_score <= -self.entry_threshold and score.direction == -1:
                sl = self._calculate_stop_loss(candles, "SHORT")
                tp = self._calculate_take_profit(current_price, sl, "SHORT")
                qty = self.calculate_position_size(
                    self.params.get("account_balance", 10000.0),
                    current_price, sl, risk_pct=1.0,
                )
                signal = StrategySignal(
                    signal_type=SignalType.SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=qty,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence=min(0.95, 0.5 + abs(score.total_score) * 0.5),
                    reason=f"Momentum SHORT: score={score.total_score:.2f} ADX={score.adx_value:.1f}",
                    metadata={
                        "rsi_score": score.rsi_score,
                        "macd_score": score.macd_score,
                        "roc_score": score.roc_score,
                        "volume_score": score.volume_score,
                        "ema_score": score.ema_score,
                        "adx": score.adx_value,
                        "divergence": score.divergence_type,
                    },
                )
                if self.check_risk(signal):
                    self.open_position("SHORT", current_price, qty, sl, tp)
                    return signal

        # Exit signals - check position
        elif self.has_position:
            self.update_position_price(current_price)
            atr = self.indicators.atr(high, low, close, self.atr_period)
            atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

            # Trailing stop
            if self.check_trailing_stop(current_price, atr_val, self.atr_stop_multiplier):
                side = self.position.side
                self.close_position(current_price, "trailing_stop")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG if side == "LONG" else SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.8,
                    reason=f"Trailing stop hit (ATR x {self.atr_stop_multiplier})",
                )

            # Momentum reversal exit
            if self.position.side == "LONG" and score.total_score <= self.exit_threshold:
                self.close_position(current_price, "momentum_reversal")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.7,
                    reason=f"Momentum reversal: score={score.total_score:.2f}",
                )
            elif self.position.side == "SHORT" and score.total_score >= -self.exit_threshold:
                self.close_position(current_price, "momentum_reversal")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.7,
                    reason=f"Momentum reversal: score={score.total_score:.2f}",
                )

            # Divergence-based early exit
            if score.divergence_detected:
                if self.position.side == "LONG" and score.divergence_type == "bearish":
                    self.close_position(current_price, "bearish_divergence")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.75,
                        reason="Bearish divergence detected - exiting long",
                    )
                elif self.position.side == "SHORT" and score.divergence_type == "bullish":
                    self.close_position(current_price, "bullish_divergence")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.75,
                        reason="Bullish divergence detected - exiting short",
                    )

        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        if self.has_position:
            self.update_position_price(price)
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        close = ohlcv.get("close", 0.0)
        if self.has_position and close > 0:
            self.update_position_price(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        price = order_info.get("price", 0.0)
        qty = order_info.get("quantity", 0.0)
        side = order_info.get("side", "")
        if price > 0 and qty > 0:
            self._logger.info("Fill received: %s %.6f @ %.4f", side, qty, price)
