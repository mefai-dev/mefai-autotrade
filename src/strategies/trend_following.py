"""
Mefai Signal Engine - Classic Trend Following Strategy

Supertrend indicator, EMA ribbon (8/13/21/34/55), Parabolic SAR,
Ichimoku Cloud analysis, ADX filter, pyramid entries on pullbacks,
and Chandelier Exit / ATR-based trailing stop.
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
class IchimokuValues:
    """Ichimoku Cloud computed values."""
    tenkan: float = 0.0       # Conversion line (9)
    kijun: float = 0.0        # Base line (26)
    senkou_a: float = 0.0     # Leading Span A
    senkou_b: float = 0.0     # Leading Span B
    chikou: float = 0.0       # Lagging Span
    cloud_top: float = 0.0
    cloud_bottom: float = 0.0
    price_vs_cloud: int = 0   # 1 = above, -1 = below, 0 = inside
    tk_cross: int = 0         # 1 = bullish cross, -1 = bearish cross, 0 = none


@dataclass
class PyramidLevel:
    """A pyramid (add-on) position level."""
    entry_price: float
    quantity: float
    entry_time: float
    level: int


class TrendFollowingStrategy(BaseStrategy):
    """
    Classic Trend Following Strategy

    Uses multiple trend indicators for confluence:
    1. Supertrend for primary direction
    2. EMA ribbon (8/13/21/34/55) for trend strength visualization
    3. Parabolic SAR for trend direction confirmation
    4. Ichimoku Cloud for multi-dimensional trend analysis
    5. ADX filter (only trade when ADX > 20)
    6. Pyramid entries on pullbacks within the trend
    7. Chandelier Exit for trailing stop
    """

    name = "TrendFollowing"
    description = "Classic trend following with Supertrend, EMA ribbon, Ichimoku"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Supertrend
        self.st_period: int = self.params.get("supertrend_period", 10)
        self.st_mult: float = self.params.get("supertrend_mult", 3.0)

        # EMA ribbon
        self.ema_periods: List[int] = self.params.get(
            "ema_periods", [8, 13, 21, 34, 55]
        )

        # Parabolic SAR
        self.sar_af_start: float = self.params.get("sar_af_start", 0.02)
        self.sar_af_step: float = self.params.get("sar_af_step", 0.02)
        self.sar_af_max: float = self.params.get("sar_af_max", 0.2)

        # Ichimoku
        self.ichimoku_tenkan: int = self.params.get("ichimoku_tenkan", 9)
        self.ichimoku_kijun: int = self.params.get("ichimoku_kijun", 26)
        self.ichimoku_senkou_b: int = self.params.get("ichimoku_senkou_b", 52)

        # ADX
        self.adx_threshold: float = self.params.get("adx_threshold", 20.0)
        self.adx_period: int = self.params.get("adx_period", 14)
        self.strong_trend_adx: float = self.params.get("strong_trend_adx", 40.0)

        # Pyramid
        self.enable_pyramid: bool = self.params.get("enable_pyramid", True)
        self.max_pyramid_levels: int = self.params.get("max_pyramid_levels", 3)
        self.pyramid_pullback_pct: float = self.params.get("pyramid_pullback_pct", 1.5)
        self.pyramid_size_reduction: float = self.params.get("pyramid_size_reduction", 0.5)

        # Trailing stop
        self.chandelier_period: int = self.params.get("chandelier_period", 22)
        self.chandelier_mult: float = self.params.get("chandelier_mult", 3.0)
        self.atr_period: int = self.params.get("atr_period", 14)

        # Risk
        self.risk_pct: float = self.params.get("risk_pct", 1.0)
        self.min_confluence: int = self.params.get("min_confluence", 3)

        # State
        self._pyramid_levels: List[PyramidLevel] = []
        self._last_supertrend_dir: int = 0
        self._last_sar_dir: int = 0
        self._prev_tenkan: float = 0.0
        self._prev_kijun: float = 0.0

    # ------------------------------------------------------------------
    # Ichimoku calculation
    # ------------------------------------------------------------------

    def _calculate_ichimoku(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ) -> IchimokuValues:
        """Calculate Ichimoku Cloud components."""
        n = len(close)
        result = IchimokuValues()

        if n < self.ichimoku_senkou_b + 26:
            return result

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        t_high = np.max(high[-self.ichimoku_tenkan:])
        t_low = np.min(low[-self.ichimoku_tenkan:])
        result.tenkan = (t_high + t_low) / 2.0

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        k_high = np.max(high[-self.ichimoku_kijun:])
        k_low = np.min(low[-self.ichimoku_kijun:])
        result.kijun = (k_high + k_low) / 2.0

        # Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
        result.senkou_a = (result.tenkan + result.kijun) / 2.0

        # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
        sb_high = np.max(high[-self.ichimoku_senkou_b:])
        sb_low = np.min(low[-self.ichimoku_senkou_b:])
        result.senkou_b = (sb_high + sb_low) / 2.0

        # Chikou Span: current close, plotted 26 periods back
        result.chikou = float(close[-1])

        # Cloud boundaries
        result.cloud_top = max(result.senkou_a, result.senkou_b)
        result.cloud_bottom = min(result.senkou_a, result.senkou_b)

        # Price vs cloud
        current = float(close[-1])
        if current > result.cloud_top:
            result.price_vs_cloud = 1
        elif current < result.cloud_bottom:
            result.price_vs_cloud = -1
        else:
            result.price_vs_cloud = 0

        # TK cross
        if self._prev_tenkan > 0 and self._prev_kijun > 0:
            if result.tenkan > result.kijun and self._prev_tenkan <= self._prev_kijun:
                result.tk_cross = 1   # bullish
            elif result.tenkan < result.kijun and self._prev_tenkan >= self._prev_kijun:
                result.tk_cross = -1  # bearish

        self._prev_tenkan = result.tenkan
        self._prev_kijun = result.kijun
        return result

    # ------------------------------------------------------------------
    # EMA ribbon analysis
    # ------------------------------------------------------------------

    def _analyze_ema_ribbon(self, close: np.ndarray) -> Tuple[int, float]:
        """
        Analyze EMA ribbon alignment.
        Returns (direction, strength).
        direction: 1 (bullish), -1 (bearish), 0 (mixed)
        strength: 0.0 - 1.0
        """
        emas = []
        for period in self.ema_periods:
            ema = self.indicators.ema(close, period)
            val = float(ema[-1]) if not np.isnan(ema[-1]) else 0.0
            emas.append(val)

        if any(v == 0 for v in emas):
            return 0, 0.0

        # Check perfect alignment
        bullish_aligned = all(emas[i] >= emas[i + 1] for i in range(len(emas) - 1))
        bearish_aligned = all(emas[i] <= emas[i + 1] for i in range(len(emas) - 1))

        if bullish_aligned:
            # Strength = how spread out the ribbon is
            spread = (emas[0] - emas[-1]) / emas[-1] * 100 if emas[-1] != 0 else 0
            strength = min(1.0, spread / 5.0)
            return 1, strength
        elif bearish_aligned:
            spread = (emas[-1] - emas[0]) / emas[0] * 100 if emas[0] != 0 else 0
            strength = min(1.0, spread / 5.0)
            return -1, strength

        # Partial alignment - count how many are in order
        bull_count = sum(1 for i in range(len(emas) - 1) if emas[i] > emas[i + 1])
        bear_count = sum(1 for i in range(len(emas) - 1) if emas[i] < emas[i + 1])
        total = len(emas) - 1

        if bull_count > bear_count:
            return 1, bull_count / total * 0.7
        elif bear_count > bull_count:
            return -1, bear_count / total * 0.7
        return 0, 0.0

    # ------------------------------------------------------------------
    # Chandelier Exit
    # ------------------------------------------------------------------

    def _chandelier_exit(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Chandelier Exit: trailing stop based on ATR from highest high / lowest low.
        Returns (long_stop, short_stop).
        """
        period = self.chandelier_period
        if len(close) < period + 1:
            return 0.0, float("inf")

        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else float(close[-1]) * 0.02

        highest = float(np.max(high[-period:]))
        lowest = float(np.min(low[-period:]))

        long_stop = highest - self.chandelier_mult * atr_val
        short_stop = lowest + self.chandelier_mult * atr_val

        return long_stop, short_stop

    # ------------------------------------------------------------------
    # Confluence scoring
    # ------------------------------------------------------------------

    def _calculate_confluence(
        self, candles: np.ndarray,
    ) -> Tuple[int, int, Dict[str, int]]:
        """
        Calculate the number of indicators agreeing on direction.
        Returns (direction, confluence_count, indicator_details).
        """
        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)

        votes: Dict[str, int] = {}

        # 1. Supertrend
        st_line, st_dir = self.indicators.supertrend(
            high, low, close, self.st_period, self.st_mult,
        )
        st_direction = int(st_dir[-1])
        votes["supertrend"] = st_direction
        self._last_supertrend_dir = st_direction

        # 2. EMA ribbon
        ribbon_dir, ribbon_strength = self._analyze_ema_ribbon(close)
        votes["ema_ribbon"] = ribbon_dir

        # 3. Parabolic SAR
        sar_vals, sar_dir = self.indicators.parabolic_sar(
            high, low, self.sar_af_start, self.sar_af_step, self.sar_af_max,
        )
        sar_direction = int(sar_dir[-1])
        votes["parabolic_sar"] = sar_direction
        self._last_sar_dir = sar_direction

        # 4. Ichimoku
        ichimoku = self._calculate_ichimoku(high, low, close)
        ich_vote = 0
        if ichimoku.price_vs_cloud == 1 and ichimoku.tenkan > ichimoku.kijun:
            ich_vote = 1
        elif ichimoku.price_vs_cloud == -1 and ichimoku.tenkan < ichimoku.kijun:
            ich_vote = -1
        votes["ichimoku"] = ich_vote

        # 5. ADX direction (using EMA cross as proxy for DI)
        ema_fast = self.indicators.ema(close, self.ema_periods[0])
        ema_slow = self.indicators.ema(close, self.ema_periods[-1])
        ef = float(ema_fast[-1]) if not np.isnan(ema_fast[-1]) else 0
        es = float(ema_slow[-1]) if not np.isnan(ema_slow[-1]) else 0
        if ef > es:
            votes["ema_cross"] = 1
        elif ef < es:
            votes["ema_cross"] = -1
        else:
            votes["ema_cross"] = 0

        # Count confluence
        bullish = sum(1 for v in votes.values() if v == 1)
        bearish = sum(1 for v in votes.values() if v == -1)

        if bullish > bearish:
            return 1, bullish, votes
        elif bearish > bullish:
            return -1, bearish, votes
        return 0, 0, votes

    # ------------------------------------------------------------------
    # Pyramid entries
    # ------------------------------------------------------------------

    def _check_pyramid_entry(
        self, current_price: float, direction: int,
    ) -> bool:
        """Check if conditions are met for a pyramid (add-on) entry."""
        if not self.enable_pyramid or not self.has_position:
            return False
        if len(self._pyramid_levels) >= self.max_pyramid_levels:
            return False
        if self.position.side == "LONG" and direction != 1:
            return False
        if self.position.side == "SHORT" and direction != -1:
            return False

        # Check pullback from last entry
        last_entry = self.position.entry_price
        if self._pyramid_levels:
            last_entry = self._pyramid_levels[-1].entry_price

        if self.position.side == "LONG":
            pullback_pct = ((last_entry - current_price) / last_entry) * 100
            if pullback_pct >= self.pyramid_pullback_pct and current_price > self.position.stop_loss:
                return True
        elif self.position.side == "SHORT":
            pullback_pct = ((current_price - last_entry) / last_entry) * 100
            if pullback_pct >= self.pyramid_pullback_pct and current_price < self.position.stop_loss:
                return True

        return False

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(
            self.ichimoku_senkou_b + 30,
            max(self.ema_periods) + 5,
            self.adx_period * 2 + 5,
            self.st_period + 20,
        )
        if len(candles) < min_len:
            return None

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        current_price = float(close[-1])

        # ADX filter
        adx_vals = self.indicators.adx(high, low, close, self.adx_period)
        adx = float(adx_vals[-1]) if not np.isnan(adx_vals[-1]) else 0.0

        if adx < self.adx_threshold:
            # Not trending - check exit only
            if self.has_position:
                long_stop, short_stop = self._chandelier_exit(high, low, close)
                self.update_position_price(current_price)
                if self.position.side == "LONG" and current_price < long_stop:
                    self.close_position(current_price, "chandelier_exit_no_trend")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.75,
                        reason=f"Chandelier exit (ADX={adx:.1f} below threshold)",
                    )
                elif self.position.side == "SHORT" and current_price > short_stop:
                    self.close_position(current_price, "chandelier_exit_no_trend")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.75,
                        reason=f"Chandelier exit (ADX={adx:.1f} below threshold)",
                    )
            return None

        # Calculate confluence
        direction, confluence, votes = self._calculate_confluence(candles)

        # Chandelier Exit levels
        long_stop, short_stop = self._chandelier_exit(high, low, close)

        # Exit checks
        if self.has_position:
            self.update_position_price(current_price)

            # Chandelier Exit
            if self.position.side == "LONG" and current_price < long_stop:
                pnl = self.close_position(current_price, "chandelier_exit")
                self._pyramid_levels.clear()
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.85,
                    reason=f"Chandelier Exit: stop={long_stop:.4f}",
                    metadata={"pnl": pnl, "chandelier_stop": long_stop},
                )

            if self.position.side == "SHORT" and current_price > short_stop:
                pnl = self.close_position(current_price, "chandelier_exit")
                self._pyramid_levels.clear()
                return StrategySignal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.85,
                    reason=f"Chandelier Exit: stop={short_stop:.4f}",
                    metadata={"pnl": pnl, "chandelier_stop": short_stop},
                )

            # Trend reversal exit
            if self.position.side == "LONG" and direction == -1 and confluence >= self.min_confluence:
                pnl = self.close_position(current_price, "trend_reversal")
                self._pyramid_levels.clear()
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.8,
                    reason=f"Trend reversal: {confluence} bearish indicators",
                    metadata={"votes": votes, "pnl": pnl},
                )

            if self.position.side == "SHORT" and direction == 1 and confluence >= self.min_confluence:
                pnl = self.close_position(current_price, "trend_reversal")
                self._pyramid_levels.clear()
                return StrategySignal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.8,
                    reason=f"Trend reversal: {confluence} bullish indicators",
                    metadata={"votes": votes, "pnl": pnl},
                )

            # Pyramid entry check
            if self._check_pyramid_entry(current_price, direction):
                level_num = len(self._pyramid_levels) + 1
                base_qty = self.position.quantity
                pyramid_qty = base_qty * (self.pyramid_size_reduction ** level_num)

                self._pyramid_levels.append(PyramidLevel(
                    entry_price=current_price,
                    quantity=pyramid_qty,
                    entry_time=time.time(),
                    level=level_num,
                ))

                signal_type = SignalType.LONG if direction == 1 else SignalType.SHORT
                return StrategySignal(
                    signal_type=signal_type,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=pyramid_qty,
                    stop_loss=long_stop if direction == 1 else short_stop,
                    confidence=0.6,
                    reason=f"Pyramid level {level_num}: pullback entry in trend",
                    metadata={"pyramid_level": level_num, "adx": adx},
                )

            return None

        # New entry - need sufficient confluence
        if confluence < self.min_confluence:
            return None

        # ATR for position sizing
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        if direction == 1:
            sl = long_stop
            tp = current_price + (current_price - sl) * 2.0  # 2:1 RR
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, self.risk_pct,
            )
            confidence = min(0.95, 0.5 + confluence * 0.1
                             + (0.1 if adx > self.strong_trend_adx else 0.0))

            signal = StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                reason=f"Trend LONG: {confluence}/5 confluence, ADX={adx:.1f}",
                metadata={
                    "votes": votes,
                    "adx": adx,
                    "confluence": confluence,
                    "chandelier_stop": long_stop,
                },
            )
            if self.check_risk(signal):
                self.open_position("LONG", current_price, qty, sl, tp)
                self._pyramid_levels.clear()
                return signal

        elif direction == -1:
            sl = short_stop
            tp = current_price - (sl - current_price) * 2.0
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, self.risk_pct,
            )
            confidence = min(0.95, 0.5 + confluence * 0.1
                             + (0.1 if adx > self.strong_trend_adx else 0.0))

            signal = StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=confidence,
                reason=f"Trend SHORT: {confluence}/5 confluence, ADX={adx:.1f}",
                metadata={
                    "votes": votes,
                    "adx": adx,
                    "confluence": confluence,
                    "chandelier_stop": short_stop,
                },
            )
            if self.check_risk(signal):
                self.open_position("SHORT", current_price, qty, sl, tp)
                self._pyramid_levels.clear()
                return signal

        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        if self.has_position:
            self.update_position_price(price)
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        if self.has_position:
            close = ohlcv.get("close", 0.0)
            if close > 0:
                self.update_position_price(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        self._logger.info("Fill: %s", order_info)
