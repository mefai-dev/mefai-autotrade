"""
Mefai Signal Engine - Breakout Detection Strategy

Horizontal support/resistance breakout, Donchian Channel (Turtle Trading),
volume-confirmed breakouts, false breakout detection, breakout pullback
entry, range detection, and ATR-based stop/target placement.
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
class SupportResistance:
    """A support or resistance level."""
    price: float
    strength: int = 1        # number of touches
    level_type: str = ""     # "support" or "resistance"
    last_touch_idx: int = 0
    broken: bool = False


@dataclass
class BreakoutEvent:
    """Record of a detected breakout."""
    price: float
    level: float
    direction: str            # "up" or "down"
    volume_confirmed: bool
    timestamp: float
    retested: bool = False
    was_false: bool = False


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Detection Strategy

    Identifies breakout opportunities using multiple methods:
    1. Horizontal S/R level breakout with strength scoring
    2. Donchian Channel breakout (Turtle Trading rules)
    3. Volume confirmation (must exceed 2x average volume)
    4. False breakout filter (wick rejection detection)
    5. Pullback entry (wait for breakout level retest)
    6. Range detection via Bollinger Band width
    7. ATR-based stop loss and take profit placement
    """

    name = "Breakout"
    description = "Breakout detection with volume confirmation and false breakout filter"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # S/R detection
        self.sr_lookback: int = self.params.get("sr_lookback", 50)
        self.sr_tolerance_pct: float = self.params.get("sr_tolerance_pct", 0.3)
        self.min_touches: int = self.params.get("min_touches", 2)

        # Donchian
        self.donchian_period: int = self.params.get("donchian_period", 20)
        self.donchian_exit_period: int = self.params.get("donchian_exit_period", 10)
        self.use_donchian: bool = self.params.get("use_donchian", True)

        # Volume
        self.volume_ma_period: int = self.params.get("volume_ma_period", 20)
        self.volume_multiplier: float = self.params.get("volume_multiplier", 2.0)
        self.require_volume: bool = self.params.get("require_volume", True)

        # False breakout filter
        self.wick_rejection_pct: float = self.params.get("wick_rejection_pct", 60.0)
        self.confirmation_bars: int = self.params.get("confirmation_bars", 2)

        # Pullback entry
        self.wait_for_pullback: bool = self.params.get("wait_for_pullback", False)
        self.pullback_tolerance_pct: float = self.params.get("pullback_tolerance_pct", 0.5)

        # Range detection
        self.bb_width_threshold: float = self.params.get("bb_width_threshold", 0.03)
        self.bb_period: int = self.params.get("bb_period", 20)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_multiplier: float = self.params.get("atr_stop_multiplier", 2.0)
        self.atr_target_multiplier: float = self.params.get("atr_target_multiplier", 3.0)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        self._sr_levels: List[SupportResistance] = []
        self._recent_breakouts: List[BreakoutEvent] = []
        self._in_range: bool = False

    # ------------------------------------------------------------------
    # Support / Resistance detection
    # ------------------------------------------------------------------

    def _find_sr_levels(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ) -> List[SupportResistance]:
        """
        Find support and resistance levels from price pivots.
        Uses swing high/low detection with touch counting.
        """
        n = len(close)
        lookback = min(self.sr_lookback, n - 2)
        if lookback < 5:
            return []

        levels: List[SupportResistance] = []
        tolerance = float(close[-1]) * (self.sr_tolerance_pct / 100.0)

        # Find swing highs and lows (local extremes with 2-bar lookback)
        for i in range(2, lookback):
            idx = n - lookback + i
            if idx < 2 or idx >= n - 2:
                continue

            # Swing high
            if high[idx] > high[idx - 1] and high[idx] > high[idx - 2] and \
               high[idx] > high[idx + 1] and high[idx] >= high[idx + 2]:
                self._add_or_merge_level(levels, float(high[idx]), "resistance", idx, tolerance)

            # Swing low
            if low[idx] < low[idx - 1] and low[idx] < low[idx - 2] and \
               low[idx] < low[idx + 1] and low[idx] <= low[idx + 2]:
                self._add_or_merge_level(levels, float(low[idx]), "support", idx, tolerance)

        # Filter by minimum touches
        levels = [l for l in levels if l.strength >= self.min_touches]

        # Sort by strength (most touches first)
        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:20]  # keep top 20

    def _add_or_merge_level(
        self,
        levels: List[SupportResistance],
        price: float,
        level_type: str,
        idx: int,
        tolerance: float,
    ) -> None:
        """Add a new level or merge with an existing close level."""
        for existing in levels:
            if abs(existing.price - price) <= tolerance:
                existing.strength += 1
                existing.last_touch_idx = max(existing.last_touch_idx, idx)
                return
        levels.append(SupportResistance(
            price=price,
            strength=1,
            level_type=level_type,
            last_touch_idx=idx,
        ))

    # ------------------------------------------------------------------
    # Breakout detection
    # ------------------------------------------------------------------

    def _detect_horizontal_breakout(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Optional[BreakoutEvent]:
        """
        Check if current price has broken above resistance or below support.
        """
        current_price = float(close[-1])
        prev_price = float(close[-2]) if len(close) > 1 else current_price

        vol_ma = self.indicators.sma(volume, self.volume_ma_period)
        avg_vol = float(vol_ma[-1]) if not np.isnan(vol_ma[-1]) else 1.0
        current_vol = float(volume[-1])
        volume_confirmed = current_vol >= avg_vol * self.volume_multiplier

        for level in self._sr_levels:
            if level.broken:
                continue

            # Breakout above resistance
            if level.level_type == "resistance":
                if prev_price <= level.price and current_price > level.price:
                    if self.require_volume and not volume_confirmed:
                        continue
                    level.broken = True
                    event = BreakoutEvent(
                        price=current_price,
                        level=level.price,
                        direction="up",
                        volume_confirmed=volume_confirmed,
                        timestamp=time.time(),
                    )
                    self._recent_breakouts.append(event)
                    return event

            # Breakout below support
            elif level.level_type == "support":
                if prev_price >= level.price and current_price < level.price:
                    if self.require_volume and not volume_confirmed:
                        continue
                    level.broken = True
                    event = BreakoutEvent(
                        price=current_price,
                        level=level.price,
                        direction="down",
                        volume_confirmed=volume_confirmed,
                        timestamp=time.time(),
                    )
                    self._recent_breakouts.append(event)
                    return event

        return None

    def _detect_donchian_breakout(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
        volume: np.ndarray,
    ) -> Optional[BreakoutEvent]:
        """Donchian Channel breakout (Turtle Trading system)."""
        if not self.use_donchian:
            return None

        dc_upper, dc_mid, dc_lower = self.indicators.donchian(
            high, low, self.donchian_period,
        )
        last = len(close) - 1
        if last < 1:
            return None
        if np.isnan(dc_upper[last]) or np.isnan(dc_lower[last]):
            return None

        current_price = float(close[last])
        prev_close = float(close[last - 1])
        prev_upper = float(dc_upper[last - 1]) if not np.isnan(dc_upper[last - 1]) else 0
        prev_lower = float(dc_lower[last - 1]) if not np.isnan(dc_lower[last - 1]) else 0

        vol_ma = self.indicators.sma(volume, self.volume_ma_period)
        avg_vol = float(vol_ma[last]) if not np.isnan(vol_ma[last]) else 1.0
        vol_confirmed = float(volume[last]) >= avg_vol * self.volume_multiplier

        # Upper breakout
        if current_price > float(dc_upper[last]) and prev_close <= prev_upper:
            return BreakoutEvent(
                price=current_price,
                level=float(dc_upper[last]),
                direction="up",
                volume_confirmed=vol_confirmed,
                timestamp=time.time(),
            )

        # Lower breakout
        if current_price < float(dc_lower[last]) and prev_close >= prev_lower:
            return BreakoutEvent(
                price=current_price,
                level=float(dc_lower[last]),
                direction="down",
                volume_confirmed=vol_confirmed,
                timestamp=time.time(),
            )

        return None

    # ------------------------------------------------------------------
    # False breakout filter
    # ------------------------------------------------------------------

    def _is_false_breakout(
        self, candles: np.ndarray, breakout: BreakoutEvent,
    ) -> bool:
        """
        Detect false breakouts via wick rejection.
        If the wick is > 60% of the candle range and price closed back
        inside the level, it is likely a false breakout.
        """
        last = candles[-1]
        open_p = float(last[1])
        high_p = float(last[2])
        low_p = float(last[3])
        close_p = float(last[4])
        body = abs(close_p - open_p)
        full_range = high_p - low_p

        if full_range == 0:
            return False

        if breakout.direction == "up":
            upper_wick = high_p - max(open_p, close_p)
            wick_pct = (upper_wick / full_range) * 100.0
            if wick_pct > self.wick_rejection_pct and close_p < breakout.level:
                breakout.was_false = True
                return True

        elif breakout.direction == "down":
            lower_wick = min(open_p, close_p) - low_p
            wick_pct = (lower_wick / full_range) * 100.0
            if wick_pct > self.wick_rejection_pct and close_p > breakout.level:
                breakout.was_false = True
                return True

        return False

    # ------------------------------------------------------------------
    # Range / squeeze detection
    # ------------------------------------------------------------------

    def _detect_range(self, close: np.ndarray) -> bool:
        """
        Detect if market is in a range using Bollinger Band width.
        Narrow BB width = ranging market = potential breakout setup.
        """
        bb_upper, bb_mid, bb_lower = self.indicators.bollinger_bands(
            close, self.bb_period, 2.0,
        )
        last = len(close) - 1
        if np.isnan(bb_upper[last]) or np.isnan(bb_lower[last]) or np.isnan(bb_mid[last]):
            return False

        bb_width = (float(bb_upper[last]) - float(bb_lower[last])) / float(bb_mid[last])
        return bb_width < self.bb_width_threshold

    # ------------------------------------------------------------------
    # Pullback check
    # ------------------------------------------------------------------

    def _check_pullback_entry(
        self, current_price: float,
    ) -> Optional[BreakoutEvent]:
        """
        If wait_for_pullback is enabled, check if price has pulled back
        to the breakout level for a retest entry.
        """
        tolerance = current_price * (self.pullback_tolerance_pct / 100.0)

        for event in self._recent_breakouts:
            if event.was_false or event.retested:
                continue

            if event.direction == "up":
                # Price pulled back to the breakout level (now support)
                if abs(current_price - event.level) <= tolerance and current_price >= event.level:
                    event.retested = True
                    return event

            elif event.direction == "down":
                if abs(current_price - event.level) <= tolerance and current_price <= event.level:
                    event.retested = True
                    return event

        return None

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(self.sr_lookback + 5, self.donchian_period + 5,
                      self.volume_ma_period + 5, self.bb_period + 5)
        if len(candles) < min_len:
            return None

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)
        current_price = float(close[-1])

        # Update S/R levels
        self._sr_levels = self._find_sr_levels(high, low, close)

        # Check range state
        self._in_range = self._detect_range(close)

        # ATR for stops
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Check exits first
        if self.has_position:
            self.update_position_price(current_price)

            # Donchian exit (shorter period)
            dc_upper_exit, _, dc_lower_exit = self.indicators.donchian(
                high, low, self.donchian_exit_period,
            )
            last = len(close) - 1

            if self.position.side == "LONG":
                if not np.isnan(dc_lower_exit[last]) and current_price < float(dc_lower_exit[last]):
                    self.close_position(current_price, "donchian_exit")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.8,
                        reason=f"Donchian exit: price below {self.donchian_exit_period}-bar low",
                    )

            elif self.position.side == "SHORT":
                if not np.isnan(dc_upper_exit[last]) and current_price > float(dc_upper_exit[last]):
                    self.close_position(current_price, "donchian_exit")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.8,
                        reason=f"Donchian exit: price above {self.donchian_exit_period}-bar high",
                    )

            # Trailing stop
            if self.check_trailing_stop(current_price, atr_val, self.atr_stop_multiplier):
                side = self.position.side
                self.close_position(current_price, "trailing_stop")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG if side == "LONG" else SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.8,
                    reason="ATR trailing stop hit",
                )

            return None

        # Entry - check for breakouts
        breakout = None

        # Check pullback entry first
        if self.wait_for_pullback:
            breakout_event = self._check_pullback_entry(current_price)
            if breakout_event is not None:
                breakout = breakout_event

        # Then check new breakouts
        if breakout is None:
            breakout = self._detect_horizontal_breakout(close, high, low, volume)

        if breakout is None:
            breakout = self._detect_donchian_breakout(high, low, close, volume)

        if breakout is None:
            return None

        # False breakout filter
        if self._is_false_breakout(candles, breakout):
            self._logger.info("False breakout detected at %.4f", breakout.level)
            return None

        # If waiting for pullback and this is a new breakout (not a retest), skip
        if self.wait_for_pullback and not breakout.retested:
            self._logger.debug("Waiting for pullback retest of %.4f", breakout.level)
            return None

        # Generate entry signal
        if breakout.direction == "up":
            sl = current_price - self.atr_stop_multiplier * atr_val
            tp = current_price + self.atr_target_multiplier * atr_val
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, self.risk_pct,
            )
            signal = StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=0.65 + (0.15 if breakout.volume_confirmed else 0.0)
                + (0.1 if self._in_range else 0.0),
                reason=f"Breakout LONG above {breakout.level:.4f}"
                + (" (volume confirmed)" if breakout.volume_confirmed else "")
                + (" (from squeeze)" if self._in_range else ""),
                metadata={
                    "breakout_level": breakout.level,
                    "volume_confirmed": breakout.volume_confirmed,
                    "from_range": self._in_range,
                    "sr_levels": len(self._sr_levels),
                    "atr": atr_val,
                    "retested": breakout.retested,
                },
            )
            if self.check_risk(signal):
                self.open_position("LONG", current_price, qty, sl, tp)
                return signal

        elif breakout.direction == "down":
            sl = current_price + self.atr_stop_multiplier * atr_val
            tp = current_price - self.atr_target_multiplier * atr_val
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, self.risk_pct,
            )
            signal = StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=0.65 + (0.15 if breakout.volume_confirmed else 0.0)
                + (0.1 if self._in_range else 0.0),
                reason=f"Breakout SHORT below {breakout.level:.4f}"
                + (" (volume confirmed)" if breakout.volume_confirmed else "")
                + (" (from squeeze)" if self._in_range else ""),
                metadata={
                    "breakout_level": breakout.level,
                    "volume_confirmed": breakout.volume_confirmed,
                    "from_range": self._in_range,
                    "sr_levels": len(self._sr_levels),
                    "atr": atr_val,
                    "retested": breakout.retested,
                },
            )
            if self.check_risk(signal):
                self.open_position("SHORT", current_price, qty, sl, tp)
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

    def get_levels(self) -> List[Dict[str, Any]]:
        """Return current S/R levels."""
        return [
            {
                "price": l.price,
                "type": l.level_type,
                "strength": l.strength,
                "broken": l.broken,
            }
            for l in self._sr_levels
        ]
