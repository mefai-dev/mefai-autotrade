"""
Mefai Signal Engine - Elliott Wave Pattern Recognition Strategy

Wave counting (impulse 5 waves, corrective 3 waves), Fibonacci
retracement for wave targets, wave validation rules, current wave
detection, target projection, and corrective pattern identification.
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
class WavePoint:
    """A point in an Elliott Wave sequence."""
    price: float
    index: int
    wave_number: int     # 0-5 for impulse, 0-3 for corrective
    wave_type: str       # "impulse" or "corrective"


@dataclass
class WavePattern:
    """A complete or in-progress Elliott Wave pattern."""
    wave_type: str                              # "impulse" or "corrective"
    direction: str                              # "bullish" or "bearish"
    points: List[WavePoint] = field(default_factory=list)
    is_valid: bool = True
    current_wave: int = 0
    completion_pct: float = 0.0
    target_price: float = 0.0
    invalidation_price: float = 0.0
    corrective_type: str = ""                   # "zigzag", "flat", "triangle", ""

    @property
    def is_complete(self) -> bool:
        if self.wave_type == "impulse":
            return len(self.points) >= 6   # 0-5
        else:
            return len(self.points) >= 4   # A-B-C

    @property
    def wave_1_length(self) -> float:
        if len(self.points) >= 2:
            return abs(self.points[1].price - self.points[0].price)
        return 0.0


class ElliottWaveStrategy(BaseStrategy):
    """
    Elliott Wave Pattern Recognition Strategy

    Identifies and trades Elliott Wave patterns:
    1. Impulse waves (5-wave pattern in trend direction)
    2. Corrective waves (3-wave pattern against trend)
    3. Fibonacci-based wave targets and retracements
    4. Wave validation rules enforcement
    5. Current wave position detection
    6. Corrective pattern identification (zigzag, flat, triangle)
    """

    name = "ElliottWave"
    description = "Elliott Wave pattern recognition with Fibonacci projections"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Wave detection
        self.min_wave_pct: float = self.params.get("min_wave_pct", 1.0)
        self.swing_lookback: int = self.params.get("swing_lookback", 5)

        # Fibonacci
        self.fib_tolerance: float = self.params.get("fib_tolerance", 0.05)
        self.wave2_max_retrace: float = self.params.get("wave2_max_retrace", 0.99)
        self.wave2_typical_retrace: float = self.params.get("wave2_typical_retrace", 0.618)
        self.wave3_min_extension: float = self.params.get("wave3_min_extension", 1.0)
        self.wave3_typical_extension: float = self.params.get("wave3_typical_extension", 1.618)
        self.wave4_max_retrace: float = self.params.get("wave4_max_retrace", 0.382)
        self.wave5_typical_extension: float = self.params.get("wave5_typical_extension", 1.0)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_mult", 2.0)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # State
        self._current_pattern: Optional[WavePattern] = None
        self._swing_highs: List[Tuple[int, float]] = []
        self._swing_lows: List[Tuple[int, float]] = []
        self._wave_history: List[WavePattern] = []

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    def _find_swings(
        self, high: np.ndarray, low: np.ndarray,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows."""
        n = len(high)
        lb = self.swing_lookback
        highs = []
        lows = []

        for i in range(lb, n - lb):
            is_high = all(high[i] >= high[i - j] and high[i] >= high[i + j]
                          for j in range(1, lb + 1))
            if is_high:
                highs.append((i, float(high[i])))

            is_low = all(low[i] <= low[i - j] and low[i] <= low[i + j]
                         for j in range(1, lb + 1))
            if is_low:
                lows.append((i, float(low[i])))

        return highs, lows

    def _build_swing_sequence(
        self, swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
    ) -> List[Tuple[int, float, str]]:
        """
        Merge swing highs and lows into alternating sequence.
        Returns [(index, price, "high"/"low"), ...]
        """
        all_swings = []
        for idx, price in swing_highs:
            all_swings.append((idx, price, "high"))
        for idx, price in swing_lows:
            all_swings.append((idx, price, "low"))

        all_swings.sort(key=lambda x: x[0])

        # Ensure alternation
        if len(all_swings) < 2:
            return all_swings

        filtered = [all_swings[0]]
        for i in range(1, len(all_swings)):
            if all_swings[i][2] != filtered[-1][2]:
                filtered.append(all_swings[i])
            else:
                # Same type - keep the more extreme one
                if all_swings[i][2] == "high" and all_swings[i][1] > filtered[-1][1]:
                    filtered[-1] = all_swings[i]
                elif all_swings[i][2] == "low" and all_swings[i][1] < filtered[-1][1]:
                    filtered[-1] = all_swings[i]

        return filtered

    # ------------------------------------------------------------------
    # Wave validation rules
    # ------------------------------------------------------------------

    def _validate_impulse(self, points: List[WavePoint]) -> Tuple[bool, str]:
        """
        Validate Elliott Wave impulse rules:
        1. Wave 2 never retraces more than 100% of Wave 1
        2. Wave 3 is never the shortest impulse wave
        3. Wave 4 never enters the price territory of Wave 1
        """
        if len(points) < 2:
            return True, ""

        p = points

        # Rule 1: Wave 2 cannot retrace more than 100% of Wave 1
        if len(points) >= 3:
            wave1 = abs(p[1].price - p[0].price)
            wave2_retrace = abs(p[2].price - p[1].price)
            if wave1 > 0 and wave2_retrace / wave1 > self.wave2_max_retrace:
                return False, "Wave 2 retraces more than 99% of Wave 1"

        # Rule 2: Wave 3 is never the shortest
        if len(points) >= 6:
            wave1 = abs(p[1].price - p[0].price)
            wave3 = abs(p[3].price - p[2].price)
            wave5 = abs(p[5].price - p[4].price)
            if wave3 < wave1 and wave3 < wave5:
                return False, "Wave 3 is the shortest impulse wave"

        # Rule 3: Wave 4 never enters Wave 1 territory
        if len(points) >= 5:
            if p[0].price < p[1].price:  # bullish
                if p[4].price < p[1].price:
                    return False, "Wave 4 overlaps with Wave 1 territory"
            else:  # bearish
                if p[4].price > p[1].price:
                    return False, "Wave 4 overlaps with Wave 1 territory"

        return True, ""

    # ------------------------------------------------------------------
    # Wave counting
    # ------------------------------------------------------------------

    def _count_impulse_waves(
        self, swings: List[Tuple[int, float, str]],
    ) -> Optional[WavePattern]:
        """
        Attempt to count a 5-wave impulse pattern from swing sequence.
        """
        if len(swings) < 6:
            return None

        # Try bullish impulse (starts at low)
        for start in range(len(swings) - 5):
            if swings[start][2] != "low":
                continue

            points = []
            for i in range(6):
                idx = start + i
                if idx >= len(swings):
                    break
                sw = swings[idx]
                points.append(WavePoint(
                    price=sw[1],
                    index=sw[0],
                    wave_number=i,
                    wave_type="impulse",
                ))

            if len(points) < 6:
                continue

            pattern = WavePattern(
                wave_type="impulse",
                direction="bullish",
                points=points,
                current_wave=5,
            )

            # Validate
            is_valid, reason = self._validate_impulse(points)
            pattern.is_valid = is_valid

            if is_valid:
                # Check minimum wave size
                wave1 = abs(points[1].price - points[0].price) / points[0].price * 100
                if wave1 < self.min_wave_pct:
                    continue

                # Calculate targets
                wave1_len = abs(points[1].price - points[0].price)
                pattern.target_price = points[4].price + wave1_len * self.wave5_typical_extension
                pattern.invalidation_price = points[0].price
                pattern.completion_pct = len(points) / 6.0 * 100

                return pattern

        # Try bearish impulse (starts at high)
        for start in range(len(swings) - 5):
            if swings[start][2] != "high":
                continue

            points = []
            for i in range(6):
                idx = start + i
                if idx >= len(swings):
                    break
                sw = swings[idx]
                points.append(WavePoint(
                    price=sw[1],
                    index=sw[0],
                    wave_number=i,
                    wave_type="impulse",
                ))

            if len(points) < 6:
                continue

            pattern = WavePattern(
                wave_type="impulse",
                direction="bearish",
                points=points,
                current_wave=5,
            )

            is_valid, reason = self._validate_impulse(points)
            pattern.is_valid = is_valid

            if is_valid:
                wave1 = abs(points[1].price - points[0].price) / points[0].price * 100
                if wave1 < self.min_wave_pct:
                    continue

                wave1_len = abs(points[1].price - points[0].price)
                pattern.target_price = points[4].price - wave1_len * self.wave5_typical_extension
                pattern.invalidation_price = points[0].price

                return pattern

        return None

    def _count_corrective_waves(
        self, swings: List[Tuple[int, float, str]],
    ) -> Optional[WavePattern]:
        """Count a 3-wave corrective pattern (A-B-C)."""
        if len(swings) < 4:
            return None

        # Check last 4 swings for corrective pattern
        recent = swings[-4:]

        points = []
        for i, sw in enumerate(recent):
            points.append(WavePoint(
                price=sw[1], index=sw[0],
                wave_number=i, wave_type="corrective",
            ))

        # Determine corrective type
        a_len = abs(points[1].price - points[0].price)
        b_len = abs(points[2].price - points[1].price)
        c_len = abs(points[3].price - points[2].price)

        if a_len == 0:
            return None

        b_retrace = b_len / a_len
        c_to_a = c_len / a_len

        corrective_type = ""
        if b_retrace < 0.618 + self.fib_tolerance:
            corrective_type = "zigzag"   # sharp A, shallow B, sharp C
        elif 0.618 - self.fib_tolerance <= b_retrace <= 1.0 + self.fib_tolerance:
            corrective_type = "flat"     # B retraces most of A
        else:
            corrective_type = "expanded_flat"

        direction = "bearish" if points[1].price < points[0].price else "bullish"

        pattern = WavePattern(
            wave_type="corrective",
            direction=direction,
            points=points,
            is_valid=True,
            current_wave=3,
            corrective_type=corrective_type,
        )

        # Target: C typically equals A in length
        if direction == "bearish":
            pattern.target_price = points[2].price - a_len
        else:
            pattern.target_price = points[2].price + a_len

        return pattern

    # ------------------------------------------------------------------
    # Fibonacci projections
    # ------------------------------------------------------------------

    def _project_wave3_target(self, pattern: WavePattern) -> float:
        """Project Wave 3 target using Fibonacci extension of Wave 1."""
        if len(pattern.points) < 3:
            return 0.0
        wave1 = abs(pattern.points[1].price - pattern.points[0].price)
        if pattern.direction == "bullish":
            return pattern.points[2].price + wave1 * self.wave3_typical_extension
        else:
            return pattern.points[2].price - wave1 * self.wave3_typical_extension

    def _project_wave5_target(self, pattern: WavePattern) -> float:
        """Project Wave 5 target."""
        if len(pattern.points) < 5:
            return 0.0
        wave1 = abs(pattern.points[1].price - pattern.points[0].price)
        if pattern.direction == "bullish":
            return pattern.points[4].price + wave1 * self.wave5_typical_extension
        else:
            return pattern.points[4].price - wave1 * self.wave5_typical_extension

    def _get_current_wave_position(
        self, pattern: WavePattern, current_price: float,
    ) -> int:
        """Determine which wave we are currently in."""
        if not pattern.points:
            return 0

        last_point = pattern.points[-1]

        if pattern.wave_type == "impulse":
            if len(pattern.points) <= 1:
                return 1
            elif len(pattern.points) == 2:
                return 2
            elif len(pattern.points) == 3:
                return 3
            elif len(pattern.points) == 4:
                return 4
            elif len(pattern.points) == 5:
                return 5
            else:
                return 5
        else:
            if len(pattern.points) <= 1:
                return 1  # Wave A
            elif len(pattern.points) == 2:
                return 2  # Wave B
            elif len(pattern.points) == 3:
                return 3  # Wave C
            else:
                return 3

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(self.swing_lookback * 2 + 10, 50, self.atr_period + 5)
        if len(candles) < min_len:
            return None

        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])

        # Find swings
        swing_highs, swing_lows = self._find_swings(high, low)
        self._swing_highs = swing_highs
        self._swing_lows = swing_lows

        swings = self._build_swing_sequence(swing_highs, swing_lows)

        # ATR
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Exit check
        if self.has_position:
            self.update_position_price(current_price)

            # Check invalidation
            if self._current_pattern and not self._current_pattern.is_valid:
                side = self.position.side
                self.close_position(current_price, "wave_invalidated")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG if side == "LONG" else SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.85,
                    reason="Elliott Wave pattern invalidated",
                )

            # Stop loss / take profit
            if self.position.stop_loss > 0:
                if self.position.side == "LONG" and current_price <= self.position.stop_loss:
                    self.close_position(current_price, "stop_loss")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol, price=current_price,
                        confidence=0.9, reason="Stop loss",
                    )
                elif self.position.side == "SHORT" and current_price >= self.position.stop_loss:
                    self.close_position(current_price, "stop_loss")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol, price=current_price,
                        confidence=0.9, reason="Stop loss",
                    )

            if self.position.take_profit > 0:
                if self.position.side == "LONG" and current_price >= self.position.take_profit:
                    self.close_position(current_price, "wave_target")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol, price=current_price,
                        confidence=0.85, reason="Wave target reached",
                    )
                elif self.position.side == "SHORT" and current_price <= self.position.take_profit:
                    self.close_position(current_price, "wave_target")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol, price=current_price,
                        confidence=0.85, reason="Wave target reached",
                    )

            return None

        # Try to find patterns
        impulse = self._count_impulse_waves(swings)
        corrective = self._count_corrective_waves(swings)

        pattern = None
        trade_reason = ""

        # Trading logic based on wave position
        if impulse and impulse.is_valid:
            current_wave = self._get_current_wave_position(impulse, current_price)

            if current_wave == 2 and len(impulse.points) >= 3:
                # End of Wave 2 correction - enter for Wave 3
                pattern = impulse
                wave3_target = self._project_wave3_target(impulse)
                trade_reason = f"Wave 3 entry (target={wave3_target:.4f})"
                pattern.target_price = wave3_target

            elif current_wave == 4 and len(impulse.points) >= 5:
                # End of Wave 4 correction - enter for Wave 5
                pattern = impulse
                wave5_target = self._project_wave5_target(impulse)
                trade_reason = f"Wave 5 entry (target={wave5_target:.4f})"
                pattern.target_price = wave5_target

        if pattern is None and corrective and corrective.is_valid:
            if corrective.is_complete:
                # Corrective pattern complete - trade the reversal
                pattern = corrective
                trade_reason = f"Corrective {corrective.corrective_type} complete - reversal entry"
                # After corrective, trade in impulse direction (opposite of correction)
                if corrective.direction == "bearish":
                    pattern.direction = "bullish"  # reversal
                else:
                    pattern.direction = "bearish"

        if pattern is None:
            return None

        self._current_pattern = pattern

        if pattern.direction == "bullish":
            sl = current_price - self.atr_stop_mult * atr_val
            tp = pattern.target_price if pattern.target_price > current_price else current_price + 3 * atr_val
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
                confidence=0.6,
                reason=f"EW {trade_reason}",
                metadata={
                    "wave_type": pattern.wave_type,
                    "direction": pattern.direction,
                    "current_wave": self._get_current_wave_position(pattern, current_price),
                    "target": pattern.target_price,
                    "invalidation": pattern.invalidation_price,
                    "corrective_type": pattern.corrective_type,
                    "wave_count": len(pattern.points),
                },
            )
            if self.check_risk(signal):
                self.open_position("LONG", current_price, qty, sl, tp)
                return signal

        elif pattern.direction == "bearish":
            sl = current_price + self.atr_stop_mult * atr_val
            tp = pattern.target_price if pattern.target_price < current_price else current_price - 3 * atr_val
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
                confidence=0.6,
                reason=f"EW {trade_reason}",
                metadata={
                    "wave_type": pattern.wave_type,
                    "direction": pattern.direction,
                    "current_wave": self._get_current_wave_position(pattern, current_price),
                    "target": pattern.target_price,
                    "invalidation": pattern.invalidation_price,
                    "corrective_type": pattern.corrective_type,
                    "wave_count": len(pattern.points),
                },
            )
            if self.check_risk(signal):
                self.open_position("SHORT", current_price, qty, sl, tp)
                return signal

        return None

    def on_tick(self, price: float, volume: float, timestamp: float) -> Optional[StrategySignal]:
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
        self._logger.info("EW fill: %s", order_info)
