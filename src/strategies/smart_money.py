"""
Mefai Signal Engine - Smart Money Concepts (SMC) Strategy

Order block identification, Fair Value Gap (FVG) detection, Break of
Structure (BOS), Change of Character (ChoCH), liquidity sweep detection,
premium/discount zones, and multi-timeframe confluence scoring.
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
class OrderBlock:
    """Institutional order block zone."""
    price_high: float
    price_low: float
    direction: str         # "bullish" or "bearish"
    index: int             # candle index where OB formed
    strength: float = 1.0  # based on impulse move size
    tested: bool = False
    invalidated: bool = False
    timestamp: float = 0.0


@dataclass
class FairValueGap:
    """Fair Value Gap (imbalance zone)."""
    high: float           # upper boundary
    low: float            # lower boundary
    direction: str        # "bullish" (gap up) or "bearish" (gap down)
    index: int
    filled: bool = False
    fill_pct: float = 0.0


@dataclass
class StructurePoint:
    """Swing high or low for structure analysis."""
    price: float
    index: int
    point_type: str       # "HH", "HL", "LH", "LL"


@dataclass
class SMCConfluence:
    """Multi-timeframe confluence scoring."""
    order_block: bool = False
    fvg: bool = False
    structure: bool = False
    liquidity_sweep: bool = False
    premium_discount: bool = False
    fibonacci: bool = False
    total_score: int = 0

    def calculate(self) -> None:
        self.total_score = sum([
            self.order_block,
            self.fvg,
            self.structure,
            self.liquidity_sweep,
            self.premium_discount,
            self.fibonacci,
        ])


class SmartMoneyStrategy(BaseStrategy):
    """
    Smart Money Concepts (SMC) Strategy

    Trades based on institutional order flow concepts:
    1. Order Block identification (last candle before impulsive move)
    2. Fair Value Gap (FVG) entry at imbalance zones
    3. Break of Structure (BOS) for trend continuation
    4. Change of Character (ChoCH) for reversal detection
    5. Liquidity sweep detection (stop hunts)
    6. Premium/Discount zones using Fibonacci
    7. Multi-timeframe confluence scoring
    """

    name = "SmartMoney"
    description = "Smart Money Concepts - order blocks, FVG, BOS, liquidity sweeps"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Structure detection
        self.swing_lookback: int = self.params.get("swing_lookback", 5)
        self.min_impulse_pct: float = self.params.get("min_impulse_pct", 1.0)

        # Order blocks
        self.ob_max_age: int = self.params.get("ob_max_age", 100)  # bars
        self.ob_max_count: int = self.params.get("ob_max_count", 10)

        # FVG
        self.fvg_min_size_pct: float = self.params.get("fvg_min_size_pct", 0.1)
        self.fvg_max_age: int = self.params.get("fvg_max_age", 50)

        # Confluence
        self.min_confluence: int = self.params.get("min_confluence", 3)

        # Fibonacci
        self.fib_levels: List[float] = self.params.get(
            "fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786]
        )
        self.premium_zone: float = self.params.get("premium_zone", 0.5)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_mult", 1.5)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # State
        self._order_blocks: List[OrderBlock] = []
        self._fvgs: List[FairValueGap] = []
        self._structure_points: List[StructurePoint] = []
        self._last_swing_high: float = 0.0
        self._last_swing_low: float = float("inf")
        self._market_structure: str = "undefined"  # "bullish", "bearish", "undefined"
        self._prev_structure: str = "undefined"

    # ------------------------------------------------------------------
    # Swing point detection
    # ------------------------------------------------------------------

    def _find_swing_points(
        self, high: np.ndarray, low: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows."""
        n = len(high)
        lb = self.swing_lookback
        swing_highs = []
        swing_lows = []

        for i in range(lb, n - lb):
            # Swing high: higher than all neighbors
            is_high = True
            for j in range(1, lb + 1):
                if high[i] <= high[i - j] or high[i] <= high[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append(i)

            # Swing low: lower than all neighbors
            is_low = True
            for j in range(1, lb + 1):
                if low[i] >= low[i - j] or low[i] >= low[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append(i)

        return swing_highs, swing_lows

    # ------------------------------------------------------------------
    # Market structure (BOS / ChoCH)
    # ------------------------------------------------------------------

    def _analyze_structure(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
        swing_highs: List[int], swing_lows: List[int],
    ) -> Tuple[str, bool, bool]:
        """
        Analyze market structure for BOS (Break of Structure) and ChoCH.

        Returns (structure, bos_detected, choch_detected).
        structure: "bullish", "bearish", or "undefined"
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "undefined", False, False

        # Get last two swing highs and lows
        sh1_idx = swing_highs[-2]
        sh2_idx = swing_highs[-1]
        sl1_idx = swing_lows[-2]
        sl2_idx = swing_lows[-1]

        sh1 = float(high[sh1_idx])
        sh2 = float(high[sh2_idx])
        sl1 = float(low[sl1_idx])
        sl2 = float(low[sl2_idx])

        current_price = float(close[-1])
        bos = False
        choch = False

        # Determine structure
        if sh2 > sh1 and sl2 > sl1:
            # Higher highs, higher lows = bullish
            new_structure = "bullish"
        elif sh2 < sh1 and sl2 < sl1:
            # Lower highs, lower lows = bearish
            new_structure = "bearish"
        else:
            new_structure = self._market_structure  # keep previous

        # BOS: continuation of current structure
        if self._market_structure == "bullish" and current_price > sh2:
            bos = True
            self._structure_points.append(StructurePoint(
                price=current_price, index=len(close) - 1, point_type="HH",
            ))
        elif self._market_structure == "bearish" and current_price < sl2:
            bos = True
            self._structure_points.append(StructurePoint(
                price=current_price, index=len(close) - 1, point_type="LL",
            ))

        # ChoCH: change of structure
        if self._market_structure == "bullish" and new_structure == "bearish":
            choch = True
        elif self._market_structure == "bearish" and new_structure == "bullish":
            choch = True

        self._prev_structure = self._market_structure
        self._market_structure = new_structure
        self._last_swing_high = sh2
        self._last_swing_low = sl2

        return new_structure, bos, choch

    # ------------------------------------------------------------------
    # Order Block detection
    # ------------------------------------------------------------------

    def _detect_order_blocks(
        self, candles: np.ndarray, swing_highs: List[int], swing_lows: List[int],
    ) -> None:
        """
        Find order blocks - the last opposite candle before an impulsive move.
        Bullish OB: last bearish candle before a strong bullish move
        Bearish OB: last bullish candle before a strong bearish move
        """
        open_p = candles[:, 1].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        n = len(close)

        new_obs: List[OrderBlock] = []

        # Check recent swing lows for bullish OBs
        for sl_idx in swing_lows[-5:]:
            if sl_idx < 1 or sl_idx >= n - 3:
                continue
            # Check if move after swing low was impulsive
            move_pct = ((high[sl_idx + 2] - low[sl_idx]) / low[sl_idx]) * 100.0
            if move_pct < self.min_impulse_pct:
                continue

            # Find last bearish candle at or before the swing low
            ob_idx = sl_idx
            while ob_idx >= max(0, sl_idx - 3):
                if close[ob_idx] < open_p[ob_idx]:  # bearish candle
                    strength = move_pct / self.min_impulse_pct
                    new_obs.append(OrderBlock(
                        price_high=float(high[ob_idx]),
                        price_low=float(low[ob_idx]),
                        direction="bullish",
                        index=ob_idx,
                        strength=min(strength, 5.0),
                        timestamp=time.time(),
                    ))
                    break
                ob_idx -= 1

        # Check recent swing highs for bearish OBs
        for sh_idx in swing_highs[-5:]:
            if sh_idx < 1 or sh_idx >= n - 3:
                continue
            move_pct = ((high[sh_idx] - low[sh_idx + 2]) / high[sh_idx]) * 100.0
            if move_pct < self.min_impulse_pct:
                continue

            ob_idx = sh_idx
            while ob_idx >= max(0, sh_idx - 3):
                if close[ob_idx] > open_p[ob_idx]:  # bullish candle
                    strength = move_pct / self.min_impulse_pct
                    new_obs.append(OrderBlock(
                        price_high=float(high[ob_idx]),
                        price_low=float(low[ob_idx]),
                        direction="bearish",
                        index=ob_idx,
                        strength=min(strength, 5.0),
                        timestamp=time.time(),
                    ))
                    break
                ob_idx -= 1

        # Merge with existing, remove old/invalidated
        self._order_blocks = [
            ob for ob in self._order_blocks
            if not ob.invalidated and (n - ob.index) < self.ob_max_age
        ]

        for new_ob in new_obs:
            # Check for duplicates
            is_dup = False
            for existing in self._order_blocks:
                if (abs(existing.price_high - new_ob.price_high) < existing.price_high * 0.001
                        and existing.direction == new_ob.direction):
                    is_dup = True
                    break
            if not is_dup:
                self._order_blocks.append(new_ob)

        # Keep only top N by strength
        self._order_blocks.sort(key=lambda x: x.strength, reverse=True)
        self._order_blocks = self._order_blocks[:self.ob_max_count]

    # ------------------------------------------------------------------
    # Fair Value Gap detection
    # ------------------------------------------------------------------

    def _detect_fvgs(self, candles: np.ndarray) -> None:
        """
        Detect Fair Value Gaps (3-candle imbalance pattern).
        Bullish FVG: candle[i+2] low > candle[i] high (gap up)
        Bearish FVG: candle[i] low > candle[i+2] high (gap down)
        """
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        n = len(high)

        new_fvgs: List[FairValueGap] = []

        for i in range(max(0, n - 20), n - 2):
            # Bullish FVG
            if low[i + 2] > high[i]:
                gap_size = low[i + 2] - high[i]
                gap_pct = (gap_size / high[i]) * 100.0
                if gap_pct >= self.fvg_min_size_pct:
                    new_fvgs.append(FairValueGap(
                        high=float(low[i + 2]),
                        low=float(high[i]),
                        direction="bullish",
                        index=i + 1,
                    ))

            # Bearish FVG
            if low[i] > high[i + 2]:
                gap_size = low[i] - high[i + 2]
                gap_pct = (gap_size / low[i]) * 100.0
                if gap_pct >= self.fvg_min_size_pct:
                    new_fvgs.append(FairValueGap(
                        high=float(low[i]),
                        low=float(high[i + 2]),
                        direction="bearish",
                        index=i + 1,
                    ))

        # Update fill status of existing FVGs
        current_price = float(candles[-1, 4])
        for fvg in self._fvgs:
            if fvg.filled:
                continue
            if fvg.direction == "bullish":
                if current_price <= fvg.low:
                    fvg.fill_pct = 100.0
                    fvg.filled = True
                elif current_price < fvg.high:
                    fvg.fill_pct = ((fvg.high - current_price) / (fvg.high - fvg.low)) * 100.0
            else:
                if current_price >= fvg.high:
                    fvg.fill_pct = 100.0
                    fvg.filled = True
                elif current_price > fvg.low:
                    fvg.fill_pct = ((current_price - fvg.low) / (fvg.high - fvg.low)) * 100.0

        # Merge new FVGs
        self._fvgs = [f for f in self._fvgs if not f.filled and (n - f.index) < self.fvg_max_age]
        for nf in new_fvgs:
            is_dup = False
            for ef in self._fvgs:
                if abs(ef.high - nf.high) < ef.high * 0.001 and ef.direction == nf.direction:
                    is_dup = True
                    break
            if not is_dup:
                self._fvgs.append(nf)

    # ------------------------------------------------------------------
    # Liquidity sweep
    # ------------------------------------------------------------------

    def _detect_liquidity_sweep(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    ) -> Optional[Tuple[str, float]]:
        """
        Detect liquidity sweeps (stop hunts).
        Price briefly breaks above/below a swing point then reverses back.
        """
        n = len(close)
        if n < 3:
            return None

        current_high = float(high[-1])
        current_low = float(low[-1])
        current_close = float(close[-1])

        # Bullish sweep: price wicked below swing low then closed above
        if self._last_swing_low < float("inf"):
            if current_low < self._last_swing_low and current_close > self._last_swing_low:
                sweep_pct = ((self._last_swing_low - current_low) / self._last_swing_low) * 100.0
                if sweep_pct > 0.05:  # minimum sweep size
                    return "bullish", self._last_swing_low

        # Bearish sweep: price wicked above swing high then closed below
        if self._last_swing_high > 0:
            if current_high > self._last_swing_high and current_close < self._last_swing_high:
                sweep_pct = ((current_high - self._last_swing_high) / self._last_swing_high) * 100.0
                if sweep_pct > 0.05:
                    return "bearish", self._last_swing_high

        return None

    # ------------------------------------------------------------------
    # Premium / Discount zones
    # ------------------------------------------------------------------

    def _get_premium_discount(self, current_price: float) -> str:
        """
        Determine if price is in premium or discount zone.
        Uses Fibonacci between last swing high and swing low.
        """
        if self._last_swing_high == 0 or self._last_swing_low == float("inf"):
            return "neutral"

        swing_range = self._last_swing_high - self._last_swing_low
        if swing_range <= 0:
            return "neutral"

        # Equilibrium (50%)
        equilibrium = self._last_swing_low + swing_range * self.premium_zone

        if current_price > equilibrium:
            return "premium"
        else:
            return "discount"

    def _get_fib_level(self, current_price: float) -> float:
        """Get the nearest Fibonacci retracement level."""
        if self._last_swing_high == 0 or self._last_swing_low == float("inf"):
            return 0.0

        swing_range = self._last_swing_high - self._last_swing_low
        if swing_range <= 0:
            return 0.0

        retracement = (self._last_swing_high - current_price) / swing_range

        nearest = 0.0
        min_dist = float("inf")
        for fib in self.fib_levels:
            dist = abs(retracement - fib)
            if dist < min_dist:
                min_dist = dist
                nearest = fib

        return nearest

    # ------------------------------------------------------------------
    # Confluence scoring
    # ------------------------------------------------------------------

    def _score_confluence(
        self, current_price: float, direction: str,
    ) -> SMCConfluence:
        """Score how many SMC concepts align at the current price."""
        conf = SMCConfluence()

        # Order block
        for ob in self._order_blocks:
            if ob.invalidated or ob.tested:
                continue
            if ob.direction == direction:
                if ob.price_low <= current_price <= ob.price_high:
                    conf.order_block = True
                    break

        # FVG
        for fvg in self._fvgs:
            if fvg.filled:
                continue
            if fvg.direction == direction:
                if fvg.low <= current_price <= fvg.high:
                    conf.fvg = True
                    break

        # Structure
        if direction == "bullish" and self._market_structure == "bullish":
            conf.structure = True
        elif direction == "bearish" and self._market_structure == "bearish":
            conf.structure = True

        # Premium/Discount
        zone = self._get_premium_discount(current_price)
        if direction == "bullish" and zone == "discount":
            conf.premium_discount = True
        elif direction == "bearish" and zone == "premium":
            conf.premium_discount = True

        # Fibonacci
        fib = self._get_fib_level(current_price)
        if fib in [0.618, 0.786, 0.5]:
            conf.fibonacci = True

        conf.calculate()
        return conf

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(self.swing_lookback * 2 + 5, 30, self.atr_period + 5)
        if len(candles) < min_len:
            return None

        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])

        # Find swing points
        swing_highs, swing_lows = self._find_swing_points(high, low)

        # Analyze structure
        structure, bos, choch = self._analyze_structure(
            high, low, close, swing_highs, swing_lows,
        )

        # Detect order blocks
        self._detect_order_blocks(candles, swing_highs, swing_lows)

        # Detect FVGs
        self._detect_fvgs(candles)

        # Detect liquidity sweep
        sweep = self._detect_liquidity_sweep(high, low, close)

        # ATR for stops
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Exit checks
        if self.has_position:
            self.update_position_price(current_price)

            # Exit on ChoCH (structure change against position)
            if choch:
                if self.position.side == "LONG" and structure == "bearish":
                    self.close_position(current_price, "choch_exit")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.85,
                        reason="Change of Character: bearish structure shift",
                    )
                elif self.position.side == "SHORT" and structure == "bullish":
                    self.close_position(current_price, "choch_exit")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol,
                        price=current_price,
                        confidence=0.85,
                        reason="Change of Character: bullish structure shift",
                    )

            # Stop loss / take profit
            if self.position.stop_loss > 0:
                if self.position.side == "LONG" and current_price <= self.position.stop_loss:
                    self.close_position(current_price, "stop_loss")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol, price=current_price,
                        confidence=0.9, reason="Stop loss hit",
                    )
                elif self.position.side == "SHORT" and current_price >= self.position.stop_loss:
                    self.close_position(current_price, "stop_loss")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol, price=current_price,
                        confidence=0.9, reason="Stop loss hit",
                    )

            return None

        # Entry signals
        # Check bullish confluence
        bull_conf = self._score_confluence(current_price, "bullish")
        bear_conf = self._score_confluence(current_price, "bearish")

        # Liquidity sweep adds to confluence
        if sweep:
            sweep_dir, sweep_level = sweep
            if sweep_dir == "bullish":
                bull_conf.liquidity_sweep = True
                bull_conf.calculate()
            elif sweep_dir == "bearish":
                bear_conf.liquidity_sweep = True
                bear_conf.calculate()

        # Bullish entry
        if bull_conf.total_score >= self.min_confluence:
            sl = current_price - self.atr_stop_mult * atr_val
            # Target: next swing high or 2:1 RR
            tp = current_price + abs(current_price - sl) * 2.0
            if self._last_swing_high > current_price:
                tp = self._last_swing_high

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
                confidence=min(0.95, 0.4 + bull_conf.total_score * 0.1),
                reason=f"SMC LONG: confluence={bull_conf.total_score}/6 "
                       f"structure={structure}",
                metadata={
                    "confluence": bull_conf.total_score,
                    "order_block": bull_conf.order_block,
                    "fvg": bull_conf.fvg,
                    "structure": structure,
                    "liquidity_sweep": bull_conf.liquidity_sweep,
                    "premium_discount": self._get_premium_discount(current_price),
                    "fib_level": self._get_fib_level(current_price),
                    "bos": bos,
                    "choch": choch,
                },
            )
            if self.check_risk(signal):
                self.open_position("LONG", current_price, qty, sl, tp)
                return signal

        # Bearish entry
        elif bear_conf.total_score >= self.min_confluence:
            sl = current_price + self.atr_stop_mult * atr_val
            tp = current_price - abs(sl - current_price) * 2.0
            if self._last_swing_low < current_price and self._last_swing_low > 0:
                tp = self._last_swing_low

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
                confidence=min(0.95, 0.4 + bear_conf.total_score * 0.1),
                reason=f"SMC SHORT: confluence={bear_conf.total_score}/6 "
                       f"structure={structure}",
                metadata={
                    "confluence": bear_conf.total_score,
                    "order_block": bear_conf.order_block,
                    "fvg": bear_conf.fvg,
                    "structure": structure,
                    "liquidity_sweep": bear_conf.liquidity_sweep,
                    "premium_discount": self._get_premium_discount(current_price),
                    "fib_level": self._get_fib_level(current_price),
                    "bos": bos,
                    "choch": choch,
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
        self._logger.info("SMC fill: %s", order_info)
