"""
Mefai Signal Engine - Order Flow Analysis Strategy

Cumulative Volume Delta (CVD), aggressive buyer/seller detection,
absorption detection, exhaustion detection, delta divergence,
and footprint chart logic.
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
class TradeFlow:
    """A single trade with aggressor classification."""
    price: float
    quantity: float
    is_buyer_aggressor: bool   # True = market buy, False = market sell
    timestamp: float


@dataclass
class FootprintBar:
    """Volume profile within a single candle (footprint chart)."""
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    delta: float = 0.0            # buy_volume - sell_volume
    max_delta_price: float = 0.0  # price level with highest delta
    min_delta_price: float = 0.0  # price level with lowest delta
    total_volume: float = 0.0
    num_trades: int = 0
    absorption_detected: bool = False
    exhaustion_detected: bool = False

    def calculate_delta(self) -> None:
        self.delta = self.buy_volume - self.sell_volume
        self.total_volume = self.buy_volume + self.sell_volume


@dataclass
class DeltaState:
    """Cumulative Volume Delta state."""
    cumulative_delta: float = 0.0
    session_delta: float = 0.0
    delta_ma: float = 0.0
    divergence_detected: bool = False
    divergence_type: str = ""       # "bullish" or "bearish"


class OrderFlowStrategy(BaseStrategy):
    """
    Order Flow Analysis Strategy

    Trades based on order flow data:
    1. Cumulative Volume Delta (CVD) - net buying vs selling pressure
    2. Aggressive buyer/seller detection from trade data
    3. Absorption: large volume at a level with no price movement
    4. Exhaustion: decreasing delta at swing points
    5. Delta divergence: price vs CVD disagreement
    6. Footprint chart analysis for precise entry
    """

    name = "OrderFlow"
    description = "Order flow analysis - CVD, delta divergence, absorption detection"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # CVD
        self.cvd_lookback: int = self.params.get("cvd_lookback", 50)
        self.cvd_ma_period: int = self.params.get("cvd_ma_period", 14)

        # Delta thresholds
        self.delta_threshold: float = self.params.get("delta_threshold", 0.3)
        self.absorption_volume_mult: float = self.params.get("absorption_volume_mult", 3.0)
        self.absorption_price_tolerance: float = self.params.get("absorption_price_tolerance", 0.1)
        self.exhaustion_delta_decline: float = self.params.get("exhaustion_delta_decline", 0.5)

        # Divergence
        self.divergence_lookback: int = self.params.get("divergence_lookback", 14)

        # Aggressive trade detection
        self.large_trade_mult: float = self.params.get("large_trade_mult", 5.0)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_mult", 2.0)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # State
        self._trade_flow: deque = deque(maxlen=10000)
        self._footprint_bars: List[FootprintBar] = []
        self._delta_state = DeltaState()
        self._cvd_history: deque = deque(maxlen=500)
        self._bar_deltas: deque = deque(maxlen=200)
        self._large_trades: deque = deque(maxlen=100)
        self._avg_trade_size: float = 0.0

    # ------------------------------------------------------------------
    # Trade flow processing
    # ------------------------------------------------------------------

    def add_trade(
        self, price: float, quantity: float, is_buyer: bool, timestamp: float,
    ) -> None:
        """
        Add a raw trade for order flow analysis.
        Called by the data feed for each trade event.
        """
        trade = TradeFlow(
            price=price,
            quantity=quantity,
            is_buyer_aggressor=is_buyer,
            timestamp=timestamp,
        )
        self._trade_flow.append(trade)

        # Update CVD
        if is_buyer:
            self._delta_state.cumulative_delta += quantity
            self._delta_state.session_delta += quantity
        else:
            self._delta_state.cumulative_delta -= quantity
            self._delta_state.session_delta -= quantity

        self._cvd_history.append(self._delta_state.cumulative_delta)

        # Track average trade size
        all_qtys = [t.quantity for t in self._trade_flow]
        if all_qtys:
            self._avg_trade_size = float(np.mean(all_qtys))

        # Detect large trades
        if self._avg_trade_size > 0 and quantity > self._avg_trade_size * self.large_trade_mult:
            self._large_trades.append(trade)

    # ------------------------------------------------------------------
    # Footprint bar construction
    # ------------------------------------------------------------------

    def _build_footprint_from_candle(
        self, ohlcv: np.ndarray, trade_data: Optional[List[TradeFlow]] = None,
    ) -> FootprintBar:
        """
        Build a footprint bar from OHLCV data.
        If trade_data is available, uses actual trade classification.
        Otherwise, estimates buy/sell volume from candle structure.
        """
        o, h, l, c, v = float(ohlcv[1]), float(ohlcv[2]), float(ohlcv[3]), float(ohlcv[4]), float(ohlcv[5])
        bar = FootprintBar(open_price=o, high_price=h, low_price=l, close_price=c)

        if trade_data:
            for trade in trade_data:
                if trade.is_buyer_aggressor:
                    bar.buy_volume += trade.quantity
                else:
                    bar.sell_volume += trade.quantity
                bar.num_trades += 1
        else:
            # Estimate from candle: close > open = more buying
            full_range = h - l
            if full_range > 0:
                body_pct = abs(c - o) / full_range
                if c > o:
                    # Bullish candle
                    bar.buy_volume = v * (0.5 + body_pct * 0.3)
                    bar.sell_volume = v - bar.buy_volume
                elif c < o:
                    # Bearish candle
                    bar.sell_volume = v * (0.5 + body_pct * 0.3)
                    bar.buy_volume = v - bar.sell_volume
                else:
                    bar.buy_volume = v * 0.5
                    bar.sell_volume = v * 0.5
            else:
                bar.buy_volume = v * 0.5
                bar.sell_volume = v * 0.5

        bar.calculate_delta()
        return bar

    # ------------------------------------------------------------------
    # Absorption detection
    # ------------------------------------------------------------------

    def _detect_absorption(self, bars: List[FootprintBar]) -> Optional[Tuple[str, float]]:
        """
        Absorption: high volume but no price movement.
        Large volume absorbed at a level = strong support/resistance.
        """
        if len(bars) < 3:
            return None

        recent = bars[-3:]
        avg_vol = np.mean([b.total_volume for b in bars[-20:]]) if len(bars) >= 20 else np.mean([b.total_volume for b in bars])

        for bar in recent:
            price_move_pct = abs(bar.close_price - bar.open_price) / bar.open_price * 100 if bar.open_price > 0 else 0

            if (bar.total_volume > avg_vol * self.absorption_volume_mult
                    and price_move_pct < self.absorption_price_tolerance):
                bar.absorption_detected = True

                if bar.delta > 0:
                    # Buying absorbed = resistance (sellers absorbing buy orders)
                    return "bearish", bar.close_price
                else:
                    # Selling absorbed = support (buyers absorbing sell orders)
                    return "bullish", bar.close_price

        return None

    # ------------------------------------------------------------------
    # Exhaustion detection
    # ------------------------------------------------------------------

    def _detect_exhaustion(
        self, bars: List[FootprintBar], close: np.ndarray,
    ) -> Optional[str]:
        """
        Exhaustion: decreasing delta magnitude at a swing point.
        Indicates the trend is running out of steam.
        """
        if len(bars) < 5:
            return None

        recent_deltas = [abs(b.delta) for b in bars[-5:]]

        # Check if delta is declining while price still trending
        delta_declining = all(recent_deltas[i] > recent_deltas[i + 1]
                              for i in range(len(recent_deltas) - 1))

        if not delta_declining:
            return None

        # Confirm price is at a swing point
        if len(close) < 5:
            return None

        recent_close = close[-5:]
        price_rising = all(recent_close[i] <= recent_close[i + 1]
                           for i in range(len(recent_close) - 1))
        price_falling = all(recent_close[i] >= recent_close[i + 1]
                            for i in range(len(recent_close) - 1))

        if price_rising:
            bars[-1].exhaustion_detected = True
            return "bearish"  # buying exhaustion at top
        elif price_falling:
            bars[-1].exhaustion_detected = True
            return "bullish"  # selling exhaustion at bottom

        return None

    # ------------------------------------------------------------------
    # Delta divergence
    # ------------------------------------------------------------------

    def _detect_delta_divergence(
        self, close: np.ndarray,
    ) -> Optional[str]:
        """
        Delta divergence: price and CVD moving in opposite directions.
        Bullish: price making lower lows, CVD making higher lows
        Bearish: price making higher highs, CVD making lower highs
        """
        if len(self._cvd_history) < self.divergence_lookback:
            return None
        if len(close) < self.divergence_lookback:
            return None

        half = self.divergence_lookback // 2
        cvd_arr = np.array(list(self._cvd_history))[-self.divergence_lookback:]
        price_arr = close[-self.divergence_lookback:]

        price_first_low = float(np.min(price_arr[:half]))
        price_second_low = float(np.min(price_arr[half:]))
        cvd_first_low = float(np.min(cvd_arr[:half]))
        cvd_second_low = float(np.min(cvd_arr[half:]))

        # Bullish divergence
        if price_second_low < price_first_low and cvd_second_low > cvd_first_low:
            self._delta_state.divergence_detected = True
            self._delta_state.divergence_type = "bullish"
            return "bullish"

        price_first_high = float(np.max(price_arr[:half]))
        price_second_high = float(np.max(price_arr[half:]))
        cvd_first_high = float(np.max(cvd_arr[:half]))
        cvd_second_high = float(np.max(cvd_arr[half:]))

        # Bearish divergence
        if price_second_high > price_first_high and cvd_second_high < cvd_first_high:
            self._delta_state.divergence_detected = True
            self._delta_state.divergence_type = "bearish"
            return "bearish"

        self._delta_state.divergence_detected = False
        return None

    # ------------------------------------------------------------------
    # CVD trend
    # ------------------------------------------------------------------

    def _get_cvd_trend(self) -> int:
        """Get CVD trend direction. Returns 1, -1, or 0."""
        if len(self._cvd_history) < 10:
            return 0
        cvd_arr = np.array(list(self._cvd_history))
        ma = np.mean(cvd_arr[-self.cvd_ma_period:]) if len(cvd_arr) >= self.cvd_ma_period else np.mean(cvd_arr)
        current = cvd_arr[-1]

        if current > ma * 1.05:
            return 1
        elif current < ma * 0.95:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        if len(candles) < max(self.cvd_lookback, self.atr_period + 5, 30):
            return None

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        current_price = float(close[-1])

        # Build footprint bars from candles
        for i in range(max(0, len(candles) - 20), len(candles)):
            if i >= len(self._footprint_bars):
                fp = self._build_footprint_from_candle(candles[i])
                self._footprint_bars.append(fp)
                self._bar_deltas.append(fp.delta)

        # ATR
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Exit checks
        if self.has_position:
            self.update_position_price(current_price)

            # Delta reversal exit
            cvd_trend = self._get_cvd_trend()
            if self.position.side == "LONG" and cvd_trend == -1:
                self.close_position(current_price, "cvd_reversal")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.75,
                    reason="CVD trend reversed to negative",
                )
            elif self.position.side == "SHORT" and cvd_trend == 1:
                self.close_position(current_price, "cvd_reversal")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.75,
                    reason="CVD trend reversed to positive",
                )

            # Exhaustion exit
            exhaustion = self._detect_exhaustion(self._footprint_bars, close)
            if exhaustion:
                if self.position.side == "LONG" and exhaustion == "bearish":
                    self.close_position(current_price, "exhaustion")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol, price=current_price,
                        confidence=0.8, reason="Buying exhaustion detected",
                    )
                elif self.position.side == "SHORT" and exhaustion == "bullish":
                    self.close_position(current_price, "exhaustion")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol, price=current_price,
                        confidence=0.8, reason="Selling exhaustion detected",
                    )

            # Stop loss
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

            return None

        # Entry signals
        signals_score: Dict[str, float] = {"bullish": 0.0, "bearish": 0.0}

        # 1. CVD trend
        cvd_trend = self._get_cvd_trend()
        if cvd_trend == 1:
            signals_score["bullish"] += 0.3
        elif cvd_trend == -1:
            signals_score["bearish"] += 0.3

        # 2. Delta divergence
        divergence = self._detect_delta_divergence(close)
        if divergence == "bullish":
            signals_score["bullish"] += 0.3
        elif divergence == "bearish":
            signals_score["bearish"] += 0.3

        # 3. Absorption
        absorption = self._detect_absorption(self._footprint_bars)
        if absorption:
            abs_dir, abs_price = absorption
            if abs_dir == "bullish":
                signals_score["bullish"] += 0.25
            elif abs_dir == "bearish":
                signals_score["bearish"] += 0.25

        # 4. Recent bar delta
        if self._footprint_bars:
            last_bar = self._footprint_bars[-1]
            if last_bar.total_volume > 0:
                delta_ratio = last_bar.delta / last_bar.total_volume
                if delta_ratio > self.delta_threshold:
                    signals_score["bullish"] += 0.2
                elif delta_ratio < -self.delta_threshold:
                    signals_score["bearish"] += 0.2

        # 5. Large trade bias
        if self._large_trades:
            recent_large = [t for t in self._large_trades
                            if time.time() - t.timestamp < 300]
            if recent_large:
                buy_vol = sum(t.quantity for t in recent_large if t.is_buyer_aggressor)
                sell_vol = sum(t.quantity for t in recent_large if not t.is_buyer_aggressor)
                if buy_vol > sell_vol * 1.5:
                    signals_score["bullish"] += 0.15
                elif sell_vol > buy_vol * 1.5:
                    signals_score["bearish"] += 0.15

        # Determine direction
        bull_score = signals_score["bullish"]
        bear_score = signals_score["bearish"]
        min_score = 0.5

        if bull_score >= min_score and bull_score > bear_score:
            sl = current_price - self.atr_stop_mult * atr_val
            tp = current_price + self.atr_stop_mult * atr_val * 2.0
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
                confidence=min(0.9, 0.4 + bull_score),
                reason=f"Order flow LONG: score={bull_score:.2f}",
                metadata={
                    "cvd_trend": cvd_trend,
                    "divergence": divergence,
                    "absorption": absorption[0] if absorption else None,
                    "bar_delta": self._footprint_bars[-1].delta if self._footprint_bars else 0,
                    "cvd": self._delta_state.cumulative_delta,
                },
            )
            if self.check_risk(signal):
                self.open_position("LONG", current_price, qty, sl, tp)
                return signal

        elif bear_score >= min_score and bear_score > bull_score:
            sl = current_price + self.atr_stop_mult * atr_val
            tp = current_price - self.atr_stop_mult * atr_val * 2.0
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
                confidence=min(0.9, 0.4 + bear_score),
                reason=f"Order flow SHORT: score={bear_score:.2f}",
                metadata={
                    "cvd_trend": cvd_trend,
                    "divergence": divergence,
                    "absorption": absorption[0] if absorption else None,
                    "bar_delta": self._footprint_bars[-1].delta if self._footprint_bars else 0,
                    "cvd": self._delta_state.cumulative_delta,
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
        self._logger.info("OF fill: %s", order_info)

    def get_flow_status(self) -> Dict[str, Any]:
        return {
            "cvd": self._delta_state.cumulative_delta,
            "session_delta": self._delta_state.session_delta,
            "divergence": self._delta_state.divergence_type if self._delta_state.divergence_detected else None,
            "large_trades_5min": len([t for t in self._large_trades if time.time() - t.timestamp < 300]),
            "avg_trade_size": self._avg_trade_size,
            "footprint_bars": len(self._footprint_bars),
        }
