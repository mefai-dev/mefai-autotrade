"""
Mefai Signal Engine - Volume Profile Strategy

Point of Control (POC) identification, Value Area High/Low, high volume
nodes as support/resistance, low volume nodes for fast price movement,
VWAP deviation bands, and session-based volume profile construction.
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
class VolumeNode:
    """A volume node in the profile."""
    price_low: float
    price_high: float
    volume: float
    price_mid: float = 0.0
    is_high_volume: bool = False
    is_low_volume: bool = False

    def __post_init__(self):
        self.price_mid = (self.price_low + self.price_high) / 2.0


@dataclass
class VolumeProfileData:
    """Complete volume profile for a session or period."""
    poc_price: float = 0.0           # Point of Control
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    total_volume: float = 0.0
    nodes: List[VolumeNode] = field(default_factory=list)
    high_volume_nodes: List[VolumeNode] = field(default_factory=list)
    low_volume_nodes: List[VolumeNode] = field(default_factory=list)


class VolumeProfileStrategy(BaseStrategy):
    """
    Volume Profile Strategy

    Analyzes the distribution of volume at different price levels:
    1. Point of Control (POC) - price level with highest volume (magnet)
    2. Value Area High/Low - zone containing 70% of total volume
    3. High Volume Nodes (HVN) - act as support/resistance
    4. Low Volume Nodes (LVN) - price tends to move fast through these
    5. VWAP deviation bands for intraday reference
    6. Session-based profile construction (daily, weekly)
    """

    name = "VolumeProfile"
    description = "Volume profile analysis with POC, value area, and VWAP"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Profile construction
        self.num_bins: int = self.params.get("num_bins", 50)
        self.value_area_pct: float = self.params.get("value_area_pct", 70.0)
        self.session_bars: int = self.params.get("session_bars", 288)  # 288 x 5m = 1 day

        # Volume node classification
        self.hvn_threshold_mult: float = self.params.get("hvn_threshold_mult", 1.5)
        self.lvn_threshold_mult: float = self.params.get("lvn_threshold_mult", 0.5)

        # VWAP
        self.vwap_std_bands: List[float] = self.params.get("vwap_std_bands", [1.0, 2.0, 3.0])

        # Trading rules
        self.trade_poc_bounce: bool = self.params.get("trade_poc_bounce", True)
        self.trade_va_edge: bool = self.params.get("trade_va_edge", True)
        self.trade_lvn_breakout: bool = self.params.get("trade_lvn_breakout", True)
        self.poc_proximity_pct: float = self.params.get("poc_proximity_pct", 0.1)
        self.va_proximity_pct: float = self.params.get("va_proximity_pct", 0.15)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_mult", 1.5)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # State
        self._current_profile: Optional[VolumeProfileData] = None
        self._previous_profile: Optional[VolumeProfileData] = None
        self._session_candle_count: int = 0

    # ------------------------------------------------------------------
    # Volume profile construction
    # ------------------------------------------------------------------

    def _build_profile(self, candles: np.ndarray) -> VolumeProfileData:
        """
        Build volume profile from OHLCV candles.
        Distributes each bar's volume across price bins proportionally.
        """
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)

        price_min = float(np.min(low))
        price_max = float(np.max(high))
        price_range = price_max - price_min

        if price_range <= 0:
            return VolumeProfileData()

        bin_size = price_range / self.num_bins
        bins = np.zeros(self.num_bins, dtype=np.float64)

        # Distribute volume across bins
        for i in range(len(candles)):
            bar_high = float(high[i])
            bar_low = float(low[i])
            bar_vol = float(volume[i])
            bar_range = bar_high - bar_low

            if bar_range <= 0 or bar_vol <= 0:
                continue

            # Find which bins this bar covers
            low_bin = max(0, int((bar_low - price_min) / bin_size))
            high_bin = min(self.num_bins - 1, int((bar_high - price_min) / bin_size))

            num_bins_covered = high_bin - low_bin + 1
            vol_per_bin = bar_vol / num_bins_covered

            for b in range(low_bin, high_bin + 1):
                bins[b] += vol_per_bin

        # Create nodes
        total_vol = float(np.sum(bins))
        avg_vol = float(np.mean(bins)) if self.num_bins > 0 else 0
        nodes: List[VolumeNode] = []

        for b in range(self.num_bins):
            node = VolumeNode(
                price_low=price_min + b * bin_size,
                price_high=price_min + (b + 1) * bin_size,
                volume=float(bins[b]),
                is_high_volume=float(bins[b]) > avg_vol * self.hvn_threshold_mult,
                is_low_volume=float(bins[b]) < avg_vol * self.lvn_threshold_mult,
            )
            nodes.append(node)

        # POC - bin with highest volume
        poc_bin = int(np.argmax(bins))
        poc_price = price_min + (poc_bin + 0.5) * bin_size

        # Value Area - expand from POC until 70% volume is captured
        va_volume_target = total_vol * (self.value_area_pct / 100.0)
        va_volume = float(bins[poc_bin])
        va_low_bin = poc_bin
        va_high_bin = poc_bin

        while va_volume < va_volume_target:
            # Expand to the side with more volume
            expand_low = va_low_bin > 0
            expand_high = va_high_bin < self.num_bins - 1

            if not expand_low and not expand_high:
                break

            low_vol = float(bins[va_low_bin - 1]) if expand_low else 0
            high_vol = float(bins[va_high_bin + 1]) if expand_high else 0

            if low_vol >= high_vol and expand_low:
                va_low_bin -= 1
                va_volume += float(bins[va_low_bin])
            elif expand_high:
                va_high_bin += 1
                va_volume += float(bins[va_high_bin])
            elif expand_low:
                va_low_bin -= 1
                va_volume += float(bins[va_low_bin])
            else:
                break

        va_high = price_min + (va_high_bin + 1) * bin_size
        va_low = price_min + va_low_bin * bin_size

        hvn_nodes = [n for n in nodes if n.is_high_volume]
        lvn_nodes = [n for n in nodes if n.is_low_volume]

        return VolumeProfileData(
            poc_price=poc_price,
            value_area_high=va_high,
            value_area_low=va_low,
            total_volume=total_vol,
            nodes=nodes,
            high_volume_nodes=hvn_nodes,
            low_volume_nodes=lvn_nodes,
        )

    # ------------------------------------------------------------------
    # VWAP with deviation bands
    # ------------------------------------------------------------------

    def _calculate_vwap_bands(
        self, candles: np.ndarray,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Calculate VWAP and standard deviation bands.
        Returns (vwap, [(upper1, lower1), (upper2, lower2), ...])
        """
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)

        typical = (high + low + close) / 3.0
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical * volume)

        if cum_vol[-1] == 0:
            return float(close[-1]), []

        vwap = float(cum_tp_vol[-1] / cum_vol[-1])

        # Standard deviation of typical price from VWAP
        squared_diff = (typical - vwap) ** 2
        variance = float(np.sum(squared_diff * volume) / cum_vol[-1])
        std_dev = np.sqrt(variance) if variance > 0 else 0.0

        bands = []
        for mult in self.vwap_std_bands:
            bands.append((vwap + mult * std_dev, vwap - mult * std_dev))

        return vwap, bands

    # ------------------------------------------------------------------
    # Signal logic
    # ------------------------------------------------------------------

    def _check_poc_bounce(
        self, current_price: float, profile: VolumeProfileData,
    ) -> Optional[Tuple[str, float]]:
        """Check if price is bouncing off POC."""
        if not self.trade_poc_bounce or profile.poc_price == 0:
            return None

        proximity = abs(current_price - profile.poc_price) / profile.poc_price * 100.0
        if proximity > self.poc_proximity_pct:
            return None

        # POC acts as support from below, resistance from above
        if current_price <= profile.poc_price:
            return "LONG", profile.poc_price
        else:
            return "SHORT", profile.poc_price

    def _check_va_edge(
        self, current_price: float, prev_price: float, profile: VolumeProfileData,
    ) -> Optional[Tuple[str, float]]:
        """Check for value area edge trades."""
        if not self.trade_va_edge:
            return None

        va_h = profile.value_area_high
        va_l = profile.value_area_low

        # Price returning to value area from above VAH -> short
        if prev_price > va_h and current_price <= va_h:
            return "SHORT", va_h

        # Price returning to value area from below VAL -> long
        if prev_price < va_l and current_price >= va_l:
            return "LONG", va_l

        return None

    def _check_lvn_breakout(
        self, current_price: float, prev_price: float, profile: VolumeProfileData,
    ) -> Optional[Tuple[str, float]]:
        """Check for price breaking through low volume nodes."""
        if not self.trade_lvn_breakout:
            return None

        for lvn in profile.low_volume_nodes:
            # Price crossing up through LVN
            if prev_price < lvn.price_low and current_price >= lvn.price_mid:
                return "LONG", lvn.price_high
            # Price crossing down through LVN
            if prev_price > lvn.price_high and current_price <= lvn.price_mid:
                return "SHORT", lvn.price_low

        return None

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        min_len = max(50, self.atr_period + 5)
        if len(candles) < min_len:
            return None

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        current_price = float(close[-1])
        prev_price = float(close[-2]) if len(close) > 1 else current_price

        # Build or update profile
        session_candles = candles[-min(len(candles), self.session_bars):]
        self._current_profile = self._build_profile(session_candles)

        profile = self._current_profile
        if profile.poc_price == 0:
            return None

        # VWAP
        vwap, vwap_bands = self._calculate_vwap_bands(session_candles)

        # ATR for stops
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Exit checks
        if self.has_position:
            self.update_position_price(current_price)

            # Exit at POC (mean reversion target)
            if self.position.side == "LONG" and current_price >= profile.poc_price:
                self.close_position(current_price, "poc_target")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.75,
                    reason=f"POC target reached: {profile.poc_price:.4f}",
                )
            elif self.position.side == "SHORT" and current_price <= profile.poc_price:
                self.close_position(current_price, "poc_target")
                return StrategySignal(
                    signal_type=SignalType.CLOSE_SHORT,
                    symbol=self.symbol,
                    price=current_price,
                    confidence=0.75,
                    reason=f"POC target reached: {profile.poc_price:.4f}",
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

        # Entry checks (priority order)
        trade_signal = None

        # 1. POC bounce
        result = self._check_poc_bounce(current_price, profile)
        if result:
            trade_signal = result

        # 2. Value area edge
        if trade_signal is None:
            result = self._check_va_edge(current_price, prev_price, profile)
            if result:
                trade_signal = result

        # 3. LVN breakout
        if trade_signal is None:
            result = self._check_lvn_breakout(current_price, prev_price, profile)
            if result:
                trade_signal = result

        if trade_signal is None:
            return None

        side, ref_price = trade_signal
        if side == "LONG":
            sl = current_price - self.atr_stop_mult * atr_val
            tp = profile.poc_price if profile.poc_price > current_price else profile.value_area_high
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
                confidence=0.65,
                reason=f"VP LONG near {ref_price:.4f} (POC={profile.poc_price:.4f})",
                metadata={
                    "poc": profile.poc_price,
                    "va_high": profile.value_area_high,
                    "va_low": profile.value_area_low,
                    "vwap": vwap,
                    "hvn_count": len(profile.high_volume_nodes),
                    "lvn_count": len(profile.low_volume_nodes),
                },
            )
            if self.check_risk(signal):
                self.open_position("LONG", current_price, qty, sl, tp)
                return signal

        elif side == "SHORT":
            sl = current_price + self.atr_stop_mult * atr_val
            tp = profile.poc_price if profile.poc_price < current_price else profile.value_area_low
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
                confidence=0.65,
                reason=f"VP SHORT near {ref_price:.4f} (POC={profile.poc_price:.4f})",
                metadata={
                    "poc": profile.poc_price,
                    "va_high": profile.value_area_high,
                    "va_low": profile.value_area_low,
                    "vwap": vwap,
                    "hvn_count": len(profile.high_volume_nodes),
                    "lvn_count": len(profile.low_volume_nodes),
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
        self._logger.info("VP fill: %s", order_info)

    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Return current volume profile data."""
        if self._current_profile is None:
            return None
        p = self._current_profile
        return {
            "poc": p.poc_price,
            "va_high": p.value_area_high,
            "va_low": p.value_area_low,
            "total_volume": p.total_volume,
            "hvn_prices": [n.price_mid for n in p.high_volume_nodes],
            "lvn_prices": [n.price_mid for n in p.low_volume_nodes],
        }
