"""
Mefai Signal Engine - Grid Trading Strategy

Production grid trading with arithmetic/geometric spacing, dynamic ATR-based
adjustment, auto range detection via Bollinger Bands, per-level PnL tracking,
and support for both spot and futures markets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import math
import numpy as np

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType, PositionInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """Represents a single grid level."""
    index: int
    price: float
    side: str              # "BUY" or "SELL"
    quantity: float
    is_filled: bool = False
    fill_price: float = 0.0
    fill_time: float = 0.0
    paired_level: int = -1   # index of the opposite grid that closes this trade
    realized_pnl: float = 0.0


@dataclass
class GridState:
    """Full state of the grid."""
    levels: List[GridLevel] = field(default_factory=list)
    upper_price: float = 0.0
    lower_price: float = 0.0
    grid_count: int = 0
    grid_type: str = "arithmetic"   # "arithmetic" or "geometric"
    total_investment: float = 0.0
    total_pnl: float = 0.0
    filled_buys: int = 0
    filled_sells: int = 0
    last_rebalance_time: float = 0.0


class GridTradingStrategy(BaseStrategy):
    """
    Grid Trading Strategy

    Places a grid of buy and sell orders within a price range.
    Profits from price oscillation within the range by buying low and
    selling high across multiple levels.

    Supports:
    - Arithmetic grid (equal price spacing)
    - Geometric grid (equal percentage spacing)
    - Dynamic volatility-based grid range (ATR)
    - Auto range from Bollinger Bands
    - Per-grid-level PnL tracking
    - Grid rebalancing on trend change
    - Spot and futures modes
    """

    name = "GridTrading"
    description = "Grid trading with dynamic range and per-level PnL tracking"
    version = "1.0.0"

    required_params = []
    param_types = {
        "num_grids": int,
        "grid_type": str,
        "total_investment": (int, float),
    }

    def _initialize_params(self) -> None:
        self.num_grids: int = self.params.get("num_grids", 10)
        self.grid_type: str = self.params.get("grid_type", "arithmetic")
        self.total_investment: float = self.params.get("total_investment", 1000.0)
        self.upper_price: float = self.params.get("upper_price", 0.0)
        self.lower_price: float = self.params.get("lower_price", 0.0)
        self.auto_range: bool = self.params.get("auto_range", True)
        self.atr_range_multiplier: float = self.params.get("atr_range_multiplier", 3.0)
        self.bb_period: int = self.params.get("bb_period", 20)
        self.bb_std: float = self.params.get("bb_std", 2.0)
        self.rebalance_on_trend: bool = self.params.get("rebalance_on_trend", True)
        self.rebalance_cooldown: float = self.params.get("rebalance_cooldown", 3600.0)
        self.market_type: str = self.params.get("market_type", "futures")
        self.profit_per_grid_pct: float = self.params.get("profit_per_grid_pct", 0.0)
        self.min_grids: int = self.params.get("min_grids", 5)
        self.max_grids: int = self.params.get("max_grids", 100)

        self.grid_state = GridState()
        self._grid_initialized = False
        self._last_price = 0.0
        self._trend_direction = 0   # 1 = up, -1 = down, 0 = neutral
        self._candle_buffer: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _detect_range(self, candles: np.ndarray) -> Tuple[float, float]:
        """
        Auto-detect grid range using Bollinger Bands and ATR.
        Returns (lower_price, upper_price).
        """
        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)

        upper_bb, middle_bb, lower_bb = self.indicators.bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        atr = self.indicators.atr(high, low, close, 14)

        last_idx = len(close) - 1
        current_price = close[last_idx]
        bb_upper = upper_bb[last_idx] if not np.isnan(upper_bb[last_idx]) else current_price * 1.05
        bb_lower = lower_bb[last_idx] if not np.isnan(lower_bb[last_idx]) else current_price * 0.95
        atr_val = atr[last_idx] if not np.isnan(atr[last_idx]) else current_price * 0.02

        # Use the wider of BB range or ATR-based range
        atr_upper = current_price + self.atr_range_multiplier * atr_val
        atr_lower = current_price - self.atr_range_multiplier * atr_val

        upper = max(bb_upper, atr_upper)
        lower = min(bb_lower, atr_lower)

        # Ensure minimum range (at least 2% of price)
        min_range = current_price * 0.02
        if upper - lower < min_range:
            mid = (upper + lower) / 2.0
            upper = mid + min_range / 2.0
            lower = mid - min_range / 2.0

        return lower, upper

    def _build_arithmetic_grid(
        self, lower: float, upper: float, count: int, qty_per_level: float,
    ) -> List[GridLevel]:
        """Build grid with equal price spacing."""
        levels = []
        step = (upper - lower) / count
        for i in range(count + 1):
            price = lower + i * step
            # Levels below current mid are BUY, above are SELL
            mid = (upper + lower) / 2.0
            side = "BUY" if price < mid else "SELL"
            levels.append(GridLevel(
                index=i,
                price=round(price, 8),
                side=side,
                quantity=qty_per_level,
                paired_level=count - i,  # symmetric pairing
            ))
        return levels

    def _build_geometric_grid(
        self, lower: float, upper: float, count: int, qty_per_level: float,
    ) -> List[GridLevel]:
        """Build grid with equal percentage spacing."""
        levels = []
        if lower <= 0:
            lower = upper * 0.01  # safety floor
        ratio = (upper / lower) ** (1.0 / count)
        for i in range(count + 1):
            price = lower * (ratio ** i)
            mid = math.sqrt(upper * lower)  # geometric mean
            side = "BUY" if price < mid else "SELL"
            levels.append(GridLevel(
                index=i,
                price=round(price, 8),
                side=side,
                quantity=qty_per_level,
                paired_level=count - i,
            ))
        return levels

    def _initialize_grid(self, candles: np.ndarray) -> None:
        """Set up the full grid from candle data."""
        if self.auto_range and (self.upper_price == 0 or self.lower_price == 0):
            self.lower_price, self.upper_price = self._detect_range(candles)

        num = max(self.min_grids, min(self.num_grids, self.max_grids))
        qty_per_level = self.total_investment / (num * self.upper_price)

        if self.grid_type == "geometric":
            levels = self._build_geometric_grid(
                self.lower_price, self.upper_price, num, qty_per_level,
            )
        else:
            levels = self._build_arithmetic_grid(
                self.lower_price, self.upper_price, num, qty_per_level,
            )

        self.grid_state = GridState(
            levels=levels,
            upper_price=self.upper_price,
            lower_price=self.lower_price,
            grid_count=num,
            grid_type=self.grid_type,
            total_investment=self.total_investment,
        )
        self._grid_initialized = True
        self._logger.info(
            "Grid initialized: %d levels, range %.4f - %.4f, type=%s",
            num, self.lower_price, self.upper_price, self.grid_type,
        )

    # ------------------------------------------------------------------
    # Grid level matching
    # ------------------------------------------------------------------

    def _find_triggered_levels(
        self, previous_price: float, current_price: float,
    ) -> List[GridLevel]:
        """Find grid levels crossed between previous and current price."""
        triggered = []
        for level in self.grid_state.levels:
            if level.is_filled:
                continue
            # Price crossed this level
            if previous_price <= level.price <= current_price:
                if level.side == "SELL":
                    triggered.append(level)
            elif previous_price >= level.price >= current_price:
                if level.side == "BUY":
                    triggered.append(level)
        return triggered

    def _fill_grid_level(
        self, level: GridLevel, fill_price: float,
    ) -> Optional[StrategySignal]:
        """Mark a grid level as filled and create signal."""
        level.is_filled = True
        level.fill_price = fill_price
        level.fill_time = time.time()

        if level.side == "BUY":
            self.grid_state.filled_buys += 1
            signal_type = SignalType.LONG
        else:
            self.grid_state.filled_sells += 1
            signal_type = SignalType.SHORT if self.market_type == "futures" else SignalType.CLOSE_LONG

        # Check for paired level profit
        paired = level.paired_level
        if 0 <= paired < len(self.grid_state.levels):
            paired_level = self.grid_state.levels[paired]
            if paired_level.is_filled:
                pnl = abs(level.price - paired_level.price) * level.quantity
                level.realized_pnl = pnl
                self.grid_state.total_pnl += pnl
                paired_level.is_filled = False  # reset for re-use

        # Calculate target based on grid profit
        if self.profit_per_grid_pct > 0:
            if level.side == "BUY":
                tp = fill_price * (1.0 + self.profit_per_grid_pct / 100.0)
            else:
                tp = fill_price * (1.0 - self.profit_per_grid_pct / 100.0)
        else:
            tp = 0.0

        signal = StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=fill_price,
            quantity=level.quantity,
            take_profit=tp,
            confidence=0.7,
            reason=f"Grid level {level.index} ({level.side}) at {level.price:.4f}",
            metadata={
                "grid_index": level.index,
                "grid_side": level.side,
                "grid_pnl": level.realized_pnl,
                "total_grid_pnl": self.grid_state.total_pnl,
                "filled_buys": self.grid_state.filled_buys,
                "filled_sells": self.grid_state.filled_sells,
            },
        )
        return signal

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def _detect_trend(self, candles: np.ndarray) -> int:
        """Detect trend direction using EMA crossover. Returns 1, -1, or 0."""
        close = candles[:, 4].astype(np.float64)
        if len(close) < 50:
            return 0
        ema_fast = self.indicators.ema(close, 20)
        ema_slow = self.indicators.ema(close, 50)
        last = len(close) - 1
        if np.isnan(ema_fast[last]) or np.isnan(ema_slow[last]):
            return 0
        diff = ema_fast[last] - ema_slow[last]
        threshold = close[last] * 0.001
        if diff > threshold:
            return 1
        elif diff < -threshold:
            return -1
        return 0

    def _should_rebalance(self, candles: np.ndarray) -> bool:
        """Check if grid needs rebalancing due to trend change."""
        if not self.rebalance_on_trend:
            return False
        now = time.time()
        if now - self.grid_state.last_rebalance_time < self.rebalance_cooldown:
            return False

        new_trend = self._detect_trend(candles)
        if new_trend != 0 and new_trend != self._trend_direction:
            self._trend_direction = new_trend
            return True
        return False

    def _rebalance_grid(self, candles: np.ndarray) -> None:
        """Rebuild the grid centered on current price."""
        close = candles[:, 4].astype(np.float64)
        current_price = close[-1]

        # Shift grid range in trend direction
        old_range = self.upper_price - self.lower_price
        if self._trend_direction == 1:
            # Uptrend: shift range up by 1/4
            shift = old_range * 0.25
            self.lower_price += shift
            self.upper_price += shift
        elif self._trend_direction == -1:
            shift = old_range * 0.25
            self.lower_price -= shift
            self.upper_price -= shift

        # Ensure current price is within range
        if current_price > self.upper_price:
            self.upper_price = current_price * 1.02
        if current_price < self.lower_price:
            self.lower_price = current_price * 0.98

        # Preserve total PnL and rebuild
        saved_pnl = self.grid_state.total_pnl
        self._initialize_grid(candles)
        self.grid_state.total_pnl = saved_pnl
        self.grid_state.last_rebalance_time = time.time()
        self._logger.info(
            "Grid rebalanced: new range %.4f - %.4f, trend=%d",
            self.lower_price, self.upper_price, self._trend_direction,
        )

    # ------------------------------------------------------------------
    # Dynamic grid count based on volatility
    # ------------------------------------------------------------------

    def _adjust_grid_count(self, candles: np.ndarray) -> None:
        """Adjust number of grids based on ATR volatility."""
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        close = candles[:, 4].astype(np.float64)
        atr = self.indicators.atr(high, low, close, 14)
        last_atr = atr[-1] if not np.isnan(atr[-1]) else 0.0
        current_price = close[-1]

        if current_price == 0 or last_atr == 0:
            return

        volatility_pct = (last_atr / current_price) * 100.0

        # Higher volatility -> fewer grids (wider spacing)
        # Lower volatility -> more grids (tighter spacing)
        if volatility_pct > 5.0:
            target_grids = self.min_grids
        elif volatility_pct > 3.0:
            target_grids = max(self.min_grids, self.num_grids // 2)
        elif volatility_pct < 1.0:
            target_grids = min(self.max_grids, self.num_grids * 2)
        else:
            target_grids = self.num_grids

        if target_grids != self.grid_state.grid_count:
            self._logger.info(
                "Adjusting grid count: %d -> %d (vol=%.2f%%)",
                self.grid_state.grid_count, target_grids, volatility_pct,
            )
            self.num_grids = target_grids

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        """
        Process candle data and return a grid trading signal.
        """
        if len(candles) < max(self.bb_period + 5, 50):
            return None

        if not self._grid_initialized:
            self._initialize_grid(candles)
            self._last_price = candles[-1, 4]
            return None

        current_price = float(candles[-1, 4])

        # Check if price is outside grid range
        if current_price > self.grid_state.upper_price * 1.05:
            self._logger.info("Price above grid range, rebalancing")
            self._rebalance_grid(candles)
            self._last_price = current_price
            return None
        if current_price < self.grid_state.lower_price * 0.95:
            self._logger.info("Price below grid range, rebalancing")
            self._rebalance_grid(candles)
            self._last_price = current_price
            return None

        # Check for trend change rebalance
        if self._should_rebalance(candles):
            self._rebalance_grid(candles)
            self._last_price = current_price
            return None

        # Find triggered levels
        if self._last_price == 0:
            self._last_price = current_price
            return None

        triggered = self._find_triggered_levels(self._last_price, current_price)
        self._last_price = current_price

        if not triggered:
            return None

        # Process the first triggered level (one signal per call)
        level = triggered[0]
        signal = self._fill_grid_level(level, current_price)
        if signal and self.check_risk(signal):
            return signal

        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        """Process individual tick for grid level crossing."""
        if not self._grid_initialized:
            return None

        if self._last_price == 0:
            self._last_price = price
            return None

        triggered = self._find_triggered_levels(self._last_price, price)
        self._last_price = price

        if not triggered:
            return None

        level = triggered[0]
        return self._fill_grid_level(level, price)

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        """Process a new bar close for grid evaluation."""
        close = ohlcv.get("close", 0.0)
        if close == 0 or not self._grid_initialized:
            return None

        prev = self._last_price if self._last_price > 0 else close
        triggered = self._find_triggered_levels(prev, close)
        self._last_price = close

        if triggered:
            return self._fill_grid_level(triggered[0], close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        """Update grid state when an order is filled externally."""
        filled_price = order_info.get("price", 0.0)
        side = order_info.get("side", "")

        # Match to nearest unfilled grid level
        best_level = None
        best_dist = float("inf")
        for level in self.grid_state.levels:
            if level.is_filled:
                continue
            if level.side != side.upper():
                continue
            dist = abs(level.price - filled_price)
            if dist < best_dist:
                best_dist = dist
                best_level = level

        if best_level is not None:
            best_level.is_filled = True
            best_level.fill_price = filled_price
            best_level.fill_time = time.time()
            self._logger.info(
                "External fill matched to grid level %d at %.4f",
                best_level.index, filled_price,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_grid_status(self) -> Dict[str, Any]:
        """Return detailed grid status."""
        filled = [l for l in self.grid_state.levels if l.is_filled]
        unfilled = [l for l in self.grid_state.levels if not l.is_filled]
        return {
            "initialized": self._grid_initialized,
            "range": [self.grid_state.lower_price, self.grid_state.upper_price],
            "grid_count": self.grid_state.grid_count,
            "grid_type": self.grid_state.grid_type,
            "filled_levels": len(filled),
            "unfilled_levels": len(unfilled),
            "filled_buys": self.grid_state.filled_buys,
            "filled_sells": self.grid_state.filled_sells,
            "total_pnl": self.grid_state.total_pnl,
            "investment": self.grid_state.total_investment,
            "roi_pct": (self.grid_state.total_pnl / self.grid_state.total_investment * 100)
            if self.grid_state.total_investment > 0 else 0.0,
            "trend": self._trend_direction,
            "levels": [
                {
                    "index": l.index,
                    "price": l.price,
                    "side": l.side,
                    "filled": l.is_filled,
                    "pnl": l.realized_pnl,
                }
                for l in self.grid_state.levels
            ],
        }
