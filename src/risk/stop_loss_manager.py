"""
Mefai Autotrade - Advanced Stop Loss Manager

Production-grade stop loss management:
- Fixed stop loss (% or absolute)
- ATR-based dynamic stop
- Chandelier Exit
- Trailing stop (fixed %, ATR-based, Parabolic SAR)
- Break-even stop
- Partial take profit
- Time-based stop
- Volatility-adjusted stop
- Support/resistance-based stop
- Guaranteed stop support
"""

import logging
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class StopType(str, Enum):
    FIXED_PCT = "fixed_pct"
    FIXED_PRICE = "fixed_price"
    ATR_BASED = "atr_based"
    CHANDELIER = "chandelier"
    TRAILING_PCT = "trailing_pct"
    TRAILING_ATR = "trailing_atr"
    PARABOLIC_SAR = "parabolic_sar"
    BREAK_EVEN = "break_even"
    TIME_BASED = "time_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    SUPPORT_RESISTANCE = "support_resistance"
    GUARANTEED = "guaranteed"


class StopAction(str, Enum):
    NONE = "none"
    UPDATE_STOP = "update_stop"
    CLOSE_FULL = "close_full"
    CLOSE_PARTIAL = "close_partial"
    MOVE_TO_BREAKEVEN = "move_to_breakeven"


@dataclass
class StopConfig:
    """Stop loss configuration."""
    # Fixed stop
    fixed_stop_pct: float = 2.0  # percent from entry
    fixed_stop_price: float = 0.0  # absolute price

    # ATR-based
    atr_multiplier: float = 2.0  # stop at entry - (ATR * multiplier)
    atr_period: int = 14

    # Chandelier Exit
    chandelier_atr_multiplier: float = 3.0
    chandelier_lookback: int = 22

    # Trailing stop
    trailing_pct: float = 1.5  # trail by this percent from highest
    trailing_activation_pct: float = 0.5  # activate trailing after this % profit
    trailing_atr_multiplier: float = 2.0

    # Parabolic SAR
    sar_acceleration: float = 0.02
    sar_max_acceleration: float = 0.2

    # Break-even
    breakeven_trigger_pct: float = 1.0  # move SL to entry after 1% profit
    breakeven_offset_pct: float = 0.1  # offset above entry for commissions

    # Partial take profit
    partial_tp_levels: List[Dict] = field(default_factory=lambda: [
        {"pct": 1.5, "close_pct": 33.0},  # close 33% at 1.5% profit
        {"pct": 3.0, "close_pct": 33.0},  # close 33% at 3.0% profit
        {"pct": 5.0, "close_pct": 34.0},  # close remaining at 5.0% profit
    ])

    # Time-based
    max_hold_hours: float = 48.0  # close if no profit after 48h
    time_stop_profit_threshold: float = 0.0  # close if PnL below this after time

    # Volatility-adjusted
    vol_high_atr_multiplier: float = 3.0  # wider stop in high vol
    vol_low_atr_multiplier: float = 1.5  # tighter stop in low vol
    vol_threshold_multiplier: float = 1.5  # vol > 1.5x average = "high"

    # Support/resistance
    sr_buffer_pct: float = 0.3  # place stop 0.3% below support

    # Guaranteed stop
    guaranteed_stop_enabled: bool = False
    guaranteed_stop_premium_pct: float = 0.5  # extra cost for guaranteed fill


@dataclass
class StopLevel:
    """Current stop loss level for a position."""
    symbol: str
    side: str
    stop_type: StopType
    stop_price: float
    entry_price: float
    current_price: float
    highest_price: float  # since entry (for longs)
    lowest_price: float  # since entry (for shorts)
    is_active: bool = True
    is_trailing: bool = False
    breakeven_triggered: bool = False
    partial_levels_hit: List[int] = field(default_factory=list)
    entry_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None


@dataclass
class StopUpdate:
    """Result of a stop loss check/update."""
    action: StopAction
    new_stop_price: float = 0.0
    close_quantity_pct: float = 0.0  # percentage of position to close
    reason: str = ""
    stop_type: StopType = StopType.FIXED_PCT
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stop Loss Manager
# ---------------------------------------------------------------------------

class StopLossManager:
    """
    Manages stop loss levels for all open positions.

    Handles multiple stop types, trailing logic, break-even moves,
    partial take profits, and time-based exits.
    """

    def __init__(self, config: Optional[StopConfig] = None):
        self.config = config or StopConfig()
        self._stops: Dict[str, StopLevel] = {}  # symbol -> StopLevel
        self._sar_state: Dict[str, Dict] = {}  # symbol -> SAR calculation state

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def register_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_type: StopType = StopType.FIXED_PCT,
        custom_stop_price: Optional[float] = None,
        atr: Optional[float] = None,
        highest_high: Optional[float] = None,
        nearest_support: Optional[float] = None,
        nearest_resistance: Optional[float] = None,
    ) -> StopLevel:
        """
        Register a new position and calculate initial stop loss.

        Args:
            symbol: Trading pair
            side: "LONG" or "SHORT"
            entry_price: Entry price
            stop_type: Type of stop loss to use
            custom_stop_price: Override calculated stop price
            atr: Current ATR value (required for ATR-based stops)
            highest_high: Highest high in lookback (for Chandelier)
            nearest_support: Nearest support level
            nearest_resistance: Nearest resistance level

        Returns:
            StopLevel with calculated stop price
        """
        now = datetime.now(timezone.utc)

        if custom_stop_price and custom_stop_price > 0:
            stop_price = custom_stop_price
        else:
            stop_price = self._calculate_initial_stop(
                side=side,
                entry_price=entry_price,
                stop_type=stop_type,
                atr=atr,
                highest_high=highest_high,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
            )

        level = StopLevel(
            symbol=symbol,
            side=side.upper(),
            stop_type=stop_type,
            stop_price=stop_price,
            entry_price=entry_price,
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            is_active=True,
            is_trailing=(stop_type in (
                StopType.TRAILING_PCT,
                StopType.TRAILING_ATR,
                StopType.PARABOLIC_SAR,
                StopType.CHANDELIER,
            )),
            entry_time=now,
            last_updated=now,
        )

        self._stops[symbol] = level

        # Initialize Parabolic SAR if needed
        if stop_type == StopType.PARABOLIC_SAR:
            self._init_sar(symbol, side, entry_price)

        logger.info(
            f"Stop registered: {symbol} {side} entry={entry_price:.4f} "
            f"stop={stop_price:.4f} type={stop_type.value}"
        )

        return level

    def remove_position(self, symbol: str) -> None:
        """Remove stop tracking for a closed position."""
        self._stops.pop(symbol, None)
        self._sar_state.pop(symbol, None)

    # ------------------------------------------------------------------
    # Stop calculation
    # ------------------------------------------------------------------

    def _calculate_initial_stop(
        self,
        side: str,
        entry_price: float,
        stop_type: StopType,
        atr: Optional[float],
        highest_high: Optional[float],
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
    ) -> float:
        """Calculate initial stop loss price based on stop type."""
        if stop_type == StopType.FIXED_PCT:
            return self._stop_fixed_pct(side, entry_price)

        elif stop_type == StopType.FIXED_PRICE:
            return self.config.fixed_stop_price

        elif stop_type in (StopType.ATR_BASED, StopType.TRAILING_ATR):
            if atr is None or atr <= 0:
                logger.warning("ATR not provided - falling back to fixed %")
                return self._stop_fixed_pct(side, entry_price)
            return self._stop_atr(side, entry_price, atr, self.config.atr_multiplier)

        elif stop_type == StopType.CHANDELIER:
            if atr is None or atr <= 0:
                return self._stop_fixed_pct(side, entry_price)
            hh = highest_high or entry_price
            return self._stop_chandelier(side, hh, atr)

        elif stop_type == StopType.TRAILING_PCT:
            return self._stop_trailing_pct(side, entry_price)

        elif stop_type == StopType.PARABOLIC_SAR:
            return self._stop_fixed_pct(side, entry_price)  # SAR will take over

        elif stop_type == StopType.VOLATILITY_ADJUSTED:
            if atr is None or atr <= 0:
                return self._stop_fixed_pct(side, entry_price)
            return self._stop_volatility_adjusted(side, entry_price, atr)

        elif stop_type == StopType.SUPPORT_RESISTANCE:
            return self._stop_support_resistance(
                side, entry_price, nearest_support, nearest_resistance
            )

        else:
            return self._stop_fixed_pct(side, entry_price)

    def _stop_fixed_pct(self, side: str, price: float) -> float:
        """Calculate fixed percentage stop."""
        distance = price * (self.config.fixed_stop_pct / 100)
        if side.upper() == "LONG":
            return price - distance
        else:
            return price + distance

    def _stop_atr(
        self,
        side: str,
        price: float,
        atr: float,
        multiplier: float,
    ) -> float:
        """Calculate ATR-based stop."""
        distance = atr * multiplier
        if side.upper() == "LONG":
            return price - distance
        else:
            return price + distance

    def _stop_chandelier(
        self,
        side: str,
        highest_high: float,
        atr: float,
    ) -> float:
        """
        Calculate Chandelier Exit stop.

        Long: Highest High - (ATR * multiplier)
        Short: Lowest Low + (ATR * multiplier)
        """
        distance = atr * self.config.chandelier_atr_multiplier
        if side.upper() == "LONG":
            return highest_high - distance
        else:
            return highest_high + distance  # For shorts, uses lowest low

    def _stop_trailing_pct(self, side: str, price: float) -> float:
        """Calculate trailing percentage stop from current price."""
        distance = price * (self.config.trailing_pct / 100)
        if side.upper() == "LONG":
            return price - distance
        else:
            return price + distance

    def _stop_volatility_adjusted(
        self,
        side: str,
        price: float,
        atr: float,
    ) -> float:
        """
        Calculate volatility-adjusted stop.

        Wider stop in high volatility, tighter in low volatility.
        """
        # For now, use the standard ATR multiplier
        # The vol adjustment happens in update() based on current conditions
        multiplier = self.config.atr_multiplier
        return self._stop_atr(side, price, atr, multiplier)

    def _stop_support_resistance(
        self,
        side: str,
        entry_price: float,
        support: Optional[float],
        resistance: Optional[float],
    ) -> float:
        """
        Calculate stop based on nearest support/resistance level.

        Long: place stop below nearest support with buffer
        Short: place stop above nearest resistance with buffer
        """
        buffer_pct = self.config.sr_buffer_pct / 100

        if side.upper() == "LONG":
            if support and support > 0:
                return support * (1 - buffer_pct)
            else:
                return self._stop_fixed_pct(side, entry_price)
        else:
            if resistance and resistance > 0:
                return resistance * (1 + buffer_pct)
            else:
                return self._stop_fixed_pct(side, entry_price)

    # ------------------------------------------------------------------
    # Parabolic SAR
    # ------------------------------------------------------------------

    def _init_sar(self, symbol: str, side: str, entry_price: float) -> None:
        """Initialize Parabolic SAR state."""
        self._sar_state[symbol] = {
            "af": self.config.sar_acceleration,
            "ep": entry_price,  # extreme point
            "sar": entry_price,
            "is_long": side.upper() == "LONG",
        }

    def _update_sar(
        self,
        symbol: str,
        high: float,
        low: float,
    ) -> float:
        """
        Update Parabolic SAR with new price data.

        Returns the new SAR value (stop price).
        """
        state = self._sar_state.get(symbol)
        if state is None:
            return 0.0

        af = state["af"]
        ep = state["ep"]
        sar = state["sar"]
        is_long = state["is_long"]

        if is_long:
            # Update extreme point and acceleration factor
            if high > ep:
                ep = high
                af = min(af + self.config.sar_acceleration, self.config.sar_max_acceleration)

            # Calculate new SAR
            new_sar = sar + af * (ep - sar)

            # SAR cannot be above the previous two lows
            new_sar = min(new_sar, low)

            # Check for reversal
            if low < new_sar:
                is_long = False
                new_sar = ep
                ep = low
                af = self.config.sar_acceleration
        else:
            if low < ep:
                ep = low
                af = min(af + self.config.sar_acceleration, self.config.sar_max_acceleration)

            new_sar = sar + af * (ep - sar)
            new_sar = max(new_sar, high)

            if high > new_sar:
                is_long = True
                new_sar = ep
                ep = high
                af = self.config.sar_acceleration

        # Update state
        state["af"] = af
        state["ep"] = ep
        state["sar"] = new_sar
        state["is_long"] = is_long

        return new_sar

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        current_price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
        normal_atr: Optional[float] = None,
        highest_high: Optional[float] = None,
    ) -> StopUpdate:
        """
        Update stop loss for a position based on current price.

        This should be called on every price update or new candle.

        Args:
            symbol: Trading pair
            current_price: Current market price
            high: Candle high (for SAR)
            low: Candle low (for SAR)
            atr: Current ATR value
            normal_atr: Normal/average ATR (for volatility adjustment)
            highest_high: Highest high in lookback period

        Returns:
            StopUpdate with any required action
        """
        level = self._stops.get(symbol)
        if level is None:
            return StopUpdate(action=StopAction.NONE, reason="No stop registered")

        if not level.is_active:
            return StopUpdate(action=StopAction.NONE, reason="Stop not active")

        now = datetime.now(timezone.utc)
        level.current_price = current_price
        level.last_updated = now

        # Update price extremes
        if current_price > level.highest_price:
            level.highest_price = current_price
        if current_price < level.lowest_price:
            level.lowest_price = current_price

        # Check if stop is hit
        stop_hit = self._is_stop_hit(level, current_price)
        if stop_hit:
            return StopUpdate(
                action=StopAction.CLOSE_FULL,
                new_stop_price=level.stop_price,
                close_quantity_pct=100.0,
                reason=f"Stop loss hit at {level.stop_price:.4f}",
                stop_type=level.stop_type,
            )

        # Check partial take profit
        partial = self._check_partial_tp(level)
        if partial:
            return partial

        # Check time-based stop
        time_stop = self._check_time_stop(level, current_price, now)
        if time_stop:
            return time_stop

        # Check break-even trigger
        be_update = self._check_breakeven(level, current_price)
        if be_update:
            return be_update

        # Update trailing stop
        trail_update = self._update_trailing(
            level, current_price, high, low, atr, normal_atr, highest_high
        )
        if trail_update:
            return trail_update

        return StopUpdate(action=StopAction.NONE)

    def _is_stop_hit(self, level: StopLevel, price: float) -> bool:
        """Check if the stop loss price has been hit."""
        if level.side == "LONG":
            return price <= level.stop_price
        else:
            return price >= level.stop_price

    def _check_partial_tp(self, level: StopLevel) -> Optional[StopUpdate]:
        """Check if any partial take profit levels have been hit."""
        if not self.config.partial_tp_levels:
            return None

        entry = level.entry_price
        price = level.current_price

        if level.side == "LONG":
            pnl_pct = (price - entry) / entry * 100
        else:
            pnl_pct = (entry - price) / entry * 100

        for i, tp_level in enumerate(self.config.partial_tp_levels):
            if i in level.partial_levels_hit:
                continue

            if pnl_pct >= tp_level["pct"]:
                level.partial_levels_hit.append(i)
                return StopUpdate(
                    action=StopAction.CLOSE_PARTIAL,
                    close_quantity_pct=tp_level["close_pct"],
                    reason=(
                        f"Partial TP level {i + 1}: {tp_level['pct']}% profit reached, "
                        f"closing {tp_level['close_pct']}%"
                    ),
                    stop_type=level.stop_type,
                    details={"tp_level_index": i, "pnl_pct": pnl_pct},
                )

        return None

    def _check_time_stop(
        self,
        level: StopLevel,
        current_price: float,
        now: datetime,
    ) -> Optional[StopUpdate]:
        """Check time-based stop."""
        if self.config.max_hold_hours <= 0:
            return None
        if level.entry_time is None:
            return None

        elapsed = (now - level.entry_time).total_seconds() / 3600
        if elapsed < self.config.max_hold_hours:
            return None

        # Check if position is profitable enough to keep
        if level.side == "LONG":
            pnl_pct = (current_price - level.entry_price) / level.entry_price * 100
        else:
            pnl_pct = (level.entry_price - current_price) / level.entry_price * 100

        if pnl_pct <= self.config.time_stop_profit_threshold:
            return StopUpdate(
                action=StopAction.CLOSE_FULL,
                close_quantity_pct=100.0,
                reason=(
                    f"Time stop: held for {elapsed:.1f}h (max {self.config.max_hold_hours:.1f}h), "
                    f"PnL={pnl_pct:.2f}%"
                ),
                stop_type=StopType.TIME_BASED,
                details={"hold_hours": elapsed, "pnl_pct": pnl_pct},
            )

        return None

    def _check_breakeven(
        self,
        level: StopLevel,
        current_price: float,
    ) -> Optional[StopUpdate]:
        """Check and apply break-even stop."""
        if level.breakeven_triggered:
            return None
        if self.config.breakeven_trigger_pct <= 0:
            return None

        if level.side == "LONG":
            pnl_pct = (current_price - level.entry_price) / level.entry_price * 100
        else:
            pnl_pct = (level.entry_price - current_price) / level.entry_price * 100

        if pnl_pct >= self.config.breakeven_trigger_pct:
            # Move stop to entry + offset
            offset = level.entry_price * (self.config.breakeven_offset_pct / 100)

            if level.side == "LONG":
                new_stop = level.entry_price + offset
            else:
                new_stop = level.entry_price - offset

            # Only move stop up for longs, down for shorts
            if level.side == "LONG" and new_stop > level.stop_price:
                level.stop_price = new_stop
                level.breakeven_triggered = True
                return StopUpdate(
                    action=StopAction.MOVE_TO_BREAKEVEN,
                    new_stop_price=new_stop,
                    reason=f"Break-even triggered at {pnl_pct:.2f}% profit",
                    stop_type=StopType.BREAK_EVEN,
                )
            elif level.side == "SHORT" and new_stop < level.stop_price:
                level.stop_price = new_stop
                level.breakeven_triggered = True
                return StopUpdate(
                    action=StopAction.MOVE_TO_BREAKEVEN,
                    new_stop_price=new_stop,
                    reason=f"Break-even triggered at {pnl_pct:.2f}% profit",
                    stop_type=StopType.BREAK_EVEN,
                )

        return None

    def _update_trailing(
        self,
        level: StopLevel,
        current_price: float,
        high: Optional[float],
        low: Optional[float],
        atr: Optional[float],
        normal_atr: Optional[float],
        highest_high: Optional[float],
    ) -> Optional[StopUpdate]:
        """Update trailing stop based on stop type."""
        if not level.is_trailing:
            return None

        # Check activation threshold for trailing
        if level.side == "LONG":
            pnl_pct = (current_price - level.entry_price) / level.entry_price * 100
        else:
            pnl_pct = (level.entry_price - current_price) / level.entry_price * 100

        if pnl_pct < self.config.trailing_activation_pct:
            return None  # Not yet profitable enough to trail

        new_stop = level.stop_price

        if level.stop_type == StopType.TRAILING_PCT:
            new_stop = self._trail_pct(level)

        elif level.stop_type == StopType.TRAILING_ATR:
            if atr and atr > 0:
                new_stop = self._trail_atr(level, atr)

        elif level.stop_type == StopType.CHANDELIER:
            if atr and atr > 0:
                hh = highest_high or level.highest_price
                new_stop = self._trail_chandelier(level, hh, atr)

        elif level.stop_type == StopType.PARABOLIC_SAR:
            h = high or current_price
            l = low or current_price
            new_stop = self._update_sar(level.symbol, h, l)

        elif level.stop_type == StopType.VOLATILITY_ADJUSTED:
            if atr and atr > 0 and normal_atr and normal_atr > 0:
                new_stop = self._trail_volatility_adjusted(level, atr, normal_atr)

        # Only move stop in favorable direction
        if level.side == "LONG" and new_stop > level.stop_price:
            old_stop = level.stop_price
            level.stop_price = new_stop
            return StopUpdate(
                action=StopAction.UPDATE_STOP,
                new_stop_price=new_stop,
                reason=(
                    f"Trailing stop moved: {old_stop:.4f} -> {new_stop:.4f}"
                ),
                stop_type=level.stop_type,
            )
        elif level.side == "SHORT" and new_stop < level.stop_price:
            old_stop = level.stop_price
            level.stop_price = new_stop
            return StopUpdate(
                action=StopAction.UPDATE_STOP,
                new_stop_price=new_stop,
                reason=(
                    f"Trailing stop moved: {old_stop:.4f} -> {new_stop:.4f}"
                ),
                stop_type=level.stop_type,
            )

        return None

    def _trail_pct(self, level: StopLevel) -> float:
        """Calculate trailing stop based on percentage."""
        if level.side == "LONG":
            ref_price = level.highest_price
            return ref_price * (1 - self.config.trailing_pct / 100)
        else:
            ref_price = level.lowest_price
            return ref_price * (1 + self.config.trailing_pct / 100)

    def _trail_atr(self, level: StopLevel, atr: float) -> float:
        """Calculate ATR-based trailing stop."""
        distance = atr * self.config.trailing_atr_multiplier
        if level.side == "LONG":
            return level.highest_price - distance
        else:
            return level.lowest_price + distance

    def _trail_chandelier(
        self,
        level: StopLevel,
        highest_high: float,
        atr: float,
    ) -> float:
        """Calculate Chandelier Exit trailing stop."""
        distance = atr * self.config.chandelier_atr_multiplier
        if level.side == "LONG":
            return highest_high - distance
        else:
            return level.lowest_price + distance

    def _trail_volatility_adjusted(
        self,
        level: StopLevel,
        atr: float,
        normal_atr: float,
    ) -> float:
        """
        Calculate volatility-adjusted trailing stop.

        Wider stop in high volatility, tighter in low volatility.
        """
        vol_ratio = atr / normal_atr if normal_atr > 0 else 1.0

        if vol_ratio > self.config.vol_threshold_multiplier:
            multiplier = self.config.vol_high_atr_multiplier
        else:
            multiplier = self.config.vol_low_atr_multiplier

        distance = atr * multiplier

        if level.side == "LONG":
            return level.highest_price - distance
        else:
            return level.lowest_price + distance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stop(self, symbol: str) -> Optional[StopLevel]:
        """Get current stop level for a symbol."""
        return self._stops.get(symbol)

    def get_all_stops(self) -> Dict[str, StopLevel]:
        """Get all active stop levels."""
        return dict(self._stops)

    def get_stop_distance(self, symbol: str) -> Optional[Dict]:
        """Get distance between current price and stop."""
        level = self._stops.get(symbol)
        if level is None:
            return None

        if level.side == "LONG":
            distance = level.current_price - level.stop_price
            distance_pct = (distance / level.current_price) * 100
        else:
            distance = level.stop_price - level.current_price
            distance_pct = (distance / level.current_price) * 100

        return {
            "symbol": symbol,
            "stop_price": level.stop_price,
            "current_price": level.current_price,
            "distance": distance,
            "distance_pct": distance_pct,
            "stop_type": level.stop_type.value,
            "is_trailing": level.is_trailing,
            "breakeven_triggered": level.breakeven_triggered,
        }

    def update_stop_price(self, symbol: str, new_price: float) -> bool:
        """Manually update stop price for a symbol."""
        level = self._stops.get(symbol)
        if level is None:
            return False

        old_price = level.stop_price
        level.stop_price = new_price
        level.last_updated = datetime.now(timezone.utc)

        logger.info(
            f"Stop manually updated: {symbol} {old_price:.4f} -> {new_price:.4f}"
        )
        return True

    def deactivate_stop(self, symbol: str) -> bool:
        """Deactivate stop for a symbol (keeps tracking but won't trigger)."""
        level = self._stops.get(symbol)
        if level is None:
            return False

        level.is_active = False
        return True

    def activate_stop(self, symbol: str) -> bool:
        """Reactivate a deactivated stop."""
        level = self._stops.get(symbol)
        if level is None:
            return False

        level.is_active = True
        return True

    def get_summary(self) -> List[Dict]:
        """Get summary of all stop levels."""
        summaries = []
        for symbol, level in self._stops.items():
            dist = self.get_stop_distance(symbol)
            summaries.append({
                "symbol": symbol,
                "side": level.side,
                "entry_price": level.entry_price,
                "current_price": level.current_price,
                "stop_price": level.stop_price,
                "stop_type": level.stop_type.value,
                "is_active": level.is_active,
                "is_trailing": level.is_trailing,
                "breakeven_triggered": level.breakeven_triggered,
                "distance_pct": dist["distance_pct"] if dist else 0,
                "partial_levels_hit": len(level.partial_levels_hit),
            })

        return summaries

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            "stops": {
                sym: {
                    "symbol": lvl.symbol,
                    "side": lvl.side,
                    "stop_type": lvl.stop_type.value,
                    "stop_price": lvl.stop_price,
                    "entry_price": lvl.entry_price,
                    "current_price": lvl.current_price,
                    "highest_price": lvl.highest_price,
                    "lowest_price": lvl.lowest_price,
                    "is_active": lvl.is_active,
                    "is_trailing": lvl.is_trailing,
                    "breakeven_triggered": lvl.breakeven_triggered,
                    "partial_levels_hit": lvl.partial_levels_hit,
                    "entry_time": lvl.entry_time.isoformat() if lvl.entry_time else None,
                }
                for sym, lvl in self._stops.items()
            },
            "config": {
                "fixed_stop_pct": self.config.fixed_stop_pct,
                "atr_multiplier": self.config.atr_multiplier,
                "trailing_pct": self.config.trailing_pct,
                "breakeven_trigger_pct": self.config.breakeven_trigger_pct,
                "max_hold_hours": self.config.max_hold_hours,
            },
        }
