"""
Mefai Signal Engine - Dollar Cost Averaging Strategy

Intelligent DCA with time-based, price-based, RSI-weighted, and
volatility-adjusted buying. Includes take-profit levels, maximum
investment cap, and schedule building.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import time
import math
import numpy as np

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class DCAOrder:
    """Record of a single DCA purchase."""
    price: float
    quantity: float
    amount: float
    timestamp: float
    trigger: str       # "time", "dip", "rsi", "volatility"
    rsi_at_buy: float = 0.0
    volatility: float = 0.0


@dataclass
class DCAState:
    """Full DCA portfolio state."""
    orders: List[DCAOrder] = field(default_factory=list)
    total_invested: float = 0.0
    total_quantity: float = 0.0
    average_cost: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_buy_time: float = 0.0
    last_buy_price: float = 0.0
    buy_count: int = 0
    partial_sells: int = 0
    highest_price_since_buy: float = 0.0

    def add_order(self, order: DCAOrder) -> None:
        self.orders.append(order)
        self.total_invested += order.amount
        self.total_quantity += order.quantity
        if self.total_quantity > 0:
            self.average_cost = self.total_invested / self.total_quantity
        self.last_buy_time = order.timestamp
        self.last_buy_price = order.price
        self.buy_count += 1

    def update_value(self, current_price: float) -> None:
        self.current_value = self.total_quantity * current_price
        self.unrealized_pnl = self.current_value - self.total_invested
        if current_price > self.highest_price_since_buy:
            self.highest_price_since_buy = current_price


class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging Strategy

    Systematic buying with multiple intelligence layers:
    - Time-based: buy at fixed intervals
    - Price-based: buy dips at configured percentage drops
    - RSI-weighted: increase buy size when RSI is oversold
    - Volatility-adjusted: reduce buy size in high volatility
    - Take-profit: partial exits at configured targets
    - Investment cap: stop buying when max investment reached
    """

    name = "DCA"
    description = "Intelligent Dollar Cost Averaging"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Time-based DCA
        self.interval_hours: float = self.params.get("interval_hours", 24.0)
        self.base_buy_amount: float = self.params.get("buy_amount", 100.0)

        # Price-based DCA
        self.dip_threshold_pct: float = self.params.get("dip_threshold_pct", 5.0)
        self.dip_multiplier: float = self.params.get("dip_multiplier", 1.5)
        self.crash_threshold_pct: float = self.params.get("crash_threshold_pct", 15.0)
        self.crash_multiplier: float = self.params.get("crash_multiplier", 3.0)

        # RSI-weighted DCA
        self.rsi_enabled: bool = self.params.get("rsi_enabled", True)
        self.rsi_period: int = self.params.get("rsi_period", 14)
        self.rsi_oversold: float = self.params.get("rsi_oversold", 30.0)
        self.rsi_very_oversold: float = self.params.get("rsi_very_oversold", 20.0)
        self.rsi_weight_factor: float = self.params.get("rsi_weight_factor", 2.0)

        # Volatility adjustment
        self.vol_adjust_enabled: bool = self.params.get("vol_adjust_enabled", True)
        self.vol_lookback: int = self.params.get("vol_lookback", 20)
        self.vol_high_threshold: float = self.params.get("vol_high_threshold", 0.04)
        self.vol_reduction: float = self.params.get("vol_reduction", 0.5)

        # Take profit
        self.take_profit_levels: List[Dict[str, float]] = self.params.get(
            "take_profit_levels",
            [
                {"pct": 10.0, "sell_pct": 25.0},
                {"pct": 25.0, "sell_pct": 25.0},
                {"pct": 50.0, "sell_pct": 25.0},
                {"pct": 100.0, "sell_pct": 25.0},
            ],
        )
        self._tp_triggered: Dict[int, bool] = {}

        # Limits
        self.max_investment: float = self.params.get("max_investment", 50000.0)
        self.max_single_buy: float = self.params.get("max_single_buy", 1000.0)
        self.min_buy_amount: float = self.params.get("min_buy_amount", 10.0)

        # State
        self.dca_state = DCAState()
        self._reference_price: float = 0.0
        self._last_rsi: float = 50.0
        self._current_volatility: float = 0.0

    # ------------------------------------------------------------------
    # Buy amount calculation
    # ------------------------------------------------------------------

    def _calculate_buy_amount(
        self, current_price: float, rsi: float, volatility: float,
    ) -> float:
        """
        Calculate the DCA buy amount considering all intelligence layers.
        """
        amount = self.base_buy_amount

        # Price-based dip multiplier
        if self._reference_price > 0:
            drop_pct = ((self._reference_price - current_price) / self._reference_price) * 100.0
            if drop_pct >= self.crash_threshold_pct:
                amount *= self.crash_multiplier
                self._logger.info("Crash buy: price dropped %.1f%%, multiplier=%.1f",
                                  drop_pct, self.crash_multiplier)
            elif drop_pct >= self.dip_threshold_pct:
                amount *= self.dip_multiplier
                self._logger.info("Dip buy: price dropped %.1f%%, multiplier=%.1f",
                                  drop_pct, self.dip_multiplier)

        # RSI weight
        if self.rsi_enabled and rsi > 0:
            if rsi <= self.rsi_very_oversold:
                amount *= self.rsi_weight_factor * 1.5
            elif rsi <= self.rsi_oversold:
                amount *= self.rsi_weight_factor
            elif rsi > 70:
                # Overbought - reduce buy amount
                amount *= 0.5

        # Volatility adjustment
        if self.vol_adjust_enabled and volatility > self.vol_high_threshold:
            amount *= self.vol_reduction

        # Enforce limits
        amount = max(self.min_buy_amount, min(amount, self.max_single_buy))

        # Check investment cap
        remaining = self.max_investment - self.dca_state.total_invested
        if remaining <= 0:
            return 0.0
        amount = min(amount, remaining)

        return round(amount, 2)

    def _calculate_volatility(self, close: np.ndarray) -> float:
        """Calculate realized volatility (standard deviation of returns)."""
        if len(close) < self.vol_lookback + 1:
            return 0.0
        returns = np.diff(np.log(close[-self.vol_lookback - 1:]))
        return float(np.std(returns))

    # ------------------------------------------------------------------
    # Time-based check
    # ------------------------------------------------------------------

    def _time_buy_due(self) -> bool:
        """Check if enough time has passed for the next scheduled buy."""
        if self.dca_state.last_buy_time == 0:
            return True
        elapsed_hours = (time.time() - self.dca_state.last_buy_time) / 3600.0
        return elapsed_hours >= self.interval_hours

    # ------------------------------------------------------------------
    # Price-based check
    # ------------------------------------------------------------------

    def _dip_buy_triggered(self, current_price: float) -> bool:
        """Check if price has dipped enough from reference for an extra buy."""
        if self._reference_price <= 0:
            return False
        drop_pct = ((self._reference_price - current_price) / self._reference_price) * 100.0
        return drop_pct >= self.dip_threshold_pct

    # ------------------------------------------------------------------
    # Take profit check
    # ------------------------------------------------------------------

    def _check_take_profit(
        self, current_price: float,
    ) -> Optional[StrategySignal]:
        """Check if any take-profit levels are hit and return sell signal."""
        if self.dca_state.total_quantity <= 0 or self.dca_state.average_cost <= 0:
            return None

        profit_pct = ((current_price - self.dca_state.average_cost)
                      / self.dca_state.average_cost) * 100.0

        for idx, level in enumerate(self.take_profit_levels):
            if idx in self._tp_triggered and self._tp_triggered[idx]:
                continue
            target_pct = level["pct"]
            sell_pct = level["sell_pct"]

            if profit_pct >= target_pct:
                sell_qty = self.dca_state.total_quantity * (sell_pct / 100.0)
                if sell_qty <= 0:
                    continue

                self._tp_triggered[idx] = True
                pnl = (current_price - self.dca_state.average_cost) * sell_qty
                self.dca_state.realized_pnl += pnl
                self.dca_state.total_quantity -= sell_qty
                self.dca_state.partial_sells += 1

                self._logger.info(
                    "Take profit level %d hit: profit=%.1f%%, selling %.6f (%.1f%%)",
                    idx, profit_pct, sell_qty, sell_pct,
                )

                return StrategySignal(
                    signal_type=SignalType.CLOSE_LONG,
                    symbol=self.symbol,
                    price=current_price,
                    quantity=sell_qty,
                    confidence=0.85,
                    reason=f"DCA take profit level {idx}: {profit_pct:.1f}% profit",
                    metadata={
                        "tp_level": idx,
                        "profit_pct": profit_pct,
                        "sell_pct": sell_pct,
                        "realized_pnl": pnl,
                        "remaining_qty": self.dca_state.total_quantity,
                        "average_cost": self.dca_state.average_cost,
                    },
                )
        return None

    # ------------------------------------------------------------------
    # Schedule builder
    # ------------------------------------------------------------------

    def build_schedule(
        self,
        start_time: float,
        num_buys: int,
    ) -> List[Dict[str, Any]]:
        """
        Build a DCA schedule for planned future buys.
        Returns list of {time, estimated_amount} entries.
        """
        schedule = []
        interval_seconds = self.interval_hours * 3600
        for i in range(num_buys):
            buy_time = start_time + i * interval_seconds
            schedule.append({
                "buy_number": i + 1,
                "scheduled_time": buy_time,
                "estimated_amount": self.base_buy_amount,
                "cumulative_invested": self.base_buy_amount * (i + 1),
            })
        return schedule

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        """
        Generate DCA buy or take-profit sell signal.
        """
        if len(candles) < max(self.rsi_period + 2, self.vol_lookback + 2):
            return None

        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])

        # Update reference price (rolling high)
        if self._reference_price == 0:
            self._reference_price = current_price
        elif current_price > self._reference_price:
            self._reference_price = current_price

        # Update state
        self.dca_state.update_value(current_price)

        # Calculate indicators
        rsi_vals = self.indicators.rsi(close, self.rsi_period)
        rsi = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else 50.0
        self._last_rsi = rsi
        self._current_volatility = self._calculate_volatility(close)

        # Check take profit first
        tp_signal = self._check_take_profit(current_price)
        if tp_signal is not None:
            return tp_signal

        # Determine if buy is triggered
        trigger = None
        if self._time_buy_due():
            trigger = "time"
        elif self._dip_buy_triggered(current_price):
            trigger = "dip"
        elif self.rsi_enabled and rsi <= self.rsi_oversold:
            # RSI-triggered extra buy (independent of schedule)
            if self.dca_state.last_buy_time > 0:
                hours_since = (time.time() - self.dca_state.last_buy_time) / 3600.0
                if hours_since >= self.interval_hours * 0.25:  # at least 25% of interval
                    trigger = "rsi"

        if trigger is None:
            return None

        # Calculate buy amount
        buy_amount = self._calculate_buy_amount(current_price, rsi, self._current_volatility)
        if buy_amount <= 0:
            self._logger.info("Max investment reached, no more buys")
            return None

        quantity = buy_amount / current_price

        # Record the order
        order = DCAOrder(
            price=current_price,
            quantity=quantity,
            amount=buy_amount,
            timestamp=time.time(),
            trigger=trigger,
            rsi_at_buy=rsi,
            volatility=self._current_volatility,
        )
        self.dca_state.add_order(order)

        # Reset reference after dip buy
        if trigger == "dip":
            self._reference_price = current_price

        signal = StrategySignal(
            signal_type=SignalType.LONG,
            symbol=self.symbol,
            price=current_price,
            quantity=quantity,
            confidence=0.6 + (0.3 if rsi < self.rsi_oversold else 0.0),
            reason=f"DCA {trigger} buy: ${buy_amount:.2f} at {current_price:.4f}",
            metadata={
                "trigger": trigger,
                "buy_amount": buy_amount,
                "rsi": rsi,
                "volatility": self._current_volatility,
                "total_invested": self.dca_state.total_invested,
                "average_cost": self.dca_state.average_cost,
                "total_quantity": self.dca_state.total_quantity,
                "buy_count": self.dca_state.buy_count,
            },
        )

        if self.check_risk(signal):
            return signal
        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        """Check take-profit levels on tick."""
        self.dca_state.update_value(price)
        return self._check_take_profit(price)

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        """Process new bar - delegated to generate_signal with buffered candles."""
        close = ohlcv.get("close", 0.0)
        if close > 0:
            self.dca_state.update_value(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        """Update state when an external fill is received."""
        price = order_info.get("price", 0.0)
        qty = order_info.get("quantity", 0.0)
        side = order_info.get("side", "BUY")

        if side.upper() == "BUY" and price > 0 and qty > 0:
            order = DCAOrder(
                price=price,
                quantity=qty,
                amount=price * qty,
                timestamp=time.time(),
                trigger="external",
            )
            self.dca_state.add_order(order)
        elif side.upper() == "SELL" and price > 0 and qty > 0:
            pnl = (price - self.dca_state.average_cost) * qty
            self.dca_state.realized_pnl += pnl
            self.dca_state.total_quantity -= qty

    def get_dca_status(self) -> Dict[str, Any]:
        return {
            "total_invested": self.dca_state.total_invested,
            "total_quantity": self.dca_state.total_quantity,
            "average_cost": self.dca_state.average_cost,
            "current_value": self.dca_state.current_value,
            "unrealized_pnl": self.dca_state.unrealized_pnl,
            "realized_pnl": self.dca_state.realized_pnl,
            "buy_count": self.dca_state.buy_count,
            "partial_sells": self.dca_state.partial_sells,
            "roi_pct": (self.dca_state.unrealized_pnl / self.dca_state.total_invested * 100)
            if self.dca_state.total_invested > 0 else 0.0,
            "last_rsi": self._last_rsi,
            "volatility": self._current_volatility,
            "max_investment_remaining": self.max_investment - self.dca_state.total_invested,
        }
