"""
Mefai Signal Engine - Market Making Strategy

Symmetric bid/ask quoting around mid price, inventory-aware skew,
volatility-adjusted spread, order refresh rate, maximum inventory limit,
PnL tracking per side, and adverse selection detection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import math
import numpy as np
from collections import deque

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """A single bid or ask quote."""
    price: float
    quantity: float
    side: str          # "BID" or "ASK"
    placed_time: float
    order_id: str = ""
    filled: bool = False
    fill_price: float = 0.0
    fill_time: float = 0.0


@dataclass
class MMInventory:
    """Tracks the market maker's inventory position."""
    long_quantity: float = 0.0
    short_quantity: float = 0.0
    net_quantity: float = 0.0      # positive = long, negative = short
    avg_entry_long: float = 0.0
    avg_entry_short: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    bid_fills: int = 0
    ask_fills: int = 0
    total_volume: float = 0.0

    def update_net(self) -> None:
        self.net_quantity = self.long_quantity - self.short_quantity

    def update_unrealized(self, mid_price: float) -> None:
        if self.net_quantity > 0:
            self.unrealized_pnl = (mid_price - self.avg_entry_long) * self.net_quantity
        elif self.net_quantity < 0:
            self.unrealized_pnl = (self.avg_entry_short - mid_price) * abs(self.net_quantity)
        else:
            self.unrealized_pnl = 0.0


@dataclass
class AdverseSelectionMetrics:
    """Metrics for detecting adverse selection (being picked off)."""
    fill_count: int = 0
    adverse_count: int = 0  # fills followed by price moving against us
    avg_adverse_move_pct: float = 0.0
    adverse_ratio: float = 0.0  # adverse_count / fill_count

    def update(self, was_adverse: bool, move_pct: float) -> None:
        self.fill_count += 1
        if was_adverse:
            self.adverse_count += 1
            # Running average
            if self.adverse_count == 1:
                self.avg_adverse_move_pct = move_pct
            else:
                self.avg_adverse_move_pct = (
                    self.avg_adverse_move_pct * (self.adverse_count - 1) + move_pct
                ) / self.adverse_count
        if self.fill_count > 0:
            self.adverse_ratio = self.adverse_count / self.fill_count


class MarketMakingStrategy(BaseStrategy):
    """
    Market Making Strategy

    Provides liquidity by continuously quoting bid and ask prices.
    Key features:
    1. Symmetric quotes around mid price with configurable spread
    2. Inventory-aware skew (shift quotes to reduce net exposure)
    3. Volatility-adjusted spread (wider in high vol environments)
    4. Configurable order refresh rate
    5. Maximum inventory limit with forced unwinding
    6. PnL tracking per side (bid vs ask)
    7. Adverse selection detection (getting picked off)
    """

    name = "MarketMaking"
    description = "Market making with inventory-aware quoting and spread management"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Spread
        self.base_spread_bps: float = self.params.get("spread_bps", 10.0)
        self.min_spread_bps: float = self.params.get("min_spread_bps", 4.0)
        self.max_spread_bps: float = self.params.get("max_spread_bps", 50.0)

        # Inventory
        self.max_inventory: float = self.params.get("max_inventory", 100.0)
        self.inventory_skew_factor: float = self.params.get("inventory_skew_factor", 0.5)
        self.force_unwind_pct: float = self.params.get("force_unwind_pct", 90.0)

        # Order management
        self.refresh_interval_sec: float = self.params.get("refresh_interval_sec", 5.0)
        self.order_size: float = self.params.get("order_size", 1.0)
        self.num_levels: int = self.params.get("num_levels", 3)
        self.level_spacing_bps: float = self.params.get("level_spacing_bps", 5.0)

        # Volatility adjustment
        self.vol_lookback: int = self.params.get("vol_lookback", 20)
        self.vol_spread_multiplier: float = self.params.get("vol_spread_multiplier", 2.0)
        self.high_vol_threshold: float = self.params.get("high_vol_threshold", 0.03)

        # Adverse selection
        self.adverse_lookback_bars: int = self.params.get("adverse_lookback_bars", 3)
        self.adverse_threshold: float = self.params.get("adverse_threshold", 0.5)
        self.widen_on_adverse: bool = self.params.get("widen_on_adverse", True)
        self.adverse_widen_mult: float = self.params.get("adverse_widen_mult", 1.5)

        # State
        self.inventory = MMInventory()
        self.adverse_metrics = AdverseSelectionMetrics()
        self._active_bids: List[Quote] = []
        self._active_asks: List[Quote] = []
        self._last_refresh_time: float = 0.0
        self._recent_fills: deque = deque(maxlen=100)
        self._mid_price: float = 0.0
        self._volatility: float = 0.0

    # ------------------------------------------------------------------
    # Spread calculation
    # ------------------------------------------------------------------

    def _calculate_spread(self, close: np.ndarray) -> float:
        """
        Calculate the optimal spread in basis points.
        Adjusts for volatility, inventory, and adverse selection.
        """
        spread_bps = self.base_spread_bps

        # Volatility adjustment
        if len(close) >= self.vol_lookback + 1:
            returns = np.diff(np.log(close[-self.vol_lookback - 1:]))
            self._volatility = float(np.std(returns))
            if self._volatility > self.high_vol_threshold:
                vol_mult = 1.0 + (self._volatility / self.high_vol_threshold - 1.0) * self.vol_spread_multiplier
                spread_bps *= vol_mult

        # Adverse selection adjustment
        if self.widen_on_adverse and self.adverse_metrics.adverse_ratio > self.adverse_threshold:
            spread_bps *= self.adverse_widen_mult

        # Clamp
        spread_bps = max(self.min_spread_bps, min(spread_bps, self.max_spread_bps))
        return spread_bps

    # ------------------------------------------------------------------
    # Inventory skew
    # ------------------------------------------------------------------

    def _calculate_skew(self) -> float:
        """
        Calculate price skew based on inventory.
        Positive net inventory -> shift quotes down (lower bid, lower ask)
        to encourage selling and discourage buying.
        Returns skew in basis points.
        """
        if self.max_inventory == 0:
            return 0.0

        inventory_ratio = self.inventory.net_quantity / self.max_inventory
        skew_bps = inventory_ratio * self.inventory_skew_factor * self.base_spread_bps
        return skew_bps

    # ------------------------------------------------------------------
    # Quote generation
    # ------------------------------------------------------------------

    def _generate_quotes(
        self, mid_price: float, spread_bps: float,
    ) -> Tuple[List[Quote], List[Quote]]:
        """
        Generate bid and ask quotes at multiple levels.
        """
        skew_bps = self._calculate_skew()
        half_spread = spread_bps / 2.0

        bids = []
        asks = []
        now = time.time()

        for level in range(self.num_levels):
            level_offset = level * self.level_spacing_bps
            # Size decreases at deeper levels
            level_size = self.order_size * (0.8 ** level)

            # Bid price (buy) - skew shifts it
            bid_bps = half_spread + level_offset + skew_bps
            bid_price = mid_price * (1.0 - bid_bps / 10000.0)

            # Ask price (sell) - skew shifts it
            ask_bps = half_spread + level_offset - skew_bps
            ask_price = mid_price * (1.0 + ask_bps / 10000.0)

            # Ensure valid prices
            if bid_price > 0 and ask_price > bid_price:
                bids.append(Quote(
                    price=round(bid_price, 8),
                    quantity=level_size,
                    side="BID",
                    placed_time=now,
                ))
                asks.append(Quote(
                    price=round(ask_price, 8),
                    quantity=level_size,
                    side="ASK",
                    placed_time=now,
                ))

        return bids, asks

    # ------------------------------------------------------------------
    # Inventory management
    # ------------------------------------------------------------------

    def _should_force_unwind(self) -> bool:
        """Check if inventory exceeds the force unwind threshold."""
        if self.max_inventory == 0:
            return False
        usage_pct = (abs(self.inventory.net_quantity) / self.max_inventory) * 100.0
        return usage_pct >= self.force_unwind_pct

    def _create_unwind_signal(self, current_price: float) -> StrategySignal:
        """Create a market order to reduce inventory."""
        if self.inventory.net_quantity > 0:
            # Too long - sell
            qty = abs(self.inventory.net_quantity) * 0.5  # unwind half
            return StrategySignal(
                signal_type=SignalType.CLOSE_LONG,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                confidence=0.9,
                reason=f"Inventory unwind: net={self.inventory.net_quantity:.4f}, "
                       f"max={self.max_inventory}",
                metadata={
                    "unwind_reason": "max_inventory",
                    "net_inventory": self.inventory.net_quantity,
                },
            )
        else:
            qty = abs(self.inventory.net_quantity) * 0.5
            return StrategySignal(
                signal_type=SignalType.CLOSE_SHORT,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                confidence=0.9,
                reason=f"Inventory unwind: net={self.inventory.net_quantity:.4f}",
                metadata={
                    "unwind_reason": "max_inventory",
                    "net_inventory": self.inventory.net_quantity,
                },
            )

    # ------------------------------------------------------------------
    # Adverse selection detection
    # ------------------------------------------------------------------

    def _check_adverse_selection(
        self, fill_side: str, fill_price: float, current_price: float,
    ) -> None:
        """
        After a fill, check if the price moved against us (adverse selection).
        If we bought (bid filled) and price went down, that is adverse.
        If we sold (ask filled) and price went up, that is adverse.
        """
        if fill_price == 0:
            return

        if fill_side == "BID":
            move_pct = ((current_price - fill_price) / fill_price) * 100.0
            was_adverse = move_pct < 0
        else:
            move_pct = ((fill_price - current_price) / fill_price) * 100.0
            was_adverse = move_pct < 0

        self.adverse_metrics.update(was_adverse, abs(move_pct))

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        """
        Generate market making quotes. Returns the most urgent signal
        (inventory unwind takes priority over quote placement).
        """
        if len(candles) < self.vol_lookback + 5:
            return None

        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])
        self._mid_price = current_price

        # Update inventory unrealized PnL
        self.inventory.update_unrealized(current_price)

        # Force unwind if over limit
        if self._should_force_unwind():
            return self._create_unwind_signal(current_price)

        # Check if quotes need refresh
        now = time.time()
        if now - self._last_refresh_time < self.refresh_interval_sec:
            return None

        # Calculate spread
        spread_bps = self._calculate_spread(close)

        # Generate new quotes
        bids, asks = self._generate_quotes(current_price, spread_bps)
        self._active_bids = bids
        self._active_asks = asks
        self._last_refresh_time = now

        # Return the best bid/ask as a signal for order placement
        if bids and asks:
            best_bid = bids[0]
            best_ask = asks[0]

            # Determine which side to signal based on inventory
            if self.inventory.net_quantity >= 0:
                # Neutral or long - prioritize selling (post ask)
                return StrategySignal(
                    signal_type=SignalType.SHORT,
                    symbol=self.symbol,
                    price=best_ask.price,
                    quantity=best_ask.quantity,
                    confidence=0.6,
                    reason=f"MM quote: bid={best_bid.price:.4f} ask={best_ask.price:.4f} "
                           f"spread={spread_bps:.1f}bps",
                    metadata={
                        "bid_price": best_bid.price,
                        "ask_price": best_ask.price,
                        "spread_bps": spread_bps,
                        "skew_bps": self._calculate_skew(),
                        "inventory": self.inventory.net_quantity,
                        "volatility": self._volatility,
                        "adverse_ratio": self.adverse_metrics.adverse_ratio,
                        "num_levels": len(bids),
                        "all_bids": [{"price": b.price, "qty": b.quantity} for b in bids],
                        "all_asks": [{"price": a.price, "qty": a.quantity} for a in asks],
                    },
                )
            else:
                # Short - prioritize buying (post bid)
                return StrategySignal(
                    signal_type=SignalType.LONG,
                    symbol=self.symbol,
                    price=best_bid.price,
                    quantity=best_bid.quantity,
                    confidence=0.6,
                    reason=f"MM quote: bid={best_bid.price:.4f} ask={best_ask.price:.4f} "
                           f"spread={spread_bps:.1f}bps",
                    metadata={
                        "bid_price": best_bid.price,
                        "ask_price": best_ask.price,
                        "spread_bps": spread_bps,
                        "skew_bps": self._calculate_skew(),
                        "inventory": self.inventory.net_quantity,
                        "volatility": self._volatility,
                        "adverse_ratio": self.adverse_metrics.adverse_ratio,
                        "num_levels": len(bids),
                        "all_bids": [{"price": b.price, "qty": b.quantity} for b in bids],
                        "all_asks": [{"price": a.price, "qty": a.quantity} for a in asks],
                    },
                )

        return None

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        """Check fills against active quotes on each tick."""
        self._mid_price = price
        self.inventory.update_unrealized(price)

        # Check bid fills
        for bid in self._active_bids:
            if not bid.filled and price <= bid.price:
                bid.filled = True
                bid.fill_price = price
                bid.fill_time = timestamp
                self.inventory.long_quantity += bid.quantity
                self.inventory.bid_fills += 1
                self.inventory.total_volume += bid.quantity * price
                self.inventory.update_net()
                self._recent_fills.append(("BID", bid.fill_price, timestamp))

        # Check ask fills
        for ask in self._active_asks:
            if not ask.filled and price >= ask.price:
                ask.filled = True
                ask.fill_price = price
                ask.fill_time = timestamp
                self.inventory.short_quantity += ask.quantity
                self.inventory.ask_fills += 1
                self.inventory.total_volume += ask.quantity * price
                self.inventory.update_net()
                self._recent_fills.append(("ASK", ask.fill_price, timestamp))

        # Check adverse selection on recent fills
        for fill_side, fill_price, fill_time in self._recent_fills:
            if timestamp - fill_time > 0 and timestamp - fill_time < 60:
                self._check_adverse_selection(fill_side, fill_price, price)

        # Force unwind check
        if self._should_force_unwind():
            return self._create_unwind_signal(price)

        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        close = ohlcv.get("close", 0.0)
        if close > 0:
            self._mid_price = close
            self.inventory.update_unrealized(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        """Update inventory from external fill notification."""
        side = order_info.get("side", "").upper()
        price = order_info.get("price", 0.0)
        qty = order_info.get("quantity", 0.0)

        if side == "BUY":
            self.inventory.long_quantity += qty
            self.inventory.bid_fills += 1
        elif side == "SELL":
            self.inventory.short_quantity += qty
            self.inventory.ask_fills += 1

        self.inventory.total_volume += qty * price
        self.inventory.update_net()
        self._recent_fills.append((
            "BID" if side == "BUY" else "ASK",
            price, time.time(),
        ))

    def get_mm_status(self) -> Dict[str, Any]:
        return {
            "mid_price": self._mid_price,
            "inventory": {
                "net": self.inventory.net_quantity,
                "long": self.inventory.long_quantity,
                "short": self.inventory.short_quantity,
                "unrealized_pnl": self.inventory.unrealized_pnl,
                "realized_pnl": self.inventory.realized_pnl,
                "bid_fills": self.inventory.bid_fills,
                "ask_fills": self.inventory.ask_fills,
                "total_volume": self.inventory.total_volume,
            },
            "spread_bps": self.base_spread_bps,
            "skew_bps": self._calculate_skew(),
            "volatility": self._volatility,
            "adverse_selection": {
                "ratio": self.adverse_metrics.adverse_ratio,
                "avg_move_pct": self.adverse_metrics.avg_adverse_move_pct,
                "total_fills": self.adverse_metrics.fill_count,
            },
            "active_bids": len([b for b in self._active_bids if not b.filled]),
            "active_asks": len([a for a in self._active_asks if not a.filled]),
        }
