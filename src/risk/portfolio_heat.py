"""
Mefai Autotrade - Portfolio Heat Map and Risk Allocation

Production-grade portfolio heat management:
- Total portfolio heat (sum of all position risks)
- Heat limit enforcement
- Per-strategy heat allocation
- Heat-based position rejection
- Visual heat map generation (data structure)
- Historical heat tracking
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class HeatLevel(str, Enum):
    COLD = "cold"  # 0-25% of limit
    WARM = "warm"  # 25-50%
    HOT = "hot"  # 50-75%
    CRITICAL = "critical"  # 75-100%
    OVERHEATED = "overheated"  # >100%


@dataclass
class HeatConfig:
    """Portfolio heat configuration."""
    # Total heat limit (percentage of equity at risk)
    max_heat_pct: float = 6.0

    # Per-strategy heat limits
    strategy_heat_limits: Dict[str, float] = field(default_factory=lambda: {
        "default": 6.0,
    })

    # Heat level thresholds (percentage of max heat)
    warm_threshold_pct: float = 25.0
    hot_threshold_pct: float = 50.0
    critical_threshold_pct: float = 75.0

    # Rejection policy
    reject_on_overheat: bool = True
    reduce_on_hot: bool = True
    hot_reduction_factor: float = 0.5  # reduce new positions by 50% when hot


@dataclass
class PositionHeat:
    """Heat contribution of a single position."""
    symbol: str
    side: str
    strategy: str
    risk_amount: float  # dollar amount at risk (to stop loss)
    risk_pct: float  # percentage of equity at risk
    entry_price: float
    stop_price: float
    quantity: float
    notional: float


@dataclass
class HeatSnapshot:
    """Point-in-time portfolio heat state."""
    timestamp: datetime
    equity: float
    total_heat_pct: float
    total_heat_amount: float
    max_heat_pct: float
    heat_level: HeatLevel
    heat_utilization_pct: float  # percentage of max heat used
    positions: List[PositionHeat]
    strategy_heat: Dict[str, float]
    is_accepting_new_trades: bool
    size_reduction_factor: float
    warnings: List[str]


@dataclass
class HeatHistoryEntry:
    """Historical heat record."""
    timestamp: datetime
    total_heat_pct: float
    heat_level: HeatLevel
    position_count: int
    equity: float


# ---------------------------------------------------------------------------
# Portfolio Heat Manager
# ---------------------------------------------------------------------------

class PortfolioHeat:
    """
    Manages portfolio heat - the total percentage of equity at risk
    across all open positions.

    Heat = sum of (risk per position) / equity * 100

    For example, if you have 3 positions each risking 1.5% of equity,
    total heat = 4.5%. If the heat limit is 6%, you have room for
    one more position at 1.5% risk.
    """

    def __init__(self, config: Optional[HeatConfig] = None):
        self.config = config or HeatConfig()
        self._positions: Dict[str, PositionHeat] = {}
        self._equity: float = 0.0
        self._history: List[HeatHistoryEntry] = []
        self._max_history = 50000

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self._equity = equity
        # Recalculate all risk percentages
        for pos in self._positions.values():
            if equity > 0:
                pos.risk_pct = (pos.risk_amount / equity) * 100

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_price: float,
        quantity: float,
        strategy: str = "default",
    ) -> None:
        """
        Add a position to heat tracking.

        Args:
            symbol: Trading pair
            side: "LONG" or "SHORT"
            entry_price: Entry price
            stop_price: Stop loss price
            quantity: Position quantity
            strategy: Strategy name for per-strategy heat tracking
        """
        risk_per_unit = abs(entry_price - stop_price)
        risk_amount = risk_per_unit * quantity
        risk_pct = (risk_amount / self._equity * 100) if self._equity > 0 else 0
        notional = quantity * entry_price

        self._positions[symbol] = PositionHeat(
            symbol=symbol,
            side=side.upper(),
            strategy=strategy,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            quantity=quantity,
            notional=notional,
        )

        logger.debug(
            f"Heat added: {symbol} risk={risk_amount:.2f} ({risk_pct:.2f}%) "
            f"strategy={strategy}"
        )

    def update_position(
        self,
        symbol: str,
        stop_price: Optional[float] = None,
        quantity: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> None:
        """Update a position's heat contribution."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        if stop_price is not None:
            pos.stop_price = stop_price

        if quantity is not None:
            pos.quantity = quantity

        # Recalculate risk
        price = current_price or pos.entry_price
        risk_per_unit = abs(price - pos.stop_price)
        pos.risk_amount = risk_per_unit * pos.quantity
        pos.risk_pct = (pos.risk_amount / self._equity * 100) if self._equity > 0 else 0
        pos.notional = pos.quantity * price

    def remove_position(self, symbol: str) -> None:
        """Remove a position from heat tracking."""
        removed = self._positions.pop(symbol, None)
        if removed:
            logger.debug(f"Heat removed: {symbol} was {removed.risk_pct:.2f}%")

    # ------------------------------------------------------------------
    # Heat calculations
    # ------------------------------------------------------------------

    def get_total_heat(self) -> float:
        """Get total portfolio heat as percentage of equity."""
        return sum(pos.risk_pct for pos in self._positions.values())

    def get_total_heat_amount(self) -> float:
        """Get total dollar amount at risk."""
        return sum(pos.risk_amount for pos in self._positions.values())

    def get_strategy_heat(self) -> Dict[str, float]:
        """Get heat breakdown by strategy."""
        strategy_heat: Dict[str, float] = {}
        for pos in self._positions.values():
            strategy_heat[pos.strategy] = (
                strategy_heat.get(pos.strategy, 0.0) + pos.risk_pct
            )
        return strategy_heat

    def get_heat_level(self) -> HeatLevel:
        """Get current heat level classification."""
        total = self.get_total_heat()
        max_heat = self.config.max_heat_pct

        if max_heat <= 0:
            return HeatLevel.COLD

        utilization = (total / max_heat) * 100

        if utilization > 100:
            return HeatLevel.OVERHEATED
        elif utilization >= self.config.critical_threshold_pct:
            return HeatLevel.CRITICAL
        elif utilization >= self.config.hot_threshold_pct:
            return HeatLevel.HOT
        elif utilization >= self.config.warm_threshold_pct:
            return HeatLevel.WARM
        else:
            return HeatLevel.COLD

    def get_remaining_heat(self) -> float:
        """Get remaining heat budget in percentage points."""
        return max(0, self.config.max_heat_pct - self.get_total_heat())

    def get_heat_utilization(self) -> float:
        """Get heat utilization as percentage of maximum."""
        if self.config.max_heat_pct <= 0:
            return 0.0
        return (self.get_total_heat() / self.config.max_heat_pct) * 100

    # ------------------------------------------------------------------
    # Trade approval
    # ------------------------------------------------------------------

    def check_new_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        quantity: float,
        strategy: str = "default",
    ) -> Dict:
        """
        Check if a new trade can be accepted based on heat limits.

        Returns:
            Dict with keys:
            - approved: bool
            - reason: str
            - adjusted_quantity: float (may be reduced)
            - heat_after: float (projected heat after trade)
            - size_multiplier: float
        """
        risk_per_unit = abs(entry_price - stop_price)
        proposed_risk = risk_per_unit * quantity
        proposed_risk_pct = (proposed_risk / self._equity * 100) if self._equity > 0 else 0

        current_heat = self.get_total_heat()
        projected_heat = current_heat + proposed_risk_pct
        heat_level = self.get_heat_level()

        result = {
            "approved": True,
            "reason": "",
            "adjusted_quantity": quantity,
            "heat_after": projected_heat,
            "size_multiplier": 1.0,
            "current_heat": current_heat,
            "proposed_risk_pct": proposed_risk_pct,
        }

        # Check total heat limit
        if projected_heat > self.config.max_heat_pct:
            if self.config.reject_on_overheat:
                remaining = self.get_remaining_heat()
                if remaining <= 0:
                    result["approved"] = False
                    result["reason"] = (
                        f"Portfolio overheated: current heat={current_heat:.2f}%, "
                        f"limit={self.config.max_heat_pct:.2f}%"
                    )
                    result["adjusted_quantity"] = 0
                    return result

                # Reduce quantity to fit within heat limit
                max_risk = self._equity * (remaining / 100)
                max_quantity = max_risk / risk_per_unit if risk_per_unit > 0 else 0
                result["adjusted_quantity"] = max_quantity
                result["size_multiplier"] = max_quantity / quantity if quantity > 0 else 0
                result["reason"] = (
                    f"Quantity reduced to fit heat limit: "
                    f"{quantity:.6f} -> {max_quantity:.6f}"
                )
                result["heat_after"] = self.config.max_heat_pct
                return result

        # Check strategy heat limit
        strategy_limit = self.config.strategy_heat_limits.get(
            strategy,
            self.config.strategy_heat_limits.get("default", self.config.max_heat_pct),
        )
        strategy_heat = self.get_strategy_heat()
        current_strategy_heat = strategy_heat.get(strategy, 0.0)

        if current_strategy_heat + proposed_risk_pct > strategy_limit:
            remaining = max(0, strategy_limit - current_strategy_heat)
            if remaining <= 0:
                result["approved"] = False
                result["reason"] = (
                    f"Strategy '{strategy}' heat limit reached: "
                    f"{current_strategy_heat:.2f}%/{strategy_limit:.2f}%"
                )
                result["adjusted_quantity"] = 0
                return result

            max_risk = self._equity * (remaining / 100)
            max_quantity = max_risk / risk_per_unit if risk_per_unit > 0 else 0
            result["adjusted_quantity"] = min(result["adjusted_quantity"], max_quantity)
            result["size_multiplier"] = result["adjusted_quantity"] / quantity if quantity > 0 else 0
            result["reason"] = f"Reduced for strategy heat limit"

        # Apply hot reduction
        if self.config.reduce_on_hot and heat_level in (HeatLevel.HOT, HeatLevel.CRITICAL):
            factor = self.config.hot_reduction_factor
            result["adjusted_quantity"] *= factor
            result["size_multiplier"] *= factor
            if not result["reason"]:
                result["reason"] = f"Reduced by {(1 - factor) * 100:.0f}% due to {heat_level.value} heat"

        return result

    # ------------------------------------------------------------------
    # Heat map (for frontend visualization)
    # ------------------------------------------------------------------

    def get_heat_map(self) -> Dict:
        """
        Generate heat map data structure for frontend visualization.

        Returns a structured dict suitable for rendering a heat map:
        - Overall heat gauge
        - Per-position heat bars
        - Per-strategy heat allocation
        """
        total_heat = self.get_total_heat()
        heat_level = self.get_heat_level()

        positions_data = []
        for pos in self._positions.values():
            heat_contribution = (
                (pos.risk_pct / total_heat * 100) if total_heat > 0 else 0
            )
            positions_data.append({
                "symbol": pos.symbol,
                "side": pos.side,
                "strategy": pos.strategy,
                "risk_pct": round(pos.risk_pct, 3),
                "risk_amount": round(pos.risk_amount, 2),
                "notional": round(pos.notional, 2),
                "heat_contribution_pct": round(heat_contribution, 1),
                "entry_price": pos.entry_price,
                "stop_price": pos.stop_price,
            })

        # Sort by risk contribution descending
        positions_data.sort(key=lambda x: x["risk_pct"], reverse=True)

        strategy_data = {}
        for strategy, heat in self.get_strategy_heat().items():
            limit = self.config.strategy_heat_limits.get(
                strategy,
                self.config.strategy_heat_limits.get("default", self.config.max_heat_pct),
            )
            strategy_data[strategy] = {
                "heat_pct": round(heat, 3),
                "limit_pct": limit,
                "utilization_pct": round(heat / limit * 100, 1) if limit > 0 else 0,
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": round(self._equity, 2),
            "total_heat_pct": round(total_heat, 3),
            "max_heat_pct": self.config.max_heat_pct,
            "heat_utilization_pct": round(self.get_heat_utilization(), 1),
            "heat_level": heat_level.value,
            "remaining_heat_pct": round(self.get_remaining_heat(), 3),
            "position_count": len(self._positions),
            "positions": positions_data,
            "strategies": strategy_data,
            "is_accepting_trades": total_heat < self.config.max_heat_pct,
        }

    # ------------------------------------------------------------------
    # Snapshot and history
    # ------------------------------------------------------------------

    def get_snapshot(self) -> HeatSnapshot:
        """Get current heat snapshot."""
        total_heat = self.get_total_heat()
        heat_level = self.get_heat_level()
        utilization = self.get_heat_utilization()

        warnings = []
        if heat_level == HeatLevel.OVERHEATED:
            warnings.append(
                f"OVERHEATED: total heat {total_heat:.2f}% exceeds "
                f"limit {self.config.max_heat_pct:.2f}%"
            )
        elif heat_level == HeatLevel.CRITICAL:
            warnings.append(
                f"Critical heat level: {total_heat:.2f}% "
                f"({utilization:.0f}% of limit)"
            )
        elif heat_level == HeatLevel.HOT:
            warnings.append(
                f"Hot: {total_heat:.2f}% ({utilization:.0f}% of limit)"
            )

        # Check per-strategy limits
        for strategy, heat in self.get_strategy_heat().items():
            limit = self.config.strategy_heat_limits.get(
                strategy,
                self.config.strategy_heat_limits.get("default", self.config.max_heat_pct),
            )
            if heat > limit * 0.9:
                warnings.append(
                    f"Strategy '{strategy}' near limit: {heat:.2f}%/{limit:.2f}%"
                )

        # Size reduction factor
        size_factor = 1.0
        if heat_level in (HeatLevel.HOT, HeatLevel.CRITICAL) and self.config.reduce_on_hot:
            size_factor = self.config.hot_reduction_factor
        if heat_level == HeatLevel.OVERHEATED:
            size_factor = 0.0

        snapshot = HeatSnapshot(
            timestamp=datetime.now(timezone.utc),
            equity=self._equity,
            total_heat_pct=total_heat,
            total_heat_amount=self.get_total_heat_amount(),
            max_heat_pct=self.config.max_heat_pct,
            heat_level=heat_level,
            heat_utilization_pct=utilization,
            positions=list(self._positions.values()),
            strategy_heat=self.get_strategy_heat(),
            is_accepting_new_trades=total_heat < self.config.max_heat_pct,
            size_reduction_factor=size_factor,
            warnings=warnings,
        )

        # Record history
        self._history.append(HeatHistoryEntry(
            timestamp=snapshot.timestamp,
            total_heat_pct=total_heat,
            heat_level=heat_level,
            position_count=len(self._positions),
            equity=self._equity,
        ))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snapshot

    def get_history(self, last_n: int = 1000) -> List[HeatHistoryEntry]:
        """Get recent heat history."""
        return self._history[-last_n:]

    def get_peak_heat(self) -> float:
        """Get the highest heat level recorded."""
        if not self._history:
            return self.get_total_heat()
        return max(h.total_heat_pct for h in self._history)

    def get_average_heat(self, last_n: int = 100) -> float:
        """Get average heat over the last N snapshots."""
        recent = self._history[-last_n:]
        if not recent:
            return 0.0
        return sum(h.total_heat_pct for h in recent) / len(recent)

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            "equity": self._equity,
            "positions": {
                sym: {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "strategy": pos.strategy,
                    "risk_amount": pos.risk_amount,
                    "risk_pct": pos.risk_pct,
                    "entry_price": pos.entry_price,
                    "stop_price": pos.stop_price,
                    "quantity": pos.quantity,
                }
                for sym, pos in self._positions.items()
            },
            "config": {
                "max_heat_pct": self.config.max_heat_pct,
                "strategy_heat_limits": self.config.strategy_heat_limits,
                "reject_on_overheat": self.config.reject_on_overheat,
            },
            "stats": {
                "total_heat_pct": self.get_total_heat(),
                "heat_level": self.get_heat_level().value,
                "position_count": len(self._positions),
                "peak_heat": self.get_peak_heat(),
            },
        }
