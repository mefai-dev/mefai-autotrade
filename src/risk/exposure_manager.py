"""
Mefai Autotrade - Portfolio Exposure Manager

Production-grade exposure monitoring and limits:
- Gross and net exposure limits
- Per-symbol and per-sector exposure
- Directional bias limits
- Correlation-adjusted exposure
- Leverage and margin monitoring
- Concentration limits
- Cross-margin vs isolated margin handling
"""

import logging
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class MarginMode(str, Enum):
    CROSS = "cross"
    ISOLATED = "isolated"


class ExposureCheckResult(str, Enum):
    APPROVED = "approved"
    REDUCED = "reduced"
    REJECTED = "rejected"


@dataclass
class ExposureConfig:
    """Exposure management configuration."""
    # Portfolio-level limits (as percentage of equity)
    max_gross_exposure_pct: float = 300.0  # 3x leverage gross
    max_net_exposure_pct: float = 100.0  # max directional tilt
    max_long_exposure_pct: float = 200.0
    max_short_exposure_pct: float = 200.0

    # Per-symbol limits
    max_symbol_exposure_pct: float = 20.0  # no single symbol > 20% of equity
    max_symbol_count: int = 20

    # Sector limits
    max_sector_exposure_pct: float = 40.0

    # Concentration
    max_concentration_pct: float = 25.0  # no single position > 25% of total exposure

    # Leverage
    max_account_leverage: float = 10.0  # total account leverage
    leverage_warning_pct: float = 70.0  # warn at 70% of max leverage

    # Margin
    margin_utilization_warning_pct: float = 70.0
    margin_utilization_limit_pct: float = 85.0

    # Correlation
    correlation_penalty_threshold: float = 0.7  # correlated above this threshold
    correlation_penalty_multiplier: float = 1.5  # count exposure as 1.5x when correlated

    # Margin mode defaults
    default_margin_mode: MarginMode = MarginMode.CROSS


@dataclass
class ExposureSnapshot:
    """Current exposure state."""
    timestamp: datetime
    equity: float
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    gross_exposure_pct: float
    net_exposure_pct: float
    long_exposure_pct: float
    short_exposure_pct: float
    account_leverage: float
    margin_used: float
    margin_available: float
    margin_utilization_pct: float
    symbol_exposures: Dict[str, float]
    sector_exposures: Dict[str, float]
    concentration_max_pct: float
    warnings: List[str]


@dataclass
class ExposureCheckResponse:
    """Result of an exposure check for a proposed trade."""
    result: ExposureCheckResult
    original_quantity: float
    adjusted_quantity: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_allowed_quantity: float = 0.0
    exposure_after_trade: Optional[ExposureSnapshot] = None


@dataclass
class SymbolExposure:
    """Exposure details for a single symbol."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    notional: float
    pct_of_equity: float
    pct_of_gross_exposure: float
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: float
    sector: str = "crypto"


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------

# Default sector classification for common crypto symbols
DEFAULT_SECTOR_MAP: Dict[str, str] = {
    # Layer 1
    "BTCUSDT": "layer1", "ETHUSDT": "layer1", "SOLUSDT": "layer1",
    "AVAXUSDT": "layer1", "ADAUSDT": "layer1", "DOTUSDT": "layer1",
    "ATOMUSDT": "layer1", "NEARUSDT": "layer1", "APTUSDT": "layer1",
    "SUIUSDT": "layer1", "SEIUSDT": "layer1", "TIAUSDT": "layer1",
    "INJUSDT": "layer1", "TONUSDT": "layer1",
    # Layer 2
    "MATICUSDT": "layer2", "ARBUSDT": "layer2", "OPUSDT": "layer2",
    "STRKUSDT": "layer2", "MANTAUSDT": "layer2", "ZKUSDT": "layer2",
    "SCROLLUSDT": "layer2",
    # DeFi
    "UNIUSDT": "defi", "AAVEUSDT": "defi", "MKRUSDT": "defi",
    "COMPUSDT": "defi", "CRVUSDT": "defi", "SUSHIUSDT": "defi",
    "SNXUSDT": "defi", "DYDXUSDT": "defi", "LDOUSDT": "defi",
    "PENDLEUSDT": "defi", "JUPUSDT": "defi",
    # Meme
    "DOGEUSDT": "meme", "SHIBUSDT": "meme", "1000PEPEUSDT": "meme",
    "1000FLOKIUSDT": "meme", "1000BONKUSDT": "meme", "WIFUSDT": "meme",
    "MEMEUSDT": "meme", "1000SHIBUSDT": "meme",
    # AI
    "FETUSDT": "ai", "RENDERUSDT": "ai", "AGIXUSDT": "ai",
    "TAOUSDT": "ai", "WLDUSDT": "ai", "ARKMUSDT": "ai",
    # Gaming
    "AXSUSDT": "gaming", "SANDUSDT": "gaming", "MANAUSDT": "gaming",
    "GALAUSDT": "gaming", "ILVUSDT": "gaming", "IMXUSDT": "gaming",
    # BNB ecosystem
    "BNBUSDT": "bnb_eco", "CAKEUSDT": "bnb_eco", "XVSUSDT": "bnb_eco",
    "BAKEUSDT": "bnb_eco",
}


# ---------------------------------------------------------------------------
# Exposure Manager
# ---------------------------------------------------------------------------

class ExposureManager:
    """
    Manages portfolio exposure limits and monitoring.

    Tracks all open positions, calculates gross/net exposure,
    enforces per-symbol and sector limits, and monitors leverage
    and margin utilization.
    """

    def __init__(
        self,
        config: Optional[ExposureConfig] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ):
        self.config = config or ExposureConfig()
        self._sector_map = sector_map or dict(DEFAULT_SECTOR_MAP)
        self._positions: Dict[str, SymbolExposure] = {}
        self._equity: float = 0.0
        self._margin_used: float = 0.0
        self._margin_available: float = 0.0
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}
        self._snapshot_history: List[ExposureSnapshot] = []
        self._max_history = 10000

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        self._equity = equity

    def update_margin(self, used: float, available: float) -> None:
        """Update margin information."""
        self._margin_used = used
        self._margin_available = available

    def update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        mark_price: float,
        leverage: int = 1,
        margin_mode: MarginMode = MarginMode.CROSS,
        unrealized_pnl: float = 0.0,
    ) -> None:
        """Update or add a position."""
        if quantity <= 0:
            # Position closed
            self._positions.pop(symbol, None)
            return

        notional = quantity * mark_price
        pct_of_equity = (notional / self._equity * 100) if self._equity > 0 else 0
        sector = self._sector_map.get(symbol, "other")

        gross = self._calculate_gross_exposure()
        pct_of_gross = (notional / gross * 100) if gross > 0 else 0

        self._positions[symbol] = SymbolExposure(
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            entry_price=entry_price,
            mark_price=mark_price,
            notional=notional,
            pct_of_equity=pct_of_equity,
            pct_of_gross_exposure=pct_of_gross,
            leverage=leverage,
            margin_mode=margin_mode,
            unrealized_pnl=unrealized_pnl,
            sector=sector,
        )

    def remove_position(self, symbol: str) -> None:
        """Remove a closed position."""
        self._positions.pop(symbol, None)

    def update_correlation_matrix(
        self,
        matrix: Dict[Tuple[str, str], float],
    ) -> None:
        """Update the correlation matrix for correlation-adjusted exposure."""
        self._correlation_matrix = matrix

    def set_sector(self, symbol: str, sector: str) -> None:
        """Set or update the sector classification for a symbol."""
        self._sector_map[symbol] = sector

    # ------------------------------------------------------------------
    # Exposure calculations
    # ------------------------------------------------------------------

    def _calculate_gross_exposure(self) -> float:
        """Total absolute notional across all positions."""
        return sum(pos.notional for pos in self._positions.values())

    def _calculate_net_exposure(self) -> float:
        """Net directional exposure (longs - shorts)."""
        long_exp = sum(
            pos.notional for pos in self._positions.values()
            if pos.side == "LONG"
        )
        short_exp = sum(
            pos.notional for pos in self._positions.values()
            if pos.side == "SHORT"
        )
        return long_exp - short_exp

    def _calculate_long_exposure(self) -> float:
        """Total long notional."""
        return sum(
            pos.notional for pos in self._positions.values()
            if pos.side == "LONG"
        )

    def _calculate_short_exposure(self) -> float:
        """Total short notional."""
        return sum(
            pos.notional for pos in self._positions.values()
            if pos.side == "SHORT"
        )

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate notional exposure per sector."""
        sectors: Dict[str, float] = {}
        for pos in self._positions.values():
            sector = self._sector_map.get(pos.symbol, "other")
            sectors[sector] = sectors.get(sector, 0.0) + pos.notional
        return sectors

    def _calculate_symbol_exposures(self) -> Dict[str, float]:
        """Calculate notional exposure per symbol."""
        return {
            symbol: pos.notional
            for symbol, pos in self._positions.items()
        }

    def _calculate_account_leverage(self) -> float:
        """Calculate total account leverage (gross exposure / equity)."""
        if self._equity <= 0:
            return 0.0
        gross = self._calculate_gross_exposure()
        return gross / self._equity

    def _calculate_margin_utilization(self) -> float:
        """Calculate margin utilization percentage."""
        total_margin = self._margin_used + self._margin_available
        if total_margin <= 0:
            return 0.0
        return (self._margin_used / total_margin) * 100

    def _calculate_concentration(self) -> float:
        """Get maximum single-position concentration as % of gross exposure."""
        gross = self._calculate_gross_exposure()
        if gross <= 0:
            return 0.0
        max_notional = max(
            (pos.notional for pos in self._positions.values()),
            default=0.0,
        )
        return (max_notional / gross) * 100

    def _calculate_correlation_adjusted_exposure(self) -> float:
        """
        Calculate correlation-adjusted gross exposure.

        Correlated positions (above threshold) count for more exposure
        since they don't provide true diversification.
        """
        if not self._correlation_matrix or len(self._positions) <= 1:
            return self._calculate_gross_exposure()

        symbols = list(self._positions.keys())
        adjusted_exposure = 0.0
        counted_pairs: Set[Tuple[str, str]] = set()

        for symbol in symbols:
            pos = self._positions[symbol]
            base_notional = pos.notional

            # Check correlation with other positions
            penalty_factor = 1.0
            for other_symbol in symbols:
                if other_symbol == symbol:
                    continue

                pair = tuple(sorted([symbol, other_symbol]))
                if pair in counted_pairs:
                    continue

                corr = self._correlation_matrix.get(
                    (symbol, other_symbol),
                    self._correlation_matrix.get((other_symbol, symbol), 0.0),
                )

                if abs(corr) >= self.config.correlation_penalty_threshold:
                    # Same direction correlation increases effective exposure
                    other_pos = self._positions[other_symbol]
                    same_direction = pos.side == other_pos.side

                    if (corr > 0 and same_direction) or (corr < 0 and not same_direction):
                        penalty_factor = max(
                            penalty_factor,
                            self.config.correlation_penalty_multiplier,
                        )

                    counted_pairs.add(pair)

            adjusted_exposure += base_notional * penalty_factor

        return adjusted_exposure

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def check_new_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        leverage: int = 1,
    ) -> ExposureCheckResponse:
        """
        Check if a proposed trade passes all exposure limits.

        May reduce the quantity if it would exceed limits, or reject
        the trade entirely if no valid quantity is possible.

        Args:
            symbol: Trading pair
            side: "LONG" or "SHORT"
            quantity: Proposed quantity
            entry_price: Expected entry price
            leverage: Leverage for this trade

        Returns:
            ExposureCheckResponse with result and any adjustments
        """
        response = ExposureCheckResponse(
            result=ExposureCheckResult.APPROVED,
            original_quantity=quantity,
            adjusted_quantity=quantity,
        )

        if self._equity <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.reasons.append("Equity is zero or negative")
            response.adjusted_quantity = 0.0
            return response

        proposed_notional = quantity * entry_price
        current_qty = quantity

        # 1. Symbol count limit
        if symbol not in self._positions and len(self._positions) >= self.config.max_symbol_count:
            response.result = ExposureCheckResult.REJECTED
            response.reasons.append(
                f"Max symbol count reached: {len(self._positions)}/{self.config.max_symbol_count}"
            )
            response.adjusted_quantity = 0.0
            return response

        # 2. Per-symbol exposure limit
        current_qty = self._check_symbol_limit(
            symbol, current_qty, entry_price, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 3. Sector exposure limit
        current_qty = self._check_sector_limit(
            symbol, side, current_qty, entry_price, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 4. Directional (side) limit
        current_qty = self._check_directional_limit(
            side, current_qty, entry_price, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 5. Net exposure limit
        current_qty = self._check_net_exposure_limit(
            side, current_qty, entry_price, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 6. Gross exposure limit
        current_qty = self._check_gross_exposure_limit(
            current_qty, entry_price, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 7. Account leverage limit
        current_qty = self._check_leverage_limit(
            current_qty, entry_price, leverage, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 8. Margin utilization
        current_qty = self._check_margin_limit(
            current_qty, entry_price, leverage, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 9. Concentration limit
        current_qty = self._check_concentration_limit(
            current_qty, entry_price, response
        )
        if current_qty <= 0:
            response.result = ExposureCheckResult.REJECTED
            response.adjusted_quantity = 0.0
            return response

        # 10. Correlation-adjusted check
        self._check_correlation_warning(symbol, side, response)

        # Set final result
        response.adjusted_quantity = current_qty
        response.max_allowed_quantity = current_qty

        if current_qty < quantity:
            response.result = ExposureCheckResult.REDUCED

        return response

    # ------------------------------------------------------------------
    # Individual limit checks
    # ------------------------------------------------------------------

    def _check_symbol_limit(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        response: ExposureCheckResponse,
    ) -> float:
        """Check per-symbol exposure limit."""
        max_symbol_notional = self._equity * (self.config.max_symbol_exposure_pct / 100)

        existing = self._positions.get(symbol)
        existing_notional = existing.notional if existing else 0.0
        new_notional = quantity * entry_price
        total_notional = existing_notional + new_notional

        if total_notional > max_symbol_notional:
            allowed_notional = max(0, max_symbol_notional - existing_notional)
            if allowed_notional <= 0:
                response.reasons.append(
                    f"Symbol {symbol} at limit: {existing_notional:.2f}/{max_symbol_notional:.2f}"
                )
                return 0.0

            reduced_qty = allowed_notional / entry_price
            response.reasons.append(
                f"Symbol exposure reduced: {symbol} {new_notional:.2f} -> {allowed_notional:.2f} "
                f"(limit: {max_symbol_notional:.2f})"
            )
            return reduced_qty

        return quantity

    def _check_sector_limit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        response: ExposureCheckResponse,
    ) -> float:
        """Check per-sector exposure limit."""
        sector = self._sector_map.get(symbol, "other")
        max_sector_notional = self._equity * (self.config.max_sector_exposure_pct / 100)

        sector_exposures = self._calculate_sector_exposure()
        current_sector_notional = sector_exposures.get(sector, 0.0)
        new_notional = quantity * entry_price
        total = current_sector_notional + new_notional

        if total > max_sector_notional:
            allowed = max(0, max_sector_notional - current_sector_notional)
            if allowed <= 0:
                response.reasons.append(
                    f"Sector '{sector}' at limit: {current_sector_notional:.2f}/{max_sector_notional:.2f}"
                )
                return 0.0

            reduced_qty = allowed / entry_price
            response.reasons.append(
                f"Sector exposure reduced: '{sector}' {new_notional:.2f} -> {allowed:.2f}"
            )
            return reduced_qty

        return quantity

    def _check_directional_limit(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        response: ExposureCheckResponse,
    ) -> float:
        """Check per-side (long/short) exposure limit."""
        new_notional = quantity * entry_price

        if side.upper() == "LONG":
            current = self._calculate_long_exposure()
            limit_pct = self.config.max_long_exposure_pct
        else:
            current = self._calculate_short_exposure()
            limit_pct = self.config.max_short_exposure_pct

        max_notional = self._equity * (limit_pct / 100)
        total = current + new_notional

        if total > max_notional:
            allowed = max(0, max_notional - current)
            if allowed <= 0:
                response.reasons.append(
                    f"{side} exposure at limit: {current:.2f}/{max_notional:.2f}"
                )
                return 0.0

            reduced_qty = allowed / entry_price
            response.reasons.append(
                f"{side} exposure reduced: limit={max_notional:.2f}"
            )
            return reduced_qty

        return quantity

    def _check_net_exposure_limit(
        self,
        side: str,
        quantity: float,
        entry_price: float,
        response: ExposureCheckResponse,
    ) -> float:
        """Check net exposure (directional bias) limit."""
        current_net = self._calculate_net_exposure()
        new_notional = quantity * entry_price

        if side.upper() == "LONG":
            projected_net = current_net + new_notional
        else:
            projected_net = current_net - new_notional

        max_net = self._equity * (self.config.max_net_exposure_pct / 100)

        if abs(projected_net) > max_net:
            # Calculate maximum allowed to stay within net limit
            if side.upper() == "LONG":
                allowed_notional = max(0, max_net - current_net)
            else:
                allowed_notional = max(0, max_net + current_net)

            if allowed_notional <= 0:
                response.reasons.append(
                    f"Net exposure limit: current_net={current_net:.2f}, "
                    f"limit={max_net:.2f}"
                )
                return 0.0

            reduced_qty = min(quantity, allowed_notional / entry_price)
            if reduced_qty < quantity:
                response.reasons.append(
                    f"Net exposure constrained: max_net={max_net:.2f}"
                )
            return reduced_qty

        return quantity

    def _check_gross_exposure_limit(
        self,
        quantity: float,
        entry_price: float,
        response: ExposureCheckResponse,
    ) -> float:
        """Check gross exposure limit."""
        current_gross = self._calculate_gross_exposure()
        new_notional = quantity * entry_price
        max_gross = self._equity * (self.config.max_gross_exposure_pct / 100)

        total = current_gross + new_notional
        if total > max_gross:
            allowed = max(0, max_gross - current_gross)
            if allowed <= 0:
                response.reasons.append(
                    f"Gross exposure at limit: {current_gross:.2f}/{max_gross:.2f}"
                )
                return 0.0

            reduced_qty = allowed / entry_price
            response.reasons.append(
                f"Gross exposure reduced: limit={max_gross:.2f}"
            )
            return reduced_qty

        return quantity

    def _check_leverage_limit(
        self,
        quantity: float,
        entry_price: float,
        leverage: int,
        response: ExposureCheckResponse,
    ) -> float:
        """Check account-level leverage limit."""
        current_gross = self._calculate_gross_exposure()
        new_notional = quantity * entry_price
        projected_gross = current_gross + new_notional

        if self._equity <= 0:
            return 0.0

        projected_leverage = projected_gross / self._equity

        if projected_leverage > self.config.max_account_leverage:
            max_notional = (self.config.max_account_leverage * self._equity) - current_gross
            if max_notional <= 0:
                response.reasons.append(
                    f"Account leverage at limit: {projected_leverage:.1f}x > "
                    f"{self.config.max_account_leverage:.1f}x"
                )
                return 0.0

            reduced_qty = max_notional / entry_price
            response.reasons.append(
                f"Leverage constrained: {projected_leverage:.1f}x -> "
                f"{self.config.max_account_leverage:.1f}x"
            )
            return reduced_qty

        # Warning check
        warning_leverage = self.config.max_account_leverage * (
            self.config.leverage_warning_pct / 100
        )
        if projected_leverage > warning_leverage:
            response.warnings.append(
                f"High leverage warning: {projected_leverage:.1f}x "
                f"(limit: {self.config.max_account_leverage:.1f}x)"
            )

        return quantity

    def _check_margin_limit(
        self,
        quantity: float,
        entry_price: float,
        leverage: int,
        response: ExposureCheckResponse,
    ) -> float:
        """Check margin utilization limit."""
        total_margin = self._margin_used + self._margin_available
        if total_margin <= 0:
            return quantity  # No margin info available, skip check

        new_margin_required = (quantity * entry_price) / max(leverage, 1)
        projected_margin_used = self._margin_used + new_margin_required
        projected_utilization = (projected_margin_used / total_margin) * 100

        if projected_utilization > self.config.margin_utilization_limit_pct:
            # Calculate max allowed
            max_margin = (
                total_margin * (self.config.margin_utilization_limit_pct / 100)
                - self._margin_used
            )
            if max_margin <= 0:
                response.reasons.append(
                    f"Margin utilization at limit: {projected_utilization:.1f}% "
                    f"(limit: {self.config.margin_utilization_limit_pct:.1f}%)"
                )
                return 0.0

            max_notional = max_margin * max(leverage, 1)
            reduced_qty = max_notional / entry_price
            response.reasons.append(
                f"Margin constrained: {projected_utilization:.1f}% -> "
                f"{self.config.margin_utilization_limit_pct:.1f}%"
            )
            return reduced_qty

        if projected_utilization > self.config.margin_utilization_warning_pct:
            response.warnings.append(
                f"Margin utilization warning: {projected_utilization:.1f}%"
            )

        return quantity

    def _check_concentration_limit(
        self,
        quantity: float,
        entry_price: float,
        response: ExposureCheckResponse,
    ) -> float:
        """Check that no single position exceeds concentration limit."""
        new_notional = quantity * entry_price
        current_gross = self._calculate_gross_exposure()
        projected_gross = current_gross + new_notional

        if projected_gross <= 0:
            return quantity

        concentration = (new_notional / projected_gross) * 100

        if concentration > self.config.max_concentration_pct:
            # Solve for max quantity: qty * price / (gross + qty * price) = limit
            # qty * price = limit * (gross + qty * price)
            # qty * price * (1 - limit) = limit * gross
            # qty = (limit * gross) / (price * (1 - limit))
            limit_frac = self.config.max_concentration_pct / 100
            if limit_frac >= 1.0:
                return quantity

            max_notional = (limit_frac * current_gross) / (1 - limit_frac)
            reduced_qty = max_notional / entry_price

            if reduced_qty < quantity:
                response.reasons.append(
                    f"Concentration limit: {concentration:.1f}% > "
                    f"{self.config.max_concentration_pct:.1f}%"
                )

            return min(quantity, reduced_qty)

        return quantity

    def _check_correlation_warning(
        self,
        symbol: str,
        side: str,
        response: ExposureCheckResponse,
    ) -> None:
        """Check for high correlation with existing positions."""
        if not self._correlation_matrix:
            return

        high_corr_positions = []

        for existing_symbol in self._positions:
            if existing_symbol == symbol:
                continue

            corr = self._correlation_matrix.get(
                (symbol, existing_symbol),
                self._correlation_matrix.get((existing_symbol, symbol), 0.0),
            )

            if abs(corr) >= self.config.correlation_penalty_threshold:
                existing_side = self._positions[existing_symbol].side
                direction = "same" if (
                    (corr > 0 and side.upper() == existing_side) or
                    (corr < 0 and side.upper() != existing_side)
                ) else "opposite"

                high_corr_positions.append(
                    f"{existing_symbol} (corr={corr:.2f}, {direction} risk)"
                )

        if high_corr_positions:
            response.warnings.append(
                f"High correlation with: {', '.join(high_corr_positions)}. "
                f"Effective exposure may be higher than nominal."
            )

    # ------------------------------------------------------------------
    # Snapshot and reporting
    # ------------------------------------------------------------------

    def get_snapshot(self) -> ExposureSnapshot:
        """Get current exposure snapshot."""
        now = datetime.now(timezone.utc)
        gross = self._calculate_gross_exposure()
        net = self._calculate_net_exposure()
        long_exp = self._calculate_long_exposure()
        short_exp = self._calculate_short_exposure()

        equity = max(self._equity, 0.001)  # avoid division by zero

        total_margin = self._margin_used + self._margin_available
        margin_util = (
            (self._margin_used / total_margin * 100) if total_margin > 0 else 0
        )

        warnings = []

        # Check for warnings
        account_leverage = gross / equity if equity > 0 else 0
        if account_leverage > self.config.max_account_leverage * 0.7:
            warnings.append(f"High leverage: {account_leverage:.1f}x")

        if margin_util > self.config.margin_utilization_warning_pct:
            warnings.append(f"High margin utilization: {margin_util:.1f}%")

        concentration = self._calculate_concentration()
        if concentration > self.config.max_concentration_pct * 0.8:
            warnings.append(f"High concentration: {concentration:.1f}%")

        snapshot = ExposureSnapshot(
            timestamp=now,
            equity=self._equity,
            gross_exposure=gross,
            net_exposure=net,
            long_exposure=long_exp,
            short_exposure=short_exp,
            gross_exposure_pct=(gross / equity * 100),
            net_exposure_pct=(net / equity * 100),
            long_exposure_pct=(long_exp / equity * 100),
            short_exposure_pct=(short_exp / equity * 100),
            account_leverage=account_leverage,
            margin_used=self._margin_used,
            margin_available=self._margin_available,
            margin_utilization_pct=margin_util,
            symbol_exposures=self._calculate_symbol_exposures(),
            sector_exposures=self._calculate_sector_exposure(),
            concentration_max_pct=concentration,
            warnings=warnings,
        )

        # Store history
        self._snapshot_history.append(snapshot)
        if len(self._snapshot_history) > self._max_history:
            self._snapshot_history = self._snapshot_history[-self._max_history:]

        return snapshot

    def get_position_details(self) -> List[SymbolExposure]:
        """Get detailed exposure for all positions."""
        return list(self._positions.values())

    def get_sector_breakdown(self) -> Dict[str, Dict]:
        """Get detailed sector breakdown."""
        sectors: Dict[str, Dict] = {}
        equity = max(self._equity, 0.001)

        for pos in self._positions.values():
            sector = self._sector_map.get(pos.symbol, "other")
            if sector not in sectors:
                sectors[sector] = {
                    "notional": 0.0,
                    "pct_of_equity": 0.0,
                    "symbols": [],
                    "long_notional": 0.0,
                    "short_notional": 0.0,
                }

            info = sectors[sector]
            info["notional"] += pos.notional
            info["symbols"].append(pos.symbol)
            if pos.side == "LONG":
                info["long_notional"] += pos.notional
            else:
                info["short_notional"] += pos.notional

        # Calculate percentages
        for sector_info in sectors.values():
            sector_info["pct_of_equity"] = sector_info["notional"] / equity * 100

        return sectors

    def get_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score (0-100).

        Based on HHI (Herfindahl-Hirschman Index).
        100 = perfectly diversified, 0 = single position.
        """
        if not self._positions:
            return 0.0

        gross = self._calculate_gross_exposure()
        if gross <= 0:
            return 0.0

        # Calculate HHI
        hhi = sum(
            (pos.notional / gross) ** 2
            for pos in self._positions.values()
        )

        # HHI ranges from 1/n (perfect diversification) to 1 (single position)
        n = len(self._positions)
        min_hhi = 1.0 / n if n > 0 else 1.0

        # Normalize to 0-100 scale
        if min_hhi >= 1.0:
            return 0.0

        score = (1.0 - hhi) / (1.0 - min_hhi) * 100
        return max(0.0, min(100.0, score))

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            "equity": self._equity,
            "margin_used": self._margin_used,
            "margin_available": self._margin_available,
            "positions": {
                sym: {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "mark_price": pos.mark_price,
                    "notional": pos.notional,
                    "leverage": pos.leverage,
                    "margin_mode": pos.margin_mode.value,
                    "sector": pos.sector,
                }
                for sym, pos in self._positions.items()
            },
            "config": {
                "max_gross_exposure_pct": self.config.max_gross_exposure_pct,
                "max_net_exposure_pct": self.config.max_net_exposure_pct,
                "max_symbol_exposure_pct": self.config.max_symbol_exposure_pct,
                "max_sector_exposure_pct": self.config.max_sector_exposure_pct,
                "max_account_leverage": self.config.max_account_leverage,
                "max_concentration_pct": self.config.max_concentration_pct,
            },
        }
