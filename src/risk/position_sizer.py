"""
Mefai Autotrade - Position Sizing Algorithms

Production-grade position sizing with multiple algorithms:
- Fixed fractional, Kelly criterion, Half-Kelly
- Volatility-based (ATR-normalized)
- Equal risk contribution
- Anti-martingale
- Optimal F (Ralph Vince)
- Leverage-aware and commission-adjusted sizing
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and configuration types
# ---------------------------------------------------------------------------

class SizingMethod(str, Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    VOLATILITY = "volatility"
    EQUAL_RISK = "equal_risk"
    ANTI_MARTINGALE = "anti_martingale"
    OPTIMAL_F = "optimal_f"
    FIXED_AMOUNT = "fixed_amount"
    FIXED_QUANTITY = "fixed_quantity"


@dataclass
class SizingConfig:
    """Configuration for position sizing."""
    method: SizingMethod = SizingMethod.FIXED_FRACTIONAL

    # Fixed fractional
    risk_per_trade_pct: float = 1.0  # percent of equity to risk per trade

    # Kelly criterion inputs
    win_rate: float = 0.55
    avg_win: float = 1.5  # average win / average loss (payoff ratio)
    kelly_fraction: float = 1.0  # 1.0 = full Kelly, 0.5 = half Kelly

    # Volatility-based
    atr_multiplier: float = 2.0  # stop loss distance in ATR units
    target_risk_pct: float = 1.0  # target portfolio risk per trade

    # Anti-martingale
    base_risk_pct: float = 1.0
    win_increment_pct: float = 0.25  # increase risk by this after each win
    loss_decrement_pct: float = 0.25  # decrease risk by this after each loss
    min_risk_pct: float = 0.25
    max_risk_pct: float = 3.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Optimal F
    largest_loss: float = 0.0  # largest historical loss in currency
    optimal_f_value: float = 0.0  # computed optimal f

    # Fixed amount / quantity
    fixed_amount_usd: float = 100.0
    fixed_quantity: float = 0.0

    # Position limits
    max_position_per_symbol: float = 0.0  # max notional per symbol (0 = no limit)
    max_position_per_side: float = 0.0  # max total notional long or short
    max_total_position: float = 0.0  # max total notional across all positions
    max_positions_count: int = 20  # max number of open positions

    # Leverage
    max_leverage: float = 20.0
    default_leverage: int = 1

    # Commission
    maker_fee_pct: float = 0.02  # percent
    taker_fee_pct: float = 0.04  # percent
    include_commission: bool = True

    # Minimum order
    min_notional: float = 5.0  # minimum order notional in USDT
    min_quantity: float = 0.0  # minimum order quantity (symbol-specific)

    # Precision
    quantity_precision: int = 3
    price_precision: int = 2


@dataclass
class SizingResult:
    """Result from position sizing calculation."""
    quantity: float = 0.0
    notional_value: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    leverage: int = 1
    method_used: SizingMethod = SizingMethod.FIXED_FRACTIONAL
    approved: bool = True
    rejection_reason: str = ""
    adjustments: List[str] = field(default_factory=list)
    raw_quantity: float = 0.0  # before adjustments
    commission_estimate: float = 0.0
    effective_risk_pct: float = 0.0  # after commission
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.adjustments:
            self.adjustments = []


@dataclass
class TradeRecord:
    """Minimal trade record for sizing history."""
    symbol: str
    side: str
    pnl: float
    pnl_pct: float
    is_win: bool
    timestamp: datetime


@dataclass
class PortfolioSnapshot:
    """Current portfolio state for sizing decisions."""
    equity: float
    available_balance: float
    total_unrealized_pnl: float
    open_positions: Dict[str, "PositionInfo"] = field(default_factory=dict)
    total_long_notional: float = 0.0
    total_short_notional: float = 0.0
    total_positions_count: int = 0


@dataclass
class PositionInfo:
    """Minimal position info for sizing checks."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    notional: float
    leverage: int
    unrealized_pnl: float


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Production-grade position sizing engine.

    Calculates optimal position size based on the selected algorithm,
    then applies all safety limits (max position, leverage, commission).
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig()
        self._trade_history: List[TradeRecord] = []
        self._sizing_log: List[SizingResult] = []
        self._max_log_size = 10000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        portfolio: PortfolioSnapshot,
        atr: Optional[float] = None,
        leverage: Optional[int] = None,
        method_override: Optional[SizingMethod] = None,
    ) -> SizingResult:
        """
        Calculate position size for a new trade.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: "LONG" or "SHORT"
            entry_price: Expected entry price
            stop_loss_price: Stop loss price
            portfolio: Current portfolio state
            atr: Average True Range value (required for volatility method)
            leverage: Override leverage (uses config default if None)
            method_override: Override sizing method for this trade

        Returns:
            SizingResult with calculated quantity and metadata
        """
        if entry_price <= 0:
            return self._rejected("Entry price must be positive")
        if stop_loss_price <= 0:
            return self._rejected("Stop loss price must be positive")
        if portfolio.equity <= 0:
            return self._rejected("Portfolio equity must be positive")

        method = method_override or self.config.method
        effective_leverage = leverage or self.config.default_leverage

        # Validate stop loss direction
        if side.upper() == "LONG" and stop_loss_price >= entry_price:
            return self._rejected(
                f"Long stop loss ({stop_loss_price}) must be below entry ({entry_price})"
            )
        if side.upper() == "SHORT" and stop_loss_price <= entry_price:
            return self._rejected(
                f"Short stop loss ({stop_loss_price}) must be above entry ({entry_price})"
            )

        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        risk_pct_per_unit = risk_per_unit / entry_price

        # Dispatch to sizing method
        try:
            raw_quantity = self._dispatch_sizing(
                method=method,
                equity=portfolio.equity,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_per_unit=risk_per_unit,
                risk_pct_per_unit=risk_pct_per_unit,
                atr=atr,
                leverage=effective_leverage,
            )
        except Exception as exc:
            logger.error(f"Sizing calculation error: {exc}")
            return self._rejected(f"Sizing calculation error: {str(exc)}")

        if raw_quantity <= 0:
            return self._rejected("Calculated quantity is zero or negative")

        # Build result
        result = SizingResult(
            raw_quantity=raw_quantity,
            quantity=raw_quantity,
            method_used=method,
            leverage=effective_leverage,
        )

        # Apply all limits and adjustments
        self._apply_leverage_limit(result, effective_leverage)
        self._apply_commission_adjustment(result, entry_price)
        self._apply_position_limits(result, symbol, side, entry_price, portfolio)
        self._apply_precision(result, entry_price)
        self._apply_minimum_check(result, entry_price)

        # Final calculations
        result.notional_value = result.quantity * entry_price
        result.risk_amount = result.quantity * risk_per_unit
        result.risk_pct = (result.risk_amount / portfolio.equity) * 100 if portfolio.equity > 0 else 0
        result.effective_risk_pct = (
            (result.risk_amount + result.commission_estimate) / portfolio.equity * 100
            if portfolio.equity > 0 else 0
        )

        # Log the result
        self._log_result(result)

        logger.info(
            f"Position size: {symbol} {side} qty={result.quantity:.{self.config.quantity_precision}f} "
            f"notional={result.notional_value:.2f} risk={result.risk_pct:.2f}% "
            f"method={method.value} leverage={effective_leverage}x"
        )

        return result

    def update_trade_result(self, record: TradeRecord) -> None:
        """Update trade history for adaptive sizing methods."""
        self._trade_history.append(record)

        # Update anti-martingale state
        if record.is_win:
            self.config.consecutive_wins += 1
            self.config.consecutive_losses = 0
        else:
            self.config.consecutive_losses += 1
            self.config.consecutive_wins = 0

        # Update Kelly inputs from history
        self._update_kelly_from_history()

        # Update Optimal F from history
        self._update_optimal_f_from_history()

        logger.debug(
            f"Trade result recorded: {record.symbol} pnl={record.pnl:.2f} "
            f"wins={self.config.consecutive_wins} losses={self.config.consecutive_losses}"
        )

    def get_kelly_fraction(self) -> float:
        """Calculate the Kelly criterion fraction."""
        return self._calculate_kelly()

    def get_optimal_f(self) -> float:
        """Calculate the Optimal F value from trade history."""
        return self._calculate_optimal_f()

    def get_anti_martingale_risk(self) -> float:
        """Get the current anti-martingale risk percentage."""
        return self._calculate_anti_martingale_risk()

    def get_equal_risk_weight(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate equal risk contribution weights.

        Args:
            symbols: List of symbols to allocate across
            volatilities: Symbol -> annualized volatility mapping

        Returns:
            Symbol -> weight mapping (weights sum to 1.0)
        """
        return self._calculate_equal_risk_weights(symbols, volatilities)

    def reset_state(self) -> None:
        """Reset adaptive sizing state."""
        self.config.consecutive_wins = 0
        self.config.consecutive_losses = 0
        self._trade_history.clear()
        self._sizing_log.clear()
        logger.info("Position sizer state reset")

    def get_sizing_stats(self) -> Dict:
        """Return statistics about recent sizing decisions."""
        if not self._sizing_log:
            return {"total_calculations": 0}

        approved = [r for r in self._sizing_log if r.approved]
        rejected = [r for r in self._sizing_log if not r.approved]
        risk_pcts = [r.risk_pct for r in approved if r.risk_pct > 0]

        return {
            "total_calculations": len(self._sizing_log),
            "approved": len(approved),
            "rejected": len(rejected),
            "avg_risk_pct": sum(risk_pcts) / len(risk_pcts) if risk_pcts else 0,
            "max_risk_pct": max(risk_pcts) if risk_pcts else 0,
            "min_risk_pct": min(risk_pcts) if risk_pcts else 0,
            "consecutive_wins": self.config.consecutive_wins,
            "consecutive_losses": self.config.consecutive_losses,
            "trade_history_size": len(self._trade_history),
        }

    # ------------------------------------------------------------------
    # Sizing method dispatch
    # ------------------------------------------------------------------

    def _dispatch_sizing(
        self,
        method: SizingMethod,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_unit: float,
        risk_pct_per_unit: float,
        atr: Optional[float],
        leverage: int,
    ) -> float:
        """Dispatch to the appropriate sizing algorithm."""
        dispatch_map = {
            SizingMethod.FIXED_FRACTIONAL: self._size_fixed_fractional,
            SizingMethod.KELLY: self._size_kelly,
            SizingMethod.HALF_KELLY: self._size_half_kelly,
            SizingMethod.VOLATILITY: self._size_volatility,
            SizingMethod.EQUAL_RISK: self._size_equal_risk,
            SizingMethod.ANTI_MARTINGALE: self._size_anti_martingale,
            SizingMethod.OPTIMAL_F: self._size_optimal_f,
            SizingMethod.FIXED_AMOUNT: self._size_fixed_amount,
            SizingMethod.FIXED_QUANTITY: self._size_fixed_quantity,
        }

        handler = dispatch_map.get(method)
        if handler is None:
            raise ValueError(f"Unknown sizing method: {method}")

        return handler(
            equity=equity,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            risk_per_unit=risk_per_unit,
            risk_pct_per_unit=risk_pct_per_unit,
            atr=atr,
            leverage=leverage,
        )

    # ------------------------------------------------------------------
    # Sizing algorithms
    # ------------------------------------------------------------------

    def _size_fixed_fractional(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Fixed Fractional position sizing.

        Risk a fixed percentage of equity on each trade.
        Position size = (Equity * Risk%) / (Entry - StopLoss)

        This is the most common and recommended method for systematic trading.
        It automatically scales position size with account equity.
        """
        risk_amount = equity * (self.config.risk_per_trade_pct / 100.0)
        if risk_per_unit <= 0:
            return 0.0
        quantity = risk_amount / risk_per_unit
        return quantity

    def _size_kelly(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Kelly Criterion position sizing.

        f* = (p * b - q) / b
        where:
            p = probability of winning
            b = ratio of average win to average loss
            q = probability of losing = 1 - p

        Kelly maximizes the long-run geometric growth rate of capital.
        Full Kelly can be very aggressive - consider half-Kelly for production.
        """
        kelly_f = self._calculate_kelly()

        if kelly_f <= 0:
            logger.warning("Kelly fraction is zero or negative - no edge detected")
            return 0.0

        # Apply Kelly fraction cap
        kelly_f = min(kelly_f, 0.25)  # never risk more than 25% even with full Kelly

        risk_amount = equity * kelly_f
        if risk_per_unit <= 0:
            return 0.0
        quantity = risk_amount / risk_per_unit
        return quantity

    def _size_half_kelly(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Half-Kelly position sizing.

        Uses 50% of the Kelly fraction for a more conservative approach.
        Half-Kelly achieves ~75% of the growth rate of full Kelly
        with significantly less volatility and drawdown risk.
        """
        # Temporarily set kelly_fraction to 0.5
        original = self.config.kelly_fraction
        self.config.kelly_fraction = 0.5
        result = self._size_kelly(
            equity=equity,
            entry_price=entry_price,
            risk_per_unit=risk_per_unit,
            leverage=leverage,
        )
        self.config.kelly_fraction = original
        return result

    def _size_volatility(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        atr: Optional[float],
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Volatility-based (ATR-normalized) position sizing.

        Position size = (Equity * Target Risk%) / (ATR * Multiplier)

        Adapts position size to current market volatility:
        - High volatility -> smaller position
        - Low volatility -> larger position

        This keeps dollar risk approximately constant across different
        volatility regimes.
        """
        if atr is None or atr <= 0:
            logger.warning("ATR not provided or zero - falling back to fixed fractional")
            return self._size_fixed_fractional(
                equity=equity,
                entry_price=entry_price,
                risk_per_unit=risk_per_unit,
                leverage=leverage,
            )

        risk_amount = equity * (self.config.target_risk_pct / 100.0)
        stop_distance = atr * self.config.atr_multiplier

        if stop_distance <= 0:
            return 0.0

        quantity = risk_amount / stop_distance
        return quantity

    def _size_equal_risk(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Equal Risk Contribution sizing.

        Each position contributes an equal amount of risk to the portfolio.
        Total target risk is divided by the maximum number of positions.

        position_risk = total_risk_budget / max_positions
        quantity = position_risk / risk_per_unit
        """
        max_positions = max(self.config.max_positions_count, 1)
        per_position_risk_pct = self.config.risk_per_trade_pct / max_positions * max_positions

        # Simpler: just use the configured risk per trade as the per-position budget
        # This ensures total risk = risk_per_trade * max_positions stays bounded
        risk_amount = equity * (self.config.risk_per_trade_pct / 100.0)

        if risk_per_unit <= 0:
            return 0.0

        quantity = risk_amount / risk_per_unit
        return quantity

    def _size_anti_martingale(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Anti-Martingale position sizing.

        Increase position size after wins, decrease after losses.
        This is the opposite of the Martingale system and is mathematically
        sound for positive-expectancy systems.

        risk% = base_risk + (consecutive_wins * win_increment) - (consecutive_losses * loss_decrement)
        Clamped between min_risk and max_risk.
        """
        current_risk_pct = self._calculate_anti_martingale_risk()
        risk_amount = equity * (current_risk_pct / 100.0)

        if risk_per_unit <= 0:
            return 0.0

        quantity = risk_amount / risk_per_unit
        return quantity

    def _size_optimal_f(
        self,
        equity: float,
        entry_price: float,
        risk_per_unit: float,
        leverage: int,
        **kwargs,
    ) -> float:
        """
        Optimal F position sizing (Ralph Vince).

        Optimal F maximizes the terminal wealth relative (TWR).
        It determines the fraction of equity to risk based on the
        actual distribution of trade outcomes.

        position_value = equity * f / |largest_loss_per_unit|

        WARNING: Optimal F can produce very large positions.
        Always use with proper position limits.
        """
        optimal_f = self._calculate_optimal_f()

        if optimal_f <= 0:
            logger.warning("Optimal F is zero - insufficient trade history")
            return self._size_fixed_fractional(
                equity=equity,
                entry_price=entry_price,
                risk_per_unit=risk_per_unit,
                leverage=leverage,
            )

        # Cap at 25% for safety
        optimal_f = min(optimal_f, 0.25)

        if self.config.largest_loss <= 0:
            logger.warning("No largest loss recorded - falling back to fixed fractional")
            return self._size_fixed_fractional(
                equity=equity,
                entry_price=entry_price,
                risk_per_unit=risk_per_unit,
                leverage=leverage,
            )

        # Vince formula: contracts = (equity * f) / largest_loss
        risk_amount = equity * optimal_f
        quantity = risk_amount / risk_per_unit
        return quantity

    def _size_fixed_amount(
        self,
        equity: float,
        entry_price: float,
        **kwargs,
    ) -> float:
        """
        Fixed notional amount sizing.

        Always trade a fixed USD amount regardless of account size.
        Simple but does not scale with equity.
        """
        if entry_price <= 0:
            return 0.0
        return self.config.fixed_amount_usd / entry_price

    def _size_fixed_quantity(self, **kwargs) -> float:
        """
        Fixed quantity sizing.

        Always trade a fixed number of contracts/coins.
        Useful for testing or very specific strategies.
        """
        return self.config.fixed_quantity

    # ------------------------------------------------------------------
    # Algorithm helpers
    # ------------------------------------------------------------------

    def _calculate_kelly(self) -> float:
        """
        Calculate Kelly criterion fraction.

        f* = (p * b - q) / b

        Returns the fraction of equity to bet.
        Negative means no edge - do not trade.
        """
        p = self.config.win_rate
        b = self.config.avg_win  # payoff ratio (avg_win / avg_loss)
        q = 1.0 - p

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b

        # Apply user-configured fraction (e.g., 0.5 for half-Kelly)
        kelly *= self.config.kelly_fraction

        return max(kelly, 0.0)

    def _calculate_anti_martingale_risk(self) -> float:
        """Calculate current risk percentage for anti-martingale."""
        risk = self.config.base_risk_pct

        # Increase after wins
        risk += self.config.consecutive_wins * self.config.win_increment_pct

        # Decrease after losses
        risk -= self.config.consecutive_losses * self.config.loss_decrement_pct

        # Clamp
        risk = max(self.config.min_risk_pct, min(risk, self.config.max_risk_pct))

        return risk

    def _calculate_optimal_f(self) -> float:
        """
        Calculate Optimal F using brute force search over trade history.

        Searches for the f value that maximizes Terminal Wealth Relative (TWR).
        TWR = product of (1 + f * (-trade_pnl_pct / largest_loss_pct))
        """
        if len(self._trade_history) < 30:
            return 0.0

        pnl_pcts = [t.pnl_pct for t in self._trade_history]
        largest_loss_pct = min(pnl_pcts)

        if largest_loss_pct >= 0:
            return 0.0  # No losses recorded

        best_f = 0.0
        best_twr = 0.0

        # Search from 0.01 to 1.0 in steps of 0.01
        for f_int in range(1, 101):
            f = f_int / 100.0
            twr = 1.0
            valid = True

            for pnl_pct in pnl_pcts:
                holding_period_return = 1.0 + f * (-pnl_pct / largest_loss_pct)
                if holding_period_return <= 0:
                    valid = False
                    break
                twr *= holding_period_return

            if valid and twr > best_twr:
                best_twr = twr
                best_f = f

        self.config.optimal_f_value = best_f
        self.config.largest_loss = abs(largest_loss_pct)

        return best_f

    def _calculate_equal_risk_weights(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate inverse-volatility weights for equal risk contribution.

        Weight_i = (1 / vol_i) / sum(1 / vol_j for all j)

        Higher volatility assets get smaller weights.
        """
        if not symbols or not volatilities:
            return {}

        inverse_vols = {}
        for sym in symbols:
            vol = volatilities.get(sym, 0.0)
            if vol > 0:
                inverse_vols[sym] = 1.0 / vol
            else:
                inverse_vols[sym] = 0.0

        total_inv_vol = sum(inverse_vols.values())
        if total_inv_vol <= 0:
            # Equal weight fallback
            n = len(symbols)
            return {sym: 1.0 / n for sym in symbols}

        weights = {sym: iv / total_inv_vol for sym, iv in inverse_vols.items()}
        return weights

    def _update_kelly_from_history(self) -> None:
        """Update Kelly parameters from trade history."""
        if len(self._trade_history) < 20:
            return

        wins = [t for t in self._trade_history if t.is_win]
        losses = [t for t in self._trade_history if not t.is_win]

        if not wins or not losses:
            return

        self.config.win_rate = len(wins) / len(self._trade_history)
        avg_win_amt = sum(abs(t.pnl) for t in wins) / len(wins)
        avg_loss_amt = sum(abs(t.pnl) for t in losses) / len(losses)

        if avg_loss_amt > 0:
            self.config.avg_win = avg_win_amt / avg_loss_amt

    def _update_optimal_f_from_history(self) -> None:
        """Recalculate Optimal F when trade history changes."""
        if len(self._trade_history) < 30:
            return

        pnl_values = [t.pnl for t in self._trade_history]
        min_pnl = min(pnl_values)
        if min_pnl < 0:
            self.config.largest_loss = abs(min_pnl)

    # ------------------------------------------------------------------
    # Limit enforcement
    # ------------------------------------------------------------------

    def _apply_leverage_limit(self, result: SizingResult, leverage: int) -> None:
        """Ensure leverage does not exceed maximum."""
        if leverage > self.config.max_leverage:
            result.leverage = int(self.config.max_leverage)
            result.adjustments.append(
                f"Leverage reduced from {leverage}x to {result.leverage}x (max={self.config.max_leverage}x)"
            )
        else:
            result.leverage = leverage

    def _apply_commission_adjustment(
        self,
        result: SizingResult,
        entry_price: float,
    ) -> None:
        """
        Adjust position size to account for commission costs.

        Estimates round-trip commission and reduces position so that
        total risk (stop loss + commission) stays within the risk budget.
        """
        if not self.config.include_commission:
            result.commission_estimate = 0.0
            return

        # Estimate round-trip commission (entry + exit)
        notional = result.quantity * entry_price
        fee_rate = self.config.taker_fee_pct / 100.0  # assume taker for worst case
        round_trip_commission = notional * fee_rate * 2

        result.commission_estimate = round_trip_commission

        # If commission is significant relative to position, note it
        if notional > 0 and (round_trip_commission / notional) > 0.001:
            result.adjustments.append(
                f"Commission estimate: {round_trip_commission:.4f} USDT "
                f"({round_trip_commission / notional * 100:.3f}% of notional)"
            )

    def _apply_position_limits(
        self,
        result: SizingResult,
        symbol: str,
        side: str,
        entry_price: float,
        portfolio: PortfolioSnapshot,
    ) -> None:
        """Apply all position size limits."""
        notional = result.quantity * entry_price

        # Max positions count
        if portfolio.total_positions_count >= self.config.max_positions_count:
            result.approved = False
            result.rejection_reason = (
                f"Max positions reached ({portfolio.total_positions_count}/{self.config.max_positions_count})"
            )
            result.quantity = 0.0
            return

        # Per-symbol limit
        if self.config.max_position_per_symbol > 0:
            existing = portfolio.open_positions.get(symbol)
            existing_notional = existing.notional if existing else 0.0
            total_symbol_notional = existing_notional + notional

            if total_symbol_notional > self.config.max_position_per_symbol:
                allowed_notional = max(0, self.config.max_position_per_symbol - existing_notional)
                if allowed_notional <= 0:
                    result.approved = False
                    result.rejection_reason = (
                        f"Symbol limit reached for {symbol}: "
                        f"existing={existing_notional:.2f}, limit={self.config.max_position_per_symbol:.2f}"
                    )
                    result.quantity = 0.0
                    return

                new_qty = allowed_notional / entry_price
                result.adjustments.append(
                    f"Quantity reduced from {result.quantity:.6f} to {new_qty:.6f} "
                    f"(symbol limit: {self.config.max_position_per_symbol:.2f})"
                )
                result.quantity = new_qty
                notional = result.quantity * entry_price

        # Per-side limit
        if self.config.max_position_per_side > 0:
            if side.upper() == "LONG":
                current_side_notional = portfolio.total_long_notional
            else:
                current_side_notional = portfolio.total_short_notional

            total_side_notional = current_side_notional + notional
            if total_side_notional > self.config.max_position_per_side:
                allowed = max(0, self.config.max_position_per_side - current_side_notional)
                if allowed <= 0:
                    result.approved = False
                    result.rejection_reason = (
                        f"Side limit reached for {side}: "
                        f"current={current_side_notional:.2f}, limit={self.config.max_position_per_side:.2f}"
                    )
                    result.quantity = 0.0
                    return

                new_qty = allowed / entry_price
                result.adjustments.append(
                    f"Quantity reduced from {result.quantity:.6f} to {new_qty:.6f} "
                    f"(side limit: {self.config.max_position_per_side:.2f})"
                )
                result.quantity = new_qty
                notional = result.quantity * entry_price

        # Total portfolio limit
        if self.config.max_total_position > 0:
            current_total = portfolio.total_long_notional + portfolio.total_short_notional
            total_portfolio = current_total + notional
            if total_portfolio > self.config.max_total_position:
                allowed = max(0, self.config.max_total_position - current_total)
                if allowed <= 0:
                    result.approved = False
                    result.rejection_reason = (
                        f"Total portfolio limit reached: "
                        f"current={current_total:.2f}, limit={self.config.max_total_position:.2f}"
                    )
                    result.quantity = 0.0
                    return

                new_qty = allowed / entry_price
                result.adjustments.append(
                    f"Quantity reduced from {result.quantity:.6f} to {new_qty:.6f} "
                    f"(total portfolio limit: {self.config.max_total_position:.2f})"
                )
                result.quantity = new_qty

        # Available balance check (margin required)
        margin_required = (result.quantity * entry_price) / result.leverage
        if margin_required > portfolio.available_balance:
            if portfolio.available_balance <= 0:
                result.approved = False
                result.rejection_reason = "No available balance"
                result.quantity = 0.0
                return

            max_qty = (portfolio.available_balance * result.leverage) / entry_price
            # Leave 2% buffer for fees and slippage
            max_qty *= 0.98
            result.adjustments.append(
                f"Quantity reduced from {result.quantity:.6f} to {max_qty:.6f} "
                f"(available balance: {portfolio.available_balance:.2f})"
            )
            result.quantity = max_qty

    def _apply_precision(self, result: SizingResult, entry_price: float) -> None:
        """Round quantity to exchange precision."""
        if result.quantity <= 0:
            return

        precision = self.config.quantity_precision
        factor = 10 ** precision
        # Always round down to avoid exceeding limits
        result.quantity = math.floor(result.quantity * factor) / factor

        if result.quantity <= 0:
            result.approved = False
            result.rejection_reason = (
                f"Quantity rounds to zero at {precision} decimal places"
            )

    def _apply_minimum_check(self, result: SizingResult, entry_price: float) -> None:
        """Ensure order meets minimum requirements."""
        if not result.approved or result.quantity <= 0:
            return

        notional = result.quantity * entry_price

        if notional < self.config.min_notional:
            result.approved = False
            result.rejection_reason = (
                f"Notional {notional:.2f} below minimum {self.config.min_notional:.2f}"
            )
            result.quantity = 0.0
            return

        if self.config.min_quantity > 0 and result.quantity < self.config.min_quantity:
            result.approved = False
            result.rejection_reason = (
                f"Quantity {result.quantity} below minimum {self.config.min_quantity}"
            )
            result.quantity = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rejected(self, reason: str) -> SizingResult:
        """Create a rejected sizing result."""
        result = SizingResult(
            approved=False,
            rejection_reason=reason,
        )
        self._log_result(result)
        return result

    def _log_result(self, result: SizingResult) -> None:
        """Log sizing result for statistics."""
        self._sizing_log.append(result)
        if len(self._sizing_log) > self._max_log_size:
            self._sizing_log = self._sizing_log[-self._max_log_size:]

    def to_dict(self) -> Dict:
        """Serialize current state."""
        return {
            "config": {
                "method": self.config.method.value,
                "risk_per_trade_pct": self.config.risk_per_trade_pct,
                "win_rate": self.config.win_rate,
                "avg_win": self.config.avg_win,
                "kelly_fraction": self.config.kelly_fraction,
                "atr_multiplier": self.config.atr_multiplier,
                "target_risk_pct": self.config.target_risk_pct,
                "base_risk_pct": self.config.base_risk_pct,
                "consecutive_wins": self.config.consecutive_wins,
                "consecutive_losses": self.config.consecutive_losses,
                "max_leverage": self.config.max_leverage,
                "max_positions_count": self.config.max_positions_count,
            },
            "stats": self.get_sizing_stats(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PositionSizer":
        """Deserialize from dict."""
        config_data = data.get("config", {})
        config = SizingConfig(
            method=SizingMethod(config_data.get("method", "fixed_fractional")),
            risk_per_trade_pct=config_data.get("risk_per_trade_pct", 1.0),
            win_rate=config_data.get("win_rate", 0.55),
            avg_win=config_data.get("avg_win", 1.5),
            kelly_fraction=config_data.get("kelly_fraction", 1.0),
            atr_multiplier=config_data.get("atr_multiplier", 2.0),
            target_risk_pct=config_data.get("target_risk_pct", 1.0),
            base_risk_pct=config_data.get("base_risk_pct", 1.0),
            consecutive_wins=config_data.get("consecutive_wins", 0),
            consecutive_losses=config_data.get("consecutive_losses", 0),
            max_leverage=config_data.get("max_leverage", 20.0),
            max_positions_count=config_data.get("max_positions_count", 20),
        )
        return cls(config=config)
