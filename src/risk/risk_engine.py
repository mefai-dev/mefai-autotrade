"""
Mefai Autotrade - Risk Engine Orchestrator

Central risk management engine that coordinates all risk modules:
- Pre-trade risk checks (sizing, exposure, drawdown, correlation, heat)
- Post-trade monitoring (drawdown tracking, circuit breaker)
- Risk check results: APPROVED, REDUCED, REJECTED
- Risk override mechanism for manual trades
- Configuration hot-reload
- Risk report generation
"""

import logging
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from pathlib import Path

from .position_sizer import PositionSizer, SizingConfig, SizingResult, PortfolioSnapshot
from .drawdown_manager import DrawdownManager, DrawdownConfig, DrawdownState
from .exposure_manager import ExposureManager, ExposureConfig, ExposureCheckResult
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from .correlation_monitor import CorrelationMonitor, CorrelationConfig
from .risk_metrics import RiskMetrics, RiskMetricsConfig
from .stop_loss_manager import StopLossManager, StopConfig, StopType
from .portfolio_heat import PortfolioHeat, HeatConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RiskDecision(str, Enum):
    APPROVED = "approved"
    REDUCED = "reduced"
    REJECTED = "rejected"


@dataclass
class RiskCheckResult:
    """Result of a complete risk check."""
    decision: RiskDecision
    original_quantity: float
    approved_quantity: float
    stop_loss_price: float
    leverage: int
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    override: bool = False


@dataclass
class RiskEngineConfig:
    """Configuration for the risk engine."""
    # Module enable/disable
    enable_position_sizing: bool = True
    enable_drawdown_protection: bool = True
    enable_exposure_limits: bool = True
    enable_circuit_breaker: bool = True
    enable_correlation_monitor: bool = True
    enable_stop_loss_manager: bool = True
    enable_portfolio_heat: bool = True

    # Override settings
    allow_overrides: bool = False
    override_requires_reason: bool = True
    max_override_size_pct: float = 5.0  # max 5% equity per override

    # Config file path for hot-reload
    config_file_path: str = ""

    # Module configs
    sizing_config: Optional[SizingConfig] = None
    drawdown_config: Optional[DrawdownConfig] = None
    exposure_config: Optional[ExposureConfig] = None
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    correlation_config: Optional[CorrelationConfig] = None
    stop_config: Optional[StopConfig] = None
    heat_config: Optional[HeatConfig] = None


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Central risk management orchestrator.

    Runs all risk checks before every trade and coordinates
    post-trade monitoring across all risk modules.

    Usage:
        engine = RiskEngine(config)
        engine.initialize(equity=10000.0)

        # Before every trade:
        result = engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG", quantity=0.1,
            entry_price=50000, stop_loss_price=49000
        )

        if result.decision != RiskDecision.REJECTED:
            # Execute trade with result.approved_quantity
            pass

        # After every trade close:
        engine.post_trade_update(symbol="BTCUSDT", pnl=150.0, pnl_pct=1.5)

        # Periodic monitoring:
        engine.update_equity(equity=10150.0)
    """

    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self.config = config or RiskEngineConfig()

        # Initialize all modules
        self.sizer = PositionSizer(self.config.sizing_config or SizingConfig())
        self.drawdown = DrawdownManager(self.config.drawdown_config or DrawdownConfig())
        self.exposure = ExposureManager(self.config.exposure_config or ExposureConfig())
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_config or CircuitBreakerConfig()
        )
        self.correlation = CorrelationMonitor(
            self.config.correlation_config or CorrelationConfig()
        )
        self.metrics = RiskMetrics(RiskMetricsConfig())
        self.stop_loss = StopLossManager(self.config.stop_config or StopConfig())
        self.heat = PortfolioHeat(self.config.heat_config or HeatConfig())

        # State
        self._equity: float = 0.0
        self._initialized: bool = False
        self._check_log: List[RiskCheckResult] = []
        self._max_log = 10000
        self._config_mtime: float = 0.0

        logger.info("Risk engine created")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, equity: float) -> None:
        """
        Initialize the risk engine with starting equity.

        Must be called before any risk checks.
        """
        self._equity = equity
        self.drawdown.initialize(equity)
        self.exposure.update_equity(equity)
        self.heat.update_equity(equity)
        self._initialized = True

        logger.info(f"Risk engine initialized with equity={equity:.2f}")

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss_price: float,
        atr: Optional[float] = None,
        leverage: int = 1,
        strategy: str = "default",
        override: bool = False,
        override_reason: str = "",
    ) -> RiskCheckResult:
        """
        Run all pre-trade risk checks.

        Args:
            symbol: Trading pair
            side: "LONG" or "SHORT"
            quantity: Proposed quantity
            entry_price: Expected entry price
            stop_loss_price: Stop loss price
            atr: Current ATR (for sizing and stops)
            leverage: Leverage for this trade
            strategy: Strategy name
            override: Set True for manual override
            override_reason: Reason for override (required if override=True)

        Returns:
            RiskCheckResult with decision and approved quantity
        """
        if not self._initialized:
            return RiskCheckResult(
                decision=RiskDecision.REJECTED,
                original_quantity=quantity,
                approved_quantity=0.0,
                stop_loss_price=stop_loss_price,
                leverage=leverage,
                reasons=["Risk engine not initialized"],
            )

        # Hot-reload config if available
        self._check_config_reload()

        result = RiskCheckResult(
            decision=RiskDecision.APPROVED,
            original_quantity=quantity,
            approved_quantity=quantity,
            stop_loss_price=stop_loss_price,
            leverage=leverage,
            override=override,
        )

        # Handle override
        if override:
            return self._handle_override(
                result, quantity, entry_price, override_reason
            )

        current_qty = quantity

        # 1. Circuit breaker check
        if self.config.enable_circuit_breaker:
            current_qty = self._check_circuit_breaker(result, current_qty)
            if current_qty <= 0:
                result.decision = RiskDecision.REJECTED
                result.approved_quantity = 0.0
                self._log_check(result)
                return result

        # 2. Drawdown check
        if self.config.enable_drawdown_protection:
            current_qty = self._check_drawdown(result, current_qty)
            if current_qty <= 0:
                result.decision = RiskDecision.REJECTED
                result.approved_quantity = 0.0
                self._log_check(result)
                return result

        # 3. Position sizing
        if self.config.enable_position_sizing:
            current_qty = self._check_position_size(
                result, symbol, side, current_qty,
                entry_price, stop_loss_price, atr, leverage,
            )
            if current_qty <= 0:
                result.decision = RiskDecision.REJECTED
                result.approved_quantity = 0.0
                self._log_check(result)
                return result

        # 4. Exposure check
        if self.config.enable_exposure_limits:
            current_qty = self._check_exposure(
                result, symbol, side, current_qty, entry_price, leverage,
            )
            if current_qty <= 0:
                result.decision = RiskDecision.REJECTED
                result.approved_quantity = 0.0
                self._log_check(result)
                return result

        # 5. Portfolio heat check
        if self.config.enable_portfolio_heat:
            current_qty = self._check_heat(
                result, symbol, entry_price, stop_loss_price,
                current_qty, strategy,
            )
            if current_qty <= 0:
                result.decision = RiskDecision.REJECTED
                result.approved_quantity = 0.0
                self._log_check(result)
                return result

        # 6. Correlation warning (advisory only, does not reduce size)
        if self.config.enable_correlation_monitor:
            self._check_correlation(result, symbol, side)

        # Final decision
        result.approved_quantity = current_qty
        if current_qty < quantity:
            result.decision = RiskDecision.REDUCED

        # Add risk metrics
        result.risk_metrics = {
            "risk_amount": current_qty * abs(entry_price - stop_loss_price),
            "risk_pct": (
                current_qty * abs(entry_price - stop_loss_price) / self._equity * 100
                if self._equity > 0 else 0
            ),
            "notional": current_qty * entry_price,
            "portfolio_heat": self.heat.get_total_heat(),
            "drawdown_state": self.drawdown.get_state().value,
            "circuit_breaker_state": self.circuit_breaker.get_state().value,
        }

        self._log_check(result)

        logger.info(
            f"Risk check: {symbol} {side} qty={quantity:.6f} -> {current_qty:.6f} "
            f"decision={result.decision.value} checks={result.checks_passed}"
        )

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_circuit_breaker(
        self,
        result: RiskCheckResult,
        quantity: float,
    ) -> float:
        """Check circuit breaker state."""
        cb_state, messages = self.circuit_breaker.check_all(self._equity)

        result.checks_passed["circuit_breaker"] = self.circuit_breaker.is_trading_allowed()
        result.warnings.extend(messages)

        if not self.circuit_breaker.is_trading_allowed():
            result.reasons.append(
                f"Circuit breaker {cb_state.value}: {self.circuit_breaker._trigger_reason}"
            )
            return 0.0

        return quantity

    def _check_drawdown(
        self,
        result: RiskCheckResult,
        quantity: float,
    ) -> float:
        """Check drawdown limits."""
        dd_snapshot = self.drawdown.get_snapshot()

        result.checks_passed["drawdown"] = self.drawdown.is_trading_allowed()

        if not self.drawdown.is_trading_allowed():
            result.reasons.append(
                f"Drawdown protection: state={dd_snapshot.state.value}, "
                f"dd={dd_snapshot.drawdown_pct:.2f}%"
            )
            return 0.0

        # Apply size multiplier from drawdown manager
        multiplier = self.drawdown.get_size_multiplier()
        if multiplier < 1.0:
            adjusted = quantity * multiplier
            result.warnings.append(
                f"Drawdown recovery mode: size reduced by "
                f"{(1 - multiplier) * 100:.0f}%"
            )
            return adjusted

        return quantity

    def _check_position_size(
        self,
        result: RiskCheckResult,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss_price: float,
        atr: Optional[float],
        leverage: int,
    ) -> float:
        """Run position sizer."""
        portfolio = PortfolioSnapshot(
            equity=self._equity,
            available_balance=self._equity * 0.9,  # conservative estimate
            total_unrealized_pnl=0.0,
        )

        sizing = self.sizer.calculate_size(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            portfolio=portfolio,
            atr=atr,
            leverage=leverage,
        )

        result.checks_passed["position_sizing"] = sizing.approved

        if not sizing.approved:
            result.reasons.append(f"Position sizing: {sizing.rejection_reason}")
            return 0.0

        # Use the smaller of proposed and calculated quantity
        sized_qty = min(quantity, sizing.quantity)
        if sized_qty < quantity:
            result.warnings.append(
                f"Position sizer reduced: {quantity:.6f} -> {sized_qty:.6f} "
                f"({sizing.method_used.value})"
            )

        result.leverage = sizing.leverage
        return sized_qty

    def _check_exposure(
        self,
        result: RiskCheckResult,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        leverage: int,
    ) -> float:
        """Check exposure limits."""
        exposure_check = self.exposure.check_new_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            leverage=leverage,
        )

        result.checks_passed["exposure"] = (
            exposure_check.result != ExposureCheckResult.REJECTED
        )
        result.warnings.extend(exposure_check.warnings)

        if exposure_check.result == ExposureCheckResult.REJECTED:
            result.reasons.extend(exposure_check.reasons)
            return 0.0

        if exposure_check.result == ExposureCheckResult.REDUCED:
            result.warnings.extend(exposure_check.reasons)
            return exposure_check.adjusted_quantity

        return quantity

    def _check_heat(
        self,
        result: RiskCheckResult,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        quantity: float,
        strategy: str,
    ) -> float:
        """Check portfolio heat limits."""
        heat_check = self.heat.check_new_trade(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_loss_price,
            quantity=quantity,
            strategy=strategy,
        )

        result.checks_passed["portfolio_heat"] = heat_check["approved"]

        if not heat_check["approved"]:
            result.reasons.append(f"Portfolio heat: {heat_check['reason']}")
            return 0.0

        adjusted = heat_check["adjusted_quantity"]
        if adjusted < quantity:
            result.warnings.append(
                f"Heat limit: qty {quantity:.6f} -> {adjusted:.6f} "
                f"({heat_check.get('reason', '')})"
            )

        return adjusted

    def _check_correlation(
        self,
        result: RiskCheckResult,
        symbol: str,
        side: str,
    ) -> None:
        """Check correlation warnings (advisory only)."""
        snapshot = self.correlation.get_snapshot()
        result.checks_passed["correlation"] = True  # advisory only

        if snapshot.warnings:
            result.warnings.extend(snapshot.warnings)

        # Check for specific high-correlation pairs
        suggestions = self.correlation.suggest_adjustments([symbol])
        for suggestion in suggestions:
            result.warnings.append(suggestion.get("reason", ""))

    def _handle_override(
        self,
        result: RiskCheckResult,
        quantity: float,
        entry_price: float,
        reason: str,
    ) -> RiskCheckResult:
        """Handle manual trade override."""
        if not self.config.allow_overrides:
            result.decision = RiskDecision.REJECTED
            result.approved_quantity = 0.0
            result.reasons.append("Overrides are not enabled")
            self._log_check(result)
            return result

        if self.config.override_requires_reason and not reason:
            result.decision = RiskDecision.REJECTED
            result.approved_quantity = 0.0
            result.reasons.append("Override requires a reason")
            self._log_check(result)
            return result

        # Apply override size limit
        max_notional = self._equity * (self.config.max_override_size_pct / 100)
        proposed_notional = quantity * entry_price

        if proposed_notional > max_notional:
            max_qty = max_notional / entry_price
            result.approved_quantity = max_qty
            result.decision = RiskDecision.REDUCED
            result.warnings.append(
                f"Override size capped: {quantity:.6f} -> {max_qty:.6f} "
                f"(max {self.config.max_override_size_pct}% of equity)"
            )
        else:
            result.approved_quantity = quantity
            result.decision = RiskDecision.APPROVED

        result.override = True
        result.warnings.append(f"OVERRIDE: {reason}")
        result.checks_passed = {k: True for k in [
            "circuit_breaker", "drawdown", "position_sizing",
            "exposure", "portfolio_heat", "correlation",
        ]}

        logger.warning(
            f"Risk OVERRIDE: qty={result.approved_quantity:.6f} reason='{reason}'"
        )

        self._log_check(result)
        return result

    # ------------------------------------------------------------------
    # Post-trade updates
    # ------------------------------------------------------------------

    def post_trade_update(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        is_win: Optional[bool] = None,
    ) -> None:
        """
        Update risk modules after a trade closes.

        Args:
            symbol: Trading pair
            pnl: Trade PnL in USD
            pnl_pct: Trade PnL as percentage
            is_win: Whether the trade was profitable (auto-detected if None)
        """
        if is_win is None:
            is_win = pnl > 0

        # Update circuit breaker
        self.circuit_breaker.record_trade_result(
            symbol=symbol,
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_win=is_win,
        )

        # Update position sizer (for adaptive methods)
        from .position_sizer import TradeRecord
        self.sizer.update_trade_result(TradeRecord(
            symbol=symbol,
            side="",
            pnl=pnl,
            pnl_pct=pnl_pct,
            is_win=is_win,
            timestamp=datetime.now(timezone.utc),
        ))

        # Update risk metrics
        self.metrics.add_trade_result(pnl, pnl_pct)

        # Remove from tracking
        self.stop_loss.remove_position(symbol)
        self.heat.remove_position(symbol)
        self.exposure.remove_position(symbol)

        logger.info(
            f"Post-trade update: {symbol} pnl={pnl:.2f} ({pnl_pct:.2f}%) "
            f"win={is_win}"
        )

    def update_equity(self, equity: float) -> Dict:
        """
        Update equity across all risk modules.

        Should be called periodically (every few seconds to minutes).

        Returns dict of any triggered alerts.
        """
        self._equity = equity

        alerts = {"drawdown": [], "circuit_breaker": [], "heat": []}

        # Update drawdown
        if self.config.enable_drawdown_protection:
            dd_snap = self.drawdown.update(equity)
            alerts["drawdown"] = dd_snap.alerts_triggered

        # Update exposure
        self.exposure.update_equity(equity)

        # Update heat
        self.heat.update_equity(equity)

        # Update metrics
        self.metrics.add_equity(equity)

        # Update circuit breaker
        if self.config.enable_circuit_breaker:
            cb_state, cb_messages = self.circuit_breaker.check_all(equity)
            alerts["circuit_breaker"] = cb_messages

        return alerts

    def update_price(
        self,
        symbol: str,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> Dict:
        """
        Update price data across risk modules.

        Returns any stop loss actions triggered.
        """
        result = {"stop_action": None}

        # Update correlation monitor
        if self.config.enable_correlation_monitor:
            self.correlation.add_price(symbol, price)

        # Update stop loss
        if self.config.enable_stop_loss_manager:
            stop_update = self.stop_loss.update(
                symbol=symbol,
                current_price=price,
                high=high,
                low=low,
                atr=atr,
            )
            if stop_update.action.value != "none":
                result["stop_action"] = {
                    "action": stop_update.action.value,
                    "symbol": symbol,
                    "stop_price": stop_update.new_stop_price,
                    "close_pct": stop_update.close_quantity_pct,
                    "reason": stop_update.reason,
                }

        # Update circuit breaker with price data
        if self.config.enable_circuit_breaker:
            self.circuit_breaker.record_price_update()
            if atr:
                self.circuit_breaker.record_atr(atr)

        return result

    def record_heartbeat(self) -> None:
        """Record exchange connectivity heartbeat."""
        self.circuit_breaker.record_heartbeat()

    # ------------------------------------------------------------------
    # Configuration hot-reload
    # ------------------------------------------------------------------

    def _check_config_reload(self) -> None:
        """Check if config file has been modified and reload if needed."""
        if not self.config.config_file_path:
            return

        path = Path(self.config.config_file_path)
        if not path.exists():
            return

        try:
            mtime = path.stat().st_mtime
            if mtime > self._config_mtime:
                self._reload_config(path)
                self._config_mtime = mtime
        except Exception as exc:
            logger.error(f"Config reload check error: {exc}")

    def _reload_config(self, path: Path) -> None:
        """Reload configuration from file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Update individual module configs
            if "sizing" in data:
                self._apply_sizing_config(data["sizing"])
            if "drawdown" in data:
                self._apply_drawdown_config(data["drawdown"])
            if "exposure" in data:
                self._apply_exposure_config(data["exposure"])
            if "circuit_breaker" in data:
                self._apply_circuit_breaker_config(data["circuit_breaker"])
            if "heat" in data:
                self._apply_heat_config(data["heat"])

            logger.info(f"Risk config reloaded from {path}")

        except Exception as exc:
            logger.error(f"Config reload error: {exc}")

    def _apply_sizing_config(self, data: Dict) -> None:
        """Apply sizing config updates."""
        config = self.sizer.config
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _apply_drawdown_config(self, data: Dict) -> None:
        """Apply drawdown config updates."""
        config = self.drawdown.config
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _apply_exposure_config(self, data: Dict) -> None:
        """Apply exposure config updates."""
        config = self.exposure.config
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _apply_circuit_breaker_config(self, data: Dict) -> None:
        """Apply circuit breaker config updates."""
        config = self.circuit_breaker.config
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _apply_heat_config(self, data: Dict) -> None:
        """Apply heat config updates."""
        config = self.heat.config
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report across all modules."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": self._equity,
            "status": {
                "circuit_breaker": self.circuit_breaker.get_state().value,
                "drawdown": self.drawdown.get_state().value,
                "trading_allowed": (
                    self.circuit_breaker.is_trading_allowed()
                    and self.drawdown.is_trading_allowed()
                ),
            },
            "drawdown": {
                "current": self.drawdown.get_current_drawdown(),
                "remaining_budget": self.drawdown.get_remaining_budget(),
                "max_drawdowns": self.drawdown.get_max_drawdowns(),
                "recovery": self.drawdown.get_recovery_progress(),
            },
            "exposure": self.exposure.get_snapshot().__dict__ if self._equity > 0 else {},
            "heat": self.heat.get_heat_map(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "correlation": {
                "regime": self.correlation.get_regime().value,
                "snapshot": self.correlation.get_snapshot().__dict__,
            },
            "stop_loss": self.stop_loss.get_summary(),
            "sizing": self.sizer.get_sizing_stats(),
            "checks": {
                "total": len(self._check_log),
                "approved": sum(
                    1 for c in self._check_log
                    if c.decision == RiskDecision.APPROVED
                ),
                "reduced": sum(
                    1 for c in self._check_log
                    if c.decision == RiskDecision.REDUCED
                ),
                "rejected": sum(
                    1 for c in self._check_log
                    if c.decision == RiskDecision.REJECTED
                ),
                "overrides": sum(1 for c in self._check_log if c.override),
            },
        }

        # Add metrics report if enough data
        if len(self.metrics._returns) >= 20:
            try:
                metrics_report = self.metrics.calculate_report(self._equity)
                report["metrics"] = {
                    "sharpe_ratio": metrics_report.sharpe_ratio,
                    "sortino_ratio": metrics_report.sortino_ratio,
                    "calmar_ratio": metrics_report.calmar_ratio,
                    "max_drawdown_pct": metrics_report.drawdown.max_drawdown_pct,
                    "win_rate": metrics_report.win_rate,
                    "profit_factor": metrics_report.profit_factor,
                    "expectancy": metrics_report.expectancy,
                    "var_95": metrics_report.var_historical.var_pct,
                    "volatility_annual": metrics_report.volatility_annual,
                }
            except Exception as exc:
                logger.error(f"Metrics report error: {exc}")
                report["metrics"] = {"error": str(exc)}

        return report

    def get_recent_checks(self, limit: int = 50) -> List[Dict]:
        """Get recent risk check results."""
        results = []
        for check in self._check_log[-limit:]:
            results.append({
                "timestamp": check.timestamp.isoformat(),
                "decision": check.decision.value,
                "original_qty": check.original_quantity,
                "approved_qty": check.approved_quantity,
                "reasons": check.reasons,
                "warnings": check.warnings,
                "checks_passed": check.checks_passed,
                "override": check.override,
            })
        return results

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def kill_switch(self) -> None:
        """Activate kill switch - immediately stop all trading."""
        self.circuit_breaker.activate_kill_switch()
        self.drawdown.force_pause("Kill switch activated")
        logger.critical("KILL SWITCH ACTIVATED via risk engine")

    def resume_trading(self) -> None:
        """Resume trading after manual intervention."""
        self.circuit_breaker.force_reset()
        self.drawdown.force_resume()
        logger.warning("Trading resumed via risk engine")

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed by all systems."""
        if not self._initialized:
            return False
        return (
            self.circuit_breaker.is_trading_allowed()
            and self.drawdown.is_trading_allowed()
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _log_check(self, result: RiskCheckResult) -> None:
        """Log a risk check result."""
        self._check_log.append(result)
        if len(self._check_log) > self._max_log:
            self._check_log = self._check_log[-self._max_log:]

    def to_dict(self) -> Dict:
        """Serialize full risk engine state."""
        return {
            "equity": self._equity,
            "initialized": self._initialized,
            "trading_allowed": self.is_trading_allowed(),
            "sizer": self.sizer.to_dict(),
            "drawdown": self.drawdown.to_dict(),
            "exposure": self.exposure.to_dict(),
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "correlation": self.correlation.to_dict(),
            "heat": self.heat.to_dict(),
            "stop_loss": self.stop_loss.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    def save_state(self, path: str) -> None:
        """Save risk engine state to file."""
        try:
            state = self.to_dict()
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Risk engine state saved to {path}")
        except Exception as exc:
            logger.error(f"Failed to save risk state: {exc}")

    def load_state(self, path: str) -> bool:
        """Load risk engine state from file."""
        try:
            with open(path, "r") as f:
                state = json.load(f)

            self._equity = state.get("equity", 0.0)
            self._initialized = state.get("initialized", False)

            if "drawdown" in state:
                self.drawdown = DrawdownManager.from_dict(state["drawdown"])
            if "circuit_breaker" in state:
                self.circuit_breaker = CircuitBreaker.from_dict(state["circuit_breaker"])
            if "sizer" in state:
                self.sizer = PositionSizer.from_dict(state["sizer"])

            logger.info(f"Risk engine state loaded from {path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load risk state: {exc}")
            return False
