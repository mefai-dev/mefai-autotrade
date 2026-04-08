"""
Tests for risk management modules: position sizing, drawdown checks,
correlation, and Kelly criterion.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from risk.position_sizer import (
    PositionSizer, SizingConfig, SizingMethod, SizingResult,
    PortfolioSnapshot, TradeRecord,
)
from risk.drawdown_manager import (
    DrawdownManager, DrawdownConfig, DrawdownState, DrawdownPeriod,
)
from risk.risk_engine import (
    RiskEngine, RiskEngineConfig, RiskDecision, RiskCheckResult,
)


# ===================================================================
# PositionSizer tests
# ===================================================================

class TestPositionSizer:

    def _make_portfolio(self, equity=10000.0):
        return PortfolioSnapshot(
            equity=equity,
            available_balance=equity * 0.9,
            total_unrealized_pnl=0.0,
        )

    def test_fixed_fractional_basic(self):
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=1.0,
        )
        sizer = PositionSizer(config)
        portfolio = self._make_portfolio(10000)

        result = sizer.calculate_size(
            symbol="BTCUSDT",
            side="LONG",
            entry_price=50000,
            stop_loss_price=49000,
            portfolio=portfolio,
        )
        # Risk = 1% of 10000 = 100. Per unit risk = 1000.
        # qty = 100 / 1000 = 0.1
        assert result.approved
        assert pytest.approx(result.quantity, abs=0.01) == 0.1

    def test_fixed_fractional_zero_stop(self):
        config = SizingConfig(method=SizingMethod.FIXED_FRACTIONAL)
        sizer = PositionSizer(config)
        portfolio = self._make_portfolio()
        result = sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=50000,
            portfolio=portfolio,
        )
        # Zero risk per unit should result in 0 qty or rejection
        assert result.quantity == 0 or not result.approved

    def test_kelly_sizing(self):
        config = SizingConfig(
            method=SizingMethod.KELLY,
            win_rate=0.6,
            avg_win=1.5,
            kelly_fraction=1.0,
        )
        sizer = PositionSizer(config)
        portfolio = self._make_portfolio()

        result = sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=49000,
            portfolio=portfolio,
        )
        assert result.approved
        assert result.method_used == SizingMethod.KELLY

    def test_half_kelly_smaller_than_full(self):
        full_config = SizingConfig(
            method=SizingMethod.KELLY,
            win_rate=0.6, avg_win=1.5, kelly_fraction=1.0,
        )
        half_config = SizingConfig(
            method=SizingMethod.HALF_KELLY,
            win_rate=0.6, avg_win=1.5, kelly_fraction=0.5,
        )
        portfolio = self._make_portfolio()

        full_sizer = PositionSizer(full_config)
        half_sizer = PositionSizer(half_config)

        full_result = full_sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=49000,
            portfolio=portfolio,
        )
        half_result = half_sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=49000,
            portfolio=portfolio,
        )
        assert half_result.quantity <= full_result.quantity

    def test_volatility_sizing_with_atr(self):
        config = SizingConfig(
            method=SizingMethod.VOLATILITY,
            target_risk_pct=1.0,
            atr_multiplier=2.0,
        )
        sizer = PositionSizer(config)
        portfolio = self._make_portfolio()

        result = sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=49000,
            portfolio=portfolio,
            atr=500.0,
        )
        assert result.approved
        assert result.quantity > 0

    def test_max_position_limit(self):
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=5.0,  # aggressive
            max_position_per_symbol=500.0,  # max notional 500
        )
        sizer = PositionSizer(config)
        portfolio = self._make_portfolio(100000)

        result = sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=49000,
            portfolio=portfolio,
        )
        # Max notional 500 at price 50000 = 0.01 max qty
        assert result.quantity <= 500 / 50000 + 0.001

    def test_min_notional_rejection(self):
        config = SizingConfig(
            method=SizingMethod.FIXED_FRACTIONAL,
            risk_per_trade_pct=0.001,  # tiny risk
            min_notional=10.0,
        )
        sizer = PositionSizer(config)
        portfolio = self._make_portfolio(100)  # small account

        result = sizer.calculate_size(
            symbol="BTCUSDT", side="LONG",
            entry_price=50000, stop_loss_price=49000,
            portfolio=portfolio,
        )
        # Quantity would be tiny. Should be rejected or zero
        if result.approved:
            notional = result.quantity * 50000
            assert notional >= config.min_notional or result.quantity == 0


# ===================================================================
# DrawdownManager tests
# ===================================================================

class TestDrawdownManager:

    def test_initial_state(self):
        dm = DrawdownManager()
        dm.initialize(10000)
        assert dm.get_state() == DrawdownState.NORMAL
        assert dm.is_trading_allowed()

    def test_drawdown_tracking(self):
        dm = DrawdownManager(DrawdownConfig(daily_limit_pct=5.0))
        dm.initialize(10000)

        snap = dm.update(9500)
        dd = dm.get_current_drawdown()
        assert dd["total"] > 0
        assert snap.drawdown_pct > 0
        assert snap.equity == 9500

    def test_new_peak_resets_drawdown(self):
        dm = DrawdownManager()
        dm.initialize(10000)
        dm.update(9500)  # drawdown
        dm.update(10500)  # new peak
        snap = dm.get_snapshot()
        assert snap.drawdown_pct == 0.0

    def test_daily_limit_breach_pauses(self):
        config = DrawdownConfig(
            daily_limit_pct=3.0,
            auto_pause_on_limit=True,
            cooldown_minutes=0,
        )
        dm = DrawdownManager(config)
        dm.initialize(10000)

        # Drop 4% (exceeds 3% daily limit)
        dm.update(9600)
        assert dm.get_state() in (DrawdownState.PAUSED, DrawdownState.RECOVERY)
        if dm.get_state() == DrawdownState.PAUSED:
            assert not dm.is_trading_allowed()

    def test_recovery_mode_multiplier(self):
        config = DrawdownConfig(
            daily_limit_pct=5.0,
            enable_recovery_mode=True,
            recovery_threshold_pct=50.0,
            recovery_size_multiplier=0.5,
        )
        dm = DrawdownManager(config)
        dm.initialize(10000)

        # Drop to 50% of daily limit (2.5% dd)
        dm.update(9750)
        multiplier = dm.get_size_multiplier()
        # Should be in recovery or normal depending on exact threshold
        assert multiplier <= 1.0

    def test_force_resume(self):
        config = DrawdownConfig(daily_limit_pct=1.0, auto_pause_on_limit=True)
        dm = DrawdownManager(config)
        dm.initialize(10000)
        dm.update(9800)  # 2% dd > 1% limit
        dm.force_resume()
        assert dm.is_trading_allowed()
        assert dm.get_state() == DrawdownState.NORMAL

    def test_force_pause(self):
        dm = DrawdownManager()
        dm.initialize(10000)
        dm.force_pause("manual test")
        assert not dm.is_trading_allowed()

    def test_remaining_budget(self):
        config = DrawdownConfig(daily_limit_pct=5.0)
        dm = DrawdownManager(config)
        dm.initialize(10000)
        dm.update(9800)  # 2% dd
        budget = dm.get_remaining_budget()
        assert budget["daily"] > 0
        assert budget["daily"] < 5.0

    def test_serialization_roundtrip(self):
        dm = DrawdownManager(DrawdownConfig(daily_limit_pct=4.0))
        dm.initialize(10000)
        dm.update(9600)

        state = dm.to_dict()
        restored = DrawdownManager.from_dict(state)
        assert restored._current_equity == dm._current_equity
        assert restored._global_peak_equity == dm._global_peak_equity

    def test_equity_history(self):
        dm = DrawdownManager()
        dm.initialize(10000)
        dm.update(9800)
        dm.update(9900)
        dm.update(10100)
        history = dm.get_equity_history()
        assert len(history) == 4  # init + 3 updates


# ===================================================================
# RiskEngine integration tests
# ===================================================================

class TestRiskEngine:

    def _make_engine(self, **config_overrides):
        config = RiskEngineConfig(
            enable_correlation_monitor=False,  # skip for unit tests
            enable_exposure_limits=False,  # skip concentration checks in unit tests
            enable_portfolio_heat=False,  # skip heat checks in unit tests
            **config_overrides,
        )
        engine = RiskEngine(config)
        engine.initialize(10000.0)
        return engine

    def test_not_initialized_rejects(self):
        engine = RiskEngine()
        result = engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG",
            quantity=0.1, entry_price=50000,
            stop_loss_price=49000,
        )
        assert result.decision == RiskDecision.REJECTED
        assert "not initialized" in result.reasons[0].lower()

    def test_basic_approval(self):
        engine = self._make_engine()
        result = engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, entry_price=50000,
            stop_loss_price=49000,
        )
        assert result.decision in (RiskDecision.APPROVED, RiskDecision.REDUCED)
        assert result.approved_quantity > 0

    def test_override_disabled_rejects(self):
        engine = self._make_engine(allow_overrides=False)
        result = engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG",
            quantity=0.1, entry_price=50000,
            stop_loss_price=49000,
            override=True, override_reason="test",
        )
        assert result.decision == RiskDecision.REJECTED

    def test_override_enabled_approves(self):
        engine = self._make_engine(allow_overrides=True)
        result = engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, entry_price=50000,
            stop_loss_price=49000,
            override=True, override_reason="manual entry",
        )
        assert result.decision in (RiskDecision.APPROVED, RiskDecision.REDUCED)
        assert result.override is True

    def test_override_without_reason_rejects(self):
        engine = self._make_engine(
            allow_overrides=True,
            override_requires_reason=True,
        )
        result = engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, entry_price=50000,
            stop_loss_price=49000,
            override=True, override_reason="",
        )
        assert result.decision == RiskDecision.REJECTED

    def test_kill_switch(self):
        engine = self._make_engine()
        assert engine.is_trading_allowed()
        engine.kill_switch()
        assert not engine.is_trading_allowed()

    def test_resume_after_kill(self):
        engine = self._make_engine()
        engine.kill_switch()
        engine.resume_trading()
        assert engine.is_trading_allowed()

    def test_post_trade_update(self):
        engine = self._make_engine()
        # Should not crash
        engine.post_trade_update(
            symbol="BTCUSDT", pnl=100, pnl_pct=1.0,
        )

    def test_update_equity(self):
        engine = self._make_engine()
        alerts = engine.update_equity(10500)
        assert isinstance(alerts, dict)
        assert engine._equity == 10500

    def test_risk_report_generation(self):
        engine = self._make_engine()
        report = engine.generate_risk_report()
        assert "equity" in report
        assert report["equity"] == 10000
        assert "status" in report
        assert "drawdown" in report

    def test_recent_checks(self):
        engine = self._make_engine()
        engine.pre_trade_check(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, entry_price=50000,
            stop_loss_price=49000,
        )
        checks = engine.get_recent_checks(10)
        assert len(checks) >= 1
        assert "decision" in checks[0]
