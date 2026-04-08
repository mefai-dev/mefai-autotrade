"""
Tests for execution engine components: OrderBuilder, TWAP/VWAP splitting,
order validation, and ExecutionResult.
"""

import pytest
import math
import time
from unittest.mock import MagicMock, AsyncMock, patch

from execution.order_builder import ExchangeRules, OrderParams
from execution.twap import TWAPConfig, TWAPSlice, TWAPProgress, TWAPStatus
from execution.vwap import VolumeProfile, VWAPConfig, VWAPStatus
from execution.execution_engine import (
    ExecutionRequest, ExecutionResult, ExecutionStatus,
    ExecutionAlgorithm, EngineMetrics,
)
from execution.fill_analyzer import FillRecord


# ===================================================================
# ExchangeRules tests
# ===================================================================

class TestExchangeRules:

    def _make_rules(self, **overrides):
        defaults = dict(
            symbol="BTCUSDT",
            tick_size=0.01,
            min_price=0.01,
            max_price=1000000,
            lot_size=0.001,
            min_quantity=0.001,
            max_quantity=1000,
            min_notional=5.0,
        )
        defaults.update(overrides)
        return ExchangeRules(**defaults)

    def test_round_price(self):
        rules = self._make_rules(tick_size=0.01)
        assert rules.round_price(50000.123) == 50000.12
        assert rules.round_price(50000.005) == 50000.0  # rounds to nearest tick

    def test_round_quantity_down(self):
        rules = self._make_rules(lot_size=0.001)
        assert rules.round_quantity(0.1234) == 0.123  # floors
        assert rules.round_quantity(0.0001) == 0.0    # below lot size

    def test_round_quantity_up(self):
        rules = self._make_rules(lot_size=0.001)
        assert rules.round_quantity_up(0.1234) == 0.124

    def test_validate_price_valid(self):
        rules = self._make_rules()
        valid, msg = rules.validate_price(50000.01)
        assert valid is True

    def test_validate_price_negative(self):
        rules = self._make_rules()
        valid, msg = rules.validate_price(-1)
        assert valid is False

    def test_validate_price_below_min(self):
        rules = self._make_rules(min_price=10.0)
        valid, msg = rules.validate_price(5.0)
        assert valid is False
        assert "below minimum" in msg

    def test_validate_price_above_max(self):
        rules = self._make_rules(max_price=100000)
        valid, msg = rules.validate_price(200000)
        assert valid is False

    def test_validate_quantity_valid(self):
        rules = self._make_rules()
        valid, msg = rules.validate_quantity(0.1)
        assert valid is True

    def test_validate_quantity_below_min(self):
        rules = self._make_rules(min_quantity=0.01)
        valid, msg = rules.validate_quantity(0.001)
        assert valid is False

    def test_validate_quantity_negative(self):
        rules = self._make_rules()
        valid, msg = rules.validate_quantity(-0.1)
        assert valid is False

    def test_validate_notional_below_min(self):
        rules = self._make_rules(min_notional=10.0)
        valid, msg = rules.validate_notional(0.0001, 50000)
        # 0.0001 * 50000 = 5 < 10
        assert valid is False

    def test_validate_notional_valid(self):
        rules = self._make_rules(min_notional=5.0)
        valid, msg = rules.validate_notional(0.001, 50000)
        # 0.001 * 50000 = 50 > 5
        assert valid is True

    def test_validate_order_full(self):
        rules = self._make_rules()
        valid, errors = rules.validate_order("BUY", "LIMIT", 0.1, 50000)
        assert valid is True
        assert len(errors) == 0

    def test_validate_order_invalid_side(self):
        rules = self._make_rules()
        valid, errors = rules.validate_order("HOLD", "LIMIT", 0.1, 50000)
        assert valid is False
        assert any("side" in e.lower() for e in errors)

    def test_validate_order_unsupported_type(self):
        rules = self._make_rules()
        valid, errors = rules.validate_order("BUY", "TRAILING_STOP", 0.1, 50000)
        assert valid is False

    def test_count_decimals(self):
        assert ExchangeRules._count_decimals(0.01) == 2
        assert ExchangeRules._count_decimals(0.001) == 3
        assert ExchangeRules._count_decimals(1.0) == 0  # "1.0000..." strips to "1." -> 0
        assert ExchangeRules._count_decimals(0.00001) == 5


# ===================================================================
# TWAPConfig tests
# ===================================================================

class TestTWAPConfig:

    def test_auto_slice_count(self):
        config = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=300,
            num_slices=0,  # auto
        )
        # 300 / 30 = 10 slices
        assert config.num_slices == 10

    def test_min_slices(self):
        config = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=10,
            num_slices=0,
        )
        assert config.num_slices >= 2

    def test_slice_interval(self):
        config = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=600,
            num_slices=10,
        )
        assert config.slice_interval == 60.0

    def test_base_slice_quantity(self):
        config = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=300,
            num_slices=5,
        )
        assert config.base_slice_quantity == 0.2

    def test_jitter_clamping(self):
        config = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=300,
            jitter_pct=0.8,
        )
        assert config.jitter_pct == 0.5  # clamped

        config2 = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=300,
            jitter_pct=-0.1,
        )
        assert config2.jitter_pct == 0


class TestTWAPSlice:

    def test_fill_pct(self):
        s = TWAPSlice(slice_id="s1", index=0, scheduled_time=100,
                      target_quantity=1.0, filled_quantity=0.5)
        assert s.fill_pct == 0.5

    def test_fill_pct_zero_target(self):
        s = TWAPSlice(slice_id="s1", index=0, scheduled_time=100,
                      target_quantity=0, filled_quantity=0)
        assert s.fill_pct == 0.0

    def test_notional_value(self):
        s = TWAPSlice(slice_id="s1", index=0, scheduled_time=100,
                      filled_quantity=0.5, fill_price=50000)
        assert s.notional_value == 25000


class TestTWAPProgress:

    def test_update_metrics(self):
        config = TWAPConfig(
            symbol="BTCUSDT", side="BUY",
            total_quantity=1.0, duration_seconds=300,
            num_slices=5,
        )
        slices = [
            TWAPSlice(slice_id=f"s{i}", index=i, scheduled_time=100 + i * 60,
                      target_quantity=0.2, filled_quantity=0.2,
                      fill_price=50000 + i * 10, status="filled")
            for i in range(3)
        ]
        progress = TWAPProgress(
            execution_id="test1", config=config,
            status=TWAPStatus.RUNNING, slices=slices,
            start_time=time.time() - 200,
        )
        progress.update()
        assert progress.total_filled == pytest.approx(0.6, rel=1e-9)
        assert progress.slices_completed == 3
        assert progress.fill_pct == pytest.approx(0.6, rel=1e-6)
        assert progress.avg_fill_price > 0


# ===================================================================
# VolumeProfile tests
# ===================================================================

class TestVolumeProfile:

    def test_flat_profile(self):
        vp = VolumeProfile(symbol="BTCUSDT", interval_minutes=30)
        frac = vp.get_volume_fraction(720)
        # Flat profile: 1 / (1440/30) = 1/48
        assert pytest.approx(frac, rel=1e-6) == 1.0 / 48

    def test_from_hourly_volumes(self):
        hourly = [100] * 24
        vp = VolumeProfile.from_hourly_volumes("BTCUSDT", hourly)
        assert len(vp.profile) == 48  # 24 hours * 2 (30-min intervals)
        # Should sum to ~1.0
        total = sum(frac for _, frac in vp.profile)
        assert pytest.approx(total, rel=1e-6) == 1.0

    def test_peak_hours(self):
        hourly = [100] * 24
        hourly[10] = 500  # peak at hour 10
        hourly[14] = 400  # second peak at hour 14
        vp = VolumeProfile.from_hourly_volumes("BTCUSDT", hourly)
        peaks = vp.get_peak_hours(2)
        assert len(peaks) == 2
        # Peak minutes should include 600 (hour 10*60)
        peak_minutes = [p[0] for p in peaks]
        assert 600 in peak_minutes or 630 in peak_minutes

    def test_volume_between(self):
        hourly = [100] * 24
        vp = VolumeProfile.from_hourly_volumes("BTCUSDT", hourly)
        vol = vp.get_volume_between(0, 1440)
        assert pytest.approx(vol, rel=0.01) == 1.0  # full day

        half = vp.get_volume_between(0, 720)
        assert pytest.approx(half, rel=0.05) == 0.5  # half day


# ===================================================================
# ExecutionRequest / ExecutionResult tests
# ===================================================================

class TestExecutionRequest:

    def test_auto_request_id(self):
        req = ExecutionRequest(symbol="BTCUSDT", side="BUY", quantity=1.0)
        assert req.request_id.startswith("exec_")
        assert len(req.request_id) > 5

    def test_explicit_request_id(self):
        req = ExecutionRequest(
            request_id="custom_123",
            symbol="BTCUSDT", side="BUY", quantity=1.0,
        )
        assert req.request_id == "custom_123"


class TestExecutionResult:

    def _make_fill(self, symbol="BTCUSDT", side="BUY",
                   filled_quantity=0.5, fill_price=50000,
                   commission=5.0, **kwargs):
        """Helper to create FillRecord with required fields."""
        return FillRecord(
            fill_id="f1", order_id="o1",
            symbol=symbol, side=side,
            order_type="MARKET",
            requested_quantity=filled_quantity,
            filled_quantity=filled_quantity,
            requested_price=fill_price,
            fill_price=fill_price,
            timestamp=time.time(),
            commission=commission,
            **kwargs,
        )

    def test_update_from_fills(self):
        req = ExecutionRequest(
            symbol="BTCUSDT", side="BUY",
            quantity=1.0, price=50000,
        )
        result = ExecutionResult(request_id=req.request_id, request=req)
        result.fills = [
            self._make_fill(filled_quantity=0.5, fill_price=50000, commission=5.0),
            self._make_fill(filled_quantity=0.5, fill_price=50010, commission=5.0),
        ]
        result.update_from_fills()
        assert result.filled_quantity == 1.0
        assert result.total_commission == 10.0
        assert result.avg_fill_price > 0
        assert result.fill_rate == 1.0

    def test_slippage_calculation_buy(self):
        req = ExecutionRequest(
            symbol="BTCUSDT", side="BUY",
            quantity=1.0, price=50000,
        )
        result = ExecutionResult(request_id=req.request_id, request=req)
        result.fills = [
            self._make_fill(filled_quantity=1.0, fill_price=50050, commission=10.0),
        ]
        result.update_from_fills()
        # Slippage for BUY = (fill - expected) / expected * 10000 bps
        assert result.slippage_bps == pytest.approx(10.0, rel=0.01)

    def test_is_complete(self):
        req = ExecutionRequest(symbol="BTCUSDT", side="BUY", quantity=1.0)
        result = ExecutionResult(
            request_id=req.request_id, request=req,
            status=ExecutionStatus.FILLED,
        )
        assert result.is_complete is True

        result.status = ExecutionStatus.EXECUTING
        assert result.is_complete is False
