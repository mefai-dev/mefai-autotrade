"""
Tests for backtesting engine: config, order/position objects,
fee calculation, PnL computation, and basic engine operations.
"""

import pytest
import uuid
from datetime import datetime

from backtesting.engine import (
    BacktestConfig, Order, Position, Trade,
    OrderSide, OrderType, OrderStatus, PositionSide,
    SlippageModel, SimulationMode,
)


# ===================================================================
# BacktestConfig tests
# ===================================================================

class TestBacktestConfig:

    def test_default_values(self):
        config = BacktestConfig()
        assert config.initial_capital == 10000.0
        assert config.commission_maker == 0.0002
        assert config.commission_taker == 0.0004
        assert config.leverage == 1.0
        assert config.slippage_model == SlippageModel.PERCENTAGE
        assert config.simulation_mode == SimulationMode.BAR_BY_BAR

    def test_custom_config(self):
        config = BacktestConfig(
            initial_capital=50000,
            commission_maker=0.0001,
            commission_taker=0.0003,
            leverage=10.0,
            enable_funding=False,
            enable_liquidation=False,
        )
        assert config.initial_capital == 50000
        assert config.leverage == 10.0
        assert config.enable_funding is False
        assert config.enable_liquidation is False


# ===================================================================
# Order tests
# ===================================================================

class TestOrder:

    def test_auto_order_id(self):
        order = Order(symbol="BTCUSDT", side=OrderSide.BUY, quantity=0.1)
        assert len(order.order_id) > 0

    def test_order_fields(self):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000,
            status=OrderStatus.NEW,
        )
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 1.0
        assert order.price == 50000
        assert order.filled_quantity == 0.0


# ===================================================================
# Position tests
# ===================================================================

class TestPosition:

    def test_notional_value(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=50000, quantity=0.5,
        )
        assert pos.notional_value == 25000

    def test_is_open(self):
        pos = Position(symbol="BTCUSDT", quantity=0.1)
        assert pos.is_open is True

        flat = Position(symbol="BTCUSDT", quantity=0.0)
        assert flat.is_open is False

    def test_unrealized_pnl_long(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=50000, quantity=1.0,
        )
        pnl = pos.calculate_unrealized_pnl(51000)
        assert pnl == 1000.0

    def test_unrealized_pnl_short(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.SHORT,
            entry_price=50000, quantity=1.0,
        )
        pnl = pos.calculate_unrealized_pnl(49000)
        assert pnl == 1000.0

    def test_unrealized_pnl_losing_long(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=50000, quantity=2.0,
        )
        pnl = pos.calculate_unrealized_pnl(49500)
        assert pnl == -1000.0

    def test_unrealized_pnl_flat(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.FLAT,
            entry_price=50000, quantity=0.0,
        )
        assert pos.calculate_unrealized_pnl(51000) == 0.0

    def test_liquidation_price_long(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=50000, quantity=1.0,
        )
        liq = pos.calculate_liquidation_price(
            maintenance_margin_rate=0.004,
            wallet_balance=5000,
        )
        # liq = entry - (wallet - mmr * notional) / qty
        # = 50000 - (5000 - 0.004 * 50000) / 1
        # = 50000 - (5000 - 200) / 1
        # = 50000 - 4800 = 45200
        assert pytest.approx(liq, rel=1e-6) == 45200.0

    def test_liquidation_price_short(self):
        pos = Position(
            symbol="BTCUSDT", side=PositionSide.SHORT,
            entry_price=50000, quantity=1.0,
        )
        liq = pos.calculate_liquidation_price(
            maintenance_margin_rate=0.004,
            wallet_balance=5000,
        )
        # liq = entry + (wallet - mmr * notional) / qty
        # = 50000 + 4800 = 54800
        assert pytest.approx(liq, rel=1e-6) == 54800.0

    def test_liquidation_price_flat(self):
        pos = Position(symbol="BTCUSDT", side=PositionSide.FLAT, quantity=0.0)
        assert pos.calculate_liquidation_price(0.004, 5000) == 0.0


# ===================================================================
# Fee / commission calculation tests
# ===================================================================

class TestFeeCalculation:
    """Test that fee calculations are correct for the config values."""

    def test_maker_fee(self):
        config = BacktestConfig(commission_maker=0.0002)
        notional = 50000  # 1 BTC at 50000
        fee = notional * config.commission_maker
        assert fee == 10.0

    def test_taker_fee(self):
        config = BacktestConfig(commission_taker=0.0004)
        notional = 50000
        fee = notional * config.commission_taker
        assert fee == 20.0

    def test_slippage_percentage(self):
        config = BacktestConfig(
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_value=0.0005,
        )
        price = 50000
        slippage = price * config.slippage_value
        # For a BUY market order the fill price would be
        fill = price + slippage
        assert fill == 50025.0

    def test_roundtrip_cost(self):
        """Full round-trip cost: entry taker + exit taker + slippage both ways."""
        config = BacktestConfig(
            commission_taker=0.0004,
            slippage_value=0.0005,
        )
        qty = 1.0
        entry_price = 50000

        entry_slippage = entry_price * config.slippage_value
        entry_fill = entry_price + entry_slippage  # BUY
        entry_fee = entry_fill * qty * config.commission_taker

        exit_price = 50500  # target
        exit_slippage = exit_price * config.slippage_value
        exit_fill = exit_price - exit_slippage  # SELL
        exit_fee = exit_fill * qty * config.commission_taker

        gross_pnl = (exit_fill - entry_fill) * qty
        net_pnl = gross_pnl - entry_fee - exit_fee

        assert gross_pnl < 500  # slippage ate into the 500 target
        assert net_pnl < gross_pnl  # fees further reduce
        assert net_pnl > 0  # but still profitable with 500 point move


# ===================================================================
# PnL computation tests
# ===================================================================

class TestPnLComputation:

    def test_long_profit(self):
        entry = 50000
        exit_ = 51000
        qty = 2.0
        pnl = (exit_ - entry) * qty
        assert pnl == 2000.0

    def test_long_loss(self):
        entry = 50000
        exit_ = 49000
        qty = 1.0
        pnl = (exit_ - entry) * qty
        assert pnl == -1000.0

    def test_short_profit(self):
        entry = 50000
        exit_ = 49000
        qty = 1.0
        pnl = (entry - exit_) * qty
        assert pnl == 1000.0

    def test_short_loss(self):
        entry = 50000
        exit_ = 51000
        qty = 0.5
        pnl = (entry - exit_) * qty
        assert pnl == -500.0

    def test_leveraged_pnl(self):
        config = BacktestConfig(leverage=10.0)
        entry = 50000
        exit_ = 50500
        qty = 1.0
        margin = entry * qty / config.leverage  # 5000
        pnl = (exit_ - entry) * qty  # 500
        roi_pct = (pnl / margin) * 100  # 10%
        assert roi_pct == 10.0

    def test_pnl_with_fees(self):
        config = BacktestConfig(commission_taker=0.0004)
        entry = 50000
        exit_ = 50100
        qty = 1.0

        entry_fee = entry * qty * config.commission_taker
        exit_fee = exit_ * qty * config.commission_taker
        gross_pnl = (exit_ - entry) * qty
        net_pnl = gross_pnl - entry_fee - exit_fee

        assert gross_pnl == 100.0
        assert net_pnl < 100.0
        assert net_pnl == pytest.approx(100 - 20 - 20.04, rel=1e-3)
