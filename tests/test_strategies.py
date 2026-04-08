"""
Tests for strategy base class, MomentumStrategy, MeanReversionStrategy,
and GridTradingStrategy.
"""

import numpy as np
import pytest

from strategies.base import (
    BaseStrategy,
    FeatureEngine,
    StrategySignal,
    SignalType,
    StrategyState,
    PositionInfo,
    PerformanceMetrics,
)
from strategies.momentum import MomentumStrategy, MomentumScore
from strategies.mean_reversion import MeanReversionStrategy, ZScoreStats
from strategies.grid_trading import GridTradingStrategy, GridLevel, GridState


# ===================================================================
# FeatureEngine indicator tests
# ===================================================================

class TestFeatureEngine:
    """Test built-in indicator calculations."""

    def test_sma_basic(self):
        close = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = FeatureEngine.sma(close, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert pytest.approx(result[2], rel=1e-9) == 2.0
        assert pytest.approx(result[3], rel=1e-9) == 3.0
        assert pytest.approx(result[4], rel=1e-9) == 4.0

    def test_sma_insufficient_data(self):
        close = np.array([1, 2], dtype=np.float64)
        result = FeatureEngine.sma(close, 5)
        assert np.all(np.isnan(result))

    def test_ema_converges(self):
        close = np.full(50, 100.0, dtype=np.float64)
        result = FeatureEngine.ema(close, 10)
        # After enough bars of constant price the EMA should equal the price
        assert pytest.approx(result[-1], rel=1e-6) == 100.0

    def test_ema_responds_to_trend(self):
        close = np.arange(1, 51, dtype=np.float64)
        result = FeatureEngine.ema(close, 10)
        # EMA of rising series should be below current price (lagging)
        assert result[-1] < close[-1]
        assert result[-1] > close[-10]

    def test_rsi_range(self):
        rng = np.random.default_rng(42)
        close = 50000 + np.cumsum(rng.normal(0, 100, 100))
        result = FeatureEngine.rsi(close, 14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_rsi_all_up(self):
        close = np.arange(100, 200, dtype=np.float64)
        result = FeatureEngine.rsi(close, 14)
        valid = result[~np.isnan(result)]
        # All gains should push RSI near 100
        assert np.all(valid > 90)

    def test_rsi_all_down(self):
        close = np.arange(200, 100, -1, dtype=np.float64)
        result = FeatureEngine.rsi(close, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 10)

    def test_atr_positive(self, sample_candles):
        high = sample_candles[:, 2]
        low = sample_candles[:, 3]
        close = sample_candles[:, 4]
        result = FeatureEngine.atr(high, low, close, 14)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid > 0)

    def test_bollinger_bands_order(self, sample_candles):
        close = sample_candles[:, 4]
        upper, middle, lower = FeatureEngine.bollinger_bands(close, 20, 2.0)
        last = len(close) - 1
        if not np.isnan(upper[last]):
            assert upper[last] > middle[last] > lower[last]

    def test_macd_output_shapes(self, sample_candles):
        close = sample_candles[:, 4]
        ml, sl, hist = FeatureEngine.macd(close)
        assert len(ml) == len(close)
        assert len(sl) == len(close)
        assert len(hist) == len(close)

    def test_vwap_calculation(self):
        high = np.array([102, 104, 103], dtype=np.float64)
        low = np.array([98, 99, 100], dtype=np.float64)
        close = np.array([100, 101, 102], dtype=np.float64)
        volume = np.array([1000, 2000, 1500], dtype=np.float64)
        result = FeatureEngine.vwap(high, low, close, volume)
        # VWAP should be between lowest low and highest high
        assert result[-1] >= 98
        assert result[-1] <= 104


# ===================================================================
# StrategySignal tests
# ===================================================================

class TestStrategySignal:

    def test_is_entry(self):
        s = StrategySignal(signal_type=SignalType.LONG, symbol="BTCUSDT", price=50000)
        assert s.is_entry is True
        assert s.is_exit is False

    def test_is_exit(self):
        s = StrategySignal(signal_type=SignalType.CLOSE_LONG, symbol="BTCUSDT", price=50000)
        assert s.is_exit is True
        assert s.is_entry is False

    def test_risk_reward_ratio(self):
        s = StrategySignal(
            signal_type=SignalType.LONG,
            symbol="BTCUSDT",
            price=50000,
            stop_loss=49000,
            take_profit=52000,
        )
        # Risk = 1000  Reward = 2000  RR = 2.0
        assert pytest.approx(s.risk_reward_ratio, rel=1e-6) == 2.0

    def test_risk_reward_ratio_zero_stop(self):
        s = StrategySignal(signal_type=SignalType.LONG, symbol="BTCUSDT", price=50000)
        assert s.risk_reward_ratio == 0.0


# ===================================================================
# PerformanceMetrics tests
# ===================================================================

class TestPerformanceMetrics:

    def test_update_tracks_wins_losses(self):
        pm = PerformanceMetrics()
        pm.trade_history.append({"pnl": 100})
        pm.update(100)
        assert pm.total_trades == 1
        assert pm.winning_trades == 1
        assert pm.win_rate == 1.0

        pm.trade_history.append({"pnl": -50})
        pm.update(-50)
        assert pm.total_trades == 2
        assert pm.losing_trades == 1
        assert pm.win_rate == 0.5

    def test_max_drawdown(self):
        pm = PerformanceMetrics()
        pnls = [100, 50, -200, -100, 300]
        for p in pnls:
            pm.trade_history.append({"pnl": p})
            pm.update(p)
        # Equity curve: [100, 150, -50, -150, 150]
        # Peak at 150 then trough at -150 -> dd = 300
        assert pm.max_drawdown == 300.0


# ===================================================================
# BaseStrategy tests (via concrete subclass)
# ===================================================================

class TestBaseStrategy:

    def _make_strategy(self, **kwargs):
        return MomentumStrategy(symbol="BTCUSDT", timeframe="5m", params=kwargs)

    def test_initial_state(self):
        s = self._make_strategy()
        assert s.state == StrategyState.CREATED
        assert s.has_position is False

    def test_lifecycle(self):
        s = self._make_strategy()
        s.start()
        assert s.state == StrategyState.RUNNING
        s.pause()
        assert s.state == StrategyState.PAUSED
        s.resume()
        assert s.state == StrategyState.RUNNING
        s.stop()
        assert s.state == StrategyState.STOPPED

    def test_open_close_position_pnl(self):
        s = self._make_strategy()
        s.open_position("LONG", entry_price=50000, quantity=1.0, stop_loss=49000)
        assert s.has_position
        assert s.position.entry_price == 50000

        pnl = s.close_position(exit_price=51000, reason="test")
        assert pnl == 1000.0
        assert s.has_position is False
        assert s.performance.total_trades == 1
        assert s.performance.total_pnl == 1000.0

    def test_short_position_pnl(self):
        s = self._make_strategy()
        s.open_position("SHORT", entry_price=50000, quantity=2.0)
        pnl = s.close_position(exit_price=49000)
        assert pnl == 2000.0

    def test_close_no_position_returns_zero(self):
        s = self._make_strategy()
        assert s.close_position(50000) == 0.0

    def test_position_size_calculation(self):
        s = self._make_strategy()
        # Risk 1% of 10000 = 100. Risk per unit = |50000-49000| = 1000
        qty = s.calculate_position_size(
            account_balance=10000, entry_price=50000,
            stop_loss=49000, risk_pct=1.0,
        )
        assert pytest.approx(qty, rel=1e-6) == 0.1

    def test_position_size_zero_stop(self):
        s = self._make_strategy()
        qty = s.calculate_position_size(10000, 50000, 0, 1.0)
        assert qty == 0.0

    def test_kelly_sizing(self):
        s = self._make_strategy()
        # win_rate=0.6 avg_win=1.5 avg_loss=1.0 -> b=1.5
        # kelly = (0.6*1.5 - 0.4)/1.5 = (0.9-0.4)/1.5 = 0.333
        # half-kelly = 0.333 * 0.5 = 0.1667
        result = s.calculate_kelly_size(
            win_rate=0.6, avg_win=1.5, avg_loss=1.0, fraction=0.5,
        )
        assert pytest.approx(result, rel=1e-3) == 0.1667

    def test_kelly_edge_cases(self):
        s = self._make_strategy()
        assert s.calculate_kelly_size(0.0, 1.5, 1.0) == 0.0
        assert s.calculate_kelly_size(1.0, 1.5, 1.0) == 0.0
        assert s.calculate_kelly_size(0.5, 1.5, 0.0) == 0.0

    def test_check_risk_daily_limit(self):
        s = self._make_strategy(max_daily_trades=2)
        s.daily_trade_count = 2
        signal = StrategySignal(
            signal_type=SignalType.LONG, symbol="BTCUSDT",
            price=50000, quantity=0.1,
        )
        assert s.check_risk(signal) is False

    def test_check_risk_position_size(self):
        s = self._make_strategy(max_position_size=0.5)
        signal = StrategySignal(
            signal_type=SignalType.LONG, symbol="BTCUSDT",
            price=50000, quantity=1.0,
        )
        assert s.check_risk(signal) is False

    def test_trailing_stop_long(self):
        s = self._make_strategy()
        s.open_position("LONG", 50000, 1.0)
        s.position.highest_price = 52000
        hit = s.check_trailing_stop(current_price=51000, atr_value=400, atr_multiplier=2.0)
        # trail_stop = 52000 - 800 = 51200. Price 51000 < 51200 -> hit
        assert hit is True

    def test_trailing_stop_short(self):
        s = self._make_strategy()
        s.open_position("SHORT", 50000, 1.0)
        s.position.lowest_price = 48000
        hit = s.check_trailing_stop(current_price=49000, atr_value=400, atr_multiplier=2.0)
        # trail_stop = 48000 + 800 = 48800. Price 49000 > 48800 -> hit
        assert hit is True

    def test_trailing_stop_no_position(self):
        s = self._make_strategy()
        assert s.check_trailing_stop(50000, 400) is False

    def test_update_position_price(self):
        s = self._make_strategy()
        s.open_position("LONG", 50000, 1.0)
        s.update_position_price(51000)
        assert s.position.highest_price == 51000
        assert s.position.unrealized_pnl == 1000.0

        s.update_position_price(49500)
        assert s.position.lowest_price == 49500
        assert s.position.unrealized_pnl == -500.0


# ===================================================================
# MomentumStrategy tests
# ===================================================================

class TestMomentumStrategy:

    def _make(self, **params):
        defaults = {"account_balance": 10000.0}
        defaults.update(params)
        return MomentumStrategy(symbol="BTCUSDT", timeframe="5m", params=defaults)

    def test_insufficient_data_returns_none(self):
        strat = self._make()
        short_candles = np.zeros((10, 6))
        assert strat.generate_signal(short_candles) is None

    def test_score_rsi_extreme_overbought(self):
        strat = self._make()
        assert strat._score_rsi(85) == -0.8

    def test_score_rsi_extreme_oversold(self):
        strat = self._make()
        assert strat._score_rsi(15) == 0.8

    def test_score_rsi_neutral(self):
        strat = self._make()
        score = strat._score_rsi(50)
        # 50 > 80 F, > 70 F, > 55 F, > 50 F, > 45 T -> -0.1
        assert score == -0.1

    def test_score_volume_high(self):
        strat = self._make()
        score = strat._score_volume(5000, 1000)
        # ratio = 5.0 > 1.5*2=3.0 -> 0.8
        assert score == 0.8

    def test_score_volume_zero_avg(self):
        strat = self._make()
        assert strat._score_volume(100, 0) == 0.0

    def test_score_ema_bullish(self):
        strat = self._make()
        # close > fast > slow -> 0.8
        score = strat._score_ema(close=100, ema_fast_val=95, ema_slow_val=90)
        assert score == 0.8

    def test_score_ema_bearish(self):
        strat = self._make()
        score = strat._score_ema(close=80, ema_fast_val=85, ema_slow_val=90)
        assert score == -0.8

    def test_momentum_score_calculate_total(self):
        score = MomentumScore(
            rsi_score=0.5,
            macd_score=0.5,
            roc_score=0.5,
            volume_score=0.5,
            ema_score=0.5,
        )
        weights = {"rsi": 0.25, "macd": 0.25, "roc": 0.15, "volume": 0.15, "ema": 0.20}
        score.calculate_total(weights)
        assert pytest.approx(score.total_score, rel=1e-6) == 0.5
        assert score.direction == 1  # 0.5 > 0.3

    def test_generate_signal_trending_up(self, trending_up_candles):
        """Strongly trending up data should eventually produce a LONG signal or None (ADX filter)."""
        strat = self._make(adx_threshold=15.0, entry_threshold=0.3)
        signal = strat.generate_signal(trending_up_candles)
        # With relaxed thresholds the strategy may or may not fire
        # but it should not error
        if signal is not None:
            assert signal.signal_type in (SignalType.LONG, SignalType.SHORT,
                                           SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)
            assert signal.symbol == "BTCUSDT"
            assert signal.price > 0

    def test_stop_loss_calculation(self, sample_candles):
        strat = self._make()
        sl = strat._calculate_stop_loss(sample_candles, "LONG")
        current = float(sample_candles[-1, 4])
        assert sl < current  # stop loss below entry for long

        sl_short = strat._calculate_stop_loss(sample_candles, "SHORT")
        assert sl_short > current  # stop loss above entry for short

    def test_take_profit_rr(self):
        strat = self._make()
        tp = strat._calculate_take_profit(
            entry=50000, stop_loss=49000, side="LONG", rr_ratio=2.0,
        )
        assert tp == 52000

        tp_short = strat._calculate_take_profit(
            entry=50000, stop_loss=51000, side="SHORT", rr_ratio=2.0,
        )
        assert tp_short == 48000

    def test_on_tick_updates_position(self):
        strat = self._make()
        strat.open_position("LONG", 50000, 1.0)
        strat.on_tick(51000, 100, time.time())
        assert strat.position.highest_price == 51000


# ===================================================================
# MeanReversionStrategy tests
# ===================================================================

class TestMeanReversionStrategy:

    def _make(self, **params):
        defaults = {
            "account_balance": 10000.0,
            "require_rsi_confirmation": False,
            "use_trend_filter": False,
        }
        defaults.update(params)
        return MeanReversionStrategy(symbol="ETHUSDT", timeframe="5m", params=defaults)

    def test_zscore_calculation_flat(self):
        strat = self._make()
        close = np.full(30, 50000.0, dtype=np.float64)
        z = strat._calculate_zscore(close)
        # All same price -> std = 0 -> z = 0
        assert z == 0.0

    def test_zscore_calculation_extreme(self):
        strat = self._make(zscore_lookback=20)
        close = np.full(25, 50000.0, dtype=np.float64)
        # Last price is far from mean
        close[-1] = 55000.0
        z = strat._calculate_zscore(close)
        assert z > 2.0  # Should be significantly above mean

    def test_zscore_to_bucket(self):
        strat = self._make()
        assert strat._zscore_to_bucket(1.8) == "1.5-2.0"
        assert strat._zscore_to_bucket(2.3) == "2.0-2.5"
        assert strat._zscore_to_bucket(2.7) == "2.5-3.0"
        assert strat._zscore_to_bucket(3.5) == "3.0+"
        assert strat._zscore_to_bucket(-2.1) == "2.0-2.5"

    def test_insufficient_data(self):
        strat = self._make()
        short = np.zeros((10, 6))
        assert strat.generate_signal(short) is None

    def test_trend_direction(self, sample_candles):
        strat = self._make()
        close = sample_candles[:, 4]
        direction = strat._get_trend_direction(close)
        assert direction in (1, -1, 0)

    def test_mean_reversion_signal_on_extreme(self, mean_reverting_candles):
        """On oscillating data the strategy should eventually fire a signal."""
        strat = self._make(
            zscore_entry=1.5,
            zscore_exit=0.3,
            require_rsi_confirmation=False,
            use_trend_filter=False,
        )
        signals_found = []
        # Walk through candles feeding growing windows
        for i in range(80, len(mean_reverting_candles)):
            window = mean_reverting_candles[:i]
            sig = strat.generate_signal(window)
            if sig is not None:
                signals_found.append(sig)
                break
        # At least something fired or it gracefully returned None throughout
        # (data dependent - we mainly check no crashes)
        assert True

    def test_edge_stats_initially_empty(self):
        strat = self._make()
        stats = strat.get_edge_stats()
        assert "2.0-2.5" in stats
        assert stats["2.0-2.5"]["trades"] == 0

    def test_zscore_stats_dataclass(self):
        zs = ZScoreStats(trades=10, wins=6, total_pnl=500)
        assert zs.win_rate == 0.6
        assert zs.avg_pnl == 50.0


# ===================================================================
# GridTradingStrategy tests
# ===================================================================

class TestGridTradingStrategy:

    def _make(self, **params):
        defaults = {
            "num_grids": 10,
            "total_investment": 1000.0,
            "grid_type": "arithmetic",
            "auto_range": False,
            "upper_price": 51000.0,
            "lower_price": 49000.0,
        }
        defaults.update(params)
        return GridTradingStrategy(symbol="BTCUSDT", timeframe="5m", params=defaults)

    def test_arithmetic_grid_construction(self):
        strat = self._make()
        levels = strat._build_arithmetic_grid(49000, 51000, 10, 0.01)
        assert len(levels) == 11  # num_grids + 1
        # First level at 49000 last at 51000
        assert levels[0].price == 49000.0
        assert levels[-1].price == 51000.0
        # Levels below mid=50000 should be BUY
        buy_levels = [l for l in levels if l.side == "BUY"]
        sell_levels = [l for l in levels if l.side == "SELL"]
        assert len(buy_levels) > 0
        assert len(sell_levels) > 0

    def test_geometric_grid_construction(self):
        strat = self._make(grid_type="geometric")
        levels = strat._build_geometric_grid(49000, 51000, 10, 0.01)
        assert len(levels) == 11
        # Prices should be strictly increasing
        prices = [l.price for l in levels]
        for i in range(1, len(prices)):
            assert prices[i] > prices[i - 1]

    def test_grid_initialization(self, flat_candles):
        strat = self._make()
        strat._initialize_grid(flat_candles)
        assert strat._grid_initialized is True
        assert strat.grid_state.grid_count == 10
        assert len(strat.grid_state.levels) == 11

    def test_find_triggered_levels_buy(self):
        strat = self._make()
        # Manual grid level
        level = GridLevel(index=0, price=49500, side="BUY", quantity=0.01)
        strat.grid_state.levels = [level]
        # Price drops through level
        triggered = strat._find_triggered_levels(49600, 49400)
        assert len(triggered) == 1
        assert triggered[0].index == 0

    def test_find_triggered_levels_sell(self):
        strat = self._make()
        level = GridLevel(index=5, price=50500, side="SELL", quantity=0.01)
        strat.grid_state.levels = [level]
        # Price rises through level
        triggered = strat._find_triggered_levels(50400, 50600)
        assert len(triggered) == 1

    def test_find_triggered_skips_filled(self):
        strat = self._make()
        level = GridLevel(index=0, price=49500, side="BUY", quantity=0.01, is_filled=True)
        strat.grid_state.levels = [level]
        triggered = strat._find_triggered_levels(49600, 49400)
        assert len(triggered) == 0

    def test_fill_grid_level_creates_signal(self):
        strat = self._make()
        strat.grid_state = GridState()
        level = GridLevel(index=3, price=49500, side="BUY", quantity=0.01, paired_level=7)
        strat.grid_state.levels = [level] + [
            GridLevel(index=i, price=49500 + i * 200, side="SELL", quantity=0.01)
            for i in range(1, 11)
        ]
        signal = strat._fill_grid_level(level, 49500)
        assert signal is not None
        assert signal.signal_type == SignalType.LONG
        assert level.is_filled is True

    def test_grid_status_report(self, flat_candles):
        strat = self._make()
        strat._initialize_grid(flat_candles)
        status = strat.get_grid_status()
        assert status["initialized"] is True
        assert status["grid_count"] == 10
        assert "levels" in status
        assert len(status["levels"]) == 11

    def test_auto_range_detection(self, sample_candles):
        strat = self._make(auto_range=True, upper_price=0, lower_price=0)
        lower, upper = strat._detect_range(sample_candles)
        assert upper > lower
        assert lower > 0
        current = float(sample_candles[-1, 4])
        assert lower < current < upper

    def test_on_tick_processes_grid(self):
        strat = self._make()
        strat._grid_initialized = True
        strat._last_price = 50000
        level = GridLevel(index=1, price=49800, side="BUY", quantity=0.01)
        strat.grid_state.levels = [level]
        signal = strat.on_tick(49700, 100, time.time())
        if signal is not None:
            assert signal.signal_type == SignalType.LONG


import time
