"""
Tests for data fetching and processing: candle data parsing,
indicator calculation via FeatureEngine, orderbook structures,
and candle store types.
"""

import pytest
import time
import numpy as np

from strategies.base import FeatureEngine
from data.candle_store import Timeframe, Candle
from data.market_data import TickData, OrderBookLevel, OrderBookSnapshot


# ===================================================================
# Timeframe tests
# ===================================================================

class TestTimeframe:

    def test_from_string(self):
        assert Timeframe.from_string("1m") == Timeframe.M1
        assert Timeframe.from_string("5m") == Timeframe.M5
        assert Timeframe.from_string("1h") == Timeframe.H1
        assert Timeframe.from_string("4h") == Timeframe.H4
        assert Timeframe.from_string("1d") == Timeframe.D1
        assert Timeframe.from_string("1w") == Timeframe.W1

    def test_from_string_case_insensitive(self):
        assert Timeframe.from_string("1H") == Timeframe.H1
        assert Timeframe.from_string("5M") == Timeframe.M5

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            Timeframe.from_string("2m")

    def test_seconds(self):
        assert Timeframe.M1.seconds == 60
        assert Timeframe.H1.seconds == 3600
        assert Timeframe.D1.seconds == 86400

    def test_can_aggregate_from(self):
        assert Timeframe.H1.can_aggregate_from(Timeframe.M1) is True
        assert Timeframe.H1.can_aggregate_from(Timeframe.M5) is True
        assert Timeframe.M1.can_aggregate_from(Timeframe.H1) is False  # can't go down
        assert Timeframe.H1.can_aggregate_from(Timeframe.H1) is False  # same

    def test_label(self):
        assert Timeframe.M5.label == "M5"
        assert Timeframe.H4.label == "H4"


# ===================================================================
# Candle tests
# ===================================================================

class TestCandle:

    def test_candle_fields(self):
        c = Candle(
            timestamp=1700000000,
            open=50000, high=50500,
            low=49800, close=50200,
            volume=1234.5,
        )
        assert c.open == 50000
        assert c.high == 50500
        assert c.low == 49800
        assert c.close == 50200
        assert c.volume == 1234.5
        assert c.is_closed is True

    def test_candle_defaults(self):
        c = Candle()
        assert c.timestamp == 0.0
        assert c.volume == 0.0
        assert c.trades == 0


# ===================================================================
# TickData tests
# ===================================================================

class TestTickData:

    def test_notional(self):
        t = TickData(price=50000, quantity=0.5)
        assert t.notional == 25000

    def test_side_buyer_maker(self):
        t = TickData(is_buyer_maker=True)
        assert t.side == "SELL"

    def test_side_not_buyer_maker(self):
        t = TickData(is_buyer_maker=False)
        assert t.side == "BUY"

    def test_to_dict(self):
        t = TickData(
            symbol="BTCUSDT", exchange="binance",
            price=50000, quantity=1.0,
            timestamp=1700000000, is_buyer_maker=False,
            trade_id="t123",
        )
        d = t.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["price"] == 50000
        assert d["trade_id"] == "t123"


# ===================================================================
# OrderBookSnapshot tests
# ===================================================================

class TestOrderBookSnapshot:

    def _make_book(self):
        bids = [
            OrderBookLevel(price=50000, quantity=1.0),
            OrderBookLevel(price=49990, quantity=2.0),
            OrderBookLevel(price=49980, quantity=3.0),
        ]
        asks = [
            OrderBookLevel(price=50010, quantity=1.5),
            OrderBookLevel(price=50020, quantity=2.5),
            OrderBookLevel(price=50030, quantity=4.0),
        ]
        return OrderBookSnapshot(
            symbol="BTCUSDT", exchange="binance",
            timestamp=time.time(),
            bids=bids, asks=asks,
        )

    def test_best_bid_ask(self):
        book = self._make_book()
        assert book.best_bid_price == 50000
        assert book.best_ask_price == 50010

    def test_mid_price(self):
        book = self._make_book()
        assert book.mid_price == 50005.0

    def test_spread(self):
        book = self._make_book()
        assert book.spread == 10.0

    def test_spread_pct(self):
        book = self._make_book()
        # 10 / 50005 * 100 = ~0.02%
        assert book.spread_pct > 0
        assert book.spread_pct < 1.0

    def test_bid_depth(self):
        book = self._make_book()
        assert book.bid_depth == 6.0  # 1 + 2 + 3

    def test_ask_depth(self):
        book = self._make_book()
        assert book.ask_depth == 8.0  # 1.5 + 2.5 + 4

    def test_imbalance(self):
        book = self._make_book()
        # bids=6 asks=8 -> imbalance = (6-8)/(6+8) = -2/14 ~= -0.143
        assert book.imbalance < 0  # more ask pressure

    def test_empty_book(self):
        book = OrderBookSnapshot(symbol="BTCUSDT")
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price == 0.0
        assert book.spread == 0.0
        assert book.imbalance == 0.0


# ===================================================================
# Indicator calculation on candle data
# ===================================================================

class TestIndicatorOnCandles:
    """Integration tests: run FeatureEngine on realistic candle arrays."""

    def test_rsi_on_sample(self, sample_candles):
        close = sample_candles[:, 4]
        rsi = FeatureEngine.rsi(close, 14)
        valid = rsi[~np.isnan(rsi)]
        assert len(valid) > 100
        assert np.min(valid) >= 0
        assert np.max(valid) <= 100

    def test_bollinger_on_sample(self, sample_candles):
        close = sample_candles[:, 4]
        upper, mid, lower = FeatureEngine.bollinger_bands(close, 20, 2.0)
        # At the last bar all three should be valid
        assert not np.isnan(upper[-1])
        assert upper[-1] > mid[-1] > lower[-1]

    def test_atr_on_sample(self, sample_candles):
        high = sample_candles[:, 2]
        low = sample_candles[:, 3]
        close = sample_candles[:, 4]
        atr = FeatureEngine.atr(high, low, close, 14)
        valid = atr[~np.isnan(atr)]
        assert len(valid) > 100
        assert np.all(valid > 0)

    def test_stochastic_range(self, sample_candles):
        high = sample_candles[:, 2]
        low = sample_candles[:, 3]
        close = sample_candles[:, 4]
        k, d = FeatureEngine.stochastic(high, low, close, 14, 3)
        valid_k = k[~np.isnan(k)]
        assert np.min(valid_k) >= 0
        assert np.max(valid_k) <= 100

    def test_donchian_channel(self, sample_candles):
        high = sample_candles[:, 2]
        low = sample_candles[:, 3]
        upper, mid, lower = FeatureEngine.donchian(high, low, 20)
        last = len(high) - 1
        assert not np.isnan(upper[last])
        assert upper[last] >= mid[last] >= lower[last]

    def test_roc_values(self, sample_candles):
        close = sample_candles[:, 4]
        roc = FeatureEngine.roc(close, 12)
        valid = roc[~np.isnan(roc)]
        assert len(valid) > 100

    def test_supertrend_direction(self, sample_candles):
        high = sample_candles[:, 2]
        low = sample_candles[:, 3]
        close = sample_candles[:, 4]
        st, direction = FeatureEngine.supertrend(high, low, close, 10, 3.0)
        assert len(st) == len(close)
        assert np.all(np.isin(direction, [1, -1]))

    def test_adx_range(self, sample_candles):
        high = sample_candles[:, 2]
        low = sample_candles[:, 3]
        close = sample_candles[:, 4]
        adx = FeatureEngine.adx(high, low, close, 14)
        valid = adx[~np.isnan(adx)]
        assert len(valid) > 0
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)
