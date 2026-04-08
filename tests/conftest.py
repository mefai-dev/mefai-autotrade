"""
Shared fixtures for mefai-autotrade test suite.
Provides mock candle data, orderbook snapshots, portfolio state,
and helper factories used across all test modules.
"""

import sys
import os
import time
import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Candle data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_candles(rng):
    """
    Generate 200 bars of realistic OHLCV candle data as a numpy array.
    Columns: [timestamp, open, high, low, close, volume]
    Simulates a trending-then-reverting price walk around 50000.
    """
    n = 200
    timestamps = np.arange(n, dtype=np.float64) * 300 + 1700000000
    prices = np.zeros(n, dtype=np.float64)
    prices[0] = 50000.0

    for i in range(1, n):
        change = rng.normal(0, 100)
        # Add slight mean reversion
        if prices[i - 1] > 52000:
            change -= 50
        elif prices[i - 1] < 48000:
            change += 50
        prices[i] = max(prices[i - 1] + change, 1000)

    opens = prices.copy()
    highs = prices + rng.uniform(50, 300, n)
    lows = prices - rng.uniform(50, 300, n)
    closes = prices + rng.normal(0, 50, n)
    volumes = rng.uniform(100, 5000, n)

    candles = np.column_stack([timestamps, opens, highs, lows, closes, volumes])
    return candles


@pytest.fixture
def trending_up_candles():
    """100 bars of steadily rising price (for momentum tests)."""
    n = 100
    timestamps = np.arange(n, dtype=np.float64) * 300 + 1700000000
    base = 50000.0
    closes = np.array([base + i * 30 for i in range(n)], dtype=np.float64)
    opens = closes - 10
    highs = closes + 50
    lows = closes - 60
    volumes = np.full(n, 1000.0)
    return np.column_stack([timestamps, opens, highs, lows, closes, volumes])


@pytest.fixture
def trending_down_candles():
    """100 bars of steadily falling price."""
    n = 100
    timestamps = np.arange(n, dtype=np.float64) * 300 + 1700000000
    base = 55000.0
    closes = np.array([base - i * 30 for i in range(n)], dtype=np.float64)
    opens = closes + 10
    highs = closes + 60
    lows = closes - 50
    volumes = np.full(n, 1000.0)
    return np.column_stack([timestamps, opens, highs, lows, closes, volumes])


@pytest.fixture
def mean_reverting_candles(rng):
    """
    200 bars oscillating around 50000. High z-score at extremes.
    Designed to trigger mean reversion signals.
    """
    n = 200
    timestamps = np.arange(n, dtype=np.float64) * 300 + 1700000000
    # Sine wave + noise
    t = np.linspace(0, 8 * np.pi, n)
    closes = 50000 + 1500 * np.sin(t) + rng.normal(0, 50, n)
    opens = closes - rng.uniform(-20, 20, n)
    highs = np.maximum(opens, closes) + rng.uniform(10, 100, n)
    lows = np.minimum(opens, closes) - rng.uniform(10, 100, n)
    volumes = rng.uniform(500, 3000, n)
    return np.column_stack([timestamps, opens, highs, lows, closes, volumes])


@pytest.fixture
def flat_candles():
    """100 bars of flat sideways price (for grid tests)."""
    n = 100
    timestamps = np.arange(n, dtype=np.float64) * 300 + 1700000000
    rng_local = np.random.default_rng(seed=99)
    base = 50000.0
    closes = base + rng_local.uniform(-200, 200, n)
    opens = closes + rng_local.uniform(-20, 20, n)
    highs = np.maximum(opens, closes) + rng_local.uniform(10, 80, n)
    lows = np.minimum(opens, closes) - rng_local.uniform(10, 80, n)
    volumes = np.full(n, 1500.0)
    return np.column_stack([timestamps, opens, highs, lows, closes, volumes])


# ---------------------------------------------------------------------------
# Orderbook fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_orderbook():
    """Simple orderbook snapshot dict."""
    return {
        "symbol": "BTCUSDT",
        "bids": [
            {"price": 50000.0, "quantity": 1.5},
            {"price": 49990.0, "quantity": 2.0},
            {"price": 49980.0, "quantity": 3.0},
            {"price": 49970.0, "quantity": 5.0},
            {"price": 49960.0, "quantity": 8.0},
        ],
        "asks": [
            {"price": 50010.0, "quantity": 1.2},
            {"price": 50020.0, "quantity": 2.5},
            {"price": 50030.0, "quantity": 3.5},
            {"price": 50040.0, "quantity": 4.0},
            {"price": 50050.0, "quantity": 6.0},
        ],
        "timestamp": time.time(),
    }


# ---------------------------------------------------------------------------
# Portfolio / account fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def portfolio_state():
    """Basic portfolio state dict."""
    return {
        "equity": 10000.0,
        "available_balance": 9000.0,
        "total_unrealized_pnl": 0.0,
        "positions": {},
    }
