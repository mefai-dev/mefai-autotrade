"""
Candle Store - OHLCV candle storage and retrieval for Mefai Autotrade.
Provides in-memory ring buffers, disk persistence, candle aggregation,
gap detection, real-time candle building, and multi-symbol concurrent storage.
"""

import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("mefai.autotrade.candle_store")


# ---------------------------------------------------------------------------
# Timeframe definitions
# ---------------------------------------------------------------------------

class Timeframe(Enum):
    """Supported timeframes with their duration in seconds."""
    M1 = 60
    M3 = 180
    M5 = 300
    M15 = 900
    M30 = 1800
    H1 = 3600
    H2 = 7200
    H4 = 14400
    H6 = 21600
    H8 = 28800
    H12 = 43200
    D1 = 86400
    W1 = 604800

    @property
    def seconds(self) -> int:
        return self.value

    @property
    def label(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, s: str) -> "Timeframe":
        """Parse timeframe from common string formats (1m, 5m, 1h, 4h, 1d, 1w)."""
        mapping = {
            "1m": cls.M1, "3m": cls.M3, "5m": cls.M5, "15m": cls.M15,
            "30m": cls.M30, "1h": cls.H1, "2h": cls.H2, "4h": cls.H4,
            "6h": cls.H6, "8h": cls.H8, "12h": cls.H12, "1d": cls.D1,
            "1w": cls.W1,
        }
        key = s.lower().strip()
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unknown timeframe: {s}")

    def can_aggregate_from(self, lower: "Timeframe") -> bool:
        """Check if this timeframe can be built from a lower one (must be exact multiple)."""
        if lower.seconds >= self.seconds:
            return False
        return self.seconds % lower.seconds == 0

    @property
    def candles_per(self) -> Dict["Timeframe", int]:
        """How many candles of the lower timeframe fit into this one."""
        result = {}
        for tf in Timeframe:
            if tf.seconds < self.seconds and self.seconds % tf.seconds == 0:
                result[tf] = self.seconds // tf.seconds
        return result


# ---------------------------------------------------------------------------
# Candle data structure
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: float = 0.0  # Unix timestamp (seconds)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    trades: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    vwap: float = 0.0
    is_closed: bool = True

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [timestamp, o, h, l, c, v]."""
        return np.array([
            self.timestamp, self.open, self.high, self.low,
            self.close, self.volume,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Candle":
        return cls(
            timestamp=float(arr[0]),
            open=float(arr[1]),
            high=float(arr[2]),
            low=float(arr[3]),
            close=float(arr[4]),
            volume=float(arr[5]) if len(arr) > 5 else 0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trades": self.trades,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "vwap": self.vwap,
            "is_closed": self.is_closed,
        }


# ---------------------------------------------------------------------------
# Ring buffer for in-memory candle storage
# ---------------------------------------------------------------------------

class CandleBuffer:
    """
    Fixed-size ring buffer for candles using numpy arrays.
    Memory-efficient storage for a single symbol and timeframe.
    Columns: [timestamp, open, high, low, close, volume, trades, buy_vol, sell_vol, vwap]
    """

    COLS = 10  # Number of data columns

    def __init__(self, capacity: int = 5000):
        self._capacity = capacity
        self._data = np.zeros((capacity, self.COLS), dtype=np.float64)
        self._head = 0
        self._count = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count >= self._capacity

    def _index(self, logical: int) -> int:
        """Convert logical index to physical ring index."""
        start = (self._head - self._count) % self._capacity
        return (start + logical) % self._capacity

    def append(self, candle: Candle) -> None:
        """Append a candle to the buffer."""
        row = np.array([
            candle.timestamp, candle.open, candle.high, candle.low,
            candle.close, candle.volume, candle.trades,
            candle.buy_volume, candle.sell_volume, candle.vwap,
        ], dtype=np.float64)

        with self._lock:
            self._data[self._head] = row
            self._head = (self._head + 1) % self._capacity
            if self._count < self._capacity:
                self._count += 1

    def update_last(self, candle: Candle) -> None:
        """Update the most recent candle in place (for building candles in real-time)."""
        if self._count == 0:
            self.append(candle)
            return

        row = np.array([
            candle.timestamp, candle.open, candle.high, candle.low,
            candle.close, candle.volume, candle.trades,
            candle.buy_volume, candle.sell_volume, candle.vwap,
        ], dtype=np.float64)

        with self._lock:
            idx = (self._head - 1) % self._capacity
            self._data[idx] = row

    def get_last(self, n: int = 1) -> List[Candle]:
        """Get the last n candles (most recent first)."""
        with self._lock:
            n = min(n, self._count)
            result = []
            for i in range(n):
                idx = (self._head - 1 - i) % self._capacity
                row = self._data[idx]
                result.append(Candle(
                    timestamp=float(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    trades=int(row[6]),
                    buy_volume=float(row[7]),
                    sell_volume=float(row[8]),
                    vwap=float(row[9]),
                ))
            return result

    def get_range(self, start_ts: float, end_ts: float) -> List[Candle]:
        """Get candles within a timestamp range."""
        with self._lock:
            result = []
            for i in range(self._count):
                idx = self._index(i)
                ts = self._data[idx, 0]
                if start_ts <= ts <= end_ts:
                    row = self._data[idx]
                    result.append(Candle(
                        timestamp=float(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        trades=int(row[6]),
                        buy_volume=float(row[7]),
                        sell_volume=float(row[8]),
                        vwap=float(row[9]),
                    ))
            return result

    def get_numpy(self, n: Optional[int] = None) -> np.ndarray:
        """Get candle data as numpy array, oldest first. Columns: [ts, o, h, l, c, v, ...]."""
        with self._lock:
            if self._count == 0:
                return np.empty((0, self.COLS), dtype=np.float64)

            count = min(n, self._count) if n is not None else self._count
            result = np.zeros((count, self.COLS), dtype=np.float64)

            start_offset = self._count - count
            for i in range(count):
                idx = self._index(start_offset + i)
                result[i] = self._data[idx]

            return result

    def get_column(self, col: int, n: Optional[int] = None) -> np.ndarray:
        """Get a single column as 1D array. Col indices: 0=ts, 1=o, 2=h, 3=l, 4=c, 5=v."""
        data = self.get_numpy(n)
        if data.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        return data[:, col]

    @property
    def timestamps(self) -> np.ndarray:
        return self.get_column(0)

    @property
    def opens(self) -> np.ndarray:
        return self.get_column(1)

    @property
    def highs(self) -> np.ndarray:
        return self.get_column(2)

    @property
    def lows(self) -> np.ndarray:
        return self.get_column(3)

    @property
    def closes(self) -> np.ndarray:
        return self.get_column(4)

    @property
    def volumes(self) -> np.ndarray:
        return self.get_column(5)

    @property
    def latest_timestamp(self) -> float:
        if self._count == 0:
            return 0.0
        with self._lock:
            idx = (self._head - 1) % self._capacity
            return float(self._data[idx, 0])

    @property
    def oldest_timestamp(self) -> float:
        if self._count == 0:
            return 0.0
        with self._lock:
            idx = self._index(0)
            return float(self._data[idx, 0])

    def clear(self) -> None:
        with self._lock:
            self._data[:] = 0.0
            self._head = 0
            self._count = 0

    @property
    def memory_bytes(self) -> int:
        return self._data.nbytes


# ---------------------------------------------------------------------------
# Candle aggregation
# ---------------------------------------------------------------------------

class CandleAggregator:
    """Build higher-timeframe candles from lower-timeframe candles."""

    @staticmethod
    def aggregate(candles: Sequence[Candle], target_tf: Timeframe) -> List[Candle]:
        """
        Aggregate a list of candles (assumed sorted by timestamp, same source tf)
        into the target timeframe.
        """
        if not candles:
            return []

        interval = target_tf.seconds
        result: List[Candle] = []
        current: Optional[Candle] = None
        current_bucket = 0

        for c in candles:
            bucket = int(c.timestamp // interval) * interval

            if current is None or bucket != current_bucket:
                # Close previous candle
                if current is not None:
                    current.is_closed = True
                    result.append(current)

                # Start new candle
                current = Candle(
                    timestamp=float(bucket),
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                    trades=c.trades,
                    buy_volume=c.buy_volume,
                    sell_volume=c.sell_volume,
                    vwap=0.0,
                    is_closed=False,
                )
                current_bucket = bucket
            else:
                # Update existing candle
                current.high = max(current.high, c.high)
                current.low = min(current.low, c.low)
                current.close = c.close
                current.volume += c.volume
                current.trades += c.trades
                current.buy_volume += c.buy_volume
                current.sell_volume += c.sell_volume

        # Finalize last candle
        if current is not None:
            result.append(current)

        # Compute VWAP for aggregated candles
        for candle in result:
            if candle.volume > 0:
                candle.vwap = candle.typical_price  # Approximation
            else:
                candle.vwap = candle.close

        return result

    @staticmethod
    def aggregate_numpy(
        data: np.ndarray, source_seconds: int, target_seconds: int
    ) -> np.ndarray:
        """
        Aggregate numpy candle array. Input columns: [ts, o, h, l, c, v].
        Returns aggregated array with same columns.
        """
        if data.shape[0] == 0:
            return np.empty((0, 6), dtype=np.float64)

        timestamps = data[:, 0]
        buckets = (timestamps // target_seconds).astype(np.int64) * target_seconds

        unique_buckets = np.unique(buckets)
        result = np.zeros((len(unique_buckets), 6), dtype=np.float64)

        for i, bucket in enumerate(unique_buckets):
            mask = buckets == bucket
            group = data[mask]
            result[i, 0] = bucket           # timestamp
            result[i, 1] = group[0, 1]      # open (first)
            result[i, 2] = np.max(group[:, 2])  # high
            result[i, 3] = np.min(group[:, 3])  # low
            result[i, 4] = group[-1, 4]     # close (last)
            result[i, 5] = np.sum(group[:, 5])  # volume

        return result


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

class GapDetector:
    """Detect and report gaps in candle data."""

    @staticmethod
    def find_gaps(
        candles: Sequence[Candle], timeframe: Timeframe, tolerance: float = 0.5
    ) -> List[Tuple[float, float, int]]:
        """
        Find gaps in candle data.
        Returns list of (gap_start, gap_end, missing_candle_count) tuples.
        Tolerance is fraction of timeframe duration allowed as drift.
        """
        if len(candles) < 2:
            return []

        interval = timeframe.seconds
        max_gap = interval * (1.0 + tolerance)
        gaps: List[Tuple[float, float, int]] = []

        for i in range(1, len(candles)):
            delta = candles[i].timestamp - candles[i - 1].timestamp
            if delta > max_gap:
                missing = int(round(delta / interval)) - 1
                gaps.append((
                    candles[i - 1].timestamp + interval,
                    candles[i].timestamp,
                    missing,
                ))

        return gaps

    @staticmethod
    def find_gaps_numpy(
        timestamps: np.ndarray, interval_seconds: int
    ) -> np.ndarray:
        """
        Find gaps in timestamp array.
        Returns Nx3 array: [gap_start, gap_end, missing_count].
        """
        if len(timestamps) < 2:
            return np.empty((0, 3), dtype=np.float64)

        diffs = np.diff(timestamps)
        gap_mask = diffs > interval_seconds * 1.5
        gap_indices = np.where(gap_mask)[0]

        if len(gap_indices) == 0:
            return np.empty((0, 3), dtype=np.float64)

        result = np.zeros((len(gap_indices), 3), dtype=np.float64)
        for i, idx in enumerate(gap_indices):
            result[i, 0] = timestamps[idx] + interval_seconds
            result[i, 1] = timestamps[idx + 1]
            result[i, 2] = round(diffs[idx] / interval_seconds) - 1

        return result

    @staticmethod
    def fill_gaps_forward(
        candles: List[Candle], timeframe: Timeframe
    ) -> List[Candle]:
        """Fill gaps by forward-filling the last known close price."""
        if len(candles) < 2:
            return list(candles)

        interval = timeframe.seconds
        filled: List[Candle] = [candles[0]]

        for i in range(1, len(candles)):
            prev = filled[-1]
            delta = candles[i].timestamp - prev.timestamp

            if delta > interval * 1.5:
                # Fill missing candles
                n_missing = int(round(delta / interval)) - 1
                for j in range(1, n_missing + 1):
                    fill_ts = prev.timestamp + interval * j
                    filled.append(Candle(
                        timestamp=fill_ts,
                        open=prev.close,
                        high=prev.close,
                        low=prev.close,
                        close=prev.close,
                        volume=0.0,
                        trades=0,
                        is_closed=True,
                    ))

            filled.append(candles[i])

        return filled


# ---------------------------------------------------------------------------
# Real-time candle builder
# ---------------------------------------------------------------------------

class RealTimeCandleBuilder:
    """
    Build candles in real-time from incoming trades/ticks.
    Maintains an open candle that gets updated with each trade,
    and emits closed candles when the time boundary is crossed.
    """

    def __init__(self, timeframe: Timeframe, on_candle_close: Optional[callable] = None):
        self._timeframe = timeframe
        self._interval = timeframe.seconds
        self._on_close = on_candle_close
        self._current: Optional[Candle] = None
        self._current_bucket: int = 0
        self._trade_count: int = 0
        self._total_value: float = 0.0  # For VWAP calculation
        self._total_volume: float = 0.0
        self._lock = threading.Lock()
        self._closed_candles: List[Candle] = []

    @property
    def current_candle(self) -> Optional[Candle]:
        return self._current

    @property
    def closed_count(self) -> int:
        return len(self._closed_candles)

    def process_trade(
        self, price: float, quantity: float, timestamp: float,
        is_buyer_maker: bool = False
    ) -> Optional[Candle]:
        """
        Process an incoming trade tick.
        Returns a closed candle if the time boundary was crossed, else None.
        """
        bucket = int(timestamp // self._interval) * self._interval
        closed_candle = None

        with self._lock:
            if self._current is None:
                # First trade ever
                self._start_new_candle(price, quantity, timestamp, bucket, is_buyer_maker)
            elif bucket != self._current_bucket:
                # Time boundary crossed - close current candle
                self._current.is_closed = True
                closed_candle = self._current
                self._closed_candles.append(closed_candle)

                # Fill any gaps between old bucket and new bucket
                gap_buckets = int((bucket - self._current_bucket) / self._interval) - 1
                if gap_buckets > 0:
                    last_close = self._current.close
                    for g in range(1, gap_buckets + 1):
                        gap_ts = self._current_bucket + self._interval * g
                        gap_candle = Candle(
                            timestamp=float(gap_ts),
                            open=last_close,
                            high=last_close,
                            low=last_close,
                            close=last_close,
                            volume=0.0,
                            is_closed=True,
                        )
                        self._closed_candles.append(gap_candle)

                # Start new candle
                self._start_new_candle(price, quantity, timestamp, bucket, is_buyer_maker)

                if self._on_close and closed_candle:
                    try:
                        self._on_close(closed_candle)
                    except Exception as exc:
                        logger.error("Candle close callback error: %s", exc)
            else:
                # Update current candle
                self._current.high = max(self._current.high, price)
                self._current.low = min(self._current.low, price)
                self._current.close = price
                self._current.volume += quantity
                self._current.trades += 1
                self._total_value += price * quantity
                self._total_volume += quantity

                if is_buyer_maker:
                    self._current.sell_volume += quantity
                else:
                    self._current.buy_volume += quantity

                if self._total_volume > 0:
                    self._current.vwap = self._total_value / self._total_volume

        return closed_candle

    def _start_new_candle(
        self, price: float, quantity: float, timestamp: float,
        bucket: int, is_buyer_maker: bool
    ) -> None:
        self._current = Candle(
            timestamp=float(bucket),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=quantity,
            trades=1,
            buy_volume=0.0 if is_buyer_maker else quantity,
            sell_volume=quantity if is_buyer_maker else 0.0,
            vwap=price,
            is_closed=False,
        )
        self._current_bucket = bucket
        self._total_value = price * quantity
        self._total_volume = quantity

    def flush(self) -> Optional[Candle]:
        """Force-close the current candle and return it."""
        with self._lock:
            if self._current is not None:
                self._current.is_closed = True
                closed = self._current
                self._closed_candles.append(closed)
                self._current = None
                self._current_bucket = 0
                self._total_value = 0.0
                self._total_volume = 0.0
                return closed
        return None

    def get_closed_candles(self, n: Optional[int] = None) -> List[Candle]:
        """Get closed candles, optionally limited to last n."""
        with self._lock:
            if n is None:
                return list(self._closed_candles)
            return list(self._closed_candles[-n:])

    def reset(self) -> None:
        with self._lock:
            self._current = None
            self._current_bucket = 0
            self._closed_candles.clear()
            self._total_value = 0.0
            self._total_volume = 0.0


# ---------------------------------------------------------------------------
# Disk persistence layer
# ---------------------------------------------------------------------------

class CandlePersistence:
    """SQLite-backed candle persistence."""

    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS candles (
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL,
        timestamp REAL NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL DEFAULT 0.0,
        trades INTEGER DEFAULT 0,
        buy_volume REAL DEFAULT 0.0,
        sell_volume REAL DEFAULT 0.0,
        vwap REAL DEFAULT 0.0,
        PRIMARY KEY (symbol, timeframe, timestamp)
    )
    """

    CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_ts
    ON candles(symbol, timeframe, timestamp)
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        import os
        db_dir = os.path.dirname(self._db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(self.CREATE_TABLE)
        self._conn.execute(self.CREATE_INDEX)
        self._conn.commit()

    def save_candles(self, symbol: str, timeframe: str, candles: Sequence[Candle]) -> int:
        """Bulk insert/replace candles. Returns number saved."""
        if not candles:
            return 0

        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                (symbol, timeframe, timestamp, open, high, low, close,
                 volume, trades, buy_volume, sell_volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (symbol, timeframe, c.timestamp, c.open, c.high, c.low,
                     c.close, c.volume, c.trades, c.buy_volume, c.sell_volume, c.vwap)
                    for c in candles
                ],
            )
            self._conn.commit()
        return len(candles)

    def load_candles(
        self, symbol: str, timeframe: str,
        start: Optional[float] = None, end: Optional[float] = None,
        limit: int = 10000,
    ) -> List[Candle]:
        """Load candles from disk."""
        clauses = ["symbol = ?", "timeframe = ?"]
        params: list = [symbol, timeframe]

        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(clauses)

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT * FROM candles WHERE {where} "
                f"ORDER BY timestamp LIMIT ?",
                params + [limit],
            )
            rows = cursor.fetchall()

        return [
            Candle(
                timestamp=float(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                trades=int(row["trades"]),
                buy_volume=float(row["buy_volume"]),
                sell_volume=float(row["sell_volume"]),
                vwap=float(row["vwap"]),
                is_closed=True,
            )
            for row in rows
        ]

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[float]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT MAX(timestamp) FROM candles WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            row = cursor.fetchone()
            return float(row[0]) if row and row[0] is not None else None

    def get_candle_count(self, symbol: str, timeframe: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            )
            return cursor.fetchone()[0]

    def delete_old_candles(self, symbol: str, timeframe: str, before: float) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM candles WHERE symbol = ? AND timeframe = ? AND timestamp < ?",
                (symbol, timeframe, before),
            )
            self._conn.commit()
            return cursor.rowcount

    def get_available_symbols(self) -> List[str]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT DISTINCT symbol FROM candles ORDER BY symbol"
            )
            return [row[0] for row in cursor.fetchall()]

    def get_available_timeframes(self, symbol: str) -> List[str]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT DISTINCT timeframe FROM candles WHERE symbol = ? ORDER BY timeframe",
                (symbol,),
            )
            return [row[0] for row in cursor.fetchall()]

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Main candle store
# ---------------------------------------------------------------------------

class CandleStore:
    """
    Unified candle storage - combines in-memory ring buffers with disk persistence.
    Provides a clean query interface for strategies and indicators.
    """

    def __init__(
        self,
        db_path: str = "candles.db",
        buffer_capacity: int = 5000,
        auto_persist: bool = True,
        persist_interval: int = 60,
        max_memory_mb: float = 512.0,
    ):
        self._db_path = db_path
        self._buffer_capacity = buffer_capacity
        self._auto_persist = auto_persist
        self._persist_interval = persist_interval
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)

        self._persistence = CandlePersistence(db_path)
        self._buffers: Dict[str, CandleBuffer] = {}  # key: "symbol:timeframe"
        self._builders: Dict[str, RealTimeCandleBuilder] = {}
        self._lock = threading.Lock()
        self._persist_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(
            "CandleStore initialized (db=%s, buffer=%d, persist=%s)",
            db_path, buffer_capacity, auto_persist,
        )

    def _buffer_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def _get_or_create_buffer(self, symbol: str, timeframe: str) -> CandleBuffer:
        key = self._buffer_key(symbol, timeframe)
        if key not in self._buffers:
            with self._lock:
                if key not in self._buffers:
                    self._buffers[key] = CandleBuffer(self._buffer_capacity)
        return self._buffers[key]

    def start(self) -> None:
        """Start background persistence thread if auto_persist is enabled."""
        if self._auto_persist and not self._running:
            self._running = True
            self._persist_thread = threading.Thread(
                target=self._persist_loop, daemon=True, name="candle-persist"
            )
            self._persist_thread.start()
            logger.info("Candle persistence thread started")

    def stop(self) -> None:
        """Stop background persistence and flush all data."""
        self._running = False
        if self._persist_thread and self._persist_thread.is_alive():
            self._persist_thread.join(timeout=10)
        self.flush_all()
        self._persistence.close()
        logger.info("CandleStore stopped")

    def _persist_loop(self) -> None:
        while self._running:
            try:
                self.flush_all()
            except Exception as exc:
                logger.error("Candle persistence error: %s", exc)
            time.sleep(self._persist_interval)

    def add_candles(self, symbol: str, timeframe: str, candles: Sequence[Candle]) -> None:
        """Add multiple candles to both buffer and persistence."""
        buf = self._get_or_create_buffer(symbol, timeframe)
        for c in candles:
            buf.append(c)

        if self._auto_persist:
            self._persistence.save_candles(symbol, timeframe, candles)

    def add_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Add a single candle."""
        buf = self._get_or_create_buffer(symbol, timeframe)
        buf.append(candle)

    def update_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Update the latest candle in the buffer (for live building)."""
        buf = self._get_or_create_buffer(symbol, timeframe)
        buf.update_last(candle)

    def process_trade(
        self, symbol: str, timeframe: str,
        price: float, quantity: float, timestamp: float,
        is_buyer_maker: bool = False,
    ) -> Optional[Candle]:
        """Process a trade tick to build candles in real-time."""
        key = self._buffer_key(symbol, timeframe)
        if key not in self._builders:
            tf = Timeframe.from_string(timeframe)
            self._builders[key] = RealTimeCandleBuilder(
                tf,
                on_candle_close=lambda c, s=symbol, t=timeframe: self.add_candle(s, t, c),
            )

        builder = self._builders[key]
        closed = builder.process_trade(price, quantity, timestamp, is_buyer_maker)

        # Also update the live candle in buffer
        if builder.current_candle:
            self.update_candle(symbol, timeframe, builder.current_candle)

        return closed

    def get_candles(
        self, symbol: str, timeframe: str,
        start: Optional[float] = None, end: Optional[float] = None,
        limit: int = 500,
    ) -> List[Candle]:
        """
        Get candles for a symbol and timeframe.
        Tries in-memory buffer first, falls back to disk if needed.
        """
        buf = self._get_or_create_buffer(symbol, timeframe)

        if start is not None and end is not None:
            candles = buf.get_range(start, end)
            if candles:
                return candles[-limit:]
            # Fallback to disk
            return self._persistence.load_candles(symbol, timeframe, start, end, limit)

        # Get last N from buffer
        candles = buf.get_last(limit)
        if candles:
            candles.reverse()  # Return oldest first
            return candles

        # Fallback to disk
        return self._persistence.load_candles(symbol, timeframe, limit=limit)

    def get_numpy(
        self, symbol: str, timeframe: str, n: Optional[int] = None
    ) -> np.ndarray:
        """Get candle data as numpy array [ts, o, h, l, c, v, ...]."""
        buf = self._get_or_create_buffer(symbol, timeframe)
        return buf.get_numpy(n)

    def get_closes(self, symbol: str, timeframe: str, n: Optional[int] = None) -> np.ndarray:
        """Shortcut to get close prices as numpy array."""
        buf = self._get_or_create_buffer(symbol, timeframe)
        return buf.closes if n is None else buf.get_column(4, n)

    def get_volumes(self, symbol: str, timeframe: str, n: Optional[int] = None) -> np.ndarray:
        buf = self._get_or_create_buffer(symbol, timeframe)
        return buf.volumes if n is None else buf.get_column(5, n)

    def get_aggregated(
        self, symbol: str, source_tf: str, target_tf: str, limit: int = 500
    ) -> List[Candle]:
        """Get candles aggregated from a lower timeframe to a higher one."""
        source = Timeframe.from_string(source_tf)
        target = Timeframe.from_string(target_tf)

        if not target.can_aggregate_from(source):
            raise ValueError(
                f"Cannot aggregate {source_tf} -> {target_tf}: "
                f"not an exact multiple"
            )

        # Get enough source candles
        ratio = target.seconds // source.seconds
        source_limit = limit * ratio + ratio  # Extra for boundary
        source_candles = self.get_candles(symbol, source_tf, limit=source_limit)
        return CandleAggregator.aggregate(source_candles, target)[-limit:]

    def flush_all(self) -> int:
        """Persist all buffered candles to disk. Returns total candles written."""
        total = 0
        with self._lock:
            keys = list(self._buffers.keys())

        for key in keys:
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            symbol, timeframe = parts
            buf = self._buffers[key]

            candles = buf.get_last(buf.count)
            if candles:
                candles.reverse()
                written = self._persistence.save_candles(symbol, timeframe, candles)
                total += written

        return total

    def cleanup_old_data(self, max_age_days: int = 90) -> Dict[str, int]:
        """Remove candles older than max_age_days from disk."""
        cutoff = time.time() - (max_age_days * 86400)
        counts: Dict[str, int] = {}

        symbols = self._persistence.get_available_symbols()
        for symbol in symbols:
            timeframes = self._persistence.get_available_timeframes(symbol)
            for tf in timeframes:
                deleted = self._persistence.delete_old_candles(symbol, tf, cutoff)
                if deleted > 0:
                    counts[f"{symbol}:{tf}"] = deleted

        if counts:
            self._persistence.vacuum()
            logger.info("Cleaned up old candles: %s", counts)

        return counts

    @property
    def memory_usage_bytes(self) -> int:
        total = 0
        for buf in self._buffers.values():
            total += buf.memory_bytes
        return total

    @property
    def memory_usage_mb(self) -> float:
        return self.memory_usage_bytes / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "buffer_count": len(self._buffers),
            "builder_count": len(self._builders),
            "memory_mb": round(self.memory_usage_mb, 2),
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
            "auto_persist": self._auto_persist,
            "running": self._running,
            "available_symbols": self._persistence.get_available_symbols(),
        }

    def __enter__(self) -> "CandleStore":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
