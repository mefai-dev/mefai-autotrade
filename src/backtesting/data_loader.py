# Mefai Signal Engine - Historical Data Loader
# Load, validate, resample, and cache historical market data from multiple sources.

import os
import csv
import json
import time
import hashlib
import logging
import sqlite3
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OHLCVBar:
    """Single OHLCV bar."""
    timestamp: datetime = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    quote_volume: float = 0.0
    trades_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades_count": self.trades_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OHLCVBar":
        ts = d.get("timestamp")
        if isinstance(ts, (int, float)):
            ts = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)
        return cls(
            timestamp=ts,
            open=float(d.get("open", 0)),
            high=float(d.get("high", 0)),
            low=float(d.get("low", 0)),
            close=float(d.get("close", 0)),
            volume=float(d.get("volume", 0)),
            quote_volume=float(d.get("quote_volume", 0)),
            trades_count=int(d.get("trades_count", 0)),
        )


class Timeframe(Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MO1 = "1M"

    @property
    def minutes(self) -> int:
        mapping = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
            "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200,
        }
        return mapping.get(self.value, 1)


@dataclass
class DataValidationResult:
    """Result of data validation checks."""
    is_valid: bool = True
    total_bars: int = 0
    gaps: List[Tuple[datetime, datetime]] = field(default_factory=list)
    duplicates: int = 0
    outliers: List[Tuple[int, str, float]] = field(default_factory=list)
    null_values: int = 0
    negative_values: int = 0
    hlc_violations: int = 0  # high < low or close outside high/low
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data source interface
# ---------------------------------------------------------------------------

class DataSource(ABC):
    """Abstract base class for custom data sources."""

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str,
              start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch OHLCV data from the source."""
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Return list of available symbols."""
        pass

    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """Return list of available timeframes."""
        pass


# ---------------------------------------------------------------------------
# CSV data source
# ---------------------------------------------------------------------------

class CSVDataSource(DataSource):
    """Load OHLCV data from CSV files."""

    COLUMN_MAPPINGS = {
        "timestamp": ["timestamp", "time", "date", "datetime", "ts", "open_time"],
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price"],
        "volume": ["volume", "vol", "v", "base_volume"],
        "quote_volume": ["quote_volume", "qv", "quote_vol"],
        "trades_count": ["trades_count", "trades", "num_trades", "count"],
    }

    def __init__(self, base_dir: str, delimiter: str = ",",
                 timestamp_format: Optional[str] = None,
                 timestamp_unit: str = "ms"):
        self.base_dir = Path(base_dir)
        self.delimiter = delimiter
        self.timestamp_format = timestamp_format
        self.timestamp_unit = timestamp_unit

    def fetch(self, symbol: str, timeframe: str = "",
              start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        # Try multiple file patterns
        patterns = [
            f"{symbol}.csv",
            f"{symbol}_{timeframe}.csv",
            f"{symbol}-{timeframe}.csv",
            f"{symbol.lower()}.csv",
            f"{symbol.upper()}.csv",
        ]

        filepath = None
        for pattern in patterns:
            candidate = self.base_dir / pattern
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            # Try exact path
            exact = Path(symbol)
            if exact.exists():
                filepath = exact
            else:
                raise FileNotFoundError(
                    f"No CSV file found for {symbol} in {self.base_dir}"
                )

        return self._load_csv(filepath, start, end)

    def _load_csv(self, filepath: Path,
                  start: Optional[datetime] = None,
                  end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        bars = []
        with open(filepath, "r", newline="") as f:
            # Detect dialect
            sample = f.read(4096)
            f.seek(0)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
            except csv.Error:
                dialect = None

            reader = csv.DictReader(f, delimiter=self.delimiter,
                                     dialect=dialect)
            column_map = self._detect_columns(reader.fieldnames or [])

            for row in reader:
                try:
                    bar = self._parse_row(row, column_map)
                    if bar is None:
                        continue
                    ts = bar["timestamp"]
                    if start and ts < start:
                        continue
                    if end and ts > end:
                        continue
                    bars.append(bar)
                except Exception as e:
                    logger.warning(f"Skipping row in {filepath}: {e}")
                    continue

        logger.info(f"Loaded {len(bars)} bars from {filepath}")
        return bars

    def _detect_columns(self, headers: List[str]) -> Dict[str, str]:
        """Map CSV headers to standard OHLCV field names."""
        column_map = {}
        headers_lower = [h.lower().strip() for h in headers]

        for field_name, aliases in self.COLUMN_MAPPINGS.items():
            for alias in aliases:
                if alias.lower() in headers_lower:
                    idx = headers_lower.index(alias.lower())
                    column_map[field_name] = headers[idx]
                    break

        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(column_map.keys())
        if missing:
            # Fall back to positional
            if len(headers) >= 6:
                column_map = {
                    "timestamp": headers[0],
                    "open": headers[1],
                    "high": headers[2],
                    "low": headers[3],
                    "close": headers[4],
                    "volume": headers[5],
                }
            else:
                raise ValueError(
                    f"Cannot map CSV columns. Missing: {missing}. "
                    f"Headers: {headers}"
                )

        return column_map

    def _parse_row(self, row: Dict[str, str],
                    column_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
        ts_raw = row.get(column_map.get("timestamp", ""), "").strip()
        if not ts_raw:
            return None

        timestamp = self._parse_timestamp(ts_raw)
        if timestamp is None:
            return None

        bar = {
            "timestamp": timestamp,
            "open": float(row.get(column_map.get("open", ""), 0)),
            "high": float(row.get(column_map.get("high", ""), 0)),
            "low": float(row.get(column_map.get("low", ""), 0)),
            "close": float(row.get(column_map.get("close", ""), 0)),
            "volume": float(row.get(column_map.get("volume", ""), 0)),
            "quote_volume": float(row.get(column_map.get("quote_volume", ""), 0)),
            "trades_count": int(float(row.get(column_map.get("trades_count", ""), 0))),
        }
        return bar

    def _parse_timestamp(self, raw: str) -> Optional[datetime]:
        if self.timestamp_format:
            try:
                return datetime.strptime(raw, self.timestamp_format)
            except ValueError:
                return None

        # Try numeric timestamp
        try:
            ts_num = float(raw)
            if ts_num > 1e15:  # microseconds
                return datetime.utcfromtimestamp(ts_num / 1e6)
            elif ts_num > 1e12:  # milliseconds
                return datetime.utcfromtimestamp(ts_num / 1000)
            elif ts_num > 1e9:  # seconds
                return datetime.utcfromtimestamp(ts_num)
            else:
                return datetime.utcfromtimestamp(ts_num)
        except (ValueError, OSError):
            pass

        # Try common datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue

        logger.warning(f"Cannot parse timestamp: {raw}")
        return None

    def get_available_symbols(self) -> List[str]:
        symbols = []
        if self.base_dir.exists():
            for f in self.base_dir.glob("*.csv"):
                name = f.stem.upper()
                # Remove timeframe suffix
                for tf in ["_1M", "_5M", "_15M", "_1H", "_4H", "_1D"]:
                    if name.endswith(tf):
                        name = name[:-len(tf)]
                        break
                if name not in symbols:
                    symbols.append(name)
        return sorted(symbols)

    def get_available_timeframes(self) -> List[str]:
        return ["1m", "5m", "15m", "1h", "4h", "1d"]


# ---------------------------------------------------------------------------
# Binance API data source
# ---------------------------------------------------------------------------

class BinanceDataSource(DataSource):
    """Load historical klines from the Binance API."""

    BASE_URL = "https://fapi.binance.com"
    MAX_LIMIT = 1500
    RATE_LIMIT_DELAY = 0.1  # seconds between requests

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 proxy_url: Optional[str] = None):
        self.base_url = base_url or self.BASE_URL
        self.api_key = api_key
        self.proxy_url = proxy_url

    def fetch(self, symbol: str, timeframe: str = "1h",
              start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required for Binance data source")

        all_bars = []
        start_ms = int(start.timestamp() * 1000) if start else None
        end_ms = int(end.timestamp() * 1000) if end else None

        url = f"{self.base_url}/fapi/v1/klines"
        if self.proxy_url:
            url = f"{self.proxy_url}/fapi/klines"

        while True:
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "limit": self.MAX_LIMIT,
            }
            if start_ms:
                params["startTime"] = start_ms
            if end_ms:
                params["endTime"] = end_ms

            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key

            try:
                resp = requests.get(url, params=params, headers=headers,
                                     timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"Binance API error: {e}")
                break

            if not data:
                break

            for kline in data:
                bar = {
                    "timestamp": datetime.utcfromtimestamp(kline[0] / 1000),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "quote_volume": float(kline[7]),
                    "trades_count": int(kline[8]),
                }
                all_bars.append(bar)

            if len(data) < self.MAX_LIMIT:
                break

            # Move start to after the last bar
            last_ts = data[-1][0]
            start_ms = last_ts + 1

            if end_ms and start_ms > end_ms:
                break

            time.sleep(self.RATE_LIMIT_DELAY)

        logger.info(
            f"Fetched {len(all_bars)} bars for {symbol} {timeframe} from Binance"
        )
        return all_bars

    def get_available_symbols(self) -> List[str]:
        try:
            import requests
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [s["symbol"] for s in data.get("symbols", [])
                    if s.get("status") == "TRADING"]
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []

    def get_available_timeframes(self) -> List[str]:
        return [
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M",
        ]


# ---------------------------------------------------------------------------
# Database data source
# ---------------------------------------------------------------------------

class DatabaseDataSource(DataSource):
    """Load historical data from SQLite or PostgreSQL databases."""

    def __init__(self, connection_string: str, table_name: str = "ohlcv",
                 db_type: str = "sqlite"):
        self.connection_string = connection_string
        self.table_name = table_name
        self.db_type = db_type
        self._connection = None

    def _get_connection(self):
        if self._connection is not None:
            return self._connection

        if self.db_type == "sqlite":
            self._connection = sqlite3.connect(self.connection_string)
            self._connection.row_factory = sqlite3.Row
        elif self.db_type == "postgresql":
            try:
                import psycopg2
                import psycopg2.extras
                self._connection = psycopg2.connect(self.connection_string)
            except ImportError:
                raise ImportError("psycopg2 required for PostgreSQL")
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        return self._connection

    def fetch(self, symbol: str, timeframe: str = "",
              start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        conditions = ["symbol = ?"]
        params = [symbol]

        if timeframe:
            conditions.append("timeframe = ?")
            params.append(timeframe)
        if start:
            conditions.append("timestamp >= ?")
            params.append(start.isoformat() if self.db_type == "sqlite"
                          else start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end.isoformat() if self.db_type == "sqlite"
                          else end)

        where = " AND ".join(conditions)

        if self.db_type == "postgresql":
            query = f"""
                SELECT timestamp, open, high, low, close, volume,
                       COALESCE(quote_volume, 0) as quote_volume,
                       COALESCE(trades_count, 0) as trades_count
                FROM {self.table_name}
                WHERE {where.replace('?', '%s')}
                ORDER BY timestamp ASC
            """
        else:
            query = f"""
                SELECT timestamp, open, high, low, close, volume,
                       COALESCE(quote_volume, 0) as quote_volume,
                       COALESCE(trades_count, 0) as trades_count
                FROM {self.table_name}
                WHERE {where}
                ORDER BY timestamp ASC
            """

        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        bars = []
        for row in rows:
            if self.db_type == "sqlite":
                ts = row[0]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                elif isinstance(ts, (int, float)):
                    ts = datetime.utcfromtimestamp(
                        ts / 1000 if ts > 1e12 else ts
                    )
                bar = {
                    "timestamp": ts,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[6]),
                    "trades_count": int(row[7]),
                }
            else:
                bar = {
                    "timestamp": row[0] if isinstance(row[0], datetime)
                                 else datetime.fromisoformat(str(row[0])),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[6]),
                    "trades_count": int(row[7]),
                }
            bars.append(bar)

        logger.info(
            f"Loaded {len(bars)} bars for {symbol} from {self.db_type} database"
        )
        return bars

    def get_available_symbols(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT symbol FROM {self.table_name}")
        return [row[0] for row in cursor.fetchall()]

    def get_available_timeframes(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"SELECT DISTINCT timeframe FROM {self.table_name}"
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

class DataValidator:
    """Validates OHLCV data for gaps, duplicates, outliers, and consistency."""

    def __init__(self, timeframe_minutes: int = 60,
                 outlier_threshold: float = 5.0):
        self.timeframe_minutes = timeframe_minutes
        self.outlier_threshold = outlier_threshold  # std devs for outlier detection

    def validate(self, bars: List[Dict[str, Any]]) -> DataValidationResult:
        result = DataValidationResult(total_bars=len(bars))

        if not bars:
            result.is_valid = False
            result.errors.append("No data to validate")
            return result

        self._check_duplicates(bars, result)
        self._check_gaps(bars, result)
        self._check_ohlc_consistency(bars, result)
        self._check_null_negative(bars, result)
        self._check_outliers(bars, result)

        if result.errors:
            result.is_valid = False

        return result

    def _check_duplicates(self, bars: List[Dict[str, Any]],
                           result: DataValidationResult) -> None:
        seen = set()
        for bar in bars:
            ts = bar["timestamp"]
            if ts in seen:
                result.duplicates += 1
            seen.add(ts)
        if result.duplicates > 0:
            result.warnings.append(
                f"Found {result.duplicates} duplicate timestamps"
            )

    def _check_gaps(self, bars: List[Dict[str, Any]],
                     result: DataValidationResult) -> None:
        expected_delta = timedelta(minutes=self.timeframe_minutes)
        tolerance = expected_delta * 0.5

        for i in range(1, len(bars)):
            ts_prev = bars[i - 1]["timestamp"]
            ts_curr = bars[i]["timestamp"]
            delta = ts_curr - ts_prev

            if delta > expected_delta + tolerance:
                result.gaps.append((ts_prev, ts_curr))

        if result.gaps:
            result.warnings.append(f"Found {len(result.gaps)} gaps in data")

    def _check_ohlc_consistency(self, bars: List[Dict[str, Any]],
                                 result: DataValidationResult) -> None:
        for i, bar in enumerate(bars):
            h, l, o, c = bar["high"], bar["low"], bar["open"], bar["close"]
            if h < l:
                result.hlc_violations += 1
            if o > h or o < l:
                result.hlc_violations += 1
            if c > h or c < l:
                result.hlc_violations += 1

        if result.hlc_violations > 0:
            result.warnings.append(
                f"Found {result.hlc_violations} OHLC consistency violations"
            )

    def _check_null_negative(self, bars: List[Dict[str, Any]],
                              result: DataValidationResult) -> None:
        for bar in bars:
            for key in ("open", "high", "low", "close", "volume"):
                val = bar.get(key)
                if val is None:
                    result.null_values += 1
                elif val < 0:
                    result.negative_values += 1

        if result.null_values > 0:
            result.errors.append(f"Found {result.null_values} null values")
        if result.negative_values > 0:
            result.errors.append(
                f"Found {result.negative_values} negative values"
            )

    def _check_outliers(self, bars: List[Dict[str, Any]],
                         result: DataValidationResult) -> None:
        if len(bars) < 20:
            return

        closes = [b["close"] for b in bars]
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                returns.append((i, ret))

        if not returns:
            return

        mean_ret = sum(r[1] for r in returns) / len(returns)
        variance = sum((r[1] - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = variance ** 0.5

        if std_ret <= 0:
            return

        for idx, ret in returns:
            z_score = abs(ret - mean_ret) / std_ret
            if z_score > self.outlier_threshold:
                result.outliers.append((idx, "close", ret))

        if result.outliers:
            result.warnings.append(
                f"Found {len(result.outliers)} potential outliers "
                f"(>{self.outlier_threshold} std devs)"
            )


# ---------------------------------------------------------------------------
# Data resampling
# ---------------------------------------------------------------------------

class DataResampler:
    """Resample OHLCV data to different timeframes."""

    @staticmethod
    def resample(bars: List[Dict[str, Any]],
                 source_tf_minutes: int,
                 target_tf_minutes: int) -> List[Dict[str, Any]]:
        """
        Resample bars from source timeframe to target timeframe.
        Target must be a multiple of source.
        """
        if target_tf_minutes <= source_tf_minutes:
            raise ValueError(
                f"Target timeframe ({target_tf_minutes}m) must be larger "
                f"than source ({source_tf_minutes}m)"
            )

        if target_tf_minutes % source_tf_minutes != 0:
            raise ValueError(
                f"Target timeframe must be a multiple of source timeframe"
            )

        ratio = target_tf_minutes // source_tf_minutes
        if not bars:
            return []

        resampled = []
        bucket = []

        for bar in bars:
            bucket.append(bar)
            if len(bucket) >= ratio:
                merged = DataResampler._merge_bars(bucket)
                resampled.append(merged)
                bucket = []

        # Handle remaining bars in the last incomplete bucket
        if bucket:
            merged = DataResampler._merge_bars(bucket)
            resampled.append(merged)

        return resampled

    @staticmethod
    def resample_by_time(bars: List[Dict[str, Any]],
                          target_tf_minutes: int) -> List[Dict[str, Any]]:
        """
        Resample bars by grouping on time boundaries.
        More accurate than fixed-ratio resampling.
        """
        if not bars:
            return []

        target_delta = timedelta(minutes=target_tf_minutes)
        resampled = []
        bucket = []
        bucket_start = None

        for bar in bars:
            ts = bar["timestamp"]
            if bucket_start is None:
                bucket_start = ts
                bucket = [bar]
                continue

            if ts - bucket_start >= target_delta:
                merged = DataResampler._merge_bars(bucket)
                resampled.append(merged)
                bucket = [bar]
                bucket_start = ts
            else:
                bucket.append(bar)

        if bucket:
            merged = DataResampler._merge_bars(bucket)
            resampled.append(merged)

        return resampled

    @staticmethod
    def _merge_bars(bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple bars into a single bar."""
        return {
            "timestamp": bars[0]["timestamp"],
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "volume": sum(b["volume"] for b in bars),
            "quote_volume": sum(b.get("quote_volume", 0) for b in bars),
            "trades_count": sum(b.get("trades_count", 0) for b in bars),
        }


# ---------------------------------------------------------------------------
# Multi-symbol alignment
# ---------------------------------------------------------------------------

class DataAligner:
    """Align multiple symbol datasets to shared timestamps."""

    @staticmethod
    def align(datasets: Dict[str, List[Dict[str, Any]]],
              method: str = "inner") -> Dict[str, List[Dict[str, Any]]]:
        """
        Align multiple symbol datasets.

        method:
            "inner" - only keep timestamps present in ALL datasets
            "outer" - keep all timestamps, forward-fill missing bars
        """
        if not datasets:
            return {}

        # Build timestamp sets for each symbol
        ts_sets = {}
        ts_maps = {}
        for symbol, bars in datasets.items():
            ts_set = set()
            ts_map = {}
            for bar in bars:
                ts = bar["timestamp"]
                ts_set.add(ts)
                ts_map[ts] = bar
            ts_sets[symbol] = ts_set
            ts_maps[symbol] = ts_map

        # Determine shared timestamps
        if method == "inner":
            common_ts = set.intersection(*ts_sets.values())
        elif method == "outer":
            common_ts = set.union(*ts_sets.values())
        else:
            raise ValueError(f"Unknown alignment method: {method}")

        sorted_ts = sorted(common_ts)

        # Build aligned datasets
        aligned = {}
        for symbol in datasets:
            aligned_bars = []
            last_bar = None
            for ts in sorted_ts:
                bar = ts_maps[symbol].get(ts)
                if bar is not None:
                    aligned_bars.append(bar)
                    last_bar = bar
                elif method == "outer" and last_bar is not None:
                    # Forward fill
                    filled = last_bar.copy()
                    filled["timestamp"] = ts
                    filled["volume"] = 0
                    filled["quote_volume"] = 0
                    filled["trades_count"] = 0
                    aligned_bars.append(filled)
            aligned[symbol] = aligned_bars

        return aligned


# ---------------------------------------------------------------------------
# Data cache
# ---------------------------------------------------------------------------

class DataCache:
    """Cache loaded data to disk for fast re-runs."""

    def __init__(self, cache_dir: str = ".backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, symbol: str, timeframe: str,
                    start: Optional[datetime],
                    end: Optional[datetime]) -> str:
        parts = [symbol, timeframe]
        if start:
            parts.append(start.isoformat())
        if end:
            parts.append(end.isoformat())
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, symbol: str, timeframe: str = "",
            start: Optional[datetime] = None,
            end: Optional[datetime] = None) -> Optional[List[Dict[str, Any]]]:
        key = self._cache_key(symbol, timeframe, start, end)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            # Convert timestamps back to datetime
            for bar in data:
                if isinstance(bar.get("timestamp"), str):
                    bar["timestamp"] = datetime.fromisoformat(bar["timestamp"])
                elif isinstance(bar.get("timestamp"), (int, float)):
                    ts = bar["timestamp"]
                    bar["timestamp"] = datetime.utcfromtimestamp(
                        ts / 1000 if ts > 1e12 else ts
                    )
            logger.info(f"Cache hit: {symbol} ({len(data)} bars)")
            return data
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def put(self, symbol: str, timeframe: str,
            start: Optional[datetime], end: Optional[datetime],
            bars: List[Dict[str, Any]]) -> None:
        key = self._cache_key(symbol, timeframe, start, end)
        cache_file = self.cache_dir / f"{key}.json"

        try:
            serializable = []
            for bar in bars:
                b = bar.copy()
                if isinstance(b.get("timestamp"), datetime):
                    b["timestamp"] = b["timestamp"].isoformat()
                serializable.append(b)

            with open(cache_file, "w") as f:
                json.dump(serializable, f)
            logger.info(f"Cached {len(bars)} bars for {symbol}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> int:
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count

    def size(self) -> int:
        return sum(1 for _ in self.cache_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Split and dividend adjustment
# ---------------------------------------------------------------------------

class AdjustmentHandler:
    """Handle stock/token split and dividend adjustments."""

    @staticmethod
    def apply_splits(bars: List[Dict[str, Any]],
                      splits: List[Tuple[datetime, float]]) -> List[Dict[str, Any]]:
        """
        Apply split adjustments to historical data.

        splits: list of (date, ratio) where ratio is the split factor.
                e.g., (date, 2.0) means 2-for-1 split.
        """
        if not splits:
            return bars

        splits_sorted = sorted(splits, key=lambda x: x[0])
        adjusted = []

        for bar in bars:
            adj_bar = bar.copy()
            cumulative_ratio = 1.0

            for split_date, split_ratio in splits_sorted:
                if bar["timestamp"] < split_date:
                    cumulative_ratio *= split_ratio

            if cumulative_ratio != 1.0:
                for field in ("open", "high", "low", "close"):
                    adj_bar[field] = bar[field] / cumulative_ratio
                adj_bar["volume"] = bar["volume"] * cumulative_ratio

            adjusted.append(adj_bar)

        return adjusted

    @staticmethod
    def apply_dividends(bars: List[Dict[str, Any]],
                         dividends: List[Tuple[datetime, float]]) -> List[Dict[str, Any]]:
        """
        Apply dividend adjustments to historical data.

        dividends: list of (ex_date, dividend_amount_per_share).
        """
        if not dividends:
            return bars

        dividends_sorted = sorted(dividends, key=lambda x: x[0])
        adjusted = []

        for bar in bars:
            adj_bar = bar.copy()
            total_div = 0.0

            for ex_date, div_amount in dividends_sorted:
                if bar["timestamp"] < ex_date:
                    total_div += div_amount

            if total_div > 0:
                adj_factor = 1.0 - total_div / bar["close"] if bar["close"] > 0 else 1.0
                adj_factor = max(adj_factor, 0.01)  # prevent negative/zero
                for field_name in ("open", "high", "low", "close"):
                    adj_bar[field_name] = bar[field_name] * adj_factor

            adjusted.append(adj_bar)

        return adjusted


# ---------------------------------------------------------------------------
# Main data loader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Unified data loading interface.

    Supports CSV, Binance API, SQLite, PostgreSQL, and custom data sources
    with caching, validation, resampling, and multi-symbol alignment.

    Usage:
        loader = DataLoader()
        bars = loader.load_csv("data/BTCUSDT_1h.csv", "BTCUSDT")
        bars = loader.load_binance("BTCUSDT", "1h", start, end)
        bars = loader.load_database("ohlcv.db", "BTCUSDT")

        # Validate
        result = loader.validate(bars)

        # Resample
        bars_4h = loader.resample(bars, source_tf=60, target_tf=240)

        # Align multiple symbols
        aligned = loader.align({"BTC": btc_bars, "ETH": eth_bars})
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache = DataCache(cache_dir) if cache_dir else None
        self.validator = DataValidator()
        self.resampler = DataResampler()
        self.aligner = DataAligner()
        self.adjuster = AdjustmentHandler()
        self._custom_sources: Dict[str, DataSource] = {}

    def register_source(self, name: str, source: DataSource) -> None:
        """Register a custom data source."""
        self._custom_sources[name] = source

    def load_csv(self, filepath: str, symbol: str = "",
                 timeframe: str = "", delimiter: str = ",",
                 timestamp_format: Optional[str] = None,
                 start: Optional[datetime] = None,
                 end: Optional[datetime] = None,
                 validate: bool = True) -> List[Dict[str, Any]]:
        """Load data from a CSV file."""
        if not symbol:
            symbol = Path(filepath).stem.upper()

        # Check cache
        if self.cache:
            cached = self.cache.get(symbol, timeframe, start, end)
            if cached:
                return cached

        source = CSVDataSource(
            base_dir=str(Path(filepath).parent),
            delimiter=delimiter,
            timestamp_format=timestamp_format,
        )
        bars = source.fetch(str(Path(filepath).name), timeframe, start, end)

        if validate and bars:
            self._log_validation(symbol, bars)

        # Cache results
        if self.cache and bars:
            self.cache.put(symbol, timeframe, start, end, bars)

        return bars

    def load_binance(self, symbol: str, timeframe: str = "1h",
                     start: Optional[datetime] = None,
                     end: Optional[datetime] = None,
                     proxy_url: Optional[str] = None,
                     api_key: Optional[str] = None,
                     validate: bool = True) -> List[Dict[str, Any]]:
        """Load data from Binance API."""
        # Check cache
        if self.cache:
            cached = self.cache.get(symbol, timeframe, start, end)
            if cached:
                return cached

        source = BinanceDataSource(proxy_url=proxy_url, api_key=api_key)
        bars = source.fetch(symbol, timeframe, start, end)

        if validate and bars:
            self._log_validation(symbol, bars)

        if self.cache and bars:
            self.cache.put(symbol, timeframe, start, end, bars)

        return bars

    def load_database(self, connection_string: str, symbol: str,
                      timeframe: str = "", table_name: str = "ohlcv",
                      db_type: str = "sqlite",
                      start: Optional[datetime] = None,
                      end: Optional[datetime] = None,
                      validate: bool = True) -> List[Dict[str, Any]]:
        """Load data from a database."""
        if self.cache:
            cached = self.cache.get(symbol, timeframe, start, end)
            if cached:
                return cached

        source = DatabaseDataSource(
            connection_string=connection_string,
            table_name=table_name,
            db_type=db_type,
        )
        bars = source.fetch(symbol, timeframe, start, end)
        source.close()

        if validate and bars:
            self._log_validation(symbol, bars)

        if self.cache and bars:
            self.cache.put(symbol, timeframe, start, end, bars)

        return bars

    def load_custom(self, source_name: str, symbol: str,
                    timeframe: str = "",
                    start: Optional[datetime] = None,
                    end: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Load data from a registered custom source."""
        source = self._custom_sources.get(source_name)
        if source is None:
            raise ValueError(f"Unknown data source: {source_name}")
        return source.fetch(symbol, timeframe, start, end)

    def validate(self, bars: List[Dict[str, Any]],
                 timeframe_minutes: int = 60,
                 outlier_threshold: float = 5.0) -> DataValidationResult:
        """Validate a dataset."""
        validator = DataValidator(
            timeframe_minutes=timeframe_minutes,
            outlier_threshold=outlier_threshold,
        )
        return validator.validate(bars)

    def resample(self, bars: List[Dict[str, Any]],
                 source_tf: int, target_tf: int) -> List[Dict[str, Any]]:
        """Resample bars to a different timeframe (in minutes)."""
        return DataResampler.resample(bars, source_tf, target_tf)

    def resample_by_time(self, bars: List[Dict[str, Any]],
                          target_tf_minutes: int) -> List[Dict[str, Any]]:
        """Resample bars using time-based grouping."""
        return DataResampler.resample_by_time(bars, target_tf_minutes)

    def align(self, datasets: Dict[str, List[Dict[str, Any]]],
              method: str = "inner") -> Dict[str, List[Dict[str, Any]]]:
        """Align multiple symbol datasets to shared timestamps."""
        return DataAligner.align(datasets, method)

    def remove_duplicates(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate timestamps, keeping the last occurrence."""
        seen = {}
        for bar in bars:
            seen[bar["timestamp"]] = bar
        return sorted(seen.values(), key=lambda x: x["timestamp"])

    def fill_gaps(self, bars: List[Dict[str, Any]],
                  timeframe_minutes: int) -> List[Dict[str, Any]]:
        """Fill gaps in data with forward-filled bars (zero volume)."""
        if not bars:
            return bars

        delta = timedelta(minutes=timeframe_minutes)
        filled = [bars[0]]

        for i in range(1, len(bars)):
            prev_ts = bars[i - 1]["timestamp"]
            curr_ts = bars[i]["timestamp"]
            expected = prev_ts + delta

            while expected < curr_ts:
                gap_bar = bars[i - 1].copy()
                gap_bar["timestamp"] = expected
                gap_bar["open"] = gap_bar["close"]
                gap_bar["high"] = gap_bar["close"]
                gap_bar["low"] = gap_bar["close"]
                gap_bar["volume"] = 0
                gap_bar["quote_volume"] = 0
                gap_bar["trades_count"] = 0
                filled.append(gap_bar)
                expected += delta

            filled.append(bars[i])

        return filled

    def apply_splits(self, bars: List[Dict[str, Any]],
                      splits: List[Tuple[datetime, float]]) -> List[Dict[str, Any]]:
        """Apply split adjustments."""
        return AdjustmentHandler.apply_splits(bars, splits)

    def apply_dividends(self, bars: List[Dict[str, Any]],
                         dividends: List[Tuple[datetime, float]]) -> List[Dict[str, Any]]:
        """Apply dividend adjustments."""
        return AdjustmentHandler.apply_dividends(bars, dividends)

    def _log_validation(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        result = self.validator.validate(bars)
        if result.warnings:
            for w in result.warnings:
                logger.warning(f"[{symbol}] {w}")
        if result.errors:
            for e in result.errors:
                logger.error(f"[{symbol}] {e}")
        if result.is_valid:
            logger.info(
                f"[{symbol}] Data validation passed: {result.total_bars} bars"
            )
