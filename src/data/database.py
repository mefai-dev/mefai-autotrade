"""
Database Layer - Production-grade persistence for Mefai Autotrade.
Supports SQLite (default) with optional PostgreSQL backend.
Includes schema migrations, connection pooling, query helpers,
aggregate analytics, data export, and retention policies.
"""

import asyncio
import csv
import io
import json
import logging
import os
import shutil
import sqlite3
import time
import threading
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, AsyncIterator, Callable, Dict, Iterator, List,
    Optional, Set, Tuple, Union,
)

logger = logging.getLogger("mefai.autotrade.database")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatabaseConfig:
    """Database connection and behavior configuration."""
    db_type: str = "sqlite"
    sqlite_path: str = "autotrade.db"
    postgres_dsn: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    echo_sql: bool = False
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -64000  # 64MB
    busy_timeout: int = 5000
    retention_days: int = 365
    archive_enabled: bool = True
    archive_path: str = "archive"
    snapshot_interval_seconds: int = 300
    max_connections: int = 20

    def validate(self) -> List[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if self.db_type not in ("sqlite", "postgres"):
            errors.append(f"Unsupported db_type: {self.db_type}")
        if self.db_type == "sqlite" and not self.sqlite_path:
            errors.append("sqlite_path is required for sqlite")
        if self.db_type == "postgres" and not self.postgres_dsn:
            errors.append("postgres_dsn is required for postgres")
        if self.pool_size < 1:
            errors.append("pool_size must be >= 1")
        if self.retention_days < 1:
            errors.append("retention_days must be >= 1")
        return errors


# ---------------------------------------------------------------------------
# Schema version tracking
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = 5

MIGRATIONS: Dict[int, List[str]] = {
    1: [
        # --- trades ---
        """
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL NOT NULL,
            pnl REAL DEFAULT 0.0,
            commission REAL DEFAULT 0.0,
            duration_seconds REAL DEFAULT 0.0,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            entry_reason TEXT,
            exit_reason TEXT,
            tags TEXT DEFAULT '[]',
            meta TEXT DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)",
        "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time)",
        "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(side)",

        # --- orders ---
        """
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            trade_id TEXT,
            exchange_order_id TEXT,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            price REAL,
            stop_price REAL,
            quantity REAL NOT NULL,
            filled_quantity REAL DEFAULT 0.0,
            average_fill_price REAL,
            commission REAL DEFAULT 0.0,
            time_in_force TEXT DEFAULT 'GTC',
            client_order_id TEXT,
            exchange TEXT,
            reduce_only INTEGER DEFAULT 0,
            post_only INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            submitted_at TEXT,
            filled_at TEXT,
            cancelled_at TEXT,
            last_update TEXT,
            reject_reason TEXT,
            meta TEXT DEFAULT '{}',
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_orders_trade_id ON orders(trade_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
        "CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_orders_exchange_order_id ON orders(exchange_order_id)",

        # --- positions ---
        """
        CREATE TABLE IF NOT EXISTS positions (
            id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            quantity REAL NOT NULL,
            unrealized_pnl REAL DEFAULT 0.0,
            realized_pnl REAL DEFAULT 0.0,
            leverage REAL DEFAULT 1.0,
            margin_used REAL DEFAULT 0.0,
            liquidation_price REAL,
            stop_loss REAL,
            take_profit REAL,
            is_open INTEGER DEFAULT 1,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            last_update TEXT,
            meta TEXT DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_positions_is_open ON positions(is_open)",

        # --- equity_snapshots ---
        """
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_equity REAL NOT NULL,
            available_balance REAL NOT NULL,
            margin_used REAL DEFAULT 0.0,
            unrealized_pnl REAL DEFAULT 0.0,
            realized_pnl_today REAL DEFAULT 0.0,
            open_position_count INTEGER DEFAULT 0,
            drawdown_pct REAL DEFAULT 0.0,
            peak_equity REAL DEFAULT 0.0,
            meta TEXT DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp)",

        # --- strategy_performance ---
        """
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            period TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            gross_profit REAL DEFAULT 0.0,
            gross_loss REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0,
            max_drawdown_pct REAL DEFAULT 0.0,
            sharpe_ratio REAL,
            sortino_ratio REAL,
            profit_factor REAL,
            avg_win REAL DEFAULT 0.0,
            avg_loss REAL DEFAULT 0.0,
            largest_win REAL DEFAULT 0.0,
            largest_loss REAL DEFAULT 0.0,
            avg_duration_seconds REAL DEFAULT 0.0,
            win_rate REAL DEFAULT 0.0,
            expectancy REAL DEFAULT 0.0,
            meta TEXT DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_strat_perf_strategy ON strategy_performance(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_strat_perf_period ON strategy_performance(period_start, period_end)",

        # --- risk_events ---
        """
        CREATE TABLE IF NOT EXISTS risk_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'INFO',
            strategy TEXT,
            symbol TEXT,
            message TEXT NOT NULL,
            details TEXT DEFAULT '{}',
            action_taken TEXT,
            timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT,
            meta TEXT DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type)",
        "CREATE INDEX IF NOT EXISTS idx_risk_events_severity ON risk_events(severity)",
        "CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp)",

        # --- schema_version ---
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            description TEXT
        )
        """,
    ],

    2: [
        "ALTER TABLE trades ADD COLUMN exchange TEXT DEFAULT ''",
        "ALTER TABLE trades ADD COLUMN timeframe TEXT DEFAULT ''",
        "ALTER TABLE trades ADD COLUMN leverage REAL DEFAULT 1.0",
    ],

    3: [
        """
        CREATE TABLE IF NOT EXISTS trade_fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            trade_id TEXT,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            commission REAL DEFAULT 0.0,
            commission_asset TEXT DEFAULT '',
            timestamp TEXT NOT NULL,
            is_maker INTEGER DEFAULT 0,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_fills_order ON trade_fills(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_fills_trade ON trade_fills(trade_id)",
    ],

    4: [
        "ALTER TABLE positions ADD COLUMN trailing_stop_distance REAL",
        "ALTER TABLE positions ADD COLUMN trailing_stop_active INTEGER DEFAULT 0",
        "ALTER TABLE positions ADD COLUMN max_price_seen REAL",
        "ALTER TABLE positions ADD COLUMN min_price_seen REAL",
    ],

    5: [
        """
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            total_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            total_commission REAL DEFAULT 0.0,
            starting_equity REAL DEFAULT 0.0,
            ending_equity REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0,
            strategies_active TEXT DEFAULT '[]',
            symbols_traded TEXT DEFAULT '[]',
            meta TEXT DEFAULT '{}'
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date)",
    ],
}


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Represents a completed or in-progress trade."""
    id: str = ""
    strategy: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    duration_seconds: float = 0.0
    entry_time: str = ""
    exit_time: Optional[str] = None
    entry_reason: str = ""
    exit_reason: str = ""
    exchange: str = ""
    timeframe: str = ""
    leverage: float = 1.0
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tags"] = json.dumps(d["tags"])
        d["meta"] = json.dumps(d["meta"])
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "TradeRecord":
        row = dict(row)
        row["tags"] = json.loads(row.get("tags", "[]") or "[]")
        row["meta"] = json.loads(row.get("meta", "{}") or "{}")
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


@dataclass
class OrderRecord:
    """Represents an order in any state."""
    id: str = ""
    trade_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    strategy: str = ""
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    status: str = "PENDING"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    quantity: float = 0.0
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    exchange: Optional[str] = None
    reduce_only: bool = False
    post_only: bool = False
    created_at: str = ""
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    last_update: Optional[str] = None
    reject_reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = json.dumps(d["meta"])
        d["reduce_only"] = int(d["reduce_only"])
        d["post_only"] = int(d["post_only"])
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "OrderRecord":
        row = dict(row)
        row["meta"] = json.loads(row.get("meta", "{}") or "{}")
        row["reduce_only"] = bool(row.get("reduce_only", 0))
        row["post_only"] = bool(row.get("post_only", 0))
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


@dataclass
class PositionRecord:
    """Current or historical position."""
    id: str = ""
    strategy: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    current_price: Optional[float] = None
    quantity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin_used: float = 0.0
    liquidation_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    is_open: bool = True
    opened_at: str = ""
    closed_at: Optional[str] = None
    last_update: Optional[str] = None
    trailing_stop_distance: Optional[float] = None
    trailing_stop_active: bool = False
    max_price_seen: Optional[float] = None
    min_price_seen: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = json.dumps(d["meta"])
        d["is_open"] = int(d["is_open"])
        d["trailing_stop_active"] = int(d["trailing_stop_active"])
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PositionRecord":
        row = dict(row)
        row["meta"] = json.loads(row.get("meta", "{}") or "{}")
        row["is_open"] = bool(row.get("is_open", 1))
        row["trailing_stop_active"] = bool(row.get("trailing_stop_active", 0))
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


@dataclass
class EquitySnapshot:
    """Point-in-time equity snapshot."""
    id: Optional[int] = None
    timestamp: str = ""
    total_equity: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    open_position_count: int = 0
    drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = json.dumps(d["meta"])
        if d["id"] is None:
            del d["id"]
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "EquitySnapshot":
        row = dict(row)
        row["meta"] = json.loads(row.get("meta", "{}") or "{}")
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyPerformance:
    """Performance summary for a strategy over a period."""
    id: Optional[int] = None
    strategy: str = ""
    period: str = ""
    period_start: str = ""
    period_end: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_duration_seconds: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = json.dumps(d["meta"])
        if d["id"] is None:
            del d["id"]
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "StrategyPerformance":
        row = dict(row)
        row["meta"] = json.loads(row.get("meta", "{}") or "{}")
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskEvent:
    """Risk management event."""
    id: Optional[int] = None
    event_type: str = ""
    severity: str = "INFO"
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[str] = None
    timestamp: str = ""
    resolved: bool = False
    resolved_at: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["details"] = json.dumps(d["details"])
        d["meta"] = json.dumps(d["meta"])
        d["resolved"] = int(d["resolved"])
        if d["id"] is None:
            del d["id"]
        return d

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "RiskEvent":
        row = dict(row)
        row["details"] = json.loads(row.get("details", "{}") or "{}")
        row["meta"] = json.loads(row.get("meta", "{}") or "{}")
        row["resolved"] = bool(row.get("resolved", 0))
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Query filter
# ---------------------------------------------------------------------------

@dataclass
class QueryFilter:
    """Generic query filter for list operations."""
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    status: Optional[str] = None
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None
    exchange: Optional[str] = None
    limit: int = 1000
    offset: int = 0
    order_by: str = "created_at"
    order_dir: str = "DESC"
    tags: Optional[List[str]] = None

    def build_where(self, time_column: str = "entry_time") -> Tuple[str, List[Any]]:
        """Build WHERE clause and params list."""
        clauses: List[str] = []
        params: List[Any] = []

        if self.strategy:
            clauses.append("strategy = ?")
            params.append(self.strategy)
        if self.symbol:
            clauses.append("symbol = ?")
            params.append(self.symbol)
        if self.side:
            clauses.append("side = ?")
            params.append(self.side)
        if self.start_time:
            clauses.append(f"{time_column} >= ?")
            params.append(self.start_time)
        if self.end_time:
            clauses.append(f"{time_column} <= ?")
            params.append(self.end_time)
        if self.status:
            clauses.append("status = ?")
            params.append(self.status)
        if self.min_pnl is not None:
            clauses.append("pnl >= ?")
            params.append(self.min_pnl)
        if self.max_pnl is not None:
            clauses.append("pnl <= ?")
            params.append(self.max_pnl)
        if self.exchange:
            clauses.append("exchange = ?")
            params.append(self.exchange)

        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

class ConnectionPool:
    """Thread-safe SQLite connection pool."""

    def __init__(self, db_path: str, max_size: int = 10, pragmas: Optional[Dict[str, Any]] = None):
        self._db_path = db_path
        self._max_size = max_size
        self._pragmas = pragmas or {}
        self._pool: List[sqlite3.Connection] = []
        self._in_use: Set[int] = set()
        self._lock = threading.Lock()
        self._created = 0

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        for pragma, value in self._pragmas.items():
            conn.execute(f"PRAGMA {pragma}={value}")
        self._created += 1
        return conn

    def acquire(self) -> sqlite3.Connection:
        with self._lock:
            for i, conn in enumerate(self._pool):
                if id(conn) not in self._in_use:
                    self._in_use.add(id(conn))
                    return conn

            if self._created < self._max_size:
                conn = self._create_connection()
                self._pool.append(conn)
                self._in_use.add(id(conn))
                return conn

        # All connections busy - wait and retry
        for _ in range(100):
            time.sleep(0.05)
            with self._lock:
                for conn in self._pool:
                    if id(conn) not in self._in_use:
                        self._in_use.add(id(conn))
                        return conn

        raise RuntimeError("Connection pool exhausted after timeout")

    def release(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            self._in_use.discard(id(conn))

    def close_all(self) -> None:
        with self._lock:
            for conn in self._pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._pool.clear()
            self._in_use.clear()
            self._created = 0

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total": len(self._pool),
                "in_use": len(self._in_use),
                "available": len(self._pool) - len(self._in_use),
                "max_size": self._max_size,
            }


# ---------------------------------------------------------------------------
# Migration manager
# ---------------------------------------------------------------------------

class MigrationManager:
    """Schema versioning and migration execution."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def get_current_version(self) -> int:
        with self._pool.connection() as conn:
            try:
                cursor = conn.execute(
                    "SELECT MAX(version) FROM schema_version"
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] is not None else 0
            except sqlite3.OperationalError:
                return 0

    def apply_migrations(self, target_version: Optional[int] = None) -> List[int]:
        """Apply pending migrations up to target_version. Returns list of applied versions."""
        if target_version is None:
            target_version = CURRENT_SCHEMA_VERSION

        current = self.get_current_version()
        applied: List[int] = []

        if current >= target_version:
            logger.info(
                "Database schema is up to date (version %d)", current
            )
            return applied

        with self._pool.connection() as conn:
            for version in range(current + 1, target_version + 1):
                if version not in MIGRATIONS:
                    logger.warning("Migration %d not found, skipping", version)
                    continue

                logger.info("Applying migration %d...", version)
                statements = MIGRATIONS[version]

                try:
                    for stmt in statements:
                        conn.execute(stmt)

                    conn.execute(
                        "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                        (version, f"Migration v{version}"),
                    )
                    conn.commit()
                    applied.append(version)
                    logger.info("Migration %d applied successfully", version)
                except Exception as exc:
                    conn.rollback()
                    logger.error("Migration %d failed: %s", version, exc)
                    raise

        return applied

    def get_migration_history(self) -> List[Dict[str, Any]]:
        with self._pool.connection() as conn:
            try:
                cursor = conn.execute(
                    "SELECT version, applied_at, description FROM schema_version ORDER BY version"
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []


# ---------------------------------------------------------------------------
# Data exporter
# ---------------------------------------------------------------------------

class DataExporter:
    """Export data to CSV and JSON formats."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def trades_to_csv(self, qf: Optional[QueryFilter] = None) -> str:
        qf = qf or QueryFilter()
        where, params = qf.build_where("entry_time")

        with self._pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM trades WHERE {where} "
                f"ORDER BY {qf.order_by} {qf.order_dir} "
                f"LIMIT ? OFFSET ?",
                params + [qf.limit, qf.offset],
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
        return output.getvalue()

    def trades_to_json(self, qf: Optional[QueryFilter] = None) -> str:
        qf = qf or QueryFilter()
        where, params = qf.build_where("entry_time")

        with self._pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM trades WHERE {where} "
                f"ORDER BY {qf.order_by} {qf.order_dir} "
                f"LIMIT ? OFFSET ?",
                params + [qf.limit, qf.offset],
            )
            rows = cursor.fetchall()

        records = []
        for row in rows:
            d = dict(row)
            for key in ("tags", "meta"):
                if key in d and isinstance(d[key], str):
                    try:
                        d[key] = json.loads(d[key])
                    except (json.JSONDecodeError, TypeError):
                        pass
            records.append(d)

        return json.dumps(records, indent=2, default=str)

    def orders_to_csv(self, qf: Optional[QueryFilter] = None) -> str:
        qf = qf or QueryFilter()
        where, params = qf.build_where("created_at")

        with self._pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM orders WHERE {where} "
                f"ORDER BY created_at {qf.order_dir} "
                f"LIMIT ? OFFSET ?",
                params + [qf.limit, qf.offset],
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
        return output.getvalue()

    def equity_to_json(self, start: Optional[str] = None, end: Optional[str] = None) -> str:
        clauses = []
        params: List[Any] = []
        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)
        where = " AND ".join(clauses) if clauses else "1=1"

        with self._pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM equity_snapshots WHERE {where} ORDER BY timestamp",
                params,
            )
            rows = cursor.fetchall()

        records = [dict(row) for row in rows]
        return json.dumps(records, indent=2, default=str)

    def export_to_file(self, table: str, filepath: str, fmt: str = "csv") -> int:
        """Export a full table to a file. Returns row count."""
        with self._pool.connection() as conn:
            cursor = conn.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()

        if not rows:
            return 0

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
        elif fmt == "json":
            with open(filepath, "w") as f:
                json.dump([dict(row) for row in rows], f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return len(rows)


# ---------------------------------------------------------------------------
# Main database manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Central database management class.
    Handles connections, schema, CRUD operations, aggregation queries,
    data export, and retention policies.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self._config = config or DatabaseConfig()
        errors = self._config.validate()
        if errors:
            raise ValueError(f"Invalid database config: {'; '.join(errors)}")

        self._pool: Optional[ConnectionPool] = None
        self._migration_mgr: Optional[MigrationManager] = None
        self._exporter: Optional[DataExporter] = None
        self._initialized = False
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None

    # --- Lifecycle ---

    def initialize(self) -> None:
        """Initialize the database, create pool, apply migrations."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if self._config.db_type == "sqlite":
                db_dir = os.path.dirname(self._config.sqlite_path)
                if db_dir:
                    os.makedirs(db_dir, exist_ok=True)

                pragmas = {
                    "journal_mode": self._config.journal_mode,
                    "synchronous": self._config.synchronous,
                    "cache_size": self._config.cache_size,
                    "busy_timeout": self._config.busy_timeout,
                }

                self._pool = ConnectionPool(
                    db_path=self._config.sqlite_path,
                    max_size=self._config.pool_size,
                    pragmas=pragmas,
                )

                self._migration_mgr = MigrationManager(self._pool)
                self._migration_mgr.apply_migrations()

                self._exporter = DataExporter(self._pool)
                self._initialized = True

                logger.info(
                    "Database initialized at %s (schema v%d)",
                    self._config.sqlite_path,
                    self._migration_mgr.get_current_version(),
                )
            else:
                raise NotImplementedError(
                    "PostgreSQL backend is not yet implemented. "
                    "Use db_type='sqlite' or contribute a PostgreSQL adapter."
                )

    def close(self) -> None:
        """Close all connections and release resources."""
        if self._pool:
            self._pool.close_all()
        self._initialized = False
        logger.info("Database connections closed")

    @property
    def pool(self) -> ConnectionPool:
        if not self._pool:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._pool

    @property
    def exporter(self) -> DataExporter:
        if not self._exporter:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._exporter

    @property
    def migrations(self) -> MigrationManager:
        if not self._migration_mgr:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._migration_mgr

    # --- Trade CRUD ---

    def insert_trade(self, trade: TradeRecord) -> str:
        """Insert a new trade record. Returns trade id."""
        data = trade.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        with self.pool.connection() as conn:
            conn.execute(
                f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()

        logger.debug("Inserted trade %s", trade.id)
        return trade.id

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        """Update trade fields. Returns True if a row was updated."""
        if "meta" in updates and isinstance(updates["meta"], dict):
            updates["meta"] = json.dumps(updates["meta"])
        if "tags" in updates and isinstance(updates["tags"], list):
            updates["tags"] = json.dumps(updates["tags"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        params = list(updates.values()) + [trade_id]

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"UPDATE trades SET {set_clause} WHERE id = ?", params
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        with self.pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
        return TradeRecord.from_row(dict(row)) if row else None

    def get_trades(self, qf: Optional[QueryFilter] = None) -> List[TradeRecord]:
        qf = qf or QueryFilter()
        where, params = qf.build_where("entry_time")

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM trades WHERE {where} "
                f"ORDER BY {qf.order_by} {qf.order_dir} "
                f"LIMIT ? OFFSET ?",
                params + [qf.limit, qf.offset],
            )
            rows = cursor.fetchall()

        return [TradeRecord.from_row(dict(row)) for row in rows]

    def delete_trade(self, trade_id: str) -> bool:
        with self.pool.connection() as conn:
            cursor = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count_trades(self, qf: Optional[QueryFilter] = None) -> int:
        qf = qf or QueryFilter()
        where, params = qf.build_where("entry_time")
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM trades WHERE {where}", params
            )
            return cursor.fetchone()[0]

    # --- Order CRUD ---

    def insert_order(self, order: OrderRecord) -> str:
        data = order.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        with self.pool.connection() as conn:
            conn.execute(
                f"INSERT INTO orders ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()

        logger.debug("Inserted order %s", order.id)
        return order.id

    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        if "meta" in updates and isinstance(updates["meta"], dict):
            updates["meta"] = json.dumps(updates["meta"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        params = list(updates.values()) + [order_id]

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"UPDATE orders SET {set_clause} WHERE id = ?", params
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        with self.pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
            row = cursor.fetchone()
        return OrderRecord.from_row(dict(row)) if row else None

    def get_orders(self, qf: Optional[QueryFilter] = None) -> List[OrderRecord]:
        qf = qf or QueryFilter()
        where, params = qf.build_where("created_at")

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM orders WHERE {where} "
                f"ORDER BY created_at {qf.order_dir} "
                f"LIMIT ? OFFSET ?",
                params + [qf.limit, qf.offset],
            )
            rows = cursor.fetchall()

        return [OrderRecord.from_row(dict(row)) for row in rows]

    def get_orders_by_trade(self, trade_id: str) -> List[OrderRecord]:
        with self.pool.connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM orders WHERE trade_id = ? ORDER BY created_at",
                (trade_id,),
            )
            rows = cursor.fetchall()
        return [OrderRecord.from_row(dict(row)) for row in rows]

    def get_active_orders(self, strategy: Optional[str] = None) -> List[OrderRecord]:
        clauses = ["status IN ('PENDING', 'SUBMITTED', 'PARTIALLY_FILLED')"]
        params: List[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)

        where = " AND ".join(clauses)
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM orders WHERE {where} ORDER BY created_at", params
            )
            rows = cursor.fetchall()
        return [OrderRecord.from_row(dict(row)) for row in rows]

    # --- Position CRUD ---

    def insert_position(self, pos: PositionRecord) -> str:
        data = pos.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        with self.pool.connection() as conn:
            conn.execute(
                f"INSERT INTO positions ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()

        logger.debug("Inserted position %s", pos.id)
        return pos.id

    def update_position(self, pos_id: str, updates: Dict[str, Any]) -> bool:
        if "meta" in updates and isinstance(updates["meta"], dict):
            updates["meta"] = json.dumps(updates["meta"])
        if "is_open" in updates and isinstance(updates["is_open"], bool):
            updates["is_open"] = int(updates["is_open"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        params = list(updates.values()) + [pos_id]

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"UPDATE positions SET {set_clause} WHERE id = ?", params
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_position(self, pos_id: str) -> Optional[PositionRecord]:
        with self.pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM positions WHERE id = ?", (pos_id,))
            row = cursor.fetchone()
        return PositionRecord.from_row(dict(row)) if row else None

    def get_open_positions(self, strategy: Optional[str] = None) -> List[PositionRecord]:
        clauses = ["is_open = 1"]
        params: List[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)

        where = " AND ".join(clauses)
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM positions WHERE {where} ORDER BY opened_at", params
            )
            rows = cursor.fetchall()
        return [PositionRecord.from_row(dict(row)) for row in rows]

    def get_positions(self, qf: Optional[QueryFilter] = None) -> List[PositionRecord]:
        qf = qf or QueryFilter()
        where, params = qf.build_where("opened_at")

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM positions WHERE {where} "
                f"ORDER BY opened_at {qf.order_dir} "
                f"LIMIT ? OFFSET ?",
                params + [qf.limit, qf.offset],
            )
            rows = cursor.fetchall()

        return [PositionRecord.from_row(dict(row)) for row in rows]

    # --- Equity snapshots ---

    def insert_equity_snapshot(self, snap: EquitySnapshot) -> int:
        data = snap.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO equity_snapshots ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_equity_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10000,
    ) -> List[EquitySnapshot]:
        clauses: List[str] = []
        params: List[Any] = []
        if start:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end:
            clauses.append("timestamp <= ?")
            params.append(end)

        where = " AND ".join(clauses) if clauses else "1=1"
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM equity_snapshots WHERE {where} "
                f"ORDER BY timestamp LIMIT ?",
                params + [limit],
            )
            rows = cursor.fetchall()
        return [EquitySnapshot.from_row(dict(row)) for row in rows]

    # --- Strategy performance ---

    def insert_strategy_performance(self, perf: StrategyPerformance) -> int:
        data = perf.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO strategy_performance ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_strategy_performance(
        self, strategy: str, period: Optional[str] = None
    ) -> List[StrategyPerformance]:
        clauses = ["strategy = ?"]
        params: List[Any] = [strategy]
        if period:
            clauses.append("period = ?")
            params.append(period)

        where = " AND ".join(clauses)
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM strategy_performance WHERE {where} "
                f"ORDER BY period_start DESC",
                params,
            )
            rows = cursor.fetchall()
        return [StrategyPerformance.from_row(dict(row)) for row in rows]

    # --- Risk events ---

    def insert_risk_event(self, event: RiskEvent) -> int:
        data = event.to_dict()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))

        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO risk_events ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_risk_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        unresolved_only: bool = False,
        limit: int = 100,
    ) -> List[RiskEvent]:
        clauses: List[str] = []
        params: List[Any] = []

        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if unresolved_only:
            clauses.append("resolved = 0")

        where = " AND ".join(clauses) if clauses else "1=1"
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"SELECT * FROM risk_events WHERE {where} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params + [limit],
            )
            rows = cursor.fetchall()
        return [RiskEvent.from_row(dict(row)) for row in rows]

    def resolve_risk_event(self, event_id: int) -> bool:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        with self.pool.connection() as conn:
            cursor = conn.execute(
                "UPDATE risk_events SET resolved = 1, resolved_at = ? WHERE id = ?",
                (now, event_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    # --- Aggregate queries ---

    def daily_pnl(self, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get daily PnL aggregation."""
        clauses: List[str] = ["exit_time IS NOT NULL"]
        params: List[Any] = []
        if start:
            clauses.append("exit_time >= ?")
            params.append(start)
        if end:
            clauses.append("exit_time <= ?")
            params.append(end)

        where = " AND ".join(clauses)
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    DATE(exit_time) as date,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    SUM(commission) as total_commission,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE {where}
                GROUP BY DATE(exit_time)
                ORDER BY date
                """,
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def monthly_pnl(self, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get monthly PnL aggregation."""
        clauses: List[str] = ["exit_time IS NOT NULL"]
        params: List[Any] = []
        if start:
            clauses.append("exit_time >= ?")
            params.append(start)
        if end:
            clauses.append("exit_time <= ?")
            params.append(end)

        where = " AND ".join(clauses)
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    STRFTIME('%Y-%m', exit_time) as month,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    SUM(commission) as total_commission,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl) as avg_pnl
                FROM trades
                WHERE {where}
                GROUP BY STRFTIME('%Y-%m', exit_time)
                ORDER BY month
                """,
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def strategy_pnl(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get per-strategy PnL summary."""
        clauses: List[str] = ["exit_time IS NOT NULL"]
        params: List[Any] = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)

        where = " AND ".join(clauses)
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    strategy,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    SUM(commission) as total_commission,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    ROUND(SUM(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) as win_rate,
                    AVG(pnl) as avg_pnl,
                    AVG(duration_seconds) as avg_duration,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE {where}
                GROUP BY strategy
                ORDER BY total_pnl DESC
                """,
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def symbol_pnl(self) -> List[Dict[str, Any]]:
        """Get per-symbol PnL summary."""
        with self.pool.connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    symbol,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(pnl) as avg_pnl
                FROM trades
                WHERE exit_time IS NOT NULL
                GROUP BY symbol
                ORDER BY total_pnl DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_drawdown_series(self) -> List[Dict[str, Any]]:
        """Compute cumulative PnL and drawdown series from equity snapshots."""
        with self.pool.connection() as conn:
            cursor = conn.execute(
                "SELECT timestamp, total_equity, peak_equity, drawdown_pct "
                "FROM equity_snapshots ORDER BY timestamp"
            )
            return [dict(row) for row in cursor.fetchall()]

    def trade_duration_distribution(self, bucket_seconds: int = 3600) -> List[Dict[str, Any]]:
        """Get distribution of trade durations in buckets."""
        with self.pool.connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    CAST(duration_seconds / ? AS INTEGER) * ? as bucket_start,
                    COUNT(*) as count,
                    AVG(pnl) as avg_pnl
                FROM trades
                WHERE exit_time IS NOT NULL AND duration_seconds > 0
                GROUP BY bucket_start
                ORDER BY bucket_start
                """,
                (bucket_seconds, bucket_seconds),
            )
            return [dict(row) for row in cursor.fetchall()]

    # --- Data retention ---

    def apply_retention_policy(self) -> Dict[str, int]:
        """Archive and delete data older than retention_days. Returns counts."""
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=self._config.retention_days)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        counts: Dict[str, int] = {}

        if self._config.archive_enabled:
            archive_dir = Path(self._config.archive_path)
            archive_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

            # Archive old trades
            with self.pool.connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE exit_time < ? AND exit_time IS NOT NULL",
                    (cutoff,),
                )
                old_trades = cursor.fetchall()

            if old_trades:
                archive_file = archive_dir / f"trades_{timestamp}.json"
                with open(archive_file, "w") as f:
                    json.dump(
                        [dict(row) for row in old_trades], f, indent=2, default=str
                    )
                logger.info("Archived %d trades to %s", len(old_trades), archive_file)

        # Delete old records
        tables_and_columns = {
            "trades": "exit_time",
            "orders": "created_at",
            "equity_snapshots": "timestamp",
            "risk_events": "timestamp",
        }

        with self.pool.connection() as conn:
            for table, col in tables_and_columns.items():
                if table == "trades":
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE {col} < ? AND {col} IS NOT NULL",
                        (cutoff,),
                    )
                else:
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE {col} < ?", (cutoff,)
                    )
                counts[table] = cursor.rowcount

            conn.commit()

        # Vacuum to reclaim space
        with self.pool.connection() as conn:
            conn.execute("VACUUM")

        logger.info("Retention policy applied. Deleted: %s", counts)
        return counts

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats: Dict[str, Any] = {"pool": self.pool.stats}

        with self.pool.connection() as conn:
            for table in ("trades", "orders", "positions", "equity_snapshots",
                          "strategy_performance", "risk_events"):
                try:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    stats[f"{table}_count"] = 0

            # Database file size
            if self._config.db_type == "sqlite":
                try:
                    stats["file_size_mb"] = round(
                        os.path.getsize(self._config.sqlite_path) / (1024 * 1024), 2
                    )
                except OSError:
                    stats["file_size_mb"] = 0

            # Schema version
            if self._migration_mgr:
                stats["schema_version"] = self._migration_mgr.get_current_version()

        return stats

    # --- Async wrappers ---

    async def async_insert_trade(self, trade: TradeRecord) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.insert_trade, trade)

    async def async_get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_trade, trade_id)

    async def async_get_trades(self, qf: Optional[QueryFilter] = None) -> List[TradeRecord]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_trades, qf)

    async def async_insert_order(self, order: OrderRecord) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.insert_order, order)

    async def async_get_orders(self, qf: Optional[QueryFilter] = None) -> List[OrderRecord]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_orders, qf)

    async def async_insert_equity_snapshot(self, snap: EquitySnapshot) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.insert_equity_snapshot, snap)

    async def async_insert_risk_event(self, event: RiskEvent) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.insert_risk_event, event)

    async def async_daily_pnl(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.daily_pnl, start, end)

    async def async_apply_retention(self) -> Dict[str, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.apply_retention_policy)

    # --- Utility ---

    def execute_raw(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL and return results. Use with caution."""
        with self.pool.connection() as conn:
            cursor = conn.execute(sql, params or [])
            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            conn.commit()
            return []

    def backup(self, dest_path: str) -> None:
        """Create a hot backup of the database."""
        if self._config.db_type != "sqlite":
            raise NotImplementedError("Backup only supported for SQLite")

        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        with self.pool.connection() as conn:
            backup_conn = sqlite3.connect(dest_path)
            try:
                conn.backup(backup_conn)
                logger.info("Database backed up to %s", dest_path)
            finally:
                backup_conn.close()

    def __enter__(self) -> "DatabaseManager":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
