"""
Event Store - Append-only event sourcing and audit trail for Mefai Autotrade.
Provides event logging, replay, querying, compression, checkpointing,
and export for post-trade analysis and state reconstruction.
"""

import gzip
import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger("mefai.autotrade.event_store")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    """Categories of events in the system."""
    TRADE = "TRADE"
    ORDER = "ORDER"
    SIGNAL = "SIGNAL"
    RISK_EVENT = "RISK_EVENT"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    POSITION = "POSITION"
    FILL = "FILL"
    CANCEL = "CANCEL"
    ERROR = "ERROR"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    STRATEGY_EVENT = "STRATEGY_EVENT"
    MARKET_DATA = "MARKET_DATA"


class EventSeverity(Enum):
    """Severity levels for events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Stored event
# ---------------------------------------------------------------------------

@dataclass
class StoredEvent:
    """Immutable event record in the event store."""
    id: Optional[int] = None
    event_type: str = ""
    severity: str = "INFO"
    timestamp: float = 0.0
    strategy: str = ""
    symbol: str = ""
    source: str = ""
    action: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    correlation_id: str = ""
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "source": self.source,
            "action": self.action,
            "data": self.data,
            "sequence": self.sequence,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StoredEvent":
        data = d.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            id=d.get("id"),
            event_type=d.get("event_type", ""),
            severity=d.get("severity", "INFO"),
            timestamp=d.get("timestamp", 0.0),
            strategy=d.get("strategy", ""),
            symbol=d.get("symbol", ""),
            source=d.get("source", ""),
            action=d.get("action", ""),
            data=data,
            sequence=d.get("sequence", 0),
            correlation_id=d.get("correlation_id", ""),
            session_id=d.get("session_id", ""),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredEvent":
        d = dict(row)
        d["data"] = json.loads(d.get("data", "{}") or "{}")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Event query
# ---------------------------------------------------------------------------

@dataclass
class EventQuery:
    """Query parameters for event retrieval."""
    event_type: Optional[str] = None
    severity: Optional[str] = None
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    source: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = 1000
    offset: int = 0
    order: str = "ASC"

    def build_where(self) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []

        if self.event_type:
            clauses.append("event_type = ?")
            params.append(self.event_type)
        if self.severity:
            clauses.append("severity = ?")
            params.append(self.severity)
        if self.strategy:
            clauses.append("strategy = ?")
            params.append(self.strategy)
        if self.symbol:
            clauses.append("symbol = ?")
            params.append(self.symbol)
        if self.source:
            clauses.append("source = ?")
            params.append(self.source)
        if self.start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(self.start_time)
        if self.end_time is not None:
            clauses.append("timestamp <= ?")
            params.append(self.end_time)
        if self.correlation_id:
            clauses.append("correlation_id = ?")
            params.append(self.correlation_id)
        if self.session_id:
            clauses.append("session_id = ?")
            params.append(self.session_id)

        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params


# ---------------------------------------------------------------------------
# Event checkpoint
# ---------------------------------------------------------------------------

@dataclass
class EventCheckpoint:
    """State snapshot for faster event replay."""
    id: Optional[int] = None
    checkpoint_name: str = ""
    timestamp: float = 0.0
    last_event_id: int = 0
    last_sequence: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "checkpoint_name": self.checkpoint_name,
            "timestamp": self.timestamp,
            "last_event_id": self.last_event_id,
            "last_sequence": self.last_sequence,
            "state": self.state,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Event replayer
# ---------------------------------------------------------------------------

class EventReplayer:
    """
    Replay events to reconstruct state.
    Supports handler registration per event type for state reconstruction.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._state: Dict[str, Any] = {}
        self._events_replayed = 0

    def register_handler(self, event_type: str, handler: Callable[[StoredEvent, Dict], None]) -> None:
        """Register a handler for a specific event type during replay."""
        self._handlers[event_type].append(handler)

    def replay(
        self,
        events: Sequence[StoredEvent],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Replay a sequence of events through registered handlers.
        Returns the final state.
        """
        self._state = dict(initial_state) if initial_state else {}
        self._events_replayed = 0

        for event in events:
            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event, self._state)
                except Exception as exc:
                    logger.error(
                        "Replay handler error for event %s (seq %d): %s",
                        event.event_type, event.sequence, exc,
                    )
            self._events_replayed += 1

        logger.info("Replayed %d events", self._events_replayed)
        return dict(self._state)

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    @property
    def events_replayed(self) -> int:
        return self._events_replayed

    def reset(self) -> None:
        self._state.clear()
        self._events_replayed = 0


# ---------------------------------------------------------------------------
# Event compressor
# ---------------------------------------------------------------------------

class EventCompressor:
    """Compress and archive old events to save space."""

    @staticmethod
    def compress_to_file(events: Sequence[StoredEvent], filepath: str) -> int:
        """
        Compress events to a gzipped JSON file.
        Returns number of events compressed.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        records = [event.to_dict() for event in events]
        json_data = json.dumps(records, default=str).encode("utf-8")

        with gzip.open(filepath, "wb") as f:
            f.write(json_data)

        logger.info(
            "Compressed %d events to %s (%.1f KB)",
            len(records), filepath, len(json_data) / 1024,
        )
        return len(records)

    @staticmethod
    def decompress_from_file(filepath: str) -> List[StoredEvent]:
        """Decompress events from a gzipped JSON file."""
        with gzip.open(filepath, "rb") as f:
            data = f.read()

        records = json.loads(data.decode("utf-8"))
        return [StoredEvent.from_dict(r) for r in records]

    @staticmethod
    def export_to_json(events: Sequence[StoredEvent], filepath: str) -> int:
        """Export events to a plain JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        records = [event.to_dict() for event in events]
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2, default=str)
        return len(records)

    @staticmethod
    def import_from_json(filepath: str) -> List[StoredEvent]:
        """Import events from a JSON file."""
        with open(filepath, "r") as f:
            records = json.load(f)
        return [StoredEvent.from_dict(r) for r in records]


# ---------------------------------------------------------------------------
# Main event store
# ---------------------------------------------------------------------------

class EventStore:
    """
    Append-only event store with SQLite backend.
    Provides event logging, querying, replay, checkpointing,
    compression, and export.
    """

    CREATE_TABLES = [
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'INFO',
            timestamp REAL NOT NULL,
            strategy TEXT DEFAULT '',
            symbol TEXT DEFAULT '',
            source TEXT DEFAULT '',
            action TEXT DEFAULT '',
            data TEXT DEFAULT '{}',
            sequence INTEGER NOT NULL,
            correlation_id TEXT DEFAULT '',
            session_id TEXT DEFAULT ''
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)",
        "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_events_strategy ON events(strategy)",
        "CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_events_sequence ON events(sequence)",
        "CREATE INDEX IF NOT EXISTS idx_events_corr_id ON events(correlation_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_name TEXT NOT NULL,
            timestamp REAL NOT NULL,
            last_event_id INTEGER NOT NULL,
            last_sequence INTEGER NOT NULL,
            state TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_checkpoints_name ON checkpoints(checkpoint_name)",
        "CREATE INDEX IF NOT EXISTS idx_checkpoints_ts ON checkpoints(timestamp)",
    ]

    def __init__(self, db_path: str = "events.db", session_id: str = ""):
        self._db_path = db_path
        self._session_id = session_id
        self._sequence = 0
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._total_events = 0

        self._initialize_db()
        logger.info("EventStore initialized at %s (session: %s)", db_path, session_id)

    def _initialize_db(self) -> None:
        db_dir = Path(self._db_path).parent
        if str(db_dir) != ".":
            db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        for sql in self.CREATE_TABLES:
            self._conn.execute(sql)
        self._conn.commit()

        # Get last sequence number
        cursor = self._conn.execute("SELECT MAX(sequence) FROM events")
        row = cursor.fetchone()
        if row and row[0] is not None:
            self._sequence = row[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM events")
        self._total_events = cursor.fetchone()[0]

    # --- Event listeners ---

    def on_event(self, event_type: str, callback: Callable[[StoredEvent], None]) -> None:
        """Register a listener for a specific event type."""
        self._listeners[event_type].append(callback)

    def on_all_events(self, callback: Callable[[StoredEvent], None]) -> None:
        """Register a listener for all events."""
        self._listeners["*"].append(callback)

    def _notify_listeners(self, event: StoredEvent) -> None:
        for cb in self._listeners.get(event.event_type, []):
            try:
                cb(event)
            except Exception as exc:
                logger.error("Event listener error: %s", exc)
        for cb in self._listeners.get("*", []):
            try:
                cb(event)
            except Exception as exc:
                logger.error("Event listener error: %s", exc)

    # --- Append events ---

    def append(self, event: StoredEvent) -> int:
        """
        Append an event to the store. Returns the event ID.
        This is the primary write operation - append only, never update.
        """
        with self._lock:
            self._sequence += 1
            event.sequence = self._sequence
            event.session_id = event.session_id or self._session_id

            if event.timestamp == 0:
                event.timestamp = time.time()

            data_json = json.dumps(event.data, default=str)

            cursor = self._conn.execute(
                """INSERT INTO events
                (event_type, severity, timestamp, strategy, symbol, source,
                 action, data, sequence, correlation_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_type, event.severity, event.timestamp,
                    event.strategy, event.symbol, event.source,
                    event.action, data_json, event.sequence,
                    event.correlation_id, event.session_id,
                ),
            )
            self._conn.commit()
            event.id = cursor.lastrowid
            self._total_events += 1

        self._notify_listeners(event)
        return event.id or 0

    def append_batch(self, events: Sequence[StoredEvent]) -> List[int]:
        """Append multiple events in a single transaction."""
        ids: List[int] = []

        with self._lock:
            for event in events:
                self._sequence += 1
                event.sequence = self._sequence
                event.session_id = event.session_id or self._session_id
                if event.timestamp == 0:
                    event.timestamp = time.time()

                data_json = json.dumps(event.data, default=str)

                cursor = self._conn.execute(
                    """INSERT INTO events
                    (event_type, severity, timestamp, strategy, symbol, source,
                     action, data, sequence, correlation_id, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.event_type, event.severity, event.timestamp,
                        event.strategy, event.symbol, event.source,
                        event.action, data_json, event.sequence,
                        event.correlation_id, event.session_id,
                    ),
                )
                event.id = cursor.lastrowid
                ids.append(event.id or 0)

            self._conn.commit()
            self._total_events += len(events)

        for event in events:
            self._notify_listeners(event)

        return ids

    # --- Convenience event creators ---

    def log_trade(
        self, strategy: str, symbol: str, action: str,
        data: Optional[Dict] = None, correlation_id: str = "",
    ) -> int:
        event = StoredEvent(
            event_type=EventType.TRADE.value,
            severity=EventSeverity.INFO.value,
            strategy=strategy,
            symbol=symbol,
            source="trade_engine",
            action=action,
            data=data or {},
            correlation_id=correlation_id,
        )
        return self.append(event)

    def log_order(
        self, strategy: str, symbol: str, action: str,
        data: Optional[Dict] = None, correlation_id: str = "",
    ) -> int:
        event = StoredEvent(
            event_type=EventType.ORDER.value,
            severity=EventSeverity.INFO.value,
            strategy=strategy,
            symbol=symbol,
            source="order_manager",
            action=action,
            data=data or {},
            correlation_id=correlation_id,
        )
        return self.append(event)

    def log_signal(
        self, strategy: str, symbol: str, action: str,
        data: Optional[Dict] = None,
    ) -> int:
        event = StoredEvent(
            event_type=EventType.SIGNAL.value,
            severity=EventSeverity.INFO.value,
            strategy=strategy,
            symbol=symbol,
            source="signal_engine",
            action=action,
            data=data or {},
        )
        return self.append(event)

    def log_risk(
        self, strategy: str, symbol: str, action: str,
        data: Optional[Dict] = None, severity: str = "WARNING",
    ) -> int:
        event = StoredEvent(
            event_type=EventType.RISK_EVENT.value,
            severity=severity,
            strategy=strategy,
            symbol=symbol,
            source="risk_manager",
            action=action,
            data=data or {},
        )
        return self.append(event)

    def log_system(
        self, source: str, action: str,
        data: Optional[Dict] = None, severity: str = "INFO",
    ) -> int:
        event = StoredEvent(
            event_type=EventType.SYSTEM_EVENT.value,
            severity=severity,
            source=source,
            action=action,
            data=data or {},
        )
        return self.append(event)

    def log_error(
        self, source: str, action: str, error_message: str,
        data: Optional[Dict] = None,
    ) -> int:
        event_data = {"error": error_message}
        if data:
            event_data.update(data)
        event = StoredEvent(
            event_type=EventType.ERROR.value,
            severity=EventSeverity.ERROR.value,
            source=source,
            action=action,
            data=event_data,
        )
        return self.append(event)

    # --- Query events ---

    def query(self, query: EventQuery) -> List[StoredEvent]:
        """Query events with filters."""
        where, params = query.build_where()

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT * FROM events WHERE {where} "
                f"ORDER BY sequence {query.order} "
                f"LIMIT ? OFFSET ?",
                params + [query.limit, query.offset],
            )
            rows = cursor.fetchall()

        return [StoredEvent.from_row(row) for row in rows]

    def get_event(self, event_id: int) -> Optional[StoredEvent]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            )
            row = cursor.fetchone()
        return StoredEvent.from_row(row) if row else None

    def get_events_since(
        self, sequence: int, limit: int = 1000
    ) -> List[StoredEvent]:
        """Get events after a given sequence number."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE sequence > ? ORDER BY sequence LIMIT ?",
                (sequence, limit),
            )
            return [StoredEvent.from_row(row) for row in cursor.fetchall()]

    def get_events_by_correlation(self, correlation_id: str) -> List[StoredEvent]:
        """Get all events with the same correlation ID (trace a transaction)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE correlation_id = ? ORDER BY sequence",
                (correlation_id,),
            )
            return [StoredEvent.from_row(row) for row in cursor.fetchall()]

    def count_events(self, query: Optional[EventQuery] = None) -> int:
        if query is None:
            return self._total_events
        where, params = query.build_where()
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT COUNT(*) FROM events WHERE {where}", params
            )
            return cursor.fetchone()[0]

    def get_event_type_counts(self) -> Dict[str, int]:
        """Get count of events by type."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
            )
            return {row["event_type"]: row["cnt"] for row in cursor.fetchall()}

    # --- Checkpointing ---

    def create_checkpoint(
        self, name: str, state: Dict[str, Any]
    ) -> EventCheckpoint:
        """Create a checkpoint of the current state for faster replay."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        with self._lock:
            # Get last event info
            cursor = self._conn.execute(
                "SELECT id, sequence FROM events ORDER BY sequence DESC LIMIT 1"
            )
            row = cursor.fetchone()
            last_id = row["id"] if row else 0
            last_seq = row["sequence"] if row else 0

            checkpoint = EventCheckpoint(
                checkpoint_name=name,
                timestamp=time.time(),
                last_event_id=last_id,
                last_sequence=last_seq,
                state=state,
                created_at=now,
            )

            cursor = self._conn.execute(
                """INSERT INTO checkpoints
                (checkpoint_name, timestamp, last_event_id, last_sequence, state, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (name, checkpoint.timestamp, last_id, last_seq, json.dumps(state, default=str), now),
            )
            self._conn.commit()
            checkpoint.id = cursor.lastrowid

        logger.info(
            "Checkpoint '%s' created at sequence %d", name, last_seq
        )
        return checkpoint

    def get_latest_checkpoint(self, name: Optional[str] = None) -> Optional[EventCheckpoint]:
        """Get the most recent checkpoint."""
        clauses = []
        params: list = []
        if name:
            clauses.append("checkpoint_name = ?")
            params.append(name)

        where = " AND ".join(clauses) if clauses else "1=1"

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT * FROM checkpoints WHERE {where} ORDER BY timestamp DESC LIMIT 1",
                params,
            )
            row = cursor.fetchone()

        if row:
            d = dict(row)
            d["state"] = json.loads(d.get("state", "{}") or "{}")
            return EventCheckpoint(**{k: v for k, v in d.items() if k in EventCheckpoint.__dataclass_fields__})
        return None

    def replay_from_checkpoint(
        self,
        checkpoint: EventCheckpoint,
        replayer: EventReplayer,
    ) -> Dict[str, Any]:
        """Replay events from a checkpoint to reconstruct current state."""
        events = self.get_events_since(checkpoint.last_sequence)
        return replayer.replay(events, initial_state=checkpoint.state)

    def list_checkpoints(self) -> List[EventCheckpoint]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM checkpoints ORDER BY timestamp DESC"
            )
            result = []
            for row in cursor.fetchall():
                d = dict(row)
                d["state"] = json.loads(d.get("state", "{}") or "{}")
                result.append(
                    EventCheckpoint(**{k: v for k, v in d.items() if k in EventCheckpoint.__dataclass_fields__})
                )
            return result

    # --- Compression and archiving ---

    def archive_old_events(
        self, before_timestamp: float, archive_dir: str = "event_archive"
    ) -> Dict[str, int]:
        """
        Archive and delete events older than the given timestamp.
        Returns counts of archived and deleted events.
        """
        # Create checkpoint first
        self.create_checkpoint("pre_archive", {})

        # Get old events
        query = EventQuery(end_time=before_timestamp, limit=100000)
        old_events = self.query(query)

        if not old_events:
            return {"archived": 0, "deleted": 0}

        # Compress to file
        ts_str = datetime.fromtimestamp(
            before_timestamp, tz=timezone.utc
        ).strftime("%Y%m%d_%H%M%S")
        archive_path = os.path.join(archive_dir, f"events_before_{ts_str}.json.gz")
        archived = EventCompressor.compress_to_file(old_events, archive_path)

        # Delete from database
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM events WHERE timestamp < ?", (before_timestamp,)
            )
            deleted = cursor.rowcount
            self._conn.commit()
            self._total_events -= deleted

        logger.info(
            "Archived %d events to %s, deleted %d from database",
            archived, archive_path, deleted,
        )
        return {"archived": archived, "deleted": deleted}

    def export_events(
        self,
        filepath: str,
        query: Optional[EventQuery] = None,
        fmt: str = "json",
    ) -> int:
        """Export events to a file."""
        events = self.query(query or EventQuery(limit=100000))

        if fmt == "json":
            return EventCompressor.export_to_json(events, filepath)
        elif fmt == "jsonl":
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                for event in events:
                    f.write(event.to_json() + "\n")
            return len(events)
        elif fmt == "gz":
            return EventCompressor.compress_to_file(events, filepath)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            type_counts = {}
            cursor = self._conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
            )
            for row in cursor.fetchall():
                type_counts[row["event_type"]] = row["cnt"]

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM checkpoints"
            )
            checkpoint_count = cursor.fetchone()[0]

            cursor = self._conn.execute(
                "SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM events"
            )
            row = cursor.fetchone()
            oldest = row["oldest"] if row else None
            newest = row["newest"] if row else None

        return {
            "total_events": self._total_events,
            "current_sequence": self._sequence,
            "session_id": self._session_id,
            "event_types": type_counts,
            "checkpoint_count": checkpoint_count,
            "oldest_event": oldest,
            "newest_event": newest,
            "listener_count": sum(len(cbs) for cbs in self._listeners.values()),
        }

    # --- Lifecycle ---

    def vacuum(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        logger.info("EventStore closed")

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
