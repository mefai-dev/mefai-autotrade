"""
Event Bus - Event-driven architecture for Mefai Autotrade.
Provides async pub/sub with priority queues, dead letter handling,
event filtering, routing, logging, and replay capabilities.
"""

import asyncio
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union,
)

logger = logging.getLogger("mefai.autotrade.event_bus")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(Enum):
    """All event types in the trading system."""
    # Order events
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    ORDER_MODIFIED = "ORDER_MODIFIED"

    # Position events
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_LIQUIDATION_WARNING = "POSITION_LIQUIDATION_WARNING"

    # Signal events
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_PROCESSED = "SIGNAL_PROCESSED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"

    # Risk events
    RISK_ALERT = "RISK_ALERT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MARGIN_WARNING = "MARGIN_WARNING"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"

    # System events
    ENGINE_STARTED = "ENGINE_STARTED"
    ENGINE_STOPPED = "ENGINE_STOPPED"
    ENGINE_PAUSED = "ENGINE_PAUSED"
    ENGINE_RESUMED = "ENGINE_RESUMED"
    ENGINE_ERROR = "ENGINE_ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    # Data events
    PRICE_UPDATE = "PRICE_UPDATE"
    KLINE_UPDATE = "KLINE_UPDATE"
    ORDERBOOK_UPDATE = "ORDERBOOK_UPDATE"
    FUNDING_RATE = "FUNDING_RATE"

    # Account events
    BALANCE_CHANGED = "BALANCE_CHANGED"
    ACCOUNT_SYNCED = "ACCOUNT_SYNCED"

    # Strategy events
    STRATEGY_STARTED = "STRATEGY_STARTED"
    STRATEGY_STOPPED = "STRATEGY_STOPPED"
    STRATEGY_ERROR = "STRATEGY_ERROR"

    # Custom / generic
    CUSTOM = "CUSTOM"


class EventPriority(Enum):
    """Event processing priority. Lower value = higher priority."""
    CRITICAL = 0    # Circuit breaker, emergency stop
    HIGH = 1        # Order fills, risk alerts
    NORMAL = 2      # Signals, position updates
    LOW = 3         # Price updates, logging
    BACKGROUND = 4  # Snapshots, cleanup


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Represents a single event in the system."""
    event_type: EventType = EventType.CUSTOM
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source: str = ""      # Component that generated the event
    target: str = ""      # Specific target handler (optional)
    correlation_id: str = ""  # For tracing related events
    ttl_seconds: float = 0.0  # Time to live (0 = no expiry)
    retry_count: int = 0
    max_retries: int = 3

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target,
            "correlation_id": self.correlation_id,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            event_id=d.get("event_id", uuid.uuid4().hex[:16]),
            event_type=EventType(d["event_type"]),
            data=d.get("data", {}),
            priority=EventPriority(d.get("priority", EventPriority.NORMAL.value)),
            timestamp=(
                datetime.fromisoformat(d["timestamp"])
                if "timestamp" in d else datetime.now(timezone.utc)
            ),
            source=d.get("source", ""),
            target=d.get("target", ""),
            correlation_id=d.get("correlation_id", ""),
            retry_count=d.get("retry_count", 0),
        )


@dataclass
class DeadLetterEntry:
    """An event that failed processing and was moved to the dead letter queue."""
    event: Event
    error: str = ""
    handler_name: str = ""
    failed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    attempts: int = 0


# ---------------------------------------------------------------------------
# Handler types
# ---------------------------------------------------------------------------

# Sync or async handler
EventHandler = Union[
    Callable[[Event], None],
    Callable[[Event], Awaitable[None]],
]


@dataclass
class HandlerRegistration:
    """Metadata for a registered event handler."""
    handler: EventHandler
    name: str = ""
    event_types: Set[EventType] = field(default_factory=set)
    priority: EventPriority = EventPriority.NORMAL
    is_async: bool = False
    filter_fn: Optional[Callable[[Event], bool]] = None
    active: bool = True
    call_count: int = 0
    error_count: int = 0
    last_called: Optional[datetime] = None
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Production event bus with async support, priority queues,
    dead letter handling, and event replay.
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        max_dead_letters: int = 1000,
        max_event_log: int = 50000,
        enable_logging: bool = True,
    ):
        self._lock = threading.RLock()

        # Handler registry: event_type -> list of handlers
        self._handlers: Dict[EventType, List[HandlerRegistration]] = defaultdict(list)
        self._global_handlers: List[HandlerRegistration] = []  # Receive all events
        self._handler_map: Dict[str, HandlerRegistration] = {}  # name -> registration

        # Async queue for ordered processing
        self._queue: Optional[asyncio.PriorityQueue] = None
        self._max_queue_size = max_queue_size

        # Dead letter queue
        self._dead_letters: List[DeadLetterEntry] = []
        self._max_dead_letters = max_dead_letters

        # Event log for replay
        self._event_log: List[Event] = []
        self._max_event_log = max_event_log
        self._enable_logging = enable_logging

        # Stats
        self._total_published = 0
        self._total_processed = 0
        self._total_errors = 0
        self._events_by_type: Dict[str, int] = defaultdict(int)

        # Processing state
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        logger.info(
            "EventBus initialized (queue=%d, dead_letters=%d)",
            max_queue_size, max_dead_letters,
        )

    # -- Handler registration -------------------------------------------------

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        name: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """
        Subscribe a handler to a specific event type.
        Returns the handler name for later unsubscription.
        """
        is_async = asyncio.iscoroutinefunction(handler)
        handler_name = name or f"handler_{uuid.uuid4().hex[:8]}"

        registration = HandlerRegistration(
            handler=handler,
            name=handler_name,
            event_types={event_type},
            priority=priority,
            is_async=is_async,
            filter_fn=filter_fn,
        )

        with self._lock:
            self._handlers[event_type].append(registration)
            self._handler_map[handler_name] = registration
            # Sort by priority
            self._handlers[event_type].sort(
                key=lambda h: h.priority.value
            )

        logger.debug(
            "Handler '%s' subscribed to %s (async=%s, priority=%s)",
            handler_name, event_type.value, is_async, priority.value,
        )
        return handler_name

    def subscribe_many(
        self,
        event_types: List[EventType],
        handler: EventHandler,
        name: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """Subscribe a handler to multiple event types."""
        is_async = asyncio.iscoroutinefunction(handler)
        handler_name = name or f"handler_{uuid.uuid4().hex[:8]}"

        registration = HandlerRegistration(
            handler=handler,
            name=handler_name,
            event_types=set(event_types),
            priority=priority,
            is_async=is_async,
            filter_fn=filter_fn,
        )

        with self._lock:
            for et in event_types:
                self._handlers[et].append(registration)
                self._handlers[et].sort(key=lambda h: h.priority.value)
            self._handler_map[handler_name] = registration

        logger.debug(
            "Handler '%s' subscribed to %d event types",
            handler_name, len(event_types),
        )
        return handler_name

    def subscribe_all(
        self,
        handler: EventHandler,
        name: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
    ) -> str:
        """Subscribe a handler to ALL event types (global handler)."""
        is_async = asyncio.iscoroutinefunction(handler)
        handler_name = name or f"global_{uuid.uuid4().hex[:8]}"

        registration = HandlerRegistration(
            handler=handler,
            name=handler_name,
            priority=priority,
            is_async=is_async,
            filter_fn=filter_fn,
        )

        with self._lock:
            self._global_handlers.append(registration)
            self._global_handlers.sort(key=lambda h: h.priority.value)
            self._handler_map[handler_name] = registration

        logger.debug("Global handler '%s' subscribed", handler_name)
        return handler_name

    def unsubscribe(self, handler_name: str) -> bool:
        """Remove a handler by name."""
        with self._lock:
            registration = self._handler_map.pop(handler_name, None)
            if registration is None:
                return False

            # Remove from type-specific handlers
            for et in registration.event_types:
                self._handlers[et] = [
                    h for h in self._handlers[et] if h.name != handler_name
                ]

            # Remove from global handlers
            self._global_handlers = [
                h for h in self._global_handlers if h.name != handler_name
            ]

        logger.debug("Handler '%s' unsubscribed", handler_name)
        return True

    def pause_handler(self, handler_name: str) -> bool:
        with self._lock:
            reg = self._handler_map.get(handler_name)
            if reg:
                reg.active = False
                return True
        return False

    def resume_handler(self, handler_name: str) -> bool:
        with self._lock:
            reg = self._handler_map.get(handler_name)
            if reg:
                reg.active = True
                return True
        return False

    # -- Event publishing -----------------------------------------------------

    def publish(self, event: Event) -> None:
        """
        Publish an event synchronously. Handlers are called in priority order.
        Use publish_async for non-blocking dispatch via the queue.
        """
        if event.is_expired:
            logger.debug(
                "Dropping expired event %s (%s)",
                event.event_id, event.event_type.value,
            )
            return

        with self._lock:
            self._total_published += 1
            self._events_by_type[event.event_type.value] += 1

            if self._enable_logging:
                self._event_log.append(event)
                if len(self._event_log) > self._max_event_log:
                    self._event_log = self._event_log[-self._max_event_log:]

        # Collect handlers
        handlers = self._get_handlers(event)

        # Dispatch
        for reg in handlers:
            self._dispatch_sync(reg, event)

    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously. If the queue processor is running,
        events are enqueued. Otherwise, handlers are called directly.
        """
        if event.is_expired:
            return

        with self._lock:
            self._total_published += 1
            self._events_by_type[event.event_type.value] += 1

            if self._enable_logging:
                self._event_log.append(event)
                if len(self._event_log) > self._max_event_log:
                    self._event_log = self._event_log[-self._max_event_log:]

        if self._queue is not None and self._running:
            # Enqueue for ordered processing
            try:
                self._queue.put_nowait(
                    (event.priority.value, time.monotonic(), event)
                )
            except asyncio.QueueFull:
                logger.warning(
                    "Event queue full, processing inline: %s",
                    event.event_type.value,
                )
                await self._process_event(event)
        else:
            await self._process_event(event)

    def emit(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "",
        correlation_id: str = "",
    ) -> Event:
        """Convenience method to create and publish an event."""
        event = Event(
            event_type=event_type,
            data=data or {},
            priority=priority,
            source=source,
            correlation_id=correlation_id,
        )
        self.publish(event)
        return event

    async def emit_async(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "",
        correlation_id: str = "",
    ) -> Event:
        """Convenience method to create and publish an event asynchronously."""
        event = Event(
            event_type=event_type,
            data=data or {},
            priority=priority,
            source=source,
            correlation_id=correlation_id,
        )
        await self.publish_async(event)
        return event

    # -- Event processing -----------------------------------------------------

    def _get_handlers(self, event: Event) -> List[HandlerRegistration]:
        """Get all handlers for an event, sorted by priority."""
        handlers: List[HandlerRegistration] = []

        with self._lock:
            # Type-specific handlers
            for reg in self._handlers.get(event.event_type, []):
                if not reg.active:
                    continue
                if event.target and reg.name != event.target:
                    continue
                if reg.filter_fn and not reg.filter_fn(event):
                    continue
                handlers.append(reg)

            # Global handlers
            for reg in self._global_handlers:
                if not reg.active:
                    continue
                if reg.filter_fn and not reg.filter_fn(event):
                    continue
                handlers.append(reg)

        # Sort by priority
        handlers.sort(key=lambda h: h.priority.value)
        return handlers

    def _dispatch_sync(
        self, registration: HandlerRegistration, event: Event
    ) -> None:
        """Dispatch event to a handler synchronously."""
        start = time.monotonic()
        try:
            if registration.is_async:
                # If we are in an async context, schedule it
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(registration.handler(event))
                except RuntimeError:
                    # No running loop - run sync anyway
                    asyncio.run(registration.handler(event))
            else:
                registration.handler(event)

            registration.call_count += 1
            registration.last_called = datetime.now(timezone.utc)
            elapsed_ms = (time.monotonic() - start) * 1000
            registration.total_latency_ms += elapsed_ms
            registration.avg_latency_ms = (
                registration.total_latency_ms / registration.call_count
            )

            with self._lock:
                self._total_processed += 1

        except Exception as e:
            registration.error_count += 1
            with self._lock:
                self._total_errors += 1
            logger.exception(
                "Error in handler '%s' for event %s: %s",
                registration.name, event.event_type.value, e,
            )
            self._send_to_dead_letter(event, str(e), registration.name)

    async def _dispatch_async(
        self, registration: HandlerRegistration, event: Event
    ) -> None:
        """Dispatch event to a handler asynchronously."""
        start = time.monotonic()
        try:
            if registration.is_async:
                await registration.handler(event)
            else:
                registration.handler(event)

            registration.call_count += 1
            registration.last_called = datetime.now(timezone.utc)
            elapsed_ms = (time.monotonic() - start) * 1000
            registration.total_latency_ms += elapsed_ms
            registration.avg_latency_ms = (
                registration.total_latency_ms / registration.call_count
            )

            with self._lock:
                self._total_processed += 1

        except Exception as e:
            registration.error_count += 1
            with self._lock:
                self._total_errors += 1
            logger.exception(
                "Error in async handler '%s' for event %s: %s",
                registration.name, event.event_type.value, e,
            )
            self._send_to_dead_letter(event, str(e), registration.name)

    async def _process_event(self, event: Event) -> None:
        """Process a single event through all matching handlers."""
        if event.is_expired:
            return

        handlers = self._get_handlers(event)
        for reg in handlers:
            await self._dispatch_async(reg, event)

    # -- Queue processor ------------------------------------------------------

    async def start(self) -> None:
        """Start the async event queue processor."""
        if self._running:
            return

        self._queue = asyncio.PriorityQueue(maxsize=self._max_queue_size)
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("EventBus queue processor started")

    async def stop(self) -> None:
        """Stop the event queue processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
        self._queue = None
        logger.info("EventBus queue processor stopped")

    async def _process_loop(self) -> None:
        """Main processing loop that dequeues and processes events."""
        while self._running:
            try:
                # Wait for next event with timeout
                try:
                    priority, timestamp, event = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_event(event)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in event processing loop")
                await asyncio.sleep(0.1)

    # -- Dead letter queue ----------------------------------------------------

    def _send_to_dead_letter(
        self, event: Event, error: str, handler_name: str
    ) -> None:
        """Move a failed event to the dead letter queue."""
        entry = DeadLetterEntry(
            event=event,
            error=error,
            handler_name=handler_name,
            attempts=event.retry_count + 1,
        )

        with self._lock:
            self._dead_letters.append(entry)
            if len(self._dead_letters) > self._max_dead_letters:
                self._dead_letters = self._dead_letters[
                    -self._max_dead_letters:
                ]

        logger.warning(
            "Event %s sent to dead letter queue (handler=%s, error=%s)",
            event.event_id, handler_name, error[:100],
        )

    def get_dead_letters(
        self, limit: int = 100
    ) -> List[DeadLetterEntry]:
        with self._lock:
            return list(self._dead_letters[-limit:])

    def retry_dead_letters(
        self,
        max_retries: int = 3,
    ) -> int:
        """Retry failed events from the dead letter queue."""
        retried = 0
        with self._lock:
            to_retry = [
                dl for dl in self._dead_letters
                if dl.attempts < max_retries
            ]
            # Remove from dead letter queue
            retry_ids = {id(dl) for dl in to_retry}
            self._dead_letters = [
                dl for dl in self._dead_letters
                if id(dl) not in retry_ids
            ]

        for dl in to_retry:
            dl.event.retry_count += 1
            self.publish(dl.event)
            retried += 1

        if retried > 0:
            logger.info("Retried %d dead letter events", retried)
        return retried

    def clear_dead_letters(self) -> int:
        with self._lock:
            count = len(self._dead_letters)
            self._dead_letters.clear()
        return count

    # -- Event log and replay -------------------------------------------------

    def get_event_log(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Query the event log with optional filters."""
        with self._lock:
            events = list(self._event_log)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    async def replay_events(
        self,
        events: List[Event],
        speed_multiplier: float = 1.0,
    ) -> int:
        """
        Replay historical events through the bus.
        speed_multiplier: 1.0 = real-time, 2.0 = 2x speed, 0 = instant.
        """
        replayed = 0
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        prev_ts: Optional[datetime] = None
        for event in sorted_events:
            # Simulate timing
            if (
                speed_multiplier > 0
                and prev_ts is not None
            ):
                delta = (event.timestamp - prev_ts).total_seconds()
                wait_time = delta / speed_multiplier
                if wait_time > 0:
                    await asyncio.sleep(min(wait_time, 5.0))

            # Re-publish with new ID to avoid duplicates
            replay_event = Event(
                event_type=event.event_type,
                data=event.data,
                priority=event.priority,
                source=f"replay:{event.source}",
                correlation_id=event.correlation_id,
            )
            await self.publish_async(replay_event)
            replayed += 1
            prev_ts = event.timestamp

        logger.info("Replayed %d events", replayed)
        return replayed

    # -- Statistics -----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            handler_stats = {}
            for name, reg in self._handler_map.items():
                handler_stats[name] = {
                    "event_types": [et.value for et in reg.event_types],
                    "active": reg.active,
                    "call_count": reg.call_count,
                    "error_count": reg.error_count,
                    "avg_latency_ms": round(reg.avg_latency_ms, 2),
                    "is_async": reg.is_async,
                }

            return {
                "total_published": self._total_published,
                "total_processed": self._total_processed,
                "total_errors": self._total_errors,
                "handler_count": len(self._handler_map),
                "global_handler_count": len(self._global_handlers),
                "dead_letter_count": len(self._dead_letters),
                "event_log_size": len(self._event_log),
                "queue_size": (
                    self._queue.qsize()
                    if self._queue is not None else 0
                ),
                "running": self._running,
                "events_by_type": dict(self._events_by_type),
                "handlers": handler_stats,
            }

    def get_handler_info(
        self, handler_name: str
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            reg = self._handler_map.get(handler_name)
            if reg is None:
                return None
            return {
                "name": reg.name,
                "event_types": [et.value for et in reg.event_types],
                "priority": reg.priority.value,
                "active": reg.active,
                "is_async": reg.is_async,
                "call_count": reg.call_count,
                "error_count": reg.error_count,
                "avg_latency_ms": round(reg.avg_latency_ms, 2),
                "last_called": (
                    reg.last_called.isoformat() if reg.last_called else None
                ),
            }

    # -- Cleanup --------------------------------------------------------------

    def clear_event_log(self) -> int:
        with self._lock:
            count = len(self._event_log)
            self._event_log.clear()
        return count

    def reset_stats(self) -> None:
        with self._lock:
            self._total_published = 0
            self._total_processed = 0
            self._total_errors = 0
            self._events_by_type.clear()
            for reg in self._handler_map.values():
                reg.call_count = 0
                reg.error_count = 0
                reg.avg_latency_ms = 0.0
                reg.total_latency_ms = 0.0
