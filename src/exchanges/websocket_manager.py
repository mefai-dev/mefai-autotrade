# mefai-autotrade - WebSocket connection manager
# Manages multiple WebSocket connections with reconnection, routing, and buffering

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class WSConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class WSSubscription:
    """Represents a WebSocket subscription."""
    subscription_id: str
    connection_id: str
    stream: str           # Stream/topic identifier
    callback: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    message_count: int = 0
    last_message_at: float = 0.0
    active: bool = True


@dataclass
class WSConnectionInfo:
    """Tracks state and metrics for a single WebSocket connection."""
    connection_id: str
    url: str
    state: WSConnectionState = WSConnectionState.DISCONNECTED
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    subscriptions: Dict[str, WSSubscription] = field(default_factory=dict)
    reconnect_attempts: int = 0
    total_messages: int = 0
    total_bytes: int = 0
    connect_time: float = 0.0
    last_message_time: float = 0.0
    last_ping_time: float = 0.0
    last_pong_time: float = 0.0
    errors: List[Tuple[float, str]] = field(default_factory=list)
    _message_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def uptime(self) -> float:
        if self.connect_time > 0:
            return time.monotonic() - self.connect_time
        return 0.0

    @property
    def bandwidth_kbps(self) -> float:
        if self.uptime > 0:
            return (self.total_bytes / 1024) / self.uptime
        return 0.0

    @property
    def messages_per_second(self) -> float:
        if self.uptime > 0:
            return self.total_messages / self.uptime
        return 0.0

    def buffer_message(self, message: Dict[str, Any]) -> None:
        """Buffer a message during reconnection."""
        self._message_buffer.append((time.monotonic(), message))

    def drain_buffer(self) -> List[Dict[str, Any]]:
        """Get and clear all buffered messages."""
        messages = [msg for _, msg in self._message_buffer]
        self._message_buffer.clear()
        return messages

    @property
    def buffer_size(self) -> int:
        return len(self._message_buffer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "url": self.url[:80],
            "state": self.state.value,
            "subscriptions": len(self.subscriptions),
            "reconnect_attempts": self.reconnect_attempts,
            "total_messages": self.total_messages,
            "total_bytes": self.total_bytes,
            "uptime_seconds": round(self.uptime, 1),
            "bandwidth_kbps": round(self.bandwidth_kbps, 2),
            "messages_per_second": round(self.messages_per_second, 2),
            "buffer_size": self.buffer_size,
        }


class WebSocketManager:
    """Manages multiple WebSocket connections with automatic reconnection,
    message routing, subscription management, and health monitoring.

    Features:
    - Multiple simultaneous connections with independent lifecycle
    - Auto-reconnect with exponential backoff
    - Message routing to registered handler callbacks
    - Heartbeat/ping-pong handling
    - Message buffering during reconnection
    - Dynamic subscribe/unsubscribe
    - Bandwidth and latency monitoring
    - Connection health tracking
    """

    def __init__(
        self,
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        reconnect_max_attempts: int = 20,
        buffer_during_reconnect: bool = True,
        max_buffer_size: int = 1000,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._reconnect_base_delay = reconnect_base_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_max_attempts = reconnect_max_attempts
        self._buffer_during_reconnect = buffer_during_reconnect
        self._max_buffer_size = max_buffer_size

        self._session = session
        self._owns_session = session is None
        self._connections: Dict[str, WSConnectionInfo] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._ping_tasks: Dict[str, asyncio.Task] = {}
        self._global_handlers: List[Callable] = []
        self._message_filters: Dict[str, Callable] = {}  # filter_id -> filter_func
        self._subscription_counter = 0
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("ws_manager")
        self._on_connect: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        self._on_reconnect: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        url: str,
        connection_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        auth_handler: Optional[Callable] = None,
        subscribe_on_connect: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Create a new WebSocket connection.

        Args:
            url: WebSocket URL to connect to.
            connection_id: Optional custom ID. Auto-generated if not provided.
            headers: Optional HTTP headers for the connection.
            auth_handler: Optional async callable for authentication after connect.
            subscribe_on_connect: Optional list of subscription messages to send after connect.

        Returns:
            Connection ID string.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True

        conn_id = connection_id or f"ws_{uuid.uuid4().hex[:12]}"

        conn_info = WSConnectionInfo(
            connection_id=conn_id,
            url=url,
        )
        self._connections[conn_id] = conn_info

        # Start the connection task
        task = asyncio.create_task(
            self._connection_loop(
                conn_id, url, headers, auth_handler, subscribe_on_connect
            )
        )
        self._tasks[conn_id] = task

        self._logger.info("Connection %s created for %s", conn_id, url[:80])
        return conn_id

    async def disconnect(self, connection_id: str, code: int = 1000) -> bool:
        """Disconnect a WebSocket connection.

        Args:
            connection_id: The connection to disconnect.
            code: WebSocket close code (1000 = normal).

        Returns:
            True if the connection was found and disconnected.
        """
        conn = self._connections.get(connection_id)
        if not conn:
            return False

        conn.state = WSConnectionState.CLOSING

        # Cancel tasks
        task = self._tasks.pop(connection_id, None)
        if task and not task.done():
            task.cancel()

        ping_task = self._ping_tasks.pop(connection_id, None)
        if ping_task and not ping_task.done():
            ping_task.cancel()

        # Close WebSocket
        if conn.ws and not conn.ws.closed:
            try:
                await conn.ws.close(code=code)
            except Exception:
                pass

        conn.state = WSConnectionState.CLOSED
        conn.ws = None

        # Remove subscriptions
        for sub in list(conn.subscriptions.values()):
            sub.active = False

        self._logger.info("Connection %s disconnected", connection_id)

        if self._on_disconnect:
            await self._fire_event(self._on_disconnect, connection_id)

        return True

    async def disconnect_all(self) -> None:
        """Disconnect all WebSocket connections."""
        conn_ids = list(self._connections.keys())
        for conn_id in conn_ids:
            await self.disconnect(conn_id)
        self._connections.clear()

        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _connection_loop(
        self,
        conn_id: str,
        url: str,
        headers: Optional[Dict[str, str]],
        auth_handler: Optional[Callable],
        subscribe_on_connect: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Main connection loop with reconnection logic."""
        conn = self._connections.get(conn_id)
        if not conn:
            return

        attempt = 0
        while True:
            try:
                conn.state = WSConnectionState.CONNECTING if attempt == 0 else WSConnectionState.RECONNECTING
                conn.reconnect_attempts = attempt

                self._logger.info(
                    "WS connecting %s to %s (attempt %d)",
                    conn_id, url[:80], attempt + 1,
                )

                ws_kwargs: Dict[str, Any] = {
                    "heartbeat": self._ping_interval,
                    "timeout": self._ping_timeout,
                }
                if headers:
                    ws_kwargs["headers"] = headers

                async with self._session.ws_connect(url, **ws_kwargs) as ws:
                    conn.ws = ws
                    conn.state = WSConnectionState.CONNECTED
                    conn.connect_time = time.monotonic()
                    attempt = 0

                    # Authenticate if handler provided
                    if auth_handler:
                        conn.state = WSConnectionState.AUTHENTICATING
                        try:
                            await auth_handler(ws)
                            conn.state = WSConnectionState.AUTHENTICATED
                        except Exception as exc:
                            self._logger.error("WS auth failed for %s: %s", conn_id, exc)
                            conn.state = WSConnectionState.CONNECTED  # Continue without auth

                    # Send initial subscriptions
                    if subscribe_on_connect:
                        for sub_msg in subscribe_on_connect:
                            await ws.send_json(sub_msg)

                    # Re-subscribe existing subscriptions (after reconnect)
                    await self._resubscribe(conn)

                    # Drain buffered messages
                    if self._buffer_during_reconnect:
                        buffered = conn.drain_buffer()
                        if buffered:
                            self._logger.info(
                                "Draining %d buffered messages for %s",
                                len(buffered), conn_id,
                            )
                            for msg in buffered:
                                await self._route_message(conn_id, msg)

                    # Fire connect callback
                    if self._on_connect:
                        await self._fire_event(self._on_connect, conn_id)
                    if attempt > 0 and self._on_reconnect:
                        await self._fire_event(self._on_reconnect, conn_id, attempt)

                    # Start ping task
                    self._ping_tasks[conn_id] = asyncio.create_task(
                        self._ping_loop(conn_id)
                    )

                    # Message receive loop
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            conn.total_messages += 1
                            conn.total_bytes += len(msg.data)
                            conn.last_message_time = time.monotonic()
                            try:
                                data = json.loads(msg.data)
                                await self._route_message(conn_id, data)
                            except json.JSONDecodeError:
                                self._logger.debug("WS non-JSON message: %s", msg.data[:100])

                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            conn.total_messages += 1
                            conn.total_bytes += len(msg.data)
                            conn.last_message_time = time.monotonic()

                        elif msg.type == aiohttp.WSMsgType.PONG:
                            conn.last_pong_time = time.monotonic()

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            exc = ws.exception()
                            self._logger.error("WS error for %s: %s", conn_id, exc)
                            break

                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            self._logger.info("WS closed for %s (code=%s)", conn_id, msg.data)
                            break

            except asyncio.CancelledError:
                self._logger.info("WS task cancelled for %s", conn_id)
                conn.state = WSConnectionState.CLOSED
                return

            except Exception as exc:
                error_msg = str(exc)
                conn.errors.append((time.monotonic(), error_msg))
                # Keep only last 50 errors
                if len(conn.errors) > 50:
                    conn.errors = conn.errors[-50:]
                self._logger.error("WS connection error for %s: %s", conn_id, error_msg)

                if self._on_error:
                    await self._fire_event(self._on_error, conn_id, exc)

            # Cancel ping task
            ping_task = self._ping_tasks.pop(conn_id, None)
            if ping_task and not ping_task.done():
                ping_task.cancel()

            # Update state
            conn.ws = None
            if conn.state == WSConnectionState.CLOSING or conn.state == WSConnectionState.CLOSED:
                return

            conn.state = WSConnectionState.DISCONNECTED

            # Reconnection logic
            attempt += 1
            if attempt > self._reconnect_max_attempts:
                self._logger.error("WS max reconnect attempts for %s", conn_id)
                conn.state = WSConnectionState.FAILED
                return

            delay = min(
                self._reconnect_base_delay * (2 ** (attempt - 1)),
                self._reconnect_max_delay,
            )
            self._logger.info("WS reconnecting %s in %.1fs (attempt %d)", conn_id, delay, attempt)
            await asyncio.sleep(delay)

    async def _ping_loop(self, connection_id: str) -> None:
        """Send periodic pings to keep the connection alive."""
        conn = self._connections.get(connection_id)
        if not conn:
            return

        while True:
            try:
                await asyncio.sleep(self._ping_interval)
                if conn.ws and not conn.ws.closed:
                    await conn.ws.ping()
                    conn.last_ping_time = time.monotonic()
                else:
                    break
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _resubscribe(self, conn: WSConnectionInfo) -> None:
        """Re-subscribe to all active subscriptions after reconnection."""
        if not conn.ws or conn.ws.closed:
            return

        for sub in conn.subscriptions.values():
            if not sub.active:
                continue
            if sub.params:
                try:
                    await conn.ws.send_json(sub.params)
                    self._logger.debug("Re-subscribed to %s on %s", sub.stream, conn.connection_id)
                except Exception as exc:
                    self._logger.error("Re-subscribe failed for %s: %s", sub.stream, exc)

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    async def _route_message(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Route an incoming message to registered handlers."""
        conn = self._connections.get(connection_id)
        if not conn:
            return

        # Apply message filters
        for filter_id, filter_func in self._message_filters.items():
            try:
                if not filter_func(data):
                    return  # Message filtered out
            except Exception:
                pass

        # Route to subscription callbacks
        matched = False
        for sub in conn.subscriptions.values():
            if not sub.active:
                continue
            try:
                result = sub.callback(data)
                if asyncio.iscoroutine(result):
                    await result
                sub.message_count += 1
                sub.last_message_at = time.monotonic()
                matched = True
            except Exception as exc:
                self._logger.error(
                    "Subscription callback error for %s: %s",
                    sub.subscription_id, exc,
                )

        # Route to global handlers
        for handler in self._global_handlers:
            try:
                result = handler(connection_id, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                self._logger.error("Global handler error: %s", exc)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        connection_id: str,
        stream: str,
        callback: Callable,
        subscribe_message: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Subscribe to a stream on a connection.

        Args:
            connection_id: Target connection.
            stream: Stream/topic identifier (for reference).
            callback: Callable to invoke with each message.
            subscribe_message: Optional subscription message to send to the server.

        Returns:
            Subscription ID.
        """
        conn = self._connections.get(connection_id)
        if not conn:
            raise ValueError(f"Connection {connection_id} not found")

        self._subscription_counter += 1
        sub_id = f"sub_{self._subscription_counter}_{uuid.uuid4().hex[:8]}"

        subscription = WSSubscription(
            subscription_id=sub_id,
            connection_id=connection_id,
            stream=stream,
            callback=callback,
            params=subscribe_message or {},
        )

        conn.subscriptions[sub_id] = subscription

        # Send subscription message if connection is active
        if subscribe_message and conn.ws and not conn.ws.closed:
            try:
                await conn.ws.send_json(subscribe_message)
            except Exception as exc:
                self._logger.error("Subscribe send failed: %s", exc)

        self._logger.debug("Subscribed %s to %s on %s", sub_id, stream, connection_id)
        return sub_id

    async def unsubscribe(
        self,
        subscription_id: str,
        unsubscribe_message: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Unsubscribe from a stream.

        Args:
            subscription_id: The subscription to remove.
            unsubscribe_message: Optional unsubscription message to send.

        Returns:
            True if found and removed.
        """
        for conn in self._connections.values():
            sub = conn.subscriptions.pop(subscription_id, None)
            if sub:
                sub.active = False

                if unsubscribe_message and conn.ws and not conn.ws.closed:
                    try:
                        await conn.ws.send_json(unsubscribe_message)
                    except Exception:
                        pass

                self._logger.debug("Unsubscribed %s", subscription_id)
                return True

        return False

    async def unsubscribe_all(self, connection_id: str) -> int:
        """Unsubscribe all subscriptions on a connection.

        Returns:
            Number of subscriptions removed.
        """
        conn = self._connections.get(connection_id)
        if not conn:
            return 0

        count = len(conn.subscriptions)
        for sub in conn.subscriptions.values():
            sub.active = False
        conn.subscriptions.clear()
        return count

    def get_subscriptions(self, connection_id: Optional[str] = None) -> List[WSSubscription]:
        """Get all subscriptions, optionally filtered by connection."""
        result = []
        for conn in self._connections.values():
            if connection_id and conn.connection_id != connection_id:
                continue
            result.extend(conn.subscriptions.values())
        return result

    # ------------------------------------------------------------------
    # Global handlers and filters
    # ------------------------------------------------------------------

    def add_global_handler(self, handler: Callable) -> None:
        """Add a global message handler that receives all messages.

        handler(connection_id: str, data: dict) -> None
        """
        self._global_handlers.append(handler)

    def remove_global_handler(self, handler: Callable) -> bool:
        """Remove a global message handler."""
        try:
            self._global_handlers.remove(handler)
            return True
        except ValueError:
            return False

    def add_message_filter(self, filter_id: str, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a message filter. If filter returns False, message is dropped.

        filter_func(data: dict) -> bool (True = pass, False = drop)
        """
        self._message_filters[filter_id] = filter_func

    def remove_message_filter(self, filter_id: str) -> bool:
        """Remove a message filter."""
        return self._message_filters.pop(filter_id, None) is not None

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def on_connect(self, callback: Callable) -> None:
        """Register callback for connection established events."""
        self._on_connect = callback

    def on_disconnect(self, callback: Callable) -> None:
        """Register callback for disconnection events."""
        self._on_disconnect = callback

    def on_error(self, callback: Callable) -> None:
        """Register callback for error events."""
        self._on_error = callback

    def on_reconnect(self, callback: Callable) -> None:
        """Register callback for reconnection events."""
        self._on_reconnect = callback

    async def _fire_event(self, callback: Callable, *args) -> None:
        """Fire an event callback safely."""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            self._logger.error("Event callback error: %s", exc)

    # ------------------------------------------------------------------
    # Send messages
    # ------------------------------------------------------------------

    async def send(self, connection_id: str, data: Dict[str, Any]) -> bool:
        """Send a JSON message on a connection.

        Returns True if sent successfully.
        """
        conn = self._connections.get(connection_id)
        if not conn or not conn.ws or conn.ws.closed:
            if self._buffer_during_reconnect and conn:
                conn.buffer_message(data)
                return False
            return False

        try:
            await conn.ws.send_json(data)
            return True
        except Exception as exc:
            self._logger.error("Send failed on %s: %s", connection_id, exc)
            if self._buffer_during_reconnect:
                conn.buffer_message(data)
            return False

    async def send_raw(self, connection_id: str, data: str) -> bool:
        """Send raw string data on a connection."""
        conn = self._connections.get(connection_id)
        if not conn or not conn.ws or conn.ws.closed:
            return False
        try:
            await conn.ws.send_str(data)
            return True
        except Exception:
            return False

    async def broadcast(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Send a message to all active connections.

        Returns {connection_id: success_bool}.
        """
        results = {}
        for conn_id in self._connections:
            results[conn_id] = await self.send(conn_id, data)
        return results

    # ------------------------------------------------------------------
    # Status and monitoring
    # ------------------------------------------------------------------

    def get_connection_state(self, connection_id: str) -> Optional[WSConnectionState]:
        """Get the state of a specific connection."""
        conn = self._connections.get(connection_id)
        return conn.state if conn else None

    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a connection."""
        conn = self._connections.get(connection_id)
        return conn.to_dict() if conn else None

    def get_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get info for all connections."""
        return {
            conn_id: conn.to_dict()
            for conn_id, conn in self._connections.items()
        }

    @property
    def active_connection_count(self) -> int:
        """Number of currently connected WebSockets."""
        return sum(
            1 for c in self._connections.values()
            if c.state in (WSConnectionState.CONNECTED, WSConnectionState.AUTHENTICATED)
        )

    @property
    def total_connection_count(self) -> int:
        """Total number of managed connections (any state)."""
        return len(self._connections)

    @property
    def total_subscriptions(self) -> int:
        """Total number of active subscriptions across all connections."""
        return sum(
            len(c.subscriptions) for c in self._connections.values()
        )

    @property
    def total_messages_received(self) -> int:
        """Total messages received across all connections."""
        return sum(c.total_messages for c in self._connections.values())

    @property
    def total_bytes_received(self) -> int:
        """Total bytes received across all connections."""
        return sum(c.total_bytes for c in self._connections.values())

    def get_bandwidth_stats(self) -> Dict[str, Any]:
        """Get aggregate bandwidth statistics."""
        total_bytes = self.total_bytes_received
        total_messages = self.total_messages_received
        connections = self.active_connection_count

        return {
            "active_connections": connections,
            "total_connections": self.total_connection_count,
            "total_subscriptions": self.total_subscriptions,
            "total_messages": total_messages,
            "total_bytes": total_bytes,
            "total_kb": round(total_bytes / 1024, 2),
            "total_mb": round(total_bytes / (1024 * 1024), 2),
            "per_connection": {
                conn_id: {
                    "messages": conn.total_messages,
                    "bytes": conn.total_bytes,
                    "bandwidth_kbps": round(conn.bandwidth_kbps, 2),
                    "msg_per_sec": round(conn.messages_per_second, 2),
                }
                for conn_id, conn in self._connections.items()
            },
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a health summary for all connections."""
        states: Dict[str, int] = {}
        unhealthy = []

        for conn in self._connections.values():
            state = conn.state.value
            states[state] = states.get(state, 0) + 1

            # Check for unhealthy indicators
            issues = []
            if conn.state == WSConnectionState.FAILED:
                issues.append("connection_failed")
            elif conn.state == WSConnectionState.DISCONNECTED:
                issues.append("disconnected")

            if conn.reconnect_attempts > 5:
                issues.append(f"high_reconnect_count({conn.reconnect_attempts})")

            if conn.last_message_time > 0:
                silence = time.monotonic() - conn.last_message_time
                if silence > 120:
                    issues.append(f"no_messages_{int(silence)}s")

            recent_errors = sum(
                1 for t, _ in conn.errors
                if time.monotonic() - t < 300
            )
            if recent_errors > 5:
                issues.append(f"errors_5m({recent_errors})")

            if issues:
                unhealthy.append({
                    "connection_id": conn.connection_id,
                    "issues": issues,
                })

        return {
            "states": states,
            "healthy": not unhealthy,
            "unhealthy_connections": unhealthy,
            "total_buffer_size": sum(c.buffer_size for c in self._connections.values()),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the WebSocket manager and all connections."""
        await self.disconnect_all()
        self._global_handlers.clear()
        self._message_filters.clear()
        self._logger.info("WebSocket manager closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __repr__(self) -> str:
        return (
            f"<WebSocketManager "
            f"connections={self.total_connection_count} "
            f"active={self.active_connection_count} "
            f"subscriptions={self.total_subscriptions}>"
        )
