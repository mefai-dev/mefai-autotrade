# mefai-autotrade - Exchange connection pool and failover management
# Manages multiple exchange instances with health monitoring and load balancing

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.exchanges.base import (
    ExchangeBase,
    ExchangeConfig,
    ExchangeError,
    NetworkError,
    RateLimitError,
    ConnectionHealthStatus,
)

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    DRAINING = "draining"  # No new requests, finishing existing ones
    STANDBY = "standby"    # Ready but not actively used (backup)


@dataclass
class ConnectionHealth:
    """Tracks health metrics for a pooled connection."""
    connection_id: str
    exchange_name: str
    state: ConnectionState = ConnectionState.STANDBY
    latency_samples: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    last_error_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_errors: int = 0
    total_requests: int = 0
    created_at: float = field(default_factory=time.monotonic)
    last_health_check: float = 0.0

    @property
    def avg_latency(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples[-50:]) / min(len(self.latency_samples), 50)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.created_at

    def record_success(self, latency_ms: float) -> None:
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 200:
            self.latency_samples = self.latency_samples[-100:]
        self.success_count += 1
        self.total_requests += 1
        self.consecutive_errors = 0
        self.last_success_time = time.monotonic()

    def record_error(self) -> None:
        self.error_count += 1
        self.total_requests += 1
        self.consecutive_errors += 1
        self.last_error_time = time.monotonic()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "exchange": self.exchange_name,
            "state": self.state.value,
            "avg_latency_ms": round(self.avg_latency, 2),
            "success_rate": round(self.success_rate * 100, 1),
            "total_requests": self.total_requests,
            "consecutive_errors": self.consecutive_errors,
            "uptime_seconds": round(self.uptime, 1),
        }


@dataclass
class PooledConnection:
    """Wraps an exchange connection with pool metadata."""
    connection_id: str
    exchange: ExchangeBase
    config: ExchangeConfig
    health: ConnectionHealth
    priority: int = 0       # Lower = higher priority (0 = primary)
    is_primary: bool = True
    tags: List[str] = field(default_factory=list)  # e.g. ["fast", "market_data"]

    def __repr__(self) -> str:
        return (
            f"<PooledConnection {self.connection_id} "
            f"[{self.exchange.exchange_name}] "
            f"state={self.health.state.value} "
            f"primary={self.is_primary}>"
        )


class ExchangeConnectionPool:
    """Connection pool managing multiple exchange instances.

    Supports:
    - Primary/backup failover
    - Health monitoring per connection
    - Auto-reconnect on failure
    - Load balancing across active connections
    - Rate limit budget tracking
    - Latency-based routing
    """

    def __init__(
        self,
        max_connections: int = 10,
        health_check_interval: float = 30.0,
        max_consecutive_errors: int = 5,
        failover_cooldown: float = 60.0,
        latency_threshold_ms: float = 5000.0,
    ):
        self._connections: Dict[str, PooledConnection] = {}
        self._max_connections = max_connections
        self._health_check_interval = health_check_interval
        self._max_consecutive_errors = max_consecutive_errors
        self._failover_cooldown = failover_cooldown
        self._latency_threshold_ms = latency_threshold_ms
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._connection_counter = 0
        self._on_failover: Optional[Callable] = None
        self._on_state_change: Optional[Callable] = None
        self._logger = logging.getLogger("connection_pool")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def add_connection(
        self,
        exchange: ExchangeBase,
        config: ExchangeConfig,
        priority: int = 0,
        is_primary: bool = True,
        tags: Optional[List[str]] = None,
        auto_initialize: bool = True,
    ) -> str:
        """Add a new exchange connection to the pool.

        Args:
            exchange: An ExchangeBase instance.
            config: The config used to create the exchange.
            priority: Lower = higher priority for selection.
            is_primary: Whether this is a primary connection.
            tags: Optional tags for filtering connections.
            auto_initialize: Whether to initialize the connection immediately.

        Returns:
            Connection ID string.
        """
        async with self._lock:
            if len(self._connections) >= self._max_connections:
                raise ExchangeError(
                    f"Connection pool full (max={self._max_connections})",
                    exchange="pool",
                )

            self._connection_counter += 1
            conn_id = f"conn_{exchange.exchange_name}_{self._connection_counter}"

            health = ConnectionHealth(
                connection_id=conn_id,
                exchange_name=exchange.exchange_name,
            )

            pooled = PooledConnection(
                connection_id=conn_id,
                exchange=exchange,
                config=config,
                health=health,
                priority=priority,
                is_primary=is_primary,
                tags=tags or [],
            )

            if auto_initialize:
                try:
                    await exchange.initialize()
                    health.state = ConnectionState.ACTIVE
                    self._logger.info("Connection %s initialized and active", conn_id)
                except Exception as exc:
                    health.state = ConnectionState.FAILED
                    health.record_error()
                    self._logger.error("Connection %s failed to initialize: %s", conn_id, exc)
            else:
                health.state = ConnectionState.STANDBY

            self._connections[conn_id] = pooled
            return conn_id

    async def remove_connection(self, connection_id: str) -> bool:
        """Remove and close a connection."""
        async with self._lock:
            pooled = self._connections.pop(connection_id, None)
            if not pooled:
                return False

        try:
            await pooled.exchange.close()
        except Exception as exc:
            self._logger.warning("Error closing connection %s: %s", connection_id, exc)

        self._logger.info("Connection %s removed", connection_id)
        return True

    def get_connection(
        self,
        exchange_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        strategy: str = "best",
    ) -> Optional[PooledConnection]:
        """Get a connection from the pool based on selection strategy.

        Args:
            exchange_name: Filter by exchange name.
            tags: Filter by tags (any match).
            strategy: Selection strategy:
                - "best": Lowest latency among active connections
                - "primary": First primary connection
                - "round_robin": Next available connection
                - "least_loaded": Connection with most rate limit budget remaining

        Returns:
            A PooledConnection or None if no suitable connection found.
        """
        candidates = self._filter_connections(exchange_name, tags)
        if not candidates:
            return None

        # Only consider active connections
        active = [c for c in candidates if c.health.state == ConnectionState.ACTIVE]
        if not active:
            # Fall back to standby connections
            active = [c for c in candidates if c.health.state == ConnectionState.STANDBY]
            if not active:
                return None

        if strategy == "primary":
            primary = [c for c in active if c.is_primary]
            if primary:
                return sorted(primary, key=lambda c: c.priority)[0]
            return sorted(active, key=lambda c: c.priority)[0]

        elif strategy == "best":
            return min(active, key=lambda c: c.health.avg_latency or float("inf"))

        elif strategy == "least_loaded":
            def rate_limit_budget(conn: PooledConnection) -> float:
                rl = conn.exchange.rate_limiter
                rl._refill()
                return rl._minute_tokens / rl.per_minute
            return max(active, key=rate_limit_budget)

        elif strategy == "round_robin":
            # Simple round-robin based on total requests
            return min(active, key=lambda c: c.health.total_requests)

        return active[0]

    def get_all_connections(
        self,
        exchange_name: Optional[str] = None,
    ) -> List[PooledConnection]:
        """Get all connections, optionally filtered by exchange."""
        return self._filter_connections(exchange_name)

    def _filter_connections(
        self,
        exchange_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[PooledConnection]:
        """Filter connections by criteria."""
        result = list(self._connections.values())
        if exchange_name:
            result = [c for c in result if c.exchange.exchange_name == exchange_name]
        if tags:
            tag_set = set(tags)
            result = [c for c in result if tag_set.intersection(c.tags)]
        return result

    # ------------------------------------------------------------------
    # Failover
    # ------------------------------------------------------------------

    async def _handle_failover(self, failed_conn: PooledConnection) -> Optional[PooledConnection]:
        """Handle failover from a failed connection to a backup.

        Returns:
            The new active connection, or None if no backup available.
        """
        failed_conn.health.state = ConnectionState.FAILED

        # Find a backup connection for the same exchange
        candidates = [
            c for c in self._connections.values()
            if c.exchange.exchange_name == failed_conn.exchange.exchange_name
            and c.connection_id != failed_conn.connection_id
            and c.health.state in (ConnectionState.STANDBY, ConnectionState.ACTIVE)
        ]

        if not candidates:
            self._logger.error(
                "No backup available for %s - connection %s failed",
                failed_conn.exchange.exchange_name, failed_conn.connection_id,
            )
            return None

        # Sort by priority (lower = better)
        candidates.sort(key=lambda c: c.priority)
        backup = candidates[0]

        # Initialize if in standby
        if backup.health.state == ConnectionState.STANDBY:
            try:
                await backup.exchange.initialize()
                backup.health.state = ConnectionState.ACTIVE
            except Exception as exc:
                self._logger.error("Backup %s failed to initialize: %s", backup.connection_id, exc)
                backup.health.state = ConnectionState.FAILED
                return None

        backup.is_primary = True
        backup.health.state = ConnectionState.ACTIVE
        self._logger.info(
            "Failover: %s -> %s for %s",
            failed_conn.connection_id, backup.connection_id,
            failed_conn.exchange.exchange_name,
        )

        if self._on_failover:
            try:
                result = self._on_failover(failed_conn, backup)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                self._logger.error("Failover callback error: %s", exc)

        return backup

    async def execute_with_failover(
        self,
        operation: Callable,
        exchange_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_attempts: int = 3,
    ) -> Any:
        """Execute an operation with automatic failover on failure.

        Args:
            operation: Async callable that takes an ExchangeBase instance.
            exchange_name: Target exchange name.
            tags: Connection filter tags.
            max_attempts: Maximum failover attempts.

        Returns:
            The result of the operation.

        Raises:
            ExchangeError: If all attempts fail.
        """
        last_error = None

        for attempt in range(max_attempts):
            conn = self.get_connection(exchange_name, tags, strategy="primary")
            if not conn:
                raise ExchangeError(
                    f"No available connection for {exchange_name or 'any exchange'}",
                    exchange="pool",
                )

            t_start = time.monotonic()
            try:
                result = await operation(conn.exchange)
                latency_ms = (time.monotonic() - t_start) * 1000
                conn.health.record_success(latency_ms)
                return result

            except (NetworkError, RateLimitError) as exc:
                conn.health.record_error()
                last_error = exc
                self._logger.warning(
                    "Connection %s error (attempt %d/%d): %s",
                    conn.connection_id, attempt + 1, max_attempts, exc,
                )

                if conn.health.consecutive_errors >= self._max_consecutive_errors:
                    backup = await self._handle_failover(conn)
                    if not backup:
                        raise

            except ExchangeError as exc:
                conn.health.record_error()
                raise  # Non-retryable errors (auth, invalid order, etc)

        raise last_error or ExchangeError("All failover attempts exhausted", exchange="pool")

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring task."""
        if self._health_check_task and not self._health_check_task.done():
            return
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._logger.info("Health monitoring started (interval=%.1fs)", self._health_check_interval)

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Health monitoring stopped")

    async def _health_check_loop(self) -> None:
        """Background loop that periodically checks all connections."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Health check error: %s", exc)

    async def _run_health_checks(self) -> None:
        """Run health checks on all connections."""
        now = time.monotonic()

        for conn in list(self._connections.values()):
            conn.health.last_health_check = now

            # Skip failed connections that are in cooldown
            if conn.health.state == ConnectionState.FAILED:
                if (now - conn.health.last_error_time) < self._failover_cooldown:
                    continue
                # Try to recover
                try:
                    if not conn.exchange.is_initialized:
                        await conn.exchange.initialize()
                    exchange_health = conn.exchange.get_health()
                    if exchange_health.connected:
                        self._update_state(conn, ConnectionState.ACTIVE)
                        conn.health.consecutive_errors = 0
                        self._logger.info("Connection %s recovered", conn.connection_id)
                except Exception as exc:
                    self._logger.debug("Recovery check failed for %s: %s", conn.connection_id, exc)
                continue

            if conn.health.state not in (ConnectionState.ACTIVE, ConnectionState.DEGRADED):
                continue

            # Check exchange health
            try:
                exchange_health = conn.exchange.get_health()
                if exchange_health.status == "healthy":
                    self._update_state(conn, ConnectionState.ACTIVE)
                elif exchange_health.status == "degraded":
                    self._update_state(conn, ConnectionState.DEGRADED)
                elif exchange_health.status == "disconnected":
                    conn.health.record_error()
                    if conn.health.consecutive_errors >= self._max_consecutive_errors:
                        self._update_state(conn, ConnectionState.FAILED)
                        await self._handle_failover(conn)
            except Exception as exc:
                conn.health.record_error()
                self._logger.warning("Health check failed for %s: %s", conn.connection_id, exc)

            # Check latency threshold
            if conn.health.avg_latency > self._latency_threshold_ms:
                self._update_state(conn, ConnectionState.DEGRADED)

    def _update_state(self, conn: PooledConnection, new_state: ConnectionState) -> None:
        """Update connection state and fire callback if changed."""
        old_state = conn.health.state
        if old_state == new_state:
            return
        conn.health.state = new_state
        self._logger.info(
            "Connection %s state: %s -> %s",
            conn.connection_id, old_state.value, new_state.value,
        )
        if self._on_state_change:
            try:
                result = self._on_state_change(conn, old_state, new_state)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Rate limit budget management
    # ------------------------------------------------------------------

    def get_rate_limit_budget(self, exchange_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get rate limit budget status for all connections."""
        result = {}
        for conn in self._connections.values():
            if exchange_name and conn.exchange.exchange_name != exchange_name:
                continue
            rl = conn.exchange.rate_limiter
            result[conn.connection_id] = rl.stats
        return result

    def get_total_budget_pct(self, exchange_name: str) -> float:
        """Get total remaining rate limit budget as percentage across all
        connections for an exchange. Useful for throttling decisions.
        """
        connections = [
            c for c in self._connections.values()
            if c.exchange.exchange_name == exchange_name
            and c.health.state == ConnectionState.ACTIVE
        ]
        if not connections:
            return 0.0

        total_remaining = 0.0
        total_capacity = 0.0
        for conn in connections:
            rl = conn.exchange.rate_limiter
            rl._refill()
            total_remaining += rl._minute_tokens
            total_capacity += rl.per_minute

        return (total_remaining / total_capacity * 100) if total_capacity > 0 else 0.0

    # ------------------------------------------------------------------
    # Pool status and callbacks
    # ------------------------------------------------------------------

    def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive pool status."""
        connections_by_state: Dict[str, int] = {}
        for conn in self._connections.values():
            state = conn.health.state.value
            connections_by_state[state] = connections_by_state.get(state, 0) + 1

        exchanges: Dict[str, int] = {}
        for conn in self._connections.values():
            name = conn.exchange.exchange_name
            exchanges[name] = exchanges.get(name, 0) + 1

        return {
            "total_connections": len(self._connections),
            "max_connections": self._max_connections,
            "connections_by_state": connections_by_state,
            "exchanges": exchanges,
            "health_monitoring_active": bool(
                self._health_check_task and not self._health_check_task.done()
            ),
            "connections": [
                c.health.to_dict() for c in self._connections.values()
            ],
        }

    def on_failover(self, callback: Callable) -> None:
        """Register a callback for failover events.
        callback(failed_conn: PooledConnection, new_conn: PooledConnection)
        """
        self._on_failover = callback

    def on_state_change(self, callback: Callable) -> None:
        """Register a callback for connection state changes.
        callback(conn: PooledConnection, old_state, new_state)
        """
        self._on_state_change = callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close_all(self) -> None:
        """Close all connections and stop health monitoring."""
        await self.stop_health_monitoring()

        for conn_id in list(self._connections.keys()):
            await self.remove_connection(conn_id)

        self._logger.info("All connections closed")

    async def __aenter__(self):
        await self.start_health_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()

    def __len__(self) -> int:
        return len(self._connections)

    def __repr__(self) -> str:
        active = sum(
            1 for c in self._connections.values()
            if c.health.state == ConnectionState.ACTIVE
        )
        return f"<ExchangeConnectionPool {active}/{len(self._connections)} active>"
