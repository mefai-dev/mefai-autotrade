"""
System Health Monitoring for Mefai Autotrade.
Monitors exchange connectivity, WebSocket status, data freshness, strategy
heartbeats, database connectivity, system resources, error rates, and
provides auto-restart on critical failure detection.
"""

import asyncio
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mefai.autotrade")


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"

    @property
    def is_ok(self) -> bool:
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "latency_ms": round(self.latency_ms, 1),
            "details": self.details,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class HealthReport:
    """Complete system health report."""
    overall_status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: List[ComponentHealth]
    system_resources: Dict[str, Any]
    error_rate_per_minute: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": round(self.uptime_seconds, 0),
            "uptime_human": self._format_uptime(),
            "error_rate_per_minute": round(self.error_rate_per_minute, 2),
            "system_resources": self.system_resources,
            "components": [c.to_dict() for c in self.components],
        }

    def _format_uptime(self) -> str:
        s = self.uptime_seconds
        days = int(s // 86400)
        hours = int((s % 86400) // 3600)
        minutes = int((s % 3600) // 60)
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"


# ---------------------------------------------------------------------------
# Health Check Functions
# ---------------------------------------------------------------------------

class _SystemResourceChecker:
    """Reads system resource usage from /proc (Linux) or psutil fallback."""

    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1]) * 1024  # Convert KB to bytes
                    info[key] = value

            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            used = total - available
            used_pct = (used / total * 100) if total > 0 else 0.0

            return {
                "total_bytes": total,
                "used_bytes": used,
                "available_bytes": available,
                "used_percent": round(used_pct, 1),
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
            }
        except (OSError, ValueError, KeyError):
            return {"total_bytes": 0, "used_bytes": 0, "used_percent": 0.0}

    @staticmethod
    def get_cpu_usage() -> Dict[str, Any]:
        """Get CPU usage from /proc/loadavg."""
        try:
            with open("/proc/loadavg", "r") as f:
                parts = f.read().strip().split()

            cpu_count = os.cpu_count() or 1
            load_1 = float(parts[0])
            load_5 = float(parts[1])
            load_15 = float(parts[2])

            return {
                "load_1m": load_1,
                "load_5m": load_5,
                "load_15m": load_15,
                "cpu_count": cpu_count,
                "load_percent_1m": round(load_1 / cpu_count * 100, 1),
                "load_percent_5m": round(load_5 / cpu_count * 100, 1),
                "load_percent_15m": round(load_15 / cpu_count * 100, 1),
            }
        except (OSError, ValueError, IndexError):
            return {
                "load_1m": 0.0,
                "load_5m": 0.0,
                "load_15m": 0.0,
                "cpu_count": os.cpu_count() or 1,
            }

    @staticmethod
    def get_disk_usage(path: str = "/") -> Dict[str, Any]:
        """Get disk usage for the given path."""
        try:
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            available = stat.f_bavail * stat.f_frsize
            used = total - free
            used_pct = (used / total * 100) if total > 0 else 0.0

            return {
                "path": path,
                "total_bytes": total,
                "used_bytes": used,
                "available_bytes": available,
                "used_percent": round(used_pct, 1),
                "total_gb": round(total / (1024 ** 3), 2),
                "available_gb": round(available / (1024 ** 3), 2),
            }
        except OSError:
            return {"path": path, "total_bytes": 0, "used_percent": 0.0}

    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """Get current process information."""
        pid = os.getpid()
        try:
            with open(f"/proc/{pid}/status", "r") as f:
                status_lines = f.readlines()

            info: Dict[str, str] = {}
            for line in status_lines:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    info[parts[0].strip()] = parts[1].strip()

            # Read RSS from /proc/pid/statm (in pages)
            with open(f"/proc/{pid}/statm", "r") as f:
                statm = f.read().strip().split()
            page_size = os.sysconf("SC_PAGE_SIZE")
            rss_bytes = int(statm[1]) * page_size
            vms_bytes = int(statm[0]) * page_size

            return {
                "pid": pid,
                "threads": int(info.get("Threads", "0")),
                "rss_bytes": rss_bytes,
                "vms_bytes": vms_bytes,
                "rss_mb": round(rss_bytes / (1024 ** 2), 1),
                "vms_mb": round(vms_bytes / (1024 ** 2), 1),
            }
        except (OSError, ValueError, IndexError):
            return {"pid": pid, "rss_mb": 0.0, "vms_mb": 0.0}


# ---------------------------------------------------------------------------
# Health Checker
# ---------------------------------------------------------------------------

class HealthChecker:
    """System health monitoring with configurable checks and auto-restart.

    Periodically runs health checks against registered components and
    tracks overall system status. Can trigger restart callbacks when
    critical failures are detected.

    Usage:
        checker = HealthChecker()
        checker.register_check("exchange", check_exchange_fn)
        checker.register_check("database", check_db_fn)
        await checker.start()
    """

    # Thresholds
    MEMORY_WARNING_PCT = 80.0
    MEMORY_CRITICAL_PCT = 95.0
    CPU_WARNING_LOAD = 80.0
    CPU_CRITICAL_LOAD = 95.0
    DISK_WARNING_PCT = 85.0
    DISK_CRITICAL_PCT = 95.0
    ERROR_RATE_WARNING = 5.0  # errors per minute
    ERROR_RATE_CRITICAL = 20.0
    DATA_FRESHNESS_WARNING_SECONDS = 30.0
    DATA_FRESHNESS_CRITICAL_SECONDS = 120.0
    HEARTBEAT_WARNING_SECONDS = 60.0
    HEARTBEAT_CRITICAL_SECONDS = 300.0
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        auto_restart_enabled: bool = True,
        restart_cooldown_seconds: float = 300.0,
        disk_path: str = "/",
    ):
        self._check_interval = check_interval_seconds
        self._auto_restart = auto_restart_enabled
        self._restart_cooldown = restart_cooldown_seconds
        self._disk_path = disk_path

        self._start_time = time.time()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Component health states
        self._components: Dict[str, ComponentHealth] = {}

        # Custom health check functions: name -> async callable returning (bool, str)
        self._checks: Dict[str, Callable] = {}

        # Strategy heartbeats: strategy_name -> last_heartbeat_time
        self._strategy_heartbeats: Dict[str, float] = {}

        # WebSocket stream status: stream_name -> (connected, last_update_time)
        self._ws_status: Dict[str, Tuple[bool, float]] = {}

        # Data freshness: data_source -> last_update_timestamp
        self._data_freshness: Dict[str, float] = {}

        # Error counter for rate calculation
        self._error_timestamps: List[float] = []
        self._error_lock = threading.Lock()

        # Restart tracking
        self._last_restart_time = 0.0
        self._restart_count = 0
        self._restart_callbacks: List[Callable] = []

        # Resource checker
        self._resources = _SystemResourceChecker()

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register_check(
        self,
        name: str,
        check_fn: Callable,
    ) -> None:
        """Register a custom health check function.

        The function should be an async callable that returns a tuple
        of (is_healthy: bool, message: str).
        """
        self._checks[name] = check_fn
        self._components[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN,
            message="Not yet checked",
            last_check=datetime.now(timezone.utc),
        )

    def register_restart_callback(self, callback: Callable) -> None:
        """Register a callback to be invoked on auto-restart."""
        self._restart_callbacks.append(callback)

    # -------------------------------------------------------------------
    # Heartbeats and Status Updates
    # -------------------------------------------------------------------

    def strategy_heartbeat(self, strategy_name: str) -> None:
        """Record a heartbeat from a strategy indicating it is running."""
        self._strategy_heartbeats[strategy_name] = time.time()

    def update_ws_status(self, stream_name: str, connected: bool) -> None:
        """Update WebSocket connection status."""
        self._ws_status[stream_name] = (connected, time.time())

    def update_data_freshness(self, source: str) -> None:
        """Mark a data source as having received fresh data."""
        self._data_freshness[source] = time.time()

    def record_error(self) -> None:
        """Record an error occurrence for rate calculation."""
        now = time.time()
        with self._error_lock:
            self._error_timestamps.append(now)
            # Keep only last 10 minutes of errors
            cutoff = now - 600
            self._error_timestamps = [
                t for t in self._error_timestamps if t > cutoff
            ]

    def get_error_rate(self, window_minutes: int = 5) -> float:
        """Get error rate per minute over the specified window."""
        now = time.time()
        cutoff = now - (window_minutes * 60)
        with self._error_lock:
            count = sum(1 for t in self._error_timestamps if t > cutoff)
        return count / max(window_minutes, 1)

    # -------------------------------------------------------------------
    # Health Check Execution
    # -------------------------------------------------------------------

    async def run_all_checks(self) -> HealthReport:
        """Execute all registered health checks and return a report."""
        components: List[ComponentHealth] = []

        # Run custom checks
        for name, check_fn in self._checks.items():
            component = await self._run_single_check(name, check_fn)
            components.append(component)

        # Built-in checks
        components.append(self._check_strategies())
        components.append(self._check_websockets())
        components.append(self._check_data_freshness())
        components.append(self._check_memory())
        components.append(self._check_cpu())
        components.append(self._check_disk())
        components.append(self._check_process())
        components.append(self._check_error_rate())

        # Store results
        with self._lock:
            for comp in components:
                self._components[comp.name] = comp

        # Determine overall status
        overall = self._determine_overall_status(components)

        # System resources
        resources = {
            "memory": self._resources.get_memory_usage(),
            "cpu": self._resources.get_cpu_usage(),
            "disk": self._resources.get_disk_usage(self._disk_path),
            "process": self._resources.get_process_info(),
        }

        report = HealthReport(
            overall_status=overall,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=time.time() - self._start_time,
            components=components,
            system_resources=resources,
            error_rate_per_minute=self.get_error_rate(),
        )

        # Auto-restart check
        if (
            self._auto_restart
            and overall == HealthStatus.UNHEALTHY
            and self._should_restart()
        ):
            await self._trigger_restart(report)

        return report

    async def _run_single_check(
        self,
        name: str,
        check_fn: Callable,
    ) -> ComponentHealth:
        """Run a single health check with timing and error handling."""
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(check_fn):
                is_healthy, message = await asyncio.wait_for(
                    check_fn(), timeout=10.0
                )
            else:
                is_healthy, message = check_fn()

            latency = (time.time() - start) * 1000

            prev = self._components.get(name)
            failures = 0
            if prev and not is_healthy:
                failures = prev.consecutive_failures + 1

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
            if is_healthy and latency > 5000:
                status = HealthStatus.DEGRADED

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                last_check=datetime.now(timezone.utc),
                latency_ms=latency,
                consecutive_failures=failures,
            )
        except asyncio.TimeoutError:
            latency = (time.time() - start) * 1000
            prev = self._components.get(name)
            failures = (prev.consecutive_failures + 1) if prev else 1
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out (>10s)",
                last_check=datetime.now(timezone.utc),
                latency_ms=latency,
                consecutive_failures=failures,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            prev = self._components.get(name)
            failures = (prev.consecutive_failures + 1) if prev else 1
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {type(e).__name__}: {e}",
                last_check=datetime.now(timezone.utc),
                latency_ms=latency,
                consecutive_failures=failures,
            )

    # -------------------------------------------------------------------
    # Built-in Checks
    # -------------------------------------------------------------------

    def _check_strategies(self) -> ComponentHealth:
        """Check if all strategies have sent a recent heartbeat."""
        now = time.time()
        if not self._strategy_heartbeats:
            return ComponentHealth(
                name="strategies",
                status=HealthStatus.UNKNOWN,
                message="No strategies registered",
                last_check=datetime.now(timezone.utc),
            )

        stale: List[str] = []
        dead: List[str] = []

        for name, last_hb in self._strategy_heartbeats.items():
            age = now - last_hb
            if age > self.HEARTBEAT_CRITICAL_SECONDS:
                dead.append(name)
            elif age > self.HEARTBEAT_WARNING_SECONDS:
                stale.append(name)

        if dead:
            return ComponentHealth(
                name="strategies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dead strategies: {', '.join(dead)}",
                last_check=datetime.now(timezone.utc),
                details={"dead": dead, "stale": stale},
            )
        elif stale:
            return ComponentHealth(
                name="strategies",
                status=HealthStatus.DEGRADED,
                message=f"Stale strategies: {', '.join(stale)}",
                last_check=datetime.now(timezone.utc),
                details={"stale": stale},
            )
        else:
            return ComponentHealth(
                name="strategies",
                status=HealthStatus.HEALTHY,
                message=f"All {len(self._strategy_heartbeats)} strategies active",
                last_check=datetime.now(timezone.utc),
                details={
                    "active_count": len(self._strategy_heartbeats),
                    "strategies": list(self._strategy_heartbeats.keys()),
                },
            )

    def _check_websockets(self) -> ComponentHealth:
        """Check WebSocket connection status."""
        if not self._ws_status:
            return ComponentHealth(
                name="websockets",
                status=HealthStatus.UNKNOWN,
                message="No WebSocket streams registered",
                last_check=datetime.now(timezone.utc),
            )

        disconnected: List[str] = []
        for stream, (connected, _) in self._ws_status.items():
            if not connected:
                disconnected.append(stream)

        if disconnected:
            return ComponentHealth(
                name="websockets",
                status=HealthStatus.UNHEALTHY,
                message=f"Disconnected: {', '.join(disconnected)}",
                last_check=datetime.now(timezone.utc),
                details={"disconnected": disconnected},
            )

        return ComponentHealth(
            name="websockets",
            status=HealthStatus.HEALTHY,
            message=f"All {len(self._ws_status)} streams connected",
            last_check=datetime.now(timezone.utc),
            details={"stream_count": len(self._ws_status)},
        )

    def _check_data_freshness(self) -> ComponentHealth:
        """Check if data sources are receiving fresh data."""
        now = time.time()
        if not self._data_freshness:
            return ComponentHealth(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                message="No data sources registered",
                last_check=datetime.now(timezone.utc),
            )

        stale: List[str] = []
        critical: List[str] = []
        for source, last_update in self._data_freshness.items():
            age = now - last_update
            if age > self.DATA_FRESHNESS_CRITICAL_SECONDS:
                critical.append(f"{source} ({age:.0f}s)")
            elif age > self.DATA_FRESHNESS_WARNING_SECONDS:
                stale.append(f"{source} ({age:.0f}s)")

        if critical:
            return ComponentHealth(
                name="data_freshness",
                status=HealthStatus.UNHEALTHY,
                message=f"Critical staleness: {', '.join(critical)}",
                last_check=datetime.now(timezone.utc),
                details={"critical": critical, "stale": stale},
            )
        elif stale:
            return ComponentHealth(
                name="data_freshness",
                status=HealthStatus.DEGRADED,
                message=f"Stale sources: {', '.join(stale)}",
                last_check=datetime.now(timezone.utc),
                details={"stale": stale},
            )

        return ComponentHealth(
            name="data_freshness",
            status=HealthStatus.HEALTHY,
            message=f"All {len(self._data_freshness)} sources fresh",
            last_check=datetime.now(timezone.utc),
        )

    def _check_memory(self) -> ComponentHealth:
        mem = self._resources.get_memory_usage()
        used_pct = mem.get("used_percent", 0.0)

        if used_pct >= self.MEMORY_CRITICAL_PCT:
            status = HealthStatus.UNHEALTHY
            msg = f"Memory critical: {used_pct:.1f}% used"
        elif used_pct >= self.MEMORY_WARNING_PCT:
            status = HealthStatus.DEGRADED
            msg = f"Memory warning: {used_pct:.1f}% used"
        else:
            status = HealthStatus.HEALTHY
            msg = f"Memory OK: {used_pct:.1f}% used"

        return ComponentHealth(
            name="memory",
            status=status,
            message=msg,
            last_check=datetime.now(timezone.utc),
            details=mem,
        )

    def _check_cpu(self) -> ComponentHealth:
        cpu = self._resources.get_cpu_usage()
        load_pct = cpu.get("load_percent_1m", 0.0)

        if load_pct >= self.CPU_CRITICAL_LOAD:
            status = HealthStatus.UNHEALTHY
            msg = f"CPU critical: {load_pct:.1f}% load"
        elif load_pct >= self.CPU_WARNING_LOAD:
            status = HealthStatus.DEGRADED
            msg = f"CPU warning: {load_pct:.1f}% load"
        else:
            status = HealthStatus.HEALTHY
            msg = f"CPU OK: {load_pct:.1f}% load"

        return ComponentHealth(
            name="cpu",
            status=status,
            message=msg,
            last_check=datetime.now(timezone.utc),
            details=cpu,
        )

    def _check_disk(self) -> ComponentHealth:
        disk = self._resources.get_disk_usage(self._disk_path)
        used_pct = disk.get("used_percent", 0.0)

        if used_pct >= self.DISK_CRITICAL_PCT:
            status = HealthStatus.UNHEALTHY
            msg = f"Disk critical: {used_pct:.1f}% used"
        elif used_pct >= self.DISK_WARNING_PCT:
            status = HealthStatus.DEGRADED
            msg = f"Disk warning: {used_pct:.1f}% used"
        else:
            status = HealthStatus.HEALTHY
            msg = f"Disk OK: {used_pct:.1f}% used"

        return ComponentHealth(
            name="disk",
            status=status,
            message=msg,
            last_check=datetime.now(timezone.utc),
            details=disk,
        )

    def _check_process(self) -> ComponentHealth:
        proc = self._resources.get_process_info()
        uptime = time.time() - self._start_time

        return ComponentHealth(
            name="process",
            status=HealthStatus.HEALTHY,
            message=f"PID {proc.get('pid')} - RSS {proc.get('rss_mb', 0):.0f}MB - "
                    f"Uptime {uptime / 3600:.1f}h",
            last_check=datetime.now(timezone.utc),
            details={
                **proc,
                "uptime_seconds": round(uptime, 0),
                "restart_count": self._restart_count,
            },
        )

    def _check_error_rate(self) -> ComponentHealth:
        rate = self.get_error_rate()

        if rate >= self.ERROR_RATE_CRITICAL:
            status = HealthStatus.UNHEALTHY
            msg = f"Error rate critical: {rate:.1f}/min"
        elif rate >= self.ERROR_RATE_WARNING:
            status = HealthStatus.DEGRADED
            msg = f"Error rate elevated: {rate:.1f}/min"
        else:
            status = HealthStatus.HEALTHY
            msg = f"Error rate normal: {rate:.1f}/min"

        return ComponentHealth(
            name="error_rate",
            status=status,
            message=msg,
            last_check=datetime.now(timezone.utc),
            details={"errors_per_minute": round(rate, 2)},
        )

    # -------------------------------------------------------------------
    # Overall Status
    # -------------------------------------------------------------------

    @staticmethod
    def _determine_overall_status(
        components: List[ComponentHealth],
    ) -> HealthStatus:
        statuses = [c.status for c in components]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.UNKNOWN
        return HealthStatus.HEALTHY

    # -------------------------------------------------------------------
    # Auto-Restart
    # -------------------------------------------------------------------

    def _should_restart(self) -> bool:
        """Determine if we should trigger an auto-restart."""
        now = time.time()
        if now - self._last_restart_time < self._restart_cooldown:
            return False

        # Check for consecutive failures on critical components
        critical_names = {"exchange", "websockets", "database"}
        for name in critical_names:
            comp = self._components.get(name)
            if comp and comp.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                return True
        return False

    async def _trigger_restart(self, report: HealthReport) -> None:
        """Trigger auto-restart via registered callbacks."""
        self._last_restart_time = time.time()
        self._restart_count += 1

        logger.critical(
            f"Auto-restart triggered (restart #{self._restart_count}). "
            f"Status: {report.overall_status.value}"
        )

        for callback in self._restart_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Restart callback failed: {e}")

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def start(self) -> None:
        """Start the periodic health check loop."""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            f"Health checker started (interval={self._check_interval}s, "
            f"auto_restart={self._auto_restart})"
        )

    async def stop(self) -> None:
        """Stop the health check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await self.run_all_checks()
            except Exception as e:
                logger.error(f"Health check loop error: {e}", exc_info=True)
            await asyncio.sleep(self._check_interval)

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        """Get the last known health of a component."""
        with self._lock:
            return self._components.get(name)

    def get_all_component_status(self) -> Dict[str, ComponentHealth]:
        """Get last known health of all components."""
        with self._lock:
            return dict(self._components)

    def get_uptime(self) -> float:
        """Get process uptime in seconds."""
        return time.time() - self._start_time

    def is_healthy(self) -> bool:
        """Quick check if the system is healthy."""
        with self._lock:
            for comp in self._components.values():
                if comp.status == HealthStatus.UNHEALTHY:
                    return False
        return True
