"""
Alert Management for Mefai Autotrade.
Configurable alert rules engine with built-in alerts for PnL thresholds,
drawdown, position sizing, price spikes, error rates, and system resources.
Supports alert deduplication, acknowledgment, escalation, and history.
"""

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("mefai.autotrade")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def level(self) -> int:
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}[self.value]


class AlertState(Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    SUPPRESSED = "SUPPRESSED"


# ---------------------------------------------------------------------------
# Alert Data
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single alert instance."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    title: str
    message: str
    state: AlertState
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    notification_sent: bool = False
    escalation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": (
                self.resolved_at.isoformat() if self.resolved_at else None
            ),
            "escalated_at": (
                self.escalated_at.isoformat() if self.escalated_at else None
            ),
            "details": self.details,
            "escalation_count": self.escalation_count,
        }

    @property
    def is_active(self) -> bool:
        return self.state in (AlertState.ACTIVE, AlertState.ESCALATED)

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


# ---------------------------------------------------------------------------
# Alert Rule Base
# ---------------------------------------------------------------------------

class AlertRule(ABC):
    """Base class for alert rules.

    Subclasses implement evaluate() which checks the current state and
    returns (should_alert, severity, title, message, details) or None.
    """

    def __init__(
        self,
        name: str,
        cooldown_seconds: float = 300.0,
        escalation_after_seconds: float = 900.0,
        enabled: bool = True,
    ):
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        self.escalation_after_seconds = escalation_after_seconds
        self.enabled = enabled
        self._last_trigger_time = 0.0

    @abstractmethod
    def evaluate(
        self, context: Dict[str, Any]
    ) -> Optional[tuple]:
        """Evaluate the rule against current context.

        Returns:
            None if no alert, or (severity, title, message, details) tuple.
        """
        ...

    def can_trigger(self) -> bool:
        """Check if cooldown has elapsed."""
        if not self.enabled:
            return False
        return time.time() - self._last_trigger_time >= self.cooldown_seconds

    def mark_triggered(self) -> None:
        self._last_trigger_time = time.time()


# ---------------------------------------------------------------------------
# Built-in Alert Rules
# ---------------------------------------------------------------------------

class PnLThresholdRule(AlertRule):
    """Alert when daily PnL exceeds a loss threshold."""

    def __init__(
        self,
        max_daily_loss: float = -500.0,
        warning_threshold: float = -300.0,
        cooldown_seconds: float = 600.0,
        **kwargs,
    ):
        super().__init__(name="pnl_threshold", cooldown_seconds=cooldown_seconds, **kwargs)
        self.max_daily_loss = max_daily_loss
        self.warning_threshold = warning_threshold

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        daily_pnl = context.get("daily_pnl", 0.0)

        if daily_pnl <= self.max_daily_loss:
            return (
                AlertSeverity.CRITICAL,
                "Daily loss limit breached",
                f"Daily PnL: {daily_pnl:+.2f} USDT (limit: {self.max_daily_loss:.2f})",
                {"daily_pnl": daily_pnl, "threshold": self.max_daily_loss},
            )
        elif daily_pnl <= self.warning_threshold:
            return (
                AlertSeverity.WARNING,
                "Daily loss warning",
                f"Daily PnL: {daily_pnl:+.2f} USDT approaching limit "
                f"({self.max_daily_loss:.2f})",
                {"daily_pnl": daily_pnl, "threshold": self.warning_threshold},
            )
        return None


class DrawdownThresholdRule(AlertRule):
    """Alert when drawdown exceeds a threshold percentage."""

    def __init__(
        self,
        max_drawdown_pct: float = 10.0,
        warning_pct: float = 5.0,
        cooldown_seconds: float = 600.0,
        **kwargs,
    ):
        super().__init__(name="drawdown_threshold", cooldown_seconds=cooldown_seconds, **kwargs)
        self.max_drawdown_pct = max_drawdown_pct
        self.warning_pct = warning_pct

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        drawdown = context.get("current_drawdown_pct", 0.0)

        if drawdown >= self.max_drawdown_pct:
            return (
                AlertSeverity.CRITICAL,
                "Maximum drawdown breached",
                f"Current drawdown: {drawdown:.2f}% (limit: {self.max_drawdown_pct:.2f}%)",
                {"drawdown_pct": drawdown, "threshold": self.max_drawdown_pct},
            )
        elif drawdown >= self.warning_pct:
            return (
                AlertSeverity.WARNING,
                "Drawdown warning",
                f"Current drawdown: {drawdown:.2f}% (limit: {self.max_drawdown_pct:.2f}%)",
                {"drawdown_pct": drawdown, "threshold": self.warning_pct},
            )
        return None


class PositionSizeRule(AlertRule):
    """Alert when a single position exceeds a portfolio percentage."""

    def __init__(
        self,
        max_position_pct: float = 25.0,
        warning_pct: float = 15.0,
        cooldown_seconds: float = 300.0,
        **kwargs,
    ):
        super().__init__(name="position_size", cooldown_seconds=cooldown_seconds, **kwargs)
        self.max_position_pct = max_position_pct
        self.warning_pct = warning_pct

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        positions = context.get("positions", [])
        portfolio_value = context.get("portfolio_value", 0.0)

        if portfolio_value <= 0:
            return None

        for pos in positions:
            notional = pos.get("notional", 0.0)
            pct = abs(notional) / portfolio_value * 100
            symbol = pos.get("symbol", "UNKNOWN")

            if pct >= self.max_position_pct:
                return (
                    AlertSeverity.CRITICAL,
                    f"Position too large: {symbol}",
                    f"{symbol} is {pct:.1f}% of portfolio "
                    f"(limit: {self.max_position_pct:.1f}%)",
                    {
                        "symbol": symbol,
                        "position_pct": pct,
                        "notional": notional,
                        "threshold": self.max_position_pct,
                    },
                )
            elif pct >= self.warning_pct:
                return (
                    AlertSeverity.WARNING,
                    f"Large position warning: {symbol}",
                    f"{symbol} is {pct:.1f}% of portfolio",
                    {
                        "symbol": symbol,
                        "position_pct": pct,
                        "notional": notional,
                    },
                )
        return None


class PriceSpikeRule(AlertRule):
    """Alert on unusual price movements."""

    def __init__(
        self,
        spike_pct: float = 5.0,
        window_minutes: int = 5,
        cooldown_seconds: float = 300.0,
        **kwargs,
    ):
        super().__init__(name="price_spike", cooldown_seconds=cooldown_seconds, **kwargs)
        self.spike_pct = spike_pct
        self.window_minutes = window_minutes

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        price_changes = context.get("price_changes", {})
        # price_changes: {symbol: {window_minutes: pct_change}}

        for symbol, changes in price_changes.items():
            change = changes.get(self.window_minutes, 0.0)
            if abs(change) >= self.spike_pct:
                severity = (
                    AlertSeverity.CRITICAL
                    if abs(change) >= self.spike_pct * 2
                    else AlertSeverity.WARNING
                )
                direction = "up" if change > 0 else "down"
                return (
                    severity,
                    f"Price spike: {symbol}",
                    f"{symbol} moved {change:+.2f}% in {self.window_minutes}min",
                    {
                        "symbol": symbol,
                        "change_pct": change,
                        "direction": direction,
                        "window_minutes": self.window_minutes,
                    },
                )
        return None


class ErrorRateRule(AlertRule):
    """Alert when error rate exceeds threshold."""

    def __init__(
        self,
        max_errors_per_minute: float = 10.0,
        warning_rate: float = 5.0,
        cooldown_seconds: float = 300.0,
        **kwargs,
    ):
        super().__init__(name="error_rate", cooldown_seconds=cooldown_seconds, **kwargs)
        self.max_rate = max_errors_per_minute
        self.warning_rate = warning_rate

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        rate = context.get("error_rate_per_minute", 0.0)

        if rate >= self.max_rate:
            return (
                AlertSeverity.CRITICAL,
                "Error rate critical",
                f"Error rate: {rate:.1f}/min (limit: {self.max_rate:.1f}/min)",
                {"error_rate": rate, "threshold": self.max_rate},
            )
        elif rate >= self.warning_rate:
            return (
                AlertSeverity.WARNING,
                "Error rate elevated",
                f"Error rate: {rate:.1f}/min (warning: {self.warning_rate:.1f}/min)",
                {"error_rate": rate, "threshold": self.warning_rate},
            )
        return None


class ResourceUsageRule(AlertRule):
    """Alert on system resource usage thresholds."""

    def __init__(
        self,
        memory_warning_pct: float = 80.0,
        memory_critical_pct: float = 95.0,
        disk_warning_pct: float = 85.0,
        disk_critical_pct: float = 95.0,
        cooldown_seconds: float = 600.0,
        **kwargs,
    ):
        super().__init__(name="resource_usage", cooldown_seconds=cooldown_seconds, **kwargs)
        self.memory_warning = memory_warning_pct
        self.memory_critical = memory_critical_pct
        self.disk_warning = disk_warning_pct
        self.disk_critical = disk_critical_pct

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        resources = context.get("system_resources", {})
        memory = resources.get("memory", {})
        disk = resources.get("disk", {})

        mem_pct = memory.get("used_percent", 0.0)
        disk_pct = disk.get("used_percent", 0.0)

        # Check memory first (usually more critical)
        if mem_pct >= self.memory_critical:
            return (
                AlertSeverity.CRITICAL,
                "Memory usage critical",
                f"Memory: {mem_pct:.1f}% used",
                {"memory_used_pct": mem_pct, "threshold": self.memory_critical},
            )
        if disk_pct >= self.disk_critical:
            return (
                AlertSeverity.CRITICAL,
                "Disk usage critical",
                f"Disk: {disk_pct:.1f}% used",
                {"disk_used_pct": disk_pct, "threshold": self.disk_critical},
            )
        if mem_pct >= self.memory_warning:
            return (
                AlertSeverity.WARNING,
                "Memory usage warning",
                f"Memory: {mem_pct:.1f}% used",
                {"memory_used_pct": mem_pct, "threshold": self.memory_warning},
            )
        if disk_pct >= self.disk_warning:
            return (
                AlertSeverity.WARNING,
                "Disk usage warning",
                f"Disk: {disk_pct:.1f}% used",
                {"disk_used_pct": disk_pct, "threshold": self.disk_warning},
            )
        return None


class StrategyDegradationRule(AlertRule):
    """Alert when a strategy performance degrades below thresholds."""

    def __init__(
        self,
        min_win_rate: float = 0.30,
        min_profit_factor: float = 0.5,
        min_trades_for_eval: int = 10,
        cooldown_seconds: float = 1800.0,
        **kwargs,
    ):
        super().__init__(
            name="strategy_degradation",
            cooldown_seconds=cooldown_seconds,
            **kwargs,
        )
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_trades = min_trades_for_eval

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        strategy_metrics = context.get("strategy_metrics", {})

        for strategy, metrics in strategy_metrics.items():
            total = metrics.get("total_trades", 0)
            if total < self.min_trades:
                continue

            win_rate = metrics.get("win_rate", 0.0)
            pf = metrics.get("profit_factor", 0.0)

            if win_rate < self.min_win_rate or pf < self.min_profit_factor:
                return (
                    AlertSeverity.WARNING,
                    f"Strategy degradation: {strategy}",
                    f"{strategy}: win_rate={win_rate:.1%}, "
                    f"profit_factor={pf:.2f} over {total} trades",
                    {
                        "strategy": strategy,
                        "win_rate": win_rate,
                        "profit_factor": pf,
                        "total_trades": total,
                    },
                )
        return None


class ExchangeAPIRule(AlertRule):
    """Alert on exchange API errors."""

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        cooldown_seconds: float = 120.0,
        **kwargs,
    ):
        super().__init__(
            name="exchange_api",
            cooldown_seconds=cooldown_seconds,
            **kwargs,
        )
        self.max_failures = max_consecutive_failures

    def evaluate(self, context: Dict[str, Any]) -> Optional[tuple]:
        exchange_status = context.get("exchange_status", {})

        for exchange, status in exchange_status.items():
            failures = status.get("consecutive_failures", 0)
            last_error = status.get("last_error", "")

            if failures >= self.max_failures:
                return (
                    AlertSeverity.CRITICAL,
                    f"Exchange API failure: {exchange}",
                    f"{exchange}: {failures} consecutive failures - {last_error}",
                    {
                        "exchange": exchange,
                        "consecutive_failures": failures,
                        "last_error": last_error,
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------

class AlertManager:
    """Central alert management system.

    Manages alert rules, evaluates them against current context,
    deduplicates alerts, handles acknowledgment and escalation,
    and maintains alert history.

    Usage:
        manager = AlertManager()
        manager.add_rule(PnLThresholdRule(max_daily_loss=-500))
        manager.add_rule(DrawdownThresholdRule(max_drawdown_pct=10))
        manager.set_notification_callback(notify_fn)
        alerts = await manager.evaluate(context)
    """

    MAX_HISTORY = 5000

    def __init__(self):
        self._lock = threading.Lock()
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}  # rule_name -> active alert
        self._alert_history: List[Alert] = []
        self._suppressed_rules: Set[str] = set()
        self._notification_callback: Optional[Callable] = None
        self._escalation_callback: Optional[Callable] = None

        # Deduplication: track alert hashes to avoid duplicates
        self._dedup_hashes: Dict[str, float] = {}  # hash -> last seen time
        self._dedup_window = 60.0  # seconds

    # -------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule."""
        self._rules.pop(rule_name, None)

    def enable_rule(self, rule_name: str) -> None:
        """Enable a disabled rule."""
        if rule_name in self._rules:
            self._rules[rule_name].enabled = True
        self._suppressed_rules.discard(rule_name)

    def disable_rule(self, rule_name: str) -> None:
        """Disable a rule."""
        if rule_name in self._rules:
            self._rules[rule_name].enabled = False

    def suppress_rule(self, rule_name: str, duration_seconds: float = 3600.0) -> None:
        """Temporarily suppress a rule."""
        self._suppressed_rules.add(rule_name)
        # Schedule unsuppression (simple approach using threading timer)
        timer = threading.Timer(
            duration_seconds, lambda: self._suppressed_rules.discard(rule_name)
        )
        timer.daemon = True
        timer.start()

    def set_notification_callback(self, callback: Callable) -> None:
        """Set the callback for sending alert notifications.

        Callback signature: async def callback(alert: Alert) -> None
        """
        self._notification_callback = callback

    def set_escalation_callback(self, callback: Callable) -> None:
        """Set the callback for escalated alerts.

        Callback signature: async def callback(alert: Alert) -> None
        """
        self._escalation_callback = callback

    def add_default_rules(self) -> None:
        """Add all default built-in rules with standard thresholds."""
        self.add_rule(PnLThresholdRule())
        self.add_rule(DrawdownThresholdRule())
        self.add_rule(PositionSizeRule())
        self.add_rule(PriceSpikeRule())
        self.add_rule(ErrorRateRule())
        self.add_rule(ResourceUsageRule())
        self.add_rule(StrategyDegradationRule())
        self.add_rule(ExchangeAPIRule())

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

    async def evaluate(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against the current context.

        Returns a list of newly triggered or updated alerts.
        """
        new_alerts: List[Alert] = []

        with self._lock:
            # Check for escalations first
            await self._check_escalations()

            for rule_name, rule in self._rules.items():
                if rule_name in self._suppressed_rules:
                    continue
                if not rule.can_trigger():
                    continue

                try:
                    result = rule.evaluate(context)
                except Exception as e:
                    logger.error(
                        f"Alert rule {rule_name} evaluation failed: {e}",
                        exc_info=True,
                    )
                    continue

                if result is None:
                    # Condition cleared - resolve active alert
                    if rule_name in self._active_alerts:
                        alert = self._active_alerts[rule_name]
                        alert.state = AlertState.RESOLVED
                        alert.resolved_at = datetime.now(timezone.utc)
                        alert.updated_at = datetime.now(timezone.utc)
                        self._alert_history.append(alert)
                        del self._active_alerts[rule_name]
                    continue

                severity, title, message, details = result

                # Deduplication check
                dedup_key = f"{rule_name}:{severity.value}:{title}"
                now = time.time()
                if dedup_key in self._dedup_hashes:
                    if now - self._dedup_hashes[dedup_key] < self._dedup_window:
                        continue
                self._dedup_hashes[dedup_key] = now

                # Create or update alert
                if rule_name in self._active_alerts:
                    # Update existing alert
                    alert = self._active_alerts[rule_name]
                    alert.severity = severity
                    alert.message = message
                    alert.details = details
                    alert.updated_at = datetime.now(timezone.utc)
                else:
                    # New alert
                    alert = Alert(
                        alert_id=str(uuid.uuid4())[:12],
                        rule_name=rule_name,
                        severity=severity,
                        title=title,
                        message=message,
                        state=AlertState.ACTIVE,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        details=details,
                    )
                    self._active_alerts[rule_name] = alert
                    new_alerts.append(alert)
                    rule.mark_triggered()

                    logger.warning(
                        f"Alert [{severity.value}]: {title} - {message}"
                    )

        # Send notifications for new alerts (outside lock)
        for alert in new_alerts:
            await self._send_notification(alert)

        # Prune dedup cache and history
        self._prune()

        return new_alerts

    async def _check_escalations(self) -> None:
        """Escalate unacknowledged alerts that have been active too long."""
        now = datetime.now(timezone.utc)

        for rule_name, alert in list(self._active_alerts.items()):
            if alert.state != AlertState.ACTIVE:
                continue

            rule = self._rules.get(rule_name)
            if not rule:
                continue

            age = (now - alert.created_at).total_seconds()
            if age >= rule.escalation_after_seconds:
                alert.state = AlertState.ESCALATED
                alert.escalated_at = now
                alert.escalation_count += 1
                alert.updated_at = now

                logger.warning(
                    f"Alert escalated: {alert.title} "
                    f"(age={age:.0f}s, escalation #{alert.escalation_count})"
                )

                if self._escalation_callback:
                    try:
                        if asyncio.iscoroutinefunction(self._escalation_callback):
                            await self._escalation_callback(alert)
                        else:
                            self._escalation_callback(alert)
                    except Exception as e:
                        logger.error(f"Escalation callback failed: {e}")

    async def _send_notification(self, alert: Alert) -> None:
        """Send notification for an alert."""
        if self._notification_callback:
            try:
                if asyncio.iscoroutinefunction(self._notification_callback):
                    await self._notification_callback(alert)
                else:
                    self._notification_callback(alert)
                alert.notification_sent = True
            except Exception as e:
                logger.error(f"Alert notification failed: {e}")

    def _prune(self) -> None:
        """Clean up old dedup entries and trim history."""
        now = time.time()
        self._dedup_hashes = {
            k: v
            for k, v in self._dedup_hashes.items()
            if now - v < self._dedup_window * 2
        }
        if len(self._alert_history) > self.MAX_HISTORY:
            self._alert_history = self._alert_history[-self.MAX_HISTORY:]

    # -------------------------------------------------------------------
    # Alert Actions
    # -------------------------------------------------------------------

    def acknowledge(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert."""
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.alert_id == alert_id:
                    alert.state = AlertState.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.now(timezone.utc)
                    alert.acknowledged_by = user
                    alert.updated_at = datetime.now(timezone.utc)
                    logger.info(
                        f"Alert acknowledged: {alert.title} by {user}"
                    )
                    return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Manually resolve an alert."""
        with self._lock:
            for rule_name, alert in list(self._active_alerts.items()):
                if alert.alert_id == alert_id:
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.updated_at = datetime.now(timezone.utc)
                    self._alert_history.append(alert)
                    del self._active_alerts[rule_name]
                    logger.info(f"Alert resolved: {alert.title}")
                    return True
        return False

    def resolve_by_rule(self, rule_name: str) -> bool:
        """Resolve an alert by rule name."""
        with self._lock:
            if rule_name in self._active_alerts:
                alert = self._active_alerts[rule_name]
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.updated_at = datetime.now(timezone.utc)
                self._alert_history.append(alert)
                del self._active_alerts[rule_name]
                return True
        return False

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity."""
        with self._lock:
            alerts = list(self._active_alerts.values())
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return sorted(
                alerts, key=lambda a: a.severity.level, reverse=True
            )

    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        rule_name: Optional[str] = None,
    ) -> List[Alert]:
        """Get alert history with optional filters."""
        with self._lock:
            alerts = list(self._alert_history)
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if rule_name:
                alerts = [a for a in alerts if a.rule_name == rule_name]
            return alerts[-limit:]

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Find an alert by ID in active alerts or history."""
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.alert_id == alert_id:
                    return alert
            for alert in self._alert_history:
                if alert.alert_id == alert_id:
                    return alert
        return None

    def get_alert_counts(self) -> Dict[str, int]:
        """Get counts of alerts by severity and state."""
        with self._lock:
            counts = {
                "active": 0,
                "acknowledged": 0,
                "escalated": 0,
                "critical": 0,
                "warning": 0,
                "info": 0,
                "total_history": len(self._alert_history),
            }
            for alert in self._active_alerts.values():
                if alert.state == AlertState.ACTIVE:
                    counts["active"] += 1
                elif alert.state == AlertState.ACKNOWLEDGED:
                    counts["acknowledged"] += 1
                elif alert.state == AlertState.ESCALATED:
                    counts["escalated"] += 1

                if alert.severity == AlertSeverity.CRITICAL:
                    counts["critical"] += 1
                elif alert.severity == AlertSeverity.WARNING:
                    counts["warning"] += 1
                else:
                    counts["info"] += 1

            return counts

    def get_rules_status(self) -> List[Dict[str, Any]]:
        """Get the status of all registered rules."""
        with self._lock:
            return [
                {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "escalation_after_seconds": rule.escalation_after_seconds,
                    "suppressed": rule.name in self._suppressed_rules,
                    "has_active_alert": rule.name in self._active_alerts,
                }
                for rule in self._rules.values()
            ]
