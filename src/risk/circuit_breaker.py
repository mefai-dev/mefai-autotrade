"""
Mefai Autotrade - Circuit Breaker System

Emergency stop mechanisms for automated trading:
- Max consecutive losses trigger
- Max loss per hour/day trigger
- Rapid drawdown trigger
- Abnormal volatility trigger
- Exchange connectivity loss trigger
- Stale data trigger
- Manual kill switch
- State machine: NORMAL -> WARNING -> TRIGGERED -> COOLDOWN
- Auto-recovery after cooldown
- Event bus integration for notifications
"""

import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Tuple
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class CircuitBreakerState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class TriggerType(str, Enum):
    CONSECUTIVE_LOSSES = "consecutive_losses"
    HOURLY_LOSS = "hourly_loss"
    DAILY_LOSS = "daily_loss"
    RAPID_DRAWDOWN = "rapid_drawdown"
    ABNORMAL_VOLATILITY = "abnormal_volatility"
    CONNECTIVITY_LOSS = "connectivity_loss"
    STALE_DATA = "stale_data"
    MANUAL_KILL = "manual_kill"
    CUSTOM = "custom"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    # Consecutive losses
    max_consecutive_losses: int = 5
    consecutive_loss_warning: int = 3  # warn at 3

    # Loss limits (percentage of equity)
    max_hourly_loss_pct: float = 2.0
    max_daily_loss_pct: float = 5.0
    hourly_loss_warning_pct: float = 1.5
    daily_loss_warning_pct: float = 3.5

    # Rapid drawdown
    rapid_dd_pct: float = 3.0  # X% drawdown
    rapid_dd_minutes: int = 15  # in Y minutes
    rapid_dd_warning_pct: float = 2.0

    # Volatility
    volatility_multiplier: float = 3.0  # trigger when ATR > 3x normal
    volatility_lookback_periods: int = 20
    volatility_warning_multiplier: float = 2.0

    # Connectivity
    max_disconnect_seconds: int = 30
    disconnect_warning_seconds: int = 10

    # Stale data
    max_stale_seconds: int = 60  # no price update for 60s
    stale_warning_seconds: int = 30

    # Cooldown
    cooldown_minutes: int = 30
    auto_recovery: bool = True

    # Kill switch
    kill_switch_active: bool = False


@dataclass
class TriggerEvent:
    """Record of a circuit breaker trigger."""
    timestamp: datetime
    trigger_type: TriggerType
    state_before: CircuitBreakerState
    state_after: CircuitBreakerState
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class LossRecord:
    """Record of a trade loss for time-based tracking."""
    timestamp: datetime
    loss_amount: float
    loss_pct: float
    symbol: str


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Emergency stop system for automated trading.

    Monitors multiple risk conditions and transitions through states:
    NORMAL -> WARNING -> TRIGGERED -> COOLDOWN -> NORMAL

    When TRIGGERED, all new trades are blocked until the cooldown period
    expires or the system is manually reset.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.NORMAL
        self._trigger_reason: str = ""
        self._trigger_type: Optional[TriggerType] = None

        # Loss tracking
        self._consecutive_losses: int = 0
        self._loss_history: deque = deque(maxlen=10000)
        self._equity_snapshots: deque = deque(maxlen=10000)  # (timestamp, equity)

        # Connectivity tracking
        self._last_heartbeat: Optional[datetime] = None
        self._last_price_update: Optional[datetime] = None
        self._connected: bool = True

        # Volatility tracking
        self._atr_history: deque = deque(maxlen=100)
        self._normal_atr: float = 0.0

        # Cooldown
        self._cooldown_end: Optional[datetime] = None
        self._triggered_at: Optional[datetime] = None

        # Event log
        self._events: List[TriggerEvent] = []
        self._max_events = 5000

        # Callbacks
        self._trigger_callbacks: List[Callable[[TriggerEvent], None]] = []
        self._state_callbacks: List[Callable[[CircuitBreakerState, CircuitBreakerState], None]] = []

        # Warning state tracking (which conditions are in warning)
        self._active_warnings: Dict[TriggerType, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        if self.config.kill_switch_active:
            return False
        return self._state in (
            CircuitBreakerState.NORMAL,
            CircuitBreakerState.WARNING,
        )

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    def get_active_warnings(self) -> Dict[str, str]:
        """Get all currently active warning conditions."""
        return {k.value: v for k, v in self._active_warnings.items()}

    def check_all(self, equity: float) -> Tuple[CircuitBreakerState, List[str]]:
        """
        Run all circuit breaker checks.

        Should be called periodically (every few seconds) and before each trade.

        Args:
            equity: Current portfolio equity

        Returns:
            Tuple of (current_state, list_of_messages)
        """
        now = datetime.now(timezone.utc)
        messages = []

        # Record equity snapshot
        self._equity_snapshots.append((now, equity))

        # Check cooldown expiry first
        if self._state == CircuitBreakerState.COOLDOWN:
            if self._check_cooldown_expiry(now):
                messages.append("Cooldown expired - returning to NORMAL")

        # If triggered or in cooldown, don't check further
        if self._state in (CircuitBreakerState.TRIGGERED, CircuitBreakerState.COOLDOWN):
            return self._state, messages

        # Kill switch override
        if self.config.kill_switch_active:
            if self._state != CircuitBreakerState.TRIGGERED:
                self._trigger(TriggerType.MANUAL_KILL, "Kill switch activated", now)
                messages.append("KILL SWITCH ACTIVE")
            return self._state, messages

        # Clear previous warnings
        self._active_warnings.clear()

        # Run all checks
        checks = [
            self._check_consecutive_losses(now),
            self._check_hourly_loss(equity, now),
            self._check_daily_loss(equity, now),
            self._check_rapid_drawdown(equity, now),
            self._check_volatility(now),
            self._check_connectivity(now),
            self._check_stale_data(now),
        ]

        for check_messages in checks:
            messages.extend(check_messages)

        # Update warning state
        if self._active_warnings and self._state == CircuitBreakerState.NORMAL:
            self._set_state(CircuitBreakerState.WARNING)
        elif not self._active_warnings and self._state == CircuitBreakerState.WARNING:
            self._set_state(CircuitBreakerState.NORMAL)

        return self._state, messages

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_consecutive_losses(self, now: datetime) -> List[str]:
        """Check consecutive loss count."""
        messages = []

        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger(
                TriggerType.CONSECUTIVE_LOSSES,
                f"Max consecutive losses reached: {self._consecutive_losses}",
                now,
            )
            messages.append(
                f"TRIGGERED: {self._consecutive_losses} consecutive losses"
            )
        elif self._consecutive_losses >= self.config.consecutive_loss_warning:
            self._active_warnings[TriggerType.CONSECUTIVE_LOSSES] = (
                f"{self._consecutive_losses} consecutive losses "
                f"(trigger at {self.config.max_consecutive_losses})"
            )
            messages.append(
                f"WARNING: {self._consecutive_losses} consecutive losses"
            )

        return messages

    def _check_hourly_loss(self, equity: float, now: datetime) -> List[str]:
        """Check loss in the last hour."""
        messages = []
        one_hour_ago = now - timedelta(hours=1)

        hourly_loss = sum(
            record.loss_amount
            for record in self._loss_history
            if record.timestamp >= one_hour_ago
        )

        if equity <= 0:
            return messages

        hourly_loss_pct = (hourly_loss / equity) * 100

        if hourly_loss_pct >= self.config.max_hourly_loss_pct:
            self._trigger(
                TriggerType.HOURLY_LOSS,
                f"Hourly loss limit: {hourly_loss_pct:.2f}% >= {self.config.max_hourly_loss_pct:.2f}%",
                now,
                {"loss_amount": hourly_loss, "loss_pct": hourly_loss_pct},
            )
            messages.append(f"TRIGGERED: Hourly loss {hourly_loss_pct:.2f}%")
        elif hourly_loss_pct >= self.config.hourly_loss_warning_pct:
            self._active_warnings[TriggerType.HOURLY_LOSS] = (
                f"Hourly loss at {hourly_loss_pct:.2f}%"
            )
            messages.append(f"WARNING: Hourly loss {hourly_loss_pct:.2f}%")

        return messages

    def _check_daily_loss(self, equity: float, now: datetime) -> List[str]:
        """Check loss in the last 24 hours."""
        messages = []
        one_day_ago = now - timedelta(hours=24)

        daily_loss = sum(
            record.loss_amount
            for record in self._loss_history
            if record.timestamp >= one_day_ago
        )

        if equity <= 0:
            return messages

        daily_loss_pct = (daily_loss / equity) * 100

        if daily_loss_pct >= self.config.max_daily_loss_pct:
            self._trigger(
                TriggerType.DAILY_LOSS,
                f"Daily loss limit: {daily_loss_pct:.2f}% >= {self.config.max_daily_loss_pct:.2f}%",
                now,
                {"loss_amount": daily_loss, "loss_pct": daily_loss_pct},
            )
            messages.append(f"TRIGGERED: Daily loss {daily_loss_pct:.2f}%")
        elif daily_loss_pct >= self.config.daily_loss_warning_pct:
            self._active_warnings[TriggerType.DAILY_LOSS] = (
                f"Daily loss at {daily_loss_pct:.2f}%"
            )
            messages.append(f"WARNING: Daily loss {daily_loss_pct:.2f}%")

        return messages

    def _check_rapid_drawdown(self, equity: float, now: datetime) -> List[str]:
        """Check for rapid drawdown (X% in Y minutes)."""
        messages = []
        cutoff = now - timedelta(minutes=self.config.rapid_dd_minutes)

        # Find peak equity in the lookback window
        recent_equities = [
            (ts, eq) for ts, eq in self._equity_snapshots
            if ts >= cutoff
        ]

        if not recent_equities:
            return messages

        peak_equity = max(eq for _, eq in recent_equities)
        if peak_equity <= 0:
            return messages

        drawdown_pct = (peak_equity - equity) / peak_equity * 100

        if drawdown_pct >= self.config.rapid_dd_pct:
            self._trigger(
                TriggerType.RAPID_DRAWDOWN,
                f"Rapid drawdown: {drawdown_pct:.2f}% in {self.config.rapid_dd_minutes} minutes",
                now,
                {"peak_equity": peak_equity, "current_equity": equity, "drawdown_pct": drawdown_pct},
            )
            messages.append(
                f"TRIGGERED: {drawdown_pct:.2f}% drawdown in "
                f"{self.config.rapid_dd_minutes} minutes"
            )
        elif drawdown_pct >= self.config.rapid_dd_warning_pct:
            self._active_warnings[TriggerType.RAPID_DRAWDOWN] = (
                f"Rapid drawdown at {drawdown_pct:.2f}%"
            )
            messages.append(f"WARNING: Rapid drawdown {drawdown_pct:.2f}%")

        return messages

    def _check_volatility(self, now: datetime) -> List[str]:
        """Check for abnormal volatility."""
        messages = []

        if len(self._atr_history) < self.config.volatility_lookback_periods:
            return messages

        if self._normal_atr <= 0:
            return messages

        # Get the most recent ATR
        current_atr = self._atr_history[-1]
        ratio = current_atr / self._normal_atr

        if ratio >= self.config.volatility_multiplier:
            self._trigger(
                TriggerType.ABNORMAL_VOLATILITY,
                f"Abnormal volatility: ATR {ratio:.1f}x normal",
                now,
                {"current_atr": current_atr, "normal_atr": self._normal_atr, "ratio": ratio},
            )
            messages.append(f"TRIGGERED: Volatility {ratio:.1f}x normal")
        elif ratio >= self.config.volatility_warning_multiplier:
            self._active_warnings[TriggerType.ABNORMAL_VOLATILITY] = (
                f"Volatility at {ratio:.1f}x normal"
            )
            messages.append(f"WARNING: Volatility {ratio:.1f}x normal")

        return messages

    def _check_connectivity(self, now: datetime) -> List[str]:
        """Check exchange connectivity."""
        messages = []

        if self._last_heartbeat is None:
            return messages

        elapsed = (now - self._last_heartbeat).total_seconds()

        if elapsed >= self.config.max_disconnect_seconds:
            self._trigger(
                TriggerType.CONNECTIVITY_LOSS,
                f"Exchange disconnected for {elapsed:.0f}s",
                now,
                {"elapsed_seconds": elapsed},
            )
            self._connected = False
            messages.append(f"TRIGGERED: Disconnected {elapsed:.0f}s")
        elif elapsed >= self.config.disconnect_warning_seconds:
            self._active_warnings[TriggerType.CONNECTIVITY_LOSS] = (
                f"No heartbeat for {elapsed:.0f}s"
            )
            messages.append(f"WARNING: No heartbeat for {elapsed:.0f}s")

        return messages

    def _check_stale_data(self, now: datetime) -> List[str]:
        """Check for stale price data."""
        messages = []

        if self._last_price_update is None:
            return messages

        elapsed = (now - self._last_price_update).total_seconds()

        if elapsed >= self.config.max_stale_seconds:
            self._trigger(
                TriggerType.STALE_DATA,
                f"Stale data: no price update for {elapsed:.0f}s",
                now,
                {"elapsed_seconds": elapsed},
            )
            messages.append(f"TRIGGERED: Stale data {elapsed:.0f}s")
        elif elapsed >= self.config.stale_warning_seconds:
            self._active_warnings[TriggerType.STALE_DATA] = (
                f"No price update for {elapsed:.0f}s"
            )
            messages.append(f"WARNING: No price update for {elapsed:.0f}s")

        return messages

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: CircuitBreakerState) -> None:
        """Transition to a new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state

        logger.info(f"Circuit breaker: {old_state.value} -> {new_state.value}")

        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as exc:
                logger.error(f"State callback error: {exc}")

    def _trigger(
        self,
        trigger_type: TriggerType,
        message: str,
        now: datetime,
        details: Optional[Dict] = None,
    ) -> None:
        """Trigger the circuit breaker."""
        if self._state == CircuitBreakerState.TRIGGERED:
            return  # Already triggered

        event = TriggerEvent(
            timestamp=now,
            trigger_type=trigger_type,
            state_before=self._state,
            state_after=CircuitBreakerState.TRIGGERED,
            message=message,
            details=details or {},
        )

        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        self._trigger_reason = message
        self._trigger_type = trigger_type
        self._triggered_at = now

        if self.config.auto_recovery and self.config.cooldown_minutes > 0:
            self._cooldown_end = now + timedelta(minutes=self.config.cooldown_minutes)

        self._set_state(CircuitBreakerState.TRIGGERED)

        logger.critical(
            f"CIRCUIT BREAKER TRIGGERED: {message} "
            f"[type={trigger_type.value}]"
        )

        # Notify callbacks
        for callback in self._trigger_callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.error(f"Trigger callback error: {exc}")

    def _check_cooldown_expiry(self, now: datetime) -> bool:
        """Check if cooldown has expired. Returns True if transitioned."""
        if self._cooldown_end is None:
            return False
        if not self.config.auto_recovery:
            return False

        if now >= self._cooldown_end:
            self._cooldown_end = None
            self._trigger_reason = ""
            self._trigger_type = None
            self._set_state(CircuitBreakerState.NORMAL)
            logger.info("Circuit breaker cooldown expired - returning to NORMAL")
            return True

        # Transition to COOLDOWN state after initial trigger
        if self._state == CircuitBreakerState.TRIGGERED:
            remaining = (self._cooldown_end - now).total_seconds()
            # Move to COOLDOWN after 10% of cooldown period
            total_cooldown = self.config.cooldown_minutes * 60
            if remaining < total_cooldown * 0.9:
                self._set_state(CircuitBreakerState.COOLDOWN)

        return False

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_trade_result(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        is_win: bool,
    ) -> None:
        """
        Record a trade result for consecutive loss and loss-rate tracking.
        """
        now = datetime.now(timezone.utc)

        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._loss_history.append(LossRecord(
                timestamp=now,
                loss_amount=abs(pnl),
                loss_pct=abs(pnl_pct),
                symbol=symbol,
            ))

    def record_heartbeat(self) -> None:
        """Record an exchange connectivity heartbeat."""
        self._last_heartbeat = datetime.now(timezone.utc)
        self._connected = True

    def record_price_update(self) -> None:
        """Record that a price update was received."""
        self._last_price_update = datetime.now(timezone.utc)

    def record_atr(self, atr: float) -> None:
        """
        Record an ATR value for volatility monitoring.

        Call this with each new candle's ATR.
        Normal ATR is calculated as the median of the lookback window.
        """
        self._atr_history.append(atr)

        if len(self._atr_history) >= self.config.volatility_lookback_periods:
            sorted_atrs = sorted(self._atr_history)
            mid = len(sorted_atrs) // 2
            self._normal_atr = sorted_atrs[mid]

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def activate_kill_switch(self) -> None:
        """Activate the manual kill switch. Blocks ALL trading."""
        self.config.kill_switch_active = True
        now = datetime.now(timezone.utc)
        self._trigger(TriggerType.MANUAL_KILL, "Kill switch activated manually", now)
        logger.critical("KILL SWITCH ACTIVATED - all trading stopped")

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch."""
        self.config.kill_switch_active = False
        logger.warning("Kill switch deactivated")

    def force_reset(self) -> None:
        """
        Force-reset the circuit breaker to NORMAL state.

        Use with extreme caution - this bypasses all safety checks.
        """
        old_state = self._state
        self._state = CircuitBreakerState.NORMAL
        self._trigger_reason = ""
        self._trigger_type = None
        self._cooldown_end = None
        self._consecutive_losses = 0
        self.config.kill_switch_active = False
        self._active_warnings.clear()

        logger.warning(
            f"Circuit breaker force-reset from {old_state.value} to NORMAL. "
            "All counters cleared."
        )

    def force_trigger(self, reason: str = "Manual trigger") -> None:
        """Manually trigger the circuit breaker."""
        now = datetime.now(timezone.utc)
        self._trigger(TriggerType.CUSTOM, reason, now)

    # ------------------------------------------------------------------
    # Status and reporting
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Get comprehensive circuit breaker status."""
        now = datetime.now(timezone.utc)

        # Calculate recent losses
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(hours=24)

        hourly_loss = sum(
            r.loss_amount for r in self._loss_history
            if r.timestamp >= one_hour_ago
        )
        daily_loss = sum(
            r.loss_amount for r in self._loss_history
            if r.timestamp >= one_day_ago
        )

        cooldown_remaining = None
        if self._cooldown_end:
            remaining = (self._cooldown_end - now).total_seconds()
            cooldown_remaining = max(0, remaining)

        return {
            "state": self._state.value,
            "is_trading_allowed": self.is_trading_allowed(),
            "kill_switch_active": self.config.kill_switch_active,
            "trigger_reason": self._trigger_reason,
            "trigger_type": self._trigger_type.value if self._trigger_type else None,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "cooldown_remaining_seconds": cooldown_remaining,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self.config.max_consecutive_losses,
            "hourly_loss": hourly_loss,
            "hourly_loss_limit": self.config.max_hourly_loss_pct,
            "daily_loss": daily_loss,
            "daily_loss_limit": self.config.max_daily_loss_pct,
            "connected": self._connected,
            "last_heartbeat": (
                self._last_heartbeat.isoformat() if self._last_heartbeat else None
            ),
            "last_price_update": (
                self._last_price_update.isoformat() if self._last_price_update else None
            ),
            "normal_atr": self._normal_atr,
            "current_atr": self._atr_history[-1] if self._atr_history else None,
            "active_warnings": {k.value: v for k, v in self._active_warnings.items()},
            "total_trigger_events": len(self._events),
        }

    def get_events(
        self,
        trigger_type: Optional[TriggerType] = None,
        limit: int = 100,
    ) -> List[TriggerEvent]:
        """Get trigger event history."""
        events = self._events
        if trigger_type:
            events = [e for e in events if e.trigger_type == trigger_type]
        return events[-limit:]

    def register_trigger_callback(
        self,
        callback: Callable[[TriggerEvent], None],
    ) -> None:
        """Register a callback for trigger events."""
        self._trigger_callbacks.append(callback)

    def register_state_callback(
        self,
        callback: Callable[[CircuitBreakerState, CircuitBreakerState], None],
    ) -> None:
        """Register a callback for state transitions."""
        self._state_callbacks.append(callback)

    def to_dict(self) -> Dict:
        """Serialize state for persistence."""
        return {
            "state": self._state.value,
            "trigger_reason": self._trigger_reason,
            "trigger_type": self._trigger_type.value if self._trigger_type else None,
            "consecutive_losses": self._consecutive_losses,
            "connected": self._connected,
            "normal_atr": self._normal_atr,
            "kill_switch_active": self.config.kill_switch_active,
            "cooldown_end": (
                self._cooldown_end.isoformat() if self._cooldown_end else None
            ),
            "triggered_at": (
                self._triggered_at.isoformat() if self._triggered_at else None
            ),
            "config": {
                "max_consecutive_losses": self.config.max_consecutive_losses,
                "max_hourly_loss_pct": self.config.max_hourly_loss_pct,
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "rapid_dd_pct": self.config.rapid_dd_pct,
                "rapid_dd_minutes": self.config.rapid_dd_minutes,
                "volatility_multiplier": self.config.volatility_multiplier,
                "cooldown_minutes": self.config.cooldown_minutes,
                "auto_recovery": self.config.auto_recovery,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CircuitBreaker":
        """Restore from serialized state."""
        config_data = data.get("config", {})
        config = CircuitBreakerConfig(
            max_consecutive_losses=config_data.get("max_consecutive_losses", 5),
            max_hourly_loss_pct=config_data.get("max_hourly_loss_pct", 2.0),
            max_daily_loss_pct=config_data.get("max_daily_loss_pct", 5.0),
            rapid_dd_pct=config_data.get("rapid_dd_pct", 3.0),
            rapid_dd_minutes=config_data.get("rapid_dd_minutes", 15),
            volatility_multiplier=config_data.get("volatility_multiplier", 3.0),
            cooldown_minutes=config_data.get("cooldown_minutes", 30),
            auto_recovery=config_data.get("auto_recovery", True),
            kill_switch_active=data.get("kill_switch_active", False),
        )

        cb = cls(config=config)
        cb._state = CircuitBreakerState(data.get("state", "normal"))
        cb._trigger_reason = data.get("trigger_reason", "")
        cb._consecutive_losses = data.get("consecutive_losses", 0)
        cb._connected = data.get("connected", True)
        cb._normal_atr = data.get("normal_atr", 0.0)

        trigger_type = data.get("trigger_type")
        if trigger_type:
            cb._trigger_type = TriggerType(trigger_type)

        cooldown_str = data.get("cooldown_end")
        if cooldown_str:
            cb._cooldown_end = datetime.fromisoformat(cooldown_str)

        triggered_str = data.get("triggered_at")
        if triggered_str:
            cb._triggered_at = datetime.fromisoformat(triggered_str)

        return cb
