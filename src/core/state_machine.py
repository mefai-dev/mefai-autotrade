"""
State Machine - Trading system state management for Mefai Autotrade.
Manages system states, transition rules, state persistence,
health checks, and graceful shutdown procedures.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("mefai.autotrade.state_machine")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradingState(Enum):
    """All possible states for the trading system."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    MAINTENANCE = "MAINTENANCE"
    ERROR = "ERROR"
    STOPPED = "STOPPED"

    @property
    def is_operational(self) -> bool:
        return self in (TradingState.RUNNING, TradingState.PAUSED)

    @property
    def can_trade(self) -> bool:
        return self == TradingState.RUNNING

    @property
    def is_terminal(self) -> bool:
        return self in (TradingState.STOPPED, TradingState.SHUTTING_DOWN)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: TradingState
    to_state: TradingState
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reason: str = ""
    triggered_by: str = ""  # Component that triggered the transition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "triggered_by": self.triggered_by,
        }


@dataclass
class HealthCheck:
    """Result of a health check."""
    component: str = ""
    healthy: bool = True
    message: str = ""
    last_check: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    check_duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "healthy": self.healthy,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "check_duration_ms": self.check_duration_ms,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    def __init__(
        self,
        from_state: TradingState,
        to_state: TradingState,
        reason: str = "",
    ):
        self.from_state = from_state
        self.to_state = to_state
        msg = (
            f"Invalid transition: {from_state.value} -> {to_state.value}"
        )
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

# Define valid state transitions
VALID_TRANSITIONS: Dict[TradingState, Set[TradingState]] = {
    TradingState.INITIALIZING: {
        TradingState.RUNNING,
        TradingState.ERROR,
        TradingState.STOPPED,
    },
    TradingState.RUNNING: {
        TradingState.PAUSED,
        TradingState.EMERGENCY_STOP,
        TradingState.SHUTTING_DOWN,
        TradingState.MAINTENANCE,
        TradingState.ERROR,
    },
    TradingState.PAUSED: {
        TradingState.RUNNING,
        TradingState.EMERGENCY_STOP,
        TradingState.SHUTTING_DOWN,
        TradingState.MAINTENANCE,
    },
    TradingState.EMERGENCY_STOP: {
        TradingState.MAINTENANCE,  # Must go through maintenance to reset
        TradingState.SHUTTING_DOWN,
        TradingState.STOPPED,
    },
    TradingState.SHUTTING_DOWN: {
        TradingState.STOPPED,
    },
    TradingState.MAINTENANCE: {
        TradingState.INITIALIZING,  # Re-init after maintenance
        TradingState.RUNNING,       # Direct resume if safe
        TradingState.STOPPED,
    },
    TradingState.ERROR: {
        TradingState.MAINTENANCE,
        TradingState.INITIALIZING,
        TradingState.STOPPED,
        TradingState.SHUTTING_DOWN,
    },
    TradingState.STOPPED: {
        TradingState.INITIALIZING,  # Can restart
    },
}

# State transition callbacks type
StateCallback = Callable[
    [TradingState, TradingState, str], None
]


class StateMachine:
    """
    Trading system state machine with transition validation,
    persistence, health checks, and callback support.
    """

    def __init__(
        self,
        initial_state: TradingState = TradingState.INITIALIZING,
        persistence_path: Optional[str] = None,
        max_history: int = 1000,
    ):
        self._lock = threading.RLock()

        # Current state
        self._state = initial_state
        self._state_entered_at = datetime.now(timezone.utc)
        self._state_data: Dict[str, Any] = {}  # Arbitrary state metadata

        # Transition history
        self._history: List[StateTransition] = []
        self._max_history = max_history

        # Persistence
        self._persistence_path = persistence_path

        # Callbacks
        self._on_enter: Dict[TradingState, List[StateCallback]] = {
            s: [] for s in TradingState
        }
        self._on_exit: Dict[TradingState, List[StateCallback]] = {
            s: [] for s in TradingState
        }
        self._on_any_transition: List[StateCallback] = []

        # Health checks
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._last_health: Dict[str, HealthCheck] = {}

        # Uptime tracking
        self._started_at = datetime.now(timezone.utc)
        self._total_running_seconds: float = 0.0
        self._last_running_start: Optional[datetime] = None

        # Try to restore state from persistence
        if persistence_path:
            self._try_restore_state()

        logger.info(
            "StateMachine initialized (state=%s)", initial_state.value
        )

    # -- State properties -----------------------------------------------------

    @property
    def state(self) -> TradingState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state == TradingState.RUNNING

    @property
    def can_trade(self) -> bool:
        return self.state.can_trade

    @property
    def is_operational(self) -> bool:
        return self.state.is_operational

    @property
    def state_duration_seconds(self) -> float:
        with self._lock:
            return (
                datetime.now(timezone.utc) - self._state_entered_at
            ).total_seconds()

    @property
    def uptime_seconds(self) -> float:
        return (
            datetime.now(timezone.utc) - self._started_at
        ).total_seconds()

    @property
    def total_running_seconds(self) -> float:
        with self._lock:
            total = self._total_running_seconds
            if (
                self._state == TradingState.RUNNING
                and self._last_running_start
            ):
                total += (
                    datetime.now(timezone.utc) - self._last_running_start
                ).total_seconds()
            return total

    # -- State transitions ----------------------------------------------------

    def transition(
        self,
        to_state: TradingState,
        reason: str = "",
        triggered_by: str = "system",
        state_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Attempt a state transition. Returns True on success.
        Raises StateTransitionError if the transition is not valid.
        """
        with self._lock:
            from_state = self._state

            if from_state == to_state:
                logger.debug(
                    "Already in state %s, no transition needed",
                    to_state.value,
                )
                return True

            # Validate transition
            valid_targets = VALID_TRANSITIONS.get(from_state, set())
            if to_state not in valid_targets:
                raise StateTransitionError(
                    from_state, to_state,
                    f"Valid targets from {from_state.value}: "
                    f"{[s.value for s in valid_targets]}",
                )

            # Track running time
            if from_state == TradingState.RUNNING and self._last_running_start:
                self._total_running_seconds += (
                    datetime.now(timezone.utc) - self._last_running_start
                ).total_seconds()
                self._last_running_start = None

            # Execute exit callbacks
            for cb in self._on_exit.get(from_state, []):
                try:
                    cb(from_state, to_state, reason)
                except Exception:
                    logger.exception(
                        "Error in exit callback for %s", from_state.value
                    )

            # Perform transition
            self._state = to_state
            self._state_entered_at = datetime.now(timezone.utc)
            if state_data:
                self._state_data = state_data
            else:
                self._state_data = {}

            if to_state == TradingState.RUNNING:
                self._last_running_start = datetime.now(timezone.utc)

            # Record transition
            transition = StateTransition(
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                triggered_by=triggered_by,
            )
            self._history.append(transition)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Execute enter callbacks
            for cb in self._on_enter.get(to_state, []):
                try:
                    cb(from_state, to_state, reason)
                except Exception:
                    logger.exception(
                        "Error in enter callback for %s", to_state.value
                    )

            # Execute any-transition callbacks
            for cb in self._on_any_transition:
                try:
                    cb(from_state, to_state, reason)
                except Exception:
                    logger.exception("Error in transition callback")

            # Persist
            self._persist_state()

        logger.info(
            "State transition: %s -> %s (reason=%s, by=%s)",
            from_state.value, to_state.value, reason, triggered_by,
        )
        return True

    def can_transition_to(self, to_state: TradingState) -> bool:
        """Check if transition to a state is valid from current state."""
        with self._lock:
            valid = VALID_TRANSITIONS.get(self._state, set())
            return to_state in valid

    def get_valid_transitions(self) -> List[TradingState]:
        """Get list of states that can be transitioned to from current state."""
        with self._lock:
            return list(VALID_TRANSITIONS.get(self._state, set()))

    # -- Convenience transition methods ---------------------------------------

    def start(self, reason: str = "System start") -> bool:
        return self.transition(
            TradingState.RUNNING, reason, "system"
        )

    def pause(self, reason: str = "User paused") -> bool:
        return self.transition(
            TradingState.PAUSED, reason, "user"
        )

    def resume(self, reason: str = "User resumed") -> bool:
        return self.transition(
            TradingState.RUNNING, reason, "user"
        )

    def emergency_stop(self, reason: str = "Emergency") -> bool:
        return self.transition(
            TradingState.EMERGENCY_STOP, reason, "risk_manager"
        )

    def enter_maintenance(self, reason: str = "Maintenance") -> bool:
        return self.transition(
            TradingState.MAINTENANCE, reason, "system"
        )

    def shutdown(self, reason: str = "Shutdown requested") -> bool:
        return self.transition(
            TradingState.SHUTTING_DOWN, reason, "system"
        )

    def mark_stopped(self, reason: str = "Shutdown complete") -> bool:
        return self.transition(
            TradingState.STOPPED, reason, "system"
        )

    def mark_error(self, reason: str = "System error") -> bool:
        return self.transition(
            TradingState.ERROR, reason, "system"
        )

    def reset(self, reason: str = "System reset") -> bool:
        """Reset to INITIALIZING state (from MAINTENANCE or ERROR)."""
        return self.transition(
            TradingState.INITIALIZING, reason, "system"
        )

    # -- Callback registration ------------------------------------------------

    def on_enter_state(
        self, state: TradingState, callback: StateCallback
    ) -> None:
        with self._lock:
            self._on_enter[state].append(callback)

    def on_exit_state(
        self, state: TradingState, callback: StateCallback
    ) -> None:
        with self._lock:
            self._on_exit[state].append(callback)

    def on_any_transition(self, callback: StateCallback) -> None:
        with self._lock:
            self._on_any_transition.append(callback)

    # -- Health checks --------------------------------------------------------

    def register_health_check(
        self,
        component: str,
        check_fn: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check function for a component."""
        with self._lock:
            self._health_checks[component] = check_fn
        logger.debug("Health check registered: %s", component)

    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results: Dict[str, HealthCheck] = {}

        with self._lock:
            checks = dict(self._health_checks)

        for component, check_fn in checks.items():
            start = time.monotonic()
            try:
                result = check_fn()
                result.check_duration_ms = (
                    time.monotonic() - start
                ) * 1000
            except Exception as e:
                result = HealthCheck(
                    component=component,
                    healthy=False,
                    message=f"Check failed: {e}",
                    check_duration_ms=(
                        time.monotonic() - start
                    ) * 1000,
                )
            results[component] = result

        with self._lock:
            self._last_health = results

        # Check if system should enter error state
        unhealthy = [
            name for name, hc in results.items() if not hc.healthy
        ]
        if unhealthy and self._state == TradingState.RUNNING:
            logger.warning(
                "Unhealthy components: %s", ", ".join(unhealthy)
            )

        return results

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        with self._lock:
            if not self._last_health:
                return True  # No checks registered/run yet
            return all(hc.healthy for hc in self._last_health.values())

    def get_last_health_results(self) -> Dict[str, HealthCheck]:
        with self._lock:
            return dict(self._last_health)

    # -- Persistence ----------------------------------------------------------

    def _persist_state(self) -> None:
        """Save current state to file for recovery after restart."""
        if not self._persistence_path:
            return

        try:
            data = {
                "state": self._state.value,
                "state_entered_at": self._state_entered_at.isoformat(),
                "state_data": self._state_data,
                "total_running_seconds": self._total_running_seconds,
                "history": [t.to_dict() for t in self._history[-50:]],
            }
            # Write atomically via temp file
            tmp_path = self._persistence_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._persistence_path)
        except Exception:
            logger.exception("Failed to persist state")

    def _try_restore_state(self) -> None:
        """Try to restore state from persistence file."""
        if not self._persistence_path or not os.path.exists(
            self._persistence_path
        ):
            return

        try:
            with open(self._persistence_path, "r") as f:
                data = json.load(f)

            saved_state = TradingState(data.get("state", "INITIALIZING"))

            # Do not restore to RUNNING - always restart through
            # INITIALIZING for safety
            if saved_state in (
                TradingState.RUNNING,
                TradingState.SHUTTING_DOWN,
            ):
                logger.info(
                    "Previous state was %s, starting in INITIALIZING",
                    saved_state.value,
                )
                self._state = TradingState.INITIALIZING
            elif saved_state == TradingState.EMERGENCY_STOP:
                logger.warning(
                    "Previous state was EMERGENCY_STOP, "
                    "starting in MAINTENANCE"
                )
                self._state = TradingState.MAINTENANCE
            else:
                self._state = saved_state

            self._total_running_seconds = data.get(
                "total_running_seconds", 0.0
            )
            self._state_data = data.get("state_data", {})

            logger.info(
                "State restored: %s (was %s)",
                self._state.value, saved_state.value,
            )
        except Exception:
            logger.exception("Failed to restore state, using default")

    # -- Query methods --------------------------------------------------------

    def get_history(
        self, limit: int = 50
    ) -> List[StateTransition]:
        with self._lock:
            return list(self._history[-limit:])

    def get_state_data(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state_data)

    def set_state_data(self, key: str, value: Any) -> None:
        with self._lock:
            self._state_data[key] = value

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "can_trade": self._state.can_trade,
                "is_operational": self._state.is_operational,
                "state_duration_seconds": self.state_duration_seconds,
                "uptime_seconds": self.uptime_seconds,
                "total_running_seconds": self.total_running_seconds,
                "transitions_count": len(self._history),
                "valid_transitions": [
                    s.value for s in self.get_valid_transitions()
                ],
                "health_checks": {
                    name: hc.to_dict()
                    for name, hc in self._last_health.items()
                },
                "is_healthy": self.is_healthy(),
                "state_data": self._state_data,
            }
