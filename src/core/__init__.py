"""
Mefai Autotrade Core Engine
Production-grade trading system core components.
"""

from .order_manager import (
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderManager,
    OrderValidationError,
    OrderNotFoundError,
)
from .position_tracker import (
    Position,
    PositionSide,
    PositionTracker,
    PositionError,
)
from .pnl_engine import (
    PnLEngine,
    PnLRecord,
    PnLSnapshot,
    PerformanceMetrics,
)
from .account_manager import (
    AccountManager,
    AccountBalance,
    MarginMode,
    AccountSnapshot,
)
from .event_bus import (
    EventBus,
    Event,
    EventType,
    EventPriority,
    EventHandler,
)
from .state_machine import (
    TradingState,
    StateMachine,
    StateTransitionError,
)
from .engine import TradingEngine

__version__ = "1.0.0"

__all__ = [
    "Order",
    "OrderStatus",
    "OrderType",
    "OrderSide",
    "TimeInForce",
    "OrderManager",
    "OrderValidationError",
    "OrderNotFoundError",
    "Position",
    "PositionSide",
    "PositionTracker",
    "PositionError",
    "PnLEngine",
    "PnLRecord",
    "PnLSnapshot",
    "PerformanceMetrics",
    "AccountManager",
    "AccountBalance",
    "MarginMode",
    "AccountSnapshot",
    "EventBus",
    "Event",
    "EventType",
    "EventPriority",
    "EventHandler",
    "TradingState",
    "StateMachine",
    "StateTransitionError",
    "TradingEngine",
]
