"""
Mefai Autotrade - Risk Management Module

Production-grade risk management for automated trading systems.
All components work together through the RiskEngine orchestrator.

Modules:
- PositionSizer: Multi-algorithm position sizing (Kelly, ATR, anti-martingale, etc.)
- DrawdownManager: Real-time drawdown tracking with daily/weekly/monthly limits
- ExposureManager: Portfolio exposure limits (gross, net, per-symbol, per-sector)
- CircuitBreaker: Emergency stop system with multiple trigger conditions
- CorrelationMonitor: Cross-position correlation analysis and regime detection
- RiskMetrics: Comprehensive risk metrics (VaR, Sharpe, Sortino, CVaR, etc.)
- StopLossManager: Advanced stop loss management (trailing, ATR, Chandelier, SAR)
- PortfolioHeat: Portfolio heat tracking and risk allocation
- RiskEngine: Central orchestrator coordinating all modules
"""

from .position_sizer import (
    PositionSizer,
    SizingConfig,
    SizingMethod,
    SizingResult,
    PortfolioSnapshot,
    PositionInfo,
    TradeRecord,
)
from .drawdown_manager import (
    DrawdownManager,
    DrawdownConfig,
    DrawdownState,
    DrawdownPeriod,
    DrawdownSnapshot,
    DrawdownAlert,
)
from .exposure_manager import (
    ExposureManager,
    ExposureConfig,
    ExposureCheckResult,
    ExposureCheckResponse,
    ExposureSnapshot,
    MarginMode,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    TriggerType,
    TriggerEvent,
)
from .correlation_monitor import (
    CorrelationMonitor,
    CorrelationConfig,
    CorrelationMethod,
    CorrelationRegime,
    CorrelationSnapshot,
    CorrelationAlert,
)
from .risk_metrics import (
    RiskMetrics,
    RiskMetricsConfig,
    VaRMethod,
    VaRResult,
    RiskReport,
    DrawdownInfo,
)
from .stop_loss_manager import (
    StopLossManager,
    StopConfig,
    StopType,
    StopAction,
    StopLevel,
    StopUpdate,
)
from .portfolio_heat import (
    PortfolioHeat,
    HeatConfig,
    HeatLevel,
    HeatSnapshot,
    PositionHeat,
)
from .risk_engine import (
    RiskEngine,
    RiskEngineConfig,
    RiskDecision,
    RiskCheckResult,
)

__version__ = "1.0.0"

__all__ = [
    # Position Sizer
    "PositionSizer",
    "SizingConfig",
    "SizingMethod",
    "SizingResult",
    "PortfolioSnapshot",
    "PositionInfo",
    "TradeRecord",
    # Drawdown Manager
    "DrawdownManager",
    "DrawdownConfig",
    "DrawdownState",
    "DrawdownPeriod",
    "DrawdownSnapshot",
    "DrawdownAlert",
    # Exposure Manager
    "ExposureManager",
    "ExposureConfig",
    "ExposureCheckResult",
    "ExposureCheckResponse",
    "ExposureSnapshot",
    "MarginMode",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "TriggerType",
    "TriggerEvent",
    # Correlation Monitor
    "CorrelationMonitor",
    "CorrelationConfig",
    "CorrelationMethod",
    "CorrelationRegime",
    "CorrelationSnapshot",
    "CorrelationAlert",
    # Risk Metrics
    "RiskMetrics",
    "RiskMetricsConfig",
    "VaRMethod",
    "VaRResult",
    "RiskReport",
    "DrawdownInfo",
    # Stop Loss Manager
    "StopLossManager",
    "StopConfig",
    "StopType",
    "StopAction",
    "StopLevel",
    "StopUpdate",
    # Portfolio Heat
    "PortfolioHeat",
    "HeatConfig",
    "HeatLevel",
    "HeatSnapshot",
    "PositionHeat",
    # Risk Engine
    "RiskEngine",
    "RiskEngineConfig",
    "RiskDecision",
    "RiskCheckResult",
]
