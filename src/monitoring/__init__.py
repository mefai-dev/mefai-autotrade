"""
Mefai Autotrade Monitoring and Alerts Module
Production-grade monitoring, dashboard, notifications, and observability.
"""

from .logger import (
    TradeLogger,
    OrderLogger,
    RiskLogger,
    SystemLogger,
    AuditLogger,
    CorrelationContext,
    setup_logging,
    get_trade_logger,
    get_order_logger,
    get_risk_logger,
    get_system_logger,
    get_audit_logger,
)
from .performance_tracker import (
    PerformanceTracker,
    RollingMetrics,
    EquityCurvePoint,
    TradeRecord,
    DrawdownInfo,
    StreakInfo,
    TimeAnalysis,
    MonthlyReturn,
)
from .health_checker import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    HealthReport,
)
from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertRule,
    AlertState,
    PnLThresholdRule,
    DrawdownThresholdRule,
    PositionSizeRule,
    PriceSpikeRule,
    ErrorRateRule,
    ResourceUsageRule,
)
from .notifier import (
    Notifier,
    TelegramNotifier,
    DiscordNotifier,
    EmailNotifier,
    NotificationPriority,
    NotificationMessage,
)
from .metrics_collector import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    MetricExporter,
)
from .dashboard_api import create_dashboard_app, DashboardState

__version__ = "1.0.0"

__all__ = [
    # Logger
    "TradeLogger",
    "OrderLogger",
    "RiskLogger",
    "SystemLogger",
    "AuditLogger",
    "CorrelationContext",
    "setup_logging",
    "get_trade_logger",
    "get_order_logger",
    "get_risk_logger",
    "get_system_logger",
    "get_audit_logger",
    # Performance
    "PerformanceTracker",
    "RollingMetrics",
    "EquityCurvePoint",
    "TradeRecord",
    "DrawdownInfo",
    "StreakInfo",
    "TimeAnalysis",
    "MonthlyReturn",
    # Health
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "HealthReport",
    # Alerting
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertRule",
    "AlertState",
    "PnLThresholdRule",
    "DrawdownThresholdRule",
    "PositionSizeRule",
    "PriceSpikeRule",
    "ErrorRateRule",
    "ResourceUsageRule",
    # Notifier
    "Notifier",
    "TelegramNotifier",
    "DiscordNotifier",
    "EmailNotifier",
    "NotificationPriority",
    "NotificationMessage",
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricExporter",
    # Dashboard
    "create_dashboard_app",
    "DashboardState",
]
