"""
Mefai Autotrade Execution Engine
Production-grade smart order execution algorithms.

Provides TWAP, VWAP, Iceberg, Sniper execution strategies,
smart order routing, market impact estimation, and execution
quality analysis.
"""

from .smart_router import (
    SmartRouter,
    Venue,
    VenueStatus,
    RoutingRule,
    RoutingDecision,
    RouteType,
    VenueMetrics,
)
from .twap import (
    TWAPExecutor,
    TWAPConfig,
    TWAPSlice,
    TWAPProgress,
    TWAPStatus,
)
from .vwap import (
    VWAPExecutor,
    VWAPConfig,
    VWAPSlice,
    VWAPProgress,
    VWAPStatus,
    VolumeProfile,
)
from .iceberg import (
    IcebergExecutor,
    IcebergConfig,
    IcebergProgress,
    IcebergStatus,
)
from .sniper import (
    SniperExecutor,
    SniperConfig,
    SniperTarget,
    SniperStatus,
)
from .market_impact import (
    MarketImpactEstimator,
    ImpactModel,
    ImpactEstimate,
    ExecutionCostAnalysis,
    AlmgrenChrissParams,
)
from .order_builder import (
    OrderBuilder,
    BracketOrder,
    OCOOrder,
    ScaledOrder,
    GridOrderSet,
    HedgeOrder,
    ExchangeRules,
)
from .fill_analyzer import (
    FillAnalyzer,
    FillRecord,
    SlippageReport,
    ExecutionQualityReport,
    BestExecutionReport,
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionAlgorithm,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)

__version__ = "1.0.0"

__all__ = [
    # Smart Router
    "SmartRouter",
    "Venue",
    "VenueStatus",
    "RoutingRule",
    "RoutingDecision",
    "RouteType",
    "VenueMetrics",
    # TWAP
    "TWAPExecutor",
    "TWAPConfig",
    "TWAPSlice",
    "TWAPProgress",
    "TWAPStatus",
    # VWAP
    "VWAPExecutor",
    "VWAPConfig",
    "VWAPSlice",
    "VWAPProgress",
    "VWAPStatus",
    "VolumeProfile",
    # Iceberg
    "IcebergExecutor",
    "IcebergConfig",
    "IcebergProgress",
    "IcebergStatus",
    # Sniper
    "SniperExecutor",
    "SniperConfig",
    "SniperTarget",
    "SniperStatus",
    # Market Impact
    "MarketImpactEstimator",
    "ImpactModel",
    "ImpactEstimate",
    "ExecutionCostAnalysis",
    "AlmgrenChrissParams",
    # Order Builder
    "OrderBuilder",
    "BracketOrder",
    "OCOOrder",
    "ScaledOrder",
    "GridOrderSet",
    "HedgeOrder",
    "ExchangeRules",
    # Fill Analyzer
    "FillAnalyzer",
    "FillRecord",
    "SlippageReport",
    "ExecutionQualityReport",
    "BestExecutionReport",
    # Execution Engine
    "ExecutionEngine",
    "ExecutionAlgorithm",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
]
