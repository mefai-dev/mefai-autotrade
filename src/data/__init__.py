"""
Mefai Autotrade Data Layer
Production-grade data management: database, caching, indicators,
market data, time series, feature store, event sourcing, and configuration.
"""

from .database import (
    DatabaseManager,
    DatabaseConfig,
    MigrationManager,
    TradeRecord,
    OrderRecord,
    PositionRecord,
    EquitySnapshot,
    StrategyPerformance,
    RiskEvent,
    QueryFilter,
    DataExporter,
)
from .candle_store import (
    CandleStore,
    Candle,
    CandleBuffer,
    CandleAggregator,
    GapDetector,
    RealTimeCandleBuilder,
)
from .indicator_cache import (
    IndicatorCache,
    CachedIndicator,
    IndicatorDependencyGraph,
    IndicatorConfig,
)
from .market_data import (
    MarketDataManager,
    OrderBookSnapshot,
    TickData,
    TimeAndSales,
    DataQualityMonitor,
    LiquidityAnalyzer,
)
from .timeseries import (
    TimeSeriesResampler,
    TimeSeriesAligner,
    RollingWindow,
    ExponentialWeighted,
    CalendarUtils,
    TimeZoneHandler,
)
from .feature_store import (
    FeatureStore,
    FeatureSchema,
    FeatureVersion,
    FeatureSelector,
    FeatureCorrelation,
)
from .event_store import (
    EventStore,
    StoredEvent,
    EventQuery,
    EventCheckpoint,
    EventReplayer,
    EventCompressor,
)
from .config_manager import (
    ConfigManager,
    ConfigSchema,
    ConfigValidator,
    SecretManager,
    ConfigWatcher,
)

__version__ = "1.0.0"

__all__ = [
    # database
    "DatabaseManager",
    "DatabaseConfig",
    "MigrationManager",
    "TradeRecord",
    "OrderRecord",
    "PositionRecord",
    "EquitySnapshot",
    "StrategyPerformance",
    "RiskEvent",
    "QueryFilter",
    "DataExporter",
    # candle_store
    "CandleStore",
    "Candle",
    "CandleBuffer",
    "CandleAggregator",
    "GapDetector",
    "RealTimeCandleBuilder",
    # indicator_cache
    "IndicatorCache",
    "CachedIndicator",
    "IndicatorDependencyGraph",
    "IndicatorConfig",
    # market_data
    "MarketDataManager",
    "OrderBookSnapshot",
    "TickData",
    "TimeAndSales",
    "DataQualityMonitor",
    "LiquidityAnalyzer",
    # timeseries
    "TimeSeriesResampler",
    "TimeSeriesAligner",
    "RollingWindow",
    "ExponentialWeighted",
    "CalendarUtils",
    "TimeZoneHandler",
    # feature_store
    "FeatureStore",
    "FeatureSchema",
    "FeatureVersion",
    "FeatureSelector",
    "FeatureCorrelation",
    # event_store
    "EventStore",
    "StoredEvent",
    "EventQuery",
    "EventCheckpoint",
    "EventReplayer",
    "EventCompressor",
    # config_manager
    "ConfigManager",
    "ConfigSchema",
    "ConfigValidator",
    "SecretManager",
    "ConfigWatcher",
]
