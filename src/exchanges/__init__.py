# mefai-autotrade exchange connectors
# Multi-exchange support with unified interface

from src.exchanges.base import (
    ExchangeBase,
    ExchangeConfig,
    UnifiedTicker,
    UnifiedOrderbook,
    UnifiedKline,
    UnifiedOrder,
    UnifiedPosition,
    UnifiedBalance,
    UnifiedTrade,
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
    MarginMode,
    TimeInForce,
    ExchangeError,
    RateLimitError,
    InsufficientBalanceError,
    OrderNotFoundError,
    InvalidOrderError,
    AuthenticationError,
    NetworkError,
    RateLimiter,
)
from src.exchanges.binance_spot import BinanceSpotExchange
from src.exchanges.binance_futures import BinanceFuturesExchange
from src.exchanges.bybit import BybitExchange
from src.exchanges.okx import OKXExchange
from src.exchanges.connection_pool import (
    ExchangeConnectionPool,
    ConnectionHealth,
    PooledConnection,
)
from src.exchanges.data_normalizer import DataNormalizer
from src.exchanges.websocket_manager import WebSocketManager, WSConnectionState

__all__ = [
    # Base
    "ExchangeBase",
    "ExchangeConfig",
    # Unified models
    "UnifiedTicker",
    "UnifiedOrderbook",
    "UnifiedKline",
    "UnifiedOrder",
    "UnifiedPosition",
    "UnifiedBalance",
    "UnifiedTrade",
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "PositionSide",
    "MarginMode",
    "TimeInForce",
    # Errors
    "ExchangeError",
    "RateLimitError",
    "InsufficientBalanceError",
    "OrderNotFoundError",
    "InvalidOrderError",
    "AuthenticationError",
    "NetworkError",
    # Exchanges
    "BinanceSpotExchange",
    "BinanceFuturesExchange",
    "BybitExchange",
    "OKXExchange",
    # Infrastructure
    "ExchangeConnectionPool",
    "ConnectionHealth",
    "PooledConnection",
    "DataNormalizer",
    "WebSocketManager",
    "WSConnectionState",
    "RateLimiter",
]

__version__ = "1.0.0"
