"""
Mefai Signal Engine - Trading Strategies Module

Production-grade trading strategies for automated execution.
Each strategy implements real entry/exit logic with proper indicator
calculations, position sizing, and risk management.
"""

from strategies.base import BaseStrategy, StrategySignal, SignalType, StrategyState
from strategies.registry import StrategyRegistry, create_strategy, list_strategies

from strategies.grid_trading import GridTradingStrategy
from strategies.dca import DCAStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from strategies.scalping import ScalpingStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.market_making import MarketMakingStrategy
from strategies.smart_money import SmartMoneyStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.volume_profile import VolumeProfileStrategy
from strategies.elliott_wave import ElliottWaveStrategy
from strategies.order_flow import OrderFlowStrategy
from strategies.machine_learning import MLStrategy
from strategies.composite import CompositeStrategy

__all__ = [
    # Base
    "BaseStrategy",
    "StrategySignal",
    "SignalType",
    "StrategyState",
    # Registry
    "StrategyRegistry",
    "create_strategy",
    "list_strategies",
    # Strategies
    "GridTradingStrategy",
    "DCAStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "ScalpingStrategy",
    "TrendFollowingStrategy",
    "ArbitrageStrategy",
    "MarketMakingStrategy",
    "SmartMoneyStrategy",
    "PairsTradingStrategy",
    "VolumeProfileStrategy",
    "ElliottWaveStrategy",
    "OrderFlowStrategy",
    "MLStrategy",
    "CompositeStrategy",
]

__version__ = "1.0.0"
