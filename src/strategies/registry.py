"""
Mefai Signal Engine - Strategy Registry

Central registry for all trading strategies. Provides factory methods to
create strategy instances from configuration, list available strategies,
and validate parameters.
"""

from typing import Any, Dict, List, Optional, Type
import logging

from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Singleton registry that maps strategy names to their classes.
    Every strategy module registers itself on import.
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}
    _descriptions: Dict[str, str] = {}
    _default_params: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        strategy_class: Type[BaseStrategy],
        description: str = "",
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a strategy class under a unique name."""
        if name in cls._strategies:
            logger.warning("Overwriting strategy registration: %s", name)
        cls._strategies[name] = strategy_class
        cls._descriptions[name] = description or strategy_class.description
        cls._default_params[name] = default_params or {}
        logger.debug("Registered strategy: %s", name)

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """Retrieve a strategy class by name."""
        return cls._strategies.get(name)

    @classmethod
    def create(
        cls,
        name: str,
        symbol: str,
        timeframe: str = "5m",
        params: Optional[Dict[str, Any]] = None,
        backtest_mode: bool = False,
    ) -> BaseStrategy:
        """
        Factory - instantiate a strategy by name.

        Merges default params with user-supplied overrides.
        """
        strategy_cls = cls._strategies.get(name)
        if strategy_cls is None:
            available = ", ".join(sorted(cls._strategies.keys()))
            raise ValueError(
                f"Unknown strategy '{name}'. Available: {available}"
            )

        merged_params = dict(cls._default_params.get(name, {}))
        if params:
            merged_params.update(params)

        instance = strategy_cls(
            symbol=symbol,
            timeframe=timeframe,
            params=merged_params,
            backtest_mode=backtest_mode,
        )
        logger.info("Created strategy '%s' for %s/%s", name, symbol, timeframe)
        return instance

    @classmethod
    def list_strategies(cls) -> List[Dict[str, Any]]:
        """Return a list of all registered strategies with metadata."""
        result = []
        for name in sorted(cls._strategies.keys()):
            strat_cls = cls._strategies[name]
            result.append({
                "name": name,
                "class": strat_cls.__name__,
                "description": cls._descriptions.get(name, ""),
                "version": getattr(strat_cls, "version", "1.0.0"),
                "default_params": cls._default_params.get(name, {}),
            })
        return result

    @classmethod
    def validate_params(cls, name: str, params: Dict[str, Any]) -> List[str]:
        """
        Validate strategy params. Returns list of error messages (empty = valid).
        """
        errors: List[str] = []
        strategy_cls = cls._strategies.get(name)
        if strategy_cls is None:
            errors.append(f"Unknown strategy: {name}")
            return errors

        # Check for required params if the class defines them
        required = getattr(strategy_cls, "required_params", [])
        for key in required:
            if key not in params:
                errors.append(f"Missing required parameter: {key}")

        # Type checks for known params
        param_types = getattr(strategy_cls, "param_types", {})
        for key, expected_type in param_types.items():
            if key in params and not isinstance(params[key], expected_type):
                errors.append(
                    f"Parameter '{key}' must be {expected_type.__name__}, "
                    f"got {type(params[key]).__name__}"
                )

        return errors

    @classmethod
    def count(cls) -> int:
        return len(cls._strategies)

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._strategies.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations (for testing)."""
        cls._strategies.clear()
        cls._descriptions.clear()
        cls._default_params.clear()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def create_strategy(
    name: str,
    symbol: str,
    timeframe: str = "5m",
    params: Optional[Dict[str, Any]] = None,
    backtest_mode: bool = False,
) -> BaseStrategy:
    """Convenience wrapper around StrategyRegistry.create()."""
    return StrategyRegistry.create(name, symbol, timeframe, params, backtest_mode)


def list_strategies() -> List[Dict[str, Any]]:
    """Convenience wrapper around StrategyRegistry.list_strategies()."""
    return StrategyRegistry.list_strategies()


# ---------------------------------------------------------------------------
# Auto-register all built-in strategies on module import
# ---------------------------------------------------------------------------

def _auto_register() -> None:
    """Import all strategy modules to trigger their registration."""
    # Each strategy module calls StrategyRegistry.register() at module level
    try:
        from strategies.grid_trading import GridTradingStrategy
        StrategyRegistry.register(
            "grid_trading", GridTradingStrategy,
            "Grid Trading - profit from ranging markets with buy/sell grid levels",
            {"num_grids": 10, "grid_type": "arithmetic", "total_investment": 1000.0},
        )
    except ImportError:
        pass

    try:
        from strategies.dca import DCAStrategy
        StrategyRegistry.register(
            "dca", DCAStrategy,
            "Dollar Cost Averaging - systematic buying with intelligence layers",
            {"interval_hours": 24, "buy_amount": 100.0, "dip_threshold_pct": 5.0},
        )
    except ImportError:
        pass

    try:
        from strategies.momentum import MomentumStrategy
        StrategyRegistry.register(
            "momentum", MomentumStrategy,
            "Multi-timeframe momentum with RSI, MACD, ADX confirmation",
            {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "adx_threshold": 25},
        )
    except ImportError:
        pass

    try:
        from strategies.mean_reversion import MeanReversionStrategy
        StrategyRegistry.register(
            "mean_reversion", MeanReversionStrategy,
            "Statistical mean reversion using Bollinger Bands and z-score",
            {"bb_period": 20, "bb_std": 2.0, "zscore_entry": 2.0, "zscore_exit": 0.5},
        )
    except ImportError:
        pass

    try:
        from strategies.breakout import BreakoutStrategy
        StrategyRegistry.register(
            "breakout", BreakoutStrategy,
            "Breakout detection with volume confirmation and false breakout filter",
            {"lookback": 20, "volume_multiplier": 2.0, "atr_stop_multiplier": 2.0},
        )
    except ImportError:
        pass

    try:
        from strategies.scalping import ScalpingStrategy
        StrategyRegistry.register(
            "scalping", ScalpingStrategy,
            "High-frequency scalping with order book and micro-trend analysis",
            {"target_pct": 0.2, "max_hold_minutes": 15, "max_trades_per_hour": 10},
        )
    except ImportError:
        pass

    try:
        from strategies.trend_following import TrendFollowingStrategy
        StrategyRegistry.register(
            "trend_following", TrendFollowingStrategy,
            "Classic trend following with Supertrend, EMA ribbon, Ichimoku",
            {"supertrend_period": 10, "supertrend_mult": 3.0, "adx_threshold": 20},
        )
    except ImportError:
        pass

    try:
        from strategies.arbitrage import ArbitrageStrategy
        StrategyRegistry.register(
            "arbitrage", ArbitrageStrategy,
            "Cross-exchange and funding rate arbitrage",
            {"min_spread_pct": 0.1, "max_exposure": 10000.0},
        )
    except ImportError:
        pass

    try:
        from strategies.market_making import MarketMakingStrategy
        StrategyRegistry.register(
            "market_making", MarketMakingStrategy,
            "Market making with inventory-aware quoting and spread management",
            {"spread_bps": 10, "max_inventory": 100.0, "refresh_interval_sec": 5},
        )
    except ImportError:
        pass

    try:
        from strategies.smart_money import SmartMoneyStrategy
        StrategyRegistry.register(
            "smart_money", SmartMoneyStrategy,
            "Smart Money Concepts - order blocks, FVG, BOS, liquidity sweeps",
            {"htf_timeframe": "4h", "ltf_timeframe": "15m", "min_confluence": 3},
        )
    except ImportError:
        pass

    try:
        from strategies.pairs_trading import PairsTradingStrategy
        StrategyRegistry.register(
            "pairs_trading", PairsTradingStrategy,
            "Statistical pairs trading with cointegration and mean reversion",
            {"zscore_entry": 2.0, "zscore_exit": 0.5, "lookback": 60},
        )
    except ImportError:
        pass

    try:
        from strategies.volume_profile import VolumeProfileStrategy
        StrategyRegistry.register(
            "volume_profile", VolumeProfileStrategy,
            "Volume profile analysis with POC, value area, and VWAP",
            {"num_bins": 50, "value_area_pct": 70.0},
        )
    except ImportError:
        pass

    try:
        from strategies.elliott_wave import ElliottWaveStrategy
        StrategyRegistry.register(
            "elliott_wave", ElliottWaveStrategy,
            "Elliott Wave pattern recognition with Fibonacci projections",
            {"min_wave_pct": 1.0, "fib_tolerance": 0.05},
        )
    except ImportError:
        pass

    try:
        from strategies.order_flow import OrderFlowStrategy
        StrategyRegistry.register(
            "order_flow", OrderFlowStrategy,
            "Order flow analysis - CVD, delta divergence, absorption detection",
            {"cvd_lookback": 50, "delta_threshold": 0.3},
        )
    except ImportError:
        pass

    try:
        from strategies.machine_learning import MLStrategy
        StrategyRegistry.register(
            "machine_learning", MLStrategy,
            "ML-integrated strategy consuming Mefai Signal Engine predictions",
            {"confidence_threshold": 0.6, "model_type": "xgboost"},
        )
    except ImportError:
        pass

    try:
        from strategies.composite import CompositeStrategy
        StrategyRegistry.register(
            "composite", CompositeStrategy,
            "Multi-strategy composite with weighted voting and conflict resolution",
            {"voting_threshold": 0.6, "min_strategies": 2},
        )
    except ImportError:
        pass


_auto_register()
