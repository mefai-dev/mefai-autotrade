# Mefai Signal Engine - Backtesting Module
# Professional-grade historical simulation engine for autotrade strategy validation.
#
# Components:
#   - engine: Core event-driven backtesting engine
#   - data_loader: Historical data loading, validation, and resampling
#   - simulator: Simulated exchange with realistic order matching
#   - optimizer: Parameter optimization (grid, random, bayesian, genetic)
#   - monte_carlo: Monte Carlo analysis and confidence intervals
#   - report: Performance reporting and analytics
#   - walk_forward: Walk-forward analysis and out-of-sample validation
#   - benchmark: Benchmark comparison and alpha/beta calculation

from src.backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from src.backtesting.data_loader import DataLoader, DataSource, OHLCVBar
from src.backtesting.simulator import SimulatedExchange, SimulatedOrderBook
from src.backtesting.optimizer import (
    StrategyOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
)
from src.backtesting.monte_carlo import MonteCarloAnalyzer, MonteCarloResult
from src.backtesting.report import BacktestReport, PerformanceSummary
from src.backtesting.walk_forward import WalkForwardAnalyzer, WalkForwardResult
from src.backtesting.benchmark import (
    BenchmarkEngine,
    BuyAndHoldBenchmark,
    RandomEntryBenchmark,
)

__version__ = "1.0.0"
__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "DataLoader",
    "DataSource",
    "OHLCVBar",
    "SimulatedExchange",
    "SimulatedOrderBook",
    "StrategyOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "GeneticOptimizer",
    "MonteCarloAnalyzer",
    "MonteCarloResult",
    "BacktestReport",
    "PerformanceSummary",
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    "BenchmarkEngine",
    "BuyAndHoldBenchmark",
    "RandomEntryBenchmark",
]
