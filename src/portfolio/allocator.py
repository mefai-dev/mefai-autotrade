"""
Mefai Signal Engine - Portfolio Allocator

Capital allocation across multiple trading strategies. Supports equal weight,
risk parity, mean-variance optimization (Markowitz), Kelly criterion, maximum
diversification, minimum variance, risk budgeting, and dynamic reallocation.

All allocation methods enforce configurable constraints (min/max per strategy)
and support time-based and drift-based rebalancing triggers.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger("mefai.autotrade.portfolio.allocator")


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class AllocationMethod(str, Enum):
    """Supported capital allocation methods."""
    EQUAL = "equal"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    KELLY = "kelly"
    MAX_DIVERSIFICATION = "max_diversification"
    MIN_VARIANCE = "min_variance"
    RISK_BUDGETING = "risk_budgeting"
    DYNAMIC = "dynamic"


class RebalanceTriggerType(str, Enum):
    """What triggers a portfolio rebalance."""
    TIME_BASED = "time_based"
    DRIFT_BASED = "drift_based"
    COMBINED = "combined"


@dataclass
class AllocationConstraint:
    """Per-strategy allocation bounds."""
    strategy_name: str
    min_weight: float = 0.0     # minimum allocation (0.0 = no minimum)
    max_weight: float = 1.0     # maximum allocation (1.0 = up to 100%)
    locked: bool = False        # if True, weight cannot change during rebalance

    def validate(self) -> List[str]:
        errors = []
        if self.min_weight < 0:
            errors.append(f"{self.strategy_name}: min_weight cannot be negative")
        if self.max_weight > 1.0:
            errors.append(f"{self.strategy_name}: max_weight cannot exceed 1.0")
        if self.min_weight > self.max_weight:
            errors.append(f"{self.strategy_name}: min_weight > max_weight")
        return errors


@dataclass
class StrategyPerformance:
    """Historical return and risk stats for a single strategy."""
    name: str
    returns: np.ndarray         # array of periodic returns
    volatility: float = 0.0     # annualized volatility
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    current_pnl: float = 0.0
    current_weight: float = 0.0 # actual current allocation


@dataclass
class AllocationResult:
    """Output of an allocation calculation."""
    method: AllocationMethod
    weights: Dict[str, float]           # strategy_name -> target weight
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_sharpe: float = 0.0
    diversification_ratio: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        total = sum(self.weights.values())
        return abs(total - 1.0) < 1e-6

    def get_trades(
        self,
        current_weights: Dict[str, float],
        total_capital: float,
    ) -> Dict[str, float]:
        """
        Calculate the dollar amount to trade for each strategy to reach
        the target allocation.
        Returns: strategy_name -> dollar_delta (positive = add, negative = reduce)
        """
        trades = {}
        for name, target_w in self.weights.items():
            current_w = current_weights.get(name, 0.0)
            delta = (target_w - current_w) * total_capital
            if abs(delta) > 0.01:  # skip dust
                trades[name] = round(delta, 2)
        return trades


@dataclass
class RebalanceTrigger:
    """Configuration for when to trigger a rebalance."""
    trigger_type: RebalanceTriggerType = RebalanceTriggerType.COMBINED
    interval_seconds: int = 86400       # default: daily
    drift_threshold_pct: float = 5.0    # rebalance if any weight drifts > 5%
    min_interval_seconds: int = 3600    # minimum time between rebalances
    last_rebalance_time: float = 0.0

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Check if rebalance should be triggered. Returns (should, reason)."""
        now = time.time()
        elapsed = now - self.last_rebalance_time

        # Minimum interval guard
        if elapsed < self.min_interval_seconds:
            return False, "min_interval_not_reached"

        if self.trigger_type == RebalanceTriggerType.TIME_BASED:
            if elapsed >= self.interval_seconds:
                return True, "time_based"
            return False, "time_not_reached"

        if self.trigger_type == RebalanceTriggerType.DRIFT_BASED:
            drift = self._max_drift(current_weights, target_weights)
            if drift >= self.drift_threshold_pct:
                return True, f"drift_{drift:.1f}pct"
            return False, f"drift_{drift:.1f}pct_below_threshold"

        # Combined: either trigger fires
        if elapsed >= self.interval_seconds:
            return True, "time_based"
        drift = self._max_drift(current_weights, target_weights)
        if drift >= self.drift_threshold_pct:
            return True, f"drift_{drift:.1f}pct"
        return False, "no_trigger"

    def _max_drift(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
    ) -> float:
        """Maximum absolute drift in percentage points across all strategies."""
        max_d = 0.0
        all_keys = set(current.keys()) | set(target.keys())
        for key in all_keys:
            c = current.get(key, 0.0)
            t = target.get(key, 0.0)
            drift = abs(c - t) * 100.0
            if drift > max_d:
                max_d = drift
        return max_d


# ---------------------------------------------------------------------------
# Portfolio Allocator
# ---------------------------------------------------------------------------

class PortfolioAllocator:
    """
    Central allocation engine. Given strategy performance data and constraints,
    computes optimal capital allocation using the selected method.
    """

    def __init__(
        self,
        method: AllocationMethod = AllocationMethod.EQUAL,
        constraints: Optional[List[AllocationConstraint]] = None,
        risk_free_rate: float = 0.04,
        rebalance_trigger: Optional[RebalanceTrigger] = None,
        total_capital: float = 10000.0,
    ):
        self.method = method
        self.constraints = {c.strategy_name: c for c in (constraints or [])}
        self.risk_free_rate = risk_free_rate
        self.trigger = rebalance_trigger or RebalanceTrigger()
        self.total_capital = total_capital

        # Allocation history for tracking
        self._history: List[AllocationResult] = []
        self._current_weights: Dict[str, float] = {}

        # Dynamic allocation state
        self._performance_window: int = 30  # lookback periods for dynamic
        self._momentum_weight: float = 0.3
        self._mean_reversion_weight: float = 0.2

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def allocate(
        self,
        strategies: List[StrategyPerformance],
        method_override: Optional[AllocationMethod] = None,
    ) -> AllocationResult:
        """
        Compute target allocation for the given strategies.
        """
        method = method_override or self.method

        if not strategies:
            return AllocationResult(method=method, weights={})

        dispatch = {
            AllocationMethod.EQUAL: self._equal_allocation,
            AllocationMethod.RISK_PARITY: self._risk_parity,
            AllocationMethod.MEAN_VARIANCE: self._mean_variance,
            AllocationMethod.KELLY: self._kelly_criterion,
            AllocationMethod.MAX_DIVERSIFICATION: self._max_diversification,
            AllocationMethod.MIN_VARIANCE: self._min_variance,
            AllocationMethod.RISK_BUDGETING: self._risk_budgeting,
            AllocationMethod.DYNAMIC: self._dynamic_allocation,
        }

        allocator_fn = dispatch.get(method)
        if allocator_fn is None:
            logger.error("Unknown allocation method: %s, falling back to equal", method)
            allocator_fn = self._equal_allocation

        result = allocator_fn(strategies)
        result = self._apply_constraints(result, strategies)

        self._history.append(result)
        self._current_weights = dict(result.weights)

        logger.info(
            "Allocation computed [%s]: %s strategies, expected_sharpe=%.3f",
            method.value, len(result.weights), result.expected_sharpe,
        )
        return result

    def should_rebalance(self) -> Tuple[bool, str]:
        """Check if portfolio needs rebalancing based on current trigger config."""
        if not self._history:
            return True, "no_previous_allocation"
        target = self._history[-1].weights
        return self.trigger.should_rebalance(self._current_weights, target)

    def update_weights(self, current_weights: Dict[str, float]) -> None:
        """Update actual portfolio weights (called after trades settle)."""
        self._current_weights = dict(current_weights)

    def get_allocation_history(self, n: int = 10) -> List[AllocationResult]:
        """Return the last n allocation results."""
        return self._history[-n:]

    def get_current_weights(self) -> Dict[str, float]:
        return dict(self._current_weights)

    # ------------------------------------------------------------------
    # Allocation methods
    # ------------------------------------------------------------------

    def _equal_allocation(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """Equal weight across all strategies."""
        n = len(strategies)
        weight = 1.0 / n if n > 0 else 0.0
        weights = {s.name: weight for s in strategies}

        # Portfolio metrics
        returns_matrix = self._build_returns_matrix(strategies)
        exp_ret, exp_vol = self._portfolio_metrics(
            np.array([weight] * n), returns_matrix,
        )

        return AllocationResult(
            method=AllocationMethod.EQUAL,
            weights=weights,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(
                np.array([weight] * n), strategies,
            ),
        )

    def _risk_parity(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Risk parity - allocate inversely proportional to volatility so each
        strategy contributes equal risk to the portfolio.
        """
        n = len(strategies)
        vols = np.array([max(s.volatility, 1e-8) for s in strategies])

        # Inverse volatility weights
        inv_vol = 1.0 / vols
        raw_weights = inv_vol / inv_vol.sum()

        # Iterative risk parity refinement
        returns_matrix = self._build_returns_matrix(strategies)
        cov = np.cov(returns_matrix, rowvar=True) if returns_matrix.shape[1] > 1 else np.diag(vols ** 2)

        # Target: each strategy contributes equal risk
        target_risk = 1.0 / n
        weights = raw_weights.copy()

        for iteration in range(50):
            port_vol = np.sqrt(weights @ cov @ weights)
            if port_vol < 1e-10:
                break
            marginal_risk = cov @ weights
            risk_contrib = weights * marginal_risk / port_vol
            total_risk = risk_contrib.sum()
            if total_risk < 1e-10:
                break
            risk_contrib_pct = risk_contrib / total_risk

            # Adjust weights to equalize risk contribution
            adjustment = target_risk - risk_contrib_pct
            weights = weights * (1.0 + 0.5 * adjustment)
            weights = np.maximum(weights, 1e-6)
            weights = weights / weights.sum()

        weight_dict = {s.name: float(w) for s, w in zip(strategies, weights)}
        exp_ret, exp_vol = self._portfolio_metrics(weights, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.RISK_PARITY,
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(weights, strategies),
            metadata={"iterations": 50},
        )

    def _mean_variance(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Markowitz mean-variance optimization. Maximizes the Sharpe ratio
        subject to allocation constraints.
        """
        n = len(strategies)
        returns_matrix = self._build_returns_matrix(strategies)
        mean_returns = np.array([np.mean(s.returns) for s in strategies])
        cov = np.cov(returns_matrix, rowvar=True) if returns_matrix.shape[1] > 1 else np.diag(
            [max(s.volatility, 1e-8) ** 2 for s in strategies]
        )

        # Ensure covariance matrix is positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        if np.min(eigvals) < 0:
            cov += np.eye(n) * (abs(np.min(eigvals)) + 1e-8)

        def neg_sharpe(w):
            port_ret = w @ mean_returns
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(port_ret - self.risk_free_rate / 252) / port_vol

        # Constraints: weights sum to 1
        constraints_scipy = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Bounds from strategy constraints
        bounds = []
        for s in strategies:
            c = self.constraints.get(s.name)
            if c:
                bounds.append((c.min_weight, c.max_weight))
            else:
                bounds.append((0.0, 1.0))

        # Initial guess: equal weight
        w0 = np.array([1.0 / n] * n)

        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_scipy,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if result.success:
            weights = result.x
        else:
            logger.warning("Mean-variance optimization failed: %s. Using equal weight.", result.message)
            weights = w0

        # Normalize to handle numerical imprecision
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        weight_dict = {s.name: float(w) for s, w in zip(strategies, weights)}
        exp_ret, exp_vol = self._portfolio_metrics(weights, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.MEAN_VARIANCE,
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(weights, strategies),
            metadata={"optimizer_success": result.success, "optimizer_message": result.message},
        )

    def _kelly_criterion(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Kelly criterion allocation. Computes the theoretically optimal fraction
        of capital for each strategy, then applies a fractional Kelly (default 0.5)
        for safety.

        Full Kelly: f = (p * b - q) / b
        where p = win_rate, q = 1-p, b = avg_win/avg_loss
        """
        kelly_fraction = 0.5  # half-Kelly for safety
        raw_weights = {}

        for s in strategies:
            if s.win_rate <= 0 or s.win_rate >= 1:
                raw_weights[s.name] = 0.0
                continue

            wins = s.returns[s.returns > 0]
            losses = s.returns[s.returns <= 0]
            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 1.0

            if avg_loss < 1e-8:
                raw_weights[s.name] = 0.0
                continue

            b = avg_win / avg_loss
            p = s.win_rate
            q = 1.0 - p
            kelly = (p * b - q) / b
            kelly = max(kelly, 0.0) * kelly_fraction
            raw_weights[s.name] = kelly

        # Normalize
        total = sum(raw_weights.values())
        if total > 0:
            weights = {k: v / total for k, v in raw_weights.items()}
        else:
            n = len(strategies)
            weights = {s.name: 1.0 / n for s in strategies}

        returns_matrix = self._build_returns_matrix(strategies)
        w_arr = np.array([weights[s.name] for s in strategies])
        exp_ret, exp_vol = self._portfolio_metrics(w_arr, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.KELLY,
            weights=weights,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(w_arr, strategies),
            metadata={"kelly_fraction": kelly_fraction, "raw_kelly": raw_weights},
        )

    def _max_diversification(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Maximum diversification portfolio. Maximizes the diversification ratio:
        DR = (weighted sum of individual volatilities) / portfolio volatility
        """
        n = len(strategies)
        vols = np.array([max(s.volatility, 1e-8) for s in strategies])
        returns_matrix = self._build_returns_matrix(strategies)
        cov = np.cov(returns_matrix, rowvar=True) if returns_matrix.shape[1] > 1 else np.diag(vols ** 2)

        def neg_div_ratio(w):
            weighted_vol_sum = w @ vols
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(weighted_vol_sum / port_vol)

        constraints_scipy = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = []
        for s in strategies:
            c = self.constraints.get(s.name)
            if c:
                bounds.append((c.min_weight, c.max_weight))
            else:
                bounds.append((0.0, 1.0))

        w0 = np.array([1.0 / n] * n)
        result = minimize(
            neg_div_ratio, w0, method="SLSQP",
            bounds=bounds, constraints=constraints_scipy,
            options={"maxiter": 500},
        )

        weights = result.x if result.success else w0
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        weight_dict = {s.name: float(w) for s, w in zip(strategies, weights)}
        exp_ret, exp_vol = self._portfolio_metrics(weights, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.MAX_DIVERSIFICATION,
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=float(-neg_div_ratio(weights)),
            metadata={"optimizer_success": result.success},
        )

    def _min_variance(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Minimum variance portfolio. Minimizes total portfolio variance
        without regard to expected returns.
        """
        n = len(strategies)
        vols = np.array([max(s.volatility, 1e-8) for s in strategies])
        returns_matrix = self._build_returns_matrix(strategies)
        cov = np.cov(returns_matrix, rowvar=True) if returns_matrix.shape[1] > 1 else np.diag(vols ** 2)

        def portfolio_variance(w):
            return w @ cov @ w

        constraints_scipy = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = []
        for s in strategies:
            c = self.constraints.get(s.name)
            if c:
                bounds.append((c.min_weight, c.max_weight))
            else:
                bounds.append((0.0, 1.0))

        w0 = np.array([1.0 / n] * n)
        result = minimize(
            portfolio_variance, w0, method="SLSQP",
            bounds=bounds, constraints=constraints_scipy,
            options={"maxiter": 500},
        )

        weights = result.x if result.success else w0
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        weight_dict = {s.name: float(w) for s, w in zip(strategies, weights)}
        exp_ret, exp_vol = self._portfolio_metrics(weights, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.MIN_VARIANCE,
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(weights, strategies),
            metadata={"portfolio_variance": float(portfolio_variance(weights))},
        )

    def _risk_budgeting(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Risk budgeting - assign a specific risk budget to each strategy.
        The optimizer finds weights such that each strategy's risk contribution
        matches its target budget.

        Risk budgets are read from constraints metadata or default to equal.
        """
        n = len(strategies)
        vols = np.array([max(s.volatility, 1e-8) for s in strategies])
        returns_matrix = self._build_returns_matrix(strategies)
        cov = np.cov(returns_matrix, rowvar=True) if returns_matrix.shape[1] > 1 else np.diag(vols ** 2)

        # Risk budgets (from constraints or equal)
        budgets = np.zeros(n)
        for i, s in enumerate(strategies):
            c = self.constraints.get(s.name)
            if c and hasattr(c, "metadata") and isinstance(getattr(c, "metadata", None), dict):
                budgets[i] = c.metadata.get("risk_budget", 1.0 / n)
            else:
                budgets[i] = 1.0 / n
        budgets = budgets / budgets.sum()

        def risk_budget_objective(w):
            """Minimize the sum of squared differences between actual and target risk contribution."""
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            marginal = cov @ w
            risk_contrib = w * marginal / port_vol
            total_risk = risk_contrib.sum()
            if total_risk < 1e-10:
                return 0.0
            actual_budgets = risk_contrib / total_risk
            return float(np.sum((actual_budgets - budgets) ** 2))

        constraints_scipy = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1.0)] * n

        w0 = budgets.copy()
        result = minimize(
            risk_budget_objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints_scipy,
            options={"maxiter": 500},
        )

        weights = result.x if result.success else np.array([1.0 / n] * n)
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        weight_dict = {s.name: float(w) for s, w in zip(strategies, weights)}
        exp_ret, exp_vol = self._portfolio_metrics(weights, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.RISK_BUDGETING,
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(weights, strategies),
            metadata={"target_budgets": dict(zip([s.name for s in strategies], budgets.tolist()))},
        )

    def _dynamic_allocation(
        self,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """
        Dynamic allocation that adjusts weights based on recent performance.
        Blends momentum (reward recent winners) with mean-reversion (expect
        underperformers to recover) and volatility scaling.

        Components:
        1. Momentum score: recent cumulative return
        2. Mean-reversion score: inverse of recent return (oversold bounce)
        3. Volatility scaling: inversely proportional to recent volatility
        4. Sharpe-weighted: reward strategies with high risk-adjusted returns
        """
        n = len(strategies)
        lookback = min(self._performance_window, min(len(s.returns) for s in strategies))
        if lookback < 2:
            return self._equal_allocation(strategies)

        momentum_scores = np.zeros(n)
        reversion_scores = np.zeros(n)
        vol_scores = np.zeros(n)
        sharpe_scores = np.zeros(n)

        for i, s in enumerate(strategies):
            recent = s.returns[-lookback:]
            cum_ret = float(np.sum(recent))
            vol = float(np.std(recent)) if len(recent) > 1 else 1e-8
            vol = max(vol, 1e-8)

            momentum_scores[i] = cum_ret
            reversion_scores[i] = -cum_ret  # inverse
            vol_scores[i] = 1.0 / vol
            sharpe_scores[i] = cum_ret / vol

        # Normalize each component to [0, 1]
        for arr in [momentum_scores, reversion_scores, vol_scores, sharpe_scores]:
            rng = arr.max() - arr.min()
            if rng > 1e-10:
                arr[:] = (arr - arr.min()) / rng
            else:
                arr[:] = 1.0 / n

        # Composite score
        composite = (
            self._momentum_weight * momentum_scores
            + self._mean_reversion_weight * reversion_scores
            + 0.3 * vol_scores
            + 0.2 * sharpe_scores
        )

        # Convert to weights
        composite = np.maximum(composite, 1e-6)
        weights = composite / composite.sum()

        weight_dict = {s.name: float(w) for s, w in zip(strategies, weights)}
        returns_matrix = self._build_returns_matrix(strategies)
        exp_ret, exp_vol = self._portfolio_metrics(weights, returns_matrix)

        return AllocationResult(
            method=AllocationMethod.DYNAMIC,
            weights=weight_dict,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            expected_sharpe=(exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0.0,
            diversification_ratio=self._diversification_ratio(weights, strategies),
            metadata={
                "momentum_scores": dict(zip([s.name for s in strategies], momentum_scores.tolist())),
                "composite_scores": dict(zip([s.name for s in strategies], composite.tolist())),
            },
        )

    # ------------------------------------------------------------------
    # Constraint application
    # ------------------------------------------------------------------

    def _apply_constraints(
        self,
        result: AllocationResult,
        strategies: List[StrategyPerformance],
    ) -> AllocationResult:
        """Apply min/max constraints and locked weights, then renormalize."""
        weights = dict(result.weights)
        locked_total = 0.0
        free_strategies = []

        # First pass: apply locked weights and min/max bounds
        for s in strategies:
            c = self.constraints.get(s.name)
            if c is None:
                free_strategies.append(s.name)
                continue

            if c.locked:
                weights[s.name] = c.min_weight  # locked at its designated weight
                locked_total += c.min_weight
            else:
                w = weights.get(s.name, 0.0)
                w = max(w, c.min_weight)
                w = min(w, c.max_weight)
                weights[s.name] = w
                free_strategies.append(s.name)

        # Renormalize free strategies to fill remaining budget
        remaining = 1.0 - locked_total
        free_total = sum(weights[n] for n in free_strategies)

        if free_total > 0 and remaining > 0:
            scale = remaining / free_total
            for name in free_strategies:
                weights[name] = weights[name] * scale
                # Re-apply bounds after scaling
                c = self.constraints.get(name)
                if c:
                    weights[name] = max(weights[name], c.min_weight)
                    weights[name] = min(weights[name], c.max_weight)

        # Final normalization
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            weights = {k: v / total for k, v in weights.items()}

        result.weights = weights
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_returns_matrix(
        self,
        strategies: List[StrategyPerformance],
    ) -> np.ndarray:
        """Build an n x T matrix of returns (one row per strategy)."""
        if not strategies:
            return np.array([]).reshape(0, 0)

        min_len = min(len(s.returns) for s in strategies)
        if min_len == 0:
            return np.zeros((len(strategies), 1))

        matrix = np.zeros((len(strategies), min_len))
        for i, s in enumerate(strategies):
            matrix[i, :] = s.returns[-min_len:]
        return matrix

    def _portfolio_metrics(
        self,
        weights: np.ndarray,
        returns_matrix: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate annualized portfolio return and volatility."""
        if returns_matrix.size == 0 or len(weights) == 0:
            return 0.0, 0.0

        mean_returns = np.mean(returns_matrix, axis=1)
        port_return = float(weights @ mean_returns) * 252  # annualize

        if returns_matrix.shape[1] > 1:
            cov = np.cov(returns_matrix, rowvar=True)
            port_var = float(weights @ cov @ weights)
            port_vol = np.sqrt(port_var) * np.sqrt(252)
        else:
            port_vol = 0.0

        return port_return, port_vol

    def _diversification_ratio(
        self,
        weights: np.ndarray,
        strategies: List[StrategyPerformance],
    ) -> float:
        """
        DR = (weighted sum of individual vols) / portfolio vol.
        DR > 1 means diversification is reducing risk.
        """
        vols = np.array([max(s.volatility, 1e-8) for s in strategies])
        weighted_vol_sum = float(weights @ vols)
        returns_matrix = self._build_returns_matrix(strategies)

        if returns_matrix.shape[1] > 1:
            cov = np.cov(returns_matrix, rowvar=True)
            port_vol = np.sqrt(float(weights @ cov @ weights))
        else:
            port_vol = weighted_vol_sum

        if port_vol < 1e-10:
            return 1.0
        return weighted_vol_sum / port_vol

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return allocator state for monitoring."""
        return {
            "method": self.method.value,
            "total_capital": self.total_capital,
            "current_weights": self._current_weights,
            "num_allocations": len(self._history),
            "last_allocation_time": self._history[-1].timestamp if self._history else 0,
            "trigger_type": self.trigger.trigger_type.value,
            "drift_threshold": self.trigger.drift_threshold_pct,
            "constraints": {
                name: {"min": c.min_weight, "max": c.max_weight, "locked": c.locked}
                for name, c in self.constraints.items()
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state."""
        return {
            "method": self.method.value,
            "total_capital": self.total_capital,
            "current_weights": self._current_weights,
            "risk_free_rate": self.risk_free_rate,
            "history_count": len(self._history),
        }
