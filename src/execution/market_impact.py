"""
Mefai Autotrade - Market Impact Estimation
Implements Almgren-Chriss optimal execution model, square-root impact,
temporary/permanent impact decomposition, and execution cost analysis.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ImpactModel(Enum):
    """Available market impact models."""
    ALMGREN_CHRISS = "almgren_chriss"
    SQUARE_ROOT = "square_root"
    LINEAR = "linear"
    POWER_LAW = "power_law"
    KYLE_LAMBDA = "kyle_lambda"


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class AlmgrenChrissParams:
    """Parameters for the Almgren-Chriss optimal execution model.

    Reference: Almgren & Chriss (2000) - "Optimal Execution of Portfolio Transactions"
    """
    # Market parameters
    daily_volume: float = 0.0       # Average daily trading volume
    daily_volatility: float = 0.0   # Daily price volatility (as fraction, e.g. 0.02 = 2%)
    bid_ask_spread: float = 0.0     # Bid-ask spread in price units

    # Impact parameters
    permanent_impact_gamma: float = 0.0  # Permanent impact coefficient
    temporary_impact_eta: float = 0.0    # Temporary impact coefficient
    temporary_impact_epsilon: float = 0.0  # Fixed component of temporary impact

    # Risk parameters
    risk_aversion: float = 1e-6     # Trader risk aversion parameter (lambda)

    # Execution parameters
    order_quantity: float = 0.0     # Total quantity to execute
    time_horizon_seconds: float = 0.0  # Execution time horizon
    num_intervals: int = 10         # Number of trading intervals
    current_price: float = 0.0     # Current market price

    @property
    def interval_length(self) -> float:
        """Length of each trading interval in seconds."""
        if self.num_intervals == 0:
            return 0.0
        return self.time_horizon_seconds / self.num_intervals

    @property
    def participation_rate(self) -> float:
        """Expected participation rate in market volume."""
        if self.daily_volume == 0 or self.time_horizon_seconds == 0:
            return 0.0
        volume_in_window = self.daily_volume * (self.time_horizon_seconds / 86400.0)
        if volume_in_window == 0:
            return 0.0
        return self.order_quantity / volume_in_window


@dataclass
class ImpactEstimate:
    """Estimated market impact of a trade."""
    model: ImpactModel
    symbol: str
    side: str
    quantity: float
    current_price: float

    # Impact components (in basis points)
    temporary_impact_bps: float = 0.0   # Temporary price impact
    permanent_impact_bps: float = 0.0   # Permanent price impact
    total_impact_bps: float = 0.0       # Total expected impact
    spread_cost_bps: float = 0.0        # Half spread cost

    # Impact in price units
    temporary_impact_price: float = 0.0
    permanent_impact_price: float = 0.0
    expected_execution_price: float = 0.0

    # Cost estimates
    total_cost_bps: float = 0.0         # Total cost including spread + impact
    total_cost_usd: float = 0.0
    optimal_execution_time_seconds: float = 0.0
    optimal_num_slices: int = 0

    # Confidence
    confidence_pct: float = 0.0         # Model confidence (based on data quality)
    variance_bps: float = 0.0           # Expected variance of impact

    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.total_impact_bps = self.temporary_impact_bps + self.permanent_impact_bps
        self.total_cost_bps = self.total_impact_bps + self.spread_cost_bps
        if self.current_price > 0 and self.quantity > 0:
            notional = self.current_price * self.quantity
            self.total_cost_usd = notional * self.total_cost_bps / 10000.0


@dataclass
class ExecutionCostAnalysis:
    """Comprehensive execution cost analysis."""
    symbol: str
    side: str
    quantity: float
    current_price: float

    # Cost breakdown
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_estimate_bps: float = 0.0
    total_cost_bps: float = 0.0
    total_cost_usd: float = 0.0

    # Comparison of execution strategies
    strategy_costs: Dict[str, float] = field(default_factory=dict)
    recommended_strategy: str = ""
    recommended_duration_seconds: float = 0.0
    recommended_num_slices: int = 0

    # Impact decay
    impact_half_life_seconds: float = 0.0
    permanent_impact_pct: float = 0.0

    timestamp: float = field(default_factory=time.time)


@dataclass
class PostTradeAnalysis:
    """Post-trade execution cost analysis."""
    execution_id: str
    symbol: str
    side: str
    quantity: float
    benchmark_price: float      # Price at decision time
    arrival_price: float        # Price at start of execution
    avg_execution_price: float  # Volume-weighted average execution price
    closing_price: float        # Price at end of execution

    # Cost components (basis points)
    implementation_shortfall_bps: float = 0.0
    delay_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0
    opportunity_cost_bps: float = 0.0
    total_cost_bps: float = 0.0

    # Comparison
    vs_vwap_bps: float = 0.0
    vs_twap_bps: float = 0.0
    market_vwap: float = 0.0
    market_twap: float = 0.0

    fill_rate: float = 0.0
    execution_time_seconds: float = 0.0

    def compute(self) -> None:
        """Compute all cost components."""
        if self.benchmark_price <= 0:
            return

        notional = self.benchmark_price * self.quantity

        # Implementation shortfall = (avg_exec - benchmark) / benchmark
        if self.side == "BUY":
            self.implementation_shortfall_bps = (
                (self.avg_execution_price - self.benchmark_price) / self.benchmark_price * 10000.0
            )
        else:
            self.implementation_shortfall_bps = (
                (self.benchmark_price - self.avg_execution_price) / self.benchmark_price * 10000.0
            )

        # Delay cost = (arrival - benchmark) / benchmark
        if self.side == "BUY":
            self.delay_cost_bps = (
                (self.arrival_price - self.benchmark_price) / self.benchmark_price * 10000.0
            )
        else:
            self.delay_cost_bps = (
                (self.benchmark_price - self.arrival_price) / self.benchmark_price * 10000.0
            )

        # Market impact = (avg_exec - arrival) / arrival
        if self.arrival_price > 0:
            if self.side == "BUY":
                self.market_impact_bps = (
                    (self.avg_execution_price - self.arrival_price) / self.arrival_price * 10000.0
                )
            else:
                self.market_impact_bps = (
                    (self.arrival_price - self.avg_execution_price) / self.arrival_price * 10000.0
                )

        # Timing cost = (closing - avg_exec) / avg_exec * (1 - fill_rate)
        if self.avg_execution_price > 0 and self.fill_rate < 1.0:
            unfilled = 1.0 - self.fill_rate
            if self.side == "BUY":
                self.timing_cost_bps = (
                    (self.closing_price - self.avg_execution_price) / self.avg_execution_price
                    * 10000.0 * unfilled
                )
            else:
                self.timing_cost_bps = (
                    (self.avg_execution_price - self.closing_price) / self.avg_execution_price
                    * 10000.0 * unfilled
                )

        # Opportunity cost of unfilled portion
        if self.fill_rate < 1.0 and self.benchmark_price > 0:
            unfilled_qty = self.quantity * (1.0 - self.fill_rate)
            if self.side == "BUY":
                self.opportunity_cost_bps = (
                    (self.closing_price - self.benchmark_price) / self.benchmark_price
                    * 10000.0 * (1.0 - self.fill_rate)
                )
            else:
                self.opportunity_cost_bps = (
                    (self.benchmark_price - self.closing_price) / self.benchmark_price
                    * 10000.0 * (1.0 - self.fill_rate)
                )

        self.total_cost_bps = (
            self.delay_cost_bps + self.market_impact_bps
            + self.timing_cost_bps + self.opportunity_cost_bps
        )

        # VWAP comparison
        if self.market_vwap > 0:
            if self.side == "BUY":
                self.vs_vwap_bps = (
                    (self.avg_execution_price - self.market_vwap) / self.market_vwap * 10000.0
                )
            else:
                self.vs_vwap_bps = (
                    (self.market_vwap - self.avg_execution_price) / self.market_vwap * 10000.0
                )

        # TWAP comparison
        if self.market_twap > 0:
            if self.side == "BUY":
                self.vs_twap_bps = (
                    (self.avg_execution_price - self.market_twap) / self.market_twap * 10000.0
                )
            else:
                self.vs_twap_bps = (
                    (self.market_twap - self.avg_execution_price) / self.market_twap * 10000.0
                )


# ---------------------------------------------------------------------------
# Impact Models
# ---------------------------------------------------------------------------

class SquareRootImpactModel:
    """Square-root market impact model.

    Impact = sigma * sqrt(Q / V) * sign(side)
    where sigma = volatility, Q = order quantity, V = daily volume

    This is the most widely observed empirical relationship in equity
    and crypto markets.
    """

    def __init__(self, impact_coefficient: float = 1.0):
        self._impact_coefficient = impact_coefficient

    def estimate(self, quantity: float, daily_volume: float,
                 volatility: float, current_price: float,
                 side: str, spread_bps: float = 0.0) -> ImpactEstimate:
        """Estimate market impact using square-root model."""
        if daily_volume <= 0 or volatility <= 0 or current_price <= 0:
            return ImpactEstimate(
                model=ImpactModel.SQUARE_ROOT,
                symbol="",
                side=side,
                quantity=quantity,
                current_price=current_price,
                confidence_pct=0.0,
            )

        participation = quantity / daily_volume
        sqrt_participation = math.sqrt(participation)

        # Total impact in return space
        impact_return = self._impact_coefficient * volatility * sqrt_participation

        # Decompose into temporary (2/3) and permanent (1/3)
        # Based on empirical findings from Bouchaud et al.
        temp_return = impact_return * 0.67
        perm_return = impact_return * 0.33

        temp_bps = temp_return * 10000.0
        perm_bps = perm_return * 10000.0
        half_spread = spread_bps / 2.0

        # Expected execution price
        total_impact_price = current_price * impact_return
        if side == "BUY":
            expected_price = current_price + total_impact_price
        else:
            expected_price = current_price - total_impact_price

        # Variance estimate
        variance = (volatility * sqrt_participation * 0.5) ** 2
        variance_bps = variance * 10000.0

        # Confidence based on participation rate
        if participation < 0.01:
            confidence = 90.0
        elif participation < 0.05:
            confidence = 75.0
        elif participation < 0.10:
            confidence = 60.0
        elif participation < 0.25:
            confidence = 40.0
        else:
            confidence = 20.0

        estimate = ImpactEstimate(
            model=ImpactModel.SQUARE_ROOT,
            symbol="",
            side=side,
            quantity=quantity,
            current_price=current_price,
            temporary_impact_bps=temp_bps,
            permanent_impact_bps=perm_bps,
            spread_cost_bps=half_spread,
            temporary_impact_price=current_price * temp_return,
            permanent_impact_price=current_price * perm_return,
            expected_execution_price=expected_price,
            confidence_pct=confidence,
            variance_bps=variance_bps,
        )
        return estimate


class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution model.

    Solves for the optimal execution trajectory that minimizes
    the combination of market impact cost and execution risk.
    """

    def compute_optimal_trajectory(self, params: AlmgrenChrissParams) -> Dict[str, Any]:
        """Compute the optimal execution trajectory.

        Returns dict with:
        - trajectory: List of (time, remaining_quantity) tuples
        - trade_list: List of quantity to trade in each interval
        - expected_cost: Expected total cost
        - variance: Variance of cost
        """
        X = params.order_quantity
        T = params.time_horizon_seconds
        N = params.num_intervals
        tau = params.interval_length
        sigma = params.daily_volatility
        eta = params.temporary_impact_eta
        gamma = params.permanent_impact_gamma
        lam = params.risk_aversion

        if X == 0 or T == 0 or N == 0:
            return {
                "trajectory": [],
                "trade_list": [],
                "expected_cost": 0.0,
                "variance": 0.0,
            }

        # Convert daily volatility to per-interval
        intervals_per_day = 86400.0 / tau if tau > 0 else 1.0
        sigma_tau = sigma / math.sqrt(intervals_per_day) if intervals_per_day > 0 else sigma

        # Compute kappa - the urgency parameter
        # kappa = sqrt(lambda * sigma^2 / eta)
        if eta <= 0:
            eta = 0.001  # Prevent division by zero
        kappa_sq = lam * sigma_tau ** 2 / eta
        kappa = math.sqrt(kappa_sq) if kappa_sq > 0 else 0.001

        # Optimal trajectory: x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
        kappa_T = kappa * N  # kappa * number of intervals
        sinh_kappa_T = math.sinh(kappa_T) if abs(kappa_T) < 700 else float("inf")

        trajectory = []
        trade_list = []
        prev_holding = X

        for j in range(N + 1):
            t_j = j
            remaining_intervals = N - j
            if sinh_kappa_T > 0 and sinh_kappa_T < float("inf"):
                holding = X * math.sinh(kappa * remaining_intervals) / sinh_kappa_T
            else:
                # Linear liquidation fallback
                holding = X * (1.0 - j / N)

            trajectory.append((j * tau, holding))

            if j > 0:
                trade = prev_holding - holding
                trade_list.append(trade)

            prev_holding = holding

        # Expected cost
        # E[cost] = 0.5 * gamma * X^2 + epsilon * X + sum(eta/tau * n_j^2 + 0.5 * gamma * n_j^2)
        epsilon = params.temporary_impact_epsilon
        expected_cost = 0.5 * gamma * X ** 2 + epsilon * abs(X)

        for n_j in trade_list:
            if tau > 0:
                expected_cost += eta / tau * n_j ** 2
            expected_cost += 0.5 * gamma * n_j ** 2

        # Variance of cost
        # Var[cost] = sigma^2 * sum(x_j^2 * tau)
        variance = 0.0
        for t, x_j in trajectory[:-1]:
            variance += sigma_tau ** 2 * x_j ** 2

        return {
            "trajectory": trajectory,
            "trade_list": trade_list,
            "expected_cost": expected_cost,
            "variance": variance,
            "std_dev": math.sqrt(variance) if variance > 0 else 0.0,
            "kappa": kappa,
            "urgency": "high" if kappa > 1.0 else "medium" if kappa > 0.1 else "low",
        }

    def estimate_impact(self, params: AlmgrenChrissParams) -> ImpactEstimate:
        """Estimate impact using the Almgren-Chriss model."""
        result = self.compute_optimal_trajectory(params)

        if not result["trade_list"] or params.current_price <= 0:
            return ImpactEstimate(
                model=ImpactModel.ALMGREN_CHRISS,
                symbol="",
                side="BUY",
                quantity=params.order_quantity,
                current_price=params.current_price,
            )

        notional = params.current_price * params.order_quantity
        expected_cost = result["expected_cost"]

        # Convert cost to bps
        if notional > 0:
            cost_bps = (expected_cost / notional) * 10000.0
        else:
            cost_bps = 0.0

        # Decompose
        temp_bps = cost_bps * 0.6  # Temporary component
        perm_bps = cost_bps * 0.4  # Permanent component
        half_spread = params.bid_ask_spread / params.current_price * 5000.0 if params.current_price > 0 else 0.0

        return ImpactEstimate(
            model=ImpactModel.ALMGREN_CHRISS,
            symbol="",
            side="BUY",
            quantity=params.order_quantity,
            current_price=params.current_price,
            temporary_impact_bps=temp_bps,
            permanent_impact_bps=perm_bps,
            spread_cost_bps=half_spread,
            temporary_impact_price=params.current_price * temp_bps / 10000.0,
            permanent_impact_price=params.current_price * perm_bps / 10000.0,
            expected_execution_price=params.current_price * (1 + cost_bps / 10000.0),
            optimal_execution_time_seconds=params.time_horizon_seconds,
            optimal_num_slices=params.num_intervals,
            variance_bps=result["std_dev"] / notional * 10000.0 if notional > 0 else 0.0,
            confidence_pct=70.0,
        )

    def compute_optimal_time(self, params: AlmgrenChrissParams) -> float:
        """Compute the optimal execution time horizon.

        Balances urgency (risk aversion) against impact cost.
        Shorter time = more impact but less risk.
        Longer time = less impact but more risk.
        """
        X = params.order_quantity
        sigma = params.daily_volatility
        eta = params.temporary_impact_eta
        lam = params.risk_aversion

        if sigma <= 0 or eta <= 0 or lam <= 0 or X <= 0:
            return 3600.0  # Default 1 hour

        # Optimal T ~ sqrt(eta * X / (lambda * sigma^2))
        optimal_T = math.sqrt(eta * abs(X) / (lam * sigma ** 2))

        # Convert to seconds (assuming parameters are in daily units)
        optimal_seconds = optimal_T * 86400.0

        # Clamp to reasonable range
        optimal_seconds = max(60.0, min(optimal_seconds, 86400.0))

        return optimal_seconds


class ImpactDecayModel:
    """Model how market impact decays over time after execution."""

    def __init__(self, half_life_seconds: float = 300.0,
                 permanent_fraction: float = 0.33):
        self._half_life = half_life_seconds
        self._permanent_fraction = permanent_fraction
        self._decay_constant = math.log(2) / half_life_seconds if half_life_seconds > 0 else 0.01

    def impact_at_time(self, initial_impact_bps: float,
                       seconds_after: float) -> float:
        """Calculate remaining impact at a given time after execution.

        Returns impact in basis points.
        """
        permanent = initial_impact_bps * self._permanent_fraction
        temporary = initial_impact_bps * (1.0 - self._permanent_fraction)

        # Exponential decay of temporary component
        decayed_temp = temporary * math.exp(-self._decay_constant * seconds_after)

        return permanent + decayed_temp

    def time_to_recover(self, initial_impact_bps: float,
                        target_impact_bps: float) -> float:
        """Calculate time for impact to decay to target level.

        Returns seconds, or -1 if target is below permanent impact.
        """
        permanent = initial_impact_bps * self._permanent_fraction
        if target_impact_bps <= permanent:
            return -1.0  # Will never decay below permanent

        temporary = initial_impact_bps * (1.0 - self._permanent_fraction)
        target_temp = target_impact_bps - permanent

        if temporary <= 0 or target_temp >= temporary:
            return 0.0

        return -math.log(target_temp / temporary) / self._decay_constant

    def get_decay_curve(self, initial_impact_bps: float,
                        duration_seconds: float,
                        num_points: int = 50) -> List[Tuple[float, float]]:
        """Generate impact decay curve as list of (time, impact_bps) points."""
        points = []
        for i in range(num_points):
            t = duration_seconds * i / (num_points - 1) if num_points > 1 else 0
            impact = self.impact_at_time(initial_impact_bps, t)
            points.append((t, impact))
        return points


# ---------------------------------------------------------------------------
# Market Impact Estimator - main interface
# ---------------------------------------------------------------------------

class MarketImpactEstimator:
    """Main interface for market impact estimation.

    Provides pre-trade and post-trade cost analysis using
    multiple impact models.
    """

    def __init__(self):
        self._sqrt_model = SquareRootImpactModel()
        self._ac_model = AlmgrenChrissModel()
        self._decay_model = ImpactDecayModel()
        self._default_commission_bps = 2.0  # Default trading commission
        self._execution_history: List[PostTradeAnalysis] = []
        logger.info("MarketImpactEstimator initialized")

    def set_commission(self, bps: float) -> None:
        """Set default commission rate in basis points."""
        self._default_commission_bps = bps

    def set_decay_params(self, half_life_seconds: float,
                         permanent_fraction: float) -> None:
        """Configure impact decay model."""
        self._decay_model = ImpactDecayModel(half_life_seconds, permanent_fraction)

    def estimate_impact(self, symbol: str, side: str, quantity: float,
                        current_price: float, daily_volume: float,
                        daily_volatility: float,
                        spread_bps: float = 5.0,
                        model: ImpactModel = ImpactModel.SQUARE_ROOT) -> ImpactEstimate:
        """Estimate market impact for a trade.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            quantity: Order quantity
            current_price: Current market price
            daily_volume: Average daily volume in base currency
            daily_volatility: Daily volatility as decimal (e.g. 0.03 = 3%)
            spread_bps: Typical bid-ask spread in basis points
            model: Which impact model to use

        Returns:
            ImpactEstimate with detailed breakdown
        """
        if model == ImpactModel.SQUARE_ROOT:
            estimate = self._sqrt_model.estimate(
                quantity, daily_volume, daily_volatility,
                current_price, side, spread_bps
            )
        elif model == ImpactModel.ALMGREN_CHRISS:
            params = AlmgrenChrissParams(
                daily_volume=daily_volume,
                daily_volatility=daily_volatility,
                bid_ask_spread=current_price * spread_bps / 10000.0,
                permanent_impact_gamma=daily_volatility * 0.1,
                temporary_impact_eta=daily_volatility * 0.05,
                temporary_impact_epsilon=current_price * spread_bps / 20000.0,
                order_quantity=quantity,
                time_horizon_seconds=3600.0,
                num_intervals=20,
                current_price=current_price,
            )
            estimate = self._ac_model.estimate_impact(params)
        elif model == ImpactModel.LINEAR:
            estimate = self._linear_estimate(
                quantity, daily_volume, daily_volatility,
                current_price, side, spread_bps
            )
        elif model == ImpactModel.POWER_LAW:
            estimate = self._power_law_estimate(
                quantity, daily_volume, daily_volatility,
                current_price, side, spread_bps
            )
        else:
            estimate = self._sqrt_model.estimate(
                quantity, daily_volume, daily_volatility,
                current_price, side, spread_bps
            )

        estimate.symbol = symbol
        return estimate

    def _linear_estimate(self, quantity: float, daily_volume: float,
                         volatility: float, price: float,
                         side: str, spread_bps: float) -> ImpactEstimate:
        """Simple linear impact model."""
        if daily_volume <= 0:
            return ImpactEstimate(
                model=ImpactModel.LINEAR, symbol="", side=side,
                quantity=quantity, current_price=price,
            )
        participation = quantity / daily_volume
        impact_bps = volatility * participation * 10000.0
        return ImpactEstimate(
            model=ImpactModel.LINEAR,
            symbol="",
            side=side,
            quantity=quantity,
            current_price=price,
            temporary_impact_bps=impact_bps * 0.6,
            permanent_impact_bps=impact_bps * 0.4,
            spread_cost_bps=spread_bps / 2.0,
            confidence_pct=50.0,
        )

    def _power_law_estimate(self, quantity: float, daily_volume: float,
                            volatility: float, price: float,
                            side: str, spread_bps: float,
                            exponent: float = 0.6) -> ImpactEstimate:
        """Power-law impact model: impact ~ (Q/V)^alpha"""
        if daily_volume <= 0:
            return ImpactEstimate(
                model=ImpactModel.POWER_LAW, symbol="", side=side,
                quantity=quantity, current_price=price,
            )
        participation = quantity / daily_volume
        impact_return = volatility * (participation ** exponent)
        impact_bps = impact_return * 10000.0
        return ImpactEstimate(
            model=ImpactModel.POWER_LAW,
            symbol="",
            side=side,
            quantity=quantity,
            current_price=price,
            temporary_impact_bps=impact_bps * 0.65,
            permanent_impact_bps=impact_bps * 0.35,
            spread_cost_bps=spread_bps / 2.0,
            confidence_pct=60.0,
        )

    def compute_execution_cost(self, symbol: str, side: str, quantity: float,
                                current_price: float, daily_volume: float,
                                daily_volatility: float,
                                spread_bps: float = 5.0,
                                commission_bps: Optional[float] = None) -> ExecutionCostAnalysis:
        """Compute comprehensive execution cost analysis.

        Compares different execution strategies and recommends optimal approach.
        """
        comm = commission_bps if commission_bps is not None else self._default_commission_bps

        # Estimate impact for different strategies
        sqrt_impact = self._sqrt_model.estimate(
            quantity, daily_volume, daily_volatility, current_price, side, spread_bps
        )

        # TWAP cost: lower impact (spread over time) but higher risk
        twap_impact_bps = sqrt_impact.total_impact_bps * 0.5
        twap_risk_bps = daily_volatility * 10000.0 * 0.1
        twap_total = twap_impact_bps + twap_risk_bps + spread_bps / 2.0 + comm

        # VWAP cost: follows volume profile, moderate impact
        vwap_impact_bps = sqrt_impact.total_impact_bps * 0.4
        vwap_risk_bps = daily_volatility * 10000.0 * 0.08
        vwap_total = vwap_impact_bps + vwap_risk_bps + spread_bps / 2.0 + comm

        # Immediate (market order): full impact, no risk
        immediate_total = sqrt_impact.total_impact_bps + spread_bps / 2.0 + comm

        # Iceberg: moderate impact
        iceberg_impact_bps = sqrt_impact.total_impact_bps * 0.6
        iceberg_total = iceberg_impact_bps + spread_bps / 2.0 + comm

        strategy_costs = {
            "immediate": immediate_total,
            "twap": twap_total,
            "vwap": vwap_total,
            "iceberg": iceberg_total,
        }

        # Find best strategy
        best = min(strategy_costs, key=strategy_costs.get)

        # Optimal execution parameters
        participation = quantity / daily_volume if daily_volume > 0 else 1.0
        if participation < 0.01:
            rec_duration = 300.0   # 5 minutes
            rec_slices = 5
        elif participation < 0.05:
            rec_duration = 1800.0  # 30 minutes
            rec_slices = 15
        elif participation < 0.10:
            rec_duration = 3600.0  # 1 hour
            rec_slices = 30
        else:
            rec_duration = 7200.0  # 2 hours
            rec_slices = 60

        # Impact decay
        half_life = max(60.0, min(600.0, 300.0 * participation * 10))
        perm_pct = 0.33

        notional = current_price * quantity
        total_cost = strategy_costs[best]

        return ExecutionCostAnalysis(
            symbol=symbol,
            side=side,
            quantity=quantity,
            current_price=current_price,
            spread_cost_bps=spread_bps / 2.0,
            market_impact_bps=sqrt_impact.total_impact_bps,
            commission_bps=comm,
            slippage_estimate_bps=sqrt_impact.variance_bps,
            total_cost_bps=total_cost,
            total_cost_usd=notional * total_cost / 10000.0,
            strategy_costs=strategy_costs,
            recommended_strategy=best,
            recommended_duration_seconds=rec_duration,
            recommended_num_slices=rec_slices,
            impact_half_life_seconds=half_life,
            permanent_impact_pct=perm_pct * 100.0,
        )

    def compute_optimal_speed(self, symbol: str, quantity: float,
                               daily_volume: float, daily_volatility: float,
                               risk_aversion: float = 1e-6) -> Dict[str, Any]:
        """Compute optimal execution speed using Almgren-Chriss."""
        params = AlmgrenChrissParams(
            daily_volume=daily_volume,
            daily_volatility=daily_volatility,
            permanent_impact_gamma=daily_volatility * 0.1,
            temporary_impact_eta=daily_volatility * 0.05,
            risk_aversion=risk_aversion,
            order_quantity=quantity,
            time_horizon_seconds=3600.0,
            num_intervals=20,
        )

        optimal_time = self._ac_model.compute_optimal_time(params)
        params.time_horizon_seconds = optimal_time
        params.num_intervals = max(2, int(optimal_time / 60))

        trajectory = self._ac_model.compute_optimal_trajectory(params)

        return {
            "optimal_time_seconds": optimal_time,
            "optimal_time_minutes": optimal_time / 60.0,
            "optimal_num_slices": params.num_intervals,
            "urgency": trajectory.get("urgency", "medium"),
            "expected_cost": trajectory.get("expected_cost", 0),
            "cost_std_dev": trajectory.get("std_dev", 0),
            "trade_schedule": trajectory.get("trade_list", []),
        }

    def analyze_post_trade(self, execution_id: str, symbol: str, side: str,
                            quantity: float, benchmark_price: float,
                            arrival_price: float, avg_execution_price: float,
                            closing_price: float, fill_rate: float = 1.0,
                            execution_time_seconds: float = 0.0,
                            market_vwap: float = 0.0,
                            market_twap: float = 0.0) -> PostTradeAnalysis:
        """Perform post-trade cost analysis."""
        analysis = PostTradeAnalysis(
            execution_id=execution_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            benchmark_price=benchmark_price,
            arrival_price=arrival_price,
            avg_execution_price=avg_execution_price,
            closing_price=closing_price,
            fill_rate=fill_rate,
            execution_time_seconds=execution_time_seconds,
            market_vwap=market_vwap,
            market_twap=market_twap,
        )
        analysis.compute()
        self._execution_history.append(analysis)
        return analysis

    def get_impact_decay_curve(self, initial_impact_bps: float,
                                duration_seconds: float = 1800.0,
                                num_points: int = 50) -> List[Tuple[float, float]]:
        """Get impact decay curve."""
        return self._decay_model.get_decay_curve(
            initial_impact_bps, duration_seconds, num_points
        )

    def get_execution_history(self, symbol: Optional[str] = None,
                               limit: int = 100) -> List[PostTradeAnalysis]:
        """Get post-trade analysis history."""
        history = self._execution_history
        if symbol:
            history = [h for h in history if h.symbol == symbol]
        return history[-limit:]

    def get_avg_implementation_shortfall(self, symbol: Optional[str] = None) -> float:
        """Get average implementation shortfall across executions."""
        history = self.get_execution_history(symbol)
        if not history:
            return 0.0
        return sum(h.implementation_shortfall_bps for h in history) / len(history)
