# Mefai Signal Engine - Strategy Parameter Optimizer
# Grid search, random search, Bayesian optimization, genetic algorithm,
# walk-forward optimization, multi-objective optimization, and overfitting detection.

import copy
import math
import random
import time
import logging
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Tuple, Callable, Set, Union, Type
)
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ParameterSpec:
    """Specification for a single parameter to optimize."""
    name: str = ""
    min_value: float = 0.0
    max_value: float = 1.0
    step: float = 0.0
    values: Optional[List[Any]] = None  # discrete values
    param_type: str = "float"  # float, int, categorical
    log_scale: bool = False  # use log scale for sampling

    def get_values(self) -> List[Any]:
        """Get all possible values for this parameter."""
        if self.values is not None:
            return self.values
        if self.param_type == "int":
            step = max(1, int(self.step)) if self.step > 0 else 1
            return list(range(int(self.min_value), int(self.max_value) + 1, step))
        if self.step > 0:
            values = []
            v = self.min_value
            while v <= self.max_value + 1e-10:
                values.append(round(v, 10))
                v += self.step
            return values
        return [self.min_value, self.max_value]

    def sample_random(self) -> Any:
        """Sample a random value from this parameter's range."""
        if self.values is not None:
            return random.choice(self.values)
        if self.param_type == "categorical":
            return random.choice(self.get_values())
        if self.log_scale and self.min_value > 0:
            log_min = math.log(self.min_value)
            log_max = math.log(self.max_value)
            val = math.exp(random.uniform(log_min, log_max))
        else:
            val = random.uniform(self.min_value, self.max_value)
        if self.param_type == "int":
            val = int(round(val))
        if self.step > 0:
            val = round(val / self.step) * self.step
            val = max(self.min_value, min(self.max_value, val))
        return val

    def mutate(self, value: Any, strength: float = 0.1) -> Any:
        """Mutate a parameter value for genetic algorithm."""
        if self.values is not None or self.param_type == "categorical":
            if random.random() < strength:
                return self.sample_random()
            return value
        range_size = self.max_value - self.min_value
        delta = random.gauss(0, range_size * strength)
        new_val = value + delta
        new_val = max(self.min_value, min(self.max_value, new_val))
        if self.param_type == "int":
            new_val = int(round(new_val))
        if self.step > 0:
            new_val = round(new_val / self.step) * self.step
        return new_val


@dataclass
class OptimizationResult:
    """Result from a single parameter combination evaluation."""
    params: Dict[str, Any] = field(default_factory=dict)
    metric_value: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    execution_time: float = 0.0
    is_sample: str = "in_sample"  # in_sample or out_of_sample


@dataclass
class FullOptimizationResult:
    """Complete results from an optimization run."""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_metric: float = 0.0
    all_results: List[OptimizationResult] = field(default_factory=list)
    total_combinations: int = 0
    evaluated_combinations: int = 0
    total_time: float = 0.0
    optimizer_type: str = ""
    parameter_specs: List[ParameterSpec] = field(default_factory=list)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    overfitting_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiObjectiveResult:
    """Result for multi-objective optimization."""
    pareto_front: List[OptimizationResult] = field(default_factory=list)
    all_results: List[OptimizationResult] = field(default_factory=list)
    dominated_count: int = 0
    non_dominated_count: int = 0


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

class ObjectiveFunction:
    """
    Defines the objective function to maximize during optimization.
    Default is Sharpe ratio, but custom objectives can be defined.
    """

    BUILT_IN = {
        "sharpe": lambda r: r.sharpe_ratio,
        "sortino": lambda r: r.metrics.get("sortino_ratio", 0),
        "total_return": lambda r: r.total_return,
        "profit_factor": lambda r: r.profit_factor,
        "calmar": lambda r: r.total_return / max(abs(r.max_drawdown), 0.01),
        "win_rate": lambda r: r.win_rate,
        "risk_adjusted": lambda r: (
            r.total_return * r.win_rate / max(abs(r.max_drawdown), 0.01)
        ),
    }

    def __init__(self, name: str = "sharpe",
                 custom_fn: Optional[Callable] = None,
                 maximize: bool = True):
        self.name = name
        self.maximize = maximize
        if custom_fn:
            self._fn = custom_fn
        elif name in self.BUILT_IN:
            self._fn = self.BUILT_IN[name]
        else:
            raise ValueError(
                f"Unknown objective: {name}. "
                f"Available: {list(self.BUILT_IN.keys())}"
            )

    def evaluate(self, result: OptimizationResult) -> float:
        val = self._fn(result)
        return val if self.maximize else -val


# ---------------------------------------------------------------------------
# Base optimizer
# ---------------------------------------------------------------------------

class StrategyOptimizer(ABC):
    """
    Abstract base class for strategy optimizers.

    Subclasses must implement _generate_param_sets() which yields
    parameter dictionaries to evaluate.
    """

    def __init__(self, param_specs: List[ParameterSpec],
                 objective: Optional[ObjectiveFunction] = None,
                 max_evaluations: int = 0,
                 verbose: bool = True):
        self.param_specs = param_specs
        self.objective = objective or ObjectiveFunction("sharpe")
        self.max_evaluations = max_evaluations
        self.verbose = verbose
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None

    @abstractmethod
    def _generate_param_sets(self) -> List[Dict[str, Any]]:
        """Generate parameter sets to evaluate."""
        pass

    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], OptimizationResult],
                 ) -> FullOptimizationResult:
        """
        Run the optimization.

        evaluate_fn: function that takes a parameter dict and returns
                     an OptimizationResult
        """
        start_time = time.time()
        param_sets = self._generate_param_sets()
        total = len(param_sets)

        if self.max_evaluations > 0 and total > self.max_evaluations:
            random.shuffle(param_sets)
            param_sets = param_sets[:self.max_evaluations]
            total = len(param_sets)

        logger.info(
            f"{self.__class__.__name__}: evaluating {total} parameter sets"
        )

        self.results = []
        self.best_result = None

        for i, params in enumerate(param_sets):
            try:
                eval_start = time.time()
                result = evaluate_fn(params)
                result.execution_time = time.time() - eval_start
                result.metric_value = self.objective.evaluate(result)
                self.results.append(result)

                if self.best_result is None or \
                   result.metric_value > self.best_result.metric_value:
                    self.best_result = result
                    if self.verbose:
                        logger.info(
                            f"  [{i+1}/{total}] New best: "
                            f"{self.objective.name}={result.metric_value:.4f} "
                            f"params={result.params}"
                        )
                elif self.verbose and (i + 1) % max(1, total // 20) == 0:
                    logger.info(f"  [{i+1}/{total}] Progress...")

            except Exception as e:
                logger.error(f"  Evaluation error for params {params}: {e}")
                continue

        elapsed = time.time() - start_time
        best_params = self.best_result.params if self.best_result else {}
        best_metric = self.best_result.metric_value if self.best_result else 0.0

        # Calculate parameter stability
        stability = self._calculate_stability()

        full_result = FullOptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            all_results=self.results,
            total_combinations=total,
            evaluated_combinations=len(self.results),
            total_time=elapsed,
            optimizer_type=self.__class__.__name__,
            parameter_specs=self.param_specs,
            stability_scores=stability,
        )

        logger.info(
            f"Optimization complete: {len(self.results)}/{total} evaluated "
            f"in {elapsed:.1f}s, best {self.objective.name}={best_metric:.4f}"
        )

        return full_result

    def _calculate_stability(self) -> Dict[str, float]:
        """
        Calculate parameter stability scores.

        For each parameter, measures how much the objective changes
        when that parameter varies while others are held near optimal.
        """
        if not self.results or self.best_result is None:
            return {}

        stability = {}
        sorted_results = sorted(
            self.results, key=lambda r: r.metric_value, reverse=True
        )
        top_n = max(1, len(sorted_results) // 10)
        top_results = sorted_results[:top_n]

        for spec in self.param_specs:
            values = [r.params.get(spec.name) for r in top_results
                      if spec.name in r.params]
            if not values or len(values) < 2:
                stability[spec.name] = 1.0
                continue

            # For numeric values, calculate coefficient of variation
            if spec.param_type in ("float", "int"):
                try:
                    numeric_vals = [float(v) for v in values]
                    mean = sum(numeric_vals) / len(numeric_vals)
                    if abs(mean) < 1e-10:
                        stability[spec.name] = 1.0
                        continue
                    variance = sum((v - mean) ** 2 for v in numeric_vals) / len(numeric_vals)
                    cv = (variance ** 0.5) / abs(mean)
                    stability[spec.name] = max(0.0, 1.0 - cv)
                except (ValueError, TypeError):
                    stability[spec.name] = 0.5
            else:
                # For categorical, check how many unique values in top results
                unique = len(set(values))
                stability[spec.name] = 1.0 / unique

        return stability


# ---------------------------------------------------------------------------
# Grid search optimizer
# ---------------------------------------------------------------------------

class GridSearchOptimizer(StrategyOptimizer):
    """
    Exhaustive grid search over all parameter combinations.

    Best for small parameter spaces (< 10000 combinations).
    """

    def _generate_param_sets(self) -> List[Dict[str, Any]]:
        param_values = {}
        for spec in self.param_specs:
            param_values[spec.name] = spec.get_values()

        # Calculate total combinations
        total = 1
        for vals in param_values.values():
            total *= len(vals)

        logger.info(f"Grid search: {total} total combinations")

        names = list(param_values.keys())
        all_values = [param_values[n] for n in names]

        param_sets = []
        for combo in itertools.product(*all_values):
            params = dict(zip(names, combo))
            param_sets.append(params)

        return param_sets


# ---------------------------------------------------------------------------
# Random search optimizer
# ---------------------------------------------------------------------------

class RandomSearchOptimizer(StrategyOptimizer):
    """
    Random sampling from the parameter space.

    More efficient than grid search for high-dimensional spaces.
    Each parameter is sampled independently.
    """

    def __init__(self, param_specs: List[ParameterSpec],
                 n_iterations: int = 100,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(param_specs, **kwargs)
        self.n_iterations = n_iterations
        if seed is not None:
            random.seed(seed)

    def _generate_param_sets(self) -> List[Dict[str, Any]]:
        param_sets = []
        seen = set()

        attempts = 0
        max_attempts = self.n_iterations * 10

        while len(param_sets) < self.n_iterations and attempts < max_attempts:
            params = {}
            for spec in self.param_specs:
                params[spec.name] = spec.sample_random()

            # Avoid exact duplicates
            key = tuple(sorted(params.items()))
            if key not in seen:
                seen.add(key)
                param_sets.append(params)
            attempts += 1

        return param_sets


# ---------------------------------------------------------------------------
# Bayesian optimizer
# ---------------------------------------------------------------------------

class BayesianOptimizer(StrategyOptimizer):
    """
    Bayesian optimization using a Gaussian Process surrogate model.

    Uses expected improvement (EI) as the acquisition function.
    Falls back to random sampling if the surrogate model is not available.

    This is a simplified implementation that does not require external
    libraries. For production use, consider using scikit-optimize or
    Optuna.
    """

    def __init__(self, param_specs: List[ParameterSpec],
                 n_initial: int = 20,
                 n_iterations: int = 100,
                 exploration_weight: float = 0.1,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(param_specs, **kwargs)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        if seed is not None:
            random.seed(seed)

        self._X: List[List[float]] = []
        self._y: List[float] = []
        self._best_y: float = float('-inf')

    def _generate_param_sets(self) -> List[Dict[str, Any]]:
        # For Bayesian optimization, we generate initial random points
        # The actual optimization loop is in optimize()
        param_sets = []
        for _ in range(self.n_initial):
            params = {spec.name: spec.sample_random() for spec in self.param_specs}
            param_sets.append(params)
        return param_sets

    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], OptimizationResult],
                 ) -> FullOptimizationResult:
        """Override optimize to implement the Bayesian optimization loop."""
        start_time = time.time()
        self.results = []
        self.best_result = None
        self._X = []
        self._y = []
        self._best_y = float('-inf')

        total = self.n_initial + self.n_iterations

        # Phase 1: Random initial exploration
        logger.info(
            f"Bayesian optimization: {self.n_initial} initial + "
            f"{self.n_iterations} optimization iterations"
        )

        for i in range(self.n_initial):
            params = {spec.name: spec.sample_random() for spec in self.param_specs}
            result = self._evaluate_and_record(params, evaluate_fn, i, total)
            if result:
                x_vec = self._params_to_vector(params)
                self._X.append(x_vec)
                self._y.append(result.metric_value)
                if result.metric_value > self._best_y:
                    self._best_y = result.metric_value

        # Phase 2: Bayesian optimization
        for i in range(self.n_iterations):
            # Generate candidate points
            candidates = self._generate_candidates(100)

            # Score candidates using surrogate model
            best_candidate = None
            best_ei = float('-inf')

            for candidate in candidates:
                x_vec = self._params_to_vector(candidate)
                ei = self._expected_improvement(x_vec)
                if ei > best_ei:
                    best_ei = ei
                    best_candidate = candidate

            if best_candidate is None:
                best_candidate = {
                    spec.name: spec.sample_random() for spec in self.param_specs
                }

            result = self._evaluate_and_record(
                best_candidate, evaluate_fn, self.n_initial + i, total
            )
            if result:
                x_vec = self._params_to_vector(best_candidate)
                self._X.append(x_vec)
                self._y.append(result.metric_value)
                if result.metric_value > self._best_y:
                    self._best_y = result.metric_value

        elapsed = time.time() - start_time
        stability = self._calculate_stability()

        best_params = self.best_result.params if self.best_result else {}
        best_metric = self.best_result.metric_value if self.best_result else 0.0

        return FullOptimizationResult(
            best_params=best_params,
            best_metric=best_metric,
            all_results=self.results,
            total_combinations=total,
            evaluated_combinations=len(self.results),
            total_time=elapsed,
            optimizer_type="BayesianOptimizer",
            parameter_specs=self.param_specs,
            stability_scores=stability,
        )

    def _evaluate_and_record(self, params: Dict[str, Any],
                              evaluate_fn: Callable,
                              idx: int, total: int) -> Optional[OptimizationResult]:
        try:
            eval_start = time.time()
            result = evaluate_fn(params)
            result.execution_time = time.time() - eval_start
            result.metric_value = self.objective.evaluate(result)
            self.results.append(result)

            if self.best_result is None or \
               result.metric_value > self.best_result.metric_value:
                self.best_result = result
                if self.verbose:
                    logger.info(
                        f"  [{idx+1}/{total}] New best: "
                        f"{self.objective.name}={result.metric_value:.4f}"
                    )
            return result
        except Exception as e:
            logger.error(f"  Evaluation error: {e}")
            return None

    def _params_to_vector(self, params: Dict[str, Any]) -> List[float]:
        """Convert parameter dict to a normalized vector."""
        vec = []
        for spec in self.param_specs:
            val = params.get(spec.name, spec.min_value)
            if spec.param_type == "categorical" and spec.values:
                try:
                    idx = spec.values.index(val)
                    vec.append(idx / max(1, len(spec.values) - 1))
                except ValueError:
                    vec.append(0.5)
            else:
                val = float(val)
                range_size = spec.max_value - spec.min_value
                if range_size > 0:
                    vec.append((val - spec.min_value) / range_size)
                else:
                    vec.append(0.5)
        return vec

    def _generate_candidates(self, n: int) -> List[Dict[str, Any]]:
        """Generate random candidate parameter sets."""
        candidates = []
        for _ in range(n):
            params = {spec.name: spec.sample_random() for spec in self.param_specs}
            candidates.append(params)
        return candidates

    def _expected_improvement(self, x: List[float]) -> float:
        """
        Calculate expected improvement using a simplified surrogate.

        Uses a weighted nearest-neighbor model instead of a full GP
        to avoid heavy dependencies.
        """
        if len(self._X) < 3:
            return random.random()

        # Calculate distances to all observed points
        distances = []
        for i, xi in enumerate(self._X):
            dist = sum((a - b) ** 2 for a, b in zip(x, xi)) ** 0.5
            distances.append((dist, self._y[i]))

        # Sort by distance
        distances.sort(key=lambda d: d[0])

        # Use k-nearest neighbors for prediction
        k = min(5, len(distances))
        neighbors = distances[:k]

        # Weighted average prediction
        total_weight = 0.0
        predicted = 0.0
        for dist, y_val in neighbors:
            w = 1.0 / max(dist + 1e-10, 1e-10)
            predicted += w * y_val
            total_weight += w
        predicted /= total_weight

        # Estimate uncertainty from neighbor variance
        mean_y = predicted
        variance = 0.0
        for dist, y_val in neighbors:
            variance += (y_val - mean_y) ** 2
        variance /= max(1, k)
        std = max(variance ** 0.5, 1e-10)

        # Expected improvement
        improvement = predicted - self._best_y
        z = improvement / std

        # Simplified normal CDF approximation
        ei = improvement * self._normal_cdf(z) + std * self._normal_pdf(z)

        # Add exploration term
        ei += self.exploration_weight * std

        return ei

    @staticmethod
    def _normal_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(x))
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * math.exp(-0.5 * x * x) * (
            t * (0.3193815 + t * (-0.3565638 + t * (1.781478 +
            t * (-1.821256 + t * 1.330274))))
        )
        if x > 0:
            return 1.0 - p
        return p


# ---------------------------------------------------------------------------
# Genetic algorithm optimizer
# ---------------------------------------------------------------------------

class GeneticOptimizer(StrategyOptimizer):
    """
    Genetic algorithm optimization with crossover, mutation, and selection.

    Uses tournament selection, uniform crossover, and Gaussian mutation.
    """

    def __init__(self, param_specs: List[ParameterSpec],
                 population_size: int = 50,
                 generations: int = 30,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.2,
                 tournament_size: int = 3,
                 elitism_count: int = 2,
                 seed: Optional[int] = None, **kwargs):
        super().__init__(param_specs, **kwargs)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        if seed is not None:
            random.seed(seed)

    def _generate_param_sets(self) -> List[Dict[str, Any]]:
        # Initial population (generated but not returned - optimize() handles the loop)
        return [
            {spec.name: spec.sample_random() for spec in self.param_specs}
            for _ in range(self.population_size)
        ]

    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], OptimizationResult],
                 ) -> FullOptimizationResult:
        """Override optimize for generational evolution."""
        start_time = time.time()
        self.results = []
        self.best_result = None

        # Initialize population
        population = self._generate_param_sets()
        fitness: List[Tuple[Dict[str, Any], float]] = []

        total_evals = self.population_size * self.generations

        eval_count = 0
        for gen in range(self.generations):
            gen_fitness = []

            for params in population:
                try:
                    eval_start = time.time()
                    result = evaluate_fn(params)
                    result.execution_time = time.time() - eval_start
                    result.metric_value = self.objective.evaluate(result)
                    self.results.append(result)
                    gen_fitness.append((params, result.metric_value))

                    if self.best_result is None or \
                       result.metric_value > self.best_result.metric_value:
                        self.best_result = result

                    eval_count += 1
                except Exception as e:
                    gen_fitness.append((params, float('-inf')))
                    logger.error(f"  Gen {gen+1} evaluation error: {e}")

            # Sort by fitness
            gen_fitness.sort(key=lambda x: x[1], reverse=True)
            fitness = gen_fitness

            best_gen_fitness = gen_fitness[0][1] if gen_fitness else 0
            avg_fitness = sum(f[1] for f in gen_fitness if f[1] > float('-inf')) / \
                          max(1, sum(1 for f in gen_fitness if f[1] > float('-inf')))

            if self.verbose:
                logger.info(
                    f"  Gen {gen+1}/{self.generations}: "
                    f"best={best_gen_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}, "
                    f"total_evals={eval_count}"
                )

            # Last generation - no need to evolve
            if gen == self.generations - 1:
                break

            # Create next generation
            next_population = []

            # Elitism - keep best individuals
            for i in range(min(self.elitism_count, len(gen_fitness))):
                next_population.append(copy.deepcopy(gen_fitness[i][0]))

            # Fill rest with offspring
            while len(next_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_select(gen_fitness)
                parent2 = self._tournament_select(gen_fitness)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)

            population = next_population

        elapsed = time.time() - start_time
        stability = self._calculate_stability()

        return FullOptimizationResult(
            best_params=self.best_result.params if self.best_result else {},
            best_metric=self.best_result.metric_value if self.best_result else 0.0,
            all_results=self.results,
            total_combinations=total_evals,
            evaluated_combinations=len(self.results),
            total_time=elapsed,
            optimizer_type="GeneticOptimizer",
            parameter_specs=self.param_specs,
            stability_scores=stability,
            metadata={
                "generations": self.generations,
                "population_size": self.population_size,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
            },
        )

    def _tournament_select(self,
                            fitness: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Select an individual via tournament selection."""
        tournament = random.sample(
            fitness, min(self.tournament_size, len(fitness))
        )
        winner = max(tournament, key=lambda x: x[1])
        return copy.deepcopy(winner[0])

    def _crossover(self, parent1: Dict[str, Any],
                    parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover between two parents."""
        child1 = {}
        child2 = {}
        for spec in self.param_specs:
            if random.random() < 0.5:
                child1[spec.name] = parent1.get(spec.name, spec.sample_random())
                child2[spec.name] = parent2.get(spec.name, spec.sample_random())
            else:
                child1[spec.name] = parent2.get(spec.name, spec.sample_random())
                child2[spec.name] = parent1.get(spec.name, spec.sample_random())
        return child1, child2

    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Gaussian mutation to an individual."""
        mutated = individual.copy()
        for spec in self.param_specs:
            if random.random() < self.mutation_rate:
                current = mutated.get(spec.name, spec.sample_random())
                mutated[spec.name] = spec.mutate(current, self.mutation_strength)
        return mutated


# ---------------------------------------------------------------------------
# Multi-objective optimizer
# ---------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using NSGA-II style non-dominated sorting.

    Optimizes for multiple objectives simultaneously (e.g., maximize Sharpe
    AND minimize drawdown) and returns the Pareto front.
    """

    def __init__(self, param_specs: List[ParameterSpec],
                 objectives: List[ObjectiveFunction],
                 population_size: int = 50,
                 generations: int = 30,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.15,
                 seed: Optional[int] = None):
        self.param_specs = param_specs
        self.objectives = objectives
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        if seed is not None:
            random.seed(seed)

    def optimize(self, evaluate_fn: Callable[[Dict[str, Any]], OptimizationResult],
                 ) -> MultiObjectiveResult:
        """Run multi-objective optimization."""
        start_time = time.time()
        all_results = []

        # Initialize population
        population = [
            {spec.name: spec.sample_random() for spec in self.param_specs}
            for _ in range(self.population_size)
        ]

        for gen in range(self.generations):
            gen_results = []

            for params in population:
                try:
                    result = evaluate_fn(params)
                    # Evaluate all objectives
                    obj_values = {}
                    for obj in self.objectives:
                        obj_values[obj.name] = obj.evaluate(result)
                    result.metrics.update(obj_values)
                    gen_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"  Multi-obj evaluation error: {e}")

            if not gen_results:
                continue

            # Non-dominated sorting
            fronts = self._non_dominated_sort(gen_results)

            if self.verbose_log and fronts:
                logger.info(
                    f"  Gen {gen+1}/{self.generations}: "
                    f"front_size={len(fronts[0])}"
                )

            # Select next generation
            if gen < self.generations - 1:
                selected = self._select_next_generation(fronts)
                population = []
                for result in selected:
                    params = result.params.copy()
                    # Apply crossover and mutation
                    if len(selected) >= 2 and random.random() < self.crossover_rate:
                        partner = random.choice(selected)
                        params = self._crossover_params(params, partner.params)
                    params = self._mutate_params(params)
                    population.append(params)

                while len(population) < self.population_size:
                    params = {
                        spec.name: spec.sample_random()
                        for spec in self.param_specs
                    }
                    population.append(params)

        # Final Pareto front
        pareto = self._get_pareto_front(all_results)

        return MultiObjectiveResult(
            pareto_front=pareto,
            all_results=all_results,
            dominated_count=len(all_results) - len(pareto),
            non_dominated_count=len(pareto),
        )

    @property
    def verbose_log(self) -> bool:
        return True

    def _dominates(self, a: OptimizationResult,
                    b: OptimizationResult) -> bool:
        """Check if solution a dominates solution b."""
        at_least_one_better = False
        for obj in self.objectives:
            a_val = a.metrics.get(obj.name, 0)
            b_val = b.metrics.get(obj.name, 0)
            if a_val < b_val:
                return False
            if a_val > b_val:
                at_least_one_better = True
        return at_least_one_better

    def _non_dominated_sort(self, results: List[OptimizationResult]
                             ) -> List[List[OptimizationResult]]:
        """Sort results into non-dominated fronts."""
        n = len(results)
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(results[i], results[j]):
                    dominated_by[i].append(j)
                elif self._dominates(results[j], results[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(results[i])

        current_front = fronts[0]
        front_idx = {id(r): 0 for r in current_front}

        while current_front:
            next_front = []
            for r in current_front:
                r_idx = next(
                    i for i, res in enumerate(results) if id(res) == id(r)
                )
                for j in dominated_by[r_idx]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(results[j])
            if next_front:
                fronts.append(next_front)
            current_front = next_front

        return fronts

    def _get_pareto_front(self, results: List[OptimizationResult]
                           ) -> List[OptimizationResult]:
        """Extract the Pareto front from all results."""
        if not results:
            return []
        fronts = self._non_dominated_sort(results)
        return fronts[0] if fronts else []

    def _select_next_generation(self, fronts: List[List[OptimizationResult]]
                                 ) -> List[OptimizationResult]:
        """Select individuals for the next generation from fronts."""
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                remaining = self.population_size - len(selected)
                # Crowding distance selection
                front_sorted = self._sort_by_crowding(front)
                selected.extend(front_sorted[:remaining])
                break
        return selected

    def _sort_by_crowding(self, front: List[OptimizationResult]
                           ) -> List[OptimizationResult]:
        """Sort a front by crowding distance (prefer diverse solutions)."""
        n = len(front)
        if n <= 2:
            return front

        distances = [0.0] * n

        for obj in self.objectives:
            vals = [(i, front[i].metrics.get(obj.name, 0)) for i in range(n)]
            vals.sort(key=lambda x: x[1])

            distances[vals[0][0]] = float('inf')
            distances[vals[-1][0]] = float('inf')

            obj_range = vals[-1][1] - vals[0][1]
            if obj_range < 1e-10:
                continue

            for k in range(1, n - 1):
                distances[vals[k][0]] += (
                    (vals[k + 1][1] - vals[k - 1][1]) / obj_range
                )

        indexed = list(zip(range(n), distances))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [front[i] for i, _ in indexed]

    def _crossover_params(self, p1: Dict[str, Any],
                           p2: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for spec in self.param_specs:
            if random.random() < 0.5:
                child[spec.name] = p1.get(spec.name, spec.sample_random())
            else:
                child[spec.name] = p2.get(spec.name, spec.sample_random())
        return child

    def _mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mutated = params.copy()
        for spec in self.param_specs:
            if random.random() < self.mutation_rate:
                current = mutated.get(spec.name, spec.sample_random())
                mutated[spec.name] = spec.mutate(current, 0.2)
        return mutated


# ---------------------------------------------------------------------------
# Overfitting detector
# ---------------------------------------------------------------------------

class OverfittingDetector:
    """
    Detects potential overfitting by comparing in-sample and out-of-sample
    performance across optimization results.
    """

    @staticmethod
    def detect(in_sample_results: List[OptimizationResult],
               out_of_sample_results: List[OptimizationResult],
               metric_name: str = "sharpe_ratio") -> Dict[str, Any]:
        """
        Compare in-sample vs out-of-sample performance.

        Returns overfitting metrics including degradation ratio,
        correlation, and a composite overfitting score.
        """
        if not in_sample_results or not out_of_sample_results:
            return {"error": "Need both IS and OOS results"}

        is_metrics = []
        oos_metrics = []

        for is_res, oos_res in zip(in_sample_results, out_of_sample_results):
            is_val = getattr(is_res, metric_name, is_res.metric_value)
            oos_val = getattr(oos_res, metric_name, oos_res.metric_value)
            is_metrics.append(is_val)
            oos_metrics.append(oos_val)

        n = len(is_metrics)
        if n == 0:
            return {"error": "No matching results"}

        # Average IS and OOS performance
        avg_is = sum(is_metrics) / n
        avg_oos = sum(oos_metrics) / n

        # Degradation ratio
        degradation = 0.0
        if avg_is != 0:
            degradation = 1.0 - (avg_oos / avg_is)

        # Correlation between IS and OOS rankings
        is_ranks = OverfittingDetector._rank(is_metrics)
        oos_ranks = OverfittingDetector._rank(oos_metrics)
        correlation = OverfittingDetector._spearman_correlation(is_ranks, oos_ranks)

        # Percentage of IS-positive results that are OOS-negative
        positive_is = sum(1 for v in is_metrics if v > 0)
        positive_oos = sum(
            1 for is_v, oos_v in zip(is_metrics, oos_metrics)
            if is_v > 0 and oos_v > 0
        )
        false_positive_rate = 0.0
        if positive_is > 0:
            false_positive_rate = 1.0 - (positive_oos / positive_is)

        # Composite overfitting score (0 = no overfitting, 1 = severe)
        score = 0.0
        if degradation > 0:
            score += min(degradation, 1.0) * 0.4
        if correlation < 1.0:
            score += (1.0 - max(correlation, -1.0)) * 0.3
        score += false_positive_rate * 0.3
        score = min(1.0, max(0.0, score))

        return {
            "overfitting_score": score,
            "degradation_ratio": degradation,
            "rank_correlation": correlation,
            "false_positive_rate": false_positive_rate,
            "avg_in_sample": avg_is,
            "avg_out_of_sample": avg_oos,
            "n_comparisons": n,
            "severity": (
                "low" if score < 0.3 else
                "moderate" if score < 0.6 else
                "high" if score < 0.8 else
                "severe"
            ),
        }

    @staticmethod
    def _rank(values: List[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * len(values)
        for rank, (idx, _) in enumerate(indexed):
            ranks[idx] = float(rank)
        return ranks

    @staticmethod
    def _spearman_correlation(ranks1: List[float],
                               ranks2: List[float]) -> float:
        n = len(ranks1)
        if n < 2:
            return 0.0
        d_squared = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))
        return 1.0 - (6 * d_squared) / (n * (n ** 2 - 1))


# ---------------------------------------------------------------------------
# Parameter sensitivity analyzer
# ---------------------------------------------------------------------------

class ParameterSensitivityAnalyzer:
    """
    Analyzes how sensitive the objective function is to small changes
    in each parameter.
    """

    @staticmethod
    def analyze(results: List[OptimizationResult],
                param_specs: List[ParameterSpec],
                metric_name: str = "metric_value") -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter sensitivity.

        Returns a dict of parameter_name -> {
            "sensitivity": float (0-1, how much output changes with input),
            "monotonicity": float (-1 to 1, direction of relationship),
            "optimal_region_start": float,
            "optimal_region_end": float,
        }
        """
        analysis = {}

        for spec in param_specs:
            if spec.param_type == "categorical":
                analysis[spec.name] = ParameterSensitivityAnalyzer._analyze_categorical(
                    results, spec, metric_name
                )
            else:
                analysis[spec.name] = ParameterSensitivityAnalyzer._analyze_numeric(
                    results, spec, metric_name
                )

        return analysis

    @staticmethod
    def _analyze_numeric(results: List[OptimizationResult],
                          spec: ParameterSpec,
                          metric_name: str) -> Dict[str, float]:
        pairs = []
        for r in results:
            val = r.params.get(spec.name)
            metric = getattr(r, metric_name, r.metric_value)
            if val is not None:
                pairs.append((float(val), metric))

        if len(pairs) < 3:
            return {"sensitivity": 0.0, "monotonicity": 0.0}

        pairs.sort(key=lambda x: x[0])
        param_vals = [p[0] for p in pairs]
        metric_vals = [p[1] for p in pairs]

        # Calculate sensitivity (normalized variance of metric across param range)
        param_range = max(param_vals) - min(param_vals)
        metric_range = max(metric_vals) - min(metric_vals) if metric_vals else 0
        mean_metric = sum(metric_vals) / len(metric_vals)
        metric_var = sum((m - mean_metric) ** 2 for m in metric_vals) / len(metric_vals)

        sensitivity = 0.0
        if abs(mean_metric) > 1e-10:
            sensitivity = min(1.0, (metric_var ** 0.5) / abs(mean_metric))

        # Calculate monotonicity (Spearman correlation)
        n = len(pairs)
        param_ranks = list(range(n))  # already sorted
        metric_ranked = ParameterSensitivityAnalyzer._rank_values(metric_vals)
        d_sq = sum((pr - mr) ** 2 for pr, mr in zip(param_ranks, metric_ranked))
        monotonicity = 1.0 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0.0

        # Find optimal region (top 20% by metric)
        sorted_by_metric = sorted(pairs, key=lambda x: x[1], reverse=True)
        top_n = max(1, len(sorted_by_metric) // 5)
        top_params = [p[0] for p in sorted_by_metric[:top_n]]

        return {
            "sensitivity": sensitivity,
            "monotonicity": monotonicity,
            "optimal_region_start": min(top_params),
            "optimal_region_end": max(top_params),
            "param_range": param_range,
            "metric_range": metric_range,
        }

    @staticmethod
    def _analyze_categorical(results: List[OptimizationResult],
                              spec: ParameterSpec,
                              metric_name: str) -> Dict[str, float]:
        groups: Dict[Any, List[float]] = defaultdict(list)
        for r in results:
            val = r.params.get(spec.name)
            metric = getattr(r, metric_name, r.metric_value)
            if val is not None:
                groups[val].append(metric)

        if len(groups) < 2:
            return {"sensitivity": 0.0, "monotonicity": 0.0}

        group_means = {k: sum(v) / len(v) for k, v in groups.items()}
        overall_mean = sum(group_means.values()) / len(group_means)

        between_var = sum(
            (m - overall_mean) ** 2 for m in group_means.values()
        ) / len(group_means)

        sensitivity = min(1.0, (between_var ** 0.5) / max(abs(overall_mean), 1e-10))

        best_category = max(group_means, key=group_means.get)

        return {
            "sensitivity": sensitivity,
            "monotonicity": 0.0,  # not applicable for categorical
            "best_category": best_category,
            "group_means": group_means,
        }

    @staticmethod
    def _rank_values(values: List[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * len(values)
        for rank, (idx, _) in enumerate(indexed):
            ranks[idx] = float(rank)
        return ranks
