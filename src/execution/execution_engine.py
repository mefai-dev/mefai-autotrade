"""
Mefai Autotrade - Execution Engine Orchestrator
Central coordination layer that routes orders through the smart router,
applies execution algorithms (TWAP, VWAP, Iceberg, Sniper), monitors
fills, handles retries and partial fills, and collects performance metrics.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .smart_router import SmartRouter, RoutingDecision, RouteType
from .twap import TWAPExecutor, TWAPConfig, TWAPProgress
from .vwap import VWAPExecutor, VWAPConfig, VWAPProgress
from .iceberg import IcebergExecutor, IcebergConfig, IcebergProgress
from .sniper import SniperExecutor, SniperConfig, SniperResult, SniperTarget
from .market_impact import MarketImpactEstimator, ImpactModel
from .order_builder import OrderBuilder, OrderParams, ExchangeRules
from .fill_analyzer import FillAnalyzer, FillRecord

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Available execution algorithms."""
    DIRECT = "direct"        # Send directly to exchange
    TWAP = "twap"            # Time-weighted average price
    VWAP = "vwap"            # Volume-weighted average price
    ICEBERG = "iceberg"      # Hidden iceberg order
    SNIPER = "sniper"        # Precision entry at specific level
    SMART = "smart"          # Auto-select based on order characteristics


class ExecutionStatus(Enum):
    """Status of an execution request."""
    PENDING = "pending"
    ROUTING = "routing"
    EXECUTING = "executing"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMED_OUT = "timed_out"


@dataclass
class ExecutionRequest:
    """Request to execute an order through the engine."""
    request_id: str = ""
    symbol: str = ""
    side: str = ""               # BUY or SELL
    quantity: float = 0.0
    price: float = 0.0           # 0 = market
    order_type: str = "LIMIT"
    time_in_force: str = "GTC"
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.DIRECT
    venue_id: Optional[str] = None  # None = auto-route

    # Algorithm-specific parameters
    algo_params: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    # TWAP: {"duration_seconds": 300, "num_slices": 10, "jitter_pct": 0.1}
    # VWAP: {"duration_seconds": 600, "mode": "aggressive"}
    # Iceberg: {"visible_quantity": 1.0, "passive_only": True}
    # Sniper: {"targets": [...], "max_wait_seconds": 120}

    # Risk parameters
    max_slippage_bps: float = 0.0    # 0 = no limit
    price_limit: float = 0.0         # Max price for BUY, min for SELL
    max_execution_time_seconds: float = 0.0  # 0 = no limit
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    reduce_only: bool = False

    # Metadata
    strategy_id: str = ""
    signal_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"exec_{uuid.uuid4().hex[:10]}"


@dataclass
class ExecutionResult:
    """Result of an execution request."""
    request_id: str
    request: ExecutionRequest
    status: ExecutionStatus = ExecutionStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    total_notional: float = 0.0
    total_commission: float = 0.0
    slippage_bps: float = 0.0
    fill_rate: float = 0.0
    order_ids: List[str] = field(default_factory=list)
    venue_id: str = ""
    routing_decision: Optional[RoutingDecision] = None
    algorithm_used: str = ""
    algo_execution_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time_ms: float = 0.0
    retries: int = 0
    error_message: str = ""
    fills: List[FillRecord] = field(default_factory=list)
    impact_estimate_bps: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_from_fills(self) -> None:
        """Recalculate metrics from individual fills."""
        if not self.fills:
            return
        self.filled_quantity = sum(f.filled_quantity for f in self.fills)
        self.total_notional = sum(f.notional_value for f in self.fills)
        self.total_commission = sum(f.commission for f in self.fills)
        if self.filled_quantity > 0:
            self.avg_fill_price = self.total_notional / self.filled_quantity
        if self.request.quantity > 0:
            self.fill_rate = self.filled_quantity / self.request.quantity
        if self.request.price > 0 and self.avg_fill_price > 0:
            if self.request.side == "BUY":
                self.slippage_bps = (
                    (self.avg_fill_price - self.request.price) / self.request.price * 10000.0
                )
            else:
                self.slippage_bps = (
                    (self.request.price - self.avg_fill_price) / self.request.price * 10000.0
                )

    @property
    def is_complete(self) -> bool:
        return self.status in (
            ExecutionStatus.FILLED, ExecutionStatus.CANCELLED,
            ExecutionStatus.FAILED, ExecutionStatus.TIMED_OUT,
        )


@dataclass
class EngineMetrics:
    """Performance metrics for the execution engine."""
    total_requests: int = 0
    total_filled: int = 0
    total_failed: int = 0
    total_retried: int = 0
    total_volume_usd: float = 0.0
    avg_execution_time_ms: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_fill_rate: float = 0.0
    by_algorithm: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    uptime_seconds: float = 0.0


class ExecutionEngine:
    """Central execution engine orchestrator.

    Coordinates order routing, execution algorithm selection,
    fill monitoring, retry handling, and performance tracking.
    """

    def __init__(self):
        # Component modules
        self._router = SmartRouter()
        self._twap = TWAPExecutor()
        self._vwap = VWAPExecutor()
        self._iceberg = IcebergExecutor()
        self._sniper = SniperExecutor()
        self._impact = MarketImpactEstimator()
        self._order_builder = OrderBuilder()
        self._fill_analyzer = FillAnalyzer()

        # Direct order execution function (must be set by user)
        self._order_executor: Optional[Callable] = None
        self._order_canceller: Optional[Callable] = None
        self._price_fetcher: Optional[Callable] = None
        self._volume_fetcher: Optional[Callable] = None
        self._depth_fetcher: Optional[Callable] = None
        self._spread_fetcher: Optional[Callable] = None

        # State
        self._active_requests: Dict[str, ExecutionResult] = {}
        self._completed_requests: deque = deque(maxlen=10000)
        self._callbacks: Dict[str, List[Callable]] = {}
        self._start_time = time.time()
        self._metrics = EngineMetrics()
        self._lock = asyncio.Lock()
        self._running = False

        # Thresholds for smart algorithm selection
        self._smart_algo_thresholds = {
            "twap_min_notional": 50000.0,   # Use TWAP above $50k
            "vwap_min_notional": 100000.0,  # Use VWAP above $100k
            "iceberg_min_notional": 25000.0,  # Use iceberg above $25k
            "max_immediate_notional": 10000.0,  # Direct below $10k
        }

        logger.info("ExecutionEngine initialized")

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    def set_order_executor(self, executor: Callable) -> None:
        """Set the direct order execution function.
        Signature: async def execute(symbol, side, quantity, order_type,
                                     price, time_in_force, venue_id) -> dict
        Returns: {"order_id": str, "filled_quantity": float,
                  "fill_price": float, "status": str, "commission": float}
        """
        self._order_executor = executor
        self._twap.set_order_executor(executor)
        self._vwap.set_order_executor(executor)
        self._iceberg.set_order_executor(executor)
        self._sniper.set_order_executor(executor)

    def set_order_canceller(self, canceller: Callable) -> None:
        """Set order cancel function."""
        self._order_canceller = canceller
        self._iceberg.set_order_canceller(canceller)

    def set_price_fetcher(self, fetcher: Callable) -> None:
        """Set price fetcher function."""
        self._price_fetcher = fetcher
        self._twap.set_price_fetcher(fetcher)
        self._vwap.set_price_fetcher(fetcher)
        self._iceberg.set_price_fetcher(fetcher)
        self._sniper.set_price_fetcher(fetcher)

    def set_volume_fetcher(self, fetcher: Callable) -> None:
        """Set volume fetcher function."""
        self._volume_fetcher = fetcher
        self._twap.set_volume_fetcher(fetcher)
        self._vwap.set_volume_fetcher(fetcher)

    def set_depth_fetcher(self, fetcher: Callable) -> None:
        """Set order book depth fetcher."""
        self._depth_fetcher = fetcher
        self._sniper.set_depth_fetcher(fetcher)

    def set_spread_fetcher(self, fetcher: Callable) -> None:
        """Set spread fetcher."""
        self._spread_fetcher = fetcher
        self._iceberg.set_spread_fetcher(fetcher)
        self._sniper.set_spread_fetcher(fetcher)

    def set_exchange_rules(self, symbol: str, rules: ExchangeRules) -> None:
        """Set exchange trading rules for a symbol."""
        self._order_builder.set_rules(symbol, rules)

    def set_smart_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set thresholds for smart algorithm selection."""
        self._smart_algo_thresholds.update(thresholds)

    @property
    def router(self) -> SmartRouter:
        """Access the smart router for venue management."""
        return self._router

    @property
    def order_builder(self) -> OrderBuilder:
        """Access the order builder."""
        return self._order_builder

    @property
    def fill_analyzer(self) -> FillAnalyzer:
        """Access the fill analyzer."""
        return self._fill_analyzer

    @property
    def impact_estimator(self) -> MarketImpactEstimator:
        """Access the market impact estimator."""
        return self._impact

    # -----------------------------------------------------------------------
    # Core execution
    # -----------------------------------------------------------------------

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute an order request.

        This is the main entry point. It will:
        1. Validate the request
        2. Route to best venue(s)
        3. Estimate market impact
        4. Select and run execution algorithm
        5. Monitor fills
        6. Handle retries
        7. Record results

        Returns ExecutionResult which is updated in real-time.
        """
        result = ExecutionResult(
            request_id=request.request_id,
            request=request,
            start_time=time.time(),
        )

        async with self._lock:
            self._active_requests[request.request_id] = result
            self._metrics.total_requests += 1

        logger.info(
            "Execution request: id=%s symbol=%s side=%s qty=%.6f algo=%s",
            request.request_id, request.symbol, request.side,
            request.quantity, request.algorithm.value,
        )

        try:
            # Step 1: Validate
            valid, errors = self._validate_request(request)
            if not valid:
                result.status = ExecutionStatus.FAILED
                result.error_message = "; ".join(errors)
                logger.warning("Request %s validation failed: %s", request.request_id, result.error_message)
                self._finalize_result(result)
                return result

            # Step 2: Round parameters
            self._round_order_params(request)

            # Step 3: Route
            result.status = ExecutionStatus.ROUTING
            routing = await self._route_order(request)
            result.routing_decision = routing
            if routing and routing.venues:
                result.venue_id = routing.primary_venue.venue_id if routing.primary_venue else ""
            else:
                # No venue available - try direct if venue_id specified
                if not request.venue_id:
                    result.status = ExecutionStatus.FAILED
                    result.error_message = "No venue available for routing"
                    self._finalize_result(result)
                    return result

            # Step 4: Estimate impact
            impact_bps = await self._estimate_impact(request)
            result.impact_estimate_bps = impact_bps

            # Step 5: Select algorithm
            algorithm = self._select_algorithm(request, impact_bps)
            result.algorithm_used = algorithm.value

            # Step 6: Execute
            result.status = ExecutionStatus.EXECUTING
            await self._execute_with_algorithm(request, result, algorithm)

            # Step 7: Finalize
            result.update_from_fills()
            if result.fill_rate >= 0.99:
                result.status = ExecutionStatus.FILLED
            elif result.filled_quantity > 0:
                result.status = ExecutionStatus.PARTIALLY_FILLED
            elif result.status == ExecutionStatus.EXECUTING:
                result.status = ExecutionStatus.FAILED

        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMED_OUT
            result.error_message = "Execution timed out"
            logger.warning("Request %s timed out", request.request_id)

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            logger.error("Request %s failed: %s", request.request_id, str(e))

        self._finalize_result(result)
        return result

    def _validate_request(self, request: ExecutionRequest) -> Tuple[bool, List[str]]:
        """Validate execution request parameters."""
        errors = []
        if not request.symbol:
            errors.append("Symbol is required")
        if request.side not in ("BUY", "SELL"):
            errors.append(f"Invalid side: {request.side}")
        if request.quantity <= 0:
            errors.append("Quantity must be positive")
        if request.order_type not in ("LIMIT", "MARKET", "STOP_MARKET",
                                       "STOP_LIMIT", "TAKE_PROFIT_MARKET"):
            errors.append(f"Invalid order type: {request.order_type}")

        # Exchange rule validation
        rules = self._order_builder.get_rules(request.symbol)
        if rules:
            valid, rule_errors = rules.validate_order(
                request.side, request.order_type,
                request.quantity, request.price
            )
            errors.extend(rule_errors)

        return len(errors) == 0, errors

    def _round_order_params(self, request: ExecutionRequest) -> None:
        """Round order parameters to exchange precision."""
        rules = self._order_builder.get_rules(request.symbol)
        if rules:
            request.quantity = rules.round_quantity(request.quantity)
            if request.price > 0:
                request.price = rules.round_price(request.price)
            if request.price_limit > 0:
                request.price_limit = rules.round_price(request.price_limit)

    async def _route_order(self, request: ExecutionRequest) -> Optional[RoutingDecision]:
        """Route order through smart router."""
        if request.venue_id:
            # Explicit venue specified - skip routing
            return None

        estimated_value = request.quantity * (request.price or 0)
        routing = await self._router.route(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            price=request.price,
            order_value_usd=estimated_value,
        )
        return routing

    async def _estimate_impact(self, request: ExecutionRequest) -> float:
        """Estimate market impact for the order."""
        # This would normally fetch daily volume and volatility
        # For now, return a basic estimate
        return 0.0

    def _select_algorithm(self, request: ExecutionRequest,
                          impact_bps: float) -> ExecutionAlgorithm:
        """Select the best execution algorithm."""
        if request.algorithm != ExecutionAlgorithm.SMART:
            return request.algorithm

        # Smart selection based on order characteristics
        estimated_notional = request.quantity * (request.price or 0)
        thresholds = self._smart_algo_thresholds

        if estimated_notional <= thresholds.get("max_immediate_notional", 10000):
            return ExecutionAlgorithm.DIRECT

        if estimated_notional >= thresholds.get("vwap_min_notional", 100000):
            return ExecutionAlgorithm.VWAP

        if estimated_notional >= thresholds.get("twap_min_notional", 50000):
            return ExecutionAlgorithm.TWAP

        if estimated_notional >= thresholds.get("iceberg_min_notional", 25000):
            return ExecutionAlgorithm.ICEBERG

        return ExecutionAlgorithm.DIRECT

    async def _execute_with_algorithm(self, request: ExecutionRequest,
                                       result: ExecutionResult,
                                       algorithm: ExecutionAlgorithm) -> None:
        """Execute using the selected algorithm."""
        venue_id = request.venue_id
        if not venue_id and result.routing_decision and result.routing_decision.primary_venue:
            venue_id = result.routing_decision.primary_venue.venue_id

        if algorithm == ExecutionAlgorithm.DIRECT:
            await self._execute_direct(request, result, venue_id)
        elif algorithm == ExecutionAlgorithm.TWAP:
            await self._execute_twap(request, result, venue_id)
        elif algorithm == ExecutionAlgorithm.VWAP:
            await self._execute_vwap(request, result, venue_id)
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            await self._execute_iceberg(request, result, venue_id)
        elif algorithm == ExecutionAlgorithm.SNIPER:
            await self._execute_sniper(request, result, venue_id)
        else:
            await self._execute_direct(request, result, venue_id)

    async def _execute_direct(self, request: ExecutionRequest,
                               result: ExecutionResult,
                               venue_id: Optional[str]) -> None:
        """Execute order directly."""
        if not self._order_executor:
            result.status = ExecutionStatus.FAILED
            result.error_message = "No order executor configured"
            return

        retries = 0
        remaining_qty = request.quantity

        while retries <= request.max_retries and remaining_qty > 0:
            try:
                start = time.time()
                exec_result = await self._order_executor(
                    symbol=request.symbol,
                    side=request.side,
                    quantity=remaining_qty,
                    order_type=request.order_type,
                    price=request.price,
                    time_in_force=request.time_in_force,
                    venue_id=venue_id,
                )
                latency_ms = (time.time() - start) * 1000.0

                if not exec_result:
                    retries += 1
                    result.retries = retries
                    result.status = ExecutionStatus.RETRYING
                    await asyncio.sleep(request.retry_delay_seconds)
                    continue

                status = exec_result.get("status", "")
                filled_qty = float(exec_result.get("filled_quantity", 0))
                fill_price = float(exec_result.get("fill_price", 0))
                order_id = exec_result.get("order_id", "")
                commission = float(exec_result.get("commission", 0))

                if order_id:
                    result.order_ids.append(order_id)

                if filled_qty > 0:
                    fill = FillRecord(
                        fill_id=f"{request.request_id}_f{len(result.fills)}",
                        order_id=order_id,
                        symbol=request.symbol,
                        side=request.side,
                        order_type=request.order_type,
                        requested_quantity=remaining_qty,
                        filled_quantity=filled_qty,
                        requested_price=request.price,
                        fill_price=fill_price,
                        timestamp=time.time(),
                        venue_id=venue_id or "",
                        is_maker=exec_result.get("is_maker", False),
                        commission=commission,
                        fill_time_ms=latency_ms,
                        execution_algo="direct",
                    )
                    result.fills.append(fill)
                    self._fill_analyzer.record_fill(fill)
                    remaining_qty -= filled_qty

                    # Record with router
                    if venue_id:
                        slippage = fill.slippage_bps
                        self._router.record_execution(
                            venue_id=venue_id,
                            symbol=request.symbol,
                            side=request.side,
                            order_type=request.order_type,
                            latency_ms=latency_ms,
                            filled=True,
                            slippage_bps=slippage,
                            expected_price=request.price,
                            actual_price=fill_price,
                            partial=filled_qty < remaining_qty + filled_qty,
                        )

                if status in ("FILLED", "filled") or remaining_qty <= 0:
                    break

                if status in ("REJECTED", "EXPIRED", "rejected", "expired"):
                    if venue_id:
                        self._router.record_execution(
                            venue_id=venue_id,
                            symbol=request.symbol,
                            side=request.side,
                            order_type=request.order_type,
                            latency_ms=latency_ms,
                            filled=False,
                        )
                    retries += 1
                    result.retries = retries
                    result.status = ExecutionStatus.RETRYING
                    await asyncio.sleep(request.retry_delay_seconds)
                    continue

                break

            except Exception as e:
                retries += 1
                result.retries = retries
                result.error_message = str(e)
                logger.error(
                    "Direct execution error (attempt %d/%d): %s",
                    retries, request.max_retries + 1, str(e),
                )
                if retries <= request.max_retries:
                    result.status = ExecutionStatus.RETRYING
                    await asyncio.sleep(request.retry_delay_seconds)

    async def _execute_twap(self, request: ExecutionRequest,
                             result: ExecutionResult,
                             venue_id: Optional[str]) -> None:
        """Execute using TWAP algorithm."""
        params = request.algo_params
        config = TWAPConfig(
            symbol=request.symbol,
            side=request.side,
            total_quantity=request.quantity,
            duration_seconds=params.get("duration_seconds", 300),
            num_slices=params.get("num_slices", 0),
            price_limit=request.price_limit or params.get("price_limit", 0),
            jitter_pct=params.get("jitter_pct", 0.10),
            volume_participation_limit=params.get("volume_participation_limit", 0),
            order_type=params.get("order_type", request.order_type),
            limit_offset_bps=params.get("limit_offset_bps", 5.0),
            cancel_on_price_move_pct=params.get("cancel_on_price_move_pct", 0),
            time_in_force=params.get("time_in_force", "IOC"),
            venue_id=venue_id,
        )

        progress = await self._twap.execute(config)
        result.algo_execution_id = progress.execution_id

        # Wait for completion
        timeout = request.max_execution_time_seconds or config.duration_seconds * 1.5
        await self._wait_for_algo_completion(
            lambda: self._twap.get_progress(progress.execution_id),
            lambda p: p.status.value in ("completed", "cancelled", "failed", "expired"),
            timeout,
        )

        # Collect fills from progress
        final = self._twap.get_progress(progress.execution_id)
        if final:
            for s in final.slices:
                if s.filled_quantity > 0:
                    fill = FillRecord(
                        fill_id=f"{request.request_id}_twap_{s.index}",
                        order_id=s.order_id or "",
                        symbol=request.symbol,
                        side=request.side,
                        order_type=config.order_type,
                        requested_quantity=s.target_quantity,
                        filled_quantity=s.filled_quantity,
                        requested_price=request.price,
                        fill_price=s.fill_price,
                        timestamp=s.actual_time or time.time(),
                        venue_id=venue_id or "",
                        fill_time_ms=s.latency_ms,
                        execution_algo="twap",
                    )
                    result.fills.append(fill)
                    self._fill_analyzer.record_fill(fill)

    async def _execute_vwap(self, request: ExecutionRequest,
                             result: ExecutionResult,
                             venue_id: Optional[str]) -> None:
        """Execute using VWAP algorithm."""
        params = request.algo_params
        config = VWAPConfig(
            symbol=request.symbol,
            side=request.side,
            total_quantity=request.quantity,
            duration_seconds=params.get("duration_seconds", 600),
            num_slices=params.get("num_slices", 0),
            price_limit=request.price_limit or params.get("price_limit", 0),
            volume_participation_limit=params.get("volume_participation_limit", 0.05),
            order_type=params.get("order_type", request.order_type),
            limit_offset_bps=params.get("limit_offset_bps", 3.0),
            time_in_force=params.get("time_in_force", "IOC"),
            venue_id=venue_id,
        )

        progress = await self._vwap.execute(config)
        result.algo_execution_id = progress.execution_id

        timeout = request.max_execution_time_seconds or config.duration_seconds * 1.5
        await self._wait_for_algo_completion(
            lambda: self._vwap.get_progress(progress.execution_id),
            lambda p: p.status.value in ("completed", "cancelled", "failed"),
            timeout,
        )

        final = self._vwap.get_progress(progress.execution_id)
        if final:
            for s in final.slices:
                if s.filled_quantity > 0:
                    fill = FillRecord(
                        fill_id=f"{request.request_id}_vwap_{s.index}",
                        order_id=s.order_id or "",
                        symbol=request.symbol,
                        side=request.side,
                        order_type=config.order_type,
                        requested_quantity=s.target_quantity,
                        filled_quantity=s.filled_quantity,
                        requested_price=request.price,
                        fill_price=s.fill_price,
                        timestamp=s.actual_time or time.time(),
                        venue_id=venue_id or "",
                        fill_time_ms=s.latency_ms,
                        execution_algo="vwap",
                    )
                    result.fills.append(fill)
                    self._fill_analyzer.record_fill(fill)

    async def _execute_iceberg(self, request: ExecutionRequest,
                                result: ExecutionResult,
                                venue_id: Optional[str]) -> None:
        """Execute using iceberg algorithm."""
        params = request.algo_params
        visible = params.get("visible_quantity", request.quantity * 0.1)
        config = IcebergConfig(
            symbol=request.symbol,
            side=request.side,
            total_quantity=request.quantity,
            visible_quantity=visible,
            visible_variation_pct=params.get("visible_variation_pct", 0.15),
            price=request.price,
            follow_market=params.get("follow_market", True),
            passive_only=params.get("passive_only", False),
            max_execution_time_seconds=request.max_execution_time_seconds or 3600,
            venue_id=venue_id,
        )

        progress = await self._iceberg.execute(config)
        result.algo_execution_id = progress.execution_id

        timeout = request.max_execution_time_seconds or config.max_execution_time_seconds * 1.2
        await self._wait_for_algo_completion(
            lambda: self._iceberg.get_progress(progress.execution_id),
            lambda p: p.status.value in ("completed", "cancelled", "failed", "expired"),
            timeout,
        )

        final = self._iceberg.get_progress(progress.execution_id)
        if final:
            for r in final.refreshes:
                if r.filled_quantity > 0:
                    fill = FillRecord(
                        fill_id=f"{request.request_id}_ice_{r.index}",
                        order_id=r.order_id or "",
                        symbol=request.symbol,
                        side=request.side,
                        order_type=config.order_type,
                        requested_quantity=r.visible_quantity,
                        filled_quantity=r.filled_quantity,
                        requested_price=request.price,
                        fill_price=r.fill_price,
                        timestamp=r.timestamp,
                        venue_id=venue_id or "",
                        fill_time_ms=r.latency_ms,
                        execution_algo="iceberg",
                    )
                    result.fills.append(fill)
                    self._fill_analyzer.record_fill(fill)

    async def _execute_sniper(self, request: ExecutionRequest,
                               result: ExecutionResult,
                               venue_id: Optional[str]) -> None:
        """Execute using sniper algorithm."""
        params = request.algo_params
        targets = []
        for i, t in enumerate(params.get("targets", [])):
            targets.append(SniperTarget(
                target_id=f"{request.request_id}_t{i}",
                price=t.get("price", request.price),
                quantity=t.get("quantity", request.quantity),
                direction=t.get("direction", "below" if request.side == "BUY" else "above"),
            ))

        if not targets and request.price > 0:
            targets.append(SniperTarget(
                target_id=f"{request.request_id}_t0",
                price=request.price,
                quantity=request.quantity,
                direction="below" if request.side == "BUY" else "above",
            ))

        config = SniperConfig(
            symbol=request.symbol,
            side=request.side,
            targets=targets,
            total_quantity=request.quantity,
            order_type=params.get("order_type", "LIMIT"),
            time_in_force=params.get("time_in_force", "IOC"),
            max_wait_seconds=params.get("max_wait_seconds", request.max_execution_time_seconds or 120),
            venue_id=venue_id,
        )

        sniper_result = await self._sniper.arm(config)
        result.algo_execution_id = sniper_result.execution_id

        timeout = config.max_wait_seconds * 1.2 if config.max_wait_seconds > 0 else 300
        await self._wait_for_algo_completion(
            lambda: self._sniper.get_result(sniper_result.execution_id),
            lambda r: r.status.value in ("filled", "partial", "expired", "cancelled", "failed", "disarmed"),
            timeout,
        )

        final = self._sniper.get_result(sniper_result.execution_id)
        if final:
            for target in config.targets:
                if target.filled and target.filled_quantity > 0:
                    fill = FillRecord(
                        fill_id=f"{request.request_id}_snp_{target.target_id}",
                        order_id=target.order_id or "",
                        symbol=request.symbol,
                        side=request.side,
                        order_type=config.order_type,
                        requested_quantity=target.quantity,
                        filled_quantity=target.filled_quantity,
                        requested_price=target.price,
                        fill_price=target.fill_price,
                        timestamp=target.fill_time,
                        venue_id=venue_id or "",
                        execution_algo="sniper",
                    )
                    result.fills.append(fill)
                    self._fill_analyzer.record_fill(fill)

    async def _wait_for_algo_completion(self, get_progress: Callable,
                                         is_done: Callable,
                                         timeout_seconds: float) -> None:
        """Wait for an algorithm execution to complete."""
        start = time.time()
        poll_interval = 0.5

        while time.time() - start < timeout_seconds:
            progress = get_progress()
            if progress and is_done(progress):
                return
            await asyncio.sleep(poll_interval)
            # Increase poll interval over time
            if time.time() - start > 30:
                poll_interval = 1.0
            if time.time() - start > 120:
                poll_interval = 2.0

    def _finalize_result(self, result: ExecutionResult) -> None:
        """Finalize an execution result."""
        result.end_time = time.time()
        result.execution_time_ms = (result.end_time - result.start_time) * 1000.0
        result.update_from_fills()

        # Update metrics
        self._metrics.total_volume_usd += result.total_notional
        if result.status == ExecutionStatus.FILLED:
            self._metrics.total_filled += 1
        elif result.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMED_OUT):
            self._metrics.total_failed += 1
        if result.retries > 0:
            self._metrics.total_retried += 1

        # Rolling averages
        n = self._metrics.total_requests
        if n > 0:
            self._metrics.avg_execution_time_ms = (
                (self._metrics.avg_execution_time_ms * (n - 1) + result.execution_time_ms) / n
            )
            self._metrics.avg_slippage_bps = (
                (self._metrics.avg_slippage_bps * (n - 1) + result.slippage_bps) / n
            )
            self._metrics.avg_fill_rate = (
                (self._metrics.avg_fill_rate * (n - 1) + result.fill_rate) / n
            )

        # Move to completed
        if result.request_id in self._active_requests:
            del self._active_requests[result.request_id]
        self._completed_requests.append(result)

        # Notify callbacks
        self._notify(result.request_id, "execution_complete", result)

        logger.info(
            "Execution %s complete: status=%s filled=%.6f/%.6f (%.1f%%) avg_price=%.6f "
            "slippage=%.2fbps time=%.0fms algo=%s",
            result.request_id, result.status.value,
            result.filled_quantity, result.request.quantity,
            result.fill_rate * 100, result.avg_fill_price,
            result.slippage_bps, result.execution_time_ms, result.algorithm_used,
        )

    # -----------------------------------------------------------------------
    # Control
    # -----------------------------------------------------------------------

    async def cancel_execution(self, request_id: str) -> bool:
        """Cancel an active execution."""
        result = self._active_requests.get(request_id)
        if not result:
            return False

        algo_id = result.algo_execution_id
        algo = result.algorithm_used

        if algo == "twap" and algo_id:
            await self._twap.cancel(algo_id)
        elif algo == "vwap" and algo_id:
            await self._vwap.cancel(algo_id)
        elif algo == "iceberg" and algo_id:
            await self._iceberg.cancel(algo_id)
        elif algo == "sniper" and algo_id:
            await self._sniper.cancel(algo_id)

        result.status = ExecutionStatus.CANCELLED
        logger.info("Execution %s cancel requested", request_id)
        return True

    # -----------------------------------------------------------------------
    # Monitoring
    # -----------------------------------------------------------------------

    def get_result(self, request_id: str) -> Optional[ExecutionResult]:
        """Get execution result by request ID."""
        result = self._active_requests.get(request_id)
        if result:
            return result
        for r in self._completed_requests:
            if r.request_id == request_id:
                return r
        return None

    def get_active_executions(self) -> Dict[str, ExecutionResult]:
        """Get all active execution requests."""
        return dict(self._active_requests)

    def get_recent_results(self, limit: int = 50) -> List[ExecutionResult]:
        """Get recent completed execution results."""
        return list(self._completed_requests)[-limit:]

    def get_metrics(self) -> EngineMetrics:
        """Get engine performance metrics."""
        self._metrics.uptime_seconds = time.time() - self._start_time
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Get engine status summary."""
        return {
            "active_executions": len(self._active_requests),
            "completed_executions": len(self._completed_requests),
            "total_requests": self._metrics.total_requests,
            "total_filled": self._metrics.total_filled,
            "total_failed": self._metrics.total_failed,
            "avg_fill_rate": self._metrics.avg_fill_rate,
            "avg_slippage_bps": self._metrics.avg_slippage_bps,
            "avg_execution_time_ms": self._metrics.avg_execution_time_ms,
            "total_volume_usd": self._metrics.total_volume_usd,
            "uptime_seconds": time.time() - self._start_time,
            "venues": len(self._router.get_active_venues()),
            "router_summary": self._router.get_routing_summary(),
        }

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def on_execution_complete(self, request_id: str, callback: Callable) -> None:
        """Register callback for execution completion."""
        if request_id not in self._callbacks:
            self._callbacks[request_id] = []
        self._callbacks[request_id].append(callback)

    def _notify(self, request_id: str, event_type: str, data: Any) -> None:
        """Notify registered callbacks."""
        for cb in self._callbacks.get(request_id, []):
            try:
                cb(event_type, data)
            except Exception as e:
                logger.error("Execution callback error: %s", str(e))
        # Clean up callbacks after completion
        self._callbacks.pop(request_id, None)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self, max_age_seconds: float = 3600.0) -> Dict[str, int]:
        """Clean up old data from all components."""
        results = {
            "twap": self._twap.cleanup(max_age_seconds),
            "vwap": self._vwap.cleanup(max_age_seconds),
            "iceberg": self._iceberg.cleanup(max_age_seconds),
            "sniper": self._sniper.cleanup(max_age_seconds),
        }

        # Clean old completed requests
        now = time.time()
        while self._completed_requests:
            oldest = self._completed_requests[0]
            if oldest.end_time > 0 and (now - oldest.end_time) > max_age_seconds:
                self._completed_requests.popleft()
            else:
                break

        return results
