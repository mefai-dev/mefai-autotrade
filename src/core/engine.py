"""
Trading Engine - Main orchestrator for Mefai Autotrade.
Ties together all core components: order manager, position tracker,
PnL engine, account manager, event bus, and state machine.
Handles the main trading loop, signal processing, strategy scheduling,
error recovery, and graceful startup/shutdown.
"""

import asyncio
import json
import logging
import os
import signal
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from .order_manager import (
    FillEvent,
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidator,
    SymbolInfo,
    TimeInForce,
)
from .position_tracker import (
    Position,
    PositionSide,
    PositionTracker,
)
from .pnl_engine import PnLEngine, PnLRecord
from .account_manager import AccountManager, MarginMode
from .event_bus import Event, EventBus, EventPriority, EventType
from .state_machine import (
    HealthCheck,
    StateMachine,
    TradingState,
    StateTransitionError,
)

logger = logging.getLogger("mefai.autotrade.engine")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Configuration for the trading engine."""
    # Identity
    engine_id: str = "mefai-autotrade-1"
    name: str = "Mefai Autotrade Engine"

    # Account defaults
    initial_balance: float = 10000.0
    default_leverage: int = 1
    default_margin_mode: str = "CROSS"

    # Fees
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004

    # Risk limits
    max_open_positions: int = 50
    max_open_orders: int = 200
    max_daily_trades: int = 500
    max_drawdown_pct: float = 20.0  # Circuit breaker
    max_single_loss_pct: float = 5.0
    max_total_margin_pct: float = 80.0
    max_single_position_pct: float = 20.0
    risk_per_trade_pct: float = 1.0

    # Trading loop
    main_loop_interval_seconds: float = 1.0
    price_update_interval_seconds: float = 0.5
    health_check_interval_seconds: float = 30.0
    snapshot_interval_seconds: float = 300.0  # 5 minutes
    order_timeout_seconds: int = 300  # 5 minutes

    # Strategy scheduling
    strategy_check_interval_seconds: float = 5.0

    # Recovery
    max_consecutive_errors: int = 10
    error_cooldown_seconds: float = 5.0
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0

    # Persistence
    state_persistence_path: str = ""
    data_dir: str = ""

    # Hot reload
    config_file_path: str = ""

    # Shutdown
    graceful_shutdown_timeout_seconds: float = 30.0
    close_positions_on_shutdown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "initial_balance": self.initial_balance,
            "default_leverage": self.default_leverage,
            "default_margin_mode": self.default_margin_mode,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "max_open_positions": self.max_open_positions,
            "max_open_orders": self.max_open_orders,
            "max_daily_trades": self.max_daily_trades,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_single_loss_pct": self.max_single_loss_pct,
            "max_total_margin_pct": self.max_total_margin_pct,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "main_loop_interval_seconds": self.main_loop_interval_seconds,
            "order_timeout_seconds": self.order_timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfig":
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    @classmethod
    def from_file(cls, path: str) -> "EngineConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """
    Base class for trading strategies. Strategies are scheduled by the engine
    and produce trading signals or direct order requests.
    """

    def __init__(self, strategy_id: str, name: str = ""):
        self.strategy_id = strategy_id
        self.name = name or strategy_id
        self.enabled = True
        self.symbols: List[str] = []
        self.interval_seconds: float = 60.0
        self.last_run: Optional[datetime] = None

    @abstractmethod
    async def on_tick(
        self,
        engine: "TradingEngine",
        prices: Dict[str, float],
    ) -> None:
        """
        Called by the engine on each strategy tick.
        Implement trading logic here.
        """
        pass

    async def on_start(self, engine: "TradingEngine") -> None:
        """Called when the strategy is started."""
        pass

    async def on_stop(self, engine: "TradingEngine") -> None:
        """Called when the strategy is stopped."""
        pass

    async def on_signal(
        self,
        engine: "TradingEngine",
        signal: Dict[str, Any],
    ) -> None:
        """Called when an external signal is received for this strategy."""
        pass

    def should_run(self) -> bool:
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self.last_run
        ).total_seconds()
        return elapsed >= self.interval_seconds


# ---------------------------------------------------------------------------
# Signal data class
# ---------------------------------------------------------------------------

@dataclass
class TradingSignal:
    """A trading signal to be processed by the engine."""
    signal_id: str = ""
    symbol: str = ""
    side: str = "BUY"  # BUY or SELL
    order_type: str = "MARKET"
    quantity: float = 0.0
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    leverage: int = 0
    strategy_id: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSignal":
        sig = cls()
        for key, value in data.items():
            if key == "timestamp" and isinstance(value, str):
                sig.timestamp = datetime.fromisoformat(value)
            elif hasattr(sig, key):
                setattr(sig, key, value)
        return sig


# ---------------------------------------------------------------------------
# Exchange interface (abstract)
# ---------------------------------------------------------------------------

class ExchangeInterface(ABC):
    """
    Abstract interface for exchange operations.
    Concrete implementations connect to Binance, etc.
    """

    @abstractmethod
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place an order on the exchange. Returns exchange response."""
        pass

    @abstractmethod
    async def cancel_order(
        self, symbol: str, order_id: str
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_order_status(
        self, symbol: str, order_id: str
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_mark_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    async def get_mark_prices(
        self, symbols: List[str]
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> None:
        pass

    @abstractmethod
    async def set_margin_mode(
        self, symbol: str, mode: str
    ) -> None:
        pass

    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        pass


# ---------------------------------------------------------------------------
# Trading Engine
# ---------------------------------------------------------------------------

class TradingEngine:
    """
    Main trading engine that orchestrates all components.
    Runs the main async trading loop, schedules strategies,
    processes signals, routes orders, and monitors risk.
    """

    def __init__(
        self,
        config: EngineConfig,
        exchange: Optional[ExchangeInterface] = None,
    ):
        self.config = config
        self.exchange = exchange

        # Core components
        self.order_validator = OrderValidator()
        self.order_manager = OrderManager(
            validator=self.order_validator,
            max_open_orders_global=config.max_open_orders,
            default_timeout_seconds=config.order_timeout_seconds,
        )
        self.position_tracker = PositionTracker(
            default_leverage=config.default_leverage,
            default_margin_mode=config.default_margin_mode,
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )
        self.pnl_engine = PnLEngine(
            initial_balance=config.initial_balance,
        )
        self.account_manager = AccountManager()
        self.event_bus = EventBus()
        self.state_machine = StateMachine(
            persistence_path=config.state_persistence_path or None,
        )

        # Strategies
        self._strategies: Dict[str, Strategy] = {}

        # Tracked symbols
        self._active_symbols: Set[str] = set()
        self._symbol_prices: Dict[str, float] = {}

        # Signal queue
        self._signal_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Error tracking
        self._consecutive_errors = 0
        self._last_error_time: Optional[datetime] = None
        self._daily_trade_count = 0
        self._daily_trade_date = ""

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # Config hot reload
        self._config_mtime: float = 0.0

        # Wire up internal event handlers
        self._setup_internal_handlers()

        logger.info(
            "TradingEngine created: %s (%s)",
            config.engine_id, config.name,
        )

    # -- Setup ----------------------------------------------------------------

    def _setup_internal_handlers(self) -> None:
        """Wire up internal event handlers and callbacks."""

        # Order fill handler - update positions and PnL
        self.order_manager.on_fill(self._on_order_filled)
        self.order_manager.on_partial_fill(self._on_order_partial_fill)
        self.order_manager.on_cancel(self._on_order_cancelled)
        self.order_manager.on_reject(self._on_order_rejected)

        # Position close handler - record PnL
        self.position_tracker.on_close(self._on_position_closed)
        self.position_tracker.on_liquidation_warning(
            self._on_liquidation_warning
        )

        # State transition logging
        self.state_machine.on_any_transition(self._on_state_transition)

        # Register health checks
        self.state_machine.register_health_check(
            "order_manager", self._health_check_order_manager
        )
        self.state_machine.register_health_check(
            "position_tracker", self._health_check_position_tracker
        )
        self.state_machine.register_health_check(
            "account", self._health_check_account
        )

    def _health_check_order_manager(self) -> HealthCheck:
        stats = self.order_manager.get_stats()
        return HealthCheck(
            component="order_manager",
            healthy=True,
            message=f"{stats['active_orders']} active orders",
            details=stats,
        )

    def _health_check_position_tracker(self) -> HealthCheck:
        stats = self.position_tracker.get_stats()
        return HealthCheck(
            component="position_tracker",
            healthy=True,
            message=f"{stats['open_positions']} open positions",
            details=stats,
        )

    def _health_check_account(self) -> HealthCheck:
        account = self.account_manager.get_active_account()
        if account is None:
            return HealthCheck(
                component="account",
                healthy=False,
                message="No active account",
            )
        bal = account.get_balance("USDT")
        return HealthCheck(
            component="account",
            healthy=bal.total > 0,
            message=f"Balance: {bal.total:.2f} USDT",
            details=bal.to_dict(),
        )

    # -- Strategy management --------------------------------------------------

    def register_strategy(self, strategy: Strategy) -> None:
        self._strategies[strategy.strategy_id] = strategy
        for symbol in strategy.symbols:
            self._active_symbols.add(symbol)
        logger.info(
            "Strategy registered: %s (%s, %d symbols)",
            strategy.strategy_id, strategy.name, len(strategy.symbols),
        )

    def unregister_strategy(self, strategy_id: str) -> bool:
        strategy = self._strategies.pop(strategy_id, None)
        if strategy is None:
            return False
        logger.info("Strategy unregistered: %s", strategy_id)
        return True

    def enable_strategy(self, strategy_id: str) -> bool:
        strategy = self._strategies.get(strategy_id)
        if strategy:
            strategy.enabled = True
            logger.info("Strategy enabled: %s", strategy_id)
            return True
        return False

    def disable_strategy(self, strategy_id: str) -> bool:
        strategy = self._strategies.get(strategy_id)
        if strategy:
            strategy.enabled = False
            logger.info("Strategy disabled: %s", strategy_id)
            return True
        return False

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> Dict[str, Strategy]:
        return dict(self._strategies)

    # -- Lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all components and start the main trading loop."""
        logger.info("Starting trading engine: %s", self.config.engine_id)

        try:
            # Create default account if none exists
            if not self.account_manager.get_active_account():
                self.account_manager.create_account(
                    name="default",
                    initial_balance=self.config.initial_balance,
                    default_leverage=self.config.default_leverage,
                    default_margin_mode=(
                        MarginMode.CROSS
                        if self.config.default_margin_mode == "CROSS"
                        else MarginMode.ISOLATED
                    ),
                    max_open_positions=self.config.max_open_positions,
                    max_total_margin_pct=self.config.max_total_margin_pct,
                    max_single_position_pct=self.config.max_single_position_pct,
                    risk_per_trade_pct=self.config.risk_per_trade_pct,
                )

            # Sync exchange info if exchange is connected
            if self.exchange:
                await self._sync_exchange_info()
                await self._sync_account()

            # Start event bus
            await self.event_bus.start()

            # Start order timeout checker
            await self.order_manager.start_timeout_checker()

            # Start strategies
            for strategy in self._strategies.values():
                try:
                    await strategy.on_start(self)
                except Exception:
                    logger.exception(
                        "Failed to start strategy %s", strategy.strategy_id
                    )

            # Transition to RUNNING
            self.state_machine.start("Engine initialization complete")

            # Start background tasks
            self._tasks.append(
                asyncio.create_task(self._main_loop())
            )
            self._tasks.append(
                asyncio.create_task(self._strategy_loop())
            )
            self._tasks.append(
                asyncio.create_task(self._signal_processor_loop())
            )
            self._tasks.append(
                asyncio.create_task(self._health_check_loop())
            )
            self._tasks.append(
                asyncio.create_task(self._snapshot_loop())
            )

            await self.event_bus.emit_async(
                EventType.ENGINE_STARTED,
                {"engine_id": self.config.engine_id},
                priority=EventPriority.HIGH,
                source="engine",
            )

            logger.info("Trading engine started successfully")

        except Exception as e:
            logger.exception("Failed to start engine")
            self.state_machine.mark_error(str(e))
            raise

    async def stop(
        self, close_positions: bool = False, reason: str = "Shutdown"
    ) -> None:
        """Gracefully shut down the engine."""
        logger.info("Stopping trading engine: %s", reason)

        try:
            self.state_machine.shutdown(reason)
        except StateTransitionError:
            logger.warning("Could not transition to SHUTTING_DOWN")

        close_positions = (
            close_positions or self.config.close_positions_on_shutdown
        )

        # Stop strategies
        for strategy in self._strategies.values():
            try:
                await strategy.on_stop(self)
            except Exception:
                logger.exception(
                    "Error stopping strategy %s", strategy.strategy_id
                )

        # Cancel all open orders
        cancelled = self.order_manager.cancel_all_orders(
            reason="Engine shutdown"
        )
        logger.info("Cancelled %d orders on shutdown", len(cancelled))

        # Close positions if requested
        if close_positions:
            await self._close_all_positions(reason="Engine shutdown")

        # Stop background tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Stop order timeout checker
        await self.order_manager.stop_timeout_checker()

        # Stop event bus
        await self.event_bus.emit_async(
            EventType.ENGINE_STOPPED,
            {"engine_id": self.config.engine_id, "reason": reason},
            priority=EventPriority.HIGH,
            source="engine",
        )
        await self.event_bus.stop()

        try:
            self.state_machine.mark_stopped(reason)
        except StateTransitionError:
            pass

        logger.info("Trading engine stopped")

    async def pause(self, reason: str = "User pause") -> None:
        """Pause trading (keep positions, stop new orders)."""
        self.state_machine.pause(reason)
        await self.event_bus.emit_async(
            EventType.ENGINE_PAUSED,
            {"reason": reason},
            priority=EventPriority.HIGH,
            source="engine",
        )
        logger.info("Engine paused: %s", reason)

    async def resume(self, reason: str = "User resume") -> None:
        """Resume trading after pause."""
        self.state_machine.resume(reason)
        await self.event_bus.emit_async(
            EventType.ENGINE_RESUMED,
            {"reason": reason},
            priority=EventPriority.HIGH,
            source="engine",
        )
        logger.info("Engine resumed: %s", reason)

    async def emergency_stop(self, reason: str = "Emergency") -> None:
        """Emergency stop - cancel all orders, optionally close positions."""
        logger.warning("EMERGENCY STOP: %s", reason)

        self.state_machine.emergency_stop(reason)

        # Cancel all orders immediately
        self.order_manager.cancel_all_orders(reason="Emergency stop")

        await self.event_bus.emit_async(
            EventType.EMERGENCY_STOP,
            {"reason": reason},
            priority=EventPriority.CRITICAL,
            source="engine",
        )

    # -- Main loops -----------------------------------------------------------

    async def _main_loop(self) -> None:
        """Main trading loop - price updates and risk monitoring."""
        interval = self.config.main_loop_interval_seconds

        while self.state_machine.state not in (
            TradingState.STOPPED, TradingState.SHUTTING_DOWN
        ):
            try:
                if not self.state_machine.is_operational:
                    await asyncio.sleep(interval)
                    continue

                # Update prices
                await self._update_prices()

                # Check risk limits
                await self._check_risk_limits()

                # Check stop loss / take profit triggers
                await self._check_sl_tp_triggers()

                # Reset daily trade counter
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._daily_trade_date:
                    self._daily_trade_count = 0
                    self._daily_trade_date = today

                # Reset error counter on success
                self._consecutive_errors = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._handle_error("main_loop", e)

            await asyncio.sleep(interval)

    async def _strategy_loop(self) -> None:
        """Strategy scheduling loop."""
        interval = self.config.strategy_check_interval_seconds

        while self.state_machine.state not in (
            TradingState.STOPPED, TradingState.SHUTTING_DOWN
        ):
            try:
                if not self.state_machine.can_trade:
                    await asyncio.sleep(interval)
                    continue

                for strategy in self._strategies.values():
                    if not strategy.should_run():
                        continue

                    try:
                        await strategy.on_tick(self, self._symbol_prices)
                        strategy.last_run = datetime.now(timezone.utc)
                    except Exception:
                        logger.exception(
                            "Error in strategy %s", strategy.strategy_id
                        )
                        await self.event_bus.emit_async(
                            EventType.STRATEGY_ERROR,
                            {
                                "strategy_id": strategy.strategy_id,
                                "error": "Tick error",
                            },
                            source="engine",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._handle_error("strategy_loop", e)

            await asyncio.sleep(interval)

    async def _signal_processor_loop(self) -> None:
        """Process incoming trading signals from the queue."""
        while self.state_machine.state not in (
            TradingState.STOPPED, TradingState.SHUTTING_DOWN
        ):
            try:
                try:
                    signal = await asyncio.wait_for(
                        self._signal_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if not self.state_machine.can_trade:
                    logger.warning(
                        "Signal rejected - engine not in RUNNING state"
                    )
                    await self.event_bus.emit_async(
                        EventType.SIGNAL_REJECTED,
                        {
                            "signal": signal.__dict__
                            if hasattr(signal, "__dict__") else str(signal),
                            "reason": "Engine not running",
                        },
                        source="engine",
                    )
                    continue

                await self._process_signal(signal)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._handle_error("signal_processor", e)

    async def _health_check_loop(self) -> None:
        """Periodic health checks."""
        interval = self.config.health_check_interval_seconds

        while self.state_machine.state not in (
            TradingState.STOPPED, TradingState.SHUTTING_DOWN
        ):
            try:
                results = self.state_machine.run_health_checks()

                unhealthy = [
                    name for name, hc in results.items() if not hc.healthy
                ]
                if unhealthy:
                    logger.warning(
                        "Unhealthy components: %s",
                        ", ".join(unhealthy),
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in health check loop")

            await asyncio.sleep(interval)

    async def _snapshot_loop(self) -> None:
        """Periodic snapshots of account state and equity."""
        interval = self.config.snapshot_interval_seconds

        while self.state_machine.state not in (
            TradingState.STOPPED, TradingState.SHUTTING_DOWN
        ):
            try:
                if self.state_machine.is_operational:
                    unrealized = (
                        self.position_tracker.get_total_unrealized_pnl()
                    )
                    open_count = len(
                        self.position_tracker.get_open_positions()
                    )
                    margin = self.position_tracker.get_total_margin_used()

                    # PnL snapshot
                    self.pnl_engine.update_equity(
                        unrealized_pnl=unrealized,
                        open_positions=open_count,
                        margin_used=margin,
                    )

                    # Account snapshot
                    account = self.account_manager.get_active_account()
                    if account:
                        account.take_snapshot(
                            unrealized_pnl=unrealized,
                            open_positions=open_count,
                            open_orders=self.order_manager.count_open_orders(),
                        )

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in snapshot loop")

            await asyncio.sleep(interval)

    # -- Signal processing ----------------------------------------------------

    async def submit_signal(self, signal: TradingSignal) -> None:
        """Submit a trading signal for processing."""
        try:
            self._signal_queue.put_nowait(signal)
            await self.event_bus.emit_async(
                EventType.SIGNAL_GENERATED,
                {"signal_id": signal.signal_id, "symbol": signal.symbol},
                source="engine",
            )
        except asyncio.QueueFull:
            logger.warning("Signal queue full, dropping signal")

    async def _process_signal(self, signal: TradingSignal) -> None:
        """Process a single trading signal into orders."""
        logger.info(
            "Processing signal: %s %s %s @ %.4f",
            signal.symbol, signal.side, signal.order_type, signal.price,
        )

        # Check daily trade limit
        if self._daily_trade_count >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            await self.event_bus.emit_async(
                EventType.SIGNAL_REJECTED,
                {
                    "signal_id": signal.signal_id,
                    "reason": "Daily trade limit reached",
                },
                source="engine",
            )
            return

        # Check if we can open more positions
        open_positions = self.position_tracker.get_open_positions()
        if len(open_positions) >= self.config.max_open_positions:
            logger.warning("Max open positions reached")
            await self.event_bus.emit_async(
                EventType.SIGNAL_REJECTED,
                {
                    "signal_id": signal.signal_id,
                    "reason": "Max positions reached",
                },
                source="engine",
            )
            return

        # Determine order parameters
        order_side = (
            OrderSide.BUY if signal.side.upper() == "BUY"
            else OrderSide.SELL
        )
        order_type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_MARKET": OrderType.STOP_MARKET,
            "STOP_LIMIT": OrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(
            signal.order_type.upper(), OrderType.MARKET
        )

        # Calculate quantity if not provided
        quantity = signal.quantity
        if quantity <= 0 and signal.price > 0:
            account = self.account_manager.get_active_account()
            if account and signal.stop_loss > 0:
                quantity = account.calculate_position_size(
                    symbol=signal.symbol,
                    entry_price=signal.price,
                    stop_loss_price=signal.stop_loss,
                )

        if quantity <= 0:
            logger.warning("Cannot determine position size for signal")
            return

        # Set leverage
        leverage = signal.leverage or self.config.default_leverage
        self.position_tracker.set_leverage(signal.symbol, leverage)

        # Get available balance
        account = self.account_manager.get_active_account()
        available = 0.0
        if account:
            bal = account.get_balance("USDT")
            available = bal.available

        # Create and submit order
        order, success, errors = self.order_manager.create_and_submit(
            symbol=signal.symbol,
            side=order_side,
            order_type=order_type,
            quantity=quantity,
            price=signal.price,
            leverage=leverage,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            available_balance=available,
            tags={"source": signal.source},
        )

        if not success:
            logger.warning(
                "Order rejected for signal %s: %s",
                signal.signal_id, errors,
            )
            await self.event_bus.emit_async(
                EventType.SIGNAL_REJECTED,
                {
                    "signal_id": signal.signal_id,
                    "errors": errors,
                },
                source="engine",
            )
            return

        # Route to exchange
        if self.exchange:
            await self._route_order_to_exchange(order)

        self._daily_trade_count += 1

        await self.event_bus.emit_async(
            EventType.SIGNAL_PROCESSED,
            {
                "signal_id": signal.signal_id,
                "order_id": order.id,
                "symbol": signal.symbol,
            },
            source="engine",
        )

        # Set up SL/TP if provided
        if signal.stop_loss > 0 or signal.take_profit > 0:
            # These will be placed after position is confirmed
            order.tags["pending_sl"] = str(signal.stop_loss)
            order.tags["pending_tp"] = str(signal.take_profit)

        logger.info(
            "Signal processed: %s -> order %s",
            signal.signal_id, order.id,
        )

    # -- Order routing --------------------------------------------------------

    async def _route_order_to_exchange(self, order: Order) -> None:
        """Send order to the exchange and handle the response."""
        if not self.exchange:
            logger.warning("No exchange connected, simulating fill")
            # Simulate immediate fill for market orders (paper trading)
            if order.order_type == OrderType.MARKET:
                price = self._symbol_prices.get(order.symbol, order.price)
                if price > 0:
                    self.order_manager.process_fill(
                        order.id, price, order.quantity,
                        commission=price * order.quantity * self.config.taker_fee,
                    )
            return

        try:
            response = await self.exchange.place_order(order)

            exchange_id = response.get("orderId", response.get("order_id", ""))
            if exchange_id:
                self.order_manager.set_exchange_order_id(
                    order.id, str(exchange_id)
                )

            # Check if immediately filled (market orders)
            status = response.get("status", "")
            if status == "FILLED":
                fills = response.get("fills", [])
                for fill_data in fills:
                    self.order_manager.process_fill(
                        order.id,
                        price=float(fill_data.get("price", 0)),
                        quantity=float(fill_data.get("qty", 0)),
                        commission=float(fill_data.get("commission", 0)),
                        commission_asset=fill_data.get(
                            "commissionAsset", "USDT"
                        ),
                        is_maker=fill_data.get("isMaker", False),
                    )

        except Exception as e:
            logger.error(
                "Failed to route order %s to exchange: %s", order.id, e
            )
            self.order_manager.reject_order(
                order.id, f"Exchange error: {e}"
            )

    # -- Price updates --------------------------------------------------------

    async def _update_prices(self) -> None:
        """Fetch latest prices and update position tracker."""
        if not self._active_symbols:
            return

        if self.exchange:
            try:
                prices = await self.exchange.get_mark_prices(
                    list(self._active_symbols)
                )
                self._symbol_prices.update(prices)
            except Exception:
                logger.debug("Price update failed, using cached prices")
                return

        # Update position tracker
        if self._symbol_prices:
            self.position_tracker.update_mark_prices(self._symbol_prices)

    # -- Risk management ------------------------------------------------------

    async def _check_risk_limits(self) -> None:
        """Check risk limits and trigger circuit breakers if needed."""

        # Check drawdown limit
        dd = self.pnl_engine.get_drawdown_info()
        if dd.current_drawdown_pct >= self.config.max_drawdown_pct:
            logger.warning(
                "CIRCUIT BREAKER: Drawdown %.2f%% exceeds limit %.2f%%",
                dd.current_drawdown_pct, self.config.max_drawdown_pct,
            )
            await self.event_bus.emit_async(
                EventType.CIRCUIT_BREAKER,
                {
                    "type": "drawdown",
                    "current": dd.current_drawdown_pct,
                    "limit": self.config.max_drawdown_pct,
                },
                priority=EventPriority.CRITICAL,
                source="risk_manager",
            )
            await self.emergency_stop(
                f"Drawdown limit {dd.current_drawdown_pct:.2f}%"
            )
            return

        # Check margin ratio
        account = self.account_manager.get_active_account()
        if account:
            margin_info = account.get_margin_info()
            if margin_info.margin_ratio >= 90.0:
                await self.event_bus.emit_async(
                    EventType.MARGIN_WARNING,
                    {
                        "margin_ratio": margin_info.margin_ratio,
                        "margin_balance": margin_info.margin_balance,
                    },
                    priority=EventPriority.CRITICAL,
                    source="risk_manager",
                )

    async def _check_sl_tp_triggers(self) -> None:
        """Check stop loss and take profit triggers for all positions."""
        for pos in self.position_tracker.get_open_positions():
            price = self._symbol_prices.get(pos.symbol)
            if price is None:
                continue

            triggered = False
            reason = ""

            # Stop loss check
            if pos.stop_loss > 0:
                if pos.side == PositionSide.LONG and price <= pos.stop_loss:
                    triggered = True
                    reason = f"Stop loss triggered at {price:.4f}"
                elif pos.side == PositionSide.SHORT and price >= pos.stop_loss:
                    triggered = True
                    reason = f"Stop loss triggered at {price:.4f}"

            # Take profit check
            if not triggered and pos.take_profit > 0:
                if pos.side == PositionSide.LONG and price >= pos.take_profit:
                    triggered = True
                    reason = f"Take profit triggered at {price:.4f}"
                elif pos.side == PositionSide.SHORT and price <= pos.take_profit:
                    triggered = True
                    reason = f"Take profit triggered at {price:.4f}"

            # Trailing stop check
            if not triggered and pos.trailing_stop_pct > 0:
                if pos.update_trailing_stop(price):
                    triggered = True
                    reason = f"Trailing stop triggered at {price:.4f}"

            if triggered:
                logger.info(
                    "Position %s SL/TP trigger: %s", pos.id, reason
                )
                await self._close_position_with_order(
                    pos, price, reason
                )

    async def _close_position_with_order(
        self,
        position: Position,
        price: float,
        reason: str,
    ) -> None:
        """Close a position by submitting a market order."""
        close_side = (
            OrderSide.SELL
            if position.side == PositionSide.LONG
            else OrderSide.BUY
        )

        order, success, errors = self.order_manager.create_and_submit(
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            reduce_only=True,
            strategy_id=position.strategy_id,
            tags={"close_reason": reason},
        )

        if success and self.exchange:
            await self._route_order_to_exchange(order)
        elif success and not self.exchange:
            # Paper trading - simulate fill
            self.order_manager.process_fill(
                order.id, price, position.quantity,
                commission=price * position.quantity * self.config.taker_fee,
            )

    async def _close_all_positions(self, reason: str = "Close all") -> None:
        """Close all open positions."""
        open_positions = self.position_tracker.get_open_positions()
        for pos in open_positions:
            price = self._symbol_prices.get(pos.symbol, pos.mark_price)
            if price > 0:
                await self._close_position_with_order(pos, price, reason)

    # -- Internal event handlers -----------------------------------------------

    def _on_order_filled(
        self, order: Order, fill: Optional[FillEvent]
    ) -> None:
        """Handle a fully filled order - open/close position."""
        if fill is None:
            return

        # Determine if this opens or closes a position
        existing = None
        open_positions = self.position_tracker.get_open_positions(
            symbol=order.symbol
        )
        for pos in open_positions:
            if (
                pos.side == PositionSide.LONG
                and order.side == OrderSide.SELL
            ):
                existing = pos
                break
            elif (
                pos.side == PositionSide.SHORT
                and order.side == OrderSide.BUY
            ):
                existing = pos
                break

        if existing and (order.reduce_only or existing is not None):
            # Close position
            try:
                self.position_tracker.close_position(
                    existing.id,
                    close_price=order.avg_fill_price,
                    quantity=order.filled_qty,
                    commission=order.commission,
                    order_id=order.id,
                )
            except Exception:
                logger.exception("Failed to close position")
        else:
            # Open position
            pos_side = (
                PositionSide.LONG
                if order.side == OrderSide.BUY
                else PositionSide.SHORT
            )
            try:
                position = self.position_tracker.open_position(
                    symbol=order.symbol,
                    side=pos_side,
                    quantity=order.filled_qty,
                    entry_price=order.avg_fill_price,
                    leverage=order.leverage,
                    strategy_id=order.strategy_id,
                    signal_id=order.signal_id,
                    order_id=order.id,
                    commission=order.commission,
                )

                # Apply pending SL/TP from signal
                pending_sl = order.tags.get("pending_sl", "0")
                pending_tp = order.tags.get("pending_tp", "0")
                sl = float(pending_sl)
                tp = float(pending_tp)

                if sl > 0:
                    self.position_tracker.modify_stop_loss(position.id, sl)
                if tp > 0:
                    self.position_tracker.modify_take_profit(position.id, tp)

            except Exception:
                logger.exception("Failed to open position")

        # Lock/unlock margin in account
        account = self.account_manager.get_active_account()
        if account:
            account.apply_commission(order.commission, reference_id=order.id)

        # Cancel linked OCO orders if any
        if order.linked_order_ids:
            self.order_manager.cancel_linked_orders(order.id)

        # Emit event
        self.event_bus.emit(
            EventType.ORDER_FILLED,
            {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.filled_qty,
                "price": order.avg_fill_price,
                "commission": order.commission,
            },
            priority=EventPriority.HIGH,
            source="engine",
        )

    def _on_order_partial_fill(
        self, order: Order, fill: Optional[FillEvent]
    ) -> None:
        self.event_bus.emit(
            EventType.ORDER_PARTIALLY_FILLED,
            {
                "order_id": order.id,
                "filled_qty": order.filled_qty,
                "remaining_qty": order.remaining_qty,
            },
            source="engine",
        )

    def _on_order_cancelled(
        self, order: Order, fill: Optional[FillEvent]
    ) -> None:
        # Release margin if any was locked
        account = self.account_manager.get_active_account()
        if account and order.notional > 0:
            margin = order.notional / max(1, order.leverage)
            remaining_margin = margin * (order.remaining_qty / order.quantity)
            if remaining_margin > 0:
                account.release_margin(
                    remaining_margin, reference_id=order.id
                )

        self.event_bus.emit(
            EventType.ORDER_CANCELLED,
            {"order_id": order.id, "reason": order.reject_reason},
            source="engine",
        )

    def _on_order_rejected(
        self, order: Order, fill: Optional[FillEvent]
    ) -> None:
        self.event_bus.emit(
            EventType.ORDER_REJECTED,
            {
                "order_id": order.id,
                "reason": order.reject_reason,
                "error_code": order.error_code,
            },
            source="engine",
        )

    def _on_position_closed(
        self, position: Position, event: Any
    ) -> None:
        """Record closed position PnL."""
        record = PnLRecord(
            record_id=position.id,
            timestamp=datetime.now(timezone.utc),
            symbol=position.symbol,
            strategy_id=position.strategy_id,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=position.mark_price,
            quantity=position.total_bought_qty or position.total_sold_qty,
            gross_pnl=position.realized_pnl + position.total_commission,
            commission=position.total_commission,
            funding_pnl=position.funding_pnl,
            net_pnl=position.realized_pnl,
            holding_time_seconds=position.holding_time_seconds,
            leverage=position.leverage,
        )
        self.pnl_engine.record_trade(record)

        # Apply PnL to account
        account = self.account_manager.get_active_account()
        if account:
            account.apply_pnl(position.realized_pnl, reference_id=position.id)
            account.release_margin(position.margin_used, reference_id=position.id)

        self.event_bus.emit(
            EventType.POSITION_CLOSED,
            {
                "position_id": position.id,
                "symbol": position.symbol,
                "side": position.side.value,
                "pnl": position.realized_pnl,
                "holding_hours": position.holding_time_hours,
            },
            priority=EventPriority.HIGH,
            source="engine",
        )

    def _on_liquidation_warning(
        self, position: Position, event: Any
    ) -> None:
        self.event_bus.emit(
            EventType.POSITION_LIQUIDATION_WARNING,
            {
                "position_id": position.id,
                "symbol": position.symbol,
                "liquidation_price": position.liquidation_price,
                "mark_price": position.mark_price,
                "distance_pct": position.liquidation_distance_pct,
            },
            priority=EventPriority.CRITICAL,
            source="engine",
        )

    def _on_state_transition(
        self,
        from_state: TradingState,
        to_state: TradingState,
        reason: str,
    ) -> None:
        logger.info(
            "State: %s -> %s (%s)",
            from_state.value, to_state.value, reason,
        )

    # -- Exchange sync --------------------------------------------------------

    async def _sync_exchange_info(self) -> None:
        """Sync exchange symbol info for validation."""
        if not self.exchange:
            return

        try:
            info = await self.exchange.get_exchange_info()
            symbols = info.get("symbols", [])

            symbol_infos = []
            for sym_data in symbols:
                si = SymbolInfo(
                    symbol=sym_data.get("symbol", ""),
                    base_asset=sym_data.get("baseAsset", ""),
                    quote_asset=sym_data.get("quoteAsset", ""),
                )
                # Parse filters
                for f in sym_data.get("filters", []):
                    ft = f.get("filterType", "")
                    if ft == "LOT_SIZE":
                        si.min_qty = float(f.get("minQty", 0))
                        si.max_qty = float(f.get("maxQty", 999999999))
                        si.qty_step = float(f.get("stepSize", 0.001))
                    elif ft == "PRICE_FILTER":
                        si.min_price = float(f.get("minPrice", 0))
                        si.max_price = float(f.get("maxPrice", 999999999))
                        si.price_step = float(f.get("tickSize", 0.01))
                    elif ft == "MIN_NOTIONAL":
                        si.min_notional = float(
                            f.get("notional", f.get("minNotional", 5))
                        )
                    elif ft == "MAX_NUM_ORDERS":
                        si.max_open_orders = int(f.get("limit", 200))

                symbol_infos.append(si)

            self.order_validator.update_symbol_info_batch(symbol_infos)
            logger.info(
                "Exchange info synced: %d symbols", len(symbol_infos)
            )
        except Exception:
            logger.exception("Failed to sync exchange info")

    async def _sync_account(self) -> None:
        """Sync account balance from exchange."""
        if not self.exchange:
            return

        try:
            info = await self.exchange.get_account_info()
            account = self.account_manager.get_active_account()
            if account:
                balances = {}
                for asset_data in info.get("assets", []):
                    asset = asset_data.get("asset", "")
                    if asset:
                        balances[asset] = {
                            "available": float(
                                asset_data.get("availableBalance", 0)
                            ),
                            "locked": float(
                                asset_data.get("initialMargin", 0)
                            ),
                        }
                self.account_manager.sync_balances_from_exchange(
                    account.account_id, balances
                )
        except Exception:
            logger.exception("Failed to sync account")

    # -- Error handling -------------------------------------------------------

    def _handle_error(self, component: str, error: Exception) -> None:
        """Handle errors with exponential backoff."""
        self._consecutive_errors += 1
        self._last_error_time = datetime.now(timezone.utc)

        logger.error(
            "Error in %s (consecutive=%d): %s",
            component, self._consecutive_errors, error,
        )

        if self._consecutive_errors >= self.config.max_consecutive_errors:
            logger.critical(
                "Max consecutive errors (%d) reached, pausing engine",
                self.config.max_consecutive_errors,
            )
            try:
                self.state_machine.pause(
                    f"Max errors reached in {component}"
                )
            except StateTransitionError:
                pass

    # -- Config hot reload ----------------------------------------------------

    def try_reload_config(self) -> bool:
        """Check if config file changed and reload if so."""
        if not self.config.config_file_path:
            return False

        try:
            mtime = os.path.getmtime(self.config.config_file_path)
            if mtime <= self._config_mtime:
                return False

            new_config = EngineConfig.from_file(self.config.config_file_path)

            # Update safe-to-change fields
            self.config.max_drawdown_pct = new_config.max_drawdown_pct
            self.config.max_daily_trades = new_config.max_daily_trades
            self.config.risk_per_trade_pct = new_config.risk_per_trade_pct
            self.config.max_single_loss_pct = new_config.max_single_loss_pct
            self.config.health_check_interval_seconds = (
                new_config.health_check_interval_seconds
            )

            self._config_mtime = mtime
            logger.info("Configuration reloaded from %s", self.config.config_file_path)
            return True
        except Exception:
            logger.exception("Failed to reload config")
            return False

    # -- Public query methods -------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            "engine_id": self.config.engine_id,
            "state": self.state_machine.state.value,
            "can_trade": self.state_machine.can_trade,
            "uptime_seconds": self.state_machine.uptime_seconds,
            "total_running_seconds": self.state_machine.total_running_seconds,
            "strategies": {
                sid: {
                    "name": s.name,
                    "enabled": s.enabled,
                    "last_run": (
                        s.last_run.isoformat() if s.last_run else None
                    ),
                }
                for sid, s in self._strategies.items()
            },
            "orders": self.order_manager.get_stats(),
            "positions": self.position_tracker.get_stats(),
            "pnl": self.pnl_engine.get_summary(),
            "account": (
                self.account_manager.get_active_account().get_summary()
                if self.account_manager.get_active_account() else None
            ),
            "events": self.event_bus.get_stats(),
            "health": {
                name: hc.to_dict()
                for name, hc in self.state_machine.get_last_health_results().items()
            },
            "active_symbols": list(self._active_symbols),
            "consecutive_errors": self._consecutive_errors,
            "daily_trade_count": self._daily_trade_count,
        }

    def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.pnl_engine.calculate_metrics()
        return metrics.to_dict()

    def get_prices(self) -> Dict[str, float]:
        """Get current price cache."""
        return dict(self._symbol_prices)

    def set_price(self, symbol: str, price: float) -> None:
        """Manually set a price (for paper trading / testing)."""
        self._symbol_prices[symbol] = price
        self._active_symbols.add(symbol)
        self.position_tracker.update_mark_price(symbol, price)
