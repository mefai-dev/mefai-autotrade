"""
Dashboard API for Mefai Autotrade.
FastAPI-based dashboard backend (port 8400) with REST endpoints for
performance data, real-time WebSocket updates, authentication,
and trading controls (pause, emergency stop).
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

logger = logging.getLogger("mefai.autotrade")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DASHBOARD_PORT = 8400
DEFAULT_API_KEY = os.environ.get("MEFAI_DASHBOARD_API_KEY", "")
CORS_ORIGINS = os.environ.get(
    "MEFAI_DASHBOARD_CORS",
    "http://localhost:3000,http://localhost:8400,https://mefai.io",
).split(",")


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class PauseRequest(BaseModel):
    paused: bool
    reason: str = ""


class EmergencyStopRequest(BaseModel):
    reason: str = "Manual emergency stop"
    close_positions: bool = True


class AlertAckRequest(BaseModel):
    alert_id: str
    user: str = "dashboard"


class AlertResolveRequest(BaseModel):
    alert_id: str


class RuleToggleRequest(BaseModel):
    rule_name: str
    enabled: bool


# ---------------------------------------------------------------------------
# Dashboard State
# ---------------------------------------------------------------------------

class TradingStatus(Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class DashboardState:
    """Shared state between the dashboard API and the trading engine.

    The trading engine writes to this state, and the dashboard reads from it.
    Thread-safe via locks on mutation methods.
    """

    trading_status: TradingStatus = TradingStatus.RUNNING
    paused_reason: str = ""
    emergency_stop_reason: str = ""

    # References to monitoring components (set during initialization)
    performance_tracker: Any = None
    health_checker: Any = None
    alert_manager: Any = None
    metrics_collector: Any = None
    notifier: Any = None

    # Live position data (updated by trading engine)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _positions: List[Dict[str, Any]] = field(default_factory=list)
    _open_orders: List[Dict[str, Any]] = field(default_factory=list)
    _recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    _strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _daily_pnl: float = 0.0
    _portfolio_value: float = 0.0
    _margin_used_pct: float = 0.0

    # Callbacks for trading control
    _pause_callback: Optional[Callable] = None
    _resume_callback: Optional[Callable] = None
    _emergency_stop_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        pause_cb: Optional[Callable] = None,
        resume_cb: Optional[Callable] = None,
        emergency_stop_cb: Optional[Callable] = None,
    ) -> None:
        self._pause_callback = pause_cb
        self._resume_callback = resume_cb
        self._emergency_stop_callback = emergency_stop_cb

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._positions = positions

    def update_orders(self, orders: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._open_orders = orders

    def update_trades(self, trades: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._recent_trades = trades

    def update_strategies(self, strategies: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            self._strategies = strategies

    def update_portfolio(
        self,
        portfolio_value: float,
        daily_pnl: float,
        margin_used_pct: float,
    ) -> None:
        with self._lock:
            self._portfolio_value = portfolio_value
            self._daily_pnl = daily_pnl
            self._margin_used_pct = margin_used_pct

    def get_positions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._positions)

    def get_orders(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._open_orders)

    def get_trades(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent_trades)

    def get_strategies(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._strategies)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "portfolio_value": self._portfolio_value,
                "daily_pnl": self._daily_pnl,
                "margin_used_pct": self._margin_used_pct,
            }


# ---------------------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------------------

class WebSocketManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info(f"Dashboard WebSocket connected. Total: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info(f"Dashboard WebSocket disconnected. Total: {len(self._connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        async with self._lock:
            dead: List[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._connections.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _verify_api_key(request: Request) -> None:
    """Verify the API key from the request header or query parameter."""
    if not DEFAULT_API_KEY:
        return  # No API key configured - skip auth

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Set X-API-Key header or api_key query param.",
        )

    # Constant-time comparison
    if not hmac.compare_digest(api_key, DEFAULT_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

def create_dashboard_app(state: DashboardState) -> FastAPI:
    """Create and configure the dashboard FastAPI application.

    Args:
        state: Shared dashboard state connected to the trading engine.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Mefai Autotrade Dashboard",
        version="1.0.0",
        docs_url="/dashboard/docs",
        redoc_url="/dashboard/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = WebSocketManager()
    _broadcast_task: Optional[asyncio.Task] = None

    # -------------------------------------------------------------------
    # Startup / Shutdown
    # -------------------------------------------------------------------

    @app.on_event("startup")
    async def on_startup():
        nonlocal _broadcast_task
        _broadcast_task = asyncio.create_task(_periodic_broadcast(state, ws_manager))
        logger.info(f"Dashboard API started on port {DASHBOARD_PORT}")

    @app.on_event("shutdown")
    async def on_shutdown():
        if _broadcast_task:
            _broadcast_task.cancel()
            try:
                await _broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("Dashboard API stopped")

    # -------------------------------------------------------------------
    # GET /dashboard/overview
    # -------------------------------------------------------------------

    @app.get("/dashboard/overview", dependencies=[Depends(_verify_api_key)])
    async def get_overview():
        """System overview - total PnL, positions, strategies, status."""
        portfolio = state.get_portfolio_summary()
        positions = state.get_positions()

        overview = {
            "trading_status": state.trading_status.value,
            "portfolio_value": portfolio["portfolio_value"],
            "daily_pnl": portfolio["daily_pnl"],
            "margin_used_pct": portfolio["margin_used_pct"],
            "open_positions": len(positions),
            "active_strategies": len(state.get_strategies()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if state.performance_tracker:
            perf = state.performance_tracker.get_overview()
            overview.update({
                "total_pnl": perf.get("total_pnl", 0),
                "total_return_pct": perf.get("total_return_pct", 0),
                "current_drawdown_pct": perf.get("current_drawdown_pct", 0),
                "max_drawdown_pct": perf.get("max_drawdown_pct", 0),
                "total_trades": perf.get("total_trades", 0),
                "win_rate": perf.get("win_rate", 0),
                "profit_factor": perf.get("profit_factor", 0),
                "sharpe_ratio": perf.get("sharpe_ratio", 0),
                "sortino_ratio": perf.get("sortino_ratio", 0),
            })

        if state.health_checker:
            overview["system_healthy"] = state.health_checker.is_healthy()
            overview["uptime_seconds"] = state.health_checker.get_uptime()

        if state.alert_manager:
            overview["alert_counts"] = state.alert_manager.get_alert_counts()

        if state.trading_status == TradingStatus.PAUSED:
            overview["paused_reason"] = state.paused_reason
        elif state.trading_status == TradingStatus.EMERGENCY_STOP:
            overview["emergency_stop_reason"] = state.emergency_stop_reason

        return overview

    # -------------------------------------------------------------------
    # GET /dashboard/equity
    # -------------------------------------------------------------------

    @app.get("/dashboard/equity", dependencies=[Depends(_verify_api_key)])
    async def get_equity(
        resolution: str = Query("hourly", regex="^(hourly|daily|weekly)$"),
        limit: int = Query(500, ge=1, le=5000),
    ):
        """Equity curve data."""
        if not state.performance_tracker:
            return {"data": [], "resolution": resolution}

        data = state.performance_tracker.get_equity_curve(resolution, limit)
        return {
            "data": data,
            "resolution": resolution,
            "count": len(data),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/positions
    # -------------------------------------------------------------------

    @app.get("/dashboard/positions", dependencies=[Depends(_verify_api_key)])
    async def get_positions():
        """Current open positions with unrealized PnL."""
        positions = state.get_positions()
        return {
            "positions": positions,
            "count": len(positions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/trades
    # -------------------------------------------------------------------

    @app.get("/dashboard/trades", dependencies=[Depends(_verify_api_key)])
    async def get_trades(
        limit: int = Query(50, ge=1, le=500),
        strategy: Optional[str] = Query(None),
        symbol: Optional[str] = Query(None),
        side: Optional[str] = Query(None),
        min_pnl: Optional[float] = Query(None),
        max_pnl: Optional[float] = Query(None),
    ):
        """Recent trade history with filters."""
        if state.performance_tracker:
            trades = state.performance_tracker.get_recent_trades(limit=500)
        else:
            trades = state.get_trades()

        # Apply filters
        if strategy:
            trades = [t for t in trades if t.get("strategy") == strategy]
        if symbol:
            trades = [t for t in trades if t.get("symbol") == symbol]
        if side:
            trades = [t for t in trades if t.get("side", "").upper() == side.upper()]
        if min_pnl is not None:
            trades = [t for t in trades if t.get("net_pnl", 0) >= min_pnl]
        if max_pnl is not None:
            trades = [t for t in trades if t.get("net_pnl", 0) <= max_pnl]

        trades = trades[:limit]

        return {
            "trades": trades,
            "count": len(trades),
            "filters": {
                "strategy": strategy,
                "symbol": symbol,
                "side": side,
                "min_pnl": min_pnl,
                "max_pnl": max_pnl,
            },
        }

    # -------------------------------------------------------------------
    # GET /dashboard/strategies
    # -------------------------------------------------------------------

    @app.get("/dashboard/strategies", dependencies=[Depends(_verify_api_key)])
    async def get_strategies():
        """Per-strategy performance breakdown."""
        result = {}

        if state.performance_tracker:
            result = state.performance_tracker.get_all_strategy_metrics()

        # Merge with live strategy state
        live_strategies = state.get_strategies()
        for name, live_data in live_strategies.items():
            if name in result:
                result[name]["live_state"] = live_data
            else:
                result[name] = {"live_state": live_data}

        return {
            "strategies": result,
            "count": len(result),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/risk
    # -------------------------------------------------------------------

    @app.get("/dashboard/risk", dependencies=[Depends(_verify_api_key)])
    async def get_risk():
        """Current risk metrics."""
        portfolio = state.get_portfolio_summary()
        positions = state.get_positions()

        risk_data = {
            "portfolio_value": portfolio["portfolio_value"],
            "margin_used_pct": portfolio["margin_used_pct"],
            "open_positions": len(positions),
            "total_exposure": sum(
                abs(p.get("notional", 0)) for p in positions
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if state.performance_tracker:
            dd = state.performance_tracker.get_drawdown_analysis()
            risk_data["drawdown"] = dd

            streak = state.performance_tracker.get_streak_info()
            risk_data["streak"] = {
                "current_streak": streak.current_streak,
                "longest_win_streak": streak.longest_win_streak,
                "longest_loss_streak": streak.longest_loss_streak,
                "current_streak_pnl": streak.current_streak_pnl,
            }

        if state.alert_manager:
            active_alerts = state.alert_manager.get_active_alerts()
            risk_data["active_alerts"] = [a.to_dict() for a in active_alerts]
            risk_data["alert_counts"] = state.alert_manager.get_alert_counts()

        return risk_data

    # -------------------------------------------------------------------
    # GET /dashboard/orders
    # -------------------------------------------------------------------

    @app.get("/dashboard/orders", dependencies=[Depends(_verify_api_key)])
    async def get_orders():
        """Open and recent orders."""
        orders = state.get_orders()
        return {
            "orders": orders,
            "count": len(orders),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/performance
    # -------------------------------------------------------------------

    @app.get("/dashboard/performance", dependencies=[Depends(_verify_api_key)])
    async def get_performance(
        window: str = Query("all", regex="^(1h|24h|7d|30d|all)$"),
    ):
        """Performance metrics - Sharpe, Sortino, win rate, profit factor, etc."""
        if not state.performance_tracker:
            return {"error": "Performance tracker not initialized"}

        metrics = state.performance_tracker.get_rolling_metrics(window)
        result = metrics.to_dict()

        # Add benchmark comparison
        benchmark = state.performance_tracker.get_benchmark_comparison()
        result["benchmark"] = benchmark

        # Add distribution stats
        distribution = state.performance_tracker.get_trade_distribution()
        result["distribution"] = distribution

        return result

    # -------------------------------------------------------------------
    # GET /dashboard/calendar
    # -------------------------------------------------------------------

    @app.get("/dashboard/calendar", dependencies=[Depends(_verify_api_key)])
    async def get_calendar():
        """Monthly returns heatmap data."""
        if not state.performance_tracker:
            return {"months": []}

        monthly = state.performance_tracker.get_monthly_returns()
        return {
            "months": [
                {
                    "year": m.year,
                    "month": m.month,
                    "return_pct": m.return_pct,
                    "trade_count": m.trade_count,
                    "win_count": m.win_count,
                    "loss_count": m.loss_count,
                    "total_pnl": m.total_pnl,
                    "best_trade_pnl": m.best_trade_pnl,
                    "worst_trade_pnl": m.worst_trade_pnl,
                }
                for m in monthly
            ],
            "count": len(monthly),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/symbols
    # -------------------------------------------------------------------

    @app.get("/dashboard/symbols", dependencies=[Depends(_verify_api_key)])
    async def get_symbols():
        """Per-symbol performance breakdown."""
        if not state.performance_tracker:
            return {"symbols": {}}

        return {
            "symbols": state.performance_tracker.get_all_symbol_metrics(),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/health
    # -------------------------------------------------------------------

    @app.get("/dashboard/health", dependencies=[Depends(_verify_api_key)])
    async def get_health():
        """System health check results."""
        if not state.health_checker:
            return {"status": "UNKNOWN", "message": "Health checker not initialized"}

        report = await state.health_checker.run_all_checks()
        return report.to_dict()

    # -------------------------------------------------------------------
    # GET /dashboard/alerts
    # -------------------------------------------------------------------

    @app.get("/dashboard/alerts", dependencies=[Depends(_verify_api_key)])
    async def get_alerts(
        severity: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        include_history: bool = Query(False),
    ):
        """Active alerts and optional history."""
        if not state.alert_manager:
            return {"active": [], "history": []}

        sev = None
        if severity:
            try:
                from .alerting import AlertSeverity
                sev = AlertSeverity(severity.upper())
            except (ValueError, ImportError):
                pass

        active = state.alert_manager.get_active_alerts(severity=sev)
        result = {
            "active": [a.to_dict() for a in active],
            "active_count": len(active),
            "counts": state.alert_manager.get_alert_counts(),
        }

        if include_history:
            history = state.alert_manager.get_alert_history(limit=limit, severity=sev)
            result["history"] = [a.to_dict() for a in history]
            result["history_count"] = len(history)

        return result

    # -------------------------------------------------------------------
    # POST /dashboard/alerts/acknowledge
    # -------------------------------------------------------------------

    @app.post("/dashboard/alerts/acknowledge", dependencies=[Depends(_verify_api_key)])
    async def acknowledge_alert(req: AlertAckRequest):
        """Acknowledge an active alert."""
        if not state.alert_manager:
            raise HTTPException(400, "Alert manager not initialized")

        success = state.alert_manager.acknowledge(req.alert_id, req.user)
        if not success:
            raise HTTPException(404, f"Alert {req.alert_id} not found or not active")

        return {"status": "acknowledged", "alert_id": req.alert_id}

    # -------------------------------------------------------------------
    # POST /dashboard/alerts/resolve
    # -------------------------------------------------------------------

    @app.post("/dashboard/alerts/resolve", dependencies=[Depends(_verify_api_key)])
    async def resolve_alert(req: AlertResolveRequest):
        """Manually resolve an alert."""
        if not state.alert_manager:
            raise HTTPException(400, "Alert manager not initialized")

        success = state.alert_manager.resolve(req.alert_id)
        if not success:
            raise HTTPException(404, f"Alert {req.alert_id} not found")

        return {"status": "resolved", "alert_id": req.alert_id}

    # -------------------------------------------------------------------
    # POST /dashboard/alerts/rules
    # -------------------------------------------------------------------

    @app.post("/dashboard/alerts/rules", dependencies=[Depends(_verify_api_key)])
    async def toggle_alert_rule(req: RuleToggleRequest):
        """Enable or disable an alert rule."""
        if not state.alert_manager:
            raise HTTPException(400, "Alert manager not initialized")

        if req.enabled:
            state.alert_manager.enable_rule(req.rule_name)
        else:
            state.alert_manager.disable_rule(req.rule_name)

        return {
            "status": "ok",
            "rule_name": req.rule_name,
            "enabled": req.enabled,
        }

    # -------------------------------------------------------------------
    # GET /dashboard/alerts/rules
    # -------------------------------------------------------------------

    @app.get("/dashboard/alerts/rules", dependencies=[Depends(_verify_api_key)])
    async def get_alert_rules():
        """Get status of all alert rules."""
        if not state.alert_manager:
            return {"rules": []}

        return {"rules": state.alert_manager.get_rules_status()}

    # -------------------------------------------------------------------
    # POST /dashboard/pause
    # -------------------------------------------------------------------

    @app.post("/dashboard/pause", dependencies=[Depends(_verify_api_key)])
    async def pause_trading(req: PauseRequest):
        """Pause or resume trading."""
        if req.paused:
            state.trading_status = TradingStatus.PAUSED
            state.paused_reason = req.reason
            logger.warning(f"Trading PAUSED via dashboard: {req.reason}")

            if state._pause_callback:
                try:
                    if asyncio.iscoroutinefunction(state._pause_callback):
                        await state._pause_callback(req.reason)
                    else:
                        state._pause_callback(req.reason)
                except Exception as e:
                    logger.error(f"Pause callback failed: {e}")

            # Broadcast to WebSocket clients
            await ws_manager.broadcast({
                "type": "trading_status",
                "status": "PAUSED",
                "reason": req.reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {"status": "PAUSED", "reason": req.reason}
        else:
            state.trading_status = TradingStatus.RUNNING
            state.paused_reason = ""
            logger.info("Trading RESUMED via dashboard")

            if state._resume_callback:
                try:
                    if asyncio.iscoroutinefunction(state._resume_callback):
                        await state._resume_callback()
                    else:
                        state._resume_callback()
                except Exception as e:
                    logger.error(f"Resume callback failed: {e}")

            await ws_manager.broadcast({
                "type": "trading_status",
                "status": "RUNNING",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {"status": "RUNNING"}

    # -------------------------------------------------------------------
    # POST /dashboard/emergency-stop
    # -------------------------------------------------------------------

    @app.post("/dashboard/emergency-stop", dependencies=[Depends(_verify_api_key)])
    async def emergency_stop(req: EmergencyStopRequest):
        """Emergency stop - halt all trading and optionally close positions."""
        state.trading_status = TradingStatus.EMERGENCY_STOP
        state.emergency_stop_reason = req.reason
        logger.critical(
            f"EMERGENCY STOP via dashboard: {req.reason} "
            f"(close_positions={req.close_positions})"
        )

        if state._emergency_stop_callback:
            try:
                if asyncio.iscoroutinefunction(state._emergency_stop_callback):
                    await state._emergency_stop_callback(req.reason, req.close_positions)
                else:
                    state._emergency_stop_callback(req.reason, req.close_positions)
            except Exception as e:
                logger.error(f"Emergency stop callback failed: {e}")

        await ws_manager.broadcast({
            "type": "emergency_stop",
            "reason": req.reason,
            "close_positions": req.close_positions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Notify
        if state.notifier:
            try:
                await state.notifier.notify_alert(
                    "CRITICAL",
                    "EMERGENCY STOP",
                    f"Trading halted: {req.reason}",
                    {"close_positions": req.close_positions},
                )
            except Exception:
                pass

        return {
            "status": "EMERGENCY_STOP",
            "reason": req.reason,
            "close_positions": req.close_positions,
        }

    # -------------------------------------------------------------------
    # GET /dashboard/metrics
    # -------------------------------------------------------------------

    @app.get("/dashboard/metrics", dependencies=[Depends(_verify_api_key)])
    async def get_metrics(format: str = Query("json", regex="^(json|prometheus|csv)$")):
        """Export collected metrics."""
        if not state.metrics_collector:
            return {"error": "Metrics collector not initialized"}

        if format == "prometheus":
            return PlainTextResponse(
                state.metrics_collector.export_prometheus(),
                media_type="text/plain",
            )
        elif format == "csv":
            return PlainTextResponse(
                state.metrics_collector.export_csv(),
                media_type="text/csv",
            )
        else:
            return state.metrics_collector.export_json()

    # -------------------------------------------------------------------
    # GET /dashboard/notifications
    # -------------------------------------------------------------------

    @app.get("/dashboard/notifications", dependencies=[Depends(_verify_api_key)])
    async def get_notification_stats():
        """Notification system statistics."""
        if not state.notifier:
            return {"error": "Notifier not initialized"}

        return state.notifier.get_stats()

    # -------------------------------------------------------------------
    # GET /dashboard/time-analysis
    # -------------------------------------------------------------------

    @app.get("/dashboard/time-analysis", dependencies=[Depends(_verify_api_key)])
    async def get_time_analysis():
        """Performance by hour of day and day of week."""
        if not state.performance_tracker:
            return {"error": "Performance tracker not initialized"}

        ta = state.performance_tracker.get_time_analysis()
        return {
            "best_hour": ta.best_hour,
            "worst_hour": ta.worst_hour,
            "best_day_of_week": ta.best_day_of_week,
            "worst_day_of_week": ta.worst_day_of_week,
            "hourly_pnl": ta.hourly_pnl,
            "hourly_trades": ta.hourly_trades,
            "hourly_win_rate": ta.hourly_win_rate,
            "daily_pnl": ta.daily_pnl,
            "daily_trades": ta.daily_trades,
        }

    # -------------------------------------------------------------------
    # GET /dashboard/timeframes
    # -------------------------------------------------------------------

    @app.get("/dashboard/timeframes", dependencies=[Depends(_verify_api_key)])
    async def get_timeframes():
        """Per-timeframe performance."""
        if not state.performance_tracker:
            return {"timeframes": {}}

        return {
            "timeframes": state.performance_tracker.get_all_timeframe_metrics(),
        }

    # -------------------------------------------------------------------
    # GET /dashboard/benchmark
    # -------------------------------------------------------------------

    @app.get("/dashboard/benchmark", dependencies=[Depends(_verify_api_key)])
    async def get_benchmark():
        """Strategy vs benchmark comparison."""
        if not state.performance_tracker:
            return {"error": "Performance tracker not initialized"}

        return state.performance_tracker.get_benchmark_comparison()

    # -------------------------------------------------------------------
    # WebSocket /ws/dashboard
    # -------------------------------------------------------------------

    @app.websocket("/ws/dashboard")
    async def websocket_endpoint(websocket: WebSocket):
        """Real-time dashboard WebSocket.

        Sends periodic updates with current PnL, positions, alerts.
        Accepts commands from clients:
          - {"type": "subscribe", "channels": ["pnl", "positions", "alerts"]}
          - {"type": "ping"}
        """
        # API key verification for WebSocket
        api_key = websocket.query_params.get("api_key", "")
        if DEFAULT_API_KEY and not hmac.compare_digest(api_key, DEFAULT_API_KEY):
            await websocket.close(code=4001, reason="Invalid API key")
            return

        await ws_manager.connect(websocket)

        try:
            # Send initial state
            await websocket.send_json({
                "type": "connected",
                "trading_status": state.trading_status.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_json(), timeout=30.0
                    )

                    msg_type = data.get("type", "")

                    if msg_type == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    elif msg_type == "subscribe":
                        # Client subscribes to specific channels
                        channels = data.get("channels", [])
                        await websocket.send_json({
                            "type": "subscribed",
                            "channels": channels,
                        })
                    elif msg_type == "get_snapshot":
                        # Send full current state
                        snapshot = _build_snapshot(state)
                        await websocket.send_json({
                            "type": "snapshot",
                            "data": snapshot,
                        })

                except asyncio.TimeoutError:
                    # Send keepalive
                    try:
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                    except Exception:
                        break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Dashboard WebSocket error: {e}")
        finally:
            await ws_manager.disconnect(websocket)

    # -------------------------------------------------------------------
    # Health endpoint (no auth)
    # -------------------------------------------------------------------

    @app.get("/health")
    async def health_check():
        """Simple health check endpoint (no auth required)."""
        return {
            "status": "ok",
            "service": "mefai-autotrade-dashboard",
            "trading_status": state.trading_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return app


# ---------------------------------------------------------------------------
# Periodic Broadcast
# ---------------------------------------------------------------------------

async def _periodic_broadcast(
    state: DashboardState,
    ws_manager: WebSocketManager,
    interval: float = 2.0,
) -> None:
    """Periodically broadcast state updates to WebSocket clients."""
    while True:
        try:
            if ws_manager.connection_count > 0:
                update = _build_ws_update(state)
                await ws_manager.broadcast(update)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

        await asyncio.sleep(interval)


def _build_ws_update(state: DashboardState) -> Dict[str, Any]:
    """Build a WebSocket update message."""
    portfolio = state.get_portfolio_summary()
    positions = state.get_positions()

    update: Dict[str, Any] = {
        "type": "update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trading_status": state.trading_status.value,
        "portfolio": portfolio,
        "position_count": len(positions),
    }

    if state.performance_tracker:
        overview = state.performance_tracker.get_overview()
        update["pnl"] = {
            "total": overview.get("total_pnl", 0),
            "daily": portfolio["daily_pnl"],
            "unrealized": overview.get("unrealized_pnl", 0),
            "drawdown_pct": overview.get("current_drawdown_pct", 0),
        }
        update["metrics"] = {
            "win_rate": overview.get("win_rate", 0),
            "total_trades": overview.get("total_trades", 0),
            "profit_factor": overview.get("profit_factor", 0),
        }

    if state.alert_manager:
        active = state.alert_manager.get_active_alerts()
        if active:
            update["alerts"] = [
                {
                    "id": a.alert_id,
                    "severity": a.severity.value,
                    "title": a.title,
                    "state": a.state.value,
                }
                for a in active[:10]
            ]

    return update


def _build_snapshot(state: DashboardState) -> Dict[str, Any]:
    """Build a full state snapshot for WebSocket clients."""
    portfolio = state.get_portfolio_summary()
    positions = state.get_positions()
    orders = state.get_orders()

    snapshot: Dict[str, Any] = {
        "trading_status": state.trading_status.value,
        "portfolio": portfolio,
        "positions": positions,
        "orders": orders,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if state.performance_tracker:
        snapshot["overview"] = state.performance_tracker.get_overview()
        snapshot["drawdown"] = state.performance_tracker.get_drawdown_analysis()
        snapshot["streak"] = {
            "current_streak": state.performance_tracker.get_streak_info().current_streak,
            "longest_win": state.performance_tracker.get_streak_info().longest_win_streak,
            "longest_loss": state.performance_tracker.get_streak_info().longest_loss_streak,
        }

    if state.alert_manager:
        active = state.alert_manager.get_active_alerts()
        snapshot["alerts"] = [a.to_dict() for a in active]
        snapshot["alert_counts"] = state.alert_manager.get_alert_counts()

    if state.health_checker:
        snapshot["health"] = {
            "healthy": state.health_checker.is_healthy(),
            "uptime": state.health_checker.get_uptime(),
        }

    return snapshot
