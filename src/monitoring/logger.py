"""
Structured Logging System for Mefai Autotrade.
JSON structured logging with separate log files, rotation, correlation IDs,
and specialized loggers for trades, orders, risk, system, and audit events.
"""

import json
import logging
import logging.handlers
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LOG_DIR = "/var/log/mefai-autotrade"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
DEFAULT_BACKUP_COUNT = 30  # 30 rotated files
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

_correlation_id: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
_trace_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "trace_context", default=None
)


# ---------------------------------------------------------------------------
# Correlation Context
# ---------------------------------------------------------------------------

class CorrelationContext:
    """Context manager that sets a correlation ID for the duration of a scope.
    Useful for tracing a single order flow across all loggers."""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())[:12]
        self._token = None

    def __enter__(self):
        self._token = _correlation_id.set(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _correlation_id.reset(self._token)
        return False

    @staticmethod
    def get_current() -> Optional[str]:
        return _correlation_id.get()

    @staticmethod
    def set_trace_context(context: Dict[str, Any]):
        _trace_context.set(context)

    @staticmethod
    def get_trace_context() -> Optional[Dict[str, Any]]:
        return _trace_context.get()


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def __init__(self, service_name: str = "mefai-autotrade"):
        super().__init__()
        self.service_name = service_name
        self._hostname = os.uname().nodename

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime(DEFAULT_DATE_FORMAT)[:-3],
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "hostname": self._hostname,
            "pid": record.process,
            "thread": record.threadName,
        }

        # Add correlation ID if present
        cid = _correlation_id.get()
        if cid:
            log_entry["correlation_id"] = cid

        # Add trace context if present
        trace = _trace_context.get()
        if trace:
            log_entry["trace"] = trace

        # Add extra fields from the record
        if hasattr(record, "extra_data") and record.extra_data:
            log_entry["data"] = record.extra_data

        # Add exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Custom Log Record Factory
# ---------------------------------------------------------------------------

class ExtraDataAdapter(logging.LoggerAdapter):
    """Logger adapter that supports extra structured data."""

    def process(self, msg, kwargs):
        extra_data = kwargs.pop("extra_data", None)
        if extra_data is None:
            extra_data = self.extra.get("extra_data")
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["extra_data"] = extra_data
        return msg, kwargs


# ---------------------------------------------------------------------------
# Log File Definitions
# ---------------------------------------------------------------------------

class LogFile(Enum):
    TRADES = "trades.log"
    ORDERS = "orders.log"
    RISK = "risk.log"
    SYSTEM = "system.log"
    ERRORS = "errors.log"
    AUDIT = "audit.log"
    PERFORMANCE = "performance.log"
    DEBUG = "debug.log"


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

_initialized = False
_lock = threading.Lock()
_loggers: Dict[str, ExtraDataAdapter] = {}


def setup_logging(
    log_dir: str = DEFAULT_LOG_DIR,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    service_name: str = "mefai-autotrade",
) -> None:
    """Initialize the logging system with structured JSON logging.

    Creates separate log files for different concerns and sets up
    rotation, formatting, and console output.

    Args:
        log_dir: Directory for log files.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.
        max_bytes: Maximum bytes per log file before rotation.
        backup_count: Number of rotated backup files to keep.
        service_name: Service identifier in log entries.
    """
    global _initialized

    with _lock:
        if _initialized:
            return

        # Ensure log directory exists
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        json_formatter = JSONFormatter(service_name=service_name)

        # Console handler with human-readable format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)

        # Create file handlers for each log type
        file_handlers: Dict[str, logging.Handler] = {}
        for log_file in LogFile:
            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path / log_file.value),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setLevel(file_level)
            handler.setFormatter(json_formatter)
            file_handlers[log_file.name] = handler

        # Error handler captures WARNING+ from all loggers
        error_handler = file_handlers["ERRORS"]
        error_handler.setLevel(logging.WARNING)

        # Configure root mefai logger
        root_logger = logging.getLogger("mefai.autotrade")
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handlers["SYSTEM"])
        root_logger.addHandler(error_handler)

        # Configure specialized loggers
        logger_map = {
            "mefai.autotrade.trades": file_handlers["TRADES"],
            "mefai.autotrade.orders": file_handlers["ORDERS"],
            "mefai.autotrade.risk": file_handlers["RISK"],
            "mefai.autotrade.audit": file_handlers["AUDIT"],
            "mefai.autotrade.performance": file_handlers["PERFORMANCE"],
            "mefai.autotrade.debug": file_handlers["DEBUG"],
        }

        for logger_name, handler in logger_map.items():
            lg = logging.getLogger(logger_name)
            lg.addHandler(handler)
            lg.addHandler(console_handler)
            lg.addHandler(error_handler)
            lg.setLevel(logging.DEBUG)
            lg.propagate = False

        _initialized = True


# ---------------------------------------------------------------------------
# Logger Getters
# ---------------------------------------------------------------------------

def _get_adapter(name: str) -> ExtraDataAdapter:
    if name not in _loggers:
        base = logging.getLogger(name)
        _loggers[name] = ExtraDataAdapter(base, {"extra_data": None})
    return _loggers[name]


def get_trade_logger() -> "TradeLogger":
    return TradeLogger()


def get_order_logger() -> "OrderLogger":
    return OrderLogger()


def get_risk_logger() -> "RiskLogger":
    return RiskLogger()


def get_system_logger() -> "SystemLogger":
    return SystemLogger()


def get_audit_logger() -> "AuditLogger":
    return AuditLogger()


# ---------------------------------------------------------------------------
# Trade Logger
# ---------------------------------------------------------------------------

class TradeLogger:
    """Specialized logger for trade events with full context."""

    def __init__(self):
        self._logger = _get_adapter("mefai.autotrade.trades")

    def trade_opened(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        signal_id: Optional[str] = None,
        timeframe: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_check_result: Optional[Dict] = None,
        leverage: int = 1,
    ) -> None:
        data = {
            "event": "trade_opened",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "notional": entry_price * quantity,
            "strategy": strategy,
            "signal_id": signal_id,
            "timeframe": timeframe,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
            "risk_check": risk_check_result,
        }
        self._logger.info(
            f"Trade opened: {side} {quantity} {symbol} @ {entry_price} "
            f"(strategy={strategy})",
            extra_data=data,
        )

    def trade_closed(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        realized_pnl: float,
        pnl_percent: float,
        duration_seconds: float,
        strategy: str,
        close_reason: str,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        data = {
            "event": "trade_closed",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "realized_pnl": realized_pnl,
            "pnl_percent": pnl_percent,
            "duration_seconds": duration_seconds,
            "duration_human": self._format_duration(duration_seconds),
            "strategy": strategy,
            "close_reason": close_reason,
            "fees": fees,
            "slippage": slippage,
            "net_pnl": realized_pnl - fees,
        }
        pnl_str = f"+{realized_pnl:.2f}" if realized_pnl >= 0 else f"{realized_pnl:.2f}"
        self._logger.info(
            f"Trade closed: {side} {symbol} PnL={pnl_str} USDT "
            f"({pnl_percent:+.2f}%) duration={data['duration_human']} "
            f"reason={close_reason}",
            extra_data=data,
        )

    def trade_modified(
        self,
        trade_id: str,
        symbol: str,
        modification: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        data = {
            "event": "trade_modified",
            "trade_id": trade_id,
            "symbol": symbol,
            "modification": modification,
            "old_value": old_value,
            "new_value": new_value,
        }
        self._logger.info(
            f"Trade modified: {symbol} {modification} "
            f"{old_value} -> {new_value}",
            extra_data=data,
        )

    def partial_close(
        self,
        trade_id: str,
        symbol: str,
        closed_quantity: float,
        remaining_quantity: float,
        partial_pnl: float,
    ) -> None:
        data = {
            "event": "partial_close",
            "trade_id": trade_id,
            "symbol": symbol,
            "closed_quantity": closed_quantity,
            "remaining_quantity": remaining_quantity,
            "partial_pnl": partial_pnl,
        }
        self._logger.info(
            f"Partial close: {symbol} closed={closed_quantity} "
            f"remaining={remaining_quantity} pnl={partial_pnl:+.2f}",
            extra_data=data,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        elif seconds < 86400:
            return f"{seconds / 3600:.1f}h"
        else:
            return f"{seconds / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Order Logger
# ---------------------------------------------------------------------------

class OrderLogger:
    """Specialized logger for order lifecycle events."""

    def __init__(self):
        self._logger = _get_adapter("mefai.autotrade.orders")

    def order_created(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        strategy: Optional[str] = None,
    ) -> None:
        data = {
            "event": "order_created",
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "time_in_force": time_in_force,
            "strategy": strategy,
        }
        price_str = f" @ {price}" if price else ""
        self._logger.info(
            f"Order created: {side} {order_type} {quantity} {symbol}{price_str}",
            extra_data=data,
        )

    def order_submitted(
        self,
        order_id: str,
        exchange_order_id: str,
        latency_ms: float,
    ) -> None:
        data = {
            "event": "order_submitted",
            "order_id": order_id,
            "exchange_order_id": exchange_order_id,
            "submission_latency_ms": latency_ms,
        }
        self._logger.info(
            f"Order submitted: {order_id} -> {exchange_order_id} "
            f"(latency={latency_ms:.1f}ms)",
            extra_data=data,
        )

    def order_filled(
        self,
        order_id: str,
        exchange_order_id: str,
        fill_price: float,
        fill_quantity: float,
        fee: float,
        fee_asset: str,
        slippage_bps: float = 0.0,
    ) -> None:
        data = {
            "event": "order_filled",
            "order_id": order_id,
            "exchange_order_id": exchange_order_id,
            "fill_price": fill_price,
            "fill_quantity": fill_quantity,
            "fee": fee,
            "fee_asset": fee_asset,
            "slippage_bps": slippage_bps,
        }
        self._logger.info(
            f"Order filled: {order_id} qty={fill_quantity} @ {fill_price} "
            f"fee={fee} {fee_asset} slippage={slippage_bps:.1f}bps",
            extra_data=data,
        )

    def order_partially_filled(
        self,
        order_id: str,
        filled_quantity: float,
        remaining_quantity: float,
        fill_price: float,
    ) -> None:
        data = {
            "event": "order_partially_filled",
            "order_id": order_id,
            "filled_quantity": filled_quantity,
            "remaining_quantity": remaining_quantity,
            "fill_price": fill_price,
        }
        self._logger.info(
            f"Order partial fill: {order_id} filled={filled_quantity} "
            f"remaining={remaining_quantity} @ {fill_price}",
            extra_data=data,
        )

    def order_cancelled(
        self,
        order_id: str,
        reason: str,
        filled_quantity: float = 0.0,
    ) -> None:
        data = {
            "event": "order_cancelled",
            "order_id": order_id,
            "reason": reason,
            "filled_quantity": filled_quantity,
        }
        self._logger.info(
            f"Order cancelled: {order_id} reason={reason}",
            extra_data=data,
        )

    def order_rejected(
        self,
        order_id: str,
        reason: str,
        error_code: Optional[str] = None,
    ) -> None:
        data = {
            "event": "order_rejected",
            "order_id": order_id,
            "reason": reason,
            "error_code": error_code,
        }
        self._logger.warning(
            f"Order rejected: {order_id} reason={reason} code={error_code}",
            extra_data=data,
        )

    def order_timeout(self, order_id: str, elapsed_seconds: float) -> None:
        data = {
            "event": "order_timeout",
            "order_id": order_id,
            "elapsed_seconds": elapsed_seconds,
        }
        self._logger.warning(
            f"Order timeout: {order_id} after {elapsed_seconds:.1f}s",
            extra_data=data,
        )


# ---------------------------------------------------------------------------
# Risk Logger
# ---------------------------------------------------------------------------

class RiskLogger:
    """Specialized logger for risk management events."""

    def __init__(self):
        self._logger = _get_adapter("mefai.autotrade.risk")

    def risk_check_passed(
        self,
        check_name: str,
        details: Dict[str, Any],
    ) -> None:
        data = {"event": "risk_check_passed", "check": check_name, **details}
        self._logger.debug(
            f"Risk check passed: {check_name}",
            extra_data=data,
        )

    def risk_check_failed(
        self,
        check_name: str,
        reason: str,
        details: Dict[str, Any],
    ) -> None:
        data = {
            "event": "risk_check_failed",
            "check": check_name,
            "reason": reason,
            **details,
        }
        self._logger.warning(
            f"Risk check FAILED: {check_name} - {reason}",
            extra_data=data,
        )

    def drawdown_update(
        self,
        current_drawdown: float,
        max_drawdown: float,
        peak_equity: float,
        current_equity: float,
    ) -> None:
        data = {
            "event": "drawdown_update",
            "current_drawdown_pct": current_drawdown,
            "max_drawdown_pct": max_drawdown,
            "peak_equity": peak_equity,
            "current_equity": current_equity,
        }
        level = logging.WARNING if current_drawdown > 5.0 else logging.DEBUG
        self._logger.log(
            level,
            f"Drawdown: current={current_drawdown:.2f}% "
            f"max={max_drawdown:.2f}%",
            extra_data=data,
        )

    def circuit_breaker_triggered(
        self,
        reason: str,
        threshold: float,
        current_value: float,
    ) -> None:
        data = {
            "event": "circuit_breaker_triggered",
            "reason": reason,
            "threshold": threshold,
            "current_value": current_value,
        }
        self._logger.critical(
            f"CIRCUIT BREAKER: {reason} (threshold={threshold}, "
            f"current={current_value})",
            extra_data=data,
        )

    def exposure_update(
        self,
        total_exposure: float,
        max_exposure: float,
        positions: int,
        margin_used_pct: float,
    ) -> None:
        data = {
            "event": "exposure_update",
            "total_exposure": total_exposure,
            "max_exposure": max_exposure,
            "positions": positions,
            "margin_used_pct": margin_used_pct,
        }
        self._logger.debug(
            f"Exposure: {total_exposure:.2f}/{max_exposure:.2f} "
            f"positions={positions} margin={margin_used_pct:.1f}%",
            extra_data=data,
        )

    def position_size_calculated(
        self,
        symbol: str,
        risk_amount: float,
        position_size: float,
        stop_distance: float,
        leverage: int,
    ) -> None:
        data = {
            "event": "position_size_calculated",
            "symbol": symbol,
            "risk_amount": risk_amount,
            "position_size": position_size,
            "stop_distance": stop_distance,
            "leverage": leverage,
        }
        self._logger.debug(
            f"Position size: {symbol} size={position_size} "
            f"risk={risk_amount} leverage={leverage}x",
            extra_data=data,
        )


# ---------------------------------------------------------------------------
# System Logger
# ---------------------------------------------------------------------------

class SystemLogger:
    """Specialized logger for system-level events."""

    def __init__(self):
        self._logger = _get_adapter("mefai.autotrade")

    def startup(self, config: Dict[str, Any]) -> None:
        # Redact sensitive fields
        safe_config = {
            k: "***" if "key" in k.lower() or "secret" in k.lower() or "token" in k.lower() else v
            for k, v in config.items()
        }
        data = {"event": "startup", "config": safe_config}
        self._logger.info("System starting up", extra_data=data)

    def shutdown(self, reason: str = "normal") -> None:
        data = {"event": "shutdown", "reason": reason}
        self._logger.info(f"System shutting down: {reason}", extra_data=data)

    def strategy_loaded(self, strategy_name: str, params: Dict[str, Any]) -> None:
        data = {
            "event": "strategy_loaded",
            "strategy": strategy_name,
            "params": params,
        }
        self._logger.info(
            f"Strategy loaded: {strategy_name}",
            extra_data=data,
        )

    def strategy_error(
        self,
        strategy_name: str,
        error: Exception,
    ) -> None:
        data = {
            "event": "strategy_error",
            "strategy": strategy_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self._logger.error(
            f"Strategy error: {strategy_name} - {error}",
            extra_data=data,
            exc_info=True,
        )

    def exchange_connected(self, exchange: str, latency_ms: float) -> None:
        data = {
            "event": "exchange_connected",
            "exchange": exchange,
            "latency_ms": latency_ms,
        }
        self._logger.info(
            f"Exchange connected: {exchange} (latency={latency_ms:.0f}ms)",
            extra_data=data,
        )

    def exchange_disconnected(self, exchange: str, reason: str) -> None:
        data = {
            "event": "exchange_disconnected",
            "exchange": exchange,
            "reason": reason,
        }
        self._logger.warning(
            f"Exchange disconnected: {exchange} - {reason}",
            extra_data=data,
        )

    def exchange_error(self, exchange: str, error: str, code: Optional[str] = None) -> None:
        data = {
            "event": "exchange_error",
            "exchange": exchange,
            "error": error,
            "code": code,
        }
        self._logger.error(
            f"Exchange error: {exchange} - {error} (code={code})",
            extra_data=data,
        )

    def websocket_status(self, stream: str, status: str, details: Optional[str] = None) -> None:
        data = {
            "event": "websocket_status",
            "stream": stream,
            "status": status,
            "details": details,
        }
        level = logging.INFO if status == "connected" else logging.WARNING
        self._logger.log(
            level,
            f"WebSocket {stream}: {status}",
            extra_data=data,
        )

    def config_changed(self, key: str, old_value: Any, new_value: Any, user: str = "system") -> None:
        data = {
            "event": "config_changed",
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "changed_by": user,
        }
        self._logger.info(
            f"Config changed: {key} = {old_value} -> {new_value} (by {user})",
            extra_data=data,
        )

    def performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        perf_logger = _get_adapter("mefai.autotrade.performance")
        data = {
            "event": "performance_metric",
            "metric": metric_name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
        }
        perf_logger.debug(
            f"Metric: {metric_name}={value}{unit}",
            extra_data=data,
        )


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """Immutable audit log for compliance-relevant events."""

    def __init__(self):
        self._logger = _get_adapter("mefai.autotrade.audit")

    def _log(self, action: str, details: Dict[str, Any], user: str = "system") -> None:
        data = {
            "event": "audit",
            "action": action,
            "user": user,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self._logger.info(f"AUDIT: {action} by {user}", extra_data=data)

    def trading_paused(self, reason: str, user: str = "system") -> None:
        self._log("trading_paused", {"reason": reason}, user)

    def trading_resumed(self, user: str = "system") -> None:
        self._log("trading_resumed", {}, user)

    def emergency_stop(self, reason: str, user: str = "system") -> None:
        self._log("emergency_stop", {"reason": reason}, user)

    def config_modified(
        self,
        section: str,
        changes: Dict[str, Any],
        user: str = "system",
    ) -> None:
        self._log(
            "config_modified",
            {"section": section, "changes": changes},
            user,
        )

    def api_key_rotated(self, exchange: str, user: str = "system") -> None:
        self._log("api_key_rotated", {"exchange": exchange}, user)

    def manual_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        user: str = "manual",
    ) -> None:
        self._log(
            "manual_order",
            {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
            user,
        )

    def strategy_toggled(
        self,
        strategy: str,
        enabled: bool,
        user: str = "system",
    ) -> None:
        state = "enabled" if enabled else "disabled"
        self._log(
            "strategy_toggled",
            {"strategy": strategy, "state": state},
            user,
        )

    def risk_parameter_changed(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        user: str = "system",
    ) -> None:
        self._log(
            "risk_parameter_changed",
            {
                "parameter": parameter,
                "old_value": old_value,
                "new_value": new_value,
            },
            user,
        )


# ---------------------------------------------------------------------------
# Log Aggregation Helpers
# ---------------------------------------------------------------------------

class LogAggregator:
    """Utility to aggregate and summarize log entries from JSON log files."""

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)

    def count_events(
        self,
        log_file: str,
        event_type: Optional[str] = None,
        since_seconds: int = 3600,
    ) -> int:
        """Count events in a log file within a time window."""
        filepath = self.log_dir / log_file
        if not filepath.exists():
            return 0

        cutoff = time.time() - since_seconds
        count = 0

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts_str = entry.get("timestamp", "")
                    try:
                        ts = datetime.strptime(
                            ts_str, "%Y-%m-%d %H:%M:%S.%f"
                        ).replace(tzinfo=timezone.utc)
                        if ts.timestamp() < cutoff:
                            continue
                    except (ValueError, TypeError):
                        continue

                    if event_type:
                        data = entry.get("data", {})
                        if data.get("event") != event_type:
                            continue

                    count += 1
        except OSError:
            pass

        return count

    def get_error_summary(self, since_seconds: int = 3600) -> Dict[str, int]:
        """Get error counts by type from the error log."""
        filepath = self.log_dir / LogFile.ERRORS.value
        if not filepath.exists():
            return {}

        cutoff = time.time() - since_seconds
        errors: Dict[str, int] = {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts_str = entry.get("timestamp", "")
                    try:
                        ts = datetime.strptime(
                            ts_str, "%Y-%m-%d %H:%M:%S.%f"
                        ).replace(tzinfo=timezone.utc)
                        if ts.timestamp() < cutoff:
                            continue
                    except (ValueError, TypeError):
                        continue

                    exc = entry.get("exception", {})
                    error_type = exc.get("type", entry.get("level", "UNKNOWN"))
                    errors[error_type] = errors.get(error_type, 0) + 1
        except OSError:
            pass

        return errors

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read the most recent trade log entries."""
        filepath = self.log_dir / LogFile.TRADES.value
        if not filepath.exists():
            return []

        trades: List[Dict[str, Any]] = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        data = entry.get("data", {})
                        if data.get("event") in ("trade_opened", "trade_closed"):
                            trades.append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

        return trades[-limit:]

    def errors_per_minute(self, window_minutes: int = 5) -> float:
        """Calculate the error rate per minute over a window."""
        count = self.count_events(
            LogFile.ERRORS.value,
            since_seconds=window_minutes * 60,
        )
        return count / max(window_minutes, 1)
