"""
Metrics Collection for Mefai Autotrade Observability.
Prometheus-compatible metrics with counters, gauges, histograms, and
custom metric registration. Supports export in Prometheus text format,
JSON, and CSV for integration with monitoring infrastructure.
"""

import csv
import io
import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mefai.autotrade")


# ---------------------------------------------------------------------------
# Metric Types
# ---------------------------------------------------------------------------

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabel:
    """A set of key-value labels for a metric."""
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        if not self.labels:
            return ""
        sorted_items = sorted(self.labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_items)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if isinstance(other, MetricLabel):
            return self.key == other.key
        return False


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class Counter:
    """Monotonically increasing counter metric.

    Only supports increment (inc) and reset operations.
    Counters are used for things like total_trades, total_orders, total_errors.
    """

    def __init__(self, name: str, description: str = "", label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._lock = threading.Lock()
        self._values: Dict[str, float] = defaultdict(float)
        self._created_at = time.time()

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented with non-negative values")
        key = MetricLabel(labels or {}).key
        with self._lock:
            self._values[key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current counter value."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        """Reset all counter values to zero."""
        with self._lock:
            self._values.clear()

    def get_all(self) -> Dict[str, float]:
        """Get all labeled values."""
        with self._lock:
            return dict(self._values)

    def to_prometheus(self) -> str:
        """Export in Prometheus text format."""
        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} counter")
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for label_key, value in sorted(self._values.items()):
                    if label_key:
                        lines.append(f"{self.name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{self.name} {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------

class Gauge:
    """Gauge metric that can increase or decrease.

    Used for things like open_positions, unrealized_pnl, portfolio_value.
    """

    def __init__(self, name: str, description: str = "", label_names: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self._lock = threading.Lock()
        self._values: Dict[str, float] = defaultdict(float)

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge to a specific value."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            self._values[key] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current gauge value."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            return self._values.get(key, 0.0)

    def get_all(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._values)

    def to_prometheus(self) -> str:
        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} gauge")
        with self._lock:
            if not self._values:
                lines.append(f"{self.name} 0")
            else:
                for label_key, value in sorted(self._values.items()):
                    if label_key:
                        lines.append(f"{self.name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{self.name} {value}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

class Histogram:
    """Histogram metric for distributions (duration, latency, slippage).

    Tracks observations in configurable buckets plus sum and count.
    """

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = label_names or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._lock = threading.Lock()

        # Per-label data: key -> {bucket -> count, sum, count}
        self._data: Dict[str, Dict[str, Any]] = {}

    def _get_or_create(self, key: str) -> Dict[str, Any]:
        if key not in self._data:
            self._data[key] = {
                "buckets": {b: 0 for b in self.buckets},
                "sum": 0.0,
                "count": 0,
                "min": float("inf"),
                "max": float("-inf"),
            }
        return self._data[key]

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            data = self._get_or_create(key)
            data["sum"] += value
            data["count"] += 1
            data["min"] = min(data["min"], value)
            data["max"] = max(data["max"], value)
            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for the histogram."""
        key = MetricLabel(labels or {}).key
        with self._lock:
            data = self._data.get(key)
            if not data or data["count"] == 0:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p90": 0.0,
                    "p99": 0.0,
                }

            count = data["count"]
            total = data["sum"]
            mean = total / count

            # Estimate percentiles from buckets
            p50 = self._estimate_percentile(data, 0.50)
            p90 = self._estimate_percentile(data, 0.90)
            p99 = self._estimate_percentile(data, 0.99)

            return {
                "count": count,
                "sum": round(total, 6),
                "mean": round(mean, 6),
                "min": round(data["min"], 6) if data["min"] != float("inf") else 0.0,
                "max": round(data["max"], 6) if data["max"] != float("-inf") else 0.0,
                "p50": round(p50, 6),
                "p90": round(p90, 6),
                "p99": round(p99, 6),
            }

    def _estimate_percentile(self, data: Dict[str, Any], percentile: float) -> float:
        """Estimate a percentile from histogram buckets."""
        count = data["count"]
        target = count * percentile
        cumulative = 0
        prev_bucket = 0.0

        sorted_buckets = sorted(
            [(b, c) for b, c in data["buckets"].items() if b != float("inf")],
            key=lambda x: x[0],
        )

        for bucket, bucket_count in sorted_buckets:
            cumulative += bucket_count
            if cumulative >= target:
                # Linear interpolation within bucket
                prev_cumulative = cumulative - bucket_count
                if bucket_count > 0:
                    fraction = (target - prev_cumulative) / bucket_count
                else:
                    fraction = 0
                return prev_bucket + fraction * (bucket - prev_bucket)
            prev_bucket = bucket

        return data["max"] if data["max"] != float("-inf") else 0.0

    def reset(self) -> None:
        with self._lock:
            self._data.clear()

    def to_prometheus(self) -> str:
        lines = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} histogram")

        with self._lock:
            for label_key, data in sorted(self._data.items()):
                label_str = f"{{{label_key}," if label_key else "{"
                cumulative = 0
                sorted_buckets = sorted(
                    [(b, c) for b, c in data["buckets"].items()],
                    key=lambda x: (x[0] == float("inf"), x[0]),
                )
                for bucket, count in sorted_buckets:
                    cumulative += count
                    le = "+Inf" if bucket == float("inf") else str(bucket)
                    lines.append(
                        f'{self.name}_bucket{{{label_key + "," if label_key else ""}le="{le}"}} {cumulative}'
                    )
                lines.append(
                    f"{self.name}_sum{{{label_key}}}" if label_key
                    else f"{self.name}_sum"
                    + f" {data['sum']}"
                )
                lines.append(
                    f"{self.name}_count{{{label_key}}}" if label_key
                    else f"{self.name}_count"
                    + f" {data['count']}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric Exporter
# ---------------------------------------------------------------------------

class MetricExporter:
    """Export metrics in various formats."""

    @staticmethod
    def to_prometheus_text(collector: "MetricsCollector") -> str:
        """Export all metrics in Prometheus text exposition format."""
        lines = [
            "# Mefai Autotrade Metrics",
            f"# Generated at {datetime.now(timezone.utc).isoformat()}",
            "",
        ]

        for metric in collector.get_all_metrics():
            lines.append(metric.to_prometheus())
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_json(collector: "MetricsCollector") -> Dict[str, Any]:
        """Export all metrics as a JSON-serializable dictionary."""
        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        for name, metric in collector._counters.items():
            result["counters"][name] = {
                "description": metric.description,
                "values": metric.get_all(),
            }

        for name, metric in collector._gauges.items():
            result["gauges"][name] = {
                "description": metric.description,
                "values": metric.get_all(),
            }

        for name, metric in collector._histograms.items():
            all_stats = {}
            with metric._lock:
                for label_key in metric._data:
                    labels = {label_key: ""} if label_key else None
                    all_stats[label_key or "default"] = metric.get_stats(
                        None if not label_key else None
                    )
            result["histograms"][name] = {
                "description": metric.description,
                "stats": all_stats,
            }

        return result

    @staticmethod
    def to_csv(collector: "MetricsCollector") -> str:
        """Export all metrics as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "type", "name", "labels", "value"])

        ts = datetime.now(timezone.utc).isoformat()

        for name, metric in collector._counters.items():
            for labels, value in metric.get_all().items():
                writer.writerow([ts, "counter", name, labels or "", value])

        for name, metric in collector._gauges.items():
            for labels, value in metric.get_all().items():
                writer.writerow([ts, "gauge", name, labels or "", value])

        for name, metric in collector._histograms.items():
            stats = metric.get_stats()
            for stat_name, value in stats.items():
                writer.writerow([ts, "histogram", f"{name}_{stat_name}", "", value])

        return output.getvalue()


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Central metrics collection registry.

    Manages all metric instances (counters, gauges, histograms) and provides
    a unified interface for metric operations. Ships with pre-registered
    trading-specific metrics.

    Usage:
        collector = MetricsCollector()
        collector.total_trades.inc(labels={"strategy": "trend"})
        collector.open_positions.set(5)
        collector.order_latency.observe(0.123)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

        # Pre-register standard trading metrics
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register the standard set of trading metrics."""

        # ---- Counters ----
        self.total_trades = self.register_counter(
            "mefai_total_trades",
            "Total number of completed trades",
            label_names=["strategy", "symbol", "side"],
        )
        self.total_orders = self.register_counter(
            "mefai_total_orders",
            "Total number of orders submitted",
            label_names=["type", "side"],
        )
        self.total_errors = self.register_counter(
            "mefai_total_errors",
            "Total number of errors",
            label_names=["component", "error_type"],
        )
        self.total_signals = self.register_counter(
            "mefai_total_signals",
            "Total number of signals received",
            label_names=["strategy", "direction"],
        )
        self.total_rejected_orders = self.register_counter(
            "mefai_rejected_orders",
            "Total number of rejected orders",
            label_names=["reason"],
        )
        self.total_risk_blocks = self.register_counter(
            "mefai_risk_blocks",
            "Total number of trades blocked by risk manager",
            label_names=["rule"],
        )
        self.total_notifications = self.register_counter(
            "mefai_notifications_sent",
            "Total notifications sent",
            label_names=["channel", "category"],
        )

        # ---- Gauges ----
        self.open_positions = self.register_gauge(
            "mefai_open_positions",
            "Number of currently open positions",
            label_names=["symbol"],
        )
        self.unrealized_pnl = self.register_gauge(
            "mefai_unrealized_pnl",
            "Current unrealized PnL in USDT",
        )
        self.realized_pnl_daily = self.register_gauge(
            "mefai_realized_pnl_daily",
            "Today's realized PnL in USDT",
        )
        self.portfolio_value = self.register_gauge(
            "mefai_portfolio_value",
            "Total portfolio value in USDT",
        )
        self.margin_used = self.register_gauge(
            "mefai_margin_used",
            "Current margin usage percentage",
        )
        self.drawdown_current = self.register_gauge(
            "mefai_drawdown_current",
            "Current drawdown percentage",
        )
        self.win_rate_rolling = self.register_gauge(
            "mefai_win_rate_rolling",
            "Rolling win rate (24h)",
        )
        self.active_strategies = self.register_gauge(
            "mefai_active_strategies",
            "Number of active strategies",
        )
        self.websocket_connections = self.register_gauge(
            "mefai_websocket_connections",
            "Number of active WebSocket connections",
        )
        self.queue_depth = self.register_gauge(
            "mefai_queue_depth",
            "Current order/notification queue depth",
            label_names=["queue_name"],
        )

        # ---- Histograms ----
        self.trade_duration = self.register_histogram(
            "mefai_trade_duration_seconds",
            "Trade duration in seconds",
            buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400, float("inf")),
        )
        self.order_latency = self.register_histogram(
            "mefai_order_latency_seconds",
            "Order submission to fill latency in seconds",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
        )
        self.fill_slippage = self.register_histogram(
            "mefai_fill_slippage_bps",
            "Fill slippage in basis points",
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, float("inf")),
        )
        self.trade_pnl = self.register_histogram(
            "mefai_trade_pnl_usdt",
            "Trade PnL distribution in USDT",
            buckets=(-500, -200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200, 500, float("inf")),
        )
        self.api_response_time = self.register_histogram(
            "mefai_api_response_seconds",
            "Exchange API response time in seconds",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
            label_names=["endpoint"],
        )
        self.signal_processing_time = self.register_histogram(
            "mefai_signal_processing_seconds",
            "Time from signal reception to order submission",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf")),
            label_names=["strategy"],
        )

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register_counter(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ) -> Counter:
        """Register a new counter metric."""
        with self._lock:
            if name in self._counters:
                return self._counters[name]
            counter = Counter(name, description, label_names)
            self._counters[name] = counter
            return counter

    def register_gauge(
        self,
        name: str,
        description: str = "",
        label_names: Optional[List[str]] = None,
    ) -> Gauge:
        """Register a new gauge metric."""
        with self._lock:
            if name in self._gauges:
                return self._gauges[name]
            gauge = Gauge(name, description, label_names)
            self._gauges[name] = gauge
            return gauge

    def register_histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
        label_names: Optional[List[str]] = None,
    ) -> Histogram:
        """Register a new histogram metric."""
        with self._lock:
            if name in self._histograms:
                return self._histograms[name]
            histogram = Histogram(name, description, buckets, label_names)
            self._histograms[name] = histogram
            return histogram

    # -------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------

    def get_counter(self, name: str) -> Optional[Counter]:
        return self._counters.get(name)

    def get_gauge(self, name: str) -> Optional[Gauge]:
        return self._gauges.get(name)

    def get_histogram(self, name: str) -> Optional[Histogram]:
        return self._histograms.get(name)

    def get_all_metrics(self) -> List[Any]:
        """Get all registered metrics."""
        with self._lock:
            metrics: List[Any] = []
            metrics.extend(self._counters.values())
            metrics.extend(self._gauges.values())
            metrics.extend(self._histograms.values())
            return metrics

    def get_metric_names(self) -> Dict[str, List[str]]:
        """Get all registered metric names by type."""
        with self._lock:
            return {
                "counters": list(self._counters.keys()),
                "gauges": list(self._gauges.keys()),
                "histograms": list(self._histograms.keys()),
            }

    # -------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        side: str,
        pnl: float,
        duration_seconds: float,
        slippage_bps: float = 0.0,
    ) -> None:
        """Record metrics for a completed trade."""
        self.total_trades.inc(labels={
            "strategy": strategy,
            "symbol": symbol,
            "side": side,
        })
        self.trade_pnl.observe(pnl)
        self.trade_duration.observe(duration_seconds)
        if slippage_bps > 0:
            self.fill_slippage.observe(slippage_bps)

    def record_order(
        self,
        order_type: str,
        side: str,
        latency_seconds: float,
    ) -> None:
        """Record metrics for an order."""
        self.total_orders.inc(labels={"type": order_type, "side": side})
        self.order_latency.observe(latency_seconds)

    def record_error(
        self,
        component: str,
        error_type: str,
    ) -> None:
        """Record an error metric."""
        self.total_errors.inc(labels={
            "component": component,
            "error_type": error_type,
        })

    def update_portfolio_state(
        self,
        portfolio_value: float,
        unrealized_pnl: float,
        realized_pnl_daily: float,
        margin_used_pct: float,
        open_position_count: int,
        drawdown_pct: float,
    ) -> None:
        """Update all portfolio-level gauges at once."""
        self.portfolio_value.set(portfolio_value)
        self.unrealized_pnl.set(unrealized_pnl)
        self.realized_pnl_daily.set(realized_pnl_daily)
        self.margin_used.set(margin_used_pct)
        self.open_positions.set(open_position_count)
        self.drawdown_current.set(drawdown_pct)

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        return MetricExporter.to_prometheus_text(self)

    def export_json(self) -> Dict[str, Any]:
        """Export all metrics as JSON."""
        return MetricExporter.to_json(self)

    def export_csv(self) -> str:
        """Export all metrics as CSV."""
        return MetricExporter.to_csv(self)

    def reset_all(self) -> None:
        """Reset all metrics to zero. Use with caution."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for gauge in self._gauges.values():
                gauge.set(0.0)
            for histogram in self._histograms.values():
                histogram.reset()
        logger.warning("All metrics reset to zero")
