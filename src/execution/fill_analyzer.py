"""
Mefai Autotrade - Fill Analyzer
Execution quality analysis: slippage tracking, fill rates,
implementation shortfall, maker/taker ratios, and best execution
report generation.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """Record of a single order fill."""
    fill_id: str
    order_id: str
    symbol: str
    side: str                    # BUY or SELL
    order_type: str              # LIMIT, MARKET, etc.
    requested_quantity: float
    filled_quantity: float
    requested_price: float       # Expected/limit price
    fill_price: float            # Actual fill price
    timestamp: float
    venue_id: str = ""
    is_maker: bool = False       # True if filled as maker
    commission: float = 0.0      # Commission paid
    commission_asset: str = ""
    market_price_at_order: float = 0.0  # Market mid price when order was placed
    fill_time_ms: float = 0.0    # Time from order to fill in ms
    execution_algo: str = ""     # TWAP, VWAP, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def slippage_bps(self) -> float:
        """Slippage in basis points. Positive = unfavorable."""
        if self.requested_price <= 0:
            if self.market_price_at_order <= 0:
                return 0.0
            ref_price = self.market_price_at_order
        else:
            ref_price = self.requested_price

        if ref_price <= 0:
            return 0.0

        if self.side == "BUY":
            return (self.fill_price - ref_price) / ref_price * 10000.0
        else:
            return (ref_price - self.fill_price) / ref_price * 10000.0

    @property
    def fill_rate(self) -> float:
        """Fill rate 0.0 - 1.0."""
        if self.requested_quantity <= 0:
            return 0.0
        return min(1.0, self.filled_quantity / self.requested_quantity)

    @property
    def notional_value(self) -> float:
        return self.filled_quantity * self.fill_price

    @property
    def is_full_fill(self) -> bool:
        return self.fill_rate >= 0.99


@dataclass
class SlippageReport:
    """Slippage analysis report."""
    symbol: str
    side: str
    period_start: float
    period_end: float
    num_fills: int = 0
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    min_slippage_bps: float = 0.0
    std_dev_slippage_bps: float = 0.0
    p95_slippage_bps: float = 0.0
    total_slippage_usd: float = 0.0
    by_order_type: Dict[str, float] = field(default_factory=dict)
    by_venue: Dict[str, float] = field(default_factory=dict)
    by_size_bucket: Dict[str, float] = field(default_factory=dict)
    worst_fills: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionQualityReport:
    """Comprehensive execution quality report."""
    period_start: float
    period_end: float
    total_orders: int = 0
    total_fills: int = 0
    total_volume_usd: float = 0.0

    # Fill metrics
    full_fill_rate: float = 0.0
    partial_fill_rate: float = 0.0
    rejection_rate: float = 0.0
    avg_fill_time_ms: float = 0.0
    p95_fill_time_ms: float = 0.0

    # Slippage
    avg_slippage_bps: float = 0.0
    total_slippage_usd: float = 0.0

    # Maker/taker
    maker_ratio: float = 0.0
    taker_ratio: float = 0.0
    maker_volume_usd: float = 0.0
    taker_volume_usd: float = 0.0

    # Commissions
    total_commission_usd: float = 0.0
    avg_commission_bps: float = 0.0
    commission_savings_vs_taker: float = 0.0

    # By category
    by_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_venue: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_algo: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_order_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BestExecutionReport:
    """Best execution compliance report."""
    report_id: str
    generated_at: float
    period_start: float
    period_end: float
    total_orders: int = 0
    orders_meeting_best_execution: int = 0
    best_execution_rate: float = 0.0
    avg_price_improvement_bps: float = 0.0
    total_price_improvement_usd: float = 0.0
    venue_comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    order_type_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    time_of_day_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fill Analyzer
# ---------------------------------------------------------------------------

class FillAnalyzer:
    """Analyzes execution quality across fills.

    Tracks slippage, fill rates, maker/taker ratios, and generates
    comprehensive execution quality reports.
    """

    def __init__(self, max_history: int = 50000):
        self._fills: deque = deque(maxlen=max_history)
        self._fills_by_symbol: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._fills_by_venue: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._fills_by_algo: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._total_orders = 0
        self._total_rejections = 0
        logger.info("FillAnalyzer initialized")

    def record_fill(self, fill: FillRecord) -> None:
        """Record a new fill."""
        self._fills.append(fill)
        self._fills_by_symbol[fill.symbol].append(fill)
        self._fills_by_venue[fill.venue_id].append(fill)
        if fill.execution_algo:
            self._fills_by_algo[fill.execution_algo].append(fill)
        self._total_orders += 1

    def record_rejection(self) -> None:
        """Record an order rejection (no fill)."""
        self._total_orders += 1
        self._total_rejections += 1

    # -----------------------------------------------------------------------
    # Slippage analysis
    # -----------------------------------------------------------------------

    def get_slippage_report(self, symbol: Optional[str] = None,
                            side: Optional[str] = None,
                            period_seconds: float = 86400.0) -> SlippageReport:
        """Generate a slippage analysis report."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, side, cutoff)

        if not fills:
            return SlippageReport(
                symbol=symbol or "ALL",
                side=side or "ALL",
                period_start=cutoff,
                period_end=time.time(),
            )

        slippages = [f.slippage_bps for f in fills]
        slippages_sorted = sorted(slippages)
        n = len(slippages)

        avg_slip = sum(slippages) / n
        median_slip = slippages_sorted[n // 2]
        max_slip = max(slippages)
        min_slip = min(slippages)
        p95_slip = slippages_sorted[int(n * 0.95)] if n > 1 else slippages[0]

        # Standard deviation
        variance = sum((s - avg_slip) ** 2 for s in slippages) / n if n > 1 else 0
        std_dev = math.sqrt(variance)

        # Total slippage in USD
        total_slip_usd = 0.0
        for f in fills:
            if f.fill_price > 0 and f.filled_quantity > 0:
                expected_notional = f.filled_quantity * (f.requested_price if f.requested_price > 0 else f.market_price_at_order)
                actual_notional = f.filled_quantity * f.fill_price
                if f.side == "BUY":
                    total_slip_usd += actual_notional - expected_notional
                else:
                    total_slip_usd += expected_notional - actual_notional

        # By order type
        by_type: Dict[str, List[float]] = defaultdict(list)
        for f in fills:
            by_type[f.order_type].append(f.slippage_bps)
        by_type_avg = {t: sum(s) / len(s) for t, s in by_type.items()}

        # By venue
        by_venue: Dict[str, List[float]] = defaultdict(list)
        for f in fills:
            by_venue[f.venue_id].append(f.slippage_bps)
        by_venue_avg = {v: sum(s) / len(s) for v, s in by_venue.items()}

        # By size bucket
        size_buckets = {"small": [], "medium": [], "large": [], "xlarge": []}
        for f in fills:
            notional = f.notional_value
            if notional < 1000:
                size_buckets["small"].append(f.slippage_bps)
            elif notional < 10000:
                size_buckets["medium"].append(f.slippage_bps)
            elif notional < 100000:
                size_buckets["large"].append(f.slippage_bps)
            else:
                size_buckets["xlarge"].append(f.slippage_bps)
        by_size = {k: sum(v) / len(v) if v else 0.0 for k, v in size_buckets.items()}

        # Worst fills
        worst = sorted(fills, key=lambda f: f.slippage_bps, reverse=True)[:5]
        worst_fills = [
            {
                "fill_id": f.fill_id,
                "symbol": f.symbol,
                "slippage_bps": f.slippage_bps,
                "notional": f.notional_value,
                "order_type": f.order_type,
                "venue": f.venue_id,
            }
            for f in worst
        ]

        return SlippageReport(
            symbol=symbol or "ALL",
            side=side or "ALL",
            period_start=cutoff,
            period_end=time.time(),
            num_fills=n,
            avg_slippage_bps=avg_slip,
            median_slippage_bps=median_slip,
            max_slippage_bps=max_slip,
            min_slippage_bps=min_slip,
            std_dev_slippage_bps=std_dev,
            p95_slippage_bps=p95_slip,
            total_slippage_usd=total_slip_usd,
            by_order_type=by_type_avg,
            by_venue=by_venue_avg,
            by_size_bucket=by_size,
            worst_fills=worst_fills,
        )

    # -----------------------------------------------------------------------
    # Fill rate analysis
    # -----------------------------------------------------------------------

    def get_fill_rate(self, symbol: Optional[str] = None,
                      venue_id: Optional[str] = None,
                      period_seconds: float = 86400.0) -> Dict[str, Any]:
        """Get fill rate statistics."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, None, cutoff, venue_id=venue_id)

        if not fills:
            return {
                "total_orders": 0,
                "full_fills": 0,
                "partial_fills": 0,
                "full_fill_rate": 0.0,
                "partial_fill_rate": 0.0,
                "avg_fill_rate": 0.0,
            }

        full = sum(1 for f in fills if f.is_full_fill)
        partial = sum(1 for f in fills if not f.is_full_fill and f.filled_quantity > 0)
        avg_rate = sum(f.fill_rate for f in fills) / len(fills)

        return {
            "total_orders": len(fills),
            "full_fills": full,
            "partial_fills": partial,
            "full_fill_rate": full / len(fills),
            "partial_fill_rate": partial / len(fills),
            "avg_fill_rate": avg_rate,
        }

    # -----------------------------------------------------------------------
    # Fill time analysis
    # -----------------------------------------------------------------------

    def get_fill_time_stats(self, symbol: Optional[str] = None,
                            period_seconds: float = 86400.0) -> Dict[str, float]:
        """Get fill time statistics."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, None, cutoff)
        times = [f.fill_time_ms for f in fills if f.fill_time_ms > 0]

        if not times:
            return {"avg_ms": 0, "median_ms": 0, "p95_ms": 0, "p99_ms": 0, "max_ms": 0}

        times_sorted = sorted(times)
        n = len(times_sorted)

        return {
            "avg_ms": sum(times) / n,
            "median_ms": times_sorted[n // 2],
            "p95_ms": times_sorted[int(n * 0.95)] if n > 1 else times_sorted[0],
            "p99_ms": times_sorted[int(n * 0.99)] if n > 1 else times_sorted[0],
            "max_ms": times_sorted[-1],
            "min_ms": times_sorted[0],
            "count": n,
        }

    # -----------------------------------------------------------------------
    # Maker/taker analysis
    # -----------------------------------------------------------------------

    def get_maker_taker_ratio(self, symbol: Optional[str] = None,
                               period_seconds: float = 86400.0) -> Dict[str, Any]:
        """Get maker vs taker fill ratio."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, None, cutoff)

        if not fills:
            return {
                "maker_count": 0, "taker_count": 0,
                "maker_ratio": 0, "taker_ratio": 0,
                "maker_volume_usd": 0, "taker_volume_usd": 0,
            }

        maker_count = sum(1 for f in fills if f.is_maker)
        taker_count = len(fills) - maker_count
        maker_volume = sum(f.notional_value for f in fills if f.is_maker)
        taker_volume = sum(f.notional_value for f in fills if not f.is_maker)

        return {
            "maker_count": maker_count,
            "taker_count": taker_count,
            "maker_ratio": maker_count / len(fills) if fills else 0,
            "taker_ratio": taker_count / len(fills) if fills else 0,
            "maker_volume_usd": maker_volume,
            "taker_volume_usd": taker_volume,
            "maker_volume_ratio": maker_volume / (maker_volume + taker_volume)
            if (maker_volume + taker_volume) > 0 else 0,
        }

    # -----------------------------------------------------------------------
    # Commission analysis
    # -----------------------------------------------------------------------

    def get_commission_stats(self, symbol: Optional[str] = None,
                             period_seconds: float = 86400.0) -> Dict[str, Any]:
        """Get commission statistics."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, None, cutoff)

        if not fills:
            return {
                "total_commission": 0, "avg_commission_per_fill": 0,
                "avg_commission_bps": 0, "total_volume_usd": 0,
            }

        total_commission = sum(f.commission for f in fills)
        total_volume = sum(f.notional_value for f in fills)
        avg_per_fill = total_commission / len(fills) if fills else 0
        avg_bps = (total_commission / total_volume * 10000) if total_volume > 0 else 0

        # By venue
        by_venue: Dict[str, Dict[str, float]] = defaultdict(lambda: {"commission": 0, "volume": 0})
        for f in fills:
            by_venue[f.venue_id]["commission"] += f.commission
            by_venue[f.venue_id]["volume"] += f.notional_value

        venue_rates = {}
        for v, data in by_venue.items():
            if data["volume"] > 0:
                venue_rates[v] = data["commission"] / data["volume"] * 10000

        return {
            "total_commission": total_commission,
            "avg_commission_per_fill": avg_per_fill,
            "avg_commission_bps": avg_bps,
            "total_volume_usd": total_volume,
            "by_venue_bps": venue_rates,
        }

    # -----------------------------------------------------------------------
    # Implementation shortfall
    # -----------------------------------------------------------------------

    def compute_implementation_shortfall(self, fills: List[FillRecord],
                                          benchmark_price: float,
                                          side: str) -> Dict[str, Any]:
        """Compute implementation shortfall for a set of fills."""
        if not fills or benchmark_price <= 0:
            return {
                "shortfall_bps": 0, "shortfall_usd": 0,
                "total_filled": 0, "avg_fill": 0,
            }

        total_filled = sum(f.filled_quantity for f in fills)
        total_notional = sum(f.notional_value for f in fills)

        if total_filled <= 0:
            return {
                "shortfall_bps": 0, "shortfall_usd": 0,
                "total_filled": 0, "avg_fill": 0,
            }

        avg_fill = total_notional / total_filled
        benchmark_notional = total_filled * benchmark_price

        if side == "BUY":
            shortfall_usd = total_notional - benchmark_notional
            shortfall_bps = (avg_fill - benchmark_price) / benchmark_price * 10000
        else:
            shortfall_usd = benchmark_notional - total_notional
            shortfall_bps = (benchmark_price - avg_fill) / benchmark_price * 10000

        return {
            "shortfall_bps": shortfall_bps,
            "shortfall_usd": shortfall_usd,
            "total_filled": total_filled,
            "avg_fill": avg_fill,
            "benchmark_price": benchmark_price,
        }

    # -----------------------------------------------------------------------
    # Execution quality report
    # -----------------------------------------------------------------------

    def generate_quality_report(self,
                                 period_seconds: float = 86400.0) -> ExecutionQualityReport:
        """Generate comprehensive execution quality report."""
        cutoff = time.time() - period_seconds
        fills = [f for f in self._fills if f.timestamp > cutoff]

        report = ExecutionQualityReport(
            period_start=cutoff,
            period_end=time.time(),
            total_orders=len(fills),
            total_fills=sum(1 for f in fills if f.filled_quantity > 0),
            total_volume_usd=sum(f.notional_value for f in fills),
        )

        if not fills:
            return report

        # Fill metrics
        full_fills = sum(1 for f in fills if f.is_full_fill)
        partial_fills = sum(
            1 for f in fills if not f.is_full_fill and f.filled_quantity > 0
        )
        report.full_fill_rate = full_fills / len(fills)
        report.partial_fill_rate = partial_fills / len(fills)
        report.rejection_rate = self._total_rejections / self._total_orders if self._total_orders > 0 else 0

        # Fill time
        times = [f.fill_time_ms for f in fills if f.fill_time_ms > 0]
        if times:
            report.avg_fill_time_ms = sum(times) / len(times)
            sorted_times = sorted(times)
            report.p95_fill_time_ms = sorted_times[int(len(sorted_times) * 0.95)]

        # Slippage
        slippages = [f.slippage_bps for f in fills]
        report.avg_slippage_bps = sum(slippages) / len(slippages)
        # Total slippage USD
        for f in fills:
            ref = f.requested_price if f.requested_price > 0 else f.market_price_at_order
            if ref > 0:
                diff = f.fill_price - ref if f.side == "BUY" else ref - f.fill_price
                report.total_slippage_usd += diff * f.filled_quantity

        # Maker/taker
        makers = sum(1 for f in fills if f.is_maker)
        report.maker_ratio = makers / len(fills)
        report.taker_ratio = 1.0 - report.maker_ratio
        report.maker_volume_usd = sum(f.notional_value for f in fills if f.is_maker)
        report.taker_volume_usd = sum(f.notional_value for f in fills if not f.is_maker)

        # Commissions
        report.total_commission_usd = sum(f.commission for f in fills)
        if report.total_volume_usd > 0:
            report.avg_commission_bps = (
                report.total_commission_usd / report.total_volume_usd * 10000
            )
        # Savings from maker fills
        typical_taker_bps = 4.0  # Assume 4bps taker
        typical_maker_bps = 2.0  # Assume 2bps maker
        if report.maker_volume_usd > 0:
            report.commission_savings_vs_taker = (
                report.maker_volume_usd * (typical_taker_bps - typical_maker_bps) / 10000
            )

        # By symbol
        by_symbol: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            by_symbol[f.symbol].append(f)
        for sym, sym_fills in by_symbol.items():
            report.by_symbol[sym] = {
                "count": len(sym_fills),
                "volume_usd": sum(f.notional_value for f in sym_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in sym_fills) / len(sym_fills),
                "fill_rate": sum(f.fill_rate for f in sym_fills) / len(sym_fills),
            }

        # By venue
        by_venue: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            by_venue[f.venue_id].append(f)
        for venue, venue_fills in by_venue.items():
            report.by_venue[venue] = {
                "count": len(venue_fills),
                "volume_usd": sum(f.notional_value for f in venue_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in venue_fills) / len(venue_fills),
                "avg_fill_time_ms": sum(f.fill_time_ms for f in venue_fills if f.fill_time_ms > 0)
                / max(1, sum(1 for f in venue_fills if f.fill_time_ms > 0)),
            }

        # By algo
        by_algo: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            algo = f.execution_algo or "direct"
            by_algo[algo].append(f)
        for algo, algo_fills in by_algo.items():
            report.by_algo[algo] = {
                "count": len(algo_fills),
                "volume_usd": sum(f.notional_value for f in algo_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in algo_fills) / len(algo_fills),
                "fill_rate": sum(f.fill_rate for f in algo_fills) / len(algo_fills),
            }

        # By order type
        by_type: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            by_type[f.order_type].append(f)
        for otype, type_fills in by_type.items():
            report.by_order_type[otype] = {
                "count": len(type_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in type_fills) / len(type_fills),
                "maker_ratio": sum(1 for f in type_fills if f.is_maker) / len(type_fills),
            }

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: ExecutionQualityReport) -> List[str]:
        """Generate actionable recommendations from quality report."""
        recs = []

        if report.avg_slippage_bps > 5.0:
            recs.append(
                f"Average slippage is {report.avg_slippage_bps:.1f}bps - "
                "consider using TWAP/VWAP for large orders"
            )

        if report.maker_ratio < 0.3:
            recs.append(
                f"Maker ratio is only {report.maker_ratio:.0%} - "
                "increase use of limit orders to reduce commission costs"
            )

        if report.full_fill_rate < 0.8:
            recs.append(
                f"Full fill rate is {report.full_fill_rate:.0%} - "
                "review limit price offsets, they may be too aggressive"
            )

        if report.avg_fill_time_ms > 500:
            recs.append(
                f"Average fill time is {report.avg_fill_time_ms:.0f}ms - "
                "check venue connectivity and consider closer venues"
            )

        if report.total_slippage_usd > 100:
            recs.append(
                f"Total slippage cost is ${report.total_slippage_usd:.2f} - "
                "use iceberg orders for large positions"
            )

        # Commission optimization
        if report.commission_savings_vs_taker > 0:
            recs.append(
                f"Maker fills saved ${report.commission_savings_vs_taker:.2f} in commissions - "
                "maintain or increase maker order usage"
            )

        # Venue-specific
        if report.by_venue:
            worst_venue = max(
                report.by_venue.items(),
                key=lambda x: x[1].get("avg_slippage_bps", 0),
            )
            if worst_venue[1].get("avg_slippage_bps", 0) > 10:
                recs.append(
                    f"Venue '{worst_venue[0]}' has high slippage "
                    f"({worst_venue[1]['avg_slippage_bps']:.1f}bps) - "
                    "consider routing to alternative venues"
                )

        return recs

    # -----------------------------------------------------------------------
    # Best execution report
    # -----------------------------------------------------------------------

    def generate_best_execution_report(self,
                                        period_seconds: float = 86400.0) -> BestExecutionReport:
        """Generate best execution compliance report."""
        import uuid
        cutoff = time.time() - period_seconds
        fills = [f for f in self._fills if f.timestamp > cutoff]

        report = BestExecutionReport(
            report_id=f"ber_{uuid.uuid4().hex[:8]}",
            generated_at=time.time(),
            period_start=cutoff,
            period_end=time.time(),
            total_orders=len(fills),
        )

        if not fills:
            return report

        # Best execution = slippage within acceptable range (< 5bps)
        threshold_bps = 5.0
        meeting_best = sum(1 for f in fills if f.slippage_bps <= threshold_bps)
        report.orders_meeting_best_execution = meeting_best
        report.best_execution_rate = meeting_best / len(fills)

        # Price improvement (negative slippage = improvement)
        improvements = [f.slippage_bps for f in fills if f.slippage_bps < 0]
        if improvements:
            report.avg_price_improvement_bps = abs(sum(improvements) / len(improvements))
            for f in fills:
                if f.slippage_bps < 0:
                    ref = f.requested_price if f.requested_price > 0 else f.market_price_at_order
                    if ref > 0:
                        improvement_usd = abs(f.fill_price - ref) * f.filled_quantity
                        report.total_price_improvement_usd += improvement_usd

        # Venue comparison
        by_venue: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            by_venue[f.venue_id].append(f)
        for venue, venue_fills in by_venue.items():
            report.venue_comparison[venue] = {
                "total_orders": len(venue_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in venue_fills) / len(venue_fills),
                "best_execution_rate": sum(
                    1 for f in venue_fills if f.slippage_bps <= threshold_bps
                ) / len(venue_fills),
                "avg_fill_time_ms": sum(f.fill_time_ms for f in venue_fills if f.fill_time_ms > 0)
                / max(1, sum(1 for f in venue_fills if f.fill_time_ms > 0)),
                "volume_usd": sum(f.notional_value for f in venue_fills),
            }

        # Order type analysis
        by_type: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            by_type[f.order_type].append(f)
        for otype, type_fills in by_type.items():
            report.order_type_analysis[otype] = {
                "count": len(type_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in type_fills) / len(type_fills),
                "best_execution_rate": sum(
                    1 for f in type_fills if f.slippage_bps <= threshold_bps
                ) / len(type_fills),
            }

        # Time of day analysis
        by_hour: Dict[str, List[FillRecord]] = defaultdict(list)
        for f in fills:
            hour = int((f.timestamp % 86400) / 3600)
            by_hour[f"{hour:02d}:00"].append(f)
        for hour_str, hour_fills in sorted(by_hour.items()):
            report.time_of_day_analysis[hour_str] = {
                "count": len(hour_fills),
                "avg_slippage_bps": sum(f.slippage_bps for f in hour_fills) / len(hour_fills),
                "volume_usd": sum(f.notional_value for f in hour_fills),
            }

        # Recommendations
        if report.best_execution_rate < 0.9:
            report.recommendations.append(
                f"Best execution rate is {report.best_execution_rate:.0%} - "
                "review routing and order type selection"
            )

        # Find best and worst venues
        if report.venue_comparison:
            best_venue = min(
                report.venue_comparison.items(),
                key=lambda x: x[1].get("avg_slippage_bps", float("inf")),
            )
            worst_venue = max(
                report.venue_comparison.items(),
                key=lambda x: x[1].get("avg_slippage_bps", 0),
            )
            if best_venue[0] != worst_venue[0]:
                report.recommendations.append(
                    f"Route more volume to '{best_venue[0]}' "
                    f"(avg slippage {best_venue[1]['avg_slippage_bps']:.1f}bps) "
                    f"and less to '{worst_venue[0]}' "
                    f"({worst_venue[1]['avg_slippage_bps']:.1f}bps)"
                )

        # Find best time of day
        if report.time_of_day_analysis:
            best_hour = min(
                report.time_of_day_analysis.items(),
                key=lambda x: x[1].get("avg_slippage_bps", float("inf")),
            )
            if best_hour[1]["count"] >= 5:
                report.recommendations.append(
                    f"Best execution quality at {best_hour[0]} UTC "
                    f"({best_hour[1]['avg_slippage_bps']:.1f}bps avg slippage)"
                )

        return report

    # -----------------------------------------------------------------------
    # Historical analysis
    # -----------------------------------------------------------------------

    def get_historical_slippage(self, symbol: Optional[str] = None,
                                 num_buckets: int = 24,
                                 period_seconds: float = 86400.0) -> List[Dict[str, Any]]:
        """Get slippage trend over time buckets."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, None, cutoff)

        if not fills:
            return []

        bucket_size = period_seconds / num_buckets
        buckets = [[] for _ in range(num_buckets)]

        for f in fills:
            bucket_idx = int((f.timestamp - cutoff) / bucket_size)
            bucket_idx = min(bucket_idx, num_buckets - 1)
            buckets[bucket_idx].append(f)

        result = []
        for i, bucket in enumerate(buckets):
            bucket_start = cutoff + i * bucket_size
            if bucket:
                avg_slip = sum(f.slippage_bps for f in bucket) / len(bucket)
                volume = sum(f.notional_value for f in bucket)
            else:
                avg_slip = 0.0
                volume = 0.0
            result.append({
                "timestamp": bucket_start,
                "avg_slippage_bps": avg_slip,
                "count": len(bucket),
                "volume_usd": volume,
            })

        return result

    def get_fill_distribution(self, symbol: Optional[str] = None,
                               period_seconds: float = 86400.0) -> Dict[str, int]:
        """Get distribution of fill rates."""
        cutoff = time.time() - period_seconds
        fills = self._get_filtered_fills(symbol, None, cutoff)

        distribution = {
            "0-10%": 0, "10-25%": 0, "25-50%": 0,
            "50-75%": 0, "75-99%": 0, "100%": 0,
        }

        for f in fills:
            rate = f.fill_rate * 100
            if rate >= 99.9:
                distribution["100%"] += 1
            elif rate >= 75:
                distribution["75-99%"] += 1
            elif rate >= 50:
                distribution["50-75%"] += 1
            elif rate >= 25:
                distribution["25-50%"] += 1
            elif rate >= 10:
                distribution["10-25%"] += 1
            else:
                distribution["0-10%"] += 1

        return distribution

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _get_filtered_fills(self, symbol: Optional[str] = None,
                            side: Optional[str] = None,
                            after_timestamp: float = 0.0,
                            venue_id: Optional[str] = None) -> List[FillRecord]:
        """Get fills matching filters."""
        if symbol:
            fills = list(self._fills_by_symbol.get(symbol, []))
        else:
            fills = list(self._fills)

        if side:
            fills = [f for f in fills if f.side == side]
        if after_timestamp > 0:
            fills = [f for f in fills if f.timestamp > after_timestamp]
        if venue_id:
            fills = [f for f in fills if f.venue_id == venue_id]

        return fills

    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary of fill analyzer state."""
        return {
            "total_fills": len(self._fills),
            "total_orders": self._total_orders,
            "total_rejections": self._total_rejections,
            "symbols_tracked": len(self._fills_by_symbol),
            "venues_tracked": len(self._fills_by_venue),
            "algos_tracked": len(self._fills_by_algo),
        }
