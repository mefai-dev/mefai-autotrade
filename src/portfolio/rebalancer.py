"""
Mefai Signal Engine - Portfolio Rebalancer

Manages the rebalancing lifecycle: detects when portfolio weights have drifted
from targets, calculates optimal trade list to restore targets, and executes
rebalances with awareness of transaction costs and tax implications.

Supports periodic (daily/weekly/monthly), threshold-based, tax-aware,
cost-aware, and gradual rebalancing modes.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("mefai.autotrade.portfolio.rebalancer")


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class RebalanceFrequency(str, Enum):
    """Periodic rebalance intervals."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @property
    def seconds(self) -> int:
        return {
            "hourly": 3600,
            "daily": 86400,
            "weekly": 604800,
            "biweekly": 1209600,
            "monthly": 2592000,
            "quarterly": 7776000,
        }[self.value]


class RebalanceMode(str, Enum):
    """How to execute the rebalance."""
    IMMEDIATE = "immediate"     # all at once
    GRADUAL = "gradual"         # spread over multiple intervals
    TAX_AWARE = "tax_aware"     # minimize realized gains
    COST_AWARE = "cost_aware"   # skip if cost > benefit


@dataclass
class RebalanceTrade:
    """A single trade required to rebalance."""
    strategy_name: str
    direction: str              # "increase" or "decrease"
    current_weight: float
    target_weight: float
    delta_weight: float         # target - current
    dollar_amount: float        # absolute dollar amount to trade
    estimated_cost: float = 0.0 # transaction cost estimate
    tax_impact: float = 0.0     # estimated tax on realized gain
    priority: int = 0           # lower = higher priority

    @property
    def net_benefit(self) -> float:
        """Net benefit of this trade after costs."""
        return abs(self.dollar_amount) - self.estimated_cost - self.tax_impact


@dataclass
class RebalanceReport:
    """Complete report for a rebalancing event."""
    timestamp: float = field(default_factory=time.time)
    mode: str = ""
    trigger_reason: str = ""
    total_capital: float = 0.0
    trades: List[RebalanceTrade] = field(default_factory=list)
    total_turnover: float = 0.0         # sum of absolute dollar trades
    total_cost: float = 0.0
    total_tax: float = 0.0
    weights_before: Dict[str, float] = field(default_factory=dict)
    weights_after: Dict[str, float] = field(default_factory=dict)
    max_drift_before: float = 0.0
    max_drift_after: float = 0.0
    skipped_trades: List[str] = field(default_factory=list)
    executed: bool = False

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def turnover_pct(self) -> float:
        if self.total_capital > 0:
            return (self.total_turnover / self.total_capital) * 100.0
        return 0.0

    def summary(self) -> str:
        lines = [
            f"Rebalance Report - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"  Mode: {self.mode}",
            f"  Trigger: {self.trigger_reason}",
            f"  Capital: ${self.total_capital:,.2f}",
            f"  Trades: {self.trade_count}",
            f"  Turnover: ${self.total_turnover:,.2f} ({self.turnover_pct:.1f}%)",
            f"  Costs: ${self.total_cost:,.2f}",
            f"  Max drift: {self.max_drift_before:.2f}% -> {self.max_drift_after:.2f}%",
        ]
        if self.skipped_trades:
            lines.append(f"  Skipped: {', '.join(self.skipped_trades)}")
        for t in self.trades:
            lines.append(
                f"    {t.strategy_name}: {t.current_weight:.1%} -> {t.target_weight:.1%} "
                f"({t.direction} ${abs(t.dollar_amount):,.2f})"
            )
        return "\n".join(lines)


@dataclass
class CostModel:
    """Transaction cost model."""
    maker_fee_bps: float = 2.0      # 0.02%
    taker_fee_bps: float = 5.0      # 0.05%
    slippage_bps: float = 3.0       # 0.03% estimated slippage
    min_trade_size: float = 10.0    # skip trades smaller than this

    def estimate_cost(self, dollar_amount: float, is_maker: bool = False) -> float:
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        total_bps = fee_bps + self.slippage_bps
        return abs(dollar_amount) * total_bps / 10000.0


@dataclass
class TaxModel:
    """Simplified tax model for rebalancing decisions."""
    short_term_rate: float = 0.37   # short-term capital gains rate
    long_term_rate: float = 0.20    # long-term capital gains rate
    holding_period_days: int = 365  # threshold for long-term

    def estimate_tax(
        self,
        realized_gain: float,
        holding_days: int,
    ) -> float:
        if realized_gain <= 0:
            return 0.0  # no tax on losses
        rate = self.long_term_rate if holding_days >= self.holding_period_days else self.short_term_rate
        return realized_gain * rate


# ---------------------------------------------------------------------------
# Portfolio Rebalancer
# ---------------------------------------------------------------------------

class PortfolioRebalancer:
    """
    Manages portfolio rebalancing with multiple strategies for when and how
    to rebalance. Supports periodic, drift-based, cost-aware, tax-aware,
    and gradual rebalancing.
    """

    def __init__(
        self,
        frequency: RebalanceFrequency = RebalanceFrequency.DAILY,
        mode: RebalanceMode = RebalanceMode.IMMEDIATE,
        drift_threshold_pct: float = 5.0,
        cost_model: Optional[CostModel] = None,
        tax_model: Optional[TaxModel] = None,
        gradual_steps: int = 3,
        gradual_interval_seconds: int = 3600,
        max_turnover_pct: float = 50.0,
        on_rebalance: Optional[Callable[[RebalanceReport], None]] = None,
    ):
        self.frequency = frequency
        self.mode = mode
        self.drift_threshold_pct = drift_threshold_pct
        self.cost_model = cost_model or CostModel()
        self.tax_model = tax_model or TaxModel()
        self.gradual_steps = gradual_steps
        self.gradual_interval_seconds = gradual_interval_seconds
        self.max_turnover_pct = max_turnover_pct
        self._on_rebalance = on_rebalance

        # State
        self._last_rebalance_time: float = 0.0
        self._reports: List[RebalanceReport] = []
        self._gradual_queue: List[RebalanceTrade] = []
        self._gradual_step: int = 0

        # Holdings tracking for tax calculations
        self._holdings: Dict[str, Dict[str, Any]] = {}
        # strategy_name -> {"entry_time": float, "cost_basis": float, "quantity": float}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Determine if rebalancing is needed."""
        now = time.time()
        elapsed = now - self._last_rebalance_time

        # Check periodic trigger
        periodic_due = elapsed >= self.frequency.seconds

        # Check drift trigger
        max_drift = self._max_drift(current_weights, target_weights)
        drift_due = max_drift >= self.drift_threshold_pct

        # Gradual mode: check if pending steps remain
        if self._gradual_queue and self._gradual_step < self.gradual_steps:
            step_elapsed = now - self._last_rebalance_time
            if step_elapsed >= self.gradual_interval_seconds:
                return True, f"gradual_step_{self._gradual_step + 1}_of_{self.gradual_steps}"

        if periodic_due and drift_due:
            return True, f"periodic_and_drift_{max_drift:.1f}pct"
        if periodic_due:
            return True, "periodic"
        if drift_due:
            return True, f"drift_{max_drift:.1f}pct"

        return False, f"no_trigger (drift={max_drift:.1f}%, elapsed={elapsed:.0f}s)"

    def calculate_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        total_capital: float,
        strategy_holdings: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> RebalanceReport:
        """
        Calculate the trades needed to rebalance from current to target weights.
        Does not execute trades - returns a report with the trade list.
        """
        report = RebalanceReport(
            mode=self.mode.value,
            total_capital=total_capital,
            weights_before=dict(current_weights),
            max_drift_before=self._max_drift(current_weights, target_weights),
        )

        if strategy_holdings:
            self._holdings = strategy_holdings

        # Calculate raw trades
        all_strategies = set(current_weights.keys()) | set(target_weights.keys())
        trades: List[RebalanceTrade] = []

        for name in sorted(all_strategies):
            current = current_weights.get(name, 0.0)
            target = target_weights.get(name, 0.0)
            delta = target - current

            if abs(delta) < 0.001:  # less than 0.1% - skip
                continue

            dollar_amount = delta * total_capital
            direction = "increase" if delta > 0 else "decrease"

            trade = RebalanceTrade(
                strategy_name=name,
                direction=direction,
                current_weight=current,
                target_weight=target,
                delta_weight=delta,
                dollar_amount=dollar_amount,
                estimated_cost=self.cost_model.estimate_cost(dollar_amount),
            )

            # Tax impact for decreases (selling)
            if direction == "decrease" and name in self._holdings:
                holding = self._holdings[name]
                entry_time = holding.get("entry_time", 0.0)
                cost_basis = holding.get("cost_basis", 0.0)
                holding_days = int((time.time() - entry_time) / 86400) if entry_time > 0 else 0
                current_value = current * total_capital
                realized_gain = (current_value - cost_basis) * (abs(delta) / current) if current > 0 else 0
                trade.tax_impact = self.tax_model.estimate_tax(realized_gain, holding_days)

            trades.append(trade)

        # Apply mode-specific filtering
        if self.mode == RebalanceMode.COST_AWARE:
            trades, skipped = self._filter_cost_aware(trades)
            report.skipped_trades = skipped
        elif self.mode == RebalanceMode.TAX_AWARE:
            trades, skipped = self._filter_tax_aware(trades)
            report.skipped_trades = skipped
        elif self.mode == RebalanceMode.GRADUAL:
            trades = self._prepare_gradual(trades)

        # Apply max turnover constraint
        trades = self._apply_turnover_limit(trades, total_capital)

        # Sort by priority (largest absolute delta first)
        trades.sort(key=lambda t: abs(t.delta_weight), reverse=True)

        report.trades = trades
        report.total_turnover = sum(abs(t.dollar_amount) for t in trades)
        report.total_cost = sum(t.estimated_cost for t in trades)
        report.total_tax = sum(t.tax_impact for t in trades)

        # Calculate expected weights after rebalance
        weights_after = dict(current_weights)
        for t in trades:
            weights_after[t.strategy_name] = weights_after.get(t.strategy_name, 0.0) + t.delta_weight
        report.weights_after = weights_after
        report.max_drift_after = self._max_drift(weights_after, target_weights)

        return report

    def execute_rebalance(
        self,
        report: RebalanceReport,
        trade_executor: Optional[Callable[[RebalanceTrade], bool]] = None,
    ) -> RebalanceReport:
        """
        Execute the trades in a rebalance report.
        trade_executor: callback that executes a single trade, returns True on success.
        """
        if not report.trades:
            logger.info("No trades to execute for rebalance")
            report.executed = True
            return report

        executed_trades = []
        for trade in report.trades:
            if abs(trade.dollar_amount) < self.cost_model.min_trade_size:
                report.skipped_trades.append(f"{trade.strategy_name} (below_min_size)")
                continue

            if trade_executor:
                success = trade_executor(trade)
                if success:
                    executed_trades.append(trade)
                    logger.info(
                        "Rebalance trade executed: %s %s $%.2f",
                        trade.direction, trade.strategy_name, abs(trade.dollar_amount),
                    )
                else:
                    logger.warning(
                        "Rebalance trade failed: %s %s",
                        trade.direction, trade.strategy_name,
                    )
            else:
                # Dry run - just log
                executed_trades.append(trade)
                logger.info(
                    "Rebalance trade (dry run): %s %s $%.2f",
                    trade.direction, trade.strategy_name, abs(trade.dollar_amount),
                )

        report.trades = executed_trades
        report.executed = True
        self._last_rebalance_time = time.time()
        self._reports.append(report)

        if self._on_rebalance:
            self._on_rebalance(report)

        logger.info(
            "Rebalance complete: %d trades, turnover=$%.2f (%.1f%%), cost=$%.2f",
            len(executed_trades), report.total_turnover,
            report.turnover_pct, report.total_cost,
        )
        return report

    def get_reports(self, n: int = 10) -> List[RebalanceReport]:
        """Return the last n rebalance reports."""
        return self._reports[-n:]

    def get_status(self) -> Dict[str, Any]:
        """Current rebalancer status."""
        return {
            "frequency": self.frequency.value,
            "mode": self.mode.value,
            "drift_threshold_pct": self.drift_threshold_pct,
            "last_rebalance_time": self._last_rebalance_time,
            "total_rebalances": len(self._reports),
            "gradual_step": self._gradual_step if self._gradual_queue else None,
            "pending_gradual_trades": len(self._gradual_queue),
            "max_turnover_pct": self.max_turnover_pct,
        }

    # ------------------------------------------------------------------
    # Mode-specific filtering
    # ------------------------------------------------------------------

    def _filter_cost_aware(
        self,
        trades: List[RebalanceTrade],
    ) -> Tuple[List[RebalanceTrade], List[str]]:
        """
        Skip trades where transaction cost exceeds the expected benefit
        of being closer to the target weight. Benefit is estimated as the
        expected return improvement from being at the correct allocation.
        """
        filtered = []
        skipped = []

        for trade in trades:
            if trade.net_benefit > 0:
                filtered.append(trade)
            else:
                skipped.append(f"{trade.strategy_name} (cost>${trade.estimated_cost:.2f}>benefit)")
                logger.debug(
                    "Skipping rebalance of %s: cost=$%.2f exceeds benefit",
                    trade.strategy_name, trade.estimated_cost,
                )

        return filtered, skipped

    def _filter_tax_aware(
        self,
        trades: List[RebalanceTrade],
    ) -> Tuple[List[RebalanceTrade], List[str]]:
        """
        Tax-aware filtering: avoid selling positions with large short-term gains.
        Prefer selling positions at a loss (tax-loss harvesting) or those
        with long-term gains.
        """
        filtered = []
        skipped = []

        # Separate buys from sells
        buys = [t for t in trades if t.direction == "increase"]
        sells = [t for t in trades if t.direction == "decrease"]

        # Always include buys
        filtered.extend(buys)

        # For sells: skip if tax impact is too high relative to drift
        for trade in sells:
            if trade.tax_impact <= 0:
                # No tax (loss or no gain) - always include
                filtered.append(trade)
            else:
                # Tax cost threshold: skip if tax > 2% of the trade
                tax_ratio = trade.tax_impact / abs(trade.dollar_amount) if trade.dollar_amount != 0 else 0
                if tax_ratio < 0.02:
                    filtered.append(trade)
                else:
                    # Check if drift is severe enough to justify the tax
                    if abs(trade.delta_weight) > 0.10:  # >10% drift overrides tax
                        filtered.append(trade)
                        logger.info(
                            "Tax-aware: accepting high tax on %s due to large drift (%.1f%%)",
                            trade.strategy_name, abs(trade.delta_weight) * 100,
                        )
                    else:
                        skipped.append(f"{trade.strategy_name} (tax=${trade.tax_impact:.2f})")
                        logger.debug(
                            "Tax-aware: skipping %s sale, tax=$%.2f on drift=%.1f%%",
                            trade.strategy_name, trade.tax_impact,
                            abs(trade.delta_weight) * 100,
                        )

        return filtered, skipped

    def _prepare_gradual(
        self,
        trades: List[RebalanceTrade],
    ) -> List[RebalanceTrade]:
        """
        Split trades into gradual steps. Each step moves a fraction of the
        total delta toward the target.
        """
        remaining_steps = max(1, self.gradual_steps - self._gradual_step)
        fraction = 1.0 / remaining_steps

        partial_trades = []
        for trade in trades:
            partial = RebalanceTrade(
                strategy_name=trade.strategy_name,
                direction=trade.direction,
                current_weight=trade.current_weight,
                target_weight=trade.target_weight,
                delta_weight=trade.delta_weight * fraction,
                dollar_amount=trade.dollar_amount * fraction,
                estimated_cost=self.cost_model.estimate_cost(trade.dollar_amount * fraction),
                tax_impact=trade.tax_impact * fraction,
                priority=trade.priority,
            )
            partial_trades.append(partial)

        # Store remaining for next steps
        self._gradual_queue = trades
        self._gradual_step += 1

        if self._gradual_step >= self.gradual_steps:
            self._gradual_queue = []
            self._gradual_step = 0

        return partial_trades

    def _apply_turnover_limit(
        self,
        trades: List[RebalanceTrade],
        total_capital: float,
    ) -> List[RebalanceTrade]:
        """Cap total turnover to max_turnover_pct of total capital."""
        if total_capital <= 0:
            return trades

        max_turnover = total_capital * (self.max_turnover_pct / 100.0)
        cumulative = 0.0
        limited = []

        # Prioritize by absolute delta (largest drift first)
        sorted_trades = sorted(trades, key=lambda t: abs(t.delta_weight), reverse=True)

        for trade in sorted_trades:
            trade_amount = abs(trade.dollar_amount)
            if cumulative + trade_amount <= max_turnover:
                limited.append(trade)
                cumulative += trade_amount
            else:
                # Partial execution to stay within limit
                remaining = max_turnover - cumulative
                if remaining > self.cost_model.min_trade_size:
                    fraction = remaining / trade_amount
                    partial = RebalanceTrade(
                        strategy_name=trade.strategy_name,
                        direction=trade.direction,
                        current_weight=trade.current_weight,
                        target_weight=trade.current_weight + trade.delta_weight * fraction,
                        delta_weight=trade.delta_weight * fraction,
                        dollar_amount=trade.dollar_amount * fraction,
                        estimated_cost=self.cost_model.estimate_cost(remaining),
                        tax_impact=trade.tax_impact * fraction,
                        priority=trade.priority,
                    )
                    limited.append(partial)
                break

        return limited

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _max_drift(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
    ) -> float:
        """Maximum absolute drift in percentage points."""
        max_d = 0.0
        all_keys = set(current.keys()) | set(target.keys())
        for key in all_keys:
            c = current.get(key, 0.0)
            t = target.get(key, 0.0)
            drift = abs(c - t) * 100.0
            if drift > max_d:
                max_d = drift
        return max_d

    def reset(self) -> None:
        """Reset rebalancer state."""
        self._last_rebalance_time = 0.0
        self._gradual_queue = []
        self._gradual_step = 0
        self._reports = []
        logger.info("Rebalancer state reset")
