"""
Mefai Autotrade - Correlation Monitor

Cross-position correlation analysis:
- Rolling correlation matrix (Pearson, Spearman)
- Exponentially weighted dynamic correlation
- Correlation regime detection (crash correlation spikes)
- Portfolio diversification score
- Correlation-adjusted VaR
- Threshold alerts and adjustment suggestions
"""

import logging
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class CorrelationMethod(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    EWMA = "ewma"


class CorrelationRegime(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


@dataclass
class CorrelationConfig:
    """Correlation monitoring configuration."""
    # Calculation
    lookback_periods: int = 60  # number of return observations
    ewma_span: int = 30  # exponential weighting span
    min_observations: int = 20  # minimum data points before calculating
    update_interval_seconds: int = 300  # recalculate every 5 minutes

    # Thresholds
    high_correlation_threshold: float = 0.7
    crisis_correlation_threshold: float = 0.85
    low_correlation_threshold: float = 0.3  # good diversification

    # Regime detection
    regime_lookback_periods: int = 20
    regime_normal_avg_corr: float = 0.35
    regime_elevated_avg_corr: float = 0.55
    regime_crisis_avg_corr: float = 0.75

    # Alerts
    alert_on_high_correlation: bool = True
    alert_on_regime_change: bool = True
    max_correlated_pairs: int = 3  # warn if more than 3 highly correlated pairs


@dataclass
class CorrelationPair:
    """Correlation between two symbols."""
    symbol_a: str
    symbol_b: str
    pearson: float = 0.0
    spearman: float = 0.0
    ewma: float = 0.0
    observation_count: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class CorrelationSnapshot:
    """Point-in-time correlation state."""
    timestamp: datetime
    regime: CorrelationRegime
    avg_correlation: float
    median_correlation: float
    max_correlation: float
    highly_correlated_pairs: List[CorrelationPair]
    diversification_score: float
    pair_count: int
    observation_count: int
    warnings: List[str]


@dataclass
class CorrelationAlert:
    """Alert from correlation monitoring."""
    timestamp: datetime
    alert_type: str  # "high_correlation", "regime_change", "crisis"
    message: str
    pairs: List[Tuple[str, str]]
    correlation_value: float
    severity: str


# ---------------------------------------------------------------------------
# Correlation Monitor
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """
    Monitors cross-position correlations in real-time.

    Calculates rolling correlation matrices, detects correlation
    regime changes, and provides diversification scoring.
    """

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self._regime = CorrelationRegime.NORMAL

        # Price/return data storage: symbol -> deque of returns
        self._returns: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_periods * 2)
        )
        self._prices: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_periods * 2)
        )

        # Correlation matrix: (symbolA, symbolB) -> CorrelationPair
        self._correlation_matrix: Dict[Tuple[str, str], CorrelationPair] = {}

        # Regime history
        self._regime_history: List[Tuple[datetime, CorrelationRegime]] = []
        self._avg_corr_history: deque = deque(maxlen=1000)

        # Alerts
        self._alerts: List[CorrelationAlert] = []
        self._max_alerts = 5000

        # Timing
        self._last_update: Optional[datetime] = None

        # Callbacks
        self._alert_callbacks: List = []
        self._regime_callbacks: List = []

    # ------------------------------------------------------------------
    # Data input
    # ------------------------------------------------------------------

    def add_price(self, symbol: str, price: float) -> None:
        """
        Add a price observation for a symbol.

        Calculates log return from the previous price.
        """
        prices = self._prices[symbol]
        prices.append(price)

        if len(prices) >= 2:
            prev_price = prices[-2]
            if prev_price > 0:
                log_return = math.log(price / prev_price)
                self._returns[symbol].append(log_return)

    def add_returns(self, symbol: str, returns: List[float]) -> None:
        """Add multiple return observations at once (for initialization)."""
        for r in returns:
            self._returns[symbol].append(r)

    # ------------------------------------------------------------------
    # Correlation calculation
    # ------------------------------------------------------------------

    def update(self) -> Optional[CorrelationSnapshot]:
        """
        Recalculate the full correlation matrix.

        Returns a snapshot if enough data is available, None otherwise.
        """
        now = datetime.now(timezone.utc)

        # Rate limiting
        if self._last_update:
            elapsed = (now - self._last_update).total_seconds()
            if elapsed < self.config.update_interval_seconds:
                return None

        symbols = [
            sym for sym, rets in self._returns.items()
            if len(rets) >= self.config.min_observations
        ]

        if len(symbols) < 2:
            return None

        self._last_update = now

        # Calculate pairwise correlations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]
                self._calculate_pair_correlation(sym_a, sym_b, now)

        # Detect regime
        self._detect_regime(now)

        # Generate snapshot
        snapshot = self._create_snapshot(now)

        # Check for alerts
        self._check_alerts(snapshot, now)

        return snapshot

    def _calculate_pair_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        now: datetime,
    ) -> None:
        """Calculate all correlation types for a pair of symbols."""
        returns_a = list(self._returns[symbol_a])
        returns_b = list(self._returns[symbol_b])

        # Align to same length (use most recent)
        min_len = min(len(returns_a), len(returns_b))
        if min_len < self.config.min_observations:
            return

        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]

        # Use the most recent `lookback_periods` observations
        n = min(min_len, self.config.lookback_periods)
        ra = returns_a[-n:]
        rb = returns_b[-n:]

        key = tuple(sorted([symbol_a, symbol_b]))
        pair = self._correlation_matrix.get(key)
        if pair is None:
            pair = CorrelationPair(symbol_a=key[0], symbol_b=key[1])
            self._correlation_matrix[key] = pair

        pair.pearson = self._pearson_correlation(ra, rb)
        pair.spearman = self._spearman_correlation(ra, rb)
        pair.ewma = self._ewma_correlation(ra, rb)
        pair.observation_count = n
        pair.last_updated = now

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        var_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return 0.0

        return cov / denom

    def _spearman_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Spearman rank correlation."""
        n = len(x)
        if n < 2:
            return 0.0

        # Rank the values
        rank_x = self._rank(x)
        rank_y = self._rank(y)

        # Pearson on ranks
        return self._pearson_correlation(rank_x, rank_y)

    def _rank(self, values: List[float]) -> List[float]:
        """Calculate ranks for a list of values (average rank for ties)."""
        n = len(values)
        indexed = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n

        i = 0
        while i < n:
            j = i
            # Find all equal values
            while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
                j += 1
            # Assign average rank to all tied values
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1

        return ranks

    def _ewma_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate exponentially weighted moving average correlation.

        More recent observations get higher weight, making the correlation
        more responsive to recent changes.
        """
        n = len(x)
        if n < 2:
            return 0.0

        span = min(self.config.ewma_span, n)
        alpha = 2.0 / (span + 1.0)

        # Calculate EWMA means
        ewma_x = x[0]
        ewma_y = y[0]
        ewma_xy = x[0] * y[0]
        ewma_x2 = x[0] ** 2
        ewma_y2 = y[0] ** 2

        for i in range(1, n):
            ewma_x = alpha * x[i] + (1 - alpha) * ewma_x
            ewma_y = alpha * y[i] + (1 - alpha) * ewma_y
            ewma_xy = alpha * x[i] * y[i] + (1 - alpha) * ewma_xy
            ewma_x2 = alpha * x[i] ** 2 + (1 - alpha) * ewma_x2
            ewma_y2 = alpha * y[i] ** 2 + (1 - alpha) * ewma_y2

        cov = ewma_xy - ewma_x * ewma_y
        var_x = ewma_x2 - ewma_x ** 2
        var_y = ewma_y2 - ewma_y ** 2

        if var_x <= 0 or var_y <= 0:
            return 0.0

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return 0.0

        corr = cov / denom
        # Clamp to [-1, 1] for numerical stability
        return max(-1.0, min(1.0, corr))

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(self, now: datetime) -> None:
        """
        Detect the current correlation regime.

        Regimes:
        - NORMAL: average correlation below elevated threshold
        - ELEVATED: average correlation above normal but below crisis
        - CRISIS: average correlation above crisis threshold (crash behavior)
        """
        if not self._correlation_matrix:
            return

        correlations = [
            abs(pair.ewma) for pair in self._correlation_matrix.values()
            if pair.observation_count >= self.config.min_observations
        ]

        if not correlations:
            return

        avg_corr = sum(correlations) / len(correlations)
        self._avg_corr_history.append(avg_corr)

        new_regime = self._regime
        if avg_corr >= self.config.regime_crisis_avg_corr:
            new_regime = CorrelationRegime.CRISIS
        elif avg_corr >= self.config.regime_elevated_avg_corr:
            new_regime = CorrelationRegime.ELEVATED
        else:
            new_regime = CorrelationRegime.NORMAL

        if new_regime != self._regime:
            old_regime = self._regime
            self._regime = new_regime
            self._regime_history.append((now, new_regime))

            logger.info(
                f"Correlation regime change: {old_regime.value} -> {new_regime.value} "
                f"(avg_corr={avg_corr:.3f})"
            )

            if self.config.alert_on_regime_change:
                alert = CorrelationAlert(
                    timestamp=now,
                    alert_type="regime_change",
                    message=(
                        f"Correlation regime: {old_regime.value} -> {new_regime.value} "
                        f"(avg_corr={avg_corr:.3f})"
                    ),
                    pairs=[],
                    correlation_value=avg_corr,
                    severity="critical" if new_regime == CorrelationRegime.CRISIS else "warning",
                )
                self._emit_alert(alert)

            for callback in self._regime_callbacks:
                try:
                    callback(old_regime, new_regime)
                except Exception as exc:
                    logger.error(f"Regime callback error: {exc}")

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _check_alerts(self, snapshot: CorrelationSnapshot, now: datetime) -> None:
        """Check for correlation alerts."""
        if not self.config.alert_on_high_correlation:
            return

        highly_correlated = snapshot.highly_correlated_pairs
        if len(highly_correlated) > self.config.max_correlated_pairs:
            alert = CorrelationAlert(
                timestamp=now,
                alert_type="high_correlation",
                message=(
                    f"Too many highly correlated pairs: {len(highly_correlated)} "
                    f"(max recommended: {self.config.max_correlated_pairs})"
                ),
                pairs=[
                    (p.symbol_a, p.symbol_b) for p in highly_correlated
                ],
                correlation_value=snapshot.avg_correlation,
                severity="warning",
            )
            self._emit_alert(alert)

    def _emit_alert(self, alert: CorrelationAlert) -> None:
        """Store and emit an alert."""
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as exc:
                logger.error(f"Alert callback error: {exc}")

    # ------------------------------------------------------------------
    # Snapshot and reporting
    # ------------------------------------------------------------------

    def _create_snapshot(self, now: datetime) -> CorrelationSnapshot:
        """Create a correlation snapshot."""
        valid_pairs = [
            pair for pair in self._correlation_matrix.values()
            if pair.observation_count >= self.config.min_observations
        ]

        if not valid_pairs:
            return CorrelationSnapshot(
                timestamp=now,
                regime=self._regime,
                avg_correlation=0.0,
                median_correlation=0.0,
                max_correlation=0.0,
                highly_correlated_pairs=[],
                diversification_score=100.0,
                pair_count=0,
                observation_count=0,
                warnings=[],
            )

        correlations = [abs(p.ewma) for p in valid_pairs]
        correlations_sorted = sorted(correlations)

        avg_corr = sum(correlations) / len(correlations)
        median_corr = correlations_sorted[len(correlations_sorted) // 2]
        max_corr = max(correlations)

        highly_correlated = [
            p for p in valid_pairs
            if abs(p.ewma) >= self.config.high_correlation_threshold
        ]

        diversification = self._calculate_diversification_score(valid_pairs)

        min_obs = min(p.observation_count for p in valid_pairs)

        warnings = []
        if self._regime == CorrelationRegime.CRISIS:
            warnings.append("CRISIS regime - correlations are elevated, diversification is reduced")
        if len(highly_correlated) > self.config.max_correlated_pairs:
            warnings.append(
                f"{len(highly_correlated)} highly correlated pairs detected"
            )

        return CorrelationSnapshot(
            timestamp=now,
            regime=self._regime,
            avg_correlation=avg_corr,
            median_correlation=median_corr,
            max_correlation=max_corr,
            highly_correlated_pairs=highly_correlated,
            diversification_score=diversification,
            pair_count=len(valid_pairs),
            observation_count=min_obs,
            warnings=warnings,
        )

    def _calculate_diversification_score(
        self,
        pairs: List[CorrelationPair],
    ) -> float:
        """
        Calculate portfolio diversification score (0-100).

        Based on average absolute pairwise correlation.
        0 = all positions perfectly correlated (no diversification)
        100 = all positions uncorrelated (perfect diversification)
        """
        if not pairs:
            return 100.0

        avg_abs_corr = sum(abs(p.ewma) for p in pairs) / len(pairs)

        # Score is inversely related to average correlation
        score = (1.0 - avg_abs_corr) * 100
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # VaR adjustment
    # ------------------------------------------------------------------

    def get_correlation_adjusted_var(
        self,
        position_vars: Dict[str, float],
    ) -> float:
        """
        Calculate correlation-adjusted portfolio VaR.

        Uses the correlation matrix to account for diversification benefit
        (or lack thereof) when calculating portfolio VaR.

        VaR_portfolio = sqrt(sum_i sum_j VaR_i * VaR_j * corr_ij)

        Args:
            position_vars: Symbol -> individual position VaR mapping

        Returns:
            Correlation-adjusted portfolio VaR
        """
        symbols = list(position_vars.keys())
        n = len(symbols)

        if n == 0:
            return 0.0
        if n == 1:
            return list(position_vars.values())[0]

        # Build correlation matrix
        portfolio_var_sq = 0.0

        for i in range(n):
            for j in range(n):
                sym_i = symbols[i]
                sym_j = symbols[j]
                var_i = position_vars[sym_i]
                var_j = position_vars[sym_j]

                if i == j:
                    corr = 1.0
                else:
                    key = tuple(sorted([sym_i, sym_j]))
                    pair = self._correlation_matrix.get(key)
                    corr = pair.ewma if pair else 0.0

                portfolio_var_sq += var_i * var_j * corr

        if portfolio_var_sq < 0:
            # Negative variance can occur with negative correlations
            # In this case, diversification benefit is very high
            return 0.0

        return math.sqrt(portfolio_var_sq)

    # ------------------------------------------------------------------
    # Adjustment suggestions
    # ------------------------------------------------------------------

    def suggest_adjustments(
        self,
        open_symbols: List[str],
    ) -> List[Dict]:
        """
        Suggest position adjustments for better decorrelation.

        Returns a list of suggestions, each with:
        - type: "reduce", "hedge", "diversify"
        - symbols: affected symbols
        - reason: explanation
        """
        suggestions = []

        # Find highly correlated same-direction pairs
        for key, pair in self._correlation_matrix.items():
            if pair.observation_count < self.config.min_observations:
                continue

            if abs(pair.ewma) < self.config.high_correlation_threshold:
                continue

            if pair.symbol_a not in open_symbols or pair.symbol_b not in open_symbols:
                continue

            if pair.ewma > 0:
                suggestions.append({
                    "type": "reduce",
                    "symbols": [pair.symbol_a, pair.symbol_b],
                    "correlation": pair.ewma,
                    "reason": (
                        f"{pair.symbol_a} and {pair.symbol_b} are highly correlated "
                        f"(r={pair.ewma:.2f}). Consider reducing one position to "
                        f"lower effective exposure."
                    ),
                })
            else:
                suggestions.append({
                    "type": "hedge",
                    "symbols": [pair.symbol_a, pair.symbol_b],
                    "correlation": pair.ewma,
                    "reason": (
                        f"{pair.symbol_a} and {pair.symbol_b} are negatively correlated "
                        f"(r={pair.ewma:.2f}). This provides natural hedging."
                    ),
                })

        # Check for sector concentration
        sector_symbols: Dict[str, List[str]] = defaultdict(list)
        for sym in open_symbols:
            # Simple sector detection from symbol name patterns
            sector_symbols["all"].append(sym)

        if len(suggestions) == 0 and len(open_symbols) <= 2:
            suggestions.append({
                "type": "diversify",
                "symbols": open_symbols,
                "correlation": 0.0,
                "reason": (
                    "Portfolio has few positions. Consider adding uncorrelated "
                    "assets to improve diversification."
                ),
            })

        return suggestions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Get the current correlation matrix as a dict of (symbolA, symbolB) -> correlation.

        Uses EWMA correlation by default.
        """
        return {
            key: pair.ewma
            for key, pair in self._correlation_matrix.items()
            if pair.observation_count >= self.config.min_observations
        }

    def get_pair_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        method: CorrelationMethod = CorrelationMethod.EWMA,
    ) -> Optional[float]:
        """Get correlation between two specific symbols."""
        key = tuple(sorted([symbol_a, symbol_b]))
        pair = self._correlation_matrix.get(key)

        if pair is None or pair.observation_count < self.config.min_observations:
            return None

        if method == CorrelationMethod.PEARSON:
            return pair.pearson
        elif method == CorrelationMethod.SPEARMAN:
            return pair.spearman
        else:
            return pair.ewma

    def get_regime(self) -> CorrelationRegime:
        """Get current correlation regime."""
        return self._regime

    def get_snapshot(self) -> CorrelationSnapshot:
        """Get current correlation snapshot."""
        return self._create_snapshot(datetime.now(timezone.utc))

    def get_alerts(self, limit: int = 100) -> List[CorrelationAlert]:
        """Get recent correlation alerts."""
        return self._alerts[-limit:]

    def get_symbols(self) -> List[str]:
        """Get list of symbols being tracked."""
        return list(self._returns.keys())

    def get_regime_history(self) -> List[Tuple[datetime, str]]:
        """Get regime change history."""
        return [(ts, regime.value) for ts, regime in self._regime_history]

    def register_alert_callback(self, callback) -> None:
        """Register a callback for correlation alerts."""
        self._alert_callbacks.append(callback)

    def register_regime_callback(self, callback) -> None:
        """Register a callback for regime changes."""
        self._regime_callbacks.append(callback)

    def clear(self) -> None:
        """Clear all data and reset."""
        self._returns.clear()
        self._prices.clear()
        self._correlation_matrix.clear()
        self._regime = CorrelationRegime.NORMAL
        self._regime_history.clear()
        self._avg_corr_history.clear()
        self._alerts.clear()
        self._last_update = None
        logger.info("Correlation monitor cleared")

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            "regime": self._regime.value,
            "symbols": list(self._returns.keys()),
            "pair_count": len(self._correlation_matrix),
            "matrix": {
                f"{k[0]}:{k[1]}": {
                    "pearson": p.pearson,
                    "spearman": p.spearman,
                    "ewma": p.ewma,
                    "observations": p.observation_count,
                }
                for k, p in self._correlation_matrix.items()
                if p.observation_count >= self.config.min_observations
            },
            "config": {
                "lookback_periods": self.config.lookback_periods,
                "ewma_span": self.config.ewma_span,
                "high_correlation_threshold": self.config.high_correlation_threshold,
            },
        }
