"""
Mefai Signal Engine - Statistical Pairs Trading Strategy

Cointegration test (Engle-Granger), spread calculation with z-score,
half-life of mean reversion (Ornstein-Uhlenbeck), dynamic hedge ratio
(rolling OLS), entry/exit at configurable SD levels, stop at 3 SD,
and pair selection algorithm.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import numpy as np

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class PairInfo:
    """Information about a trading pair relationship."""
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    correlation: float
    cointegration_pvalue: float
    half_life: float
    spread_mean: float
    spread_std: float
    is_cointegrated: bool


@dataclass
class PairsPosition:
    """Tracks a pairs trade (two legs)."""
    symbol_a: str
    symbol_b: str
    side_a: str             # "LONG" or "SHORT"
    side_b: str
    qty_a: float
    qty_b: float
    entry_price_a: float
    entry_price_b: float
    entry_spread: float
    entry_zscore: float
    entry_time: float
    hedge_ratio: float


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical Pairs Trading Strategy

    Trades the spread between two correlated/cointegrated assets:
    1. Engle-Granger cointegration test to validate pair
    2. Spread z-score calculation for entry/exit
    3. Ornstein-Uhlenbeck half-life for holding period estimation
    4. Rolling OLS for dynamic hedge ratio
    5. Entry at 2 SD, exit at 0.5 SD, stop at 3 SD
    6. Pair selection algorithm scans for cointegrated pairs
    """

    name = "PairsTrading"
    description = "Statistical pairs trading with cointegration and mean reversion"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Pair configuration
        self.symbol_b: str = self.params.get("symbol_b", "")
        self.lookback: int = self.params.get("lookback", 60)
        self.hedge_ratio_window: int = self.params.get("hedge_ratio_window", 30)

        # Z-score thresholds
        self.zscore_entry: float = self.params.get("zscore_entry", 2.0)
        self.zscore_exit: float = self.params.get("zscore_exit", 0.5)
        self.zscore_stop: float = self.params.get("zscore_stop", 3.0)

        # Cointegration
        self.coint_pvalue_threshold: float = self.params.get("coint_pvalue_threshold", 0.05)
        self.min_correlation: float = self.params.get("min_correlation", 0.7)
        self.min_half_life: float = self.params.get("min_half_life", 1.0)
        self.max_half_life: float = self.params.get("max_half_life", 50.0)

        # Risk
        self.risk_pct: float = self.params.get("risk_pct", 1.0)
        self.max_notional: float = self.params.get("max_notional", 10000.0)

        # State
        self._price_history_a: List[float] = []
        self._price_history_b: List[float] = []
        self._pair_info: Optional[PairInfo] = None
        self._pairs_position: Optional[PairsPosition] = None
        self._last_hedge_ratio: float = 1.0
        self._spread_history: List[float] = []

    # ------------------------------------------------------------------
    # Statistical calculations
    # ------------------------------------------------------------------

    def _ols_regression(
        self, y: np.ndarray, x: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Ordinary Least Squares regression.
        Returns (slope, intercept, residuals).
        """
        n = len(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0:
            return 1.0, 0.0, y - x

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        residuals = y - (slope * x + intercept)

        return float(slope), float(intercept), residuals

    def _adf_test(self, series: np.ndarray) -> float:
        """
        Simplified Augmented Dickey-Fuller test.
        Returns approximate p-value.
        Uses the ADF test statistic compared to critical values.
        """
        n = len(series)
        if n < 20:
            return 1.0

        # First difference
        diff = np.diff(series)
        lagged = series[:-1]

        # Regression: diff = alpha + beta * lagged + error
        x = lagged
        y = diff

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0:
            return 1.0

        beta = ss_xy / ss_xx
        residuals = y - (beta * x + (y_mean - beta * x_mean))
        se_beta = np.sqrt(np.sum(residuals ** 2) / (n - 3)) / np.sqrt(ss_xx)

        if se_beta == 0:
            return 1.0

        t_stat = beta / se_beta

        # Approximate p-value from ADF critical values
        # Critical values for n=100: 1%=-3.51, 5%=-2.89, 10%=-2.58
        if t_stat < -3.51:
            return 0.01
        elif t_stat < -2.89:
            return 0.05
        elif t_stat < -2.58:
            return 0.10
        elif t_stat < -1.95:
            return 0.20
        else:
            return 0.50

    def _test_cointegration(
        self, series_a: np.ndarray, series_b: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Engle-Granger cointegration test.
        1. Regress A on B
        2. Test residuals for stationarity (ADF)
        Returns (p_value, hedge_ratio, spread).
        """
        slope, intercept, residuals = self._ols_regression(series_a, series_b)
        p_value = self._adf_test(residuals)
        return p_value, slope, residuals

    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
        Half-life = -ln(2) / theta, where theta is estimated from
        regression of spread changes on lagged spread.
        """
        if len(spread) < 10:
            return float("inf")

        lagged = spread[:-1]
        diff = np.diff(spread)

        x_mean = np.mean(lagged)
        y_mean = np.mean(diff)

        ss_xy = np.sum((lagged - x_mean) * (diff - y_mean))
        ss_xx = np.sum((lagged - x_mean) ** 2)

        if ss_xx == 0:
            return float("inf")

        theta = ss_xy / ss_xx

        if theta >= 0:
            return float("inf")  # not mean reverting

        half_life = -np.log(2) / theta
        return float(half_life)

    def _rolling_hedge_ratio(
        self, series_a: np.ndarray, series_b: np.ndarray,
    ) -> float:
        """Calculate rolling OLS hedge ratio over recent window."""
        window = min(self.hedge_ratio_window, len(series_a))
        if window < 10:
            return 1.0

        a = series_a[-window:]
        b = series_b[-window:]
        slope, _, _ = self._ols_regression(a, b)
        return slope

    # ------------------------------------------------------------------
    # Spread and z-score
    # ------------------------------------------------------------------

    def _calculate_spread(
        self, price_a: float, price_b: float, hedge_ratio: float,
    ) -> float:
        """Calculate the spread between the two assets."""
        return price_a - hedge_ratio * price_b

    def _calculate_zscore(self, spread: float) -> float:
        """Calculate z-score of current spread."""
        if len(self._spread_history) < 10:
            return 0.0

        window = self._spread_history[-self.lookback:]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1))

        if std == 0:
            return 0.0

        return (spread - mean) / std

    # ------------------------------------------------------------------
    # Pair selection
    # ------------------------------------------------------------------

    def scan_pairs(
        self, price_data: Dict[str, np.ndarray],
    ) -> List[PairInfo]:
        """
        Scan all symbol pairs for cointegration.
        price_data: {symbol: np.ndarray of closes}
        Returns list of cointegrated pairs sorted by quality.
        """
        symbols = list(price_data.keys())
        pairs: List[PairInfo] = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a = symbols[i]
                sym_b = symbols[j]
                prices_a = price_data[sym_a]
                prices_b = price_data[sym_b]

                min_len = min(len(prices_a), len(prices_b))
                if min_len < 30:
                    continue

                a = prices_a[-min_len:]
                b = prices_b[-min_len:]

                # Correlation check
                corr = float(np.corrcoef(a, b)[0, 1])
                if abs(corr) < self.min_correlation:
                    continue

                # Cointegration test
                p_value, hedge_ratio, spread = self._test_cointegration(a, b)
                is_coint = p_value < self.coint_pvalue_threshold

                if not is_coint:
                    continue

                # Half-life
                half_life = self._calculate_half_life(spread)
                if half_life < self.min_half_life or half_life > self.max_half_life:
                    continue

                pairs.append(PairInfo(
                    symbol_a=sym_a,
                    symbol_b=sym_b,
                    hedge_ratio=hedge_ratio,
                    correlation=corr,
                    cointegration_pvalue=p_value,
                    half_life=half_life,
                    spread_mean=float(np.mean(spread)),
                    spread_std=float(np.std(spread, ddof=1)),
                    is_cointegrated=True,
                ))

        # Sort by p-value (lower is better)
        pairs.sort(key=lambda p: p.cointegration_pvalue)
        return pairs

    # ------------------------------------------------------------------
    # Pair validation
    # ------------------------------------------------------------------

    def _validate_pair(self) -> bool:
        """Validate that the current pair is still cointegrated."""
        if len(self._price_history_a) < 30 or len(self._price_history_b) < 30:
            return False

        a = np.array(self._price_history_a[-self.lookback:])
        b = np.array(self._price_history_b[-self.lookback:])

        p_value, hedge_ratio, spread = self._test_cointegration(a, b)
        self._last_hedge_ratio = hedge_ratio

        is_valid = p_value < self.coint_pvalue_threshold
        if is_valid:
            half_life = self._calculate_half_life(spread)
            is_valid = self.min_half_life <= half_life <= self.max_half_life

            if is_valid:
                self._pair_info = PairInfo(
                    symbol_a=self.symbol,
                    symbol_b=self.symbol_b,
                    hedge_ratio=hedge_ratio,
                    correlation=float(np.corrcoef(a, b)[0, 1]),
                    cointegration_pvalue=p_value,
                    half_life=half_life,
                    spread_mean=float(np.mean(spread)),
                    spread_std=float(np.std(spread, ddof=1)),
                    is_cointegrated=True,
                )

        return is_valid

    # ------------------------------------------------------------------
    # Update prices for symbol B
    # ------------------------------------------------------------------

    def update_pair_price(self, price_b: float) -> None:
        """Update the price of the second symbol. Called externally."""
        self._price_history_b.append(price_b)
        # Keep history bounded
        if len(self._price_history_b) > self.lookback * 3:
            self._price_history_b = self._price_history_b[-self.lookback * 2:]

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        if len(candles) < 30:
            return None

        close = candles[:, 4].astype(np.float64)
        current_price_a = float(close[-1])

        # Update price history for symbol A
        self._price_history_a.append(current_price_a)
        if len(self._price_history_a) > self.lookback * 3:
            self._price_history_a = self._price_history_a[-self.lookback * 2:]

        # Need price history for both symbols
        if len(self._price_history_b) < 30:
            return None

        current_price_b = self._price_history_b[-1]

        # Validate pair periodically
        if self._pair_info is None or len(self._price_history_a) % 20 == 0:
            if not self._validate_pair():
                if self._pairs_position is not None:
                    # Pair broke down - exit
                    return self._exit_pairs_trade(current_price_a, current_price_b, "pair_breakdown")
                return None

        # Dynamic hedge ratio
        if len(self._price_history_a) >= self.hedge_ratio_window:
            a = np.array(self._price_history_a[-self.hedge_ratio_window:])
            b = np.array(self._price_history_b[-self.hedge_ratio_window:])
            self._last_hedge_ratio = self._rolling_hedge_ratio(a, b)

        # Calculate spread and z-score
        spread = self._calculate_spread(current_price_a, current_price_b, self._last_hedge_ratio)
        self._spread_history.append(spread)
        if len(self._spread_history) > self.lookback * 3:
            self._spread_history = self._spread_history[-self.lookback * 2:]

        zscore = self._calculate_zscore(spread)

        # Exit check
        if self._pairs_position is not None:
            return self._evaluate_exit(zscore, current_price_a, current_price_b)

        # Entry check
        if abs(zscore) >= self.zscore_entry:
            return self._evaluate_entry(zscore, current_price_a, current_price_b, spread)

        return None

    def _evaluate_entry(
        self, zscore: float, price_a: float, price_b: float, spread: float,
    ) -> Optional[StrategySignal]:
        """Evaluate pairs trade entry."""
        notional = min(self.max_notional, self.params.get("account_balance", 10000.0) * 0.5)

        if zscore > self.zscore_entry:
            # Spread is too high - sell A, buy B (expect spread to narrow)
            qty_a = notional / price_a / 2.0
            qty_b = (notional / 2.0) / price_b * self._last_hedge_ratio

            self._pairs_position = PairsPosition(
                symbol_a=self.symbol,
                symbol_b=self.symbol_b,
                side_a="SHORT",
                side_b="LONG",
                qty_a=qty_a,
                qty_b=qty_b,
                entry_price_a=price_a,
                entry_price_b=price_b,
                entry_spread=spread,
                entry_zscore=zscore,
                entry_time=time.time(),
                hedge_ratio=self._last_hedge_ratio,
            )

            return StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=price_a,
                quantity=qty_a,
                confidence=min(0.85, 0.5 + abs(zscore) * 0.1),
                reason=f"Pairs trade: SHORT {self.symbol} / LONG {self.symbol_b}, z={zscore:.2f}",
                metadata={
                    "pair_symbol": self.symbol_b,
                    "pair_side": "LONG",
                    "pair_quantity": qty_b,
                    "pair_price": price_b,
                    "zscore": zscore,
                    "hedge_ratio": self._last_hedge_ratio,
                    "spread": spread,
                    "half_life": self._pair_info.half_life if self._pair_info else 0,
                },
            )

        elif zscore < -self.zscore_entry:
            # Spread is too low - buy A, sell B (expect spread to widen)
            qty_a = notional / price_a / 2.0
            qty_b = (notional / 2.0) / price_b * self._last_hedge_ratio

            self._pairs_position = PairsPosition(
                symbol_a=self.symbol,
                symbol_b=self.symbol_b,
                side_a="LONG",
                side_b="SHORT",
                qty_a=qty_a,
                qty_b=qty_b,
                entry_price_a=price_a,
                entry_price_b=price_b,
                entry_spread=spread,
                entry_zscore=zscore,
                entry_time=time.time(),
                hedge_ratio=self._last_hedge_ratio,
            )

            return StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=price_a,
                quantity=qty_a,
                confidence=min(0.85, 0.5 + abs(zscore) * 0.1),
                reason=f"Pairs trade: LONG {self.symbol} / SHORT {self.symbol_b}, z={zscore:.2f}",
                metadata={
                    "pair_symbol": self.symbol_b,
                    "pair_side": "SHORT",
                    "pair_quantity": qty_b,
                    "pair_price": price_b,
                    "zscore": zscore,
                    "hedge_ratio": self._last_hedge_ratio,
                    "spread": spread,
                    "half_life": self._pair_info.half_life if self._pair_info else 0,
                },
            )

        return None

    def _evaluate_exit(
        self, zscore: float, price_a: float, price_b: float,
    ) -> Optional[StrategySignal]:
        """Check exit conditions for pairs trade."""
        if self._pairs_position is None:
            return None

        should_exit = False
        reason = ""

        pp = self._pairs_position

        # Z-score reverted
        if pp.side_a == "SHORT" and zscore <= self.zscore_exit:
            should_exit = True
            reason = f"Spread reverted: z={zscore:.2f}"
        elif pp.side_a == "LONG" and zscore >= -self.zscore_exit:
            should_exit = True
            reason = f"Spread reverted: z={zscore:.2f}"

        # Stop loss
        if pp.side_a == "SHORT" and zscore > self.zscore_stop:
            should_exit = True
            reason = f"Z-score stop: z={zscore:.2f} > {self.zscore_stop}"
        elif pp.side_a == "LONG" and zscore < -self.zscore_stop:
            should_exit = True
            reason = f"Z-score stop: z={zscore:.2f} < -{self.zscore_stop}"

        if should_exit:
            return self._exit_pairs_trade(price_a, price_b, reason)

        return None

    def _exit_pairs_trade(
        self, price_a: float, price_b: float, reason: str,
    ) -> StrategySignal:
        """Close the pairs trade."""
        pp = self._pairs_position

        # Calculate PnL
        if pp.side_a == "LONG":
            pnl_a = (price_a - pp.entry_price_a) * pp.qty_a
            pnl_b = (pp.entry_price_b - price_b) * pp.qty_b
        else:
            pnl_a = (pp.entry_price_a - price_a) * pp.qty_a
            pnl_b = (price_b - pp.entry_price_b) * pp.qty_b

        total_pnl = pnl_a + pnl_b
        self.performance.trade_history.append({
            "type": "pairs",
            "pnl": total_pnl,
            "pnl_a": pnl_a,
            "pnl_b": pnl_b,
            "reason": reason,
        })
        self.performance.update(total_pnl)

        signal_type = SignalType.CLOSE_LONG if pp.side_a == "LONG" else SignalType.CLOSE_SHORT
        self._pairs_position = None

        return StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=price_a,
            quantity=pp.qty_a,
            confidence=0.8,
            reason=f"Pairs exit: {reason} PnL=${total_pnl:.2f}",
            metadata={
                "pair_symbol": pp.symbol_b,
                "pnl_a": pnl_a,
                "pnl_b": pnl_b,
                "total_pnl": total_pnl,
            },
        )

    def on_tick(self, price: float, volume: float, timestamp: float) -> Optional[StrategySignal]:
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        self._logger.info("Pairs fill: %s", order_info)
