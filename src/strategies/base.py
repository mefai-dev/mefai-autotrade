"""
Mefai Signal Engine - Base Strategy

Abstract base class for all trading strategies. Provides the interface contract,
indicator access, position management helpers, risk integration, logging,
performance tracking, and backtest compatibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import time
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class SignalType(str, Enum):
    """Direction of a trading signal."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    NEUTRAL = "NEUTRAL"


class StrategyState(str, Enum):
    """Lifecycle state of a strategy instance."""
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class StrategySignal:
    """
    Output of a strategy's generate_signal() method.

    Every field is concrete - no placeholders. The execution layer uses these
    values directly to place orders on the exchange.
    """
    signal_type: SignalType
    symbol: str
    price: float
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0          # 0.0 - 1.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_entry(self) -> bool:
        return self.signal_type in (SignalType.LONG, SignalType.SHORT)

    @property
    def is_exit(self) -> bool:
        return self.signal_type in (SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)

    @property
    def risk_reward_ratio(self) -> float:
        if self.stop_loss == 0 or self.take_profit == 0 or self.price == 0:
            return 0.0
        risk = abs(self.price - self.stop_loss)
        reward = abs(self.take_profit - self.price)
        if risk == 0:
            return 0.0
        return reward / risk


@dataclass
class PositionInfo:
    """Tracks an open position managed by a strategy."""
    symbol: str
    side: str                        # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    highest_price: float = 0.0       # for trailing stop
    lowest_price: float = 0.0        # for trailing stop
    entry_time: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Rolling performance stats kept by every strategy instance."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    def update(self, pnl: float) -> None:
        """Update metrics after a closed trade."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.equity_curve.append(self.total_pnl)

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        wins = [t["pnl"] for t in self.trade_history if t.get("pnl", 0) > 0]
        losses = [t["pnl"] for t in self.trade_history if t.get("pnl", 0) <= 0]
        self.avg_win = float(np.mean(wins)) if wins else 0.0
        self.avg_loss = float(np.mean(losses)) if losses else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        total_win = sum(wins) if wins else 0.0
        self.profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        # Max drawdown from equity curve
        if len(self.equity_curve) >= 2:
            peak = self.equity_curve[0]
            max_dd = 0.0
            for val in self.equity_curve:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown = max_dd

        # Consecutive wins/losses
        current_streak = 0
        streak_type = None
        for t in self.trade_history:
            p = t.get("pnl", 0)
            if p > 0:
                if streak_type == "win":
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = "win"
                self.max_consecutive_wins = max(self.max_consecutive_wins, current_streak)
            else:
                if streak_type == "loss":
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = "loss"
                self.max_consecutive_losses = max(self.max_consecutive_losses, current_streak)


# ---------------------------------------------------------------------------
# Feature / Indicator Engine (lightweight, built-in)
# ---------------------------------------------------------------------------

class FeatureEngine:
    """
    Provides real indicator calculations consumed by strategies.
    Works on numpy arrays of OHLCV data.
    """

    @staticmethod
    def sma(close: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        out = np.full_like(close, np.nan)
        if len(close) < period:
            return out
        cumsum = np.cumsum(close)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        out[period - 1:] = cumsum[period - 1:] / period
        return out

    @staticmethod
    def ema(close: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        out = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) < period:
            return out
        multiplier = 2.0 / (period + 1)
        out[period - 1] = np.mean(close[:period])
        for i in range(period, len(close)):
            out[i] = (close[i] - out[i - 1]) * multiplier + out[i - 1]
        return out

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index (Wilder's smoothing)."""
        out = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) < period + 1:
            return out
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        if avg_loss == 0:
            out[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[period] = 100.0 - (100.0 / (1.0 + rs))
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                out[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return out

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> np.ndarray:
        """Average True Range."""
        out = np.full(len(close), np.nan, dtype=np.float64)
        if len(close) < period + 1:
            return out
        tr = np.zeros(len(close), dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        out[period] = np.mean(tr[1:period + 1])
        for i in range(period + 1, len(close)):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
        return out

    @staticmethod
    def macd(close: np.ndarray, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD line, signal line, histogram."""
        fe = FeatureEngine
        fast_ema = fe.ema(close, fast)
        slow_ema = fe.ema(close, slow)
        macd_line = fast_ema - slow_ema
        signal_line = fe.ema(macd_line[~np.isnan(macd_line)], signal)
        # Align signal line
        full_signal = np.full_like(close, np.nan)
        offset = len(close) - len(signal_line)
        full_signal[offset:] = signal_line
        histogram = macd_line - full_signal
        return macd_line, full_signal, histogram

    @staticmethod
    def bollinger_bands(close: np.ndarray, period: int = 20,
                        std_dev: float = 2.0
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Upper band, middle band (SMA), lower band."""
        fe = FeatureEngine
        middle = fe.sma(close, period)
        rolling_std = np.full_like(close, np.nan)
        for i in range(period - 1, len(close)):
            rolling_std[i] = np.std(close[i - period + 1:i + 1], ddof=0)
        upper = middle + std_dev * rolling_std
        lower = middle - std_dev * rolling_std
        return upper, middle, lower

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> np.ndarray:
        """Average Directional Index."""
        n = len(close)
        out = np.full(n, np.nan, dtype=np.float64)
        if n < 2 * period + 1:
            return out

        up_move = np.zeros(n)
        down_move = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up_move[i] = high[i] - high[i - 1]
            down_move[i] = low[i - 1] - low[i]
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr_val = np.zeros(n)
        plus_di_arr = np.zeros(n)
        minus_di_arr = np.zeros(n)

        atr_val[period] = np.sum(tr[1:period + 1])
        smooth_plus = np.sum(plus_dm[1:period + 1])
        smooth_minus = np.sum(minus_dm[1:period + 1])
        plus_di_arr[period] = 100.0 * smooth_plus / atr_val[period] if atr_val[period] != 0 else 0
        minus_di_arr[period] = 100.0 * smooth_minus / atr_val[period] if atr_val[period] != 0 else 0

        for i in range(period + 1, n):
            atr_val[i] = atr_val[i - 1] - atr_val[i - 1] / period + tr[i]
            smooth_plus = smooth_plus - smooth_plus / period + plus_dm[i]
            smooth_minus = smooth_minus - smooth_minus / period + minus_dm[i]
            if atr_val[i] != 0:
                plus_di_arr[i] = 100.0 * smooth_plus / atr_val[i]
                minus_di_arr[i] = 100.0 * smooth_minus / atr_val[i]

        dx = np.zeros(n)
        for i in range(period, n):
            denom = plus_di_arr[i] + minus_di_arr[i]
            if denom != 0:
                dx[i] = 100.0 * abs(plus_di_arr[i] - minus_di_arr[i]) / denom

        # First ADX = mean of first 'period' DX values
        start_idx = 2 * period
        if start_idx < n:
            out[start_idx] = np.mean(dx[period:start_idx + 1])
            for i in range(start_idx + 1, n):
                out[i] = (out[i - 1] * (period - 1) + dx[i]) / period

        return out

    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 10, multiplier: float = 3.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (supertrend_line, direction). direction: 1 = up, -1 = down."""
        fe = FeatureEngine
        atr_vals = fe.atr(high, low, close, period)
        n = len(close)
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        supertrend = np.zeros(n)
        direction = np.ones(n)

        for i in range(n):
            mid = (high[i] + low[i]) / 2.0
            atr_v = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
            upper_band[i] = mid + multiplier * atr_v
            lower_band[i] = mid - multiplier * atr_v

        for i in range(1, n):
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                pass
            else:
                lower_band[i] = lower_band[i - 1]

            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                pass
            else:
                upper_band[i] = upper_band[i - 1]

            if supertrend[i - 1] == upper_band[i - 1]:
                if close[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
            else:
                if close[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1

        return supertrend, direction

    @staticmethod
    def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             volume: np.ndarray) -> np.ndarray:
        """Volume Weighted Average Price (cumulative)."""
        typical = (high + low + close) / 3.0
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical * volume)
        out = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
        return out

    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int = 14, d_period: int = 3
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic %K and %D."""
        n = len(close)
        k = np.full(n, np.nan)
        for i in range(k_period - 1, n):
            highest = np.max(high[i - k_period + 1:i + 1])
            lowest = np.min(low[i - k_period + 1:i + 1])
            denom = highest - lowest
            if denom == 0:
                k[i] = 50.0
            else:
                k[i] = 100.0 * (close[i] - lowest) / denom
        d = FeatureEngine.sma(k[~np.isnan(k)], d_period)
        full_d = np.full(n, np.nan)
        offset = n - len(d)
        full_d[offset:] = d
        return k, full_d

    @staticmethod
    def donchian(high: np.ndarray, low: np.ndarray,
                 period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian channel: upper, middle, lower."""
        n = len(high)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(period - 1, n):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        middle = (upper + lower) / 2.0
        return upper, middle, lower

    @staticmethod
    def keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        ema_period: int = 20, atr_period: int = 10,
                        multiplier: float = 1.5
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channel: upper, middle (EMA), lower."""
        fe = FeatureEngine
        middle = fe.ema(close, ema_period)
        atr_vals = fe.atr(high, low, close, atr_period)
        upper = middle + multiplier * atr_vals
        lower = middle - multiplier * atr_vals
        return upper, middle, lower

    @staticmethod
    def roc(close: np.ndarray, period: int = 12) -> np.ndarray:
        """Rate of Change (percentage)."""
        out = np.full(len(close), np.nan)
        for i in range(period, len(close)):
            if close[i - period] != 0:
                out[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
        return out

    @staticmethod
    def parabolic_sar(high: np.ndarray, low: np.ndarray,
                      af_start: float = 0.02, af_step: float = 0.02,
                      af_max: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Parabolic SAR. Returns (sar_values, direction). direction: 1=bull, -1=bear."""
        n = len(high)
        sar = np.zeros(n)
        direction = np.ones(n)
        af = af_start
        ep = low[0]
        sar[0] = high[0]
        direction[0] = -1

        for i in range(1, n):
            if direction[i - 1] == 1:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = min(sar[i], low[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 2])
                if low[i] < sar[i]:
                    direction[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    direction[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, af_max)
            else:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = max(sar[i], high[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 2])
                if high[i] > sar[i]:
                    direction[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    direction[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, af_max)

        return sar, direction


# ---------------------------------------------------------------------------
# Abstract Base Strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base for every trading strategy.

    Subclasses MUST implement:
      - generate_signal(candles) -> Optional[StrategySignal]
      - on_tick(price, volume, timestamp) -> Optional[StrategySignal]
      - on_bar(ohlcv_bar) -> Optional[StrategySignal]
      - on_fill(order_info) -> None
    """

    # Class-level description (override in subclass)
    name: str = "BaseStrategy"
    description: str = "Abstract base strategy"
    version: str = "1.0.0"

    def __init__(
        self,
        symbol: str,
        timeframe: str = "5m",
        params: Optional[Dict[str, Any]] = None,
        backtest_mode: bool = False,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}
        self.backtest_mode = backtest_mode

        self.state = StrategyState.CREATED
        self.indicators = FeatureEngine()
        self.position: Optional[PositionInfo] = None
        self.performance = PerformanceMetrics()
        self._logger = logging.getLogger(f"strategy.{self.name}.{symbol}")

        # Risk limits (can be overridden per-strategy)
        self.max_position_size: float = self.params.get("max_position_size", 1.0)
        self.max_drawdown_pct: float = self.params.get("max_drawdown_pct", 10.0)
        self.max_daily_trades: int = self.params.get("max_daily_trades", 50)
        self.daily_trade_count: int = 0
        self.daily_pnl: float = 0.0

        self._initialize_params()

    def _initialize_params(self) -> None:
        """Hook for subclasses to set up default params. Called from __init__."""
        pass

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signal(
        self,
        candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        """
        Primary signal generation from OHLCV candles.

        Args:
            candles: numpy array with columns [timestamp, open, high, low, close, volume]

        Returns:
            StrategySignal or None if no trade.
        """
        ...

    @abstractmethod
    def on_tick(
        self,
        price: float,
        volume: float,
        timestamp: float,
    ) -> Optional[StrategySignal]:
        """Called on every tick/trade update."""
        ...

    @abstractmethod
    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        """Called when a new bar closes."""
        ...

    @abstractmethod
    def on_fill(self, order_info: Dict[str, Any]) -> None:
        """Called when an order is filled."""
        ...

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def open_position(
        self,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> PositionInfo:
        """Create and store a new position."""
        self.position = PositionInfo(
            symbol=self.symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._logger.info(
            "Opened %s position: qty=%.6f entry=%.4f sl=%.4f tp=%.4f",
            side, quantity, entry_price, stop_loss, take_profit,
        )
        return self.position

    def close_position(self, exit_price: float, reason: str = "") -> float:
        """Close the current position, return realized PnL."""
        if self.position is None:
            return 0.0

        if self.position.side == "LONG":
            pnl = (exit_price - self.position.entry_price) * self.position.quantity
        else:
            pnl = (self.position.entry_price - exit_price) * self.position.quantity

        self.performance.trade_history.append({
            "symbol": self.symbol,
            "side": self.position.side,
            "entry_price": self.position.entry_price,
            "exit_price": exit_price,
            "quantity": self.position.quantity,
            "pnl": pnl,
            "reason": reason,
            "duration": time.time() - self.position.entry_time,
        })
        self.performance.update(pnl)
        self.daily_pnl += pnl
        self.daily_trade_count += 1

        self._logger.info(
            "Closed %s position at %.4f  PnL=%.4f  reason=%s",
            self.position.side, exit_price, pnl, reason,
        )
        self.position = None
        return pnl

    def update_position_price(self, current_price: float) -> None:
        """Update the position's unrealized PnL and high/low watermarks."""
        if self.position is None:
            return
        if current_price > self.position.highest_price:
            self.position.highest_price = current_price
        if current_price < self.position.lowest_price:
            self.position.lowest_price = current_price
        if self.position.side == "LONG":
            self.position.unrealized_pnl = (
                (current_price - self.position.entry_price) * self.position.quantity
            )
        else:
            self.position.unrealized_pnl = (
                (self.position.entry_price - current_price) * self.position.quantity
            )

    @property
    def has_position(self) -> bool:
        return self.position is not None

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def check_risk(self, signal: StrategySignal) -> bool:
        """
        Pre-trade risk gate. Returns True if the trade is allowed.
        """
        # Daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            self._logger.warning("Daily trade limit reached (%d)", self.max_daily_trades)
            return False

        # Max position size
        if signal.quantity > self.max_position_size:
            self._logger.warning(
                "Position size %.6f exceeds max %.6f",
                signal.quantity, self.max_position_size,
            )
            return False

        # Max drawdown
        if self.performance.max_drawdown > 0:
            dd_pct = (self.performance.max_drawdown / max(abs(self.performance.total_pnl), 1.0)) * 100
            if dd_pct > self.max_drawdown_pct:
                self._logger.warning("Max drawdown exceeded: %.2f%%", dd_pct)
                return False

        # Signal confidence threshold
        min_confidence = self.params.get("min_confidence", 0.0)
        if signal.confidence < min_confidence:
            self._logger.debug("Signal confidence %.2f below threshold %.2f",
                               signal.confidence, min_confidence)
            return False

        return True

    # ------------------------------------------------------------------
    # Position sizing helpers
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = 1.0,
    ) -> float:
        """
        Risk-based position sizing.
        risk_pct: percentage of account to risk on a single trade (e.g. 1.0 = 1%).
        """
        if entry_price == 0 or stop_loss == 0:
            return 0.0
        risk_amount = account_balance * (risk_pct / 100.0)
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0.0
        qty = risk_amount / risk_per_unit
        return min(qty, self.max_position_size)

    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5,
    ) -> float:
        """
        Half-Kelly position sizing.
        fraction: Kelly fraction (0.5 = half-Kelly for safety).
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        b = avg_win / abs(avg_loss)
        kelly = (win_rate * b - (1 - win_rate)) / b
        kelly = max(kelly, 0.0)
        return kelly * fraction

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------

    def check_trailing_stop(
        self,
        current_price: float,
        atr_value: float,
        atr_multiplier: float = 2.0,
    ) -> bool:
        """
        ATR-based trailing stop. Returns True if stop is hit.
        """
        if self.position is None:
            return False

        if self.position.side == "LONG":
            trail_stop = self.position.highest_price - atr_multiplier * atr_value
            if current_price <= trail_stop:
                return True
        else:
            trail_stop = self.position.lowest_price + atr_multiplier * atr_value
            if current_price >= trail_stop:
                return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.state = StrategyState.RUNNING
        self._logger.info("Strategy %s started for %s", self.name, self.symbol)

    def stop(self) -> None:
        self.state = StrategyState.STOPPED
        self._logger.info("Strategy %s stopped for %s", self.name, self.symbol)

    def pause(self) -> None:
        self.state = StrategyState.PAUSED

    def resume(self) -> None:
        self.state = StrategyState.RUNNING

    def reset_daily_counters(self) -> None:
        self.daily_trade_count = 0
        self.daily_pnl = 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "state": self.state.value,
            "has_position": self.has_position,
            "daily_trades": self.daily_trade_count,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.performance.total_pnl,
            "total_trades": self.performance.total_trades,
            "win_rate": self.performance.win_rate,
            "max_drawdown": self.performance.max_drawdown,
        }
