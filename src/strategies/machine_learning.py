"""
Mefai Signal Engine - ML-Integrated Strategy

Wrapper for Mefai Signal Engine predictions, XGBoost signal consumption,
transformer prediction integration, confidence filtering, regime-aware
trading, and dynamic parameter adjustment based on model accuracy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import numpy as np
from collections import deque

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Prediction from an ML model."""
    model_name: str
    direction: int           # 1 = long, -1 = short, 0 = neutral
    confidence: float        # 0.0 - 1.0
    predicted_return: float  # expected return percentage
    features_used: int
    timestamp: float
    regime: str = ""         # "trending", "ranging", "volatile"


@dataclass
class ModelPerformance:
    """Track model prediction accuracy."""
    model_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    recent_accuracy: float = 0.0    # last 50 predictions
    total_return: float = 0.0
    predictions: deque = field(default_factory=lambda: deque(maxlen=200))
    recent_correct: deque = field(default_factory=lambda: deque(maxlen=50))

    def record(self, was_correct: bool, actual_return: float) -> None:
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        self.accuracy = self.correct_predictions / self.total_predictions
        self.total_return += actual_return
        self.predictions.append({
            "correct": was_correct,
            "return": actual_return,
            "timestamp": time.time(),
        })
        self.recent_correct.append(1 if was_correct else 0)
        if len(self.recent_correct) > 0:
            self.recent_accuracy = sum(self.recent_correct) / len(self.recent_correct)


class MLStrategy(BaseStrategy):
    """
    ML-Integrated Strategy

    Consumes predictions from external ML models (Mefai Signal Engine):
    1. XGBoost signal consumption via API
    2. Transformer model prediction integration
    3. Confidence threshold filtering (only trade high-confidence)
    4. Regime-aware trading (reduce size in sideways/volatile)
    5. Dynamic parameter adjustment based on model accuracy
    6. Multi-model ensemble for robust signals
    """

    name = "MachineLearning"
    description = "ML-integrated strategy consuming Mefai Signal Engine predictions"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Confidence
        self.confidence_threshold: float = self.params.get("confidence_threshold", 0.6)
        self.high_confidence: float = self.params.get("high_confidence", 0.8)

        # Models
        self.model_type: str = self.params.get("model_type", "ensemble")
        self.enabled_models: List[str] = self.params.get(
            "enabled_models", ["xgboost", "transformer", "signal_engine"]
        )
        self.ensemble_method: str = self.params.get("ensemble_method", "weighted_average")
        self.min_models_agree: int = self.params.get("min_models_agree", 2)

        # Regime
        self.regime_detection: bool = self.params.get("regime_detection", True)
        self.trending_size_mult: float = self.params.get("trending_size_mult", 1.0)
        self.ranging_size_mult: float = self.params.get("ranging_size_mult", 0.5)
        self.volatile_size_mult: float = self.params.get("volatile_size_mult", 0.3)

        # Dynamic adjustment
        self.min_accuracy_to_trade: float = self.params.get("min_accuracy_to_trade", 0.52)
        self.accuracy_lookback: int = self.params.get("accuracy_lookback", 50)
        self.reduce_on_losing_streak: bool = self.params.get("reduce_on_losing_streak", True)
        self.max_losing_streak: int = self.params.get("max_losing_streak", 5)

        # Risk
        self.atr_period: int = self.params.get("atr_period", 14)
        self.atr_stop_mult: float = self.params.get("atr_stop_mult", 2.0)
        self.risk_pct: float = self.params.get("risk_pct", 1.0)

        # State
        self._pending_predictions: Dict[str, MLPrediction] = {}
        self._model_performance: Dict[str, ModelPerformance] = {}
        for model in self.enabled_models:
            self._model_performance[model] = ModelPerformance(model_name=model)

        self._current_regime: str = "unknown"
        self._losing_streak: int = 0
        self._entry_prediction: Optional[MLPrediction] = None

    # ------------------------------------------------------------------
    # Prediction ingestion
    # ------------------------------------------------------------------

    def submit_prediction(
        self,
        model_name: str,
        direction: int,
        confidence: float,
        predicted_return: float = 0.0,
        features_used: int = 0,
        regime: str = "",
    ) -> None:
        """
        Submit a prediction from an external ML model.
        Called by the data feed or prediction service.
        """
        if model_name not in self.enabled_models:
            return

        pred = MLPrediction(
            model_name=model_name,
            direction=direction,
            confidence=confidence,
            predicted_return=predicted_return,
            features_used=features_used,
            timestamp=time.time(),
            regime=regime,
        )
        self._pending_predictions[model_name] = pred
        self._logger.debug("Prediction from %s: dir=%d conf=%.2f",
                           model_name, direction, confidence)

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(self, candles: np.ndarray) -> str:
        """
        Detect market regime using ADX and volatility.
        Returns "trending", "ranging", or "volatile".
        """
        if not self.regime_detection:
            return "trending"

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)

        # ADX for trend
        adx_vals = self.indicators.adx(high, low, close, self.atr_period)
        adx = float(adx_vals[-1]) if not np.isnan(adx_vals[-1]) else 20.0

        # Volatility
        if len(close) > 20:
            returns = np.diff(np.log(close[-21:]))
            vol = float(np.std(returns))
        else:
            vol = 0.02

        if adx > 25:
            if vol > 0.04:
                return "volatile"
            return "trending"
        else:
            if vol > 0.04:
                return "volatile"
            return "ranging"

    def _get_size_multiplier(self) -> float:
        """Get position size multiplier based on regime."""
        if self._current_regime == "trending":
            return self.trending_size_mult
        elif self._current_regime == "ranging":
            return self.ranging_size_mult
        elif self._current_regime == "volatile":
            return self.volatile_size_mult
        return 1.0

    # ------------------------------------------------------------------
    # Ensemble logic
    # ------------------------------------------------------------------

    def _ensemble_predictions(self) -> Optional[MLPrediction]:
        """
        Combine predictions from multiple models into a single signal.
        """
        active_preds = {}
        stale_cutoff = time.time() - 300.0  # predictions older than 5 min are stale

        for name, pred in self._pending_predictions.items():
            if pred.timestamp > stale_cutoff:
                active_preds[name] = pred

        if len(active_preds) < self.min_models_agree:
            return None

        if self.ensemble_method == "weighted_average":
            return self._weighted_average_ensemble(active_preds)
        elif self.ensemble_method == "majority_vote":
            return self._majority_vote_ensemble(active_preds)
        else:
            return self._weighted_average_ensemble(active_preds)

    def _weighted_average_ensemble(
        self, predictions: Dict[str, MLPrediction],
    ) -> Optional[MLPrediction]:
        """Weighted average of predictions, weighted by recent accuracy."""
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        weighted_return = 0.0

        for name, pred in predictions.items():
            perf = self._model_performance.get(name)
            weight = perf.recent_accuracy if perf and perf.recent_accuracy > 0 else 0.5
            weight *= pred.confidence

            weighted_direction += pred.direction * weight
            weighted_confidence += pred.confidence * weight
            weighted_return += pred.predicted_return * weight
            total_weight += weight

        if total_weight == 0:
            return None

        avg_direction = weighted_direction / total_weight
        avg_confidence = weighted_confidence / total_weight
        avg_return = weighted_return / total_weight

        direction = 1 if avg_direction > 0.3 else (-1 if avg_direction < -0.3 else 0)
        if direction == 0:
            return None

        return MLPrediction(
            model_name="ensemble",
            direction=direction,
            confidence=avg_confidence,
            predicted_return=avg_return,
            features_used=sum(p.features_used for p in predictions.values()),
            timestamp=time.time(),
            regime=self._current_regime,
        )

    def _majority_vote_ensemble(
        self, predictions: Dict[str, MLPrediction],
    ) -> Optional[MLPrediction]:
        """Simple majority vote - most models agree."""
        long_votes = 0
        short_votes = 0
        total_confidence = 0.0
        total_return = 0.0

        for pred in predictions.values():
            if pred.confidence < self.confidence_threshold:
                continue
            if pred.direction == 1:
                long_votes += 1
            elif pred.direction == -1:
                short_votes += 1
            total_confidence += pred.confidence
            total_return += pred.predicted_return

        num_voted = long_votes + short_votes
        if num_voted < self.min_models_agree:
            return None

        if long_votes > short_votes and long_votes >= self.min_models_agree:
            return MLPrediction(
                model_name="ensemble_vote",
                direction=1,
                confidence=total_confidence / num_voted,
                predicted_return=total_return / num_voted,
                features_used=0,
                timestamp=time.time(),
                regime=self._current_regime,
            )
        elif short_votes > long_votes and short_votes >= self.min_models_agree:
            return MLPrediction(
                model_name="ensemble_vote",
                direction=-1,
                confidence=total_confidence / num_voted,
                predicted_return=total_return / num_voted,
                features_used=0,
                timestamp=time.time(),
                regime=self._current_regime,
            )

        return None

    # ------------------------------------------------------------------
    # Accuracy-based adjustments
    # ------------------------------------------------------------------

    def _should_trade(self) -> bool:
        """Check if model accuracy is sufficient to continue trading."""
        for name, perf in self._model_performance.items():
            if perf.total_predictions >= self.accuracy_lookback:
                if perf.recent_accuracy < self.min_accuracy_to_trade:
                    self._logger.warning(
                        "Model %s accuracy %.2f below minimum %.2f - pausing",
                        name, perf.recent_accuracy, self.min_accuracy_to_trade,
                    )
                    return False

        if self.reduce_on_losing_streak and self._losing_streak >= self.max_losing_streak:
            self._logger.warning(
                "Losing streak %d reached max %d - pausing",
                self._losing_streak, self.max_losing_streak,
            )
            return False

        return True

    def _adjust_risk_for_accuracy(self) -> float:
        """Adjust risk percentage based on recent model accuracy."""
        best_accuracy = 0.0
        for perf in self._model_performance.values():
            if perf.recent_accuracy > best_accuracy:
                best_accuracy = perf.recent_accuracy

        if best_accuracy > 0.7:
            return self.risk_pct * 1.5
        elif best_accuracy > 0.6:
            return self.risk_pct
        elif best_accuracy > 0.55:
            return self.risk_pct * 0.7
        else:
            return self.risk_pct * 0.5

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        if len(candles) < self.atr_period + 5:
            return None

        close = candles[:, 4].astype(np.float64)
        high = candles[:, 2].astype(np.float64)
        low = candles[:, 3].astype(np.float64)
        current_price = float(close[-1])

        # Update regime
        self._current_regime = self._detect_regime(candles)

        # ATR
        atr = self.indicators.atr(high, low, close, self.atr_period)
        atr_val = float(atr[-1]) if not np.isnan(atr[-1]) else current_price * 0.02

        # Exit check
        if self.has_position:
            self.update_position_price(current_price)

            if self.position.stop_loss > 0:
                if self.position.side == "LONG" and current_price <= self.position.stop_loss:
                    self._losing_streak += 1
                    self.close_position(current_price, "stop_loss")
                    if self._entry_prediction:
                        for perf in self._model_performance.values():
                            perf.record(False, -abs(current_price - self.position.entry_price) / self.position.entry_price * 100 if self.position else 0)
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol, price=current_price,
                        confidence=0.9, reason="ML stop loss",
                    )
                elif self.position.side == "SHORT" and current_price >= self.position.stop_loss:
                    self._losing_streak += 1
                    self.close_position(current_price, "stop_loss")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol, price=current_price,
                        confidence=0.9, reason="ML stop loss",
                    )

            if self.position.take_profit > 0:
                if self.position.side == "LONG" and current_price >= self.position.take_profit:
                    self._losing_streak = 0
                    self.close_position(current_price, "take_profit")
                    for perf in self._model_performance.values():
                        perf.record(True, abs(current_price - self.position.entry_price) / self.position.entry_price * 100 if self.position else 0)
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_LONG,
                        symbol=self.symbol, price=current_price,
                        confidence=0.85, reason="ML take profit",
                    )
                elif self.position.side == "SHORT" and current_price <= self.position.take_profit:
                    self._losing_streak = 0
                    self.close_position(current_price, "take_profit")
                    return StrategySignal(
                        signal_type=SignalType.CLOSE_SHORT,
                        symbol=self.symbol, price=current_price,
                        confidence=0.85, reason="ML take profit",
                    )

            return None

        # Check if we should trade
        if not self._should_trade():
            return None

        # Get ensemble prediction
        prediction = self._ensemble_predictions()
        if prediction is None:
            return None

        # Confidence filter
        if prediction.confidence < self.confidence_threshold:
            return None

        # Regime-based sizing
        size_mult = self._get_size_multiplier()
        adjusted_risk = self._adjust_risk_for_accuracy()

        if prediction.direction == 1:
            sl = current_price - self.atr_stop_mult * atr_val
            tp = current_price + self.atr_stop_mult * atr_val * 2.0
            if prediction.predicted_return > 0:
                tp = current_price * (1.0 + prediction.predicted_return / 100.0)
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, adjusted_risk,
            ) * size_mult

            signal = StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=prediction.confidence,
                reason=f"ML LONG: {prediction.model_name} conf={prediction.confidence:.2f} "
                       f"regime={self._current_regime}",
                metadata={
                    "model": prediction.model_name,
                    "predicted_return": prediction.predicted_return,
                    "regime": self._current_regime,
                    "size_multiplier": size_mult,
                    "adjusted_risk": adjusted_risk,
                    "models_used": list(self._pending_predictions.keys()),
                },
            )
            if self.check_risk(signal):
                self._entry_prediction = prediction
                self.open_position("LONG", current_price, qty, sl, tp)
                return signal

        elif prediction.direction == -1:
            sl = current_price + self.atr_stop_mult * atr_val
            tp = current_price - self.atr_stop_mult * atr_val * 2.0
            if prediction.predicted_return > 0:
                tp = current_price * (1.0 - prediction.predicted_return / 100.0)
            qty = self.calculate_position_size(
                self.params.get("account_balance", 10000.0),
                current_price, sl, adjusted_risk,
            ) * size_mult

            signal = StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=current_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                confidence=prediction.confidence,
                reason=f"ML SHORT: {prediction.model_name} conf={prediction.confidence:.2f} "
                       f"regime={self._current_regime}",
                metadata={
                    "model": prediction.model_name,
                    "predicted_return": prediction.predicted_return,
                    "regime": self._current_regime,
                    "size_multiplier": size_mult,
                    "adjusted_risk": adjusted_risk,
                    "models_used": list(self._pending_predictions.keys()),
                },
            )
            if self.check_risk(signal):
                self._entry_prediction = prediction
                self.open_position("SHORT", current_price, qty, sl, tp)
                return signal

        return None

    def on_tick(self, price: float, volume: float, timestamp: float) -> Optional[StrategySignal]:
        if self.has_position:
            self.update_position_price(price)
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        if self.has_position:
            close = ohlcv.get("close", 0.0)
            if close > 0:
                self.update_position_price(close)
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        self._logger.info("ML fill: %s", order_info)

    def get_ml_status(self) -> Dict[str, Any]:
        return {
            "regime": self._current_regime,
            "models": {
                name: {
                    "total_predictions": perf.total_predictions,
                    "accuracy": perf.accuracy,
                    "recent_accuracy": perf.recent_accuracy,
                    "total_return": perf.total_return,
                }
                for name, perf in self._model_performance.items()
            },
            "pending_predictions": {
                name: {
                    "direction": pred.direction,
                    "confidence": pred.confidence,
                    "age_sec": time.time() - pred.timestamp,
                }
                for name, pred in self._pending_predictions.items()
            },
            "losing_streak": self._losing_streak,
            "should_trade": self._should_trade(),
        }
