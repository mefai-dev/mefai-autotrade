"""
Market Data Manager - Real-time market data aggregation for Mefai Autotrade.
Provides consolidated orderbook views, best bid/ask across venues,
tick-by-tick storage, time and sales, data quality monitoring,
and liquidity analysis.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("mefai.autotrade.market_data")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TickData:
    """Single trade tick from an exchange."""
    symbol: str = ""
    exchange: str = ""
    price: float = 0.0
    quantity: float = 0.0
    timestamp: float = 0.0
    is_buyer_maker: bool = False
    trade_id: str = ""

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    @property
    def side(self) -> str:
        return "SELL" if self.is_buyer_maker else "BUY"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp,
            "is_buyer_maker": self.is_buyer_maker,
            "trade_id": self.trade_id,
        }


@dataclass
class OrderBookLevel:
    """Single price level in an orderbook."""
    price: float = 0.0
    quantity: float = 0.0
    order_count: int = 0
    exchange: str = ""

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class OrderBookSnapshot:
    """Snapshot of an orderbook at a point in time."""
    symbol: str = ""
    exchange: str = ""
    timestamp: float = 0.0
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    sequence: int = 0

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def best_bid_price(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask_price(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid_price + self.best_ask_price) / 2.0
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.best_ask_price - self.best_bid_price
        return 0.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 100.0
        return 0.0

    @property
    def bid_depth(self) -> float:
        """Total bid-side quantity."""
        return sum(level.quantity for level in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total ask-side quantity."""
        return sum(level.quantity for level in self.asks)

    @property
    def imbalance(self) -> float:
        """Order book imbalance: positive = more bids, negative = more asks."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total

    def depth_at_price(self, price: float, side: str = "bid") -> float:
        """Get cumulative depth up to a price level."""
        total = 0.0
        levels = self.bids if side == "bid" else self.asks
        for level in levels:
            if side == "bid" and level.price >= price:
                total += level.quantity
            elif side == "ask" and level.price <= price:
                total += level.quantity
        return total

    def volume_weighted_price(self, quantity: float, side: str = "buy") -> float:
        """
        Estimate fill price for a given quantity using orderbook depth.
        Side = "buy" walks up asks, "sell" walks down bids.
        """
        levels = self.asks if side == "buy" else self.bids
        remaining = quantity
        total_cost = 0.0

        for level in levels:
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            remaining -= fill_qty
            if remaining <= 0:
                break

        filled = quantity - max(0, remaining)
        if filled > 0:
            return total_cost / filled
        return 0.0


@dataclass
class TimeAndSalesEntry:
    """Single entry in time and sales log."""
    timestamp: float = 0.0
    symbol: str = ""
    price: float = 0.0
    quantity: float = 0.0
    side: str = ""
    exchange: str = ""
    is_block_trade: bool = False

    @property
    def notional(self) -> float:
        return self.price * self.quantity


class TimeAndSales:
    """
    Time and sales log - recent trade history for a symbol.
    Thread-safe with configurable max size.
    """

    def __init__(self, max_entries: int = 10000):
        self._entries: Deque[TimeAndSalesEntry] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._total_buy_volume = 0.0
        self._total_sell_volume = 0.0
        self._total_buy_notional = 0.0
        self._total_sell_notional = 0.0
        self._trade_count = 0
        self._last_price = 0.0

    def add(self, entry: TimeAndSalesEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            self._trade_count += 1
            self._last_price = entry.price

            if entry.side == "BUY":
                self._total_buy_volume += entry.quantity
                self._total_buy_notional += entry.notional
            else:
                self._total_sell_volume += entry.quantity
                self._total_sell_notional += entry.notional

    def add_tick(self, tick: TickData) -> None:
        entry = TimeAndSalesEntry(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            price=tick.price,
            quantity=tick.quantity,
            side=tick.side,
            exchange=tick.exchange,
            is_block_trade=tick.notional > 100000,  # 100k USD threshold
        )
        self.add(entry)

    def get_recent(self, n: int = 100) -> List[TimeAndSalesEntry]:
        with self._lock:
            entries = list(self._entries)
        return entries[-n:]

    def get_block_trades(self, min_notional: float = 100000) -> List[TimeAndSalesEntry]:
        with self._lock:
            return [e for e in self._entries if e.notional >= min_notional]

    def get_volume_profile(self, period_seconds: float = 300) -> Dict[str, float]:
        """Get buy/sell volume breakdown for the last N seconds."""
        cutoff = time.time() - period_seconds
        buy_vol = 0.0
        sell_vol = 0.0
        buy_count = 0
        sell_count = 0

        with self._lock:
            for entry in reversed(list(self._entries)):
                if entry.timestamp < cutoff:
                    break
                if entry.side == "BUY":
                    buy_vol += entry.quantity
                    buy_count += 1
                else:
                    sell_vol += entry.quantity
                    sell_count += 1

        total_vol = buy_vol + sell_vol
        return {
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "total_volume": total_vol,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_ratio": buy_vol / total_vol if total_vol > 0 else 0.5,
            "period_seconds": period_seconds,
        }

    def get_vwap(self, period_seconds: float = 300) -> float:
        """Volume-weighted average price for the last N seconds."""
        cutoff = time.time() - period_seconds
        total_notional = 0.0
        total_volume = 0.0

        with self._lock:
            for entry in reversed(list(self._entries)):
                if entry.timestamp < cutoff:
                    break
                total_notional += entry.notional
                total_volume += entry.quantity

        return total_notional / total_volume if total_volume > 0 else self._last_price

    @property
    def last_price(self) -> float:
        return self._last_price

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_buy_volume = 0.0
            self._total_sell_volume = 0.0
            self._total_buy_notional = 0.0
            self._total_sell_notional = 0.0
            self._trade_count = 0


# ---------------------------------------------------------------------------
# Data quality monitoring
# ---------------------------------------------------------------------------

class DataQualityMonitor:
    """
    Monitors the quality and freshness of incoming market data.
    Detects stale data, price anomalies, and connectivity issues.
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 10.0,
        price_deviation_pct: float = 5.0,
        min_tick_rate: float = 0.1,
    ):
        self._stale_threshold = stale_threshold_seconds
        self._price_deviation_pct = price_deviation_pct
        self._min_tick_rate = min_tick_rate

        self._last_tick_time: Dict[str, float] = {}
        self._last_prices: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self._tick_counts: Dict[str, int] = defaultdict(int)
        self._tick_windows: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))
        self._anomaly_counts: Dict[str, int] = defaultdict(int)
        self._quality_scores: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable) -> None:
        """Register a callback for data quality alerts."""
        self._callbacks.append(callback)

    def _fire_alert(self, alert_type: str, symbol: str, details: Dict[str, Any]) -> None:
        for cb in self._callbacks:
            try:
                cb(alert_type, symbol, details)
            except Exception as exc:
                logger.error("Quality alert callback error: %s", exc)

    def on_tick(self, tick: TickData) -> Dict[str, Any]:
        """
        Process an incoming tick for quality monitoring.
        Returns a dict with quality assessment.
        """
        now = time.time()
        key = f"{tick.symbol}:{tick.exchange}"
        issues: List[str] = []

        with self._lock:
            # Check for stale data
            last_time = self._last_tick_time.get(key, 0)
            if last_time > 0:
                gap = now - last_time
                if gap > self._stale_threshold:
                    issues.append(f"stale_data:gap={gap:.1f}s")
                    self._fire_alert("STALE_DATA", tick.symbol, {
                        "exchange": tick.exchange,
                        "gap_seconds": gap,
                    })

            self._last_tick_time[key] = now

            # Price sanity check
            recent_prices = self._last_prices[key]
            if len(recent_prices) > 10:
                median_price = float(np.median(list(recent_prices)))
                if median_price > 0:
                    deviation = abs(tick.price - median_price) / median_price * 100
                    if deviation > self._price_deviation_pct:
                        issues.append(f"price_anomaly:dev={deviation:.2f}%")
                        self._anomaly_counts[key] += 1
                        self._fire_alert("PRICE_ANOMALY", tick.symbol, {
                            "exchange": tick.exchange,
                            "price": tick.price,
                            "median": median_price,
                            "deviation_pct": deviation,
                        })

            recent_prices.append(tick.price)

            # Tick rate monitoring
            self._tick_counts[key] += 1
            self._tick_windows[key].append(now)

            # Compute quality score (0-100)
            score = self._compute_quality_score(key)
            self._quality_scores[key] = score

        return {
            "symbol": tick.symbol,
            "exchange": tick.exchange,
            "quality_score": score,
            "issues": issues,
            "tick_count": self._tick_counts[key],
        }

    def _compute_quality_score(self, key: str) -> float:
        """Compute a 0-100 quality score for a data feed."""
        score = 100.0

        # Penalize for staleness
        last_time = self._last_tick_time.get(key, 0)
        if last_time > 0:
            age = time.time() - last_time
            if age > self._stale_threshold:
                score -= min(30, age * 3)

        # Penalize for anomalies
        anomalies = self._anomaly_counts.get(key, 0)
        if anomalies > 0:
            score -= min(20, anomalies * 2)

        # Penalize for low tick rate
        window = self._tick_windows.get(key)
        if window and len(window) > 1:
            duration = window[-1] - window[0]
            if duration > 0:
                rate = len(window) / duration
                if rate < self._min_tick_rate:
                    score -= min(20, (self._min_tick_rate - rate) * 100)

        return max(0.0, min(100.0, score))

    def is_stale(self, symbol: str, exchange: str = "") -> bool:
        """Check if data for a symbol is stale."""
        key = f"{symbol}:{exchange}" if exchange else symbol
        with self._lock:
            # Try exact match first
            if key in self._last_tick_time:
                return (time.time() - self._last_tick_time[key]) > self._stale_threshold
            # Try symbol-only match
            for k, t in self._last_tick_time.items():
                if k.startswith(f"{symbol}:"):
                    return (time.time() - t) > self._stale_threshold
        return True

    def get_quality_score(self, symbol: str, exchange: str = "") -> float:
        key = f"{symbol}:{exchange}" if exchange else symbol
        with self._lock:
            if key in self._quality_scores:
                return self._quality_scores[key]
            for k, score in self._quality_scores.items():
                if k.startswith(f"{symbol}:"):
                    return score
        return 0.0

    def get_all_scores(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._quality_scores)

    def get_feed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data feeds."""
        now = time.time()
        status: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            for key, last_time in self._last_tick_time.items():
                status[key] = {
                    "last_tick_age": now - last_time,
                    "is_stale": (now - last_time) > self._stale_threshold,
                    "tick_count": self._tick_counts.get(key, 0),
                    "anomaly_count": self._anomaly_counts.get(key, 0),
                    "quality_score": self._quality_scores.get(key, 0),
                }

        return status

    def reset(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                keys_to_remove = [k for k in self._last_tick_time if k.startswith(f"{symbol}:")]
                for k in keys_to_remove:
                    self._last_tick_time.pop(k, None)
                    self._last_prices.pop(k, None)
                    self._tick_counts.pop(k, None)
                    self._tick_windows.pop(k, None)
                    self._anomaly_counts.pop(k, None)
                    self._quality_scores.pop(k, None)
            else:
                self._last_tick_time.clear()
                self._last_prices.clear()
                self._tick_counts.clear()
                self._tick_windows.clear()
                self._anomaly_counts.clear()
                self._quality_scores.clear()


# ---------------------------------------------------------------------------
# Liquidity analysis
# ---------------------------------------------------------------------------

class LiquidityAnalyzer:
    """Analyze market liquidity from orderbook and trade data."""

    @staticmethod
    def estimate_slippage(
        orderbook: OrderBookSnapshot, quantity: float, side: str = "buy"
    ) -> Dict[str, float]:
        """
        Estimate slippage for a given order size.
        Returns mid price, expected fill price, slippage in absolute and pct.
        """
        mid = orderbook.mid_price
        if mid == 0:
            return {"mid_price": 0, "fill_price": 0, "slippage": 0, "slippage_pct": 0}

        fill_price = orderbook.volume_weighted_price(quantity, side)
        slippage = abs(fill_price - mid)
        slippage_pct = (slippage / mid) * 100 if mid > 0 else 0

        return {
            "mid_price": mid,
            "fill_price": fill_price,
            "slippage": slippage,
            "slippage_pct": slippage_pct,
            "quantity": quantity,
            "side": side,
        }

    @staticmethod
    def compute_market_depth(
        orderbook: OrderBookSnapshot, levels: int = 10
    ) -> Dict[str, Any]:
        """Compute market depth metrics."""
        bid_levels = orderbook.bids[:levels]
        ask_levels = orderbook.asks[:levels]

        bid_depth = sum(l.quantity for l in bid_levels)
        ask_depth = sum(l.quantity for l in ask_levels)
        bid_notional = sum(l.notional for l in bid_levels)
        ask_notional = sum(l.notional for l in ask_levels)

        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bid_notional": bid_notional,
            "ask_notional": ask_notional,
            "total_depth": total_depth,
            "imbalance": imbalance,
            "spread": orderbook.spread,
            "spread_pct": orderbook.spread_pct,
            "levels_analyzed": levels,
        }

    @staticmethod
    def detect_walls(
        orderbook: OrderBookSnapshot, multiplier: float = 3.0
    ) -> Dict[str, List[OrderBookLevel]]:
        """
        Detect large orders (walls) in the orderbook.
        A wall is a level with quantity > multiplier * average level quantity.
        """
        bid_walls: List[OrderBookLevel] = []
        ask_walls: List[OrderBookLevel] = []

        if orderbook.bids:
            avg_bid = sum(l.quantity for l in orderbook.bids) / len(orderbook.bids)
            threshold = avg_bid * multiplier
            bid_walls = [l for l in orderbook.bids if l.quantity >= threshold]

        if orderbook.asks:
            avg_ask = sum(l.quantity for l in orderbook.asks) / len(orderbook.asks)
            threshold = avg_ask * multiplier
            ask_walls = [l for l in orderbook.asks if l.quantity >= threshold]

        return {"bid_walls": bid_walls, "ask_walls": ask_walls}

    @staticmethod
    def compute_liquidity_score(
        orderbook: OrderBookSnapshot,
        recent_volume: float = 0.0,
        spread_weight: float = 0.4,
        depth_weight: float = 0.3,
        volume_weight: float = 0.3,
    ) -> float:
        """
        Compute a 0-100 liquidity score.
        Higher = more liquid market.
        """
        score = 0.0

        # Spread component (lower spread = higher score)
        spread_pct = orderbook.spread_pct
        if spread_pct > 0:
            spread_score = max(0, 100 - spread_pct * 1000)
        else:
            spread_score = 100.0
        score += spread_score * spread_weight

        # Depth component
        total_depth = orderbook.bid_depth + orderbook.ask_depth
        if total_depth > 0:
            depth_score = min(100, total_depth * 10)  # Normalized
        else:
            depth_score = 0.0
        score += depth_score * depth_weight

        # Volume component
        if recent_volume > 0:
            volume_score = min(100, recent_volume / 100)
        else:
            volume_score = 0.0
        score += volume_score * volume_weight

        return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Main market data manager
# ---------------------------------------------------------------------------

class MarketDataManager:
    """
    Central market data manager.
    Aggregates data from multiple exchanges, maintains orderbooks,
    provides consolidated views, and monitors data quality.
    """

    def __init__(
        self,
        max_tick_history: int = 50000,
        stale_threshold: float = 10.0,
        block_trade_threshold: float = 100000.0,
    ):
        self._orderbooks: Dict[str, Dict[str, OrderBookSnapshot]] = defaultdict(dict)
        self._time_and_sales: Dict[str, TimeAndSales] = {}
        self._quality_monitor = DataQualityMonitor(stale_threshold)
        self._liquidity_analyzer = LiquidityAnalyzer()

        self._max_tick_history = max_tick_history
        self._block_trade_threshold = block_trade_threshold

        self._tick_callbacks: List[Callable] = []
        self._orderbook_callbacks: List[Callable] = []
        self._lock = threading.Lock()

        self._symbols: Set[str] = set()
        self._exchanges: Set[str] = set()
        self._last_prices: Dict[str, float] = {}
        self._total_ticks_processed = 0

        logger.info("MarketDataManager initialized")

    # --- Subscriptions ---

    def on_tick(self, callback: Callable[[TickData], None]) -> None:
        """Register a callback for incoming ticks."""
        self._tick_callbacks.append(callback)

    def on_orderbook_update(self, callback: Callable[[OrderBookSnapshot], None]) -> None:
        """Register a callback for orderbook updates."""
        self._orderbook_callbacks.append(callback)

    # --- Tick processing ---

    def process_tick(self, tick: TickData) -> None:
        """Process an incoming trade tick."""
        self._total_ticks_processed += 1
        self._symbols.add(tick.symbol)
        self._exchanges.add(tick.exchange)
        self._last_prices[tick.symbol] = tick.price

        # Store in time and sales
        tas = self._get_time_and_sales(tick.symbol)
        tas.add_tick(tick)

        # Quality monitoring
        self._quality_monitor.on_tick(tick)

        # Fire callbacks
        for cb in self._tick_callbacks:
            try:
                cb(tick)
            except Exception as exc:
                logger.error("Tick callback error: %s", exc)

    def process_ticks(self, ticks: List[TickData]) -> None:
        """Process multiple ticks (batch)."""
        for tick in ticks:
            self.process_tick(tick)

    # --- Orderbook management ---

    def update_orderbook(self, snapshot: OrderBookSnapshot) -> None:
        """Update the orderbook for a symbol on an exchange."""
        with self._lock:
            self._orderbooks[snapshot.symbol][snapshot.exchange] = snapshot

        self._symbols.add(snapshot.symbol)
        self._exchanges.add(snapshot.exchange)

        for cb in self._orderbook_callbacks:
            try:
                cb(snapshot)
            except Exception as exc:
                logger.error("Orderbook callback error: %s", exc)

    def get_orderbook(
        self, symbol: str, exchange: Optional[str] = None
    ) -> Optional[OrderBookSnapshot]:
        """Get orderbook for a symbol. If no exchange specified, returns first available."""
        with self._lock:
            books = self._orderbooks.get(symbol, {})
            if exchange:
                return books.get(exchange)
            if books:
                return next(iter(books.values()))
        return None

    def get_consolidated_orderbook(
        self, symbol: str, depth: int = 20
    ) -> OrderBookSnapshot:
        """
        Build a consolidated orderbook from all exchanges.
        Merges bids and asks, sorted by best price.
        """
        with self._lock:
            books = self._orderbooks.get(symbol, {})

        all_bids: List[OrderBookLevel] = []
        all_asks: List[OrderBookLevel] = []

        for exchange, book in books.items():
            for bid in book.bids:
                level = OrderBookLevel(
                    price=bid.price,
                    quantity=bid.quantity,
                    order_count=bid.order_count,
                    exchange=exchange,
                )
                all_bids.append(level)
            for ask in book.asks:
                level = OrderBookLevel(
                    price=ask.price,
                    quantity=ask.quantity,
                    order_count=ask.order_count,
                    exchange=exchange,
                )
                all_asks.append(level)

        # Sort: bids descending, asks ascending
        all_bids.sort(key=lambda x: x.price, reverse=True)
        all_asks.sort(key=lambda x: x.price)

        # Merge same-price levels
        merged_bids = self._merge_levels(all_bids)[:depth]
        merged_asks = self._merge_levels(all_asks)[:depth]

        return OrderBookSnapshot(
            symbol=symbol,
            exchange="CONSOLIDATED",
            timestamp=time.time(),
            bids=merged_bids,
            asks=merged_asks,
        )

    @staticmethod
    def _merge_levels(levels: List[OrderBookLevel]) -> List[OrderBookLevel]:
        """Merge levels at the same price."""
        if not levels:
            return []

        merged: List[OrderBookLevel] = []
        current = levels[0]

        for level in levels[1:]:
            if abs(level.price - current.price) < 1e-10:
                current = OrderBookLevel(
                    price=current.price,
                    quantity=current.quantity + level.quantity,
                    order_count=current.order_count + level.order_count,
                    exchange="MERGED",
                )
            else:
                merged.append(current)
                current = level

        merged.append(current)
        return merged

    # --- Best bid/ask ---

    def get_best_bid(self, symbol: str) -> Optional[OrderBookLevel]:
        """Get the best bid across all exchanges."""
        best: Optional[OrderBookLevel] = None
        with self._lock:
            for exchange, book in self._orderbooks.get(symbol, {}).items():
                if book.bids:
                    bid = book.bids[0]
                    if best is None or bid.price > best.price:
                        best = OrderBookLevel(
                            price=bid.price,
                            quantity=bid.quantity,
                            exchange=exchange,
                        )
        return best

    def get_best_ask(self, symbol: str) -> Optional[OrderBookLevel]:
        """Get the best ask across all exchanges."""
        best: Optional[OrderBookLevel] = None
        with self._lock:
            for exchange, book in self._orderbooks.get(symbol, {}).items():
                if book.asks:
                    ask = book.asks[0]
                    if best is None or ask.price < best.price:
                        best = OrderBookLevel(
                            price=ask.price,
                            quantity=ask.quantity,
                            exchange=exchange,
                        )
        return best

    def get_bbo(self, symbol: str) -> Dict[str, Any]:
        """Get best bid and offer (BBO) across all exchanges."""
        best_bid = self.get_best_bid(symbol)
        best_ask = self.get_best_ask(symbol)

        mid = 0.0
        spread = 0.0
        if best_bid and best_ask:
            mid = (best_bid.price + best_ask.price) / 2.0
            spread = best_ask.price - best_bid.price

        return {
            "symbol": symbol,
            "best_bid": best_bid.price if best_bid else 0.0,
            "best_bid_qty": best_bid.quantity if best_bid else 0.0,
            "best_bid_exchange": best_bid.exchange if best_bid else "",
            "best_ask": best_ask.price if best_ask else 0.0,
            "best_ask_qty": best_ask.quantity if best_ask else 0.0,
            "best_ask_exchange": best_ask.exchange if best_ask else "",
            "mid_price": mid,
            "spread": spread,
            "spread_pct": (spread / mid * 100) if mid > 0 else 0.0,
        }

    def get_vwap_mid(self, symbol: str, depth_qty: float = 100.0) -> float:
        """Volume-weighted mid price using orderbook depth."""
        book = self.get_consolidated_orderbook(symbol)
        bid_vwap = book.volume_weighted_price(depth_qty, "sell")
        ask_vwap = book.volume_weighted_price(depth_qty, "buy")
        if bid_vwap > 0 and ask_vwap > 0:
            return (bid_vwap + ask_vwap) / 2.0
        return book.mid_price

    # --- Time and sales ---

    def _get_time_and_sales(self, symbol: str) -> TimeAndSales:
        if symbol not in self._time_and_sales:
            self._time_and_sales[symbol] = TimeAndSales(self._max_tick_history)
        return self._time_and_sales[symbol]

    def get_time_and_sales(self, symbol: str) -> TimeAndSales:
        return self._get_time_and_sales(symbol)

    def get_recent_trades(self, symbol: str, n: int = 100) -> List[TimeAndSalesEntry]:
        tas = self._time_and_sales.get(symbol)
        if tas:
            return tas.get_recent(n)
        return []

    def get_block_trades(self, symbol: str) -> List[TimeAndSalesEntry]:
        tas = self._time_and_sales.get(symbol)
        if tas:
            return tas.get_block_trades(self._block_trade_threshold)
        return []

    # --- Market data snapshot ---

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get a complete market data snapshot for a symbol."""
        bbo = self.get_bbo(symbol)
        tas = self._time_and_sales.get(symbol)

        snapshot: Dict[str, Any] = {
            "symbol": symbol,
            "timestamp": time.time(),
            **bbo,
        }

        if tas:
            snapshot["last_price"] = tas.last_price
            snapshot["trade_count"] = tas.trade_count
            snapshot["volume_profile"] = tas.get_volume_profile(300)
            snapshot["vwap_5m"] = tas.get_vwap(300)
        else:
            snapshot["last_price"] = self._last_prices.get(symbol, 0)
            snapshot["trade_count"] = 0

        # Quality
        snapshot["quality_score"] = self._quality_monitor.get_quality_score(symbol)
        snapshot["is_stale"] = self._quality_monitor.is_stale(symbol)

        return snapshot

    def get_snapshot_for_strategy(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data snapshots for multiple symbols (for strategy consumption)."""
        return {symbol: self.get_snapshot(symbol) for symbol in symbols}

    # --- Liquidity ---

    def get_slippage_estimate(
        self, symbol: str, quantity: float, side: str = "buy"
    ) -> Dict[str, float]:
        book = self.get_consolidated_orderbook(symbol)
        return self._liquidity_analyzer.estimate_slippage(book, quantity, side)

    def get_liquidity_score(self, symbol: str) -> float:
        book = self.get_consolidated_orderbook(symbol)
        tas = self._time_and_sales.get(symbol)
        recent_vol = 0.0
        if tas:
            profile = tas.get_volume_profile(300)
            recent_vol = profile["total_volume"]
        return self._liquidity_analyzer.compute_liquidity_score(book, recent_vol)

    # --- Quality ---

    @property
    def quality_monitor(self) -> DataQualityMonitor:
        return self._quality_monitor

    # --- Status ---

    def get_last_price(self, symbol: str) -> float:
        return self._last_prices.get(symbol, 0.0)

    def get_active_symbols(self) -> Set[str]:
        return set(self._symbols)

    def get_active_exchanges(self) -> Set[str]:
        return set(self._exchanges)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols_tracked": len(self._symbols),
            "exchanges_connected": len(self._exchanges),
            "total_ticks_processed": self._total_ticks_processed,
            "orderbooks_maintained": sum(
                len(books) for books in self._orderbooks.values()
            ),
            "time_and_sales_entries": sum(
                tas.entry_count for tas in self._time_and_sales.values()
            ),
            "feed_status": self._quality_monitor.get_feed_status(),
        }

    def clear(self, symbol: Optional[str] = None) -> None:
        """Clear all data or data for a specific symbol."""
        with self._lock:
            if symbol:
                self._orderbooks.pop(symbol, None)
                self._time_and_sales.pop(symbol, None)
                self._last_prices.pop(symbol, None)
                self._symbols.discard(symbol)
                self._quality_monitor.reset(symbol)
            else:
                self._orderbooks.clear()
                self._time_and_sales.clear()
                self._last_prices.clear()
                self._symbols.clear()
                self._exchanges.clear()
                self._quality_monitor.reset()
                self._total_ticks_processed = 0
