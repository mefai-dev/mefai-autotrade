# mefai-autotrade - Data normalizer for multi-exchange support
# Converts exchange-specific data formats into unified models

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.exchanges.base import (
    UnifiedTicker,
    UnifiedOrderbook,
    UnifiedKline,
    UnifiedOrder,
    UnifiedPosition,
    UnifiedBalance,
    UnifiedTrade,
    OrderbookLevel,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    MarginMode,
    TimeInForce,
    current_timestamp_ms,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol mapping tables
# ---------------------------------------------------------------------------

# Unified format: BASE/QUOTE (e.g. BTC/USDT)
# Exchange formats:
#   Binance: BTCUSDT
#   Bybit:   BTCUSDT
#   OKX:     BTC-USDT (spot), BTC-USDT-SWAP (swap)

COMMON_QUOTE_ASSETS = ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB", "EUR", "GBP", "TRY", "DAI", "TUSD", "FDUSD"]

# Known base assets that end with common quote suffixes (disambiguation)
AMBIGUOUS_BASES = {
    "ETHBTC": ("ETH", "BTC"),
    "BNBBTC": ("BNB", "BTC"),
    "BNBETH": ("BNB", "ETH"),
    "USDCUSDT": ("USDC", "USDT"),
    "BUSDUSDT": ("BUSD", "USDT"),
    "TUSDUSDT": ("TUSD", "USDT"),
    "FDUSDUSDT": ("FDUSD", "USDT"),
    "DAIUSDT": ("DAI", "USDT"),
    "EURUSDT": ("EUR", "USDT"),
}


class SymbolMapper:
    """Maps symbols between exchange-specific and unified formats.

    Handles the different symbol conventions across exchanges:
    - Binance Spot/Futures: BTCUSDT
    - OKX: BTC-USDT, BTC-USDT-SWAP
    - Bybit: BTCUSDT (same as Binance)
    - Unified: BTC/USDT
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, str], str] = {}  # (symbol, exchange) -> unified
        self._reverse_cache: Dict[Tuple[str, str], str] = {}  # (unified, exchange) -> exchange_symbol
        self._custom_mappings: Dict[str, Dict[str, str]] = {}  # exchange -> {exchange_sym: unified}

    def add_custom_mapping(self, exchange: str, exchange_symbol: str, unified_symbol: str) -> None:
        """Add a custom symbol mapping for an exchange."""
        if exchange not in self._custom_mappings:
            self._custom_mappings[exchange] = {}
        self._custom_mappings[exchange][exchange_symbol] = unified_symbol

    def to_unified(self, symbol: str, exchange: str) -> str:
        """Convert an exchange-specific symbol to unified format (BASE/QUOTE)."""
        cache_key = (symbol, exchange)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check custom mappings first
        custom = self._custom_mappings.get(exchange, {})
        if symbol in custom:
            result = custom[symbol]
            self._cache[cache_key] = result
            return result

        result = self._convert_to_unified(symbol, exchange)
        self._cache[cache_key] = result
        return result

    def to_exchange(self, unified_symbol: str, exchange: str) -> str:
        """Convert a unified symbol (BTC/USDT) to exchange-specific format."""
        cache_key = (unified_symbol, exchange)
        if cache_key in self._reverse_cache:
            return self._reverse_cache[cache_key]

        result = self._convert_to_exchange(unified_symbol, exchange)
        self._reverse_cache[cache_key] = result
        return result

    def _convert_to_unified(self, symbol: str, exchange: str) -> str:
        """Internal conversion to unified format."""
        # Already in unified format
        if "/" in symbol:
            return symbol

        # OKX format: BTC-USDT or BTC-USDT-SWAP
        if "-" in symbol:
            parts = symbol.split("-")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"

        # Binance/Bybit format: BTCUSDT
        # Check ambiguous pairs first
        if symbol in AMBIGUOUS_BASES:
            base, quote = AMBIGUOUS_BASES[symbol]
            return f"{base}/{quote}"

        # Try to split by known quote assets (longest match first)
        for quote in sorted(COMMON_QUOTE_ASSETS, key=len, reverse=True):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[:-len(quote)]
                if base:
                    return f"{base}/{quote}"

        # Fallback: return as-is
        return symbol

    def _convert_to_exchange(self, unified: str, exchange: str) -> str:
        """Internal conversion to exchange format."""
        if "/" not in unified:
            return unified

        base, quote = unified.split("/", 1)

        if exchange in ("binance_spot", "binance_futures", "bybit"):
            return f"{base}{quote}"
        elif exchange == "okx":
            return f"{base}-{quote}"

        # Default: concatenate
        return f"{base}{quote}"

    def clear_cache(self) -> None:
        """Clear the symbol mapping cache."""
        self._cache.clear()
        self._reverse_cache.clear()


class DataNormalizer:
    """Normalizes market data across different exchanges.

    Provides utility methods to convert exchange-specific data formats
    into the unified models defined in base.py. Used by the connection
    pool and strategy engine to ensure exchange-agnostic data handling.
    """

    def __init__(self):
        self.symbol_mapper = SymbolMapper()
        self._precision_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Symbol normalization
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str, exchange: str) -> str:
        """Normalize a symbol to unified format."""
        return self.symbol_mapper.to_unified(symbol, exchange)

    def denormalize_symbol(self, unified_symbol: str, exchange: str) -> str:
        """Convert unified symbol back to exchange-specific format."""
        return self.symbol_mapper.to_exchange(unified_symbol, exchange)

    # ------------------------------------------------------------------
    # Ticker normalization
    # ------------------------------------------------------------------

    def normalize_ticker(self, ticker: UnifiedTicker, target_symbol: Optional[str] = None) -> UnifiedTicker:
        """Normalize a ticker, optionally remapping the symbol."""
        if target_symbol:
            ticker.symbol = target_symbol
        elif "/" not in ticker.symbol:
            ticker.symbol = self.normalize_symbol(ticker.symbol, ticker.exchange)
        return ticker

    def merge_tickers(self, tickers: List[UnifiedTicker]) -> Dict[str, List[UnifiedTicker]]:
        """Group tickers by unified symbol, useful for cross-exchange comparison."""
        result: Dict[str, List[UnifiedTicker]] = {}
        for t in tickers:
            unified = self.normalize_symbol(t.symbol, t.exchange)
            if unified not in result:
                result[unified] = []
            result[unified].append(t)
        return result

    def best_ticker(self, tickers: List[UnifiedTicker]) -> Optional[UnifiedTicker]:
        """Given tickers from multiple exchanges for the same symbol,
        return the one with the best (tightest) spread.
        """
        if not tickers:
            return None

        best = None
        best_spread = float("inf")
        for t in tickers:
            if t.ask_price > 0 and t.bid_price > 0:
                spread = t.ask_price - t.bid_price
                if spread < best_spread:
                    best_spread = spread
                    best = t
        return best or tickers[0]

    # ------------------------------------------------------------------
    # Orderbook normalization
    # ------------------------------------------------------------------

    def normalize_orderbook(
        self,
        orderbook: UnifiedOrderbook,
        target_symbol: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> UnifiedOrderbook:
        """Normalize an orderbook, optionally truncating depth."""
        if target_symbol:
            orderbook.symbol = target_symbol
        elif "/" not in orderbook.symbol:
            orderbook.symbol = self.normalize_symbol(orderbook.symbol, orderbook.exchange)

        if max_depth:
            orderbook.bids = orderbook.bids[:max_depth]
            orderbook.asks = orderbook.asks[:max_depth]

        # Ensure correct sort order
        orderbook.bids.sort(key=lambda l: l.price, reverse=True)
        orderbook.asks.sort(key=lambda l: l.price)

        return orderbook

    def merge_orderbooks(
        self,
        orderbooks: List[UnifiedOrderbook],
        max_depth: int = 50,
    ) -> UnifiedOrderbook:
        """Merge orderbooks from multiple exchanges into a combined view.

        Bids and asks are merged and sorted by price. This creates a
        virtual "super orderbook" showing liquidity across exchanges.
        """
        if not orderbooks:
            raise ValueError("Cannot merge empty orderbook list")

        all_bids: List[Tuple[float, float, str]] = []  # (price, qty, exchange)
        all_asks: List[Tuple[float, float, str]] = []

        for book in orderbooks:
            for level in book.bids:
                all_bids.append((level.price, level.quantity, book.exchange))
            for level in book.asks:
                all_asks.append((level.price, level.quantity, book.exchange))

        # Sort and merge same-price levels
        all_bids.sort(key=lambda x: x[0], reverse=True)
        all_asks.sort(key=lambda x: x[0])

        merged_bids = self._merge_levels(all_bids, max_depth)
        merged_asks = self._merge_levels(all_asks, max_depth)

        return UnifiedOrderbook(
            symbol=orderbooks[0].symbol,
            exchange="merged",
            bids=merged_bids,
            asks=merged_asks,
            timestamp=max(b.timestamp for b in orderbooks),
        )

    @staticmethod
    def _merge_levels(
        levels: List[Tuple[float, float, str]],
        max_depth: int,
    ) -> List[OrderbookLevel]:
        """Merge price levels, combining quantities at the same price."""
        merged: Dict[float, float] = {}
        for price, qty, _ in levels:
            merged[price] = merged.get(price, 0) + qty

        result = [
            OrderbookLevel(price=p, quantity=q)
            for p, q in sorted(merged.items(), reverse=True)
        ]
        return result[:max_depth]

    # ------------------------------------------------------------------
    # Kline normalization
    # ------------------------------------------------------------------

    def normalize_kline(self, kline: UnifiedKline, target_symbol: Optional[str] = None) -> UnifiedKline:
        """Normalize a kline entry."""
        if target_symbol:
            kline.symbol = target_symbol
        elif "/" not in kline.symbol:
            kline.symbol = self.normalize_symbol(kline.symbol, kline.exchange)
        return kline

    def align_klines(
        self,
        klines_by_exchange: Dict[str, List[UnifiedKline]],
    ) -> Dict[int, Dict[str, UnifiedKline]]:
        """Align klines from multiple exchanges by open_time.

        Returns a dict mapping open_time -> {exchange_name: kline}.
        Useful for cross-exchange OHLCV comparison.
        """
        aligned: Dict[int, Dict[str, UnifiedKline]] = {}
        for exchange_name, klines in klines_by_exchange.items():
            for k in klines:
                if k.open_time not in aligned:
                    aligned[k.open_time] = {}
                aligned[k.open_time][exchange_name] = k
        return dict(sorted(aligned.items()))

    # ------------------------------------------------------------------
    # Order normalization
    # ------------------------------------------------------------------

    def normalize_order(self, order: UnifiedOrder, target_symbol: Optional[str] = None) -> UnifiedOrder:
        """Normalize an order."""
        if target_symbol:
            order.symbol = target_symbol
        elif "/" not in order.symbol and "-" not in order.symbol:
            order.symbol = self.normalize_symbol(order.symbol, order.exchange)
        return order

    # ------------------------------------------------------------------
    # Position normalization
    # ------------------------------------------------------------------

    def normalize_position(self, position: UnifiedPosition, target_symbol: Optional[str] = None) -> UnifiedPosition:
        """Normalize a position."""
        if target_symbol:
            position.symbol = target_symbol
        elif "/" not in position.symbol and "-" not in position.symbol:
            position.symbol = self.normalize_symbol(position.symbol, position.exchange)
        return position

    def aggregate_positions(
        self,
        positions: List[UnifiedPosition],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate positions across exchanges by unified symbol.

        Returns a dict mapping unified_symbol to aggregated data:
        {
            "BTC/USDT": {
                "total_long": 0.5,
                "total_short": 0.2,
                "net_size": 0.3,
                "total_unrealized_pnl": 123.45,
                "positions": [...],
            }
        }
        """
        result: Dict[str, Dict[str, Any]] = {}

        for pos in positions:
            unified = self.normalize_symbol(pos.symbol, pos.exchange)
            if unified not in result:
                result[unified] = {
                    "total_long": 0.0,
                    "total_short": 0.0,
                    "net_size": 0.0,
                    "total_unrealized_pnl": 0.0,
                    "total_notional": 0.0,
                    "exchanges": [],
                    "positions": [],
                }

            entry = result[unified]
            if pos.side == PositionSide.LONG or (pos.side == PositionSide.BOTH and pos.size > 0):
                entry["total_long"] += pos.size
                entry["net_size"] += pos.size
            elif pos.side == PositionSide.SHORT:
                entry["total_short"] += pos.size
                entry["net_size"] -= pos.size

            entry["total_unrealized_pnl"] += pos.unrealized_pnl
            entry["total_notional"] += pos.notional
            if pos.exchange not in entry["exchanges"]:
                entry["exchanges"].append(pos.exchange)
            entry["positions"].append(pos)

        return result

    # ------------------------------------------------------------------
    # Balance normalization
    # ------------------------------------------------------------------

    def normalize_balance(self, balance: UnifiedBalance) -> UnifiedBalance:
        """Normalize a balance entry (no symbol conversion needed)."""
        return balance

    def aggregate_balances(
        self,
        balances: List[UnifiedBalance],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate balances across exchanges by asset.

        Returns:
        {
            "USDT": {
                "total_free": 1000.0,
                "total_locked": 200.0,
                "total": 1200.0,
                "total_usd_value": 1200.0,
                "exchanges": {"binance_spot": {...}, "bybit": {...}},
            }
        }
        """
        result: Dict[str, Dict[str, Any]] = {}
        for b in balances:
            if b.asset not in result:
                result[b.asset] = {
                    "total_free": 0.0,
                    "total_locked": 0.0,
                    "total": 0.0,
                    "total_usd_value": 0.0,
                    "exchanges": {},
                }

            entry = result[b.asset]
            entry["total_free"] += b.free
            entry["total_locked"] += b.locked
            entry["total"] += b.total
            entry["total_usd_value"] += b.usd_value
            entry["exchanges"][b.exchange] = {
                "free": b.free,
                "locked": b.locked,
                "total": b.total,
                "usd_value": b.usd_value,
            }

        return result

    # ------------------------------------------------------------------
    # Trade normalization
    # ------------------------------------------------------------------

    def normalize_trade(self, trade: UnifiedTrade, target_symbol: Optional[str] = None) -> UnifiedTrade:
        """Normalize a trade."""
        if target_symbol:
            trade.symbol = target_symbol
        elif "/" not in trade.symbol:
            trade.symbol = self.normalize_symbol(trade.symbol, trade.exchange)
        return trade

    # ------------------------------------------------------------------
    # Timestamp normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_timestamp(ts: Any) -> int:
        """Convert any timestamp format to UTC milliseconds.

        Handles:
        - Already in milliseconds (13 digits)
        - Seconds (10 digits) -> multiply by 1000
        - String timestamps (ISO 8601) -> parse
        - None -> current time
        """
        if ts is None:
            return current_timestamp_ms()

        if isinstance(ts, str):
            # Try to parse ISO format
            try:
                from datetime import datetime, timezone
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except (ValueError, AttributeError):
                pass
            # Try numeric string
            try:
                ts = int(ts)
            except ValueError:
                return current_timestamp_ms()

        ts = int(ts)

        # 10 digits = seconds, 13 digits = milliseconds, 16 digits = microseconds
        if ts < 1e12:
            return ts * 1000  # Seconds to ms
        elif ts > 1e15:
            return ts // 1000  # Microseconds to ms
        return ts

    # ------------------------------------------------------------------
    # Precision handling
    # ------------------------------------------------------------------

    def set_precision(
        self,
        symbol: str,
        exchange: str,
        price_precision: int,
        quantity_precision: int,
    ) -> None:
        """Cache precision info for a symbol on an exchange."""
        key = (symbol, exchange)
        self._precision_cache[key] = {
            "price": price_precision,
            "quantity": quantity_precision,
        }

    def get_precision(self, symbol: str, exchange: str) -> Optional[Dict[str, int]]:
        """Get cached precision for a symbol."""
        return self._precision_cache.get((symbol, exchange))

    def adjust_price(self, price: float, symbol: str, exchange: str) -> float:
        """Adjust price to the correct precision for the exchange."""
        prec = self._precision_cache.get((symbol, exchange))
        if not prec:
            return price
        p = prec["price"]
        factor = 10 ** p
        return int(price * factor) / factor

    def adjust_quantity(self, quantity: float, symbol: str, exchange: str) -> float:
        """Adjust quantity to the correct precision for the exchange."""
        prec = self._precision_cache.get((symbol, exchange))
        if not prec:
            return quantity
        p = prec["quantity"]
        factor = 10 ** p
        return int(quantity * factor) / factor

    # ------------------------------------------------------------------
    # Cross-exchange price comparison
    # ------------------------------------------------------------------

    def compute_price_spread(
        self,
        tickers: List[UnifiedTicker],
    ) -> Dict[str, Any]:
        """Compute price spread across exchanges for arbitrage detection.

        Returns:
        {
            "min_price": {"exchange": "binance", "price": 60000.0},
            "max_price": {"exchange": "okx", "price": 60010.0},
            "spread": 10.0,
            "spread_pct": 0.017,
            "opportunity": True if spread > threshold
        }
        """
        if len(tickers) < 2:
            return {"spread": 0, "spread_pct": 0, "opportunity": False}

        valid = [t for t in tickers if t.last_price > 0]
        if len(valid) < 2:
            return {"spread": 0, "spread_pct": 0, "opportunity": False}

        min_t = min(valid, key=lambda t: t.last_price)
        max_t = max(valid, key=lambda t: t.last_price)

        spread = max_t.last_price - min_t.last_price
        spread_pct = (spread / min_t.last_price * 100) if min_t.last_price > 0 else 0

        return {
            "min_price": {"exchange": min_t.exchange, "price": min_t.last_price},
            "max_price": {"exchange": max_t.exchange, "price": max_t.last_price},
            "spread": round(spread, 8),
            "spread_pct": round(spread_pct, 4),
            "opportunity": spread_pct > 0.05,  # > 0.05% spread threshold
            "tickers": valid,
        }

    def compute_funding_spread(
        self,
        funding_rates: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare funding rates across exchanges.

        Args:
            funding_rates: {exchange_name: funding_rate_dict}

        Returns spread and best exchange to be long/short.
        """
        if len(funding_rates) < 2:
            return {"spread": 0}

        rates = {}
        for exchange, data in funding_rates.items():
            rate = data.get("funding_rate", data.get("last_funding_rate", 0))
            if rate is not None:
                rates[exchange] = float(rate)

        if len(rates) < 2:
            return {"spread": 0}

        min_exchange = min(rates, key=rates.get)
        max_exchange = max(rates, key=rates.get)
        spread = rates[max_exchange] - rates[min_exchange]

        return {
            "min_rate": {"exchange": min_exchange, "rate": rates[min_exchange]},
            "max_rate": {"exchange": max_exchange, "rate": rates[max_exchange]},
            "spread": round(spread, 8),
            "spread_pct": round(spread * 100, 6),
            "best_long": min_exchange,   # Long where funding is cheapest
            "best_short": max_exchange,  # Short where funding is most expensive
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.symbol_mapper.clear_cache()
        self._precision_cache.clear()
