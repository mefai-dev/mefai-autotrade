# mefai-autotrade - Binance Spot exchange connector
# Full REST + WebSocket implementation for Binance Spot trading

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from src.exchanges.base import (
    ExchangeBase,
    ExchangeConfig,
    ExchangeError,
    AuthenticationError,
    InsufficientBalanceError,
    InvalidOrderError,
    NetworkError,
    OrderNotFoundError,
    RateLimitError,
    UnifiedAccountInfo,
    UnifiedBalance,
    UnifiedKline,
    UnifiedOrder,
    UnifiedOrderbook,
    UnifiedPosition,
    UnifiedTicker,
    UnifiedTrade,
    OrderbookLevel,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    PositionSide,
    MarginMode,
    current_timestamp_ms,
    retry_async,
)

logger = logging.getLogger(__name__)

# Binance Spot base URLs
BINANCE_SPOT_REST = "https://api.binance.com"
BINANCE_SPOT_REST_TESTNET = "https://testnet.binance.vision"
BINANCE_SPOT_WS = "wss://stream.binance.com:9443/ws"
BINANCE_SPOT_WS_TESTNET = "wss://testnet.binance.vision/ws"

# Order type mapping to Binance API values
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
    OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT",
    OrderType.LIMIT_MAKER: "LIMIT_MAKER",
}

# Reverse mapping from Binance to unified
_ORDER_TYPE_REV = {v: k for k, v in _ORDER_TYPE_MAP.items()}
_ORDER_TYPE_REV["STOP_LOSS"] = OrderType.STOP_MARKET
_ORDER_TYPE_REV["TAKE_PROFIT"] = OrderType.TAKE_PROFIT_MARKET

_ORDER_STATUS_MAP = {
    "NEW": OrderStatus.NEW,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
    "PENDING_CANCEL": OrderStatus.CANCELED,
    "EXPIRED_IN_MATCH": OrderStatus.EXPIRED,
}

_TIF_MAP = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.GTX: "GTC",
}


class BinanceSpotExchange(ExchangeBase):
    """Binance Spot exchange connector with full REST and WebSocket support.

    Supports all spot order types, account management, market data, and
    real-time streaming via WebSocket with automatic reconnection.
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__(config, "binance_spot")
        self._base_url = config.base_url_override or (
            BINANCE_SPOT_REST_TESTNET if config.testnet else BINANCE_SPOT_REST
        )
        self._ws_base_url = config.ws_url_override or (
            BINANCE_SPOT_WS_TESTNET if config.testnet else BINANCE_SPOT_WS
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._ws_tasks: Dict[str, asyncio.Task] = {}
        self._listen_key: Optional[str] = None
        self._listen_key_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._sub_counter = 0
        # Exchange info cache - symbol filters, precision, lot sizes
        self._symbol_info: Dict[str, Dict[str, Any]] = {}
        self._server_time_offset: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "X-MBX-APIKEY": self.config.api_key,
                **self.config.custom_headers,
            },
        )
        # Sync server time and load exchange info
        await self._sync_server_time()
        await self._load_exchange_info()
        self._initialized = True
        self._logger.info("Binance Spot connector initialized (testnet=%s)", self.config.testnet)

    async def close(self) -> None:
        # Cancel listen key keepalive
        if self._listen_key_task and not self._listen_key_task.done():
            self._listen_key_task.cancel()
        # Close listen key
        if self._listen_key:
            try:
                await self._close_listen_key()
            except Exception:
                pass
        # Cancel all WS tasks
        for task in self._ws_tasks.values():
            if not task.done():
                task.cancel()
        # Close all WS connections
        for ws in self._ws_connections.values():
            if not ws.closed:
                await ws.close()
        self._ws_connections.clear()
        self._ws_tasks.clear()
        self._subscriptions.clear()
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
        self._initialized = False
        self._logger.info("Binance Spot connector closed")

    # ------------------------------------------------------------------
    # Internal HTTP methods
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Any:
        """Make an HTTP request to the Binance Spot API.

        Args:
            method: HTTP method (GET, POST, DELETE, PUT).
            path: API path (e.g. /api/v3/ticker/price).
            params: Query parameters.
            signed: Whether to sign the request with HMAC-SHA256.
            weight: Rate limit weight for this endpoint.

        Returns:
            Parsed JSON response.

        Raises:
            Various ExchangeError subclasses depending on error code.
        """
        await self._throttle(weight)

        url = f"{self._base_url}{path}"
        params = params or {}

        # Filter None values
        params = {k: v for k, v in params.items() if v is not None}

        if signed:
            params["timestamp"] = current_timestamp_ms() + self._server_time_offset
            params["recvWindow"] = self.config.recv_window
            query = self._build_query(params)
            signature = self._sign_hmac_sha256(query)
            params["signature"] = signature

        t_start = time.monotonic()
        try:
            async with self._session.request(method, url, params=params) as resp:
                latency = (time.monotonic() - t_start) * 1000
                self._health.record_latency(latency)

                data = await resp.json()

                if resp.status == 429 or resp.status == 418:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    raise RateLimitError(
                        f"Rate limited: {data.get('msg', '')}",
                        retry_after=retry_after,
                        exchange=self.exchange_name,
                    )

                if resp.status != 200:
                    self._handle_error(data, resp.status)

                return data

        except aiohttp.ClientError as exc:
            self._health.record_error(str(exc))
            raise NetworkError(
                f"Network error: {exc}",
                exchange=self.exchange_name,
            ) from exc

    def _handle_error(self, data: Dict[str, Any], status: int) -> None:
        """Map Binance error codes to unified exceptions."""
        code = data.get("code", 0)
        msg = data.get("msg", f"HTTP {status}")

        if code in (-2015, -2014, -2008):
            raise AuthenticationError(msg, code=code, exchange=self.exchange_name)
        elif code in (-2010, -1013):
            raise InsufficientBalanceError(msg, code=code, exchange=self.exchange_name)
        elif code == -2011:
            raise OrderNotFoundError(msg, code=code, exchange=self.exchange_name)
        elif code in (-1102, -1100, -1104, -1106, -1111, -1112, -1114, -1115, -1116, -1117, -1118, -1119, -1120, -1121):
            raise InvalidOrderError(msg, code=code, exchange=self.exchange_name)
        elif code == -1003:
            raise RateLimitError(msg, code=code, exchange=self.exchange_name)
        else:
            raise ExchangeError(msg, code=code, exchange=self.exchange_name)

    async def _sync_server_time(self) -> None:
        """Synchronize with Binance server time to avoid timestamp errors."""
        try:
            local_ts = current_timestamp_ms()
            data = await self._request("GET", "/api/v3/time", weight=1)
            server_ts = data["serverTime"]
            self._server_time_offset = server_ts - local_ts
            self._logger.debug("Server time offset: %dms", self._server_time_offset)
        except Exception as exc:
            self._logger.warning("Failed to sync server time: %s", exc)
            self._server_time_offset = 0

    async def _load_exchange_info(self) -> None:
        """Load and cache exchange info (trading rules, filters, precision)."""
        data = await self._request("GET", "/api/v3/exchangeInfo", weight=10)
        for sym_info in data.get("symbols", []):
            symbol = sym_info["symbol"]
            filters = {}
            for f in sym_info.get("filters", []):
                filters[f["filterType"]] = f

            self._symbol_info[symbol] = {
                "status": sym_info["status"],
                "base_asset": sym_info["baseAsset"],
                "quote_asset": sym_info["quoteAsset"],
                "base_precision": sym_info.get("baseAssetPrecision", 8),
                "quote_precision": sym_info.get("quoteAssetPrecision", 8),
                "order_types": sym_info.get("orderTypes", []),
                "filters": filters,
                "min_notional": float(filters.get("NOTIONAL", {}).get("minNotional", 0) or
                                     filters.get("MIN_NOTIONAL", {}).get("minNotional", 0)),
                "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", "0.01")),
                "step_size": float(filters.get("LOT_SIZE", {}).get("stepSize", "0.00000100")),
                "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", "0.00000100")),
                "max_qty": float(filters.get("LOT_SIZE", {}).get("maxQty", "9000000")),
                "min_price": float(filters.get("PRICE_FILTER", {}).get("minPrice", "0")),
                "max_price": float(filters.get("PRICE_FILTER", {}).get("maxPrice", "0")),
            }

        self._exchange_info_cache = data
        self._exchange_info_ts = time.monotonic()
        self._logger.info("Loaded exchange info for %d symbols", len(self._symbol_info))

    async def _fetch_exchange_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v3/exchangeInfo", weight=10)

    # ------------------------------------------------------------------
    # Quantity and price precision helpers
    # ------------------------------------------------------------------

    def _adjust_quantity(self, symbol: str, quantity: float) -> str:
        """Adjust quantity to match exchange lot size filter."""
        info = self._symbol_info.get(symbol)
        if not info:
            return str(quantity)

        step_size = info["step_size"]
        if step_size <= 0:
            return str(quantity)

        precision = self._step_to_precision(step_size)
        adjusted = self._truncate(quantity, precision)
        return f"{adjusted:.{precision}f}"

    def _adjust_price(self, symbol: str, price: float) -> str:
        """Adjust price to match exchange tick size filter."""
        info = self._symbol_info.get(symbol)
        if not info:
            return str(price)

        tick_size = info["tick_size"]
        if tick_size <= 0:
            return str(price)

        precision = self._step_to_precision(tick_size)
        adjusted = self._truncate(price, precision)
        return f"{adjusted:.{precision}f}"

    @staticmethod
    def _step_to_precision(step: float) -> int:
        """Convert step size to decimal precision."""
        s = f"{step:.10f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0

    @staticmethod
    def _truncate(value: float, precision: int) -> float:
        """Truncate a float to the given decimal precision (no rounding)."""
        factor = 10 ** precision
        return int(value * factor) / factor

    def _validate_order_params(
        self,
        symbol: str,
        order_type: OrderType,
        quantity: float,
        price: Optional[float],
    ) -> None:
        """Validate order parameters against exchange filters."""
        info = self._symbol_info.get(symbol)
        if not info:
            return  # Skip validation if no info cached

        if info["status"] != "TRADING":
            raise InvalidOrderError(
                f"Symbol {symbol} is not trading (status={info['status']})",
                exchange=self.exchange_name,
            )

        if quantity < info["min_qty"]:
            raise InvalidOrderError(
                f"Quantity {quantity} below minimum {info['min_qty']}",
                exchange=self.exchange_name,
            )

        if quantity > info["max_qty"]:
            raise InvalidOrderError(
                f"Quantity {quantity} above maximum {info['max_qty']}",
                exchange=self.exchange_name,
            )

        if price is not None and info["min_price"] > 0:
            if price < info["min_price"]:
                raise InvalidOrderError(
                    f"Price {price} below minimum {info['min_price']}",
                    exchange=self.exchange_name,
                )
            if info["max_price"] > 0 and price > info["max_price"]:
                raise InvalidOrderError(
                    f"Price {price} above maximum {info['max_price']}",
                    exchange=self.exchange_name,
                )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        data = await self._request("GET", "/api/v3/ticker/24hr", {"symbol": symbol}, weight=1)
        return self._parse_ticker(data)

    async def get_tickers(self) -> List[UnifiedTicker]:
        """Get all tickers (weight=40)."""
        data = await self._request("GET", "/api/v3/ticker/24hr", weight=40)
        return [self._parse_ticker(t) for t in data]

    def _parse_ticker(self, data: Dict[str, Any]) -> UnifiedTicker:
        return UnifiedTicker(
            symbol=data["symbol"],
            exchange=self.exchange_name,
            last_price=float(data.get("lastPrice", 0)),
            bid_price=float(data.get("bidPrice", 0)),
            ask_price=float(data.get("askPrice", 0)),
            bid_qty=float(data.get("bidQty", 0)),
            ask_qty=float(data.get("askQty", 0)),
            high_24h=float(data.get("highPrice", 0)),
            low_24h=float(data.get("lowPrice", 0)),
            volume_24h=float(data.get("volume", 0)),
            quote_volume_24h=float(data.get("quoteVolume", 0)),
            price_change_24h=float(data.get("priceChange", 0)),
            price_change_pct=float(data.get("priceChangePercent", 0)),
            timestamp=int(data.get("closeTime", current_timestamp_ms())),
            raw=data,
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> UnifiedOrderbook:
        valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
        actual_limit = min(l for l in valid_limits if l >= limit) if limit <= 5000 else 5000
        weight = 1 if actual_limit <= 100 else (5 if actual_limit <= 500 else (10 if actual_limit <= 1000 else 50))

        data = await self._request(
            "GET", "/api/v3/depth",
            {"symbol": symbol, "limit": actual_limit},
            weight=weight,
        )
        return UnifiedOrderbook(
            symbol=symbol,
            exchange=self.exchange_name,
            bids=[OrderbookLevel(float(p), float(q)) for p, q in data.get("bids", [])],
            asks=[OrderbookLevel(float(p), float(q)) for p, q in data.get("asks", [])],
            timestamp=current_timestamp_ms(),
            last_update_id=data.get("lastUpdateId"),
            raw=data,
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedKline]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/api/v3/klines", params, weight=1)
        klines = []
        for k in data:
            klines.append(UnifiedKline(
                symbol=symbol,
                exchange=self.exchange_name,
                interval=interval,
                open_time=int(k[0]),
                close_time=int(k[6]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                quote_volume=float(k[7]),
                trades=int(k[8]),
                taker_buy_volume=float(k[9]),
                taker_buy_quote_volume=float(k[10]),
                is_closed=True,
                raw=k,
            ))
        return klines

    async def get_trades(self, symbol: str, limit: int = 500) -> List[UnifiedTrade]:
        data = await self._request(
            "GET", "/api/v3/trades",
            {"symbol": symbol, "limit": min(limit, 1000)},
            weight=1,
        )
        trades = []
        for t in data:
            trades.append(UnifiedTrade(
                symbol=symbol,
                exchange=self.exchange_name,
                trade_id=str(t["id"]),
                price=float(t["price"]),
                quantity=float(t["qty"]),
                quote_quantity=float(t.get("quoteQty", float(t["price"]) * float(t["qty"]))),
                side=OrderSide.SELL if t.get("isBuyerMaker", False) else OrderSide.BUY,
                timestamp=int(t["time"]),
                is_maker=t.get("isBuyerMaker", False),
                raw=t,
            ))
        return trades

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self, asset: Optional[str] = None) -> List[UnifiedBalance]:
        data = await self._request("GET", "/api/v3/account", signed=True, weight=10)
        balances = []
        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            total = free + locked
            if total == 0 and asset is None:
                continue
            if asset and b["asset"] != asset:
                continue
            balances.append(UnifiedBalance(
                asset=b["asset"],
                exchange=self.exchange_name,
                free=free,
                locked=locked,
                total=total,
                usd_value=0.0,  # Caller can enrich with price data
                raw=b,
            ))
        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[UnifiedPosition]:
        """Spot exchange does not have positions. Returns empty list."""
        return []

    async def get_account_info(self) -> UnifiedAccountInfo:
        data = await self._request("GET", "/api/v3/account", signed=True, weight=10)
        balances = []
        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            total = free + locked
            if total == 0:
                continue
            balances.append(UnifiedBalance(
                asset=b["asset"],
                exchange=self.exchange_name,
                free=free,
                locked=locked,
                total=total,
                usd_value=0.0,
                raw=b,
            ))

        return UnifiedAccountInfo(
            exchange=self.exchange_name,
            account_type="SPOT",
            can_trade=data.get("canTrade", False),
            can_withdraw=data.get("canWithdraw", False),
            can_deposit=data.get("canDeposit", False),
            balances=balances,
            total_usd_value=0.0,
            timestamp=data.get("updateTime", current_timestamp_ms()),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        post_only: bool = False,
        client_order_id: Optional[str] = None,
        position_side: Optional[PositionSide] = None,
        callback_rate: Optional[float] = None,
        **kwargs,
    ) -> UnifiedOrder:
        self._validate_order_params(symbol, order_type, quantity, price)

        if order_type == OrderType.OCO:
            return await self._place_oco_order(
                symbol, side, quantity, price,
                stop_price=stop_price,
                stop_limit_price=kwargs.get("stop_limit_price"),
                client_order_id=client_order_id,
            )

        binance_type = _ORDER_TYPE_MAP.get(order_type, order_type.value)
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": binance_type,
            "quantity": self._adjust_quantity(symbol, quantity),
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        # Price for limit orders
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT, OrderType.LIMIT_MAKER):
            if price is None:
                raise InvalidOrderError(
                    f"Price required for {order_type.value} orders",
                    exchange=self.exchange_name,
                )
            params["price"] = self._adjust_price(symbol, price)

        # Time in force (not for MARKET or LIMIT_MAKER)
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            params["timeInForce"] = _TIF_MAP.get(time_in_force, "GTC")

        # Stop price
        if stop_price and order_type in (OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT, OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET):
            params["stopPrice"] = self._adjust_price(symbol, stop_price)

        params["newOrderRespType"] = "FULL"

        data = await self._request("POST", "/api/v3/order", params, signed=True, weight=1)
        return self._parse_order(data)

    async def _place_oco_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float],
        stop_price: Optional[float] = None,
        stop_limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> UnifiedOrder:
        """Place an OCO (One-Cancels-the-Other) order."""
        if price is None or stop_price is None:
            raise InvalidOrderError(
                "OCO orders require both price and stop_price",
                exchange=self.exchange_name,
            )

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "quantity": self._adjust_quantity(symbol, quantity),
            "price": self._adjust_price(symbol, price),
            "stopPrice": self._adjust_price(symbol, stop_price),
        }

        if stop_limit_price:
            params["stopLimitPrice"] = self._adjust_price(symbol, stop_limit_price)
            params["stopLimitTimeInForce"] = "GTC"

        if client_order_id:
            params["listClientOrderId"] = client_order_id

        data = await self._request("POST", "/api/v3/order/oco", params, signed=True, weight=1)

        # OCO returns an orderListId plus multiple orders - return the first
        orders = data.get("orderReports", data.get("orders", []))
        if orders:
            return self._parse_order(orders[0])

        return UnifiedOrder(
            symbol=symbol,
            exchange=self.exchange_name,
            order_id=str(data.get("orderListId", "")),
            client_order_id=client_order_id or "",
            side=side,
            order_type=OrderType.OCO,
            status=OrderStatus.NEW,
            price=price or 0,
            stop_price=stop_price or 0,
            quantity=quantity,
            filled_quantity=0,
            remaining_quantity=quantity,
            average_price=0,
            commission=0,
            commission_asset="",
            created_at=data.get("transactionTime", current_timestamp_ms()),
            raw=data,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        params = {"symbol": symbol, "orderId": int(order_id)}
        data = await self._request("DELETE", "/api/v3/order", params, signed=True, weight=1)
        return self._parse_order(data)

    async def cancel_all_orders(self, symbol: str) -> List[UnifiedOrder]:
        """Cancel all open orders for a symbol."""
        params = {"symbol": symbol}
        data = await self._request("DELETE", "/api/v3/openOrders", params, signed=True, weight=1)
        return [self._parse_order(o) for o in data]

    async def modify_order(
        self,
        symbol: str,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> UnifiedOrder:
        """Modify an order using Binance's cancel-replace endpoint."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "cancelOrderId": int(order_id),
            "cancelReplaceMode": "STOP_ON_FAILURE",
        }

        # We need to get the original order first
        original = await self.get_order(symbol, order_id)

        binance_type = _ORDER_TYPE_MAP.get(original.order_type, original.order_type.value)
        params["side"] = original.side.value
        params["type"] = binance_type

        if quantity:
            params["quantity"] = self._adjust_quantity(symbol, quantity)
        else:
            params["quantity"] = self._adjust_quantity(symbol, original.quantity)

        if price:
            params["price"] = self._adjust_price(symbol, price)
        elif original.price > 0:
            params["price"] = self._adjust_price(symbol, original.price)

        if original.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            params["timeInForce"] = "GTC"

        data = await self._request("POST", "/api/v3/order/cancelReplace", params, signed=True, weight=1)
        new_order = data.get("newOrderResponse", data)
        return self._parse_order(new_order)

    async def get_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        params = {"symbol": symbol, "orderId": int(order_id)}
        data = await self._request("GET", "/api/v3/order", params, signed=True, weight=2)
        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {}
        weight = 3
        if symbol:
            params["symbol"] = symbol
            weight = 3
        else:
            weight = 40

        data = await self._request("GET", "/api/v3/openOrders", params, signed=True, weight=weight)
        return [self._parse_order(o) for o in data]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedOrder]:
        if not symbol:
            raise InvalidOrderError(
                "Binance requires symbol for order history",
                exchange=self.exchange_name,
            )
        params: Dict[str, Any] = {
            "symbol": symbol,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/api/v3/allOrders", params, signed=True, weight=10)
        return [self._parse_order(o) for o in data]

    async def get_my_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get account trade history for a symbol."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        return await self._request("GET", "/api/v3/myTrades", params, signed=True, weight=10)

    def _parse_order(self, data: Dict[str, Any]) -> UnifiedOrder:
        """Parse a Binance order response into UnifiedOrder."""
        raw_type = data.get("type", "MARKET")
        raw_status = data.get("status", "NEW")

        # Calculate commission from fills if available
        commission = 0.0
        commission_asset = ""
        fills = data.get("fills", [])
        if fills:
            for f in fills:
                commission += float(f.get("commission", 0))
            commission_asset = fills[0].get("commissionAsset", "") if fills else ""

        orig_qty = float(data.get("origQty", 0))
        executed_qty = float(data.get("executedQty", 0))

        return UnifiedOrder(
            symbol=data.get("symbol", ""),
            exchange=self.exchange_name,
            order_id=str(data.get("orderId", "")),
            client_order_id=data.get("clientOrderId", ""),
            side=OrderSide(data.get("side", "BUY")),
            order_type=_ORDER_TYPE_REV.get(raw_type, OrderType.MARKET),
            status=_ORDER_STATUS_MAP.get(raw_status, OrderStatus.NEW),
            price=float(data.get("price", 0)),
            stop_price=float(data.get("stopPrice", 0)),
            quantity=orig_qty,
            filled_quantity=executed_qty,
            remaining_quantity=orig_qty - executed_qty,
            average_price=float(data.get("avgPrice", 0)) or (
                float(data.get("cummulativeQuoteQty", 0)) / executed_qty
                if executed_qty > 0 else 0
            ),
            commission=commission,
            commission_asset=commission_asset,
            time_in_force=TimeInForce.GTC,
            created_at=int(data.get("time", data.get("transactTime", 0))),
            updated_at=int(data.get("updateTime", data.get("transactTime", 0))),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Futures stubs (not applicable for spot)
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        self._logger.warning("set_leverage called on spot exchange - no effect")
        return False

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        self._logger.warning("set_margin_mode called on spot exchange - no effect")
        return False

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "funding_rate": 0.0, "note": "not applicable for spot"}

    async def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        ticker = await self.get_ticker(symbol)
        return {"symbol": symbol, "mark_price": ticker.last_price, "index_price": ticker.last_price}

    # ------------------------------------------------------------------
    # WebSocket - Listen key management
    # ------------------------------------------------------------------

    async def _create_listen_key(self) -> str:
        """Create a new listen key for user data stream."""
        data = await self._request("POST", "/api/v3/userDataStream", weight=1)
        return data["listenKey"]

    async def _keepalive_listen_key(self) -> None:
        """Keep the listen key alive (must be called every 30 minutes)."""
        if not self._listen_key:
            return
        await self._request(
            "PUT", "/api/v3/userDataStream",
            {"listenKey": self._listen_key},
            weight=1,
        )

    async def _close_listen_key(self) -> None:
        """Close the listen key."""
        if not self._listen_key:
            return
        await self._request(
            "DELETE", "/api/v3/userDataStream",
            {"listenKey": self._listen_key},
            weight=1,
        )
        self._listen_key = None

    async def _listen_key_keepalive_loop(self) -> None:
        """Background task to keep the listen key alive."""
        while True:
            try:
                await asyncio.sleep(1200)  # Every 20 minutes
                await self._keepalive_listen_key()
                self._logger.debug("Listen key keepalive sent")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Listen key keepalive failed: %s", exc)
                # Try to create a new one
                try:
                    self._listen_key = await self._create_listen_key()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # WebSocket - Connection and message handling
    # ------------------------------------------------------------------

    async def _ws_connect(self, url: str, stream_id: str) -> None:
        """Connect to a WebSocket stream with automatic reconnection."""
        attempt = 0
        max_attempts = self.config.ws_reconnect_max_attempts
        base_delay = self.config.ws_reconnect_base_delay
        max_delay = self.config.ws_reconnect_max_delay

        while True:
            try:
                self._logger.info("WS connecting: %s (attempt %d)", stream_id, attempt + 1)
                async with self._session.ws_connect(
                    url,
                    heartbeat=self.config.ws_ping_interval,
                    timeout=self.config.ws_ping_timeout,
                ) as ws:
                    self._ws_connections[stream_id] = ws
                    attempt = 0  # Reset on successful connect
                    self._health.set_ws_status(True, len(self._ws_connections))
                    self._logger.info("WS connected: %s", stream_id)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                self._handle_ws_message(stream_id, data)
                            except json.JSONDecodeError:
                                self._logger.warning("WS invalid JSON: %s", msg.data[:200])
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self._logger.error("WS error: %s", ws.exception())
                            break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                            self._logger.info("WS closed: %s", stream_id)
                            break

            except asyncio.CancelledError:
                self._logger.info("WS task cancelled: %s", stream_id)
                break
            except Exception as exc:
                self._logger.error("WS connection error for %s: %s", stream_id, exc)
                self._health.record_error(f"ws_{stream_id}: {exc}")

            # Remove from active connections
            self._ws_connections.pop(stream_id, None)
            self._health.set_ws_status(
                bool(self._ws_connections),
                len(self._ws_connections),
            )

            if not self.config.enable_ws_reconnect:
                break

            attempt += 1
            if attempt > max_attempts:
                self._logger.error("WS max reconnect attempts reached for %s", stream_id)
                break

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            self._logger.info("WS reconnecting %s in %.1fs", stream_id, delay)
            await asyncio.sleep(delay)

    def _handle_ws_message(self, stream_id: str, data: Dict[str, Any]) -> None:
        """Route incoming WebSocket messages to registered callbacks."""
        event_type = data.get("e", "")
        self._health.record_heartbeat()

        # Market data streams
        if event_type == "24hrTicker":
            ticker = self._parse_ws_ticker(data)
            self._dispatch_callback(f"ticker:{data.get('s', '')}", ticker)

        elif event_type == "depthUpdate":
            book = self._parse_ws_depth(data)
            self._dispatch_callback(f"depth:{data.get('s', '')}", book)

        elif event_type == "trade":
            trade = self._parse_ws_trade(data)
            self._dispatch_callback(f"trade:{data.get('s', '')}", trade)

        elif event_type == "kline":
            kline = self._parse_ws_kline(data)
            k_data = data.get("k", {})
            self._dispatch_callback(f"kline:{k_data.get('s', '')}:{k_data.get('i', '')}", kline)

        # User data events
        elif event_type == "executionReport":
            order = self._parse_ws_order(data)
            self._dispatch_callback("user:order", order)

        elif event_type == "outboundAccountPosition":
            balances = self._parse_ws_balance(data)
            for b in balances:
                self._dispatch_callback("user:balance", b)

        elif event_type == "balanceUpdate":
            # Single asset balance change
            self._dispatch_callback("user:balance_update", data)

        else:
            # Combined stream format - data might be in "data" field
            if "stream" in data and "data" in data:
                self._handle_ws_message(stream_id, data["data"])

    # ------------------------------------------------------------------
    # WebSocket message parsers
    # ------------------------------------------------------------------

    def _parse_ws_ticker(self, data: Dict[str, Any]) -> UnifiedTicker:
        return UnifiedTicker(
            symbol=data.get("s", ""),
            exchange=self.exchange_name,
            last_price=float(data.get("c", 0)),
            bid_price=float(data.get("b", 0)),
            ask_price=float(data.get("a", 0)),
            bid_qty=float(data.get("B", 0)),
            ask_qty=float(data.get("A", 0)),
            high_24h=float(data.get("h", 0)),
            low_24h=float(data.get("l", 0)),
            volume_24h=float(data.get("v", 0)),
            quote_volume_24h=float(data.get("q", 0)),
            price_change_24h=float(data.get("p", 0)),
            price_change_pct=float(data.get("P", 0)),
            timestamp=int(data.get("E", current_timestamp_ms())),
            raw=data,
        )

    def _parse_ws_depth(self, data: Dict[str, Any]) -> UnifiedOrderbook:
        return UnifiedOrderbook(
            symbol=data.get("s", ""),
            exchange=self.exchange_name,
            bids=[OrderbookLevel(float(p), float(q)) for p, q in data.get("b", [])],
            asks=[OrderbookLevel(float(p), float(q)) for p, q in data.get("a", [])],
            timestamp=int(data.get("E", current_timestamp_ms())),
            last_update_id=data.get("u"),
            raw=data,
        )

    def _parse_ws_trade(self, data: Dict[str, Any]) -> UnifiedTrade:
        return UnifiedTrade(
            symbol=data.get("s", ""),
            exchange=self.exchange_name,
            trade_id=str(data.get("t", "")),
            price=float(data.get("p", 0)),
            quantity=float(data.get("q", 0)),
            quote_quantity=float(data.get("p", 0)) * float(data.get("q", 0)),
            side=OrderSide.SELL if data.get("m", False) else OrderSide.BUY,
            timestamp=int(data.get("T", current_timestamp_ms())),
            is_maker=data.get("m", False),
            raw=data,
        )

    def _parse_ws_kline(self, data: Dict[str, Any]) -> UnifiedKline:
        k = data.get("k", {})
        return UnifiedKline(
            symbol=k.get("s", ""),
            exchange=self.exchange_name,
            interval=k.get("i", ""),
            open_time=int(k.get("t", 0)),
            close_time=int(k.get("T", 0)),
            open=float(k.get("o", 0)),
            high=float(k.get("h", 0)),
            low=float(k.get("l", 0)),
            close=float(k.get("c", 0)),
            volume=float(k.get("v", 0)),
            quote_volume=float(k.get("q", 0)),
            trades=int(k.get("n", 0)),
            taker_buy_volume=float(k.get("V", 0)),
            taker_buy_quote_volume=float(k.get("Q", 0)),
            is_closed=k.get("x", False),
            raw=data,
        )

    def _parse_ws_order(self, data: Dict[str, Any]) -> UnifiedOrder:
        raw_type = data.get("o", "MARKET")
        raw_status = data.get("X", "NEW")
        orig_qty = float(data.get("q", 0))
        filled_qty = float(data.get("z", 0))
        cum_quote = float(data.get("Z", 0))

        return UnifiedOrder(
            symbol=data.get("s", ""),
            exchange=self.exchange_name,
            order_id=str(data.get("i", "")),
            client_order_id=data.get("c", ""),
            side=OrderSide(data.get("S", "BUY")),
            order_type=_ORDER_TYPE_REV.get(raw_type, OrderType.MARKET),
            status=_ORDER_STATUS_MAP.get(raw_status, OrderStatus.NEW),
            price=float(data.get("p", 0)),
            stop_price=float(data.get("P", 0)),
            quantity=orig_qty,
            filled_quantity=filled_qty,
            remaining_quantity=orig_qty - filled_qty,
            average_price=cum_quote / filled_qty if filled_qty > 0 else 0,
            commission=float(data.get("n", 0)),
            commission_asset=data.get("N", ""),
            created_at=int(data.get("O", 0)),
            updated_at=int(data.get("T", 0)),
            raw=data,
        )

    def _parse_ws_balance(self, data: Dict[str, Any]) -> List[UnifiedBalance]:
        balances = []
        for b in data.get("B", []):
            free = float(b.get("f", 0))
            locked = float(b.get("l", 0))
            balances.append(UnifiedBalance(
                asset=b.get("a", ""),
                exchange=self.exchange_name,
                free=free,
                locked=locked,
                total=free + locked,
                usd_value=0.0,
                raw=b,
            ))
        return balances

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    def _next_sub_id(self) -> str:
        self._sub_counter += 1
        return f"sub_{self._sub_counter}_{uuid.uuid4().hex[:8]}"

    async def subscribe_ticker(self, symbol: str, callback: Callable[[UnifiedTicker], Any]) -> str:
        sub_id = self._next_sub_id()
        stream_name = f"{symbol.lower()}@ticker"
        stream_key = f"ticker:{symbol}"
        self._register_callback(stream_key, callback)

        url = f"{self._ws_base_url}/{stream_name}"
        task = asyncio.create_task(self._ws_connect(url, sub_id))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {
            "type": "ticker",
            "symbol": symbol,
            "stream": stream_name,
            "key": stream_key,
        }
        return sub_id

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[UnifiedOrderbook], Any], depth: int = 20
    ) -> str:
        sub_id = self._next_sub_id()
        stream_name = f"{symbol.lower()}@depth@100ms"
        stream_key = f"depth:{symbol}"
        self._register_callback(stream_key, callback)

        url = f"{self._ws_base_url}/{stream_name}"
        task = asyncio.create_task(self._ws_connect(url, sub_id))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {
            "type": "depth",
            "symbol": symbol,
            "stream": stream_name,
            "key": stream_key,
        }
        return sub_id

    async def subscribe_trades(self, symbol: str, callback: Callable[[UnifiedTrade], Any]) -> str:
        sub_id = self._next_sub_id()
        stream_name = f"{symbol.lower()}@trade"
        stream_key = f"trade:{symbol}"
        self._register_callback(stream_key, callback)

        url = f"{self._ws_base_url}/{stream_name}"
        task = asyncio.create_task(self._ws_connect(url, sub_id))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {
            "type": "trade",
            "symbol": symbol,
            "stream": stream_name,
            "key": stream_key,
        }
        return sub_id

    async def subscribe_klines(
        self, symbol: str, interval: str, callback: Callable[[UnifiedKline], Any]
    ) -> str:
        sub_id = self._next_sub_id()
        stream_name = f"{symbol.lower()}@kline_{interval}"
        stream_key = f"kline:{symbol}:{interval}"
        self._register_callback(stream_key, callback)

        url = f"{self._ws_base_url}/{stream_name}"
        task = asyncio.create_task(self._ws_connect(url, sub_id))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {
            "type": "kline",
            "symbol": symbol,
            "interval": interval,
            "stream": stream_name,
            "key": stream_key,
        }
        return sub_id

    async def subscribe_user_data(
        self,
        on_order: Optional[Callable[[UnifiedOrder], Any]] = None,
        on_position: Optional[Callable[[UnifiedPosition], Any]] = None,
        on_balance: Optional[Callable[[UnifiedBalance], Any]] = None,
    ) -> str:
        if not self.config.api_key:
            raise AuthenticationError(
                "API key required for user data stream",
                exchange=self.exchange_name,
            )

        self._listen_key = await self._create_listen_key()
        sub_id = self._next_sub_id()

        if on_order:
            self._register_callback("user:order", on_order)
        if on_balance:
            self._register_callback("user:balance", on_balance)

        url = f"{self._ws_base_url}/{self._listen_key}"
        task = asyncio.create_task(self._ws_connect(url, sub_id))
        self._ws_tasks[sub_id] = task

        # Start keepalive loop
        self._listen_key_task = asyncio.create_task(self._listen_key_keepalive_loop())

        self._subscriptions[sub_id] = {
            "type": "user_data",
            "listen_key": self._listen_key,
            "key": "user",
        }
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        sub = self._subscriptions.pop(subscription_id, None)
        if not sub:
            return False

        # Cancel the WS task
        task = self._ws_tasks.pop(subscription_id, None)
        if task and not task.done():
            task.cancel()

        # Close the WS connection
        ws = self._ws_connections.pop(subscription_id, None)
        if ws and not ws.closed:
            await ws.close()

        # Remove callbacks
        key = sub.get("key", "")
        if key:
            self._remove_callbacks(key)

        # Close listen key if user data
        if sub.get("type") == "user_data":
            await self._close_listen_key()
            if self._listen_key_task and not self._listen_key_task.done():
                self._listen_key_task.cancel()

        self._logger.info("Unsubscribed: %s", subscription_id)
        return True

    @property
    def active_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._subscriptions)
