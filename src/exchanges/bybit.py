# mefai-autotrade - Bybit V5 exchange connector
# Unified API supporting spot, linear, inverse, and option categories

import asyncio
import hashlib
import hmac
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
    PositionSide,
    MarginMode,
    TimeInForce,
    current_timestamp_ms,
)

logger = logging.getLogger(__name__)

# Bybit V5 API URLs
BYBIT_REST = "https://api.bybit.com"
BYBIT_REST_TESTNET = "https://api-testnet.bybit.com"
BYBIT_WS_PUBLIC = "wss://stream.bybit.com/v5/public"
BYBIT_WS_PRIVATE = "wss://stream.bybit.com/v5/private"
BYBIT_WS_PUBLIC_TESTNET = "wss://stream-testnet.bybit.com/v5/public"
BYBIT_WS_PRIVATE_TESTNET = "wss://stream-testnet.bybit.com/v5/private"

# Order type mapping
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "Market",
    OrderType.LIMIT: "Limit",
}

_ORDER_TYPE_REV = {
    "Market": OrderType.MARKET,
    "Limit": OrderType.LIMIT,
}

_ORDER_STATUS_MAP = {
    "New": OrderStatus.NEW,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELED,
    "PartiallyFilledCanceled": OrderStatus.CANCELED,
    "Rejected": OrderStatus.REJECTED,
    "Deactivated": OrderStatus.EXPIRED,
    "Triggered": OrderStatus.NEW,
    "Active": OrderStatus.NEW,
    "Untriggered": OrderStatus.PENDING,
}

_TIF_MAP = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.GTX: "PostOnly",
}


class BybitExchange(ExchangeBase):
    """Bybit V5 unified API exchange connector.

    Supports spot, linear (USDT perpetual), inverse perpetual, and option
    categories through the V5 unified API. All methods accept a category
    parameter to route to the correct market type.
    """

    def __init__(self, config: ExchangeConfig, category: str = "linear"):
        """Initialize Bybit connector.

        Args:
            config: Exchange configuration.
            category: Default trading category - spot, linear, inverse, option.
        """
        super().__init__(config, "bybit")
        self._category = category
        self._base_url = config.base_url_override or (
            BYBIT_REST_TESTNET if config.testnet else BYBIT_REST
        )
        self._ws_public_url = (
            f"{BYBIT_WS_PUBLIC_TESTNET}/{category}" if config.testnet
            else f"{BYBIT_WS_PUBLIC}/{category}"
        )
        self._ws_private_url = (
            BYBIT_WS_PRIVATE_TESTNET if config.testnet else BYBIT_WS_PRIVATE
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._ws_tasks: Dict[str, asyncio.Task] = {}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._sub_counter = 0
        self._symbol_info: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        )
        await self._load_exchange_info()
        self._initialized = True
        self._logger.info("Bybit connector initialized (category=%s, testnet=%s)", self._category, self.config.testnet)

    async def close(self) -> None:
        for task in self._ws_tasks.values():
            if not task.done():
                task.cancel()
        for ws in self._ws_connections.values():
            if not ws.closed:
                await ws.close()
        self._ws_connections.clear()
        self._ws_tasks.clear()
        self._subscriptions.clear()
        if self._session and not self._session.closed:
            await self._session.close()
        self._initialized = False
        self._logger.info("Bybit connector closed")

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def _sign_request(self, params: Dict[str, Any], timestamp: int, recv_window: int) -> str:
        """Generate Bybit V5 HMAC-SHA256 signature.

        Signature string: timestamp + api_key + recv_window + query_string
        """
        if not self.config.api_secret:
            return ""

        # For GET: sorted query string. For POST: JSON body string.
        param_str = ""
        if params:
            # Sort for GET requests
            sorted_params = sorted(params.items())
            param_str = "&".join(f"{k}={v}" for k, v in sorted_params if v is not None)

        sign_str = f"{timestamp}{self.config.api_key}{recv_window}{param_str}"
        return hmac.new(
            self.config.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _sign_post_request(self, body: str, timestamp: int, recv_window: int) -> str:
        """Sign a POST request body."""
        sign_str = f"{timestamp}{self.config.api_key}{recv_window}{body}"
        return hmac.new(
            self.config.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Any:
        await self._throttle(weight)
        url = f"{self._base_url}{path}"
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}

        if signed and self.config.api_key:
            timestamp = current_timestamp_ms()
            recv_window = self.config.recv_window
            headers["X-BAPI-API-KEY"] = self.config.api_key
            headers["X-BAPI-TIMESTAMP"] = str(timestamp)
            headers["X-BAPI-RECV-WINDOW"] = str(recv_window)

            if method == "POST":
                body = json.dumps(params) if params else ""
                headers["X-BAPI-SIGN"] = self._sign_post_request(body, timestamp, recv_window)
                headers["Content-Type"] = "application/json"
            else:
                headers["X-BAPI-SIGN"] = self._sign_request(params, timestamp, recv_window)

        t_start = time.monotonic()
        try:
            if method == "POST":
                async with self._session.post(url, json=params, headers=headers) as resp:
                    latency = (time.monotonic() - t_start) * 1000
                    self._health.record_latency(latency)
                    data = await resp.json()
            else:
                async with self._session.get(url, params=params, headers=headers) as resp:
                    latency = (time.monotonic() - t_start) * 1000
                    self._health.record_latency(latency)
                    data = await resp.json()

            ret_code = data.get("retCode", 0)
            if ret_code != 0:
                self._handle_error(data)

            return data.get("result", data)

        except aiohttp.ClientError as exc:
            self._health.record_error(str(exc))
            raise NetworkError(f"Network error: {exc}", exchange=self.exchange_name) from exc

    def _handle_error(self, data: Dict[str, Any]) -> None:
        code = data.get("retCode", 0)
        msg = data.get("retMsg", "Unknown error")

        if code in (10003, 10004, 10005, 33004):
            raise AuthenticationError(msg, code=code, exchange=self.exchange_name)
        elif code in (110001, 110045, 110043):
            raise InsufficientBalanceError(msg, code=code, exchange=self.exchange_name)
        elif code == 110001:
            raise OrderNotFoundError(msg, code=code, exchange=self.exchange_name)
        elif code in (10001, 10014, 110007, 110008, 110009, 110012, 110013, 110014, 110015, 110016, 110017, 110018):
            raise InvalidOrderError(msg, code=code, exchange=self.exchange_name)
        elif code == 10006:
            raise RateLimitError(msg, code=code, exchange=self.exchange_name)
        else:
            raise ExchangeError(msg, code=code, exchange=self.exchange_name)

    async def _load_exchange_info(self) -> None:
        data = await self._request(
            "GET", "/v5/market/instruments-info",
            {"category": self._category, "limit": 1000},
        )
        for item in data.get("list", []):
            symbol = item["symbol"]
            lot_filter = item.get("lotSizeFilter", {})
            price_filter = item.get("priceFilter", {})
            leverage_filter = item.get("leverageFilter", {})

            self._symbol_info[symbol] = {
                "status": item.get("status", ""),
                "base_coin": item.get("baseCoin", ""),
                "quote_coin": item.get("quoteCoin", ""),
                "settle_coin": item.get("settleCoin", ""),
                "contract_type": item.get("contractType", ""),
                "launch_time": item.get("launchTime", ""),
                "delivery_time": item.get("deliveryTime", ""),
                "min_qty": float(lot_filter.get("minOrderQty", "0.001")),
                "max_qty": float(lot_filter.get("maxOrderQty", "100000")),
                "qty_step": float(lot_filter.get("qtyStep", "0.001")),
                "min_price": float(price_filter.get("minPrice", "0.01")),
                "max_price": float(price_filter.get("maxPrice", "999999")),
                "tick_size": float(price_filter.get("tickSize", "0.01")),
                "min_leverage": float(leverage_filter.get("minLeverage", "1")),
                "max_leverage": float(leverage_filter.get("maxLeverage", "100")),
                "leverage_step": float(leverage_filter.get("leverageStep", "0.01")),
                "min_notional": float(lot_filter.get("minNotionalValue", "5")),
                "unified_margin": item.get("isPreListing", False),
            }
        self._exchange_info_cache = data
        self._exchange_info_ts = time.monotonic()
        self._logger.info("Loaded Bybit info for %d %s symbols", len(self._symbol_info), self._category)

    async def _fetch_exchange_info(self) -> Dict[str, Any]:
        return await self._request(
            "GET", "/v5/market/instruments-info",
            {"category": self._category, "limit": 1000},
        )

    # ------------------------------------------------------------------
    # Precision helpers
    # ------------------------------------------------------------------

    def _adjust_quantity(self, symbol: str, quantity: float) -> str:
        info = self._symbol_info.get(symbol)
        if not info:
            return str(quantity)
        step = info["qty_step"]
        precision = self._step_precision(step)
        adjusted = int(quantity / step) * step
        return f"{adjusted:.{precision}f}"

    def _adjust_price(self, symbol: str, price: float) -> str:
        info = self._symbol_info.get(symbol)
        if not info:
            return str(price)
        tick = info["tick_size"]
        precision = self._step_precision(tick)
        adjusted = int(price / tick) * tick
        return f"{adjusted:.{precision}f}"

    @staticmethod
    def _step_precision(step: float) -> int:
        s = f"{step:.10f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        data = await self._request(
            "GET", "/v5/market/tickers",
            {"category": self._category, "symbol": symbol},
        )
        items = data.get("list", [])
        if not items:
            raise ExchangeError(f"No ticker data for {symbol}", exchange=self.exchange_name)
        return self._parse_ticker(items[0])

    async def get_tickers(self, category: Optional[str] = None) -> List[UnifiedTicker]:
        cat = category or self._category
        data = await self._request("GET", "/v5/market/tickers", {"category": cat})
        return [self._parse_ticker(t) for t in data.get("list", [])]

    def _parse_ticker(self, data: Dict[str, Any]) -> UnifiedTicker:
        return UnifiedTicker(
            symbol=data.get("symbol", ""),
            exchange=self.exchange_name,
            last_price=float(data.get("lastPrice", 0)),
            bid_price=float(data.get("bid1Price", 0)),
            ask_price=float(data.get("ask1Price", 0)),
            bid_qty=float(data.get("bid1Size", 0)),
            ask_qty=float(data.get("ask1Size", 0)),
            high_24h=float(data.get("highPrice24h", 0)),
            low_24h=float(data.get("lowPrice24h", 0)),
            volume_24h=float(data.get("volume24h", 0)),
            quote_volume_24h=float(data.get("turnover24h", 0)),
            price_change_24h=float(data.get("price24hPcnt", 0)) * float(data.get("prevPrice24h", 0)) if data.get("prevPrice24h") else 0,
            price_change_pct=float(data.get("price24hPcnt", 0)) * 100,
            timestamp=current_timestamp_ms(),
            raw=data,
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> UnifiedOrderbook:
        actual_limit = min(limit, 200)
        data = await self._request(
            "GET", "/v5/market/orderbook",
            {"category": self._category, "symbol": symbol, "limit": actual_limit},
        )
        return UnifiedOrderbook(
            symbol=symbol,
            exchange=self.exchange_name,
            bids=[OrderbookLevel(float(p), float(q)) for p, q in data.get("b", [])],
            asks=[OrderbookLevel(float(p), float(q)) for p, q in data.get("a", [])],
            timestamp=int(data.get("ts", current_timestamp_ms())),
            last_update_id=int(data.get("u", 0)),
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
        # Bybit uses different interval format: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M
        bybit_interval = self._convert_interval(interval)
        params: Dict[str, Any] = {
            "category": self._category,
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        data = await self._request("GET", "/v5/market/kline", params)
        klines = []
        for k in data.get("list", []):
            klines.append(UnifiedKline(
                symbol=symbol,
                exchange=self.exchange_name,
                interval=interval,
                open_time=int(k[0]),
                close_time=int(k[0]) + self._interval_ms(interval) - 1,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                quote_volume=float(k[6]) if len(k) > 6 else 0,
                trades=0,
                is_closed=True,
                raw=k,
            ))
        # Bybit returns newest first, reverse to match standard chronological order
        klines.reverse()
        return klines

    async def get_trades(self, symbol: str, limit: int = 500) -> List[UnifiedTrade]:
        data = await self._request(
            "GET", "/v5/market/recent-trade",
            {"category": self._category, "symbol": symbol, "limit": min(limit, 1000)},
        )
        trades = []
        for t in data.get("list", []):
            side_str = t.get("side", "Buy")
            trades.append(UnifiedTrade(
                symbol=symbol,
                exchange=self.exchange_name,
                trade_id=str(t.get("execId", "")),
                price=float(t.get("price", 0)),
                quantity=float(t.get("size", 0)),
                quote_quantity=float(t.get("price", 0)) * float(t.get("size", 0)),
                side=OrderSide.BUY if side_str == "Buy" else OrderSide.SELL,
                timestamp=int(t.get("time", current_timestamp_ms())),
                is_maker=t.get("isBlockTrade", False),
                raw=t,
            ))
        return trades

    @staticmethod
    def _convert_interval(interval: str) -> str:
        """Convert standard interval to Bybit format."""
        mapping = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M",
        }
        return mapping.get(interval, interval)

    @staticmethod
    def _interval_ms(interval: str) -> int:
        """Convert interval string to milliseconds."""
        mapping = {
            "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000, "30m": 1800000,
            "1h": 3600000, "2h": 7200000, "4h": 14400000, "6h": 21600000, "12h": 43200000,
            "1d": 86400000, "1w": 604800000, "1M": 2592000000,
        }
        return mapping.get(interval, 60000)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_balance(self, asset: Optional[str] = None) -> List[UnifiedBalance]:
        account_type = "UNIFIED" if self._category != "spot" else "SPOT"
        params: Dict[str, Any] = {"accountType": account_type}
        if asset:
            params["coin"] = asset
        data = await self._request("GET", "/v5/account/wallet-balance", params, signed=True)

        balances = []
        for account in data.get("list", []):
            for coin in account.get("coin", []):
                free = float(coin.get("availableToWithdraw", 0))
                locked = float(coin.get("locked", 0))
                total = float(coin.get("walletBalance", 0))
                usd_val = float(coin.get("usdValue", 0))
                if total == 0 and asset is None:
                    continue
                balances.append(UnifiedBalance(
                    asset=coin.get("coin", ""),
                    exchange=self.exchange_name,
                    free=free,
                    locked=locked,
                    total=total,
                    usd_value=usd_val,
                    raw=coin,
                ))
        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[UnifiedPosition]:
        if self._category == "spot":
            return []

        params: Dict[str, Any] = {"category": self._category}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", "/v5/position/list", params, signed=True)
        positions = []
        for p in data.get("list", []):
            size = float(p.get("size", 0))
            if size == 0 and symbol is None:
                continue

            side_str = p.get("side", "")
            if side_str == "Buy":
                pos_side = PositionSide.LONG
            elif side_str == "Sell":
                pos_side = PositionSide.SHORT
            else:
                pos_side = PositionSide.BOTH

            positions.append(UnifiedPosition(
                symbol=p.get("symbol", ""),
                exchange=self.exchange_name,
                side=pos_side,
                size=size,
                entry_price=float(p.get("avgPrice", 0)),
                mark_price=float(p.get("markPrice", 0)),
                liquidation_price=float(p.get("liqPrice", 0)),
                unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                realized_pnl=float(p.get("cumRealisedPnl", 0)),
                leverage=int(float(p.get("leverage", 1))),
                margin_mode=MarginMode.ISOLATED if p.get("tradeMode", 0) == 1 else MarginMode.CROSS,
                margin=float(p.get("positionIM", 0)),
                notional=float(p.get("positionValue", 0)),
                adl_quantile=float(p.get("adlRankIndicator", 0)),
                timestamp=int(p.get("updatedTime", current_timestamp_ms())),
                raw=p,
            ))
        return positions

    async def get_account_info(self) -> UnifiedAccountInfo:
        balances = await self.get_balance()
        positions = await self.get_positions()

        total_usd = sum(b.usd_value for b in balances)
        total_unrealized = sum(p.unrealized_pnl for p in positions)

        return UnifiedAccountInfo(
            exchange=self.exchange_name,
            account_type=self._category.upper(),
            can_trade=True,
            can_withdraw=True,
            can_deposit=True,
            balances=balances,
            total_usd_value=total_usd,
            total_unrealized_pnl=total_unrealized,
            positions=positions,
            timestamp=current_timestamp_ms(),
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
        bybit_side = "Buy" if side == OrderSide.BUY else "Sell"
        bybit_type = _ORDER_TYPE_MAP.get(order_type, "Market")

        params: Dict[str, Any] = {
            "category": self._category,
            "symbol": symbol,
            "side": bybit_side,
            "orderType": bybit_type,
            "qty": self._adjust_quantity(symbol, quantity),
        }

        if client_order_id:
            params["orderLinkId"] = client_order_id

        if price and order_type in (OrderType.LIMIT, OrderType.LIMIT_MAKER):
            params["price"] = self._adjust_price(symbol, price)

        if post_only:
            params["timeInForce"] = "PostOnly"
        elif order_type == OrderType.LIMIT:
            params["timeInForce"] = _TIF_MAP.get(time_in_force, "GTC")

        if stop_price:
            params["triggerPrice"] = self._adjust_price(symbol, stop_price)
            params["triggerDirection"] = kwargs.get("trigger_direction", 1)

        if reduce_only and self._category != "spot":
            params["reduceOnly"] = True

        if position_side and self._category != "spot":
            params["positionIdx"] = 1 if position_side == PositionSide.LONG else 2

        if callback_rate:
            params["trailingStop"] = str(callback_rate)

        # Take profit / stop loss
        tp_price = kwargs.get("take_profit")
        sl_price = kwargs.get("stop_loss")
        if tp_price:
            params["takeProfit"] = self._adjust_price(symbol, tp_price)
        if sl_price:
            params["stopLoss"] = self._adjust_price(symbol, sl_price)

        data = await self._request("POST", "/v5/order/create", params, signed=True)
        return self._parse_order_response(data, symbol, side, order_type, quantity, price)

    async def cancel_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        params = {
            "category": self._category,
            "symbol": symbol,
            "orderId": order_id,
        }
        data = await self._request("POST", "/v5/order/cancel", params, signed=True)
        return UnifiedOrder(
            symbol=symbol,
            exchange=self.exchange_name,
            order_id=data.get("orderId", order_id),
            client_order_id=data.get("orderLinkId", ""),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.CANCELED,
            price=0, stop_price=0, quantity=0,
            filled_quantity=0, remaining_quantity=0,
            average_price=0, commission=0, commission_asset="",
            raw=data,
        )

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders."""
        params: Dict[str, Any] = {"category": self._category}
        if symbol:
            params["symbol"] = symbol
        await self._request("POST", "/v5/order/cancel-all", params, signed=True)
        return True

    async def modify_order(
        self,
        symbol: str,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> UnifiedOrder:
        params: Dict[str, Any] = {
            "category": self._category,
            "symbol": symbol,
            "orderId": order_id,
        }
        if quantity:
            params["qty"] = self._adjust_quantity(symbol, quantity)
        if price:
            params["price"] = self._adjust_price(symbol, price)

        data = await self._request("POST", "/v5/order/amend", params, signed=True)
        return UnifiedOrder(
            symbol=symbol,
            exchange=self.exchange_name,
            order_id=data.get("orderId", order_id),
            client_order_id=data.get("orderLinkId", ""),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            price=price or 0, stop_price=0,
            quantity=quantity or 0,
            filled_quantity=0, remaining_quantity=quantity or 0,
            average_price=0, commission=0, commission_asset="",
            raw=data,
        )

    async def get_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        params = {
            "category": self._category,
            "symbol": symbol,
            "orderId": order_id,
        }
        data = await self._request("GET", "/v5/order/realtime", params, signed=True)
        items = data.get("list", [])
        if not items:
            raise OrderNotFoundError(f"Order {order_id} not found", exchange=self.exchange_name)
        return self._parse_order_detail(items[0])

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {"category": self._category}
        if symbol:
            params["symbol"] = symbol
        data = await self._request("GET", "/v5/order/realtime", params, signed=True)
        return [self._parse_order_detail(o) for o in data.get("list", [])]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {
            "category": self._category,
            "limit": min(limit, 50),
        }
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/v5/order/history", params, signed=True)
        return [self._parse_order_detail(o) for o in data.get("list", [])]

    def _parse_order_response(
        self,
        data: Dict[str, Any],
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float],
    ) -> UnifiedOrder:
        """Parse a create-order response (limited fields)."""
        return UnifiedOrder(
            symbol=symbol,
            exchange=self.exchange_name,
            order_id=data.get("orderId", ""),
            client_order_id=data.get("orderLinkId", ""),
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            price=price or 0,
            stop_price=0,
            quantity=quantity,
            filled_quantity=0,
            remaining_quantity=quantity,
            average_price=0,
            commission=0,
            commission_asset="",
            created_at=current_timestamp_ms(),
            raw=data,
        )

    def _parse_order_detail(self, data: Dict[str, Any]) -> UnifiedOrder:
        """Parse a full order detail from realtime/history endpoint."""
        raw_type = data.get("orderType", "Market")
        raw_status = data.get("orderStatus", "New")
        side_str = data.get("side", "Buy")
        qty = float(data.get("qty", 0))
        filled = float(data.get("cumExecQty", 0))
        cum_value = float(data.get("cumExecValue", 0))

        return UnifiedOrder(
            symbol=data.get("symbol", ""),
            exchange=self.exchange_name,
            order_id=data.get("orderId", ""),
            client_order_id=data.get("orderLinkId", ""),
            side=OrderSide.BUY if side_str == "Buy" else OrderSide.SELL,
            order_type=_ORDER_TYPE_REV.get(raw_type, OrderType.MARKET),
            status=_ORDER_STATUS_MAP.get(raw_status, OrderStatus.NEW),
            price=float(data.get("price", 0)),
            stop_price=float(data.get("triggerPrice", 0)),
            quantity=qty,
            filled_quantity=filled,
            remaining_quantity=qty - filled,
            average_price=float(data.get("avgPrice", 0)) or (cum_value / filled if filled > 0 else 0),
            commission=float(data.get("cumExecFee", 0)),
            commission_asset="",
            time_in_force=TimeInForce.GTC,
            reduce_only=data.get("reduceOnly", False),
            created_at=int(data.get("createdTime", 0)),
            updated_at=int(data.get("updatedTime", 0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Futures specific
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        if self._category == "spot":
            return False
        try:
            await self._request(
                "POST", "/v5/position/set-leverage",
                {
                    "category": self._category,
                    "symbol": symbol,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage),
                },
                signed=True,
            )
            return True
        except ExchangeError as exc:
            # 110043 means leverage not modified (already set)
            if hasattr(exc, "code") and exc.code == 110043:
                return True
            self._logger.error("Failed to set leverage: %s", exc)
            return False

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        if self._category == "spot":
            return False
        trade_mode = 1 if mode == MarginMode.ISOLATED else 0
        try:
            await self._request(
                "POST", "/v5/position/switch-isolated",
                {
                    "category": self._category,
                    "symbol": symbol,
                    "tradeMode": trade_mode,
                    "buyLeverage": "10",
                    "sellLeverage": "10",
                },
                signed=True,
            )
            return True
        except ExchangeError as exc:
            if hasattr(exc, "code") and exc.code == 110026:
                return True
            self._logger.error("Failed to set margin mode: %s", exc)
            return False

    async def set_position_mode(self, hedge_mode: bool) -> bool:
        """Set position mode: 0 = one-way, 3 = hedge."""
        try:
            await self._request(
                "POST", "/v5/position/switch-mode",
                {"category": self._category, "mode": 3 if hedge_mode else 0},
                signed=True,
            )
            return True
        except ExchangeError:
            return False

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        if self._category == "spot":
            return {"symbol": symbol, "funding_rate": 0.0}
        data = await self._request(
            "GET", "/v5/market/funding/history",
            {"category": self._category, "symbol": symbol, "limit": 1},
        )
        items = data.get("list", [])
        if not items:
            return {"symbol": symbol, "funding_rate": 0.0}
        item = items[0]
        return {
            "symbol": symbol,
            "funding_rate": float(item.get("fundingRate", 0)),
            "funding_rate_timestamp": int(item.get("fundingRateTimestamp", 0)),
        }

    async def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        data = await self._request(
            "GET", "/v5/market/tickers",
            {"category": self._category, "symbol": symbol},
        )
        items = data.get("list", [])
        if not items:
            return {"symbol": symbol, "mark_price": 0, "index_price": 0}
        item = items[0]
        return {
            "symbol": symbol,
            "mark_price": float(item.get("markPrice", 0)),
            "index_price": float(item.get("indexPrice", 0)),
            "last_price": float(item.get("lastPrice", 0)),
        }

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _ws_connect(self, url: str, stream_id: str, topics: List[str], is_private: bool = False) -> None:
        attempt = 0
        while True:
            try:
                async with self._session.ws_connect(url, heartbeat=20) as ws:
                    self._ws_connections[stream_id] = ws
                    attempt = 0
                    self._health.set_ws_status(True, len(self._ws_connections))

                    # Authenticate for private channels
                    if is_private and self.config.api_key:
                        await self._ws_authenticate(ws)

                    # Subscribe to topics
                    sub_msg = {"op": "subscribe", "args": topics}
                    await ws.send_json(sub_msg)
                    self._logger.info("WS subscribed to %s", topics)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                self._handle_ws_message(stream_id, data)
                            except json.JSONDecodeError:
                                pass
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE,
                                          aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                            break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._health.record_error(f"ws: {exc}")

            self._ws_connections.pop(stream_id, None)
            self._health.set_ws_status(bool(self._ws_connections), len(self._ws_connections))

            if not self.config.enable_ws_reconnect:
                break
            attempt += 1
            if attempt > self.config.ws_reconnect_max_attempts:
                break
            delay = min(self.config.ws_reconnect_base_delay * (2 ** (attempt - 1)), self.config.ws_reconnect_max_delay)
            await asyncio.sleep(delay)

    async def _ws_authenticate(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Authenticate a private WebSocket connection."""
        expires = int(time.time() * 1000) + 10000
        sign_str = f"GET/realtime{expires}"
        signature = hmac.new(
            self.config.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        auth_msg = {
            "op": "auth",
            "args": [self.config.api_key, expires, signature],
        }
        await ws.send_json(auth_msg)
        # Wait for auth response
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=5)
            if msg.type == aiohttp.WSMsgType.TEXT:
                resp = json.loads(msg.data)
                if resp.get("success", False):
                    self._logger.info("WS authentication successful")
                else:
                    self._logger.error("WS authentication failed: %s", resp)
        except asyncio.TimeoutError:
            self._logger.warning("WS auth response timeout")

    def _handle_ws_message(self, stream_id: str, data: Dict[str, Any]) -> None:
        self._health.record_heartbeat()

        # Skip ping/pong and subscription confirmations
        op = data.get("op", "")
        if op in ("pong", "subscribe", "auth"):
            return

        topic = data.get("topic", "")
        msg_data = data.get("data", data)

        if not topic:
            return

        # Market data topics
        if topic.startswith("tickers."):
            if isinstance(msg_data, dict):
                ticker = self._parse_ticker(msg_data)
                symbol = topic.split(".")[1] if "." in topic else ""
                self._dispatch_callback(f"ticker:{symbol}", ticker)

        elif topic.startswith("orderbook."):
            parts = topic.split(".")
            symbol = parts[1] if len(parts) > 1 else ""
            if isinstance(msg_data, dict):
                book = UnifiedOrderbook(
                    symbol=symbol,
                    exchange=self.exchange_name,
                    bids=[OrderbookLevel(float(p), float(q)) for p, q in msg_data.get("b", [])],
                    asks=[OrderbookLevel(float(p), float(q)) for p, q in msg_data.get("a", [])],
                    timestamp=int(data.get("ts", current_timestamp_ms())),
                    last_update_id=int(data.get("u", 0)),
                    raw=data,
                )
                self._dispatch_callback(f"depth:{symbol}", book)

        elif topic.startswith("publicTrade."):
            symbol = topic.split(".")[1] if "." in topic else ""
            if isinstance(msg_data, list):
                for t in msg_data:
                    trade = UnifiedTrade(
                        symbol=symbol,
                        exchange=self.exchange_name,
                        trade_id=str(t.get("i", "")),
                        price=float(t.get("p", 0)),
                        quantity=float(t.get("v", 0)),
                        quote_quantity=float(t.get("p", 0)) * float(t.get("v", 0)),
                        side=OrderSide.BUY if t.get("S", "") == "Buy" else OrderSide.SELL,
                        timestamp=int(t.get("T", current_timestamp_ms())),
                        raw=t,
                    )
                    self._dispatch_callback(f"trade:{symbol}", trade)

        elif topic.startswith("kline."):
            parts = topic.split(".")
            interval = parts[1] if len(parts) > 1 else ""
            symbol = parts[2] if len(parts) > 2 else ""
            if isinstance(msg_data, list):
                for k in msg_data:
                    kline = UnifiedKline(
                        symbol=symbol,
                        exchange=self.exchange_name,
                        interval=interval,
                        open_time=int(k.get("start", 0)),
                        close_time=int(k.get("end", 0)),
                        open=float(k.get("open", 0)),
                        high=float(k.get("high", 0)),
                        low=float(k.get("low", 0)),
                        close=float(k.get("close", 0)),
                        volume=float(k.get("volume", 0)),
                        quote_volume=float(k.get("turnover", 0)),
                        trades=0,
                        is_closed=k.get("confirm", False),
                        raw=k,
                    )
                    self._dispatch_callback(f"kline:{symbol}:{interval}", kline)

        # Private topics
        elif topic == "order":
            if isinstance(msg_data, list):
                for o in msg_data:
                    order = self._parse_order_detail(o)
                    self._dispatch_callback("user:order", order)

        elif topic == "position":
            if isinstance(msg_data, list):
                for p in msg_data:
                    size = float(p.get("size", 0))
                    side_str = p.get("side", "")
                    pos_side = PositionSide.LONG if side_str == "Buy" else (
                        PositionSide.SHORT if side_str == "Sell" else PositionSide.BOTH
                    )
                    position = UnifiedPosition(
                        symbol=p.get("symbol", ""),
                        exchange=self.exchange_name,
                        side=pos_side,
                        size=size,
                        entry_price=float(p.get("entryPrice", 0)),
                        mark_price=float(p.get("markPrice", 0)),
                        liquidation_price=float(p.get("liqPrice", 0)),
                        unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                        realized_pnl=float(p.get("cumRealisedPnl", 0)),
                        leverage=int(float(p.get("leverage", 1))),
                        margin_mode=MarginMode.ISOLATED if p.get("tradeMode", 0) == 1 else MarginMode.CROSS,
                        margin=float(p.get("positionIM", 0)),
                        notional=float(p.get("positionValue", 0)),
                        adl_quantile=float(p.get("adlRankIndicator", 0)),
                        timestamp=int(p.get("updatedTime", current_timestamp_ms())),
                        raw=p,
                    )
                    self._dispatch_callback("user:position", position)

        elif topic == "wallet":
            if isinstance(msg_data, list):
                for account in msg_data:
                    for coin in account.get("coin", []):
                        balance = UnifiedBalance(
                            asset=coin.get("coin", ""),
                            exchange=self.exchange_name,
                            free=float(coin.get("availableToWithdraw", 0)),
                            locked=float(coin.get("locked", 0)),
                            total=float(coin.get("walletBalance", 0)),
                            usd_value=float(coin.get("usdValue", 0)),
                            raw=coin,
                        )
                        self._dispatch_callback("user:balance", balance)

    # -- Subscription methods --

    def _next_sub_id(self) -> str:
        self._sub_counter += 1
        return f"bsub_{self._sub_counter}_{uuid.uuid4().hex[:8]}"

    async def subscribe_ticker(self, symbol: str, callback: Callable[[UnifiedTicker], Any]) -> str:
        sub_id = self._next_sub_id()
        topic = f"tickers.{symbol}"
        self._register_callback(f"ticker:{symbol}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, [topic]))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "ticker", "symbol": symbol, "topic": topic}
        return sub_id

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[UnifiedOrderbook], Any], depth: int = 20
    ) -> str:
        sub_id = self._next_sub_id()
        # Bybit supports depth levels: 1, 50, 200, 500
        book_depth = 50 if depth <= 50 else (200 if depth <= 200 else 500)
        topic = f"orderbook.{book_depth}.{symbol}"
        self._register_callback(f"depth:{symbol}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, [topic]))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "depth", "symbol": symbol, "topic": topic}
        return sub_id

    async def subscribe_trades(self, symbol: str, callback: Callable[[UnifiedTrade], Any]) -> str:
        sub_id = self._next_sub_id()
        topic = f"publicTrade.{symbol}"
        self._register_callback(f"trade:{symbol}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, [topic]))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "trade", "symbol": symbol, "topic": topic}
        return sub_id

    async def subscribe_klines(
        self, symbol: str, interval: str, callback: Callable[[UnifiedKline], Any]
    ) -> str:
        sub_id = self._next_sub_id()
        bybit_interval = self._convert_interval(interval)
        topic = f"kline.{bybit_interval}.{symbol}"
        self._register_callback(f"kline:{symbol}:{interval}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, [topic]))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "kline", "symbol": symbol, "topic": topic}
        return sub_id

    async def subscribe_user_data(
        self,
        on_order: Optional[Callable[[UnifiedOrder], Any]] = None,
        on_position: Optional[Callable[[UnifiedPosition], Any]] = None,
        on_balance: Optional[Callable[[UnifiedBalance], Any]] = None,
    ) -> str:
        if not self.config.api_key:
            raise AuthenticationError("API key required", exchange=self.exchange_name)

        sub_id = self._next_sub_id()
        topics = []
        if on_order:
            self._register_callback("user:order", on_order)
            topics.append("order")
        if on_position:
            self._register_callback("user:position", on_position)
            topics.append("position")
        if on_balance:
            self._register_callback("user:balance", on_balance)
            topics.append("wallet")

        if not topics:
            topics = ["order", "position", "wallet"]

        task = asyncio.create_task(
            self._ws_connect(self._ws_private_url, sub_id, topics, is_private=True)
        )
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "user_data", "topics": topics}
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        sub = self._subscriptions.pop(subscription_id, None)
        if not sub:
            return False
        task = self._ws_tasks.pop(subscription_id, None)
        if task and not task.done():
            task.cancel()
        ws = self._ws_connections.pop(subscription_id, None)
        if ws and not ws.closed:
            await ws.close()
        return True

    @property
    def active_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._subscriptions)

    @property
    def category(self) -> str:
        return self._category
