# mefai-autotrade - Binance USDT-M Futures exchange connector
# Full REST + WebSocket implementation for Binance Futures trading

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
    PositionSide,
    MarginMode,
    TimeInForce,
    current_timestamp_ms,
    retry_async,
)

logger = logging.getLogger(__name__)

# Binance Futures base URLs
FUTURES_REST = "https://fapi.binance.com"
FUTURES_REST_TESTNET = "https://testnet.binancefuture.com"
FUTURES_WS = "wss://fstream.binance.com/ws"
FUTURES_WS_TESTNET = "wss://stream.binancefuture.com/ws"

# Order type mapping
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_MARKET",
    OrderType.STOP_LIMIT: "STOP",
    OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
    OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT",
    OrderType.TRAILING_STOP_MARKET: "TRAILING_STOP_MARKET",
    OrderType.LIMIT_MAKER: "LIMIT",
}

_ORDER_TYPE_REV = {
    "MARKET": OrderType.MARKET,
    "LIMIT": OrderType.LIMIT,
    "STOP": OrderType.STOP_LIMIT,
    "STOP_MARKET": OrderType.STOP_MARKET,
    "TAKE_PROFIT": OrderType.TAKE_PROFIT_LIMIT,
    "TAKE_PROFIT_MARKET": OrderType.TAKE_PROFIT_MARKET,
    "TRAILING_STOP_MARKET": OrderType.TRAILING_STOP_MARKET,
}

_ORDER_STATUS_MAP = {
    "NEW": OrderStatus.NEW,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
    "NEW_INSURANCE": OrderStatus.NEW,
    "NEW_ADL": OrderStatus.NEW,
}

_TIF_MAP = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.GTX: "GTX",
    TimeInForce.GTD: "GTD",
}

# Meme coin prefix mapping for Binance Futures
MEME_COIN_PREFIXES = {
    "PEPE": "1000PEPE",
    "FLOKI": "1000FLOKI",
    "SHIB": "1000SHIB",
    "BONK": "1000BONK",
    "LUNC": "1000LUNC",
    "SATS": "1000SATS",
    "RATS": "1000RATS",
}


class BinanceFuturesExchange(ExchangeBase):
    """Binance USDT-M Futures exchange connector.

    Full implementation supporting all futures order types, position management,
    leverage/margin control, funding rates, and real-time WebSocket streaming
    including liquidation feeds and user data.
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__(config, "binance_futures")
        self._base_url = config.base_url_override or (
            FUTURES_REST_TESTNET if config.testnet else FUTURES_REST
        )
        self._ws_base_url = config.ws_url_override or (
            FUTURES_WS_TESTNET if config.testnet else FUTURES_WS
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._ws_tasks: Dict[str, asyncio.Task] = {}
        self._listen_key: Optional[str] = None
        self._listen_key_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._sub_counter = 0
        self._symbol_info: Dict[str, Dict[str, Any]] = {}
        self._server_time_offset: int = 0
        self._position_mode_dual: bool = False  # False = one-way mode, True = hedge mode

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
        await self._sync_server_time()
        await self._load_exchange_info()
        if self.config.api_key:
            await self._load_position_mode()
        self._initialized = True
        self._logger.info(
            "Binance Futures connector initialized (testnet=%s, dual_position=%s)",
            self.config.testnet, self._position_mode_dual,
        )

    async def close(self) -> None:
        if self._listen_key_task and not self._listen_key_task.done():
            self._listen_key_task.cancel()
        if self._listen_key:
            try:
                await self._close_listen_key()
            except Exception:
                pass
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
        self._logger.info("Binance Futures connector closed")

    # ------------------------------------------------------------------
    # Internal HTTP
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

        if signed:
            params["timestamp"] = current_timestamp_ms() + self._server_time_offset
            params["recvWindow"] = self.config.recv_window
            query = self._build_query(params)
            params["signature"] = self._sign_hmac_sha256(query)

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
            raise NetworkError(f"Network error: {exc}", exchange=self.exchange_name) from exc

    def _handle_error(self, data: Dict[str, Any], status: int) -> None:
        code = data.get("code", 0)
        msg = data.get("msg", f"HTTP {status}")

        if code in (-2015, -2014, -2008):
            raise AuthenticationError(msg, code=code, exchange=self.exchange_name)
        elif code in (-2019, -2010):
            raise InsufficientBalanceError(msg, code=code, exchange=self.exchange_name)
        elif code == -2011:
            raise OrderNotFoundError(msg, code=code, exchange=self.exchange_name)
        elif code in (-4061, -4062, -4063, -4164, -1102, -1116):
            raise InvalidOrderError(msg, code=code, exchange=self.exchange_name)
        elif code == -1003:
            raise RateLimitError(msg, code=code, exchange=self.exchange_name)
        else:
            raise ExchangeError(msg, code=code, exchange=self.exchange_name)

    async def _sync_server_time(self) -> None:
        try:
            local_ts = current_timestamp_ms()
            data = await self._request("GET", "/fapi/v1/time", weight=1)
            self._server_time_offset = data["serverTime"] - local_ts
        except Exception as exc:
            self._logger.warning("Failed to sync server time: %s", exc)
            self._server_time_offset = 0

    async def _load_exchange_info(self) -> None:
        data = await self._request("GET", "/fapi/v1/exchangeInfo", weight=1)
        for sym_info in data.get("symbols", []):
            symbol = sym_info["symbol"]
            filters = {}
            for f in sym_info.get("filters", []):
                filters[f["filterType"]] = f

            self._symbol_info[symbol] = {
                "status": sym_info.get("status", ""),
                "pair": sym_info.get("pair", ""),
                "contract_type": sym_info.get("contractType", ""),
                "delivery_date": sym_info.get("deliveryDate", 0),
                "onboard_date": sym_info.get("onboardDate", 0),
                "base_asset": sym_info.get("baseAsset", ""),
                "quote_asset": sym_info.get("quoteAsset", ""),
                "margin_asset": sym_info.get("marginAsset", ""),
                "price_precision": sym_info.get("pricePrecision", 2),
                "quantity_precision": sym_info.get("quantityPrecision", 3),
                "base_precision": sym_info.get("baseAssetPrecision", 8),
                "quote_precision": sym_info.get("quotePrecision", 8),
                "order_types": sym_info.get("orderTypes", []),
                "time_in_force": sym_info.get("timeInForce", []),
                "filters": filters,
                "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", "0.01")),
                "step_size": float(filters.get("LOT_SIZE", {}).get("stepSize", "0.001")),
                "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", "0.001")),
                "max_qty": float(filters.get("LOT_SIZE", {}).get("maxQty", "1000")),
                "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("notional", "5")),
            }
        self._exchange_info_cache = data
        self._exchange_info_ts = time.monotonic()
        self._logger.info("Loaded futures exchange info for %d symbols", len(self._symbol_info))

    async def _fetch_exchange_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/fapi/v1/exchangeInfo", weight=1)

    async def _load_position_mode(self) -> None:
        """Load current position mode (one-way or hedge)."""
        try:
            data = await self._request("GET", "/fapi/v1/positionSide/dual", signed=True, weight=30)
            self._position_mode_dual = data.get("dualSidePosition", False)
        except Exception as exc:
            self._logger.warning("Failed to load position mode: %s", exc)

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def normalize_symbol(self, symbol: str) -> str:
        """Normalize a symbol, applying meme coin prefix if needed.

        Example: PEPEUSDT -> 1000PEPEUSDT
        """
        base = symbol.replace("USDT", "").replace("BUSD", "")
        if base in MEME_COIN_PREFIXES:
            prefix = MEME_COIN_PREFIXES[base]
            suffix = symbol[len(base):]
            return f"{prefix}{suffix}"
        return symbol

    def _adjust_quantity(self, symbol: str, quantity: float) -> str:
        info = self._symbol_info.get(symbol)
        if not info:
            return str(quantity)
        precision = info.get("quantity_precision", 3)
        factor = 10 ** precision
        adjusted = int(quantity * factor) / factor
        return f"{adjusted:.{precision}f}"

    def _adjust_price(self, symbol: str, price: float) -> str:
        info = self._symbol_info.get(symbol)
        if not info:
            return str(price)
        precision = info.get("price_precision", 2)
        factor = 10 ** precision
        adjusted = int(price * factor) / factor
        return f"{adjusted:.{precision}f}"

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> UnifiedTicker:
        data = await self._request("GET", "/fapi/v1/ticker/24hr", {"symbol": symbol}, weight=1)
        return self._parse_ticker(data)

    async def get_tickers(self) -> List[UnifiedTicker]:
        data = await self._request("GET", "/fapi/v1/ticker/24hr", weight=40)
        return [self._parse_ticker(t) for t in data]

    def _parse_ticker(self, data: Dict[str, Any]) -> UnifiedTicker:
        return UnifiedTicker(
            symbol=data["symbol"],
            exchange=self.exchange_name,
            last_price=float(data.get("lastPrice", 0)),
            bid_price=float(data.get("bidPrice", 0) if "bidPrice" in data else 0),
            ask_price=float(data.get("askPrice", 0) if "askPrice" in data else 0),
            bid_qty=float(data.get("bidQty", 0) if "bidQty" in data else 0),
            ask_qty=float(data.get("askQty", 0) if "askQty" in data else 0),
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
        valid_limits = [5, 10, 20, 50, 100, 500, 1000]
        actual_limit = min(l for l in valid_limits if l >= limit) if limit <= 1000 else 1000
        weight = 2 if actual_limit <= 50 else (5 if actual_limit <= 100 else (10 if actual_limit <= 500 else 20))

        data = await self._request(
            "GET", "/fapi/v1/depth",
            {"symbol": symbol, "limit": actual_limit},
            weight=weight,
        )
        return UnifiedOrderbook(
            symbol=symbol,
            exchange=self.exchange_name,
            bids=[OrderbookLevel(float(p), float(q)) for p, q in data.get("bids", [])],
            asks=[OrderbookLevel(float(p), float(q)) for p, q in data.get("asks", [])],
            timestamp=int(data.get("T", current_timestamp_ms())),
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
            "limit": min(limit, 1500),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/fapi/v1/klines", params, weight=5)
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
            "GET", "/fapi/v1/trades",
            {"symbol": symbol, "limit": min(limit, 1000)},
            weight=5,
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

    async def get_agg_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get compressed/aggregate trades."""
        params: Dict[str, Any] = {"symbol": symbol, "limit": min(limit, 1000)}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("GET", "/fapi/v1/aggTrades", params, weight=20)

    # ------------------------------------------------------------------
    # Account and positions
    # ------------------------------------------------------------------

    async def get_balance(self, asset: Optional[str] = None) -> List[UnifiedBalance]:
        data = await self._request("GET", "/fapi/v2/balance", signed=True, weight=5)
        balances = []
        for b in data:
            if asset and b["asset"] != asset:
                continue
            balance = float(b.get("balance", 0))
            available = float(b.get("availableBalance", 0))
            locked = balance - available
            if balance == 0 and asset is None:
                continue
            balances.append(UnifiedBalance(
                asset=b["asset"],
                exchange=self.exchange_name,
                free=available,
                locked=max(locked, 0),
                total=balance,
                usd_value=float(b.get("balance", 0)),  # USDT-M is already in USD terms
                raw=b,
            ))
        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[UnifiedPosition]:
        data = await self._request("GET", "/fapi/v2/positionRisk", signed=True, weight=5)
        positions = []
        for p in data:
            size = float(p.get("positionAmt", 0))
            if size == 0 and symbol is None:
                continue
            if symbol and p["symbol"] != symbol:
                continue

            pos_side_str = p.get("positionSide", "BOTH")
            if pos_side_str == "LONG":
                pos_side = PositionSide.LONG
            elif pos_side_str == "SHORT":
                pos_side = PositionSide.SHORT
            else:
                pos_side = PositionSide.BOTH

            entry_price = float(p.get("entryPrice", 0))
            mark_price = float(p.get("markPrice", 0))
            liq_price = float(p.get("liquidationPrice", 0))
            leverage = int(p.get("leverage", 1))
            margin_type = p.get("marginType", "cross")
            unrealized_pnl = float(p.get("unRealizedProfit", 0))
            notional = float(p.get("notional", abs(size) * mark_price))

            positions.append(UnifiedPosition(
                symbol=p["symbol"],
                exchange=self.exchange_name,
                side=pos_side,
                size=abs(size),
                entry_price=entry_price,
                mark_price=mark_price,
                liquidation_price=liq_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0.0,
                leverage=leverage,
                margin_mode=MarginMode.ISOLATED if margin_type == "isolated" else MarginMode.CROSS,
                margin=float(p.get("isolatedMargin", 0)) if margin_type == "isolated" else 0,
                notional=abs(notional),
                adl_quantile=float(p.get("adlQuantile", 0)),
                timestamp=int(p.get("updateTime", current_timestamp_ms())),
                raw=p,
            ))
        return positions

    async def get_account_info(self) -> UnifiedAccountInfo:
        data = await self._request("GET", "/fapi/v2/account", signed=True, weight=5)

        balances = []
        for a in data.get("assets", []):
            balance = float(a.get("walletBalance", 0))
            available = float(a.get("availableBalance", 0))
            if balance == 0:
                continue
            balances.append(UnifiedBalance(
                asset=a["asset"],
                exchange=self.exchange_name,
                free=available,
                locked=max(balance - available, 0),
                total=balance,
                usd_value=balance,
                raw=a,
            ))

        positions = []
        for p in data.get("positions", []):
            size = float(p.get("positionAmt", 0))
            if size == 0:
                continue
            pos_side_str = p.get("positionSide", "BOTH")
            if pos_side_str == "LONG":
                pos_side = PositionSide.LONG
            elif pos_side_str == "SHORT":
                pos_side = PositionSide.SHORT
            else:
                pos_side = PositionSide.BOTH

            positions.append(UnifiedPosition(
                symbol=p.get("symbol", ""),
                exchange=self.exchange_name,
                side=pos_side,
                size=abs(size),
                entry_price=float(p.get("entryPrice", 0)),
                mark_price=0,
                liquidation_price=0,
                unrealized_pnl=float(p.get("unrealizedProfit", 0)),
                realized_pnl=0,
                leverage=int(p.get("leverage", 1)),
                margin_mode=MarginMode.ISOLATED if p.get("isolated", False) else MarginMode.CROSS,
                margin=float(p.get("isolatedWallet", 0)),
                notional=float(p.get("notional", 0)),
                adl_quantile=0,
                timestamp=int(p.get("updateTime", 0)),
                raw=p,
            ))

        total_margin = float(data.get("totalMarginBalance", 0))
        available_margin = float(data.get("availableBalance", 0))
        total_unrealized = float(data.get("totalUnrealizedProfit", 0))
        total_wallet = float(data.get("totalWalletBalance", 0))
        maint_margin = float(data.get("totalMaintMargin", 0))
        margin_ratio = (maint_margin / total_margin * 100) if total_margin > 0 else 0

        return UnifiedAccountInfo(
            exchange=self.exchange_name,
            account_type="FUTURES",
            can_trade=data.get("canTrade", False),
            can_withdraw=data.get("canWithdraw", False),
            can_deposit=data.get("canDeposit", False),
            balances=balances,
            total_usd_value=total_wallet,
            total_margin=total_margin,
            available_margin=available_margin,
            total_unrealized_pnl=total_unrealized,
            margin_ratio=margin_ratio,
            positions=positions,
            timestamp=int(data.get("updateTime", current_timestamp_ms())),
            raw=data,
        )

    async def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed position risk information."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v2/positionRisk", params, signed=True, weight=5)

    async def get_adl_quantile(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ADL (Auto-Deleveraging) quantile estimation."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/adlQuantile", params, signed=True, weight=5)

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
        binance_type = _ORDER_TYPE_MAP.get(order_type, order_type.value)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": binance_type,
            "quantity": self._adjust_quantity(symbol, quantity),
        }

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        # Position side for hedge mode
        if self._position_mode_dual and position_side:
            params["positionSide"] = position_side.value
        elif self._position_mode_dual:
            # Default: BUY->LONG, SELL->SHORT
            params["positionSide"] = "LONG" if side == OrderSide.BUY else "SHORT"

        # Price for limit orders
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT, OrderType.LIMIT_MAKER):
            if price is None:
                raise InvalidOrderError(
                    f"Price required for {order_type.value}",
                    exchange=self.exchange_name,
                )
            params["price"] = self._adjust_price(symbol, price)

        # Time in force
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            if post_only:
                params["timeInForce"] = "GTX"
            else:
                params["timeInForce"] = _TIF_MAP.get(time_in_force, "GTC")

        # Stop/trigger price
        if stop_price and order_type in (
            OrderType.STOP_MARKET, OrderType.STOP_LIMIT,
            OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT,
        ):
            params["stopPrice"] = self._adjust_price(symbol, stop_price)

        # Trailing stop callback rate
        if order_type == OrderType.TRAILING_STOP_MARKET:
            if callback_rate is None:
                raise InvalidOrderError(
                    "callback_rate required for TRAILING_STOP_MARKET",
                    exchange=self.exchange_name,
                )
            params["callbackRate"] = str(callback_rate)
            if stop_price:
                params["activationPrice"] = self._adjust_price(symbol, stop_price)

        # Reduce only (not valid in hedge mode)
        if reduce_only and not self._position_mode_dual:
            params["reduceOnly"] = "true"

        # Working type
        working_type = kwargs.get("working_type")
        if working_type:
            params["workingType"] = working_type

        # Close position flag
        close_position = kwargs.get("close_position", False)
        if close_position:
            params["closePosition"] = "true"
            params.pop("quantity", None)

        params["newOrderRespType"] = "RESULT"

        data = await self._request("POST", "/fapi/v1/order", params, signed=True, weight=1)
        return self._parse_order(data)

    async def place_batch_orders(self, orders: List[Dict[str, Any]]) -> List[UnifiedOrder]:
        """Place up to 5 orders in a single request.

        Each order dict should contain: symbol, side, type, quantity, etc.
        """
        if len(orders) > 5:
            raise InvalidOrderError("Max 5 batch orders allowed", exchange=self.exchange_name)

        batch = []
        for o in orders:
            params: Dict[str, Any] = {
                "symbol": o["symbol"],
                "side": o["side"],
                "type": o["type"],
            }
            if "quantity" in o:
                params["quantity"] = o["quantity"]
            if "price" in o:
                params["price"] = o["price"]
            if "stopPrice" in o:
                params["stopPrice"] = o["stopPrice"]
            if "timeInForce" in o:
                params["timeInForce"] = o["timeInForce"]
            if "reduceOnly" in o:
                params["reduceOnly"] = str(o["reduceOnly"]).lower()
            if "positionSide" in o:
                params["positionSide"] = o["positionSide"]
            if "newClientOrderId" in o:
                params["newClientOrderId"] = o["newClientOrderId"]
            batch.append(params)

        data = await self._request(
            "POST", "/fapi/v1/batchOrders",
            {"batchOrders": json.dumps(batch)},
            signed=True, weight=5,
        )
        results = []
        for item in data:
            if "code" in item and item["code"] != 200:
                self._logger.error("Batch order failed: %s", item)
                continue
            results.append(self._parse_order(item))
        return results

    async def cancel_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        params = {"symbol": symbol, "orderId": int(order_id)}
        data = await self._request("DELETE", "/fapi/v1/order", params, signed=True, weight=1)
        return self._parse_order(data)

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        await self._request(
            "DELETE", "/fapi/v1/allOpenOrders",
            {"symbol": symbol},
            signed=True, weight=1,
        )
        return True

    async def modify_order(
        self,
        symbol: str,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> UnifiedOrder:
        """Modify an existing order (Binance futures supports direct modification)."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "orderId": int(order_id),
        }
        if quantity:
            params["quantity"] = self._adjust_quantity(symbol, quantity)
        if price:
            params["price"] = self._adjust_price(symbol, price)

        params["side"] = "BUY"  # Required but doesn't change
        params["timestamp"] = current_timestamp_ms() + self._server_time_offset

        data = await self._request("PUT", "/fapi/v1/order", params, signed=True, weight=1)
        return self._parse_order(data)

    async def get_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        params = {"symbol": symbol, "orderId": int(order_id)}
        data = await self._request("GET", "/fapi/v1/order", params, signed=True, weight=1)
        return self._parse_order(data)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {}
        weight = 1
        if symbol:
            params["symbol"] = symbol
        else:
            weight = 40
        data = await self._request("GET", "/fapi/v1/openOrders", params, signed=True, weight=weight)
        return [self._parse_order(o) for o in data]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/fapi/v1/allOrders", params, signed=True, weight=5)
        return [self._parse_order(o) for o in data]

    def _parse_order(self, data: Dict[str, Any]) -> UnifiedOrder:
        raw_type = data.get("type", data.get("origType", "MARKET"))
        raw_status = data.get("status", "NEW")
        orig_qty = float(data.get("origQty", 0))
        executed_qty = float(data.get("executedQty", 0))
        cum_quote = float(data.get("cumQuote", 0))

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
                cum_quote / executed_qty if executed_qty > 0 else 0
            ),
            commission=float(data.get("commission", 0)),
            commission_asset=data.get("commissionAsset", ""),
            time_in_force=TimeInForce.GTC,
            reduce_only=data.get("reduceOnly", False),
            post_only=data.get("timeInForce", "") == "GTX",
            created_at=int(data.get("time", data.get("updateTime", 0))),
            updated_at=int(data.get("updateTime", 0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Futures specific - Leverage, margin, funding
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            await self._request(
                "POST", "/fapi/v1/leverage",
                {"symbol": symbol, "leverage": leverage},
                signed=True, weight=1,
            )
            self._logger.info("Set leverage for %s to %dx", symbol, leverage)
            return True
        except ExchangeError as exc:
            self._logger.error("Failed to set leverage: %s", exc)
            return False

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        margin_type = "ISOLATED" if mode == MarginMode.ISOLATED else "CROSSED"
        try:
            await self._request(
                "POST", "/fapi/v1/marginType",
                {"symbol": symbol, "marginType": margin_type},
                signed=True, weight=1,
            )
            self._logger.info("Set margin mode for %s to %s", symbol, mode.value)
            return True
        except ExchangeError as exc:
            # -4046 means already in that mode - not an error
            if hasattr(exc, "code") and exc.code == -4046:
                return True
            self._logger.error("Failed to set margin mode: %s", exc)
            return False

    async def set_position_mode(self, dual_side: bool) -> bool:
        """Set position mode: True for hedge mode, False for one-way mode."""
        try:
            await self._request(
                "POST", "/fapi/v1/positionSide/dual",
                {"dualSidePosition": str(dual_side).lower()},
                signed=True, weight=1,
            )
            self._position_mode_dual = dual_side
            self._logger.info("Set position mode to %s", "hedge" if dual_side else "one-way")
            return True
        except ExchangeError as exc:
            # -4059 means already in that mode
            if hasattr(exc, "code") and exc.code == -4059:
                self._position_mode_dual = dual_side
                return True
            self._logger.error("Failed to set position mode: %s", exc)
            return False

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        data = await self._request(
            "GET", "/fapi/v1/premiumIndex",
            {"symbol": symbol},
            weight=1,
        )
        return {
            "symbol": data.get("symbol", symbol),
            "mark_price": float(data.get("markPrice", 0)),
            "index_price": float(data.get("indexPrice", 0)),
            "last_funding_rate": float(data.get("lastFundingRate", 0)),
            "next_funding_time": int(data.get("nextFundingTime", 0)),
            "interest_rate": float(data.get("interestRate", 0)),
            "timestamp": int(data.get("time", current_timestamp_ms())),
        }

    async def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get historical funding rate data."""
        params: Dict[str, Any] = {
            "symbol": symbol,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("GET", "/fapi/v1/fundingRate", params, weight=1)

    async def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        data = await self._request(
            "GET", "/fapi/v1/premiumIndex",
            {"symbol": symbol},
            weight=1,
        )
        return {
            "symbol": data.get("symbol", symbol),
            "mark_price": float(data.get("markPrice", 0)),
            "index_price": float(data.get("indexPrice", 0)),
            "estimated_settle_price": float(data.get("estimatedSettlePrice", 0)),
            "last_funding_rate": float(data.get("lastFundingRate", 0)),
            "next_funding_time": int(data.get("nextFundingTime", 0)),
        }

    async def get_income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get income history (realized PnL, funding, commission, etc).

        income_type options: REALIZED_PNL, FUNDING_FEE, COMMISSION, INSURANCE_CLEAR,
                             TRANSFER, WELCOME_BONUS, API_REBATE, etc.
        """
        params: Dict[str, Any] = {"limit": min(limit, 1000)}
        if symbol:
            params["symbol"] = symbol
        if income_type:
            params["incomeType"] = income_type
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return await self._request("GET", "/fapi/v1/income", params, signed=True, weight=30)

    async def get_force_orders(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        auto_close_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get liquidation/ADL order history.

        auto_close_type: LIQUIDATION or ADL
        """
        params: Dict[str, Any] = {"limit": min(limit, 100)}
        if symbol:
            params["symbol"] = symbol
        if auto_close_type:
            params["autoCloseType"] = auto_close_type

        return await self._request("GET", "/fapi/v1/forceOrders", params, signed=True, weight=20)

    async def get_commission_rate(self, symbol: str) -> Dict[str, Any]:
        """Get current commission rate for a symbol."""
        data = await self._request(
            "GET", "/fapi/v1/commissionRate",
            {"symbol": symbol},
            signed=True, weight=20,
        )
        return {
            "symbol": data.get("symbol", symbol),
            "maker_rate": float(data.get("makerCommissionRate", 0)),
            "taker_rate": float(data.get("takerCommissionRate", 0)),
        }

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _ws_connect(self, url: str, stream_id: str) -> None:
        attempt = 0
        while True:
            try:
                self._logger.info("WS connecting: %s (attempt %d)", stream_id, attempt + 1)
                async with self._session.ws_connect(
                    url,
                    heartbeat=self.config.ws_ping_interval,
                    timeout=self.config.ws_ping_timeout,
                ) as ws:
                    self._ws_connections[stream_id] = ws
                    attempt = 0
                    self._health.set_ws_status(True, len(self._ws_connections))
                    self._logger.info("WS connected: %s", stream_id)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                self._handle_ws_message(stream_id, data)
                            except json.JSONDecodeError:
                                pass
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                            break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._health.record_error(f"ws_{stream_id}: {exc}")

            self._ws_connections.pop(stream_id, None)
            self._health.set_ws_status(bool(self._ws_connections), len(self._ws_connections))

            if not self.config.enable_ws_reconnect:
                break
            attempt += 1
            if attempt > self.config.ws_reconnect_max_attempts:
                break
            delay = min(
                self.config.ws_reconnect_base_delay * (2 ** (attempt - 1)),
                self.config.ws_reconnect_max_delay,
            )
            await asyncio.sleep(delay)

    def _handle_ws_message(self, stream_id: str, data: Dict[str, Any]) -> None:
        event_type = data.get("e", "")
        self._health.record_heartbeat()

        if event_type == "24hrTicker" or event_type == "24hrMiniTicker":
            ticker = self._parse_ticker(data)
            self._dispatch_callback(f"ticker:{data.get('s', '')}", ticker)

        elif event_type == "depthUpdate":
            book = UnifiedOrderbook(
                symbol=data.get("s", ""),
                exchange=self.exchange_name,
                bids=[OrderbookLevel(float(p), float(q)) for p, q in data.get("b", [])],
                asks=[OrderbookLevel(float(p), float(q)) for p, q in data.get("a", [])],
                timestamp=int(data.get("E", current_timestamp_ms())),
                last_update_id=data.get("u"),
                raw=data,
            )
            self._dispatch_callback(f"depth:{data.get('s', '')}", book)

        elif event_type == "aggTrade":
            trade = UnifiedTrade(
                symbol=data.get("s", ""),
                exchange=self.exchange_name,
                trade_id=str(data.get("a", "")),
                price=float(data.get("p", 0)),
                quantity=float(data.get("q", 0)),
                quote_quantity=float(data.get("p", 0)) * float(data.get("q", 0)),
                side=OrderSide.SELL if data.get("m", False) else OrderSide.BUY,
                timestamp=int(data.get("T", current_timestamp_ms())),
                is_maker=data.get("m", False),
                raw=data,
            )
            self._dispatch_callback(f"trade:{data.get('s', '')}", trade)

        elif event_type == "kline":
            k = data.get("k", {})
            kline = UnifiedKline(
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
            self._dispatch_callback(f"kline:{k.get('s', '')}:{k.get('i', '')}", kline)

        elif event_type == "markPriceUpdate":
            self._dispatch_callback(f"markPrice:{data.get('s', '')}", data)

        elif event_type == "forceOrder":
            # Liquidation event
            o = data.get("o", {})
            self._dispatch_callback("liquidation", {
                "symbol": o.get("s", ""),
                "side": o.get("S", ""),
                "order_type": o.get("o", ""),
                "quantity": float(o.get("q", 0)),
                "price": float(o.get("p", 0)),
                "avg_price": float(o.get("ap", 0)),
                "status": o.get("X", ""),
                "time": int(o.get("T", 0)),
                "raw": data,
            })

        # User data events
        elif event_type == "ORDER_TRADE_UPDATE":
            order_data = data.get("o", {})
            order = self._parse_ws_order_update(order_data)
            self._dispatch_callback("user:order", order)

        elif event_type == "ACCOUNT_UPDATE":
            account_data = data.get("a", {})
            # Balance updates
            for b in account_data.get("B", []):
                balance = UnifiedBalance(
                    asset=b.get("a", ""),
                    exchange=self.exchange_name,
                    free=float(b.get("cw", 0)),
                    locked=0,
                    total=float(b.get("wb", 0)),
                    usd_value=float(b.get("wb", 0)),
                    raw=b,
                )
                self._dispatch_callback("user:balance", balance)
            # Position updates
            for p in account_data.get("P", []):
                size = float(p.get("pa", 0))
                pos_side = p.get("ps", "BOTH")
                position = UnifiedPosition(
                    symbol=p.get("s", ""),
                    exchange=self.exchange_name,
                    side=PositionSide.LONG if pos_side == "LONG" else (
                        PositionSide.SHORT if pos_side == "SHORT" else PositionSide.BOTH
                    ),
                    size=abs(size),
                    entry_price=float(p.get("ep", 0)),
                    mark_price=0,
                    liquidation_price=0,
                    unrealized_pnl=float(p.get("up", 0)),
                    realized_pnl=float(p.get("cr", 0)),
                    leverage=0,
                    margin_mode=MarginMode.ISOLATED if p.get("mt", "") == "isolated" else MarginMode.CROSS,
                    margin=float(p.get("iw", 0)),
                    notional=0,
                    adl_quantile=0,
                    timestamp=int(data.get("E", current_timestamp_ms())),
                    raw=p,
                )
                self._dispatch_callback("user:position", position)

        elif event_type == "MARGIN_CALL":
            self._dispatch_callback("user:margin_call", data)

        elif event_type == "ACCOUNT_CONFIG_UPDATE":
            self._dispatch_callback("user:config", data)

        # Combined stream format
        elif "stream" in data and "data" in data:
            self._handle_ws_message(stream_id, data["data"])

    def _parse_ws_order_update(self, data: Dict[str, Any]) -> UnifiedOrder:
        raw_type = data.get("o", "MARKET")
        raw_status = data.get("X", "NEW")
        orig_qty = float(data.get("q", 0))
        filled_qty = float(data.get("z", 0))

        return UnifiedOrder(
            symbol=data.get("s", ""),
            exchange=self.exchange_name,
            order_id=str(data.get("i", "")),
            client_order_id=data.get("c", ""),
            side=OrderSide(data.get("S", "BUY")),
            order_type=_ORDER_TYPE_REV.get(raw_type, OrderType.MARKET),
            status=_ORDER_STATUS_MAP.get(raw_status, OrderStatus.NEW),
            price=float(data.get("p", 0)),
            stop_price=float(data.get("sp", 0)),
            quantity=orig_qty,
            filled_quantity=filled_qty,
            remaining_quantity=orig_qty - filled_qty,
            average_price=float(data.get("ap", 0)),
            commission=float(data.get("n", 0)),
            commission_asset=data.get("N", ""),
            reduce_only=data.get("R", False),
            created_at=int(data.get("T", 0)),
            updated_at=int(data.get("T", 0)),
            raw=data,
        )

    # -- Listen key management --

    async def _create_listen_key(self) -> str:
        data = await self._request("POST", "/fapi/v1/listenKey", weight=1)
        return data["listenKey"]

    async def _keepalive_listen_key(self) -> None:
        if not self._listen_key:
            return
        await self._request("PUT", "/fapi/v1/listenKey", weight=1)

    async def _close_listen_key(self) -> None:
        if not self._listen_key:
            return
        await self._request("DELETE", "/fapi/v1/listenKey", weight=1)
        self._listen_key = None

    async def _listen_key_keepalive_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                await self._keepalive_listen_key()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.error("Listen key keepalive failed: %s", exc)
                try:
                    self._listen_key = await self._create_listen_key()
                except Exception:
                    pass

    # -- Subscription methods --

    def _next_sub_id(self) -> str:
        self._sub_counter += 1
        return f"fsub_{self._sub_counter}_{uuid.uuid4().hex[:8]}"

    async def subscribe_ticker(self, symbol: str, callback: Callable[[UnifiedTicker], Any]) -> str:
        sub_id = self._next_sub_id()
        stream = f"{symbol.lower()}@ticker"
        self._register_callback(f"ticker:{symbol}", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "ticker", "symbol": symbol, "key": f"ticker:{symbol}"}
        return sub_id

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[UnifiedOrderbook], Any], depth: int = 20
    ) -> str:
        sub_id = self._next_sub_id()
        stream = f"{symbol.lower()}@depth@100ms"
        self._register_callback(f"depth:{symbol}", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "depth", "symbol": symbol, "key": f"depth:{symbol}"}
        return sub_id

    async def subscribe_trades(self, symbol: str, callback: Callable[[UnifiedTrade], Any]) -> str:
        sub_id = self._next_sub_id()
        stream = f"{symbol.lower()}@aggTrade"
        self._register_callback(f"trade:{symbol}", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "trade", "symbol": symbol, "key": f"trade:{symbol}"}
        return sub_id

    async def subscribe_klines(
        self, symbol: str, interval: str, callback: Callable[[UnifiedKline], Any]
    ) -> str:
        sub_id = self._next_sub_id()
        stream = f"{symbol.lower()}@kline_{interval}"
        self._register_callback(f"kline:{symbol}:{interval}", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "kline", "symbol": symbol, "key": f"kline:{symbol}:{interval}"}
        return sub_id

    async def subscribe_mark_price(
        self, symbol: str, callback: Callable, update_speed: str = "1s"
    ) -> str:
        """Subscribe to mark price updates (1s or 3s)."""
        sub_id = self._next_sub_id()
        stream = f"{symbol.lower()}@markPrice@{update_speed}"
        self._register_callback(f"markPrice:{symbol}", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "markPrice", "symbol": symbol, "key": f"markPrice:{symbol}"}
        return sub_id

    async def subscribe_liquidations(self, symbol: str, callback: Callable) -> str:
        """Subscribe to forced liquidation events for a symbol."""
        sub_id = self._next_sub_id()
        stream = f"{symbol.lower()}@forceOrder"
        self._register_callback("liquidation", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "liquidation", "symbol": symbol, "key": "liquidation"}
        return sub_id

    async def subscribe_all_liquidations(self, callback: Callable) -> str:
        """Subscribe to all forced liquidation events across all symbols."""
        sub_id = self._next_sub_id()
        stream = "!forceOrder@arr"
        self._register_callback("liquidation", callback)
        url = f"{self._ws_base_url}/{stream}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._subscriptions[sub_id] = {"type": "all_liquidations", "key": "liquidation"}
        return sub_id

    async def subscribe_user_data(
        self,
        on_order: Optional[Callable[[UnifiedOrder], Any]] = None,
        on_position: Optional[Callable[[UnifiedPosition], Any]] = None,
        on_balance: Optional[Callable[[UnifiedBalance], Any]] = None,
    ) -> str:
        if not self.config.api_key:
            raise AuthenticationError("API key required", exchange=self.exchange_name)

        self._listen_key = await self._create_listen_key()
        sub_id = self._next_sub_id()

        if on_order:
            self._register_callback("user:order", on_order)
        if on_position:
            self._register_callback("user:position", on_position)
        if on_balance:
            self._register_callback("user:balance", on_balance)

        url = f"{self._ws_base_url}/{self._listen_key}"
        self._ws_tasks[sub_id] = asyncio.create_task(self._ws_connect(url, sub_id))
        self._listen_key_task = asyncio.create_task(self._listen_key_keepalive_loop())
        self._subscriptions[sub_id] = {"type": "user_data", "key": "user"}
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
        key = sub.get("key", "")
        if key:
            self._remove_callbacks(key)
        if sub.get("type") == "user_data":
            await self._close_listen_key()
            if self._listen_key_task and not self._listen_key_task.done():
                self._listen_key_task.cancel()
        return True

    @property
    def active_subscriptions(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._subscriptions)

    @property
    def position_mode_dual(self) -> bool:
        return self._position_mode_dual
