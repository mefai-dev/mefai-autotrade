# mefai-autotrade - OKX exchange connector
# Full REST + WebSocket implementation for OKX trading

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone
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

# OKX API URLs
OKX_REST = "https://www.okx.com"
OKX_REST_DEMO = "https://www.okx.com"  # Same URL, uses x-simulated-trading header
OKX_WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"
OKX_WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"
OKX_WS_BUSINESS = "wss://ws.okx.com:8443/ws/v5/business"
OKX_WS_PUBLIC_DEMO = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
OKX_WS_PRIVATE_DEMO = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"

# Order type mapping
_ORDER_TYPE_MAP = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.LIMIT_MAKER: "post_only",
    OrderType.STOP_MARKET: "trigger",
    OrderType.STOP_LIMIT: "trigger",
    OrderType.TAKE_PROFIT_MARKET: "trigger",
    OrderType.TAKE_PROFIT_LIMIT: "trigger",
}

_ORDER_STATUS_MAP = {
    "live": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "mmp_canceled": OrderStatus.CANCELED,
}

_TIF_MAP = {
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
}

# OKX instrument type mapping
_INST_TYPE_MAP = {
    "spot": "SPOT",
    "futures": "FUTURES",
    "swap": "SWAP",
    "option": "OPTION",
    "margin": "MARGIN",
}


class OKXExchange(ExchangeBase):
    """OKX exchange connector with full REST and WebSocket support.

    Supports spot, futures (delivery), swap (perpetual), and option trading
    through OKX's unified account model. All methods use the V5 API.
    """

    def __init__(self, config: ExchangeConfig, inst_type: str = "SWAP"):
        """Initialize OKX connector.

        Args:
            config: Exchange configuration. Passphrase is required for OKX.
            inst_type: Default instrument type - SPOT, SWAP, FUTURES, OPTION.
        """
        super().__init__(config, "okx")
        self._inst_type = inst_type
        self._base_url = config.base_url_override or OKX_REST
        self._ws_public_url = OKX_WS_PUBLIC_DEMO if config.testnet else OKX_WS_PUBLIC
        self._ws_private_url = OKX_WS_PRIVATE_DEMO if config.testnet else OKX_WS_PRIVATE
        self._ws_business_url = OKX_WS_BUSINESS
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
        self._logger.info("OKX connector initialized (inst_type=%s, testnet=%s)", self._inst_type, self.config.testnet)

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
        self._logger.info("OKX connector closed")

    # ------------------------------------------------------------------
    # OKX signature (HMAC-SHA256 + Base64)
    # ------------------------------------------------------------------

    def _sign_okx(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate OKX API signature.

        signature = Base64(HMAC-SHA256(timestamp + method + path + body, secret))
        """
        message = f"{timestamp}{method}{path}{body}"
        mac = hmac.new(
            self.config.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _get_timestamp(self) -> str:
        """ISO 8601 timestamp for OKX API."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _build_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Build authenticated headers for OKX API requests."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        if self.config.testnet:
            headers["x-simulated-trading"] = "1"

        if self.config.api_key:
            timestamp = self._get_timestamp()
            signature = self._sign_okx(timestamp, method, path, body)
            headers["OK-ACCESS-KEY"] = self.config.api_key
            headers["OK-ACCESS-SIGN"] = signature
            headers["OK-ACCESS-TIMESTAMP"] = timestamp
            headers["OK-ACCESS-PASSPHRASE"] = self.config.passphrase

        return headers

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

        params = {k: v for k, v in (params or {}).items() if v is not None}
        body = ""
        url = f"{self._base_url}{path}"

        if method == "GET" and params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
            full_path = f"{path}?{query}"
        else:
            full_path = path
            if method == "POST" and params:
                body = json.dumps(params)

        headers = self._build_headers(method.upper(), full_path, body) if signed else {
            "Content-Type": "application/json",
        }
        if self.config.testnet:
            headers["x-simulated-trading"] = "1"

        t_start = time.monotonic()
        try:
            if method == "GET":
                async with self._session.get(url, headers=headers) as resp:
                    latency = (time.monotonic() - t_start) * 1000
                    self._health.record_latency(latency)
                    data = await resp.json()
            else:
                async with self._session.post(url, data=body, headers=headers) as resp:
                    latency = (time.monotonic() - t_start) * 1000
                    self._health.record_latency(latency)
                    data = await resp.json()

            code = data.get("code", "0")
            if code != "0":
                self._handle_error(data)

            return data.get("data", [])

        except aiohttp.ClientError as exc:
            self._health.record_error(str(exc))
            raise NetworkError(f"Network error: {exc}", exchange=self.exchange_name) from exc

    def _handle_error(self, data: Dict[str, Any]) -> None:
        code = data.get("code", "0")
        msg = data.get("msg", "Unknown error")

        # Also check sCode in data items
        items = data.get("data", [])
        if items and isinstance(items, list) and len(items) > 0:
            s_code = items[0].get("sCode", "0")
            s_msg = items[0].get("sMsg", msg)
            if s_code != "0":
                code = s_code
                msg = s_msg

        code_int = int(code) if code.isdigit() else 0

        if code_int in (50100, 50101, 50102, 50103, 50104, 50105):
            raise AuthenticationError(msg, code=code_int, exchange=self.exchange_name)
        elif code_int in (51008, 51127, 51131):
            raise InsufficientBalanceError(msg, code=code_int, exchange=self.exchange_name)
        elif code_int in (51400, 51401, 51402, 51403):
            raise OrderNotFoundError(msg, code=code_int, exchange=self.exchange_name)
        elif code_int in (51000, 51001, 51002, 51003, 51004, 51005, 51006, 51007, 51009, 51010, 51011, 51012, 51020):
            raise InvalidOrderError(msg, code=code_int, exchange=self.exchange_name)
        elif code_int == 50011:
            raise RateLimitError(msg, code=code_int, exchange=self.exchange_name)
        else:
            raise ExchangeError(msg, code=code_int, exchange=self.exchange_name)

    async def _load_exchange_info(self) -> None:
        data = await self._request("GET", "/api/v5/public/instruments", {"instType": self._inst_type})
        for item in data:
            inst_id = item.get("instId", "")
            self._symbol_info[inst_id] = {
                "inst_type": item.get("instType", ""),
                "uly": item.get("uly", ""),
                "base_ccy": item.get("baseCcy", ""),
                "quote_ccy": item.get("quoteCcy", ""),
                "settle_ccy": item.get("settlCcy", ""),
                "ct_val": float(item.get("ctVal", 1)),
                "ct_mult": float(item.get("ctMult", 1)),
                "ct_type": item.get("ctType", ""),
                "lot_sz": float(item.get("lotSz", "0.001")),
                "min_sz": float(item.get("minSz", "0.001")),
                "tick_sz": float(item.get("tickSz", "0.01")),
                "state": item.get("state", ""),
                "lever": item.get("lever", ""),
                "max_lever": float(item.get("lever", "100")),
                "inst_family": item.get("instFamily", ""),
            }
        self._exchange_info_cache = {"list": data}
        self._exchange_info_ts = time.monotonic()
        self._logger.info("Loaded OKX info for %d %s instruments", len(self._symbol_info), self._inst_type)

    async def _fetch_exchange_info(self) -> Dict[str, Any]:
        data = await self._request("GET", "/api/v5/public/instruments", {"instType": self._inst_type})
        return {"list": data}

    # ------------------------------------------------------------------
    # Symbol conversion helpers
    # ------------------------------------------------------------------

    def to_okx_symbol(self, symbol: str) -> str:
        """Convert unified symbol to OKX format.
        BTCUSDT -> BTC-USDT (spot), BTC-USDT-SWAP (swap)
        """
        # Already in OKX format
        if "-" in symbol:
            return symbol

        # Try common patterns
        for quote in ("USDT", "USDC", "BTC", "ETH"):
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                if self._inst_type == "SPOT":
                    return f"{base}-{quote}"
                elif self._inst_type == "SWAP":
                    return f"{base}-{quote}-SWAP"
                elif self._inst_type == "FUTURES":
                    return f"{base}-{quote}"
        return symbol

    def from_okx_symbol(self, inst_id: str) -> str:
        """Convert OKX instrument ID to unified format.
        BTC-USDT -> BTCUSDT, BTC-USDT-SWAP -> BTCUSDT
        """
        parts = inst_id.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}{parts[1]}"
        return inst_id

    # ------------------------------------------------------------------
    # Precision helpers
    # ------------------------------------------------------------------

    def _adjust_quantity(self, inst_id: str, quantity: float) -> str:
        info = self._symbol_info.get(inst_id)
        if not info:
            return str(quantity)
        lot_sz = info["lot_sz"]
        precision = self._step_precision(lot_sz)
        adjusted = int(quantity / lot_sz) * lot_sz
        return f"{adjusted:.{precision}f}"

    def _adjust_price(self, inst_id: str, price: float) -> str:
        info = self._symbol_info.get(inst_id)
        if not info:
            return str(price)
        tick_sz = info["tick_sz"]
        precision = self._step_precision(tick_sz)
        adjusted = int(price / tick_sz) * tick_sz
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
        inst_id = self.to_okx_symbol(symbol)
        data = await self._request("GET", "/api/v5/market/ticker", {"instId": inst_id})
        if not data:
            raise ExchangeError(f"No ticker for {inst_id}", exchange=self.exchange_name)
        return self._parse_ticker(data[0])

    async def get_tickers(self, inst_type: Optional[str] = None) -> List[UnifiedTicker]:
        it = inst_type or self._inst_type
        data = await self._request("GET", "/api/v5/market/tickers", {"instType": it})
        return [self._parse_ticker(t) for t in data]

    def _parse_ticker(self, data: Dict[str, Any]) -> UnifiedTicker:
        last = float(data.get("last", 0))
        open_24h = float(data.get("open24h", 0))
        change = last - open_24h if open_24h > 0 else 0
        change_pct = (change / open_24h * 100) if open_24h > 0 else 0

        return UnifiedTicker(
            symbol=data.get("instId", ""),
            exchange=self.exchange_name,
            last_price=last,
            bid_price=float(data.get("bidPx", 0)),
            ask_price=float(data.get("askPx", 0)),
            bid_qty=float(data.get("bidSz", 0)),
            ask_qty=float(data.get("askSz", 0)),
            high_24h=float(data.get("high24h", 0)),
            low_24h=float(data.get("low24h", 0)),
            volume_24h=float(data.get("vol24h", 0)),
            quote_volume_24h=float(data.get("volCcy24h", 0)),
            price_change_24h=change,
            price_change_pct=change_pct,
            timestamp=int(data.get("ts", current_timestamp_ms())),
            raw=data,
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> UnifiedOrderbook:
        inst_id = self.to_okx_symbol(symbol)
        # OKX supports sz: 1-400 (books), 1-5 (books5), bbo-tbt
        actual_limit = min(limit, 400)
        data = await self._request(
            "GET", "/api/v5/market/books",
            {"instId": inst_id, "sz": actual_limit},
        )
        if not data:
            raise ExchangeError(f"No orderbook for {inst_id}", exchange=self.exchange_name)

        book = data[0]
        return UnifiedOrderbook(
            symbol=inst_id,
            exchange=self.exchange_name,
            bids=[OrderbookLevel(float(b[0]), float(b[1])) for b in book.get("bids", [])],
            asks=[OrderbookLevel(float(a[0]), float(a[1])) for a in book.get("asks", [])],
            timestamp=int(book.get("ts", current_timestamp_ms())),
            raw=book,
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedKline]:
        inst_id = self.to_okx_symbol(symbol)
        okx_bar = self._convert_interval(interval)
        params: Dict[str, Any] = {
            "instId": inst_id,
            "bar": okx_bar,
            "limit": str(min(limit, 300)),
        }
        if start_time:
            params["after"] = str(start_time)
        if end_time:
            params["before"] = str(end_time)

        data = await self._request("GET", "/api/v5/market/candles", params)
        klines = []
        for k in data:
            klines.append(UnifiedKline(
                symbol=inst_id,
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
                is_closed=k[8] == "1" if len(k) > 8 else True,
                raw=k,
            ))
        # OKX returns newest first
        klines.reverse()
        return klines

    async def get_trades(self, symbol: str, limit: int = 500) -> List[UnifiedTrade]:
        inst_id = self.to_okx_symbol(symbol)
        data = await self._request(
            "GET", "/api/v5/market/trades",
            {"instId": inst_id, "limit": str(min(limit, 500))},
        )
        trades = []
        for t in data:
            trades.append(UnifiedTrade(
                symbol=inst_id,
                exchange=self.exchange_name,
                trade_id=str(t.get("tradeId", "")),
                price=float(t.get("px", 0)),
                quantity=float(t.get("sz", 0)),
                quote_quantity=float(t.get("px", 0)) * float(t.get("sz", 0)),
                side=OrderSide.BUY if t.get("side", "") == "buy" else OrderSide.SELL,
                timestamp=int(t.get("ts", current_timestamp_ms())),
                raw=t,
            ))
        return trades

    @staticmethod
    def _convert_interval(interval: str) -> str:
        mapping = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
            "1d": "1D", "1w": "1W", "1M": "1M",
        }
        return mapping.get(interval, interval)

    @staticmethod
    def _interval_ms(interval: str) -> int:
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
        params: Dict[str, Any] = {}
        if asset:
            params["ccy"] = asset
        data = await self._request("GET", "/api/v5/account/balance", params, signed=True)
        balances = []
        for account in data:
            for detail in account.get("details", []):
                avail = float(detail.get("availBal", 0))
                frozen = float(detail.get("frozenBal", 0))
                total = float(detail.get("cashBal", 0))
                eq = float(detail.get("eq", 0))
                if total == 0 and eq == 0 and asset is None:
                    continue
                balances.append(UnifiedBalance(
                    asset=detail.get("ccy", ""),
                    exchange=self.exchange_name,
                    free=avail,
                    locked=frozen,
                    total=total or (avail + frozen),
                    usd_value=float(detail.get("eqUsd", 0)),
                    raw=detail,
                ))
        return balances

    async def get_positions(self, symbol: Optional[str] = None) -> List[UnifiedPosition]:
        if self._inst_type == "SPOT":
            return []

        params: Dict[str, Any] = {"instType": self._inst_type}
        if symbol:
            params["instId"] = self.to_okx_symbol(symbol)

        data = await self._request("GET", "/api/v5/account/positions", params, signed=True)
        positions = []
        for p in data:
            pos_amt = float(p.get("pos", 0))
            if pos_amt == 0 and symbol is None:
                continue

            pos_side_str = p.get("posSide", "net")
            if pos_side_str == "long":
                pos_side = PositionSide.LONG
            elif pos_side_str == "short":
                pos_side = PositionSide.SHORT
            else:
                pos_side = PositionSide.BOTH

            mgn_mode = p.get("mgnMode", "cross")

            positions.append(UnifiedPosition(
                symbol=p.get("instId", ""),
                exchange=self.exchange_name,
                side=pos_side,
                size=abs(pos_amt),
                entry_price=float(p.get("avgPx", 0)),
                mark_price=float(p.get("markPx", 0)),
                liquidation_price=float(p.get("liqPx", 0)) if p.get("liqPx") else 0,
                unrealized_pnl=float(p.get("upl", 0)),
                realized_pnl=float(p.get("realizedPnl", 0)),
                leverage=int(float(p.get("lever", 1))),
                margin_mode=MarginMode.ISOLATED if mgn_mode == "isolated" else MarginMode.CROSS,
                margin=float(p.get("margin", 0)) or float(p.get("imr", 0)),
                notional=float(p.get("notionalUsd", 0)),
                adl_quantile=float(p.get("adl", 0)),
                timestamp=int(p.get("uTime", current_timestamp_ms())),
                raw=p,
            ))
        return positions

    async def get_account_info(self) -> UnifiedAccountInfo:
        data = await self._request("GET", "/api/v5/account/balance", signed=True)
        balances = []
        total_eq = 0.0
        available_eq = 0.0

        for account in data:
            total_eq = float(account.get("totalEq", 0))
            available_eq = float(account.get("availBal", 0)) or float(account.get("availEq", 0))

            for detail in account.get("details", []):
                avail = float(detail.get("availBal", 0))
                frozen = float(detail.get("frozenBal", 0))
                total = float(detail.get("cashBal", 0))
                if total == 0 and avail == 0:
                    continue
                balances.append(UnifiedBalance(
                    asset=detail.get("ccy", ""),
                    exchange=self.exchange_name,
                    free=avail,
                    locked=frozen,
                    total=total or (avail + frozen),
                    usd_value=float(detail.get("eqUsd", 0)),
                    raw=detail,
                ))

        positions = await self.get_positions() if self._inst_type != "SPOT" else []
        total_unrealized = sum(p.unrealized_pnl for p in positions)

        return UnifiedAccountInfo(
            exchange=self.exchange_name,
            account_type=self._inst_type,
            can_trade=True,
            can_withdraw=True,
            can_deposit=True,
            balances=balances,
            total_usd_value=total_eq,
            available_margin=available_eq,
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
        inst_id = self.to_okx_symbol(symbol)

        # Determine trade mode
        td_mode = kwargs.get("td_mode", "cross")
        if self._inst_type == "SPOT":
            td_mode = "cash"

        okx_type = "market" if order_type == OrderType.MARKET else "limit"
        if post_only:
            okx_type = "post_only"

        params: Dict[str, Any] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": "buy" if side == OrderSide.BUY else "sell",
            "ordType": okx_type,
            "sz": self._adjust_quantity(inst_id, quantity),
        }

        if client_order_id:
            params["clOrdId"] = client_order_id

        if price and order_type != OrderType.MARKET:
            params["px"] = self._adjust_price(inst_id, price)

        if position_side and self._inst_type != "SPOT":
            ps = "long" if position_side == PositionSide.LONG else "short"
            params["posSide"] = ps

        if reduce_only:
            params["reduceOnly"] = True

        if stop_price:
            # OKX uses algo orders for stop/TP/SL
            return await self._place_algo_order(
                inst_id, side, quantity, price, stop_price,
                order_type, td_mode, position_side, client_order_id,
            )

        # Take profit / stop loss attached to order
        tp_price = kwargs.get("take_profit")
        sl_price = kwargs.get("stop_loss")
        if tp_price:
            params["tpTriggerPx"] = self._adjust_price(inst_id, tp_price)
            params["tpOrdPx"] = "-1"  # Market
        if sl_price:
            params["slTriggerPx"] = self._adjust_price(inst_id, sl_price)
            params["slOrdPx"] = "-1"  # Market

        data = await self._request("POST", "/api/v5/trade/order", params, signed=True)
        if not data:
            raise ExchangeError("Empty response from place_order", exchange=self.exchange_name)

        result = data[0]
        if result.get("sCode", "0") != "0":
            raise InvalidOrderError(
                result.get("sMsg", "Order failed"),
                code=int(result.get("sCode", 0)),
                exchange=self.exchange_name,
            )

        return UnifiedOrder(
            symbol=inst_id,
            exchange=self.exchange_name,
            order_id=result.get("ordId", ""),
            client_order_id=result.get("clOrdId", client_order_id or ""),
            side=side,
            order_type=order_type,
            status=OrderStatus.NEW,
            price=price or 0,
            stop_price=stop_price or 0,
            quantity=quantity,
            filled_quantity=0,
            remaining_quantity=quantity,
            average_price=0,
            commission=0,
            commission_asset="",
            created_at=current_timestamp_ms(),
            raw=result,
        )

    async def _place_algo_order(
        self,
        inst_id: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float],
        stop_price: float,
        order_type: OrderType,
        td_mode: str,
        position_side: Optional[PositionSide],
        client_order_id: Optional[str],
    ) -> UnifiedOrder:
        """Place an algo order (stop, take profit, trailing stop)."""
        params: Dict[str, Any] = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": "buy" if side == OrderSide.BUY else "sell",
            "ordType": "conditional",
            "sz": self._adjust_quantity(inst_id, quantity),
        }

        if client_order_id:
            params["clOrdId"] = client_order_id

        if position_side:
            params["posSide"] = "long" if position_side == PositionSide.LONG else "short"

        # Stop price as trigger
        if order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            params["slTriggerPx"] = self._adjust_price(inst_id, stop_price)
            if price:
                params["slOrdPx"] = self._adjust_price(inst_id, price)
            else:
                params["slOrdPx"] = "-1"  # Market price

        elif order_type in (OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT):
            params["tpTriggerPx"] = self._adjust_price(inst_id, stop_price)
            if price:
                params["tpOrdPx"] = self._adjust_price(inst_id, price)
            else:
                params["tpOrdPx"] = "-1"

        data = await self._request("POST", "/api/v5/trade/order-algo", params, signed=True)
        if not data:
            raise ExchangeError("Empty algo order response", exchange=self.exchange_name)

        result = data[0]
        if result.get("sCode", "0") != "0":
            raise InvalidOrderError(
                result.get("sMsg", "Algo order failed"),
                code=int(result.get("sCode", 0)),
                exchange=self.exchange_name,
            )

        return UnifiedOrder(
            symbol=inst_id,
            exchange=self.exchange_name,
            order_id=result.get("algoId", ""),
            client_order_id=result.get("clOrdId", client_order_id or ""),
            side=side,
            order_type=order_type,
            status=OrderStatus.PENDING,
            price=price or 0,
            stop_price=stop_price,
            quantity=quantity,
            filled_quantity=0,
            remaining_quantity=quantity,
            average_price=0,
            commission=0,
            commission_asset="",
            created_at=current_timestamp_ms(),
            raw=result,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        inst_id = self.to_okx_symbol(symbol)
        data = await self._request(
            "POST", "/api/v5/trade/cancel-order",
            {"instId": inst_id, "ordId": order_id},
            signed=True,
        )
        if not data:
            raise ExchangeError("Empty cancel response", exchange=self.exchange_name)

        result = data[0]
        return UnifiedOrder(
            symbol=inst_id,
            exchange=self.exchange_name,
            order_id=result.get("ordId", order_id),
            client_order_id=result.get("clOrdId", ""),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.CANCELED,
            price=0, stop_price=0, quantity=0,
            filled_quantity=0, remaining_quantity=0,
            average_price=0, commission=0, commission_asset="",
            raw=result,
        )

    async def modify_order(
        self,
        symbol: str,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> UnifiedOrder:
        inst_id = self.to_okx_symbol(symbol)
        params: Dict[str, Any] = {
            "instId": inst_id,
            "ordId": order_id,
        }
        if quantity:
            params["newSz"] = self._adjust_quantity(inst_id, quantity)
        if price:
            params["newPx"] = self._adjust_price(inst_id, price)

        data = await self._request("POST", "/api/v5/trade/amend-order", params, signed=True)
        if not data:
            raise ExchangeError("Empty amend response", exchange=self.exchange_name)

        result = data[0]
        return UnifiedOrder(
            symbol=inst_id,
            exchange=self.exchange_name,
            order_id=result.get("ordId", order_id),
            client_order_id=result.get("clOrdId", ""),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            price=price or 0, stop_price=0,
            quantity=quantity or 0,
            filled_quantity=0, remaining_quantity=quantity or 0,
            average_price=0, commission=0, commission_asset="",
            raw=result,
        )

    async def get_order(self, symbol: str, order_id: str) -> UnifiedOrder:
        inst_id = self.to_okx_symbol(symbol)
        data = await self._request(
            "GET", "/api/v5/trade/order",
            {"instId": inst_id, "ordId": order_id},
            signed=True,
        )
        if not data:
            raise OrderNotFoundError(f"Order {order_id} not found", exchange=self.exchange_name)
        return self._parse_order_detail(data[0])

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {"instType": self._inst_type}
        if symbol:
            params["instId"] = self.to_okx_symbol(symbol)
        data = await self._request("GET", "/api/v5/trade/orders-pending", params, signed=True)
        return [self._parse_order_detail(o) for o in data]

    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[UnifiedOrder]:
        params: Dict[str, Any] = {
            "instType": self._inst_type,
            "limit": str(min(limit, 100)),
        }
        if symbol:
            params["instId"] = self.to_okx_symbol(symbol)
        if start_time:
            params["begin"] = str(start_time)
        if end_time:
            params["end"] = str(end_time)

        data = await self._request("GET", "/api/v5/trade/orders-history", params, signed=True)
        return [self._parse_order_detail(o) for o in data]

    def _parse_order_detail(self, data: Dict[str, Any]) -> UnifiedOrder:
        raw_status = data.get("state", "live")
        side_str = data.get("side", "buy")
        raw_type = data.get("ordType", "market")
        sz = float(data.get("sz", 0))
        filled = float(data.get("accFillSz", 0) or data.get("fillSz", 0))

        if raw_type == "market":
            ot = OrderType.MARKET
        elif raw_type == "limit":
            ot = OrderType.LIMIT
        elif raw_type == "post_only":
            ot = OrderType.LIMIT_MAKER
        else:
            ot = OrderType.MARKET

        return UnifiedOrder(
            symbol=data.get("instId", ""),
            exchange=self.exchange_name,
            order_id=data.get("ordId", ""),
            client_order_id=data.get("clOrdId", ""),
            side=OrderSide.BUY if side_str == "buy" else OrderSide.SELL,
            order_type=ot,
            status=_ORDER_STATUS_MAP.get(raw_status, OrderStatus.NEW),
            price=float(data.get("px", 0) or 0),
            stop_price=0,
            quantity=sz,
            filled_quantity=filled,
            remaining_quantity=sz - filled,
            average_price=float(data.get("avgPx", 0) or 0),
            commission=abs(float(data.get("fee", 0) or 0)),
            commission_asset=data.get("feeCcy", ""),
            reduce_only=data.get("reduceOnly", "") == "true",
            created_at=int(data.get("cTime", 0)),
            updated_at=int(data.get("uTime", 0)),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Futures specific
    # ------------------------------------------------------------------

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        inst_id = self.to_okx_symbol(symbol)
        try:
            await self._request(
                "POST", "/api/v5/account/set-leverage",
                {
                    "instId": inst_id,
                    "lever": str(leverage),
                    "mgnMode": "cross",
                },
                signed=True,
            )
            return True
        except ExchangeError as exc:
            self._logger.error("Failed to set leverage: %s", exc)
            return False

    async def set_margin_mode(self, symbol: str, mode: MarginMode) -> bool:
        # OKX sets margin mode per account, not per symbol
        pos_mode = "long_short_mode" if mode == MarginMode.ISOLATED else "net_mode"
        try:
            await self._request(
                "POST", "/api/v5/account/set-position-mode",
                {"posMode": pos_mode},
                signed=True,
            )
            return True
        except ExchangeError as exc:
            self._logger.error("Failed to set position mode: %s", exc)
            return False

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        if self._inst_type == "SPOT":
            return {"symbol": symbol, "funding_rate": 0.0}
        inst_id = self.to_okx_symbol(symbol)
        data = await self._request(
            "GET", "/api/v5/public/funding-rate",
            {"instId": inst_id},
        )
        if not data:
            return {"symbol": inst_id, "funding_rate": 0.0}
        item = data[0]
        return {
            "symbol": inst_id,
            "funding_rate": float(item.get("fundingRate", 0)),
            "next_funding_rate": float(item.get("nextFundingRate", 0)) if item.get("nextFundingRate") else None,
            "next_funding_time": int(item.get("nextFundingTime", 0)),
            "funding_time": int(item.get("fundingTime", 0)),
        }

    async def get_mark_price(self, symbol: str) -> Dict[str, Any]:
        inst_id = self.to_okx_symbol(symbol)
        data = await self._request(
            "GET", "/api/v5/public/mark-price",
            {"instType": self._inst_type, "instId": inst_id},
        )
        if not data:
            return {"symbol": inst_id, "mark_price": 0}
        item = data[0]
        return {
            "symbol": inst_id,
            "mark_price": float(item.get("markPx", 0)),
            "timestamp": int(item.get("ts", 0)),
        }

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _ws_connect(self, url: str, stream_id: str, channels: List[Dict[str, str]], is_private: bool = False) -> None:
        attempt = 0
        while True:
            try:
                async with self._session.ws_connect(url, heartbeat=25) as ws:
                    self._ws_connections[stream_id] = ws
                    attempt = 0
                    self._health.set_ws_status(True, len(self._ws_connections))

                    if is_private:
                        await self._ws_authenticate(ws)

                    # Subscribe
                    sub_msg = {"op": "subscribe", "args": channels}
                    await ws.send_json(sub_msg)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                # Ping-pong
                                if data == "pong" or data.get("op") == "pong":
                                    continue
                                self._handle_ws_message(stream_id, data)
                            except (json.JSONDecodeError, AttributeError):
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
        """Authenticate a private OKX WebSocket connection."""
        timestamp = str(int(time.time()))
        sign_str = f"{timestamp}GET/users/self/verify"
        signature = base64.b64encode(
            hmac.new(
                self.config.api_secret.encode("utf-8"),
                sign_str.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        auth_msg = {
            "op": "login",
            "args": [{
                "apiKey": self.config.api_key,
                "passphrase": self.config.passphrase,
                "timestamp": timestamp,
                "sign": signature,
            }],
        }
        await ws.send_json(auth_msg)
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=5)
            if msg.type == aiohttp.WSMsgType.TEXT:
                resp = json.loads(msg.data)
                if resp.get("event") == "login":
                    self._logger.info("OKX WS authentication successful")
                else:
                    self._logger.error("OKX WS auth response: %s", resp)
        except asyncio.TimeoutError:
            self._logger.warning("OKX WS auth timeout")

    def _handle_ws_message(self, stream_id: str, data: Dict[str, Any]) -> None:
        self._health.record_heartbeat()

        event = data.get("event", "")
        if event in ("subscribe", "unsubscribe", "login", "error"):
            if event == "error":
                self._logger.error("OKX WS error: %s", data)
            return

        arg = data.get("arg", {})
        channel = arg.get("channel", "")
        inst_id = arg.get("instId", "")
        msg_data = data.get("data", [])

        if channel == "tickers" and msg_data:
            for item in msg_data:
                ticker = self._parse_ticker(item)
                self._dispatch_callback(f"ticker:{inst_id}", ticker)

        elif channel.startswith("books") and msg_data:
            for item in msg_data:
                book = UnifiedOrderbook(
                    symbol=inst_id,
                    exchange=self.exchange_name,
                    bids=[OrderbookLevel(float(b[0]), float(b[1])) for b in item.get("bids", [])],
                    asks=[OrderbookLevel(float(a[0]), float(a[1])) for a in item.get("asks", [])],
                    timestamp=int(item.get("ts", current_timestamp_ms())),
                    raw=data,
                )
                self._dispatch_callback(f"depth:{inst_id}", book)

        elif channel == "trades" and msg_data:
            for t in msg_data:
                trade = UnifiedTrade(
                    symbol=inst_id,
                    exchange=self.exchange_name,
                    trade_id=str(t.get("tradeId", "")),
                    price=float(t.get("px", 0)),
                    quantity=float(t.get("sz", 0)),
                    quote_quantity=float(t.get("px", 0)) * float(t.get("sz", 0)),
                    side=OrderSide.BUY if t.get("side", "") == "buy" else OrderSide.SELL,
                    timestamp=int(t.get("ts", current_timestamp_ms())),
                    raw=t,
                )
                self._dispatch_callback(f"trade:{inst_id}", trade)

        elif channel.startswith("candle") and msg_data:
            interval = channel.replace("candle", "")
            for k in msg_data:
                kline = UnifiedKline(
                    symbol=inst_id,
                    exchange=self.exchange_name,
                    interval=interval,
                    open_time=int(k[0]),
                    close_time=int(k[0]) + 60000,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    quote_volume=float(k[6]) if len(k) > 6 else 0,
                    trades=0,
                    is_closed=k[8] == "1" if len(k) > 8 else False,
                    raw=k,
                )
                self._dispatch_callback(f"kline:{inst_id}:{interval}", kline)

        # Private channels
        elif channel == "orders" and msg_data:
            for o in msg_data:
                order = self._parse_order_detail(o)
                self._dispatch_callback("user:order", order)

        elif channel == "positions" and msg_data:
            for p in msg_data:
                pos_amt = float(p.get("pos", 0))
                pos_side_str = p.get("posSide", "net")
                if pos_side_str == "long":
                    pos_side = PositionSide.LONG
                elif pos_side_str == "short":
                    pos_side = PositionSide.SHORT
                else:
                    pos_side = PositionSide.BOTH

                position = UnifiedPosition(
                    symbol=p.get("instId", ""),
                    exchange=self.exchange_name,
                    side=pos_side,
                    size=abs(pos_amt),
                    entry_price=float(p.get("avgPx", 0)),
                    mark_price=float(p.get("markPx", 0)),
                    liquidation_price=float(p.get("liqPx", 0) or 0),
                    unrealized_pnl=float(p.get("upl", 0)),
                    realized_pnl=float(p.get("realizedPnl", 0)),
                    leverage=int(float(p.get("lever", 1))),
                    margin_mode=MarginMode.ISOLATED if p.get("mgnMode") == "isolated" else MarginMode.CROSS,
                    margin=float(p.get("margin", 0) or 0),
                    notional=float(p.get("notionalUsd", 0)),
                    adl_quantile=float(p.get("adl", 0)),
                    timestamp=int(p.get("uTime", current_timestamp_ms())),
                    raw=p,
                )
                self._dispatch_callback("user:position", position)

        elif channel == "account" and msg_data:
            for account in msg_data:
                for detail in account.get("details", []):
                    balance = UnifiedBalance(
                        asset=detail.get("ccy", ""),
                        exchange=self.exchange_name,
                        free=float(detail.get("availBal", 0)),
                        locked=float(detail.get("frozenBal", 0)),
                        total=float(detail.get("cashBal", 0)),
                        usd_value=float(detail.get("eqUsd", 0)),
                        raw=detail,
                    )
                    self._dispatch_callback("user:balance", balance)

    # -- Subscription methods --

    def _next_sub_id(self) -> str:
        self._sub_counter += 1
        return f"osub_{self._sub_counter}_{uuid.uuid4().hex[:8]}"

    async def subscribe_ticker(self, symbol: str, callback: Callable[[UnifiedTicker], Any]) -> str:
        inst_id = self.to_okx_symbol(symbol)
        sub_id = self._next_sub_id()
        channels = [{"channel": "tickers", "instId": inst_id}]
        self._register_callback(f"ticker:{inst_id}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, channels))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "ticker", "inst_id": inst_id}
        return sub_id

    async def subscribe_orderbook(
        self, symbol: str, callback: Callable[[UnifiedOrderbook], Any], depth: int = 20
    ) -> str:
        inst_id = self.to_okx_symbol(symbol)
        sub_id = self._next_sub_id()
        # books for full, books5 for top 5, bbo-tbt for best bid/offer
        channel = "books" if depth > 5 else "books5"
        channels = [{"channel": channel, "instId": inst_id}]
        self._register_callback(f"depth:{inst_id}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, channels))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "depth", "inst_id": inst_id}
        return sub_id

    async def subscribe_trades(self, symbol: str, callback: Callable[[UnifiedTrade], Any]) -> str:
        inst_id = self.to_okx_symbol(symbol)
        sub_id = self._next_sub_id()
        channels = [{"channel": "trades", "instId": inst_id}]
        self._register_callback(f"trade:{inst_id}", callback)
        task = asyncio.create_task(self._ws_connect(self._ws_public_url, sub_id, channels))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "trade", "inst_id": inst_id}
        return sub_id

    async def subscribe_klines(
        self, symbol: str, interval: str, callback: Callable[[UnifiedKline], Any]
    ) -> str:
        inst_id = self.to_okx_symbol(symbol)
        sub_id = self._next_sub_id()
        okx_bar = self._convert_interval(interval)
        channel = f"candle{okx_bar}"
        channels = [{"channel": channel, "instId": inst_id}]
        self._register_callback(f"kline:{inst_id}:{interval}", callback)
        task = asyncio.create_task(self._ws_business_url and
            self._ws_connect(self._ws_business_url, sub_id, channels) or
            self._ws_connect(self._ws_public_url, sub_id, channels))
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "kline", "inst_id": inst_id, "interval": interval}
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
        channels = []

        if on_order:
            self._register_callback("user:order", on_order)
            channels.append({"channel": "orders", "instType": self._inst_type})
        if on_position:
            self._register_callback("user:position", on_position)
            channels.append({"channel": "positions", "instType": self._inst_type})
        if on_balance:
            self._register_callback("user:balance", on_balance)
            channels.append({"channel": "account"})

        if not channels:
            channels = [
                {"channel": "orders", "instType": self._inst_type},
                {"channel": "positions", "instType": self._inst_type},
                {"channel": "account"},
            ]

        task = asyncio.create_task(
            self._ws_connect(self._ws_private_url, sub_id, channels, is_private=True)
        )
        self._ws_tasks[sub_id] = task
        self._subscriptions[sub_id] = {"type": "user_data", "channels": channels}
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
    def inst_type(self) -> str:
        return self._inst_type
