"""
Mefai Signal Engine - Arbitrage Strategy

Spot-futures basis spread, funding rate arbitrage, cross-exchange price
discrepancy, triangular arbitrage, latency-aware execution, transaction
cost calculation, and minimum spread threshold.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import math
import numpy as np

from strategies.base import (
    BaseStrategy, FeatureEngine, StrategySignal, SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    arb_type: str          # "basis", "funding", "cross_exchange", "triangular"
    spread_pct: float
    expected_profit: float
    transaction_cost: float
    net_profit: float
    buy_venue: str
    sell_venue: str
    buy_price: float
    sell_price: float
    quantity: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    expired: bool = False


@dataclass
class FundingRateData:
    """Funding rate information for a symbol."""
    symbol: str
    rate: float              # current funding rate (e.g. 0.01 = 1%)
    next_funding_time: float
    predicted_rate: float
    interval_hours: int = 8


@dataclass
class ExchangePrice:
    """Price from a specific exchange."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    timestamp: float
    latency_ms: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        if self.mid == 0:
            return 0.0
        return ((self.ask - self.bid) / self.mid) * 100.0


class ArbitrageStrategy(BaseStrategy):
    """
    Cross-Exchange and Funding Rate Arbitrage Strategy

    Identifies and executes multiple types of arbitrage:
    1. Spot-Futures basis spread (buy spot, sell futures when premium is high)
    2. Funding rate arbitrage (long spot + short perp when funding > threshold)
    3. Cross-exchange price discrepancy (buy on exchange A, sell on exchange B)
    4. Triangular arbitrage (BTC/USDT -> ETH/BTC -> ETH/USDT cycle)
    5. Latency-aware execution with timing analysis
    6. Full transaction cost calculation including fees and slippage
    """

    name = "Arbitrage"
    description = "Cross-exchange and funding rate arbitrage"
    version = "1.0.0"

    def _initialize_params(self) -> None:
        # Spread thresholds
        self.min_spread_pct: float = self.params.get("min_spread_pct", 0.1)
        self.min_funding_spread: float = self.params.get("min_funding_spread", 0.05)
        self.min_basis_spread: float = self.params.get("min_basis_spread", 0.2)
        self.min_triangular_spread: float = self.params.get("min_triangular_spread", 0.15)

        # Exposure limits
        self.max_exposure: float = self.params.get("max_exposure", 10000.0)
        self.max_per_trade: float = self.params.get("max_per_trade", 5000.0)
        self.max_open_arbs: int = self.params.get("max_open_arbs", 5)

        # Fees
        self.maker_fee_pct: float = self.params.get("maker_fee_pct", 0.02)
        self.taker_fee_pct: float = self.params.get("taker_fee_pct", 0.04)
        self.withdrawal_fee: float = self.params.get("withdrawal_fee", 0.0)
        self.slippage_pct: float = self.params.get("slippage_pct", 0.01)

        # Latency
        self.max_latency_ms: float = self.params.get("max_latency_ms", 500.0)
        self.price_staleness_ms: float = self.params.get("price_staleness_ms", 2000.0)

        # Funding rate
        self.funding_annualized_threshold: float = self.params.get(
            "funding_annualized_threshold", 20.0
        )

        # State
        self._exchange_prices: Dict[str, ExchangePrice] = {}
        self._funding_rates: Dict[str, FundingRateData] = {}
        self._spot_prices: Dict[str, float] = {}
        self._futures_prices: Dict[str, float] = {}
        self._open_arbs: List[ArbitrageOpportunity] = []
        self._total_exposure: float = 0.0
        self._arb_history: List[ArbitrageOpportunity] = []

    # ------------------------------------------------------------------
    # Price management
    # ------------------------------------------------------------------

    def update_exchange_price(
        self, exchange: str, symbol: str, bid: float, ask: float,
        latency_ms: float = 0.0,
    ) -> None:
        """Update price from an exchange. Called by the data feed."""
        key = f"{exchange}:{symbol}"
        self._exchange_prices[key] = ExchangePrice(
            exchange=exchange, symbol=symbol,
            bid=bid, ask=ask,
            timestamp=time.time(), latency_ms=latency_ms,
        )

    def update_spot_price(self, symbol: str, price: float) -> None:
        self._spot_prices[symbol] = price

    def update_futures_price(self, symbol: str, price: float) -> None:
        self._futures_prices[symbol] = price

    def update_funding_rate(
        self, symbol: str, rate: float, next_time: float,
        predicted_rate: float = 0.0,
    ) -> None:
        self._funding_rates[symbol] = FundingRateData(
            symbol=symbol, rate=rate,
            next_funding_time=next_time,
            predicted_rate=predicted_rate,
        )

    def _is_price_fresh(self, price: ExchangePrice) -> bool:
        """Check if a price quote is still fresh enough to trade on."""
        age_ms = (time.time() - price.timestamp) * 1000.0
        return age_ms <= self.price_staleness_ms

    # ------------------------------------------------------------------
    # Transaction cost
    # ------------------------------------------------------------------

    def _calculate_round_trip_cost(
        self, notional: float, is_cross_exchange: bool = False,
    ) -> float:
        """
        Calculate total transaction cost for a round-trip trade.
        Includes maker/taker fees, slippage, and withdrawal fees.
        """
        # Entry fee (taker for speed)
        entry_fee = notional * (self.taker_fee_pct / 100.0)
        # Exit fee (maker if limit order)
        exit_fee = notional * (self.maker_fee_pct / 100.0)
        # Slippage on both sides
        slippage = notional * (self.slippage_pct / 100.0) * 2.0
        # Withdrawal fee for cross-exchange
        withdrawal = self.withdrawal_fee if is_cross_exchange else 0.0

        return entry_fee + exit_fee + slippage + withdrawal

    # ------------------------------------------------------------------
    # Basis spread (spot vs futures)
    # ------------------------------------------------------------------

    def _scan_basis_spread(self) -> Optional[ArbitrageOpportunity]:
        """
        Scan for spot-futures basis spread opportunities.
        When futures premium is high, buy spot + sell futures.
        When futures are at discount, buy futures + sell spot.
        """
        for symbol in self._spot_prices:
            if symbol not in self._futures_prices:
                continue

            spot = self._spot_prices[symbol]
            futures = self._futures_prices[symbol]

            if spot == 0:
                continue

            basis_pct = ((futures - spot) / spot) * 100.0

            if abs(basis_pct) < self.min_basis_spread:
                continue

            notional = min(self.max_per_trade, self.max_exposure - self._total_exposure)
            if notional <= 0:
                continue

            cost = self._calculate_round_trip_cost(notional)
            quantity = notional / spot

            if basis_pct > 0:
                # Futures premium - buy spot, sell futures
                expected_profit = notional * (basis_pct / 100.0)
                net = expected_profit - cost
                if net <= 0:
                    continue
                return ArbitrageOpportunity(
                    arb_type="basis",
                    spread_pct=basis_pct,
                    expected_profit=expected_profit,
                    transaction_cost=cost,
                    net_profit=net,
                    buy_venue="spot",
                    sell_venue="futures",
                    buy_price=spot,
                    sell_price=futures,
                    quantity=quantity,
                    confidence=min(0.9, 0.5 + basis_pct * 0.1),
                )
            else:
                # Futures discount - buy futures, sell spot
                expected_profit = notional * (abs(basis_pct) / 100.0)
                net = expected_profit - cost
                if net <= 0:
                    continue
                return ArbitrageOpportunity(
                    arb_type="basis",
                    spread_pct=abs(basis_pct),
                    expected_profit=expected_profit,
                    transaction_cost=cost,
                    net_profit=net,
                    buy_venue="futures",
                    sell_venue="spot",
                    buy_price=futures,
                    sell_price=spot,
                    quantity=quantity,
                    confidence=min(0.9, 0.5 + abs(basis_pct) * 0.1),
                )

        return None

    # ------------------------------------------------------------------
    # Funding rate arbitrage
    # ------------------------------------------------------------------

    def _scan_funding_arb(self) -> Optional[ArbitrageOpportunity]:
        """
        Scan for funding rate arbitrage.
        When funding is very positive (longs pay shorts), go long spot + short perp
        to collect funding payments while being market neutral.
        """
        for symbol, funding in self._funding_rates.items():
            if symbol not in self._spot_prices:
                continue

            # Annualize the funding rate (8h intervals = 3x per day = 1095x per year)
            annualized = funding.rate * 1095.0 * 100.0  # as percentage

            if abs(annualized) < self.funding_annualized_threshold:
                continue

            spot_price = self._spot_prices.get(symbol, 0.0)
            if spot_price == 0:
                continue

            notional = min(self.max_per_trade, self.max_exposure - self._total_exposure)
            if notional <= 0:
                continue

            # Expected profit from one funding payment
            funding_profit = notional * abs(funding.rate)
            cost = self._calculate_round_trip_cost(notional)

            # Must be profitable after costs within a few funding periods
            if funding_profit * 3.0 < cost:
                continue

            quantity = notional / spot_price

            if funding.rate > 0:
                # Positive funding - longs pay shorts
                # Strategy: long spot + short perp = collect funding
                return ArbitrageOpportunity(
                    arb_type="funding",
                    spread_pct=annualized,
                    expected_profit=funding_profit,
                    transaction_cost=cost,
                    net_profit=funding_profit - cost / 3.0,
                    buy_venue="spot",
                    sell_venue="perp",
                    buy_price=spot_price,
                    sell_price=spot_price,
                    quantity=quantity,
                    confidence=min(0.85, 0.4 + abs(annualized) * 0.01),
                )
            else:
                # Negative funding - shorts pay longs
                # Strategy: short spot (or sell) + long perp = collect funding
                return ArbitrageOpportunity(
                    arb_type="funding",
                    spread_pct=abs(annualized),
                    expected_profit=funding_profit,
                    transaction_cost=cost,
                    net_profit=funding_profit - cost / 3.0,
                    buy_venue="perp",
                    sell_venue="spot",
                    buy_price=spot_price,
                    sell_price=spot_price,
                    quantity=quantity,
                    confidence=min(0.85, 0.4 + abs(annualized) * 0.01),
                )

        return None

    # ------------------------------------------------------------------
    # Cross-exchange arbitrage
    # ------------------------------------------------------------------

    def _scan_cross_exchange(self) -> Optional[ArbitrageOpportunity]:
        """
        Scan for cross-exchange price discrepancies.
        Buy on exchange with lower ask, sell on exchange with higher bid.
        """
        # Group prices by symbol
        symbol_prices: Dict[str, List[ExchangePrice]] = {}
        for key, ep in self._exchange_prices.items():
            if not self._is_price_fresh(ep):
                continue
            if ep.symbol not in symbol_prices:
                symbol_prices[ep.symbol] = []
            symbol_prices[ep.symbol].append(ep)

        best_opportunity: Optional[ArbitrageOpportunity] = None
        best_net = 0.0

        for symbol, prices in symbol_prices.items():
            if len(prices) < 2:
                continue

            # Find best bid and best ask across exchanges
            for i in range(len(prices)):
                for j in range(len(prices)):
                    if i == j:
                        continue

                    buy_exchange = prices[i]   # buy at ask
                    sell_exchange = prices[j]   # sell at bid

                    if buy_exchange.ask >= sell_exchange.bid:
                        continue

                    spread_pct = ((sell_exchange.bid - buy_exchange.ask)
                                  / buy_exchange.ask) * 100.0

                    if spread_pct < self.min_spread_pct:
                        continue

                    # Latency check
                    if buy_exchange.latency_ms > self.max_latency_ms:
                        continue
                    if sell_exchange.latency_ms > self.max_latency_ms:
                        continue

                    notional = min(self.max_per_trade,
                                   self.max_exposure - self._total_exposure)
                    if notional <= 0:
                        continue

                    cost = self._calculate_round_trip_cost(notional, is_cross_exchange=True)
                    expected = notional * (spread_pct / 100.0)
                    net = expected - cost

                    if net > best_net:
                        best_net = net
                        quantity = notional / buy_exchange.ask
                        best_opportunity = ArbitrageOpportunity(
                            arb_type="cross_exchange",
                            spread_pct=spread_pct,
                            expected_profit=expected,
                            transaction_cost=cost,
                            net_profit=net,
                            buy_venue=buy_exchange.exchange,
                            sell_venue=sell_exchange.exchange,
                            buy_price=buy_exchange.ask,
                            sell_price=sell_exchange.bid,
                            quantity=quantity,
                            confidence=min(0.85, 0.4 + spread_pct * 0.15),
                            latency_ms=max(buy_exchange.latency_ms, sell_exchange.latency_ms),
                        )

        return best_opportunity

    # ------------------------------------------------------------------
    # Triangular arbitrage
    # ------------------------------------------------------------------

    def _scan_triangular(
        self,
        pair_a: str,   # e.g. "BTCUSDT"
        pair_b: str,   # e.g. "ETHBTC"
        pair_c: str,   # e.g. "ETHUSDT"
    ) -> Optional[ArbitrageOpportunity]:
        """
        Triangular arbitrage: check if going through three pairs
        yields a profit. Example cycle:
        USDT -> BTC -> ETH -> USDT

        The prices must form a profitable cycle:
        1 USDT buys (1/ask_BTCUSDT) BTC
        That BTC buys (amount * bid_ETHBTC) ETH -- wait, this is backwards
        Actually: 1 USDT -> buy BTC at ask -> sell BTC for ETH at bid -> sell ETH at bid

        Forward: USDT -> BTC/USDT(ask) -> ETH/BTC(ask) -> ETH/USDT(bid)
        """
        key_a = None
        key_b = None
        key_c = None

        for key, ep in self._exchange_prices.items():
            if ep.symbol == pair_a and self._is_price_fresh(ep):
                key_a = ep
            elif ep.symbol == pair_b and self._is_price_fresh(ep):
                key_b = ep
            elif ep.symbol == pair_c and self._is_price_fresh(ep):
                key_c = ep

        if not (key_a and key_b and key_c):
            return None

        # Forward path: USDT -> buy BTC -> buy ETH with BTC -> sell ETH for USDT
        start_amount = 1000.0  # USDT
        btc_amount = start_amount / key_a.ask           # buy BTC with USDT
        eth_amount = btc_amount / key_b.ask              # buy ETH with BTC
        end_amount = eth_amount * key_c.bid              # sell ETH for USDT

        forward_profit_pct = ((end_amount - start_amount) / start_amount) * 100.0

        # Reverse path: USDT -> buy ETH -> sell ETH for BTC -> sell BTC for USDT
        eth_amount_r = start_amount / key_c.ask
        btc_amount_r = eth_amount_r * key_b.bid
        end_amount_r = btc_amount_r * key_a.bid

        reverse_profit_pct = ((end_amount_r - start_amount) / start_amount) * 100.0

        # Use the more profitable direction
        if forward_profit_pct > reverse_profit_pct and forward_profit_pct > self.min_triangular_spread:
            notional = min(self.max_per_trade, self.max_exposure - self._total_exposure)
            cost = self._calculate_round_trip_cost(notional) * 1.5  # 3 trades
            expected = notional * (forward_profit_pct / 100.0)
            net = expected - cost
            if net > 0:
                return ArbitrageOpportunity(
                    arb_type="triangular",
                    spread_pct=forward_profit_pct,
                    expected_profit=expected,
                    transaction_cost=cost,
                    net_profit=net,
                    buy_venue="forward",
                    sell_venue=f"{pair_a}->{pair_b}->{pair_c}",
                    buy_price=key_a.ask,
                    sell_price=key_c.bid,
                    quantity=notional / key_a.ask,
                    confidence=min(0.8, 0.3 + forward_profit_pct * 0.2),
                )

        elif reverse_profit_pct > self.min_triangular_spread:
            notional = min(self.max_per_trade, self.max_exposure - self._total_exposure)
            cost = self._calculate_round_trip_cost(notional) * 1.5
            expected = notional * (reverse_profit_pct / 100.0)
            net = expected - cost
            if net > 0:
                return ArbitrageOpportunity(
                    arb_type="triangular",
                    spread_pct=reverse_profit_pct,
                    expected_profit=expected,
                    transaction_cost=cost,
                    net_profit=net,
                    buy_venue="reverse",
                    sell_venue=f"{pair_c}->{pair_b}->{pair_a}",
                    buy_price=key_c.ask,
                    sell_price=key_a.bid,
                    quantity=notional / key_c.ask,
                    confidence=min(0.8, 0.3 + reverse_profit_pct * 0.2),
                )

        return None

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def generate_signal(
        self, candles: np.ndarray,
    ) -> Optional[StrategySignal]:
        """
        Scan all arbitrage types and return the best opportunity.
        Candles are used mainly for spot price reference.
        """
        if len(candles) < 5:
            return None

        close = candles[:, 4].astype(np.float64)
        current_price = float(close[-1])

        # Update spot price from candles
        self._spot_prices[self.symbol] = current_price

        # Check exposure limit
        if self._total_exposure >= self.max_exposure:
            return None
        if len(self._open_arbs) >= self.max_open_arbs:
            return None

        # Scan all arb types (priority order)
        best: Optional[ArbitrageOpportunity] = None

        # 1. Cross-exchange (fastest to expire)
        cross = self._scan_cross_exchange()
        if cross and (best is None or cross.net_profit > best.net_profit):
            best = cross

        # 2. Basis spread
        basis = self._scan_basis_spread()
        if basis and (best is None or basis.net_profit > best.net_profit):
            best = basis

        # 3. Funding rate
        funding = self._scan_funding_arb()
        if funding and (best is None or funding.net_profit > best.net_profit):
            best = funding

        # 4. Triangular (check common triangles)
        triangles = [
            ("BTCUSDT", "ETHBTC", "ETHUSDT"),
            ("BTCUSDT", "BNBBTC", "BNBUSDT"),
            ("ETHUSDT", "BNBETH", "BNBUSDT"),
        ]
        for a, b, c in triangles:
            tri = self._scan_triangular(a, b, c)
            if tri and (best is None or tri.net_profit > best.net_profit):
                best = tri

        if best is None:
            return None

        # Convert to signal
        self._open_arbs.append(best)
        self._total_exposure += best.buy_price * best.quantity

        signal_type = SignalType.LONG  # buy side of the arb
        signal = StrategySignal(
            signal_type=signal_type,
            symbol=self.symbol,
            price=best.buy_price,
            quantity=best.quantity,
            confidence=best.confidence,
            reason=f"Arb {best.arb_type}: spread={best.spread_pct:.3f}% "
                   f"net=${best.net_profit:.2f} ({best.buy_venue}->{best.sell_venue})",
            metadata={
                "arb_type": best.arb_type,
                "spread_pct": best.spread_pct,
                "expected_profit": best.expected_profit,
                "transaction_cost": best.transaction_cost,
                "net_profit": best.net_profit,
                "buy_venue": best.buy_venue,
                "sell_venue": best.sell_venue,
                "buy_price": best.buy_price,
                "sell_price": best.sell_price,
                "latency_ms": best.latency_ms,
            },
        )
        return signal

    def on_tick(
        self, price: float, volume: float, timestamp: float,
    ) -> Optional[StrategySignal]:
        self._spot_prices[self.symbol] = price
        return None

    def on_bar(self, ohlcv: Dict[str, float]) -> Optional[StrategySignal]:
        close = ohlcv.get("close", 0.0)
        if close > 0:
            self._spot_prices[self.symbol] = close
        return None

    def on_fill(self, order_info: Dict[str, Any]) -> None:
        self._logger.info("Arb fill: %s", order_info)

    def get_arb_status(self) -> Dict[str, Any]:
        return {
            "open_arbs": len(self._open_arbs),
            "total_exposure": self._total_exposure,
            "max_exposure": self.max_exposure,
            "exchange_prices": len(self._exchange_prices),
            "funding_rates": {
                k: {"rate": v.rate, "annualized": v.rate * 1095 * 100}
                for k, v in self._funding_rates.items()
            },
            "history_count": len(self._arb_history),
        }
