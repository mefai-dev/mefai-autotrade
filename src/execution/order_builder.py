"""
Mefai Autotrade - Order Builder
Constructs complex order types: OCO, bracket, scaled entry, grid,
and hedge orders. Handles exchange rule validation, lot size rounding,
price tick rounding, and minimum notional checks.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exchange rules
# ---------------------------------------------------------------------------

@dataclass
class ExchangeRules:
    """Exchange trading rules for a symbol."""
    symbol: str
    # Price rules
    tick_size: float = 0.01          # Minimum price increment
    min_price: float = 0.0
    max_price: float = 0.0
    # Quantity rules
    lot_size: float = 0.001          # Minimum quantity increment
    min_quantity: float = 0.001
    max_quantity: float = 0.0        # 0 = no limit
    # Notional rules
    min_notional: float = 5.0        # Minimum order value in quote currency
    max_notional: float = 0.0        # 0 = no limit
    # Order type support
    supported_order_types: List[str] = field(default_factory=lambda: [
        "LIMIT", "MARKET", "STOP_MARKET", "STOP_LIMIT",
        "TAKE_PROFIT_MARKET", "TAKE_PROFIT_LIMIT",
    ])
    # Other
    max_open_orders: int = 200
    base_asset_precision: int = 8
    quote_asset_precision: int = 8
    multiplier: float = 1.0          # Contract multiplier (futures)

    def round_price(self, price: float) -> float:
        """Round price to valid tick size."""
        if self.tick_size <= 0:
            return price
        # Use Decimal-style rounding to avoid floating point issues
        ticks = round(price / self.tick_size)
        result = ticks * self.tick_size
        # Round to appropriate precision
        decimals = self._count_decimals(self.tick_size)
        return round(result, decimals)

    def round_quantity(self, quantity: float) -> float:
        """Round quantity to valid lot size."""
        if self.lot_size <= 0:
            return quantity
        # Always round DOWN for quantity (don't exceed intended quantity)
        lots = math.floor(quantity / self.lot_size)
        result = lots * self.lot_size
        decimals = self._count_decimals(self.lot_size)
        return round(result, decimals)

    def round_quantity_up(self, quantity: float) -> float:
        """Round quantity UP to valid lot size."""
        if self.lot_size <= 0:
            return quantity
        lots = math.ceil(quantity / self.lot_size)
        result = lots * self.lot_size
        decimals = self._count_decimals(self.lot_size)
        return round(result, decimals)

    def validate_price(self, price: float) -> Tuple[bool, str]:
        """Validate a price against exchange rules."""
        if price <= 0:
            return False, "Price must be positive"
        if self.min_price > 0 and price < self.min_price:
            return False, f"Price {price} below minimum {self.min_price}"
        if self.max_price > 0 and price > self.max_price:
            return False, f"Price {price} above maximum {self.max_price}"
        rounded = self.round_price(price)
        if abs(rounded - price) > self.tick_size * 0.01:
            return False, f"Price {price} not on tick grid (tick={self.tick_size})"
        return True, ""

    def validate_quantity(self, quantity: float) -> Tuple[bool, str]:
        """Validate quantity against exchange rules."""
        if quantity <= 0:
            return False, "Quantity must be positive"
        if quantity < self.min_quantity:
            return False, f"Quantity {quantity} below minimum {self.min_quantity}"
        if self.max_quantity > 0 and quantity > self.max_quantity:
            return False, f"Quantity {quantity} above maximum {self.max_quantity}"
        return True, ""

    def validate_notional(self, quantity: float, price: float) -> Tuple[bool, str]:
        """Validate notional value."""
        notional = quantity * price * self.multiplier
        if notional < self.min_notional:
            return False, f"Notional {notional:.2f} below minimum {self.min_notional}"
        if self.max_notional > 0 and notional > self.max_notional:
            return False, f"Notional {notional:.2f} above maximum {self.max_notional}"
        return True, ""

    def validate_order(self, side: str, order_type: str,
                       quantity: float, price: float = 0.0) -> Tuple[bool, List[str]]:
        """Full validation of an order. Returns (valid, list_of_errors)."""
        errors = []
        if side not in ("BUY", "SELL"):
            errors.append(f"Invalid side: {side}")
        if order_type not in self.supported_order_types:
            errors.append(f"Unsupported order type: {order_type}")
        valid_qty, msg = self.validate_quantity(quantity)
        if not valid_qty:
            errors.append(msg)
        if price > 0:
            valid_price, msg = self.validate_price(price)
            if not valid_price:
                errors.append(msg)
            valid_notional, msg = self.validate_notional(quantity, price)
            if not valid_notional:
                errors.append(msg)
        return len(errors) == 0, errors

    @staticmethod
    def _count_decimals(value: float) -> int:
        """Count decimal places in a float value."""
        s = f"{value:.10f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0


# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------

@dataclass
class OrderParams:
    """Generic order parameters."""
    symbol: str
    side: str
    quantity: float
    order_type: str
    price: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    time_in_force: str = "GTC"
    reduce_only: bool = False
    close_position: bool = False
    callback_rate: float = 0.0    # For trailing stop
    working_type: str = "CONTRACT_PRICE"
    label: str = ""               # User-defined label
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BracketOrder:
    """Bracket order: entry + stop loss + take profit."""
    entry: OrderParams
    stop_loss: OrderParams
    take_profit: OrderParams
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def get_all_orders(self) -> List[OrderParams]:
        return [self.entry, self.stop_loss, self.take_profit]

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk-reward ratio."""
        if self.entry.price <= 0:
            return 0.0
        if self.entry.side == "BUY":
            risk = self.entry.price - self.stop_loss.stop_price
            reward = self.take_profit.price - self.entry.price
        else:
            risk = self.stop_loss.stop_price - self.entry.price
            reward = self.entry.price - self.take_profit.price
        if risk <= 0:
            return 0.0
        return reward / risk


@dataclass
class OCOOrder:
    """One-Cancels-Other order."""
    order_a: OrderParams  # Usually take profit (limit)
    order_b: OrderParams  # Usually stop loss (stop market)
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def get_all_orders(self) -> List[OrderParams]:
        return [self.order_a, self.order_b]


@dataclass
class ScaledOrder:
    """Scaled entry with multiple orders at different price levels."""
    orders: List[OrderParams]
    total_quantity: float = 0.0
    avg_entry_price: float = 0.0
    price_range_pct: float = 0.0
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def get_all_orders(self) -> List[OrderParams]:
        return list(self.orders)

    def compute_stats(self) -> None:
        """Compute aggregate statistics."""
        self.total_quantity = sum(o.quantity for o in self.orders)
        if self.total_quantity > 0:
            self.avg_entry_price = sum(
                o.price * o.quantity for o in self.orders
            ) / self.total_quantity
        if self.orders:
            prices = [o.price for o in self.orders if o.price > 0]
            if len(prices) >= 2:
                self.price_range_pct = (max(prices) - min(prices)) / min(prices) * 100


@dataclass
class GridOrderSet:
    """Grid of buy and sell orders at regular price intervals."""
    buy_orders: List[OrderParams]
    sell_orders: List[OrderParams]
    grid_spacing_pct: float = 0.0
    num_levels: int = 0
    upper_price: float = 0.0
    lower_price: float = 0.0
    total_buy_quantity: float = 0.0
    total_sell_quantity: float = 0.0
    total_investment: float = 0.0
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def get_all_orders(self) -> List[OrderParams]:
        return self.buy_orders + self.sell_orders


@dataclass
class HedgeOrder:
    """Hedge order - opposite position on same or different venue."""
    primary: OrderParams
    hedge: OrderParams
    hedge_venue_id: Optional[str] = None
    hedge_ratio: float = 1.0       # 1.0 = full hedge
    correlation: float = 1.0       # Expected correlation between instruments
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def get_all_orders(self) -> List[OrderParams]:
        return [self.primary, self.hedge]


# ---------------------------------------------------------------------------
# Order Builder
# ---------------------------------------------------------------------------

class OrderBuilder:
    """Builds complex order structures with exchange rule validation.

    All prices and quantities are automatically rounded to exchange
    precision, and orders are validated against minimum notional,
    lot size, and tick size rules.
    """

    def __init__(self):
        self._exchange_rules: Dict[str, ExchangeRules] = {}
        logger.info("OrderBuilder initialized")

    def set_rules(self, symbol: str, rules: ExchangeRules) -> None:
        """Set exchange rules for a symbol."""
        self._exchange_rules[symbol] = rules

    def get_rules(self, symbol: str) -> Optional[ExchangeRules]:
        """Get exchange rules for a symbol."""
        return self._exchange_rules.get(symbol)

    def _get_or_default_rules(self, symbol: str) -> ExchangeRules:
        """Get rules or create default."""
        rules = self._exchange_rules.get(symbol)
        if not rules:
            rules = ExchangeRules(symbol=symbol)
        return rules

    # -----------------------------------------------------------------------
    # Bracket orders
    # -----------------------------------------------------------------------

    def build_bracket(self, symbol: str, side: str, quantity: float,
                      entry_price: float, stop_loss_price: float,
                      take_profit_price: float,
                      entry_type: str = "LIMIT",
                      time_in_force: str = "GTC") -> BracketOrder:
        """Build a bracket order (entry + SL + TP).

        Args:
            symbol: Trading pair
            side: BUY or SELL for the entry
            quantity: Position size
            entry_price: Entry limit price
            stop_loss_price: Stop loss trigger price
            take_profit_price: Take profit price
            entry_type: LIMIT or MARKET for entry
            time_in_force: Time in force for entry order
        """
        rules = self._get_or_default_rules(symbol)
        errors = []

        # Round values
        qty = rules.round_quantity(quantity)
        entry_p = rules.round_price(entry_price)
        sl_p = rules.round_price(stop_loss_price)
        tp_p = rules.round_price(take_profit_price)

        # Validate bracket logic
        exit_side = "SELL" if side == "BUY" else "BUY"
        if side == "BUY":
            if sl_p >= entry_p:
                errors.append(f"Stop loss {sl_p} must be below entry {entry_p} for BUY")
            if tp_p <= entry_p:
                errors.append(f"Take profit {tp_p} must be above entry {entry_p} for BUY")
        else:
            if sl_p <= entry_p:
                errors.append(f"Stop loss {sl_p} must be above entry {entry_p} for SELL")
            if tp_p >= entry_p:
                errors.append(f"Take profit {tp_p} must be below entry {entry_p} for SELL")

        # Validate against exchange rules
        valid, qty_errors = rules.validate_order(side, entry_type, qty, entry_p)
        errors.extend(qty_errors)

        entry = OrderParams(
            symbol=symbol, side=side, quantity=qty,
            order_type=entry_type, price=entry_p,
            time_in_force=time_in_force, label="bracket_entry",
        )
        stop_loss = OrderParams(
            symbol=symbol, side=exit_side, quantity=qty,
            order_type="STOP_MARKET", stop_price=sl_p,
            time_in_force="GTC", reduce_only=True, label="bracket_sl",
        )
        take_profit = OrderParams(
            symbol=symbol, side=exit_side, quantity=qty,
            order_type="TAKE_PROFIT_MARKET", price=tp_p,
            time_in_force="GTC", reduce_only=True, label="bracket_tp",
        )

        bracket = BracketOrder(
            entry=entry, stop_loss=stop_loss, take_profit=take_profit,
            valid=len(errors) == 0, validation_errors=errors,
        )
        return bracket

    # -----------------------------------------------------------------------
    # OCO orders
    # -----------------------------------------------------------------------

    def build_oco(self, symbol: str, side: str, quantity: float,
                  limit_price: float, stop_price: float,
                  time_in_force: str = "GTC") -> OCOOrder:
        """Build a One-Cancels-Other order.

        Typically used for exit: limit order at take profit + stop at stop loss.

        Args:
            symbol: Trading pair
            side: BUY or SELL (exit side)
            quantity: Quantity for each leg
            limit_price: Limit order price (take profit)
            stop_price: Stop order trigger price (stop loss)
        """
        rules = self._get_or_default_rules(symbol)
        errors = []

        qty = rules.round_quantity(quantity)
        lp = rules.round_price(limit_price)
        sp = rules.round_price(stop_price)

        # Validate OCO logic
        if side == "SELL":
            if sp >= lp:
                errors.append(f"For SELL OCO: stop {sp} must be below limit {lp}")
        else:
            if sp <= lp:
                errors.append(f"For BUY OCO: stop {sp} must be above limit {lp}")

        valid, order_errors = rules.validate_order(side, "LIMIT", qty, lp)
        errors.extend(order_errors)

        order_a = OrderParams(
            symbol=symbol, side=side, quantity=qty,
            order_type="LIMIT", price=lp,
            time_in_force=time_in_force, reduce_only=True,
            label="oco_limit",
        )
        order_b = OrderParams(
            symbol=symbol, side=side, quantity=qty,
            order_type="STOP_MARKET", stop_price=sp,
            time_in_force="GTC", reduce_only=True,
            label="oco_stop",
        )

        return OCOOrder(
            order_a=order_a, order_b=order_b,
            valid=len(errors) == 0, validation_errors=errors,
        )

    # -----------------------------------------------------------------------
    # Scaled entry orders
    # -----------------------------------------------------------------------

    def build_scaled_entry(self, symbol: str, side: str,
                           total_quantity: float,
                           price_start: float, price_end: float,
                           num_orders: int = 5,
                           distribution: str = "linear",
                           time_in_force: str = "GTC") -> ScaledOrder:
        """Build scaled entry orders at multiple price levels.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            total_quantity: Total quantity across all orders
            price_start: First price level (closest to market)
            price_end: Last price level (furthest from market)
            num_orders: Number of orders to create
            distribution: "linear", "exponential", or "fibonacci"
        """
        rules = self._get_or_default_rules(symbol)
        errors = []

        if num_orders < 2:
            num_orders = 2
        if num_orders > 50:
            num_orders = 50

        # Generate price levels
        prices = self._generate_price_levels(
            price_start, price_end, num_orders, distribution
        )

        # Generate quantity distribution
        quantities = self._generate_quantity_distribution(
            total_quantity, num_orders, distribution
        )

        orders = []
        for i, (price, qty) in enumerate(zip(prices, quantities)):
            p = rules.round_price(price)
            q = rules.round_quantity(qty)

            # Check minimum notional
            if q > 0 and p > 0:
                notional = q * p
                if notional < rules.min_notional:
                    # Increase quantity to meet minimum
                    min_qty = rules.min_notional / p
                    q = rules.round_quantity_up(min_qty)

            valid, order_errors = rules.validate_order(side, "LIMIT", q, p)
            if not valid:
                errors.extend([f"Order {i+1}: {e}" for e in order_errors])

            order = OrderParams(
                symbol=symbol, side=side, quantity=q,
                order_type="LIMIT", price=p,
                time_in_force=time_in_force,
                label=f"scaled_{i+1}of{num_orders}",
            )
            orders.append(order)

        scaled = ScaledOrder(
            orders=orders,
            valid=len(errors) == 0,
            validation_errors=errors,
        )
        scaled.compute_stats()
        return scaled

    def _generate_price_levels(self, start: float, end: float,
                                num: int, distribution: str) -> List[float]:
        """Generate price levels based on distribution type."""
        if distribution == "exponential":
            if start <= 0 or end <= 0:
                return [start + (end - start) * i / (num - 1) for i in range(num)]
            log_start = math.log(start)
            log_end = math.log(end)
            return [
                math.exp(log_start + (log_end - log_start) * i / (num - 1))
                for i in range(num)
            ]

        elif distribution == "fibonacci":
            # Fibonacci spacing: wider gaps as we go further
            fibs = [1, 1]
            while len(fibs) < num:
                fibs.append(fibs[-1] + fibs[-2])
            fibs = fibs[:num]
            total_fib = sum(fibs)
            cumulative = 0.0
            prices = []
            for f in fibs:
                cumulative += f
                frac = cumulative / total_fib
                price = start + (end - start) * frac
                prices.append(price)
            return prices

        else:  # linear
            if num == 1:
                return [start]
            return [start + (end - start) * i / (num - 1) for i in range(num)]

    def _generate_quantity_distribution(self, total: float, num: int,
                                         distribution: str) -> List[float]:
        """Generate quantity distribution across orders."""
        if distribution == "exponential":
            # More quantity at better prices (further from market)
            weights = [math.exp(0.5 * i) for i in range(num)]
        elif distribution == "fibonacci":
            fibs = [1, 1]
            while len(fibs) < num:
                fibs.append(fibs[-1] + fibs[-2])
            weights = fibs[:num]
        else:  # linear - equal distribution
            weights = [1.0] * num

        total_weight = sum(weights)
        return [total * w / total_weight for w in weights]

    # -----------------------------------------------------------------------
    # Grid orders
    # -----------------------------------------------------------------------

    def build_grid(self, symbol: str, upper_price: float,
                   lower_price: float, num_grids: int,
                   total_investment: float,
                   current_price: float,
                   grid_type: str = "arithmetic",
                   quantity_per_grid: float = 0.0) -> GridOrderSet:
        """Build a grid of buy and sell orders.

        Args:
            symbol: Trading pair
            upper_price: Upper bound of grid
            lower_price: Lower bound of grid
            num_grids: Number of grid lines
            total_investment: Total capital to deploy
            current_price: Current market price
            grid_type: "arithmetic" (equal price gaps) or "geometric" (equal % gaps)
            quantity_per_grid: Fixed quantity per grid level (0 = auto from investment)
        """
        rules = self._get_or_default_rules(symbol)
        errors = []

        if num_grids < 2:
            errors.append("Need at least 2 grid levels")
            return GridOrderSet(
                buy_orders=[], sell_orders=[], valid=False,
                validation_errors=errors,
            )

        if upper_price <= lower_price:
            errors.append("Upper price must be above lower price")
            return GridOrderSet(
                buy_orders=[], sell_orders=[], valid=False,
                validation_errors=errors,
            )

        # Generate grid levels
        if grid_type == "geometric":
            ratio = (upper_price / lower_price) ** (1.0 / num_grids)
            levels = [lower_price * (ratio ** i) for i in range(num_grids + 1)]
        else:
            step = (upper_price - lower_price) / num_grids
            levels = [lower_price + step * i for i in range(num_grids + 1)]

        levels = [rules.round_price(l) for l in levels]

        # Calculate quantity per grid
        if quantity_per_grid <= 0:
            avg_price = (upper_price + lower_price) / 2
            if avg_price > 0 and num_grids > 0:
                quantity_per_grid = total_investment / (num_grids * avg_price)
            else:
                quantity_per_grid = 0

        qty_per_grid = rules.round_quantity(quantity_per_grid)

        # Build orders: buy below current price, sell above
        buy_orders = []
        sell_orders = []
        grid_spacing = (upper_price - lower_price) / num_grids
        grid_spacing_pct = grid_spacing / lower_price * 100.0 if lower_price > 0 else 0.0

        for level in levels:
            if level < current_price * 0.999:
                # Buy order below market
                valid, order_errors = rules.validate_order("BUY", "LIMIT", qty_per_grid, level)
                if not valid:
                    errors.extend(order_errors)
                order = OrderParams(
                    symbol=symbol, side="BUY", quantity=qty_per_grid,
                    order_type="LIMIT", price=level, time_in_force="GTC",
                    label=f"grid_buy_{level:.2f}",
                )
                buy_orders.append(order)
            elif level > current_price * 1.001:
                # Sell order above market
                valid, order_errors = rules.validate_order("SELL", "LIMIT", qty_per_grid, level)
                if not valid:
                    errors.extend(order_errors)
                order = OrderParams(
                    symbol=symbol, side="SELL", quantity=qty_per_grid,
                    order_type="LIMIT", price=level, time_in_force="GTC",
                    label=f"grid_sell_{level:.2f}",
                )
                sell_orders.append(order)

        total_buy = sum(o.quantity * o.price for o in buy_orders)
        total_sell_qty = sum(o.quantity for o in sell_orders)

        return GridOrderSet(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            grid_spacing_pct=grid_spacing_pct,
            num_levels=num_grids,
            upper_price=upper_price,
            lower_price=lower_price,
            total_buy_quantity=sum(o.quantity for o in buy_orders),
            total_sell_quantity=total_sell_qty,
            total_investment=total_buy,
            valid=len(errors) == 0,
            validation_errors=errors,
        )

    # -----------------------------------------------------------------------
    # Hedge orders
    # -----------------------------------------------------------------------

    def build_hedge(self, primary_symbol: str, primary_side: str,
                    primary_quantity: float, primary_price: float,
                    hedge_symbol: Optional[str] = None,
                    hedge_venue_id: Optional[str] = None,
                    hedge_ratio: float = 1.0,
                    hedge_price: float = 0.0,
                    correlation: float = 1.0) -> HedgeOrder:
        """Build a hedge order pair.

        Args:
            primary_symbol: Primary position symbol
            primary_side: BUY or SELL
            primary_quantity: Primary position quantity
            primary_price: Primary order price
            hedge_symbol: Hedge instrument (None = same symbol)
            hedge_venue_id: Venue for hedge (None = same venue)
            hedge_ratio: Hedge ratio (1.0 = full hedge)
            hedge_price: Hedge order price (0 = market)
            correlation: Expected correlation between instruments
        """
        hedge_sym = hedge_symbol or primary_symbol
        hedge_side = "SELL" if primary_side == "BUY" else "BUY"
        hedge_qty = primary_quantity * hedge_ratio

        if correlation != 1.0 and correlation > 0:
            hedge_qty = hedge_qty / correlation

        rules_primary = self._get_or_default_rules(primary_symbol)
        rules_hedge = self._get_or_default_rules(hedge_sym)

        errors = []
        p_qty = rules_primary.round_quantity(primary_quantity)
        p_price = rules_primary.round_price(primary_price)
        h_qty = rules_hedge.round_quantity(hedge_qty)
        h_price = rules_hedge.round_price(hedge_price) if hedge_price > 0 else 0.0

        valid_p, p_errors = rules_primary.validate_order(
            primary_side, "LIMIT", p_qty, p_price
        )
        errors.extend(p_errors)

        h_type = "LIMIT" if h_price > 0 else "MARKET"
        if h_price > 0:
            valid_h, h_errors = rules_hedge.validate_order(
                hedge_side, h_type, h_qty, h_price
            )
            errors.extend(h_errors)

        primary = OrderParams(
            symbol=primary_symbol, side=primary_side, quantity=p_qty,
            order_type="LIMIT", price=p_price, time_in_force="GTC",
            label="hedge_primary",
        )
        hedge = OrderParams(
            symbol=hedge_sym, side=hedge_side, quantity=h_qty,
            order_type=h_type, price=h_price, time_in_force="GTC" if h_price > 0 else "IOC",
            label="hedge_offset",
        )

        return HedgeOrder(
            primary=primary, hedge=hedge,
            hedge_venue_id=hedge_venue_id,
            hedge_ratio=hedge_ratio,
            correlation=correlation,
            valid=len(errors) == 0,
            validation_errors=errors,
        )

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------

    def validate_order_params(self, params: OrderParams) -> Tuple[bool, List[str]]:
        """Validate order parameters against exchange rules."""
        rules = self._get_or_default_rules(params.symbol)
        return rules.validate_order(
            params.side, params.order_type, params.quantity, params.price
        )

    def round_order_params(self, params: OrderParams) -> OrderParams:
        """Round order parameters to exchange precision."""
        rules = self._get_or_default_rules(params.symbol)
        params.quantity = rules.round_quantity(params.quantity)
        if params.price > 0:
            params.price = rules.round_price(params.price)
        if params.stop_price > 0:
            params.stop_price = rules.round_price(params.stop_price)
        if params.take_profit_price > 0:
            params.take_profit_price = rules.round_price(params.take_profit_price)
        return params

    def compute_min_quantity(self, symbol: str, price: float) -> float:
        """Compute minimum order quantity given a price."""
        rules = self._get_or_default_rules(symbol)
        min_qty = rules.min_quantity
        if price > 0 and rules.min_notional > 0:
            notional_min_qty = rules.min_notional / price
            min_qty = max(min_qty, notional_min_qty)
        return rules.round_quantity_up(min_qty)

    def compute_max_quantity(self, symbol: str, price: float,
                             available_balance: float) -> float:
        """Compute maximum order quantity given balance."""
        rules = self._get_or_default_rules(symbol)
        if price <= 0:
            return 0.0
        max_qty = available_balance / price
        if rules.max_quantity > 0:
            max_qty = min(max_qty, rules.max_quantity)
        return rules.round_quantity(max_qty)

    def get_supported_symbols(self) -> List[str]:
        """Get all symbols with registered rules."""
        return list(self._exchange_rules.keys())

    def get_rules_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get a summary of exchange rules for a symbol."""
        rules = self._exchange_rules.get(symbol)
        if not rules:
            return None
        return {
            "symbol": rules.symbol,
            "tick_size": rules.tick_size,
            "lot_size": rules.lot_size,
            "min_quantity": rules.min_quantity,
            "max_quantity": rules.max_quantity,
            "min_notional": rules.min_notional,
            "supported_types": rules.supported_order_types,
            "multiplier": rules.multiplier,
        }
