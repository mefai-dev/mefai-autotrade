"""
Account Manager - Account state management for Mefai Autotrade.
Handles balance tracking, margin calculations, leverage management,
multi-account support, and account snapshots.
"""

import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mefai.autotrade.account_manager")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarginMode(Enum):
    CROSS = "CROSS"
    ISOLATED = "ISOLATED"


class BalanceChangeType(Enum):
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    TRADE_PNL = "TRADE_PNL"
    COMMISSION = "COMMISSION"
    FUNDING = "FUNDING"
    TRANSFER = "TRANSFER"
    MARGIN_LOCK = "MARGIN_LOCK"
    MARGIN_RELEASE = "MARGIN_RELEASE"
    LIQUIDATION = "LIQUIDATION"
    ADJUSTMENT = "ADJUSTMENT"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AccountBalance:
    """Balance state for a single asset."""
    asset: str = "USDT"
    available: float = 0.0
    locked: float = 0.0  # Locked in open orders / margin

    @property
    def total(self) -> float:
        return self.available + self.locked

    def lock(self, amount: float) -> bool:
        if amount > self.available:
            return False
        self.available -= amount
        self.locked += amount
        return True

    def unlock(self, amount: float) -> bool:
        if amount > self.locked:
            amount = self.locked
        self.locked -= amount
        self.available += amount
        return True

    def add(self, amount: float) -> None:
        self.available += amount

    def deduct(self, amount: float) -> bool:
        if amount > self.available:
            return False
        self.available -= amount
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "available": self.available,
            "locked": self.locked,
            "total": self.total,
        }


@dataclass
class BalanceChange:
    """Record of a balance change."""
    change_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    asset: str = "USDT"
    amount: float = 0.0
    change_type: BalanceChangeType = BalanceChangeType.ADJUSTMENT
    before_balance: float = 0.0
    after_balance: float = 0.0
    reference_id: str = ""  # Order ID, position ID, etc.
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "amount": self.amount,
            "change_type": self.change_type.value,
            "before_balance": self.before_balance,
            "after_balance": self.after_balance,
            "reference_id": self.reference_id,
            "notes": self.notes,
        }


@dataclass
class MarginInfo:
    """Margin state for the account."""
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    margin_balance: float = 0.0
    available_margin: float = 0.0
    margin_ratio: float = 0.0  # Percentage
    margin_mode: MarginMode = MarginMode.CROSS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
            "margin_balance": self.margin_balance,
            "available_margin": self.available_margin,
            "margin_ratio": self.margin_ratio,
            "margin_mode": self.margin_mode.value,
        }


@dataclass
class AccountSnapshot:
    """Point-in-time account snapshot for history tracking."""
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    account_id: str = ""
    total_balance: float = 0.0
    available_balance: float = 0.0
    locked_balance: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    margin_ratio: float = 0.0
    open_positions: int = 0
    open_orders: int = 0
    equity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "account_id": self.account_id,
            "total_balance": self.total_balance,
            "available_balance": self.available_balance,
            "locked_balance": self.locked_balance,
            "unrealized_pnl": self.unrealized_pnl,
            "margin_used": self.margin_used,
            "margin_ratio": self.margin_ratio,
            "open_positions": self.open_positions,
            "open_orders": self.open_orders,
            "equity": self.equity,
        }


@dataclass
class LeverageConfig:
    """Leverage configuration per symbol."""
    symbol: str = ""
    leverage: int = 1
    max_leverage: int = 125
    margin_mode: MarginMode = MarginMode.CROSS
    max_notional: float = 0.0  # Max position notional at this leverage


@dataclass
class AccountConfig:
    """Configuration for a trading account."""
    account_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "default"
    exchange: str = "binance"
    account_type: str = "futures"  # spot, futures, margin
    default_leverage: int = 1
    default_margin_mode: MarginMode = MarginMode.CROSS
    max_open_positions: int = 50
    max_total_margin_pct: float = 80.0  # Max % of balance as margin
    max_single_position_pct: float = 20.0  # Max % of balance in one position
    risk_per_trade_pct: float = 1.0  # Max risk per trade
    is_active: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class Account:
    """
    Single trading account with balance management, margin tracking,
    and leverage configuration.
    """

    def __init__(self, config: AccountConfig):
        self.config = config
        self._lock = threading.RLock()

        # Balances by asset
        self._balances: Dict[str, AccountBalance] = {}

        # Leverage per symbol
        self._leverage_configs: Dict[str, LeverageConfig] = {}

        # Margin info
        self._margin = MarginInfo(margin_mode=config.default_margin_mode)

        # Balance change history
        self._balance_history: List[BalanceChange] = []
        self._max_history = 10000

        # Snapshots
        self._snapshots: List[AccountSnapshot] = []
        self._max_snapshots = 2000

        logger.info(
            "Account created: %s (%s/%s, leverage=%d)",
            config.account_id, config.exchange, config.account_type,
            config.default_leverage,
        )

    @property
    def account_id(self) -> str:
        return self.config.account_id

    @property
    def name(self) -> str:
        return self.config.name

    # -- Balance operations ---------------------------------------------------

    def get_balance(self, asset: str = "USDT") -> AccountBalance:
        with self._lock:
            if asset not in self._balances:
                self._balances[asset] = AccountBalance(asset=asset)
            return self._balances[asset]

    def get_all_balances(self) -> Dict[str, AccountBalance]:
        with self._lock:
            return dict(self._balances)

    def set_balance(
        self,
        asset: str,
        available: float,
        locked: float = 0.0,
    ) -> None:
        """Set balance directly (e.g., from exchange sync)."""
        with self._lock:
            bal = self.get_balance(asset)
            old_total = bal.total
            bal.available = available
            bal.locked = locked

            self._record_change(
                asset=asset,
                amount=bal.total - old_total,
                change_type=BalanceChangeType.ADJUSTMENT,
                before=old_total,
                after=bal.total,
                notes="Balance sync from exchange",
            )

    def deposit(
        self,
        amount: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")

        with self._lock:
            bal = self.get_balance(asset)
            before = bal.total
            bal.add(amount)

            self._record_change(
                asset=asset,
                amount=amount,
                change_type=BalanceChangeType.DEPOSIT,
                before=before,
                after=bal.total,
                reference_id=reference_id,
                notes=f"Deposit {amount:.4f} {asset}",
            )

        logger.info(
            "Account %s: deposited %.4f %s",
            self.account_id, amount, asset,
        )

    def withdraw(
        self,
        amount: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> bool:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")

        with self._lock:
            bal = self.get_balance(asset)
            before = bal.total
            if not bal.deduct(amount):
                logger.warning(
                    "Withdrawal failed: insufficient balance "
                    "(requested=%.4f, available=%.4f)",
                    amount, bal.available,
                )
                return False

            self._record_change(
                asset=asset,
                amount=-amount,
                change_type=BalanceChangeType.WITHDRAWAL,
                before=before,
                after=bal.total,
                reference_id=reference_id,
                notes=f"Withdrawal {amount:.4f} {asset}",
            )

        logger.info(
            "Account %s: withdrew %.4f %s",
            self.account_id, amount, asset,
        )
        return True

    def lock_margin(
        self,
        amount: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> bool:
        """Lock balance as margin for a position/order."""
        with self._lock:
            bal = self.get_balance(asset)
            before = bal.available
            if not bal.lock(amount):
                return False

            self._record_change(
                asset=asset,
                amount=-amount,
                change_type=BalanceChangeType.MARGIN_LOCK,
                before=before,
                after=bal.available,
                reference_id=reference_id,
                notes=f"Margin locked {amount:.4f} {asset}",
            )
        return True

    def release_margin(
        self,
        amount: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> bool:
        """Release locked margin."""
        with self._lock:
            bal = self.get_balance(asset)
            before = bal.available
            bal.unlock(amount)

            self._record_change(
                asset=asset,
                amount=amount,
                change_type=BalanceChangeType.MARGIN_RELEASE,
                before=before,
                after=bal.available,
                reference_id=reference_id,
                notes=f"Margin released {amount:.4f} {asset}",
            )
        return True

    def apply_pnl(
        self,
        pnl: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> None:
        """Apply trade PnL to available balance."""
        with self._lock:
            bal = self.get_balance(asset)
            before = bal.total
            bal.add(pnl)

            self._record_change(
                asset=asset,
                amount=pnl,
                change_type=BalanceChangeType.TRADE_PNL,
                before=before,
                after=bal.total,
                reference_id=reference_id,
                notes=f"Trade PnL {pnl:+.4f} {asset}",
            )

    def apply_commission(
        self,
        amount: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> None:
        """Deduct commission from available balance."""
        with self._lock:
            bal = self.get_balance(asset)
            before = bal.total
            bal.available -= amount

            self._record_change(
                asset=asset,
                amount=-amount,
                change_type=BalanceChangeType.COMMISSION,
                before=before,
                after=bal.total,
                reference_id=reference_id,
                notes=f"Commission {amount:.6f} {asset}",
            )

    def apply_funding(
        self,
        amount: float,
        asset: str = "USDT",
        reference_id: str = "",
    ) -> None:
        """Apply funding payment (positive = received, negative = paid)."""
        with self._lock:
            bal = self.get_balance(asset)
            before = bal.total
            bal.add(amount)

            self._record_change(
                asset=asset,
                amount=amount,
                change_type=BalanceChangeType.FUNDING,
                before=before,
                after=bal.total,
                reference_id=reference_id,
                notes=f"Funding {amount:+.4f} {asset}",
            )

    def _record_change(
        self,
        asset: str,
        amount: float,
        change_type: BalanceChangeType,
        before: float,
        after: float,
        reference_id: str = "",
        notes: str = "",
    ) -> None:
        """Record a balance change in history."""
        change = BalanceChange(
            asset=asset,
            amount=amount,
            change_type=change_type,
            before_balance=before,
            after_balance=after,
            reference_id=reference_id,
            notes=notes,
        )
        self._balance_history.append(change)
        if len(self._balance_history) > self._max_history:
            self._balance_history = self._balance_history[
                -self._max_history:
            ]

    # -- Leverage management --------------------------------------------------

    def set_leverage(
        self,
        symbol: str,
        leverage: int,
        max_notional: float = 0.0,
    ) -> None:
        if leverage < 1 or leverage > 125:
            raise ValueError(
                f"Leverage must be between 1 and 125 (got {leverage})"
            )
        with self._lock:
            config = self._leverage_configs.get(symbol, LeverageConfig(symbol=symbol))
            config.leverage = leverage
            if max_notional > 0:
                config.max_notional = max_notional
            self._leverage_configs[symbol] = config

        logger.info(
            "Account %s: leverage for %s set to %dx",
            self.account_id, symbol, leverage,
        )

    def get_leverage(self, symbol: str) -> int:
        with self._lock:
            config = self._leverage_configs.get(symbol)
            if config:
                return config.leverage
            return self.config.default_leverage

    def set_margin_mode(
        self,
        symbol: str,
        mode: MarginMode,
    ) -> None:
        with self._lock:
            config = self._leverage_configs.get(symbol, LeverageConfig(symbol=symbol))
            config.margin_mode = mode
            self._leverage_configs[symbol] = config

        logger.info(
            "Account %s: margin mode for %s set to %s",
            self.account_id, symbol, mode.value,
        )

    def get_margin_mode(self, symbol: str) -> MarginMode:
        with self._lock:
            config = self._leverage_configs.get(symbol)
            if config:
                return config.margin_mode
            return self.config.default_margin_mode

    # -- Margin calculations --------------------------------------------------

    def update_margin(
        self,
        initial_margin: float,
        maintenance_margin: float,
        unrealized_pnl: float = 0.0,
    ) -> MarginInfo:
        """Update margin information from position data."""
        with self._lock:
            usdt_balance = self.get_balance("USDT")
            margin_balance = usdt_balance.total + unrealized_pnl

            available_margin = margin_balance - initial_margin
            margin_ratio = (
                (maintenance_margin / margin_balance) * 100.0
                if margin_balance > 0 else 100.0
            )

            self._margin = MarginInfo(
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                margin_balance=margin_balance,
                available_margin=max(0.0, available_margin),
                margin_ratio=margin_ratio,
                margin_mode=self.config.default_margin_mode,
            )
            return MarginInfo(
                initial_margin=self._margin.initial_margin,
                maintenance_margin=self._margin.maintenance_margin,
                margin_balance=self._margin.margin_balance,
                available_margin=self._margin.available_margin,
                margin_ratio=self._margin.margin_ratio,
                margin_mode=self._margin.margin_mode,
            )

    def get_margin_info(self) -> MarginInfo:
        with self._lock:
            return MarginInfo(
                initial_margin=self._margin.initial_margin,
                maintenance_margin=self._margin.maintenance_margin,
                margin_balance=self._margin.margin_balance,
                available_margin=self._margin.available_margin,
                margin_ratio=self._margin.margin_ratio,
                margin_mode=self._margin.margin_mode,
            )

    def calculate_required_margin(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ) -> float:
        """Calculate required initial margin for a new position."""
        leverage = self.get_leverage(symbol)
        notional = quantity * price
        return notional / leverage

    def can_open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        asset: str = "USDT",
    ) -> Tuple[bool, str]:
        """Check if account has sufficient margin for a new position."""
        required = self.calculate_required_margin(symbol, quantity, price)

        with self._lock:
            bal = self.get_balance(asset)

            if required > bal.available:
                return False, (
                    f"Insufficient margin: need {required:.4f}, "
                    f"available {bal.available:.4f}"
                )

            # Check max margin usage
            total = bal.total
            total_margin = bal.locked + required
            margin_pct = (total_margin / total) * 100.0 if total > 0 else 100.0

            if margin_pct > self.config.max_total_margin_pct:
                return False, (
                    f"Would exceed max margin usage: "
                    f"{margin_pct:.1f}% > {self.config.max_total_margin_pct:.1f}%"
                )

            # Check single position size
            position_pct = (required / total) * 100.0 if total > 0 else 100.0
            if position_pct > self.config.max_single_position_pct:
                return False, (
                    f"Position too large: {position_pct:.1f}% > "
                    f"{self.config.max_single_position_pct:.1f}% limit"
                )

        return True, "OK"

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        risk_pct: Optional[float] = None,
        asset: str = "USDT",
    ) -> float:
        """
        Calculate position size based on risk percentage.
        Uses the distance to stop loss to determine size.
        """
        risk = risk_pct if risk_pct is not None else self.config.risk_per_trade_pct

        with self._lock:
            bal = self.get_balance(asset)
            risk_amount = bal.total * (risk / 100.0)

        price_diff = abs(entry_price - stop_loss_price)
        if price_diff <= 0:
            return 0.0

        raw_qty = risk_amount / price_diff
        return raw_qty

    # -- Snapshots ------------------------------------------------------------

    def take_snapshot(
        self,
        unrealized_pnl: float = 0.0,
        open_positions: int = 0,
        open_orders: int = 0,
    ) -> AccountSnapshot:
        with self._lock:
            bal = self.get_balance("USDT")
            equity = bal.total + unrealized_pnl

            snapshot = AccountSnapshot(
                account_id=self.account_id,
                total_balance=bal.total,
                available_balance=bal.available,
                locked_balance=bal.locked,
                unrealized_pnl=unrealized_pnl,
                margin_used=self._margin.initial_margin,
                margin_ratio=self._margin.margin_ratio,
                open_positions=open_positions,
                open_orders=open_orders,
                equity=equity,
            )

            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots = self._snapshots[-self._max_snapshots:]

            return snapshot

    def get_snapshots(
        self, limit: int = 100
    ) -> List[AccountSnapshot]:
        with self._lock:
            return list(self._snapshots[-limit:])

    def get_balance_history(
        self,
        asset: str = "USDT",
        change_type: Optional[BalanceChangeType] = None,
        limit: int = 100,
    ) -> List[BalanceChange]:
        with self._lock:
            changes = [
                c for c in self._balance_history
                if c.asset == asset
            ]
            if change_type:
                changes = [c for c in changes if c.change_type == change_type]
            return changes[-limit:]

    # -- Summary --------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            balances = {
                asset: bal.to_dict()
                for asset, bal in self._balances.items()
            }
            return {
                "account_id": self.account_id,
                "name": self.config.name,
                "exchange": self.config.exchange,
                "account_type": self.config.account_type,
                "is_active": self.config.is_active,
                "balances": balances,
                "margin": self._margin.to_dict(),
                "leverage_configs": {
                    sym: {
                        "leverage": lc.leverage,
                        "margin_mode": lc.margin_mode.value,
                    }
                    for sym, lc in self._leverage_configs.items()
                },
            }


# ---------------------------------------------------------------------------
# AccountManager - multi-account support
# ---------------------------------------------------------------------------

class AccountManager:
    """
    Manages multiple trading accounts. Supports switching between accounts,
    copy trading scenarios, and aggregate reporting.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._accounts: Dict[str, Account] = {}
        self._active_account_id: Optional[str] = None

        # Callbacks
        self._on_balance_change: List[Callable[[Account, BalanceChange], None]] = []
        self._on_margin_warning: List[Callable[[Account, MarginInfo], None]] = []

        logger.info("AccountManager initialized")

    # -- Callback registration -----------------------------------------------

    def on_balance_change(
        self, callback: Callable[[Account, BalanceChange], None]
    ) -> None:
        self._on_balance_change.append(callback)

    def on_margin_warning(
        self, callback: Callable[[Account, MarginInfo], None]
    ) -> None:
        self._on_margin_warning.append(callback)

    # -- Account management ---------------------------------------------------

    def create_account(
        self,
        name: str = "default",
        exchange: str = "binance",
        account_type: str = "futures",
        initial_balance: float = 0.0,
        default_leverage: int = 1,
        default_margin_mode: MarginMode = MarginMode.CROSS,
        max_open_positions: int = 50,
        max_total_margin_pct: float = 80.0,
        max_single_position_pct: float = 20.0,
        risk_per_trade_pct: float = 1.0,
    ) -> Account:
        config = AccountConfig(
            name=name,
            exchange=exchange,
            account_type=account_type,
            default_leverage=default_leverage,
            default_margin_mode=default_margin_mode,
            max_open_positions=max_open_positions,
            max_total_margin_pct=max_total_margin_pct,
            max_single_position_pct=max_single_position_pct,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        account = Account(config)

        if initial_balance > 0:
            account.deposit(initial_balance)

        with self._lock:
            self._accounts[account.account_id] = account
            if self._active_account_id is None:
                self._active_account_id = account.account_id

        logger.info(
            "Account created: %s (%s) with %.2f USDT",
            name, account.account_id, initial_balance,
        )
        return account

    def get_account(
        self, account_id: Optional[str] = None
    ) -> Optional[Account]:
        with self._lock:
            aid = account_id or self._active_account_id
            if aid is None:
                return None
            return self._accounts.get(aid)

    def get_active_account(self) -> Optional[Account]:
        return self.get_account()

    def set_active_account(self, account_id: str) -> bool:
        with self._lock:
            if account_id not in self._accounts:
                return False
            self._active_account_id = account_id
        logger.info("Active account set to %s", account_id)
        return True

    def get_all_accounts(self) -> List[Account]:
        with self._lock:
            return list(self._accounts.values())

    def remove_account(self, account_id: str) -> bool:
        with self._lock:
            if account_id not in self._accounts:
                return False
            del self._accounts[account_id]
            if self._active_account_id == account_id:
                remaining = list(self._accounts.keys())
                self._active_account_id = remaining[0] if remaining else None
        logger.info("Account %s removed", account_id)
        return True

    # -- Aggregate queries ----------------------------------------------------

    def get_total_equity(
        self, unrealized_pnl_map: Optional[Dict[str, float]] = None
    ) -> float:
        """Total equity across all accounts."""
        total = 0.0
        with self._lock:
            for account in self._accounts.values():
                bal = account.get_balance("USDT")
                upnl = (
                    unrealized_pnl_map.get(account.account_id, 0.0)
                    if unrealized_pnl_map else 0.0
                )
                total += bal.total + upnl
        return total

    def get_aggregate_summary(self) -> Dict[str, Any]:
        """Summary across all accounts."""
        with self._lock:
            summaries = {}
            total_balance = 0.0
            total_available = 0.0
            total_locked = 0.0

            for account in self._accounts.values():
                summary = account.get_summary()
                summaries[account.account_id] = summary
                bal = account.get_balance("USDT")
                total_balance += bal.total
                total_available += bal.available
                total_locked += bal.locked

            return {
                "active_account": self._active_account_id,
                "total_accounts": len(self._accounts),
                "total_balance": total_balance,
                "total_available": total_available,
                "total_locked": total_locked,
                "accounts": summaries,
            }

    # -- Margin monitoring ----------------------------------------------------

    def check_margin_health(
        self, warning_threshold: float = 80.0
    ) -> List[Tuple[str, MarginInfo]]:
        """Check margin health across all accounts. Returns accounts at risk."""
        at_risk = []
        with self._lock:
            for account in self._accounts.values():
                margin = account.get_margin_info()
                if margin.margin_ratio >= warning_threshold:
                    at_risk.append((account.account_id, margin))
                    for cb in self._on_margin_warning:
                        try:
                            cb(account, margin)
                        except Exception:
                            logger.exception("Error in margin warning callback")

        if at_risk:
            logger.warning(
                "Margin warning: %d accounts at risk", len(at_risk)
            )
        return at_risk

    # -- Copy trading helpers -------------------------------------------------

    def sync_balances_from_exchange(
        self,
        account_id: str,
        balances: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Sync balances from exchange data.
        balances format: {"USDT": {"available": 1000, "locked": 200}, ...}
        """
        account = self.get_account(account_id)
        if account is None:
            logger.warning("Account %s not found for sync", account_id)
            return

        for asset, amounts in balances.items():
            account.set_balance(
                asset=asset,
                available=amounts.get("available", 0.0),
                locked=amounts.get("locked", 0.0),
            )

        logger.info(
            "Account %s balances synced (%d assets)",
            account_id, len(balances),
        )

    # -- Export / Import ------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_account": self._active_account_id,
                "accounts": {
                    aid: account.get_summary()
                    for aid, account in self._accounts.items()
                },
            }
