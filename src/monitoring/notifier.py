"""
Multi-channel Notification System for Mefai Autotrade.
Supports Telegram bot, Discord webhooks, and SMTP email with async
message queuing, priority levels, filtering, and cooldown management.
"""

import asyncio
import logging
import smtplib
import ssl
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger("mefai.autotrade")


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class NotificationPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class NotificationMessage:
    """A notification message to be sent through one or more channels."""
    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channel: str = "all"  # "telegram", "discord", "email", or "all"
    category: str = "general"  # "trade", "alert", "daily_report", "system"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_parse_mode: str = "HTML"

    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    discord_trades_webhook_url: str = ""
    discord_alerts_webhook_url: str = ""

    # Email
    email_enabled: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Filtering
    min_pnl_notify: float = 1.0  # Minimum PnL to send trade notification
    cooldown_seconds: float = 30.0  # Minimum seconds between same-type notifications
    quiet_hours_start: int = -1  # -1 = disabled. Hour (0-23) to start quiet period
    quiet_hours_end: int = -1
    max_queue_size: int = 1000


# ---------------------------------------------------------------------------
# Telegram Notifier
# ---------------------------------------------------------------------------

class TelegramNotifier:
    """Send notifications via Telegram Bot API."""

    API_BASE = "https://api.telegram.org/bot{token}"

    def __init__(self, bot_token: str, chat_id: str, parse_mode: str = "HTML"):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.parse_mode = parse_mode
        self._api_url = self.API_BASE.format(token=bot_token)

    async def send_message(self, text: str, disable_preview: bool = True) -> bool:
        """Send a text message to the configured chat."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed - Telegram notifications disabled")
            return False

        url = f"{self._api_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": disable_preview,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        body = await resp.text()
                        logger.error(
                            f"Telegram API error: status={resp.status} body={body}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def format_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        leverage: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> str:
        direction_emoji = "LONG" if side.upper() == "BUY" else "SHORT"
        notional = entry_price * quantity

        lines = [
            f"<b>Trade Opened - {direction_emoji}</b>",
            f"",
            f"Symbol: <code>{symbol}</code>",
            f"Direction: {side.upper()}",
            f"Entry: <code>{entry_price}</code>",
            f"Quantity: <code>{quantity}</code>",
            f"Notional: <code>{notional:.2f} USDT</code>",
            f"Leverage: {leverage}x",
            f"Strategy: {strategy}",
        ]
        if stop_loss:
            lines.append(f"Stop Loss: <code>{stop_loss}</code>")
        if take_profit:
            lines.append(f"Take Profit: <code>{take_profit}</code>")

        return "\n".join(lines)

    def format_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        realized_pnl: float,
        pnl_percent: float,
        duration: str,
        strategy: str,
        close_reason: str,
    ) -> str:
        result = "WIN" if realized_pnl >= 0 else "LOSS"
        pnl_sign = "+" if realized_pnl >= 0 else ""

        lines = [
            f"<b>Trade Closed - {result}</b>",
            f"",
            f"Symbol: <code>{symbol}</code>",
            f"Direction: {side.upper()}",
            f"Entry: <code>{entry_price}</code>",
            f"Exit: <code>{exit_price}</code>",
            f"PnL: <code>{pnl_sign}{realized_pnl:.2f} USDT ({pnl_sign}{pnl_percent:.2f}%)</code>",
            f"Duration: {duration}",
            f"Strategy: {strategy}",
            f"Reason: {close_reason}",
        ]

        return "\n".join(lines)

    def format_daily_summary(
        self,
        date: str,
        total_pnl: float,
        trade_count: int,
        win_count: int,
        loss_count: int,
        win_rate: float,
        equity: float,
        drawdown: float,
        best_trade: float,
        worst_trade: float,
    ) -> str:
        pnl_sign = "+" if total_pnl >= 0 else ""

        lines = [
            f"<b>Daily Summary - {date}</b>",
            f"",
            f"PnL: <code>{pnl_sign}{total_pnl:.2f} USDT</code>",
            f"Trades: {trade_count} (W:{win_count} / L:{loss_count})",
            f"Win Rate: {win_rate:.1%}",
            f"Equity: <code>{equity:.2f} USDT</code>",
            f"Drawdown: {drawdown:.2f}%",
            f"Best Trade: <code>+{best_trade:.2f} USDT</code>",
            f"Worst Trade: <code>{worst_trade:.2f} USDT</code>",
        ]

        return "\n".join(lines)

    def format_risk_alert(
        self,
        severity: str,
        title: str,
        message: str,
    ) -> str:
        severity_label = {
            "CRITICAL": "CRITICAL",
            "WARNING": "WARNING",
            "INFO": "INFO",
        }.get(severity, severity)

        lines = [
            f"<b>Alert [{severity_label}]</b>",
            f"",
            f"<b>{title}</b>",
            f"{message}",
        ]

        return "\n".join(lines)

    def format_system_status(
        self,
        event: str,
        details: str = "",
    ) -> str:
        lines = [
            f"<b>System: {event}</b>",
        ]
        if details:
            lines.append(f"")
            lines.append(details)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord Notifier
# ---------------------------------------------------------------------------

class DiscordNotifier:
    """Send notifications via Discord webhooks with rich embeds."""

    COLOR_PROFIT = 0x00FF00  # Green
    COLOR_LOSS = 0xFF0000    # Red
    COLOR_INFO = 0x3498DB    # Blue
    COLOR_WARNING = 0xFFA500  # Orange
    COLOR_CRITICAL = 0xFF0000  # Red

    def __init__(
        self,
        webhook_url: str,
        trades_webhook_url: Optional[str] = None,
        alerts_webhook_url: Optional[str] = None,
    ):
        self.webhook_url = webhook_url
        self.trades_url = trades_webhook_url or webhook_url
        self.alerts_url = alerts_webhook_url or webhook_url

    async def send_embed(
        self,
        webhook_url: str,
        title: str,
        description: str,
        color: int,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
    ) -> bool:
        """Send a Discord embed message."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed - Discord notifications disabled")
            return False

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if fields:
            embed["fields"] = fields
        if footer:
            embed["footer"] = {"text": footer}

        payload = {
            "username": "Mefai Autotrade",
            "embeds": [embed],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 204):
                        return True
                    else:
                        body = await resp.text()
                        logger.error(
                            f"Discord webhook error: status={resp.status} body={body}"
                        )
                        return False
        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    async def send_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        leverage: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        notional = entry_price * quantity
        direction = "LONG" if side.upper() == "BUY" else "SHORT"

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Direction", "value": direction, "inline": True},
            {"name": "Leverage", "value": f"{leverage}x", "inline": True},
            {"name": "Entry Price", "value": str(entry_price), "inline": True},
            {"name": "Quantity", "value": str(quantity), "inline": True},
            {"name": "Notional", "value": f"{notional:.2f} USDT", "inline": True},
            {"name": "Strategy", "value": strategy, "inline": True},
        ]
        if stop_loss:
            fields.append({"name": "Stop Loss", "value": str(stop_loss), "inline": True})
        if take_profit:
            fields.append({"name": "Take Profit", "value": str(take_profit), "inline": True})

        return await self.send_embed(
            self.trades_url,
            f"Trade Opened - {direction} {symbol}",
            f"New {direction} position on {symbol}",
            self.COLOR_INFO,
            fields=fields,
            footer=f"Strategy: {strategy}",
        )

    async def send_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        realized_pnl: float,
        pnl_percent: float,
        duration: str,
        strategy: str,
        close_reason: str,
    ) -> bool:
        color = self.COLOR_PROFIT if realized_pnl >= 0 else self.COLOR_LOSS
        result = "WIN" if realized_pnl >= 0 else "LOSS"
        pnl_sign = "+" if realized_pnl >= 0 else ""

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Result", "value": result, "inline": True},
            {"name": "PnL", "value": f"{pnl_sign}{realized_pnl:.2f} USDT", "inline": True},
            {"name": "PnL %", "value": f"{pnl_sign}{pnl_percent:.2f}%", "inline": True},
            {"name": "Entry", "value": str(entry_price), "inline": True},
            {"name": "Exit", "value": str(exit_price), "inline": True},
            {"name": "Duration", "value": duration, "inline": True},
            {"name": "Close Reason", "value": close_reason, "inline": True},
        ]

        return await self.send_embed(
            self.trades_url,
            f"Trade Closed - {result} {symbol}",
            f"{pnl_sign}{realized_pnl:.2f} USDT ({pnl_sign}{pnl_percent:.2f}%)",
            color,
            fields=fields,
            footer=f"Strategy: {strategy}",
        )

    async def send_alert(
        self,
        severity: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        color_map = {
            "CRITICAL": self.COLOR_CRITICAL,
            "WARNING": self.COLOR_WARNING,
            "INFO": self.COLOR_INFO,
        }
        color = color_map.get(severity, self.COLOR_INFO)

        fields = []
        if details:
            for key, value in details.items():
                fields.append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value),
                    "inline": True,
                })

        return await self.send_embed(
            self.alerts_url,
            f"[{severity}] {title}",
            message,
            color,
            fields=fields or None,
        )

    async def send_daily_summary(
        self,
        date: str,
        total_pnl: float,
        trade_count: int,
        win_rate: float,
        equity: float,
        drawdown: float,
    ) -> bool:
        color = self.COLOR_PROFIT if total_pnl >= 0 else self.COLOR_LOSS
        pnl_sign = "+" if total_pnl >= 0 else ""

        fields = [
            {"name": "PnL", "value": f"{pnl_sign}{total_pnl:.2f} USDT", "inline": True},
            {"name": "Trades", "value": str(trade_count), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.1%}", "inline": True},
            {"name": "Equity", "value": f"{equity:.2f} USDT", "inline": True},
            {"name": "Drawdown", "value": f"{drawdown:.2f}%", "inline": True},
        ]

        return await self.send_embed(
            self.webhook_url,
            f"Daily Summary - {date}",
            f"Daily PnL: {pnl_sign}{total_pnl:.2f} USDT",
            color,
            fields=fields,
        )


# ---------------------------------------------------------------------------
# Email Notifier
# ---------------------------------------------------------------------------

class EmailNotifier:
    """Send notifications via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send_email(
        self,
        subject: str,
        html_body: str,
        plain_body: Optional[str] = None,
    ) -> bool:
        """Send an email synchronously."""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            if plain_body:
                msg.attach(MIMEText(plain_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            if self.use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls(context=context)
                    server.login(self.username, self.password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    async def send_email_async(
        self,
        subject: str,
        html_body: str,
        plain_body: Optional[str] = None,
    ) -> bool:
        """Send email in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.send_email, subject, html_body, plain_body
        )

    def format_daily_report(
        self,
        date: str,
        total_pnl: float,
        trade_count: int,
        win_count: int,
        loss_count: int,
        win_rate: float,
        equity: float,
        drawdown: float,
        trades: List[Dict[str, Any]],
    ) -> str:
        pnl_sign = "+" if total_pnl >= 0 else ""
        pnl_color = "#00cc00" if total_pnl >= 0 else "#cc0000"

        trade_rows = ""
        for t in trades[:20]:  # Max 20 trades in email
            t_pnl = t.get("net_pnl", 0)
            t_color = "#00cc00" if t_pnl >= 0 else "#cc0000"
            t_sign = "+" if t_pnl >= 0 else ""
            trade_rows += f"""
            <tr>
                <td style="padding:4px 8px">{t.get('symbol', '')}</td>
                <td style="padding:4px 8px">{t.get('side', '')}</td>
                <td style="padding:4px 8px;color:{t_color}">{t_sign}{t_pnl:.2f}</td>
                <td style="padding:4px 8px">{t.get('strategy', '')}</td>
                <td style="padding:4px 8px">{t.get('close_reason', '')}</td>
            </tr>"""

        return f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#1a1a1a;color:#e0e0e0;padding:20px">
            <h2 style="color:#d4a843">Mefai Autotrade - Daily Report</h2>
            <p style="color:#aaa">{date}</p>

            <table style="margin:20px 0;border-collapse:collapse">
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Daily PnL</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333;color:{pnl_color};font-weight:bold">
                        {pnl_sign}{total_pnl:.2f} USDT
                    </td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Trades</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">
                        {trade_count} (W:{win_count} / L:{loss_count})
                    </td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Win Rate</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{win_rate:.1%}</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Equity</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{equity:.2f} USDT</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Drawdown</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{drawdown:.2f}%</td>
                </tr>
            </table>

            <h3 style="color:#d4a843;margin-top:30px">Trades</h3>
            <table style="border-collapse:collapse;width:100%">
                <thead>
                    <tr style="border-bottom:2px solid #333">
                        <th style="padding:4px 8px;text-align:left">Symbol</th>
                        <th style="padding:4px 8px;text-align:left">Side</th>
                        <th style="padding:4px 8px;text-align:left">PnL</th>
                        <th style="padding:4px 8px;text-align:left">Strategy</th>
                        <th style="padding:4px 8px;text-align:left">Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {trade_rows}
                </tbody>
            </table>

            <p style="color:#666;margin-top:30px;font-size:12px">
                Mefai Autotrade - Automated Trading System
            </p>
        </body>
        </html>
        """

    def format_weekly_report(
        self,
        week_start: str,
        week_end: str,
        total_pnl: float,
        trade_count: int,
        win_rate: float,
        equity: float,
        max_drawdown: float,
        sharpe: float,
        profit_factor: float,
        daily_returns: List[Dict[str, Any]],
    ) -> str:
        pnl_sign = "+" if total_pnl >= 0 else ""
        pnl_color = "#00cc00" if total_pnl >= 0 else "#cc0000"

        day_rows = ""
        for d in daily_returns:
            d_pnl = d.get("pnl", 0)
            d_color = "#00cc00" if d_pnl >= 0 else "#cc0000"
            d_sign = "+" if d_pnl >= 0 else ""
            day_rows += f"""
            <tr>
                <td style="padding:4px 8px">{d.get('date', '')}</td>
                <td style="padding:4px 8px;color:{d_color}">{d_sign}{d_pnl:.2f}</td>
                <td style="padding:4px 8px">{d.get('trades', 0)}</td>
            </tr>"""

        return f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#1a1a1a;color:#e0e0e0;padding:20px">
            <h2 style="color:#d4a843">Mefai Autotrade - Weekly Report</h2>
            <p style="color:#aaa">{week_start} to {week_end}</p>

            <table style="margin:20px 0;border-collapse:collapse">
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Weekly PnL</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333;color:{pnl_color};font-weight:bold">
                        {pnl_sign}{total_pnl:.2f} USDT
                    </td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Trades</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{trade_count}</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Win Rate</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{win_rate:.1%}</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Equity</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{equity:.2f} USDT</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Max Drawdown</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{max_drawdown:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Sharpe Ratio</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{sharpe:.2f}</td>
                </tr>
                <tr>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">Profit Factor</td>
                    <td style="padding:8px 16px;border-bottom:1px solid #333">{profit_factor:.2f}</td>
                </tr>
            </table>

            <h3 style="color:#d4a843;margin-top:30px">Daily Breakdown</h3>
            <table style="border-collapse:collapse;width:100%">
                <thead>
                    <tr style="border-bottom:2px solid #333">
                        <th style="padding:4px 8px;text-align:left">Date</th>
                        <th style="padding:4px 8px;text-align:left">PnL</th>
                        <th style="padding:4px 8px;text-align:left">Trades</th>
                    </tr>
                </thead>
                <tbody>{day_rows}</tbody>
            </table>

            <p style="color:#666;margin-top:30px;font-size:12px">
                Mefai Autotrade - Automated Trading System
            </p>
        </body>
        </html>
        """

    def format_critical_alert(
        self,
        severity: str,
        title: str,
        message: str,
        details: Dict[str, Any],
    ) -> str:
        detail_rows = ""
        for key, value in details.items():
            detail_rows += f"""
            <tr>
                <td style="padding:4px 8px;border-bottom:1px solid #333">{key.replace('_', ' ').title()}</td>
                <td style="padding:4px 8px;border-bottom:1px solid #333">{value}</td>
            </tr>"""

        border_color = "#cc0000" if severity == "CRITICAL" else "#FFA500"

        return f"""
        <html>
        <body style="font-family:Arial,sans-serif;background:#1a1a1a;color:#e0e0e0;padding:20px">
            <div style="border-left:4px solid {border_color};padding-left:16px">
                <h2 style="color:{border_color}">[{severity}] {title}</h2>
                <p>{message}</p>
            </div>

            <table style="margin:20px 0;border-collapse:collapse">
                {detail_rows}
            </table>

            <p style="color:#666;margin-top:30px;font-size:12px">
                Mefai Autotrade - Automated Alert
            </p>
        </body>
        </html>
        """


# ---------------------------------------------------------------------------
# Main Notifier (Unified Queue)
# ---------------------------------------------------------------------------

class Notifier:
    """Unified notification system with async queue and filtering.

    Routes messages to the appropriate channel(s) based on configuration,
    handles rate limiting, cooldowns, and quiet hours.

    Usage:
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="123:ABC",
            telegram_chat_id="-100123",
        )
        notifier = Notifier(config)
        await notifier.start()
        await notifier.notify_trade_opened(symbol="BTCUSDT", ...)
        await notifier.stop()
    """

    def __init__(self, config: NotificationConfig):
        self.config = config
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Cooldown tracking: category -> last_send_time
        self._cooldowns: Dict[str, float] = {}

        # Message counters
        self._sent_count = 0
        self._failed_count = 0
        self._filtered_count = 0

        # Initialize channel notifiers
        self._telegram: Optional[TelegramNotifier] = None
        self._discord: Optional[DiscordNotifier] = None
        self._email: Optional[EmailNotifier] = None

        if config.telegram_enabled and config.telegram_bot_token:
            self._telegram = TelegramNotifier(
                config.telegram_bot_token,
                config.telegram_chat_id,
                config.telegram_parse_mode,
            )

        if config.discord_enabled and config.discord_webhook_url:
            self._discord = DiscordNotifier(
                config.discord_webhook_url,
                config.discord_trades_webhook_url,
                config.discord_alerts_webhook_url,
            )

        if config.email_enabled and config.smtp_username:
            self._email = EmailNotifier(
                config.smtp_host,
                config.smtp_port,
                config.smtp_username,
                config.smtp_password,
                config.email_from,
                config.email_to,
                config.smtp_use_tls,
            )

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def start(self) -> None:
        """Start the notification queue processor."""
        if self._running:
            return
        self._running = True
        self._queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._task = asyncio.create_task(self._process_queue())
        logger.info("Notification system started")

    async def stop(self) -> None:
        """Stop the notification queue processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Process remaining messages
        while not self._queue.empty():
            try:
                msg = self._queue.get_nowait()
                await self._dispatch(msg)
            except asyncio.QueueEmpty:
                break
        logger.info(
            f"Notification system stopped. "
            f"Sent: {self._sent_count}, Failed: {self._failed_count}, "
            f"Filtered: {self._filtered_count}"
        )

    async def _process_queue(self) -> None:
        """Process messages from the queue."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(msg)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Notification queue error: {e}", exc_info=True)

    # -------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------

    def _should_send(self, msg: NotificationMessage) -> bool:
        """Check if a message should be sent based on filters."""
        # Cooldown check
        now = time.time()
        cooldown_key = f"{msg.category}:{msg.channel}"
        last_sent = self._cooldowns.get(cooldown_key, 0)
        if now - last_sent < self.config.cooldown_seconds:
            # Critical priority bypasses cooldown
            if msg.priority != NotificationPriority.CRITICAL:
                self._filtered_count += 1
                return False

        # Quiet hours check (critical always goes through)
        if (
            msg.priority != NotificationPriority.CRITICAL
            and self.config.quiet_hours_start >= 0
        ):
            current_hour = datetime.now(timezone.utc).hour
            start = self.config.quiet_hours_start
            end = self.config.quiet_hours_end
            if start < end:
                if start <= current_hour < end:
                    self._filtered_count += 1
                    return False
            else:  # Wraps midnight
                if current_hour >= start or current_hour < end:
                    self._filtered_count += 1
                    return False

        # PnL threshold for trade notifications
        if msg.category == "trade":
            pnl = abs(msg.data.get("realized_pnl", 0))
            if pnl < self.config.min_pnl_notify and pnl > 0:
                self._filtered_count += 1
                return False

        return True

    # -------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------

    async def _dispatch(self, msg: NotificationMessage) -> None:
        """Route a message to the appropriate channel(s)."""
        if not self._should_send(msg):
            return

        now = time.time()
        cooldown_key = f"{msg.category}:{msg.channel}"

        channels_to_send = []
        if msg.channel == "all":
            if self._telegram:
                channels_to_send.append("telegram")
            if self._discord:
                channels_to_send.append("discord")
            if msg.priority == NotificationPriority.CRITICAL and self._email:
                channels_to_send.append("email")
        else:
            channels_to_send.append(msg.channel)

        success = False
        for channel in channels_to_send:
            try:
                if channel == "telegram" and self._telegram:
                    result = await self._telegram.send_message(msg.body)
                    success = success or result
                elif channel == "discord" and self._discord:
                    result = await self._discord.send_embed(
                        self._discord.webhook_url,
                        msg.title,
                        msg.body,
                        self._get_discord_color(msg),
                    )
                    success = success or result
                elif channel == "email" and self._email:
                    result = await self._email.send_email_async(
                        f"Mefai Autotrade - {msg.title}",
                        msg.body,
                    )
                    success = success or result
            except Exception as e:
                logger.error(f"Notification dispatch error ({channel}): {e}")

        if success:
            self._sent_count += 1
            self._cooldowns[cooldown_key] = now
        else:
            self._failed_count += 1

    @staticmethod
    def _get_discord_color(msg: NotificationMessage) -> int:
        if msg.category == "trade":
            pnl = msg.data.get("realized_pnl", 0)
            return DiscordNotifier.COLOR_PROFIT if pnl >= 0 else DiscordNotifier.COLOR_LOSS
        if msg.priority == NotificationPriority.CRITICAL:
            return DiscordNotifier.COLOR_CRITICAL
        if msg.priority == NotificationPriority.HIGH:
            return DiscordNotifier.COLOR_WARNING
        return DiscordNotifier.COLOR_INFO

    # -------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------

    async def _enqueue(self, msg: NotificationMessage) -> None:
        """Add a message to the queue (non-blocking)."""
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning("Notification queue full - dropping message")
            self._filtered_count += 1

    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        leverage: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Notify about a new trade."""
        body = ""
        if self._telegram:
            body = self._telegram.format_trade_opened(
                symbol, side, entry_price, quantity, strategy,
                leverage, stop_loss, take_profit,
            )

        await self._enqueue(NotificationMessage(
            title=f"Trade Opened: {side} {symbol}",
            body=body or f"Opened {side} {symbol} @ {entry_price}",
            priority=NotificationPriority.NORMAL,
            category="trade",
            data={
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "quantity": quantity,
                "strategy": strategy,
                "leverage": leverage,
            },
        ))

        # Also send Discord embed if available
        if self._discord:
            await self._discord.send_trade_opened(
                symbol, side, entry_price, quantity, strategy,
                leverage, stop_loss, take_profit,
            )

    async def notify_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        realized_pnl: float,
        pnl_percent: float,
        duration: str,
        strategy: str,
        close_reason: str,
    ) -> None:
        """Notify about a closed trade."""
        # Filter small trades
        if abs(realized_pnl) < self.config.min_pnl_notify:
            return

        body = ""
        if self._telegram:
            body = self._telegram.format_trade_closed(
                symbol, side, entry_price, exit_price, quantity,
                realized_pnl, pnl_percent, duration, strategy, close_reason,
            )

        await self._enqueue(NotificationMessage(
            title=f"Trade Closed: {symbol}",
            body=body or f"Closed {symbol} PnL={realized_pnl:+.2f}",
            priority=NotificationPriority.NORMAL,
            category="trade",
            data={
                "symbol": symbol,
                "realized_pnl": realized_pnl,
                "pnl_percent": pnl_percent,
            },
        ))

        if self._discord:
            await self._discord.send_trade_closed(
                symbol, side, entry_price, exit_price,
                realized_pnl, pnl_percent, duration, strategy, close_reason,
            )

    async def notify_alert(
        self,
        severity: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a risk/system alert."""
        priority = {
            "CRITICAL": NotificationPriority.CRITICAL,
            "WARNING": NotificationPriority.HIGH,
            "INFO": NotificationPriority.NORMAL,
        }.get(severity, NotificationPriority.NORMAL)

        body = ""
        if self._telegram:
            body = self._telegram.format_risk_alert(severity, title, message)

        await self._enqueue(NotificationMessage(
            title=f"[{severity}] {title}",
            body=body or f"[{severity}] {title}: {message}",
            priority=priority,
            category="alert",
            data=details or {},
        ))

        if self._discord:
            await self._discord.send_alert(severity, title, message, details)

        # Email for critical alerts
        if severity == "CRITICAL" and self._email:
            html = self._email.format_critical_alert(
                severity, title, message, details or {}
            )
            await self._email.send_email_async(
                f"CRITICAL ALERT: {title}", html
            )

    async def notify_daily_summary(
        self,
        date: str,
        total_pnl: float,
        trade_count: int,
        win_count: int,
        loss_count: int,
        win_rate: float,
        equity: float,
        drawdown: float,
        best_trade: float = 0.0,
        worst_trade: float = 0.0,
        trades: Optional[List[Dict]] = None,
    ) -> None:
        """Send daily performance summary."""
        if self._telegram:
            body = self._telegram.format_daily_summary(
                date, total_pnl, trade_count, win_count, loss_count,
                win_rate, equity, drawdown, best_trade, worst_trade,
            )
            await self._enqueue(NotificationMessage(
                title=f"Daily Summary - {date}",
                body=body,
                priority=NotificationPriority.NORMAL,
                category="daily_report",
            ))

        if self._discord:
            await self._discord.send_daily_summary(
                date, total_pnl, trade_count, win_rate, equity, drawdown,
            )

        if self._email:
            html = self._email.format_daily_report(
                date, total_pnl, trade_count, win_count, loss_count,
                win_rate, equity, drawdown, trades or [],
            )
            await self._email.send_email_async(
                f"Mefai Daily Report - {date}", html
            )

    async def notify_system_event(
        self,
        event: str,
        details: str = "",
    ) -> None:
        """Send system event notification (startup, shutdown, error)."""
        body = ""
        if self._telegram:
            body = self._telegram.format_system_status(event, details)

        await self._enqueue(NotificationMessage(
            title=f"System: {event}",
            body=body or f"System: {event} - {details}",
            priority=NotificationPriority.HIGH,
            category="system",
        ))

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get notification system statistics."""
        return {
            "sent": self._sent_count,
            "failed": self._failed_count,
            "filtered": self._filtered_count,
            "queue_size": self._queue.qsize(),
            "channels": {
                "telegram": self._telegram is not None,
                "discord": self._discord is not None,
                "email": self._email is not None,
            },
        }
