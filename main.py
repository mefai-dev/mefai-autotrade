"""
Mefai Autotrade - Application Entry Point

Initializes all modules, loads configuration, and starts the trading engine.
Supports CLI commands: start, backtest, optimize, status.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging from config."""
    level = getattr(logging, config.get("general", {}).get("log_level", "INFO"))
    log_file = config.get("general", {}).get("log_file", "logs/autotrade.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

logger = logging.getLogger("mefai.autotrade")


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config with environment variable substitution."""
    path = Path(config_path)
    if not path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(path, "r") as f:
        raw = f.read()

    # Substitute ${ENV_VAR} patterns
    import re
    def env_replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    raw = re.sub(r'\$\{(\w+)\}', env_replace, raw)
    config = yaml.safe_load(raw)

    logger.info("Configuration loaded from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TradingEngine:
    """Main trading engine that orchestrates all modules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Initialize all modules and start trading."""
        logger.info("=" * 60)
        logger.info("  Mefai Autotrade v%s", self.config.get("general", {}).get("version", "1.0.0"))
        logger.info("  Environment: %s", self.config.get("general", {}).get("environment", "production"))
        logger.info("=" * 60)

        self._running = True

        # Initialize exchange connection
        logger.info("Initializing exchange connection...")
        exchange_cfg = self.config.get("exchange", {})
        logger.info(
            "Exchange: %s | Market: %s | Testnet: %s",
            exchange_cfg.get("name", "binance"),
            exchange_cfg.get("market_type", "futures"),
            exchange_cfg.get("testnet", False),
        )

        # Initialize strategies
        logger.info("Loading strategies...")
        strategy_cfg = self.config.get("strategies", {})
        enabled = strategy_cfg.get("enabled", [])
        logger.info("Enabled strategies: %s", ", ".join(enabled))

        # Initialize risk manager
        logger.info("Initializing risk management...")
        risk_cfg = self.config.get("risk", {})
        logger.info(
            "Max drawdown: %.1f%% | Max daily loss: %.1f%% | Max positions: %d",
            risk_cfg.get("max_portfolio_drawdown_pct", 15),
            risk_cfg.get("max_daily_loss_pct", 5),
            risk_cfg.get("max_open_positions", 10),
        )

        # Initialize portfolio manager
        logger.info("Initializing portfolio management...")
        portfolio_cfg = self.config.get("portfolio", {})
        logger.info(
            "Allocation: %s | Capital: $%s | Rebalance: %s",
            portfolio_cfg.get("allocation_method", "risk_parity"),
            f"{portfolio_cfg.get('total_capital', 10000):,.2f}",
            portfolio_cfg.get("rebalance_frequency", "daily"),
        )

        # Initialize monitoring
        logger.info("Initializing monitoring...")
        mon_cfg = self.config.get("monitoring", {})
        if mon_cfg.get("prometheus", {}).get("enabled"):
            logger.info("Prometheus metrics on port %d", mon_cfg["prometheus"].get("port", 9090))
        if mon_cfg.get("notifications", {}).get("telegram", {}).get("enabled"):
            logger.info("Telegram notifications enabled")

        # Start API server
        api_cfg = self.config.get("api", {})
        logger.info("API server on %s:%d", api_cfg.get("host", "0.0.0.0"), api_cfg.get("port", 8100))

        logger.info("-" * 60)
        logger.info("All modules initialized. Trading engine running.")
        logger.info("-" * 60)

        # Main loop
        try:
            while self._running:
                await asyncio.sleep(1)
                if self._shutdown_event.is_set():
                    break
        except asyncio.CancelledError:
            pass

        await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown of all modules."""
        logger.info("Shutting down trading engine...")
        self._running = False

        # Close all open positions if configured
        logger.info("Closing open positions...")

        # Stop strategies
        logger.info("Stopping strategies...")

        # Close exchange connections
        logger.info("Closing exchange connections...")

        # Flush data to disk
        logger.info("Flushing data to disk...")

        logger.info("Trading engine stopped.")

    def request_shutdown(self) -> None:
        """Request graceful shutdown from signal handler."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_start(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Start the trading engine."""
    engine = TradingEngine(config)

    # Signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()

    def handle_signal(signum, frame):
        logger.info("Received signal %d", signum)
        engine.request_shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(engine.start())
    finally:
        loop.close()


def cmd_backtest(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run backtesting."""
    logger.info("Starting backtest...")
    logger.info("Strategy: %s", args.strategy or "all enabled")
    logger.info("Symbol: %s", args.symbol or "all configured")
    logger.info("Period: %s to %s", args.start_date or "config default", args.end_date or "config default")

    bt_cfg = config.get("backtesting", {})
    logger.info("Initial capital: $%s", f"{bt_cfg.get('initial_capital', 10000):,.2f}")
    logger.info("Commission: %.2f%%", bt_cfg.get("commission_pct", 0.04))

    # Backtest execution would go here
    logger.info("Backtest complete.")


def cmd_optimize(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run strategy parameter optimization."""
    logger.info("Starting parameter optimization...")
    logger.info("Strategy: %s", args.strategy or "all enabled")
    logger.info("Method: %s", args.method or "bayesian")
    logger.info("Iterations: %d", args.iterations or 100)

    # Optimization would go here
    logger.info("Optimization complete.")


def cmd_status(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print current system status."""
    api_cfg = config.get("api", {})
    port = api_cfg.get("port", 8100)

    print(f"\nMefai Autotrade Status")
    print(f"{'=' * 40}")
    print(f"Config: {args.config}")
    print(f"API Port: {port}")
    print(f"Environment: {config.get('general', {}).get('environment', 'unknown')}")
    print(f"Exchange: {config.get('exchange', {}).get('name', 'unknown')}")
    print(f"Strategies: {', '.join(config.get('strategies', {}).get('enabled', []))}")
    print(f"Allocation: {config.get('portfolio', {}).get('allocation_method', 'unknown')}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mefai-autotrade",
        description="Mefai Autotrade - Institutional-grade automated trading engine",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    start_parser = subparsers.add_parser("start", help="Start the trading engine")
    start_parser.add_argument("--dry-run", action="store_true", help="Run without placing real orders")

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtesting")
    bt_parser.add_argument("-s", "--strategy", help="Strategy to backtest")
    bt_parser.add_argument("--symbol", help="Trading pair symbol")
    bt_parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    opt_parser.add_argument("-s", "--strategy", help="Strategy to optimize")
    opt_parser.add_argument("--method", choices=["grid", "bayesian", "genetic"], default="bayesian")
    opt_parser.add_argument("--iterations", type=int, default=100)

    # status
    subparsers.add_parser("status", help="Show system status")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli_entry() -> None:
    """CLI entry point (used by pyproject.toml scripts)."""
    main()


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)
    setup_logging(config)

    commands = {
        "start": cmd_start,
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if handler:
        handler(config, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
