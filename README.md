# Mefai Autotrade

[![CI](https://github.com/mefai-dev/mefai-autotrade/actions/workflows/ci.yml/badge.svg)](https://github.com/mefai-dev/mefai-autotrade/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)


**Institutional grade automated trading engine for cryptocurrency markets.**

Built for professional traders and funds that need reliable and battle tested infrastructure for algorithmic trading across multiple exchanges. Mefai Autotrade combines 20+ trading strategies, advanced portfolio management, intelligent risk controls, and production grade execution into a single cohesive system.

```
    MARKET DATA          STRATEGIES           EXECUTION          EXCHANGE
   ___________        _____________        ____________        __________
  |           |      |             |      |            |      |          |
  | Binance   |----->| Momentum    |----->| TWAP/VWAP  |----->| Binance  |
  | WebSocket |      | Trend       |      | Iceberg    |      | Futures  |
  | REST API  |      | Mean Rev    |      | Smart Route|      |__________|
  | Tick Data |      | Breakout    |      |____________|
  |___________|      | Grid        |           |              ___________
       |             | Scalping    |           |             |           |
       |             | Smart Money |           |------------>| Bybit     |
       |             | ML Engine   |           |             | OKX       |
       |             | Arbitrage   |           |             |___________|
       |             | + 12 more   |           |
       |             |_____________|           |
       |                  |                    |
       |                  v                    |
       |           _______________             |
       |          |               |            |
       |--------->| RISK ENGINE   |<-----------|
       |          |               |
       |          | Position Size |        ________________
       |          | Stop Loss     |       |                |
       |          | Drawdown      |------>| PORTFOLIO MGR  |
       |          | Circuit Break |       |                |
       |          | Correlation   |       | Allocation     |
       |          |_______________|       | Rebalancing    |
       |                                  | Attribution    |
       |          _______________         | Copy Trading   |
       |         |               |        |________________|
       |-------->| BACKTESTER    |
       |         |               |        ________________
       |         | Walk Forward  |       |                |
       |         | Monte Carlo   |------>| MONITORING     |
       |         | Optimization  |       |                |
       |         |_______________|       | Prometheus     |
                                         | Telegram       |
                                         | Discord        |
                                         | WebSocket      |
                                         |________________|
```

---

## Table of Contents

- [Features](#features)
- [Trading Strategies](#trading-strategies)
- [Risk Management](#risk-management)
- [Portfolio Management](#portfolio-management)
- [Execution Algorithms](#execution-algorithms)
- [Exchange Support](#exchange-support)
- [Backtesting](#backtesting)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [ML Training](#ml-training)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

### Core Engine
- **20+ trading strategies** from scalping to portfolio-level allocation
- **Multi-exchange support** via CCXT with automatic failover
- **Real time execution** with sub-second signal-to-order latency
- **Strategy registry** with factory pattern for dynamic strategy loading
- **Composite strategies** with weighted voting and conflict resolution

### Risk Management
- Position sizing (risk-based, Kelly criterion, ATR-based)
- Multi-level stop loss (fixed, trailing, ATR-based, time-based)
- Portfolio-level drawdown protection with circuit breakers
- Correlation filter to prevent concentrated exposure
- Daily and weekly loss limits with automatic trading halt

### Portfolio Management
- 8 allocation methods (equal, risk parity, Markowitz, Kelly, and more)
- Automated rebalancing (periodic, drift-based, tax-aware, cost-aware)
- Performance attribution (Brinson-Fachler decomposition)
- Factor analysis and regime detection
- Copy trading hub with master/follower architecture

### Execution
- 4 execution algorithms (TWAP, VWAP, Iceberg, Smart Routing)
- Order lifecycle management with partial fill tracking
- Slippage estimation and transaction cost analysis
- Automatic retry with exponential backoff

### Backtesting
- Event-driven backtester with realistic fill simulation
- Walk-forward analysis with rolling optimization
- Monte Carlo simulation for robustness testing
- Parameter optimization (grid search, Bayesian, genetic)

### Machine Learning
- XGBoost, Random Forest, and Neural Network models
- Automatic feature engineering from technical indicators
- Scheduled model retraining with validation
- Ensemble voting across multiple model types
- Hidden Markov Model for regime detection

### Monitoring
- Prometheus metrics export
- Telegram and Discord notifications
- Real time WebSocket dashboard feed
- Health check endpoint for orchestration systems

---

## Trading Strategies

| # | Strategy | Description | Timeframes | Style |
|---|----------|-------------|------------|-------|
| 1 | **Momentum** | Multi indicator momentum with RSI, MACD, ADX confluence | 5m, 15m | Swing |
| 2 | **Trend Following** | Supertrend + EMA ribbon + Ichimoku cloud | 15m, 1h | Trend |
| 3 | **Mean Reversion** | Bollinger Band + z-score statistical mean reversion | 5m, 15m | Mean Rev |
| 4 | **Breakout** | Range breakout with volume confirmation and false breakout filter | 15m, 1h | Breakout |
| 5 | **Grid Trading** | Arithmetic/geometric grid for ranging markets | 1m, 5m | Grid |
| 6 | **Scalping** | High frequency micro-trend scalping with spread filter | 1m | Scalp |
| 7 | **Smart Money** | ICT concepts - order blocks, FVG, BOS, liquidity sweeps | 15m, 4h | SMC |
| 8 | **Machine Learning** | XGBoost/RF ensemble consuming 100+ engineered features | 5m | ML |
| 9 | **DCA** | Intelligent dollar cost averaging with dip detection | 1h, 4h | DCA |
| 10 | **Arbitrage** | Cross exchange spread and funding rate arbitrage | Real time | Arb |
| 11 | **Market Making** | Inventory aware quoting with dynamic spread adjustment | 1m | MM |
| 12 | **Pairs Trading** | Cointegration based statistical pairs with z score entry/exit | 5m, 15m | Stat Arb |
| 13 | **Volume Profile** | POC, value area, and VWAP-based entry/exit | 5m, 15m | Volume |
| 14 | **Elliott Wave** | Automated wave counting with Fibonacci projections | 1h, 4h | Wave |
| 15 | **Order Flow** | CVD divergence, delta analysis, absorption detection | 1m, 5m | Flow |
| 16 | **Composite** | Multi-strategy voting with configurable thresholds | Any | Meta |

### Built-in Technical Indicators

Every strategy has access to 16 built-in indicators via the `FeatureEngine`:

- SMA, EMA (any period)
- RSI (Wilder's smoothing)
- MACD (line, signal, histogram)
- ATR (Average True Range)
- Bollinger Bands (upper, middle, lower)
- ADX (Average Directional Index)
- Supertrend (with direction)
- VWAP (cumulative)
- Stochastic (%K, %D)
- Donchian Channels
- Keltner Channels
- Rate of Change (ROC)
- Parabolic SAR

All indicators are computed using pure NumPy for maximum performance - no external TA library dependency.

---

## Risk Management

### Position-Level Controls

| Control | Description | Default |
|---------|-------------|---------|
| Risk-based sizing | Risk X% of capital per trade based on stop distance | 1% |
| Kelly criterion sizing | Optimal fraction with configurable Kelly multiplier | Half-Kelly |
| Maximum position size | Hard cap per position as % of capital | 10% |
| Stop loss | Fixed percentage, ATR-based, or trailing | 2% |
| Take profit | Single target or multiple partial take-profit levels | 4% |
| Trailing stop | ATR-based trailing with configurable activation | 2x ATR |
| Time-based exit | Close position after maximum holding period | Configurable |

### Portfolio-Level Controls

| Control | Description | Default |
|---------|-------------|---------|
| Max drawdown | Halt all trading when portfolio drawdown exceeds threshold | 15% |
| Daily loss limit | Stop trading for the day after X% loss | 5% |
| Weekly loss limit | Reduce position sizes after X% weekly loss | 10% |
| Max open positions | Hard limit on concurrent positions | 10 |
| Correlation filter | Block new positions correlated > threshold with existing | 0.85 |
| Circuit breaker | Pause after N consecutive losses | 5 losses |
| Dynamic leverage | Reduce leverage automatically in high-volatility regimes | Enabled |

---

## Portfolio Management

### Allocation Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Equal** | Equal capital to each strategy | Simplicity, no historical data needed |
| **Risk Parity** | Equal risk contribution per strategy | Balanced risk exposure |
| **Mean-Variance** | Markowitz optimization maximizing Sharpe ratio | Maximum risk-adjusted return |
| **Kelly Criterion** | Theoretically optimal fraction (half-Kelly for safety) | High-conviction strategies |
| **Max Diversification** | Maximize diversification ratio | Reducing concentration risk |
| **Min Variance** | Minimize total portfolio variance | Capital preservation |
| **Risk Budgeting** | Assign specific risk budgets per strategy | Custom risk targets |
| **Dynamic** | Adaptive blend of momentum, mean-reversion, vol-scaling | Changing market conditions |

### Rebalancing Modes

| Mode | Description |
|------|-------------|
| **Immediate** | Execute all rebalancing trades at once |
| **Gradual** | Spread rebalancing over N steps to reduce market impact |
| **Tax-Aware** | Avoid realizing short-term gains, prefer tax-loss harvesting |
| **Cost-Aware** | Skip rebalancing trades where cost exceeds expected benefit |

### Performance Analytics

- **Return calculation** - Time-weighted (TWRR) and money-weighted (MWRR/IRR)
- **Attribution analysis** - Brinson-Fachler decomposition (allocation, selection, interaction)
- **Factor analysis** - Multi-factor regression with beta, alpha, R-squared
- **Regime analysis** - Performance breakdown across bull, bear, sideways, crash, recovery
- **Risk decomposition** - Euler decomposition of per-strategy risk contribution
- **Diversification benefit** - Quantified vol reduction, effective number of strategies
- **Benchmark tracking** - Correlation, beta, alpha, tracking error, information ratio

### Copy Trading

- Master/follower architecture with independent risk controls
- Configurable size multiplier, fixed size, or proportional sizing
- Delay configuration (copy after N seconds)
- Filter by symbol, strategy, direction, and minimum confidence
- Per-master performance tracking with leaderboard
- Follower-side daily trade limits, loss limits, and max exposure

---

## Execution Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **TWAP** | Time-Weighted Average Price - split order into equal slices over time | Large orders, reduce timing risk |
| **VWAP** | Volume-Weighted Average Price - participate proportionally to market volume | Match market VWAP benchmark |
| **Iceberg** | Show only a fraction of total order, replenish on fill | Hide large order intent |
| **Smart Routing** | Check orderbook depth, prefer maker orders, route to best venue | Minimize execution cost |
| **Market** | Immediate market order execution | Time-sensitive signals |

All algorithms include:
- Automatic retry with exponential backoff (3 attempts default)
- Order timeout detection (30 seconds default)
- Slippage tolerance check (reject fill if slippage exceeds threshold)
- Reduce-only mode for closing positions

---

## Exchange Support

| Exchange | Spot | Futures | Margin | Status |
|----------|------|---------|--------|--------|
| Binance | Yes | Yes | Yes | Primary |
| Bybit | Yes | Yes | Yes | Failover |
| OKX | Yes | Yes | Yes | Failover |

Additional exchanges available through CCXT:
Bitget, Gate.io, KuCoin, MEXC, Huobi, Kraken, Coinbase, and 100+ more.

All exchanges support:
- REST API for order management and account queries
- WebSocket for real-time market data and order updates
- Testnet/paper trading mode
- Rate limit management with configurable buffer

---

## Backtesting

### Features

- **Event-driven architecture** matching the live trading engine
- **Realistic simulation** with configurable commission and slippage
- **Walk-forward analysis** with rolling train/test windows
- **Parameter optimization** via grid search, Bayesian optimization, or genetic algorithms
- **Multi-strategy backtesting** with portfolio-level analytics
- **Trade-by-trade analysis** with entry/exit visualization

### Configuration

```yaml
backtesting:
  initial_capital: 10000.0
  commission_pct: 0.04
  slippage_pct: 0.02
  walk_forward:
    train_periods: 60      # days
    test_periods: 20       # days
```

### Running a Backtest

```bash
# Backtest a single strategy
python main.py backtest -s momentum --symbol BTCUSDT --start-date 2024-01-01 --end-date 2025-01-01

# Backtest all enabled strategies
python main.py backtest --start-date 2024-01-01

# Optimize parameters
python main.py optimize -s momentum --method bayesian --iterations 200
```

---

## Monitoring and Alerting

### Prometheus Metrics

Exposed at `http://localhost:9090/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `autotrade_portfolio_value` | Gauge | Current portfolio value in USDT |
| `autotrade_open_positions` | Gauge | Number of open positions |
| `autotrade_total_pnl` | Counter | Cumulative realized PnL |
| `autotrade_daily_pnl` | Gauge | Current day PnL |
| `autotrade_trade_count` | Counter | Total trades executed |
| `autotrade_win_rate` | Gauge | Rolling win rate |
| `autotrade_max_drawdown` | Gauge | Current max drawdown |
| `autotrade_order_latency_ms` | Histogram | Order placement latency |
| `autotrade_strategy_signals` | Counter | Signals generated per strategy |
| `autotrade_risk_blocks` | Counter | Trades blocked by risk engine |

### Notification Channels

**Telegram** - Trade alerts, daily summaries, error notifications, circuit breaker alerts, drawdown warnings.

**Discord** - Same alert types via webhook integration.

**WebSocket** - Real time feed for custom dashboards at `/ws`.

### Alert Thresholds

| Alert | Threshold | Severity |
|-------|-----------|----------|
| Drawdown warning | 5% | Warning |
| Drawdown critical | 10% | Critical |
| Daily loss | 3% | Warning |
| Consecutive losses | 3 | Warning |
| Position held too long | 24 hours | Warning |
| Order latency | 500ms | Warning |

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Binance API key (futures-enabled)
- 2GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/mefai-dev/mefai-autotrade.git
cd mefai-autotrade

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Configuration

```bash
# Copy and edit configuration
cp config.yaml config.local.yaml

# Set environment variables
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

### Running

```bash
# Start trading engine
python main.py -c config.local.yaml start

# Start in dry-run mode (no real orders)
python main.py start --dry-run

# Check status
python main.py status

# Run a backtest
python main.py backtest -s momentum --symbol BTCUSDT

# Optimize strategy parameters
python main.py optimize -s momentum --method bayesian
```

### Paper Trading

Set `testnet: true` in the exchange configuration to use Binance Testnet. All strategies and risk controls work identically in testnet mode - only the exchange endpoint changes.

---

## Configuration Reference

The complete configuration is in `config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `general` | Application name, version, log level, timezone |
| `exchange` | API credentials, market type, leverage, failover exchanges |
| `strategies` | Which strategies are enabled, per-strategy parameters |
| `risk` | Position sizing, stop loss, drawdown limits, circuit breakers |
| `execution` | Order types, execution algorithms, slippage tolerance |
| `data` | Database path, data retention, WebSocket settings |
| `portfolio` | Allocation method, rebalancing, constraints, copy trading |
| `monitoring` | Prometheus, health check, Telegram/Discord notifications |
| `api` | HTTP server host, port, authentication, rate limits |
| `backtesting` | Historical data, commission, walk-forward settings |
| `ml` | Model directory, training schedule, feature selection, ensemble |

Environment variables override config values using `${VAR_NAME}` syntax. The `.env` file is automatically loaded via `python-dotenv`.

---

## API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | Engine status and portfolio summary |
| `GET` | `/api/positions` | Current open positions |
| `GET` | `/api/trades` | Trade history (paginated) |
| `GET` | `/api/strategies` | List registered strategies and their status |
| `GET` | `/api/performance` | Portfolio performance metrics |
| `GET` | `/api/allocation` | Current allocation weights |
| `POST` | `/api/rebalance` | Trigger manual rebalance |
| `POST` | `/api/strategy/{name}/pause` | Pause a strategy |
| `POST` | `/api/strategy/{name}/resume` | Resume a strategy |
| `GET` | `/api/backtest/results` | Backtest results |
| `GET` | `/api/copy/masters` | Copy trading master leaderboard |
| `GET` | `/api/copy/followers/{id}` | Follower stats |
| `GET` | `/metrics` | Prometheus metrics |

### WebSocket

Connect to `/ws` for real-time updates:

```json
{"type": "trade", "data": {"symbol": "BTCUSDT", "side": "LONG", "price": 65432.10}}
{"type": "position", "data": {"symbol": "BTCUSDT", "pnl": 123.45, "unrealized": true}}
{"type": "signal", "data": {"strategy": "momentum", "symbol": "ETHUSDT", "confidence": 0.82}}
{"type": "rebalance", "data": {"trades": 3, "turnover_pct": 8.5}}
```

---

## Performance Metrics

### Strategy-Level

| Metric | Description |
|--------|-------------|
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / gross loss |
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Avg Win / Avg Loss | Average winning vs losing trade |
| Max Consecutive Wins/Losses | Longest streak |
| Average Holding Time | Mean position duration |

### Portfolio-Level

| Metric | Description |
|--------|-------------|
| TWRR | Time-weighted rate of return (cash-flow adjusted) |
| MWRR/IRR | Money-weighted return considering timing of flows |
| Calmar Ratio | Annualized return / max drawdown |
| Information Ratio | Active return / tracking error vs benchmark |
| Diversification Ratio | Weighted vol sum / portfolio vol (>1 = diversified) |
| Effective Strategies | PCA-based count of independent return sources |
| VaR (95%) | Value at Risk - maximum expected loss at 95% confidence |
| CVaR (95%) | Conditional VaR - expected loss beyond VaR threshold |

---

## ML Training

### Data Requirements

- Minimum 1,000 rows of historical OHLCV data per symbol before training
- More data produces better models (10,000+ rows recommended)
- Data is automatically collected and stored in SQLite during live trading

### Feature Engineering

The ML pipeline automatically engineers 100+ features from raw OHLCV data:

- Technical indicators (RSI, MACD, BB, ADX, Stochastic, etc.)
- Price-based features (returns, log returns, volatility, momentum)
- Volume features (relative volume, OBV, volume profile stats)
- Order flow features (CVD, delta, absorption - when available)
- Time features (hour of day, day of week, session)

### Training

```bash
# Train models for all configured symbols
python -m src.strategies.machine_learning --train

# Models are saved to the configured model_dir (default: models/)
# Retraining runs automatically on the configured cron schedule
```

### Ensemble

When ensemble mode is enabled, the system trains XGBoost, Random Forest, and Neural Network models independently. Predictions are combined via soft voting (probability averaging) for more robust signals.

---

## Project Structure

```
mefai-autotrade/
|-- main.py                          # Application entry point and CLI
|-- config.yaml                      # Complete configuration file
|-- requirements.txt                 # Python dependencies
|-- pyproject.toml                   # Package metadata
|-- README.md                        # This file
|
|-- src/
|   |-- core/
|   |   |-- __init__.py
|   |   |-- order_manager.py         # Order lifecycle management
|   |
|   |-- strategies/
|   |   |-- __init__.py
|   |   |-- base.py                  # BaseStrategy + FeatureEngine (16 indicators)
|   |   |-- registry.py              # Strategy registry with auto-registration
|   |   |-- momentum.py              # RSI + MACD + ADX momentum
|   |   |-- trend_following.py       # Supertrend + EMA ribbon
|   |   |-- mean_reversion.py        # Bollinger Band + z-score
|   |   |-- breakout.py              # Range breakout + volume
|   |   |-- grid_trading.py          # Arithmetic/geometric grid
|   |   |-- scalping.py              # High frequency micro-trend
|   |   |-- smart_money.py           # ICT order blocks, FVG, BOS
|   |   |-- machine_learning.py      # XGBoost/RF/NN ensemble
|   |   |-- dca.py                   # Dollar cost averaging
|   |   |-- arbitrage.py             # Cross exchange arbitrage
|   |   |-- market_making.py         # Inventory aware market making
|   |   |-- pairs_trading.py         # Statistical pairs
|   |   |-- volume_profile.py        # POC + value area
|   |   |-- elliott_wave.py          # Wave counting + Fibonacci
|   |   |-- order_flow.py            # CVD + delta analysis
|   |   |-- composite.py             # Multi-strategy voting
|   |
|   |-- portfolio/
|   |   |-- __init__.py
|   |   |-- allocator.py             # 8 allocation methods
|   |   |-- rebalancer.py            # 4 rebalancing modes
|   |   |-- performance.py           # Attribution, factor, regime analysis
|   |   |-- copy_trading.py          # Master/follower copy trading hub
|   |
|   |-- risk/
|   |   |-- __init__.py
|   |   |-- risk_manager.py          # Portfolio risk engine
|   |   |-- position_sizer.py        # Risk-based position sizing
|   |   |-- circuit_breaker.py       # Trading halt logic
|   |
|   |-- execution/
|   |   |-- __init__.py
|   |   |-- executor.py              # Order execution engine
|   |   |-- algorithms.py            # TWAP, VWAP, Iceberg
|   |   |-- smart_router.py          # Multi-venue routing
|   |
|   |-- exchanges/
|   |   |-- __init__.py
|   |   |-- binance.py               # Binance REST + WebSocket
|   |   |-- ccxt_adapter.py          # CCXT multi-exchange adapter
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- market_data.py           # Real time data feed
|   |   |-- database.py              # SQLite persistence
|   |   |-- historical.py            # Historical data loader
|   |
|   |-- backtesting/
|   |   |-- __init__.py
|   |   |-- engine.py                # Backtest engine
|   |   |-- optimizer.py             # Parameter optimization
|   |   |-- simulator.py             # Fill simulation
|   |
|   |-- monitoring/
|       |-- __init__.py
|       |-- prometheus.py            # Metrics export
|       |-- notifications.py         # Telegram + Discord
|       |-- health.py                # Health check endpoint
|
|-- models/                          # Trained ML model files
|-- data/                            # SQLite databases and historical data
|-- logs/                            # Application logs
|-- tests/                           # Test suite
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## License

MIT License

Copyright (c) 2026 Mefai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
