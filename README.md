# Binance 15-Minute Anomaly Bot

An experimental Binance USDT-M futures toolkit for real-time anomaly
monitoring, order execution, historical backtesting, parameter exploration,
and report generation.

> [!CAUTION]
> This project can submit real leveraged futures orders. It is research
> software, not investment advice. Start with `DRY_RUN=1` and
> `USE_TESTNET=1`, use exchange-side least-privilege API permissions, and
> review every strategy and risk parameter before considering production use.

## Features

- asynchronous Binance REST, market WebSocket, and user-data WebSocket clients;
- dynamic USDT-M universe selection and symbol filters;
- price/volume anomaly signals with optional Bollinger and volatility filters;
- long/short execution, position synchronization, stop loss, staged take profit,
  trailing exits, and runtime risk controls;
- SQLite operational telemetry;
- portfolio-aware backtests and parameter sweeps over local 20-second bars.

## Requirements

- Python 3.10 or newer;
- Binance Futures testnet credentials for end-to-end runtime testing;
- local historical data for backtesting. Bulk market data is intentionally not
  stored in Git.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[research,reports]"
cp .env.example .env
```

Populate `.env` with separate testnet credentials. Keep the safe defaults:

```dotenv
DRY_RUN=1
USE_TESTNET=1
```

Production order submission is fail-closed. It requires all three explicit
settings: `DRY_RUN=0`, `USE_TESTNET=0`, and
`CONFIRM_LIVE_TRADING=I_UNDERSTAND`.

## Run the bot

```bash
binance-anomaly-bot --config configs/strategy.yaml
# equivalent after installation:
python -m binance_anomaly_bot --config configs/strategy.yaml
```

The default database is `data/runtime/bot.sqlite3`. Existing databases are
preserved. Deletion requires the explicit `--reset-database` option.

The program performs network and account checks even in dry-run mode. Do not
run it with credentials or an account you are not prepared to query.

## Backtesting

Historical files use this layout:

```text
data/raw/<dataset>/<SYMBOL>/20s/YYYY-MM-DD.csv
```

Example:

```bash
python -m backtests.run_backtest \
  --config configs/strategy.yaml \
  --symbols_file configs/symbols/symbols_top60.txt \
  --data data/raw/before_2026_02_01 \
  --use_symbol_range \
  --use_long_strategy \
  --out artifacts/backtests/local_run
```

Generated output belongs in `artifacts/` and is ignored by Git. A small curated
output example is available in `examples/results/multi_strategy_before_final/`.
Backtest output is hypothetical and does not establish future performance.

## Data tools

```bash
python scripts/data/download_aggtrades_20s_rest.py --help
python scripts/data/scan_missing_data.py
python scripts/data/fill_missing_data.py
```

See [data/README.md](data/README.md) for the local data policy.

## Reports

Install the `reports` optional dependency group, then run:

```bash
python -m backtests.generate_report \
  --trades artifacts/backtests/local_run/trades.csv \
  --summary artifacts/backtests/local_run/summary.json \
  --out artifacts/reports/local_run
```

## Repository layout

```text
src/binance_anomaly_bot/  installable live-trading package
backtests/                backtest, optimization, and reporting modules
scripts/                  data and manually invoked operational utilities
configs/                  strategy configuration and symbol lists
tests/                    offline safety and import tests
docs/                     architecture and implementation notes
examples/                 small tracked examples only
data/                     ignored local market/runtime data
artifacts/                ignored generated research output
logs/                     ignored runtime logs
```

Detailed component boundaries are documented in
[docs/architecture.md](docs/architecture.md).

## Development

```bash
python -m pip install -e ".[research,dev]"
ruff check .
pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before
opening a pull request or reporting a vulnerability.

## License

Licensed under the [Apache License 2.0](LICENSE).
