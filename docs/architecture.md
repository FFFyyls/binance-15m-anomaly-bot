# Architecture

## Runtime data flow

```text
Binance REST ───────┐
                    ├─> universe + symbol metadata
Market WebSocket ───┤
                    ├─> strategy engine ─> order execution
User WebSocket ─────┘          │                 │
                               └──── SQLite <────┘
```

`binance_anomaly_bot.cli` owns process startup and shutdown. It loads the YAML
configuration and local environment, performs connectivity and account checks,
constructs the exchange clients, builds the trading universe, and starts the
asynchronous workers.

`binance_client` implements signed and public USDT-M REST calls with time
synchronization, request pacing, retry, and rate-limit backoff. `market_ws`
maintains book, mark-price, and aggregate-trade state. `user_ws` consumes account
and order updates.

`strategy` owns signal generation, entry evaluation, leverage management,
position reconciliation, stop and take-profit orders, time exits, daily-loss
limits, and WebSocket-health controls. It is intentionally kept compatible with
the current strategy behavior; future work should split signal, execution,
position, and risk responsibilities into smaller modules.

`storage` persists operational telemetry in SQLite. Existing databases are
preserved by default. Runtime state is local and must never be committed.

## Research path

`backtests.run_backtest` loads symbol/day CSV files, reconstructs the configured
signal and execution assumptions, applies portfolio constraints, and writes
trade tables plus a JSON summary. Optimization and sweep modules call the same
backtest functions with generated configurations. `generate_report` converts a
completed run into tabular and visual reports.

Live execution and backtesting share configuration concepts but are separate
engines. A backtest result is not evidence that the live engine will produce the
same fills, latency, funding, liquidity, or risk outcome.

## Trust boundaries

- `.env` contains local credentials and is never tracked.
- `data/`, `artifacts/`, and `logs/` contain potentially sensitive or large
  runtime state and are ignored.
- Network responses and exchange errors are untrusted inputs.
- Manual order utilities require an additional confirmation when testnet is
  disabled.
- The repository does not provide custody, withdrawal, or secret-management
  functionality.
