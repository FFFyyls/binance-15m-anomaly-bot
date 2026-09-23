# Public Repository Standardization Design

## Objective

Prepare the Binance anomaly-trading project for a public GitHub repository under Apache License 2.0 without publishing credentials, account activity, runtime databases, logs, bulk market data, caches, or uncurated experiment artifacts.

## Repository architecture

The project will use a conventional Python `src` layout. Runtime code will live in `src/binance_anomaly_bot/`, research code in `backtests/`, operational utilities in `scripts/`, configuration in `configs/`, automated checks in `tests/`, and human documentation in `docs/`. The installable package will expose a console command and support `python -m binance_anomaly_bot` after installation. Existing Chinese identifiers inside SQLite tables and user-facing log messages may remain because changing persisted schemas is outside this repository-organization task; file and directory names, imports, commands, and documentation will be English.

Large or sensitive local state will use predictable ignored locations: `data/` for market/runtime data, `artifacts/` for research output, `logs/` for runtime logs, and `tmp/` for transient files. A small, curated backtest example may be tracked under `examples/results/` to demonstrate output format and reproducibility without publishing the full experiment archive.

## Security boundaries

The real `.env` remains local, receives mode `0600`, and is ignored. Its safety defaults are changed to dry-run and testnet, while credentials are neither printed nor copied. The example environment contains placeholders and conservative defaults. The trading entry point will never delete an existing database implicitly; database reset becomes an explicit option. Parent directories are created safely.

Manual order-submission utilities must be fail-closed. They default to testnet and require an explicit command-line acknowledgement before any production order can be submitted. Git ignore rules cover every local data location and generated artifact class. The staged Git index will be scanned for secret-like material and forbidden file categories before the initial release commit.

## Testing and release readiness

Behavioral tests cover database preservation, explicit reset, safe runtime defaults, and live-order confirmation. Static checks cover imports and syntax after the package rename. GitHub Actions will run tests and lint checks on supported Python versions. Dependencies move to `pyproject.toml`, with reporting and development dependencies in optional groups. `requirements.txt` remains as a compatibility installation file.

The public repository includes Apache License 2.0, a rewritten README, contribution and security guidance, a data-policy document, and a concise architecture description. No remote repository is created or pushed automatically; the local repository will be initialized on `main`, validated, and left ready for the user to connect to GitHub.
