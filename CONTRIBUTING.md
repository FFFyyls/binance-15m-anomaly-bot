# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[research,dev]"
```

Run the local quality gates before submitting a pull request:

```bash
ruff check .
pytest
python -m build
```

## Safety requirements

- Never commit `.env`, API credentials, databases, logs, account exports, or
  full market datasets.
- Tests must not contact Binance or submit orders.
- New order-submission paths must default to dry-run/testnet and require an
  explicit production acknowledgement.
- Database deletion or reset must require an explicit user action.
- Include regression tests for execution and risk-control changes.

Use focused commits and explain changes to fill, fee, slippage, liquidation, or
risk assumptions in the pull request description.

By submitting a contribution, you agree that it is licensed under Apache
License 2.0 as described in `LICENSE`.
