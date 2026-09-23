# Public Repository Standardization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert the local trading project into a secure, documented, testable, Apache-2.0 public Python repository ready for its first GitHub push.

**Architecture:** Use an installable `src/binance_anomaly_bot` package with separate `backtests`, `scripts`, `configs`, `tests`, and documentation. Keep all credentials and high-volume/runtime state outside Git through explicit ignored local directories.

**Tech Stack:** Python 3.10+, asyncio, aiohttp, websockets, pandas, NumPy, PyYAML, SQLite, pytest, Ruff, GitHub Actions.

---

### Task 1: Establish the public-repository boundary

**Files:**
- Modify: `.gitignore`
- Modify: `.env`
- Modify: `.env.example`
- Create: `LICENSE`
- Create: `SECURITY.md`

**Steps:**
1. Expand ignore rules for credentials, virtual environments, databases, logs, data, artifacts, reports, caches, and temporary files while keeping documented samples trackable.
2. Set local `.env` safety toggles to dry-run/testnet without reading credentials into output and set file mode `0600`.
3. Replace example secrets with placeholders and document every supported environment variable.
4. Add the canonical Apache License 2.0 text and responsible disclosure instructions.
5. Verify ignored-file behavior with `git check-ignore` after repository initialization.

### Task 2: Normalize filesystem and Python imports

**Files:**
- Move: `核心代码/*.py` to `src/binance_anomaly_bot/`
- Move: `回测系统/*.py` to `backtests/`
- Move: `实用工具/*.py` and root maintenance scripts to `scripts/`
- Move: configuration files to `configs/`
- Move: local data/results/logs into ignored standard directories
- Create: `src/binance_anomaly_bot/__init__.py`
- Create: `src/binance_anomaly_bot/__main__.py`

**Steps:**
1. Move source files with explicit source and destination paths.
2. Rewrite internal imports to English package paths.
3. Rewrite default data/config/symbol paths.
4. Preserve one curated result example and archive remaining generated output under ignored `artifacts/`.
5. Remove confirmed disposable caches, virtual environment, empty outputs, `.DS_Store`, and obsolete organization plans.
6. Parse every Python source file with `ast.parse`; expected result: zero errors.

### Task 3: Make database startup non-destructive using TDD

**Files:**
- Create: `tests/test_runtime_paths.py`
- Modify: `src/binance_anomaly_bot/cli.py`

**Steps:**
1. Write tests proving an existing database is preserved by default and deleted only with an explicit reset flag.
2. Run the tests and confirm they fail because the safe path helper does not exist.
3. Add a small helper that creates the parent directory and performs an explicit reset only when requested.
4. Add `--config`, `--database`, and `--reset-database` CLI options without weakening dry-run defaults.
5. Run the focused tests and then the full suite; expected result: all pass.

### Task 4: Harden manual order submission using TDD

**Files:**
- Create: `tests/test_order_safety.py`
- Modify: `scripts/manual/submit_test_stop_order.py`

**Steps:**
1. Write tests for the production-order guard: testnet is allowed; production without explicit acknowledgement is rejected.
2. Run the tests and confirm the unsafe existing behavior fails the guard expectation.
3. Implement the minimal guard and `--confirm-live-order` switch.
4. Run focused and full tests; expected result: all pass without network access.

### Task 5: Add packaging, dependency, and automation metadata

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_imports.py`

**Steps:**
1. Add a test importing the package and backtest entry point; verify it fails before package wiring is complete.
2. Define project metadata, runtime dependencies, optional report/dev dependencies, console entry point, pytest, and Ruff settings.
3. Keep `requirements.txt` as a simple compatibility list matching runtime dependencies.
4. Add CI for Python 3.10-3.12 with install, Ruff, and pytest steps.
5. Build/install the project in a clean temporary virtual environment and run the full suite.

### Task 6: Rewrite public documentation

**Files:**
- Rewrite: `README.md`
- Create: `docs/architecture.md`
- Create: `data/README.md`
- Create: `CONTRIBUTING.md`
- Create: `CHANGELOG.md`

**Steps:**
1. Document installation, dry-run/testnet-first operation, CLI commands, data layout, backtesting, reporting, and risk warnings.
2. Document component ownership and data flow.
3. Document how users obtain their own market data without committing it.
4. Document contribution, testing, and security expectations.
5. Validate every documented local path and CLI `--help` command.

### Task 7: Initialize and audit Git

**Files:**
- Create: local `.git/` metadata only

**Steps:**
1. Initialize the repository with branch `main` and create a working branch for standardization if needed.
2. Verify `.env`, databases, market data, logs, virtual environments, and artifacts are ignored.
3. Stage with `git add .`, inspect `git status`, `git diff --cached --stat`, and the complete staged file list.
4. Search staged files for credential patterns and confirm the real Binance key/secret have no matches.
5. Reject any staged file larger than 5 MiB or any forbidden local-state path.
6. Run the full test, lint, packaging, and documentation command set.
7. Create the local initial commit only if Git identity is configured; never create a remote or push.

## Review focus

- Credentials or account/runtime data accidentally staged.
- Any code path capable of live orders without two explicit signals.
- Database deletion without explicit opt-in.
- Stale Chinese/legacy filesystem references that make documented commands fail.
- Backtest imports broken by the package rename.
- Generated data accidentally included in the distributable package.
