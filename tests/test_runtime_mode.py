import pytest

from binance_anomaly_bot import cli


def test_missing_mode_variables_default_to_dry_run_testnet() -> None:
    assert hasattr(cli, "resolve_runtime_mode"), "fail-closed runtime mode resolver is missing"

    dry_run, use_testnet = cli.resolve_runtime_mode({})

    assert dry_run is True
    assert use_testnet is True


def test_live_production_requires_confirmation_phrase() -> None:
    assert hasattr(cli, "resolve_runtime_mode"), "fail-closed runtime mode resolver is missing"

    with pytest.raises(RuntimeError, match="CONFIRM_LIVE_TRADING"):
        cli.resolve_runtime_mode({"DRY_RUN": "0", "USE_TESTNET": "0"})


def test_live_production_accepts_exact_confirmation_phrase() -> None:
    assert hasattr(cli, "resolve_runtime_mode"), "fail-closed runtime mode resolver is missing"

    mode = cli.resolve_runtime_mode(
        {
            "DRY_RUN": "0",
            "USE_TESTNET": "0",
            "CONFIRM_LIVE_TRADING": "I_UNDERSTAND",
        }
    )

    assert mode == (False, False)


def test_invalid_mode_value_is_rejected() -> None:
    assert hasattr(cli, "resolve_runtime_mode"), "fail-closed runtime mode resolver is missing"

    with pytest.raises(ValueError, match="DRY_RUN"):
        cli.resolve_runtime_mode({"DRY_RUN": "false", "USE_TESTNET": "1"})
