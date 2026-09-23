def test_runtime_package_imports() -> None:
    import binance_anomaly_bot
    from binance_anomaly_bot import cli

    assert binance_anomaly_bot.__version__ == "0.1.0"
    assert callable(cli.run)


def test_backtest_module_imports() -> None:
    from backtests import run_backtest

    assert callable(run_backtest.summarize)
