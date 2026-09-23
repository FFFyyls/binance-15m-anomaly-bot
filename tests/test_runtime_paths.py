from pathlib import Path

from binance_anomaly_bot import cli


def test_existing_database_is_preserved_by_default(tmp_path: Path) -> None:
    database = tmp_path / "runtime" / "bot.sqlite3"
    database.parent.mkdir()
    database.write_bytes(b"existing database")

    assert hasattr(cli, "prepare_database_path"), "safe database path helper is missing"
    resolved = cli.prepare_database_path(database, reset=False)

    assert resolved == database
    assert database.read_bytes() == b"existing database"


def test_database_is_removed_only_with_explicit_reset(tmp_path: Path) -> None:
    database = tmp_path / "runtime" / "bot.sqlite3"
    database.parent.mkdir()
    database.write_bytes(b"replace me")

    assert hasattr(cli, "prepare_database_path"), "safe database path helper is missing"
    cli.prepare_database_path(database, reset=True)

    assert not database.exists()


def test_cli_defaults_do_not_reset_database() -> None:
    assert hasattr(cli, "build_parser"), "runtime argument parser is missing"
    args = cli.build_parser().parse_args([])

    assert args.reset_database is False
    assert args.config == Path("configs/strategy.yaml")
    assert args.database == Path("data/runtime/bot.sqlite3")
