import asyncio
from pathlib import Path
import sqlite3

import pytest

from binance_anomaly_bot.storage import SqliteStore


def _initialize(path: Path) -> SqliteStore:
    store = SqliteStore(str(path))
    asyncio.run(store.init())
    return store


def test_historical_rows_survive_store_restart(tmp_path: Path) -> None:
    database = tmp_path / "bot.sqlite3"
    first = _initialize(database)
    first.conn.execute(
        "INSERT INTO trade_log (时间戳, symbol, 方向, 事件) VALUES (?, ?, ?, ?)",
        ("2026-09-20 00:00:00.000", "BTCUSDT", "BUY", "TEST"),
    )
    first.conn.commit()
    first.conn.close()

    second = _initialize(database)
    count = second.conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
    second.conn.close()

    assert count == 1


def test_transient_position_rows_are_cleared_on_restart(tmp_path: Path) -> None:
    database = tmp_path / "bot.sqlite3"
    first = _initialize(database)
    first.conn.execute(
        "INSERT INTO positions_live (symbol, 方向) VALUES (?, ?)",
        ("BTCUSDT", "BUY"),
    )
    first.conn.commit()
    first.conn.close()

    second = _initialize(database)
    count = second.conn.execute("SELECT COUNT(*) FROM positions_live").fetchone()[0]
    second.conn.close()

    assert count == 0


def test_schema_mismatch_refuses_destructive_reset(tmp_path: Path) -> None:
    database = tmp_path / "bot.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE meta(键 TEXT PRIMARY KEY, 值 TEXT)")
    connection.execute("INSERT INTO meta(键, 值) VALUES('schema_version', '2')")
    connection.execute("CREATE TABLE historical_record(value TEXT)")
    connection.execute("INSERT INTO historical_record(value) VALUES('keep me')")
    connection.commit()
    connection.close()

    store = SqliteStore(str(database))
    with pytest.raises(RuntimeError, match="schema version"):
        asyncio.run(store.init())

    connection = sqlite3.connect(database)
    value = connection.execute("SELECT value FROM historical_record").fetchone()[0]
    connection.close()

    assert value == "keep me"


def test_unversioned_existing_database_is_preserved(tmp_path: Path) -> None:
    database = tmp_path / "bot.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE historical_record(value TEXT)")
    connection.execute("INSERT INTO historical_record(value) VALUES('keep me')")
    connection.commit()
    connection.close()

    store = SqliteStore(str(database))
    with pytest.raises(RuntimeError, match="no recognized schema version"):
        asyncio.run(store.init())

    connection = sqlite3.connect(database)
    value = connection.execute("SELECT value FROM historical_record").fetchone()[0]
    connection.close()

    assert value == "keep me"
