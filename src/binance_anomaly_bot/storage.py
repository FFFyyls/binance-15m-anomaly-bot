import asyncio
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

SCHEMA_VERSION = 3
_TZ_CN = timezone(timedelta(hours=8))
_ERROR_DETAIL_MAX = 8000

def _format_ts(ts_ms):
    if ts_ms is None:
        return None
    if isinstance(ts_ms, str):
        return ts_ms
    try:
        ts_val = float(ts_ms)
    except (TypeError, ValueError):
        return None
    if ts_val <= 0:
        return None
    dt = datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc).astimezone(_TZ_CN)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def _format_detail(detail):
    if detail is None:
        return None
    if isinstance(detail, (dict, list, tuple)):
        try:
            text = json.dumps(detail, ensure_ascii=True, separators=(",", ":"), default=str)
        except Exception:
            text = str(detail)
    else:
        text = str(detail)
    if len(text) > _ERROR_DETAIL_MAX:
        return text[:_ERROR_DETAIL_MAX] + "...(truncated)"
    return text

class SqliteStore:
    def __init__(self, path: str):
        self.path = path
        self.conn: Optional[sqlite3.Connection] = None
        self.lock = asyncio.Lock()

    async def init(self):
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS meta(键 TEXT PRIMARY KEY, 值 TEXT)")
        cur.execute("SELECT 值 FROM meta WHERE 键='schema_version'")
        row = cur.fetchone()
        if row and row[0] != str(SCHEMA_VERSION):
            actual_version = row[0]
            self.conn.close()
            self.conn = None
            raise RuntimeError(
                f"Unsupported database schema version {actual_version}; "
                f"expected {SCHEMA_VERSION}. Back up the database and use "
                "--reset-database only if history can be discarded."
            )
        if not row:
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT IN ('meta', 'sqlite_sequence')"
            )
            existing_tables = [name for (name,) in cur.fetchall()]
            if existing_tables:
                self.conn.close()
                self.conn = None
                raise RuntimeError(
                    "Database has no recognized schema version; back it up and "
                    "use --reset-database only if history can be discarded."
                )
            self._reset_schema(cur)
            cur.execute("INSERT OR REPLACE INTO meta(键, 值) VALUES(?,?)", ("schema_version", str(SCHEMA_VERSION)))
        self._ensure_latency_table(cur)
        self._ensure_trade_log_columns(cur)
        self._ensure_binance_views(cur)
        self._ensure_signal_proximity_table(cur)
        self._ensure_signal_debug_table(cur)
        self._ensure_error_log_table(cur)
        self._clear_transient_tables(cur)
        self.conn.commit()

    def _reset_schema(self, cur):
        cur.execute("DROP VIEW IF EXISTS entry_latency_cn")
        cur.execute("DROP VIEW IF EXISTS 涨跌幅阈值监控")
        cur.execute("DROP VIEW IF EXISTS 当前仓位")
        cur.execute("DROP VIEW IF EXISTS 历史成交")
        cur.execute("DROP VIEW IF EXISTS 历史仓位")
        cur.execute("DROP VIEW IF EXISTS 历史委托")
        cur.execute("DROP TABLE IF EXISTS events")
        cur.execute("DROP TABLE IF EXISTS positions")
        cur.execute("DROP TABLE IF EXISTS trigger_log")
        cur.execute("DROP TABLE IF EXISTS trade_log")
        cur.execute("DROP TABLE IF EXISTS positions_live")
        cur.execute("DROP TABLE IF EXISTS orders")
        cur.execute("DROP TABLE IF EXISTS entry_latency")
        cur.execute("DROP TABLE IF EXISTS signal_proximity")
        cur.execute("DROP TABLE IF EXISTS error_log")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trigger_log(
            编号 INTEGER PRIMARY KEY AUTOINCREMENT,
            时间戳 INTEGER,
            symbol TEXT,
            方向 TEXT,
            收益率15m REAL,
            成交量15m REAL,
            基础成交量15m REAL,
            成交量基线 REAL,
            收盘价15m REAL,
            触发原因 TEXT,
            进场价格 REAL,
            进场原因 TEXT,
            截止时间戳 INTEGER
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_log(
            编号 INTEGER PRIMARY KEY AUTOINCREMENT,
            时间戳 INTEGER,
            持仓编号 INTEGER,
            symbol TEXT,
            方向 TEXT,
            事件 TEXT,
            交易类型 TEXT,
            是否追仓 INTEGER,
            数量 REAL,
            价格 REAL,
            订单编号 INTEGER,
            状态 TEXT,
            原因 TEXT,
            进场价格 REAL,
            止损价 REAL,
            峰值价 REAL,
            止盈阶段 INTEGER,
            盈亏 REAL,
            盈亏比例 REAL,
            信号价格 REAL,
            滑点bps REAL,
            手续费 REAL,
            手续费币种 TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions_live(
            symbol TEXT,
            方向 TEXT,
            持仓编号 INTEGER,
            时间戳 INTEGER,
            持仓价格 REAL,
            持仓数量 REAL,
            实时价格 REAL,
            实时盈亏 REAL,
            实时盈亏比例 REAL,
            止损价 REAL,
            止盈价 REAL,
            峰值价 REAL,
            止盈阶段 INTEGER,
            状态 TEXT,
            PRIMARY KEY(symbol, 方向)
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders(
            编号 INTEGER PRIMARY KEY AUTOINCREMENT,
            时间戳 INTEGER,
            symbol TEXT,
            客户订单编号 TEXT,
            订单编号 INTEGER,
            方向 TEXT,
            类型 TEXT,
            价格 REAL,
            数量 REAL,
            状态 TEXT,
            均价 REAL,
            成交数量 REAL
        )""")
        self._ensure_latency_table(cur)
        self._ensure_trade_log_columns(cur)
        self._ensure_binance_views(cur)
        self._ensure_signal_proximity_table(cur)
        self._ensure_signal_debug_table(cur)
        self._ensure_error_log_table(cur)

    def _ensure_latency_table(self, cur):
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entry_latency(
            编号 INTEGER PRIMARY KEY AUTOINCREMENT,
            信号时间戳 INTEGER,
            成交时间戳 INTEGER,
            延迟ms INTEGER,
            symbol TEXT,
            方向 TEXT,
            订单编号 INTEGER,
            信号价格 REAL,
            成交价格 REAL,
            滑点bps REAL,
            挂单时间戳 INTEGER
        )""")
        cur.execute("PRAGMA table_info(entry_latency)")
        cols = {row[1] for row in cur.fetchall()}
        if "挂单时间戳" not in cols:
            cur.execute("ALTER TABLE entry_latency ADD COLUMN 挂单时间戳 INTEGER")
        if "信号到挂单延迟ms" not in cols:
            cur.execute("ALTER TABLE entry_latency ADD COLUMN 信号到挂单延迟ms INTEGER")
        if "信号到成交延迟ms" not in cols:
            cur.execute("ALTER TABLE entry_latency ADD COLUMN 信号到成交延迟ms INTEGER")
        cur.execute("""
        CREATE VIEW IF NOT EXISTS entry_latency_cn AS
        SELECT
            信号时间戳 AS 信号时间戳,
            挂单时间戳 AS 挂单时间戳,
            成交时间戳 AS 成交时间戳,
            信号到挂单延迟ms AS 信号到挂单延迟ms,
            信号到成交延迟ms AS 信号到成交延迟ms,
            symbol AS 币种,
            方向 AS 方向,
            订单编号 AS 订单编号,
            信号价格 AS 信号价格,
            成交价格 AS 成交价格,
            滑点bps AS 滑点基点
        FROM entry_latency
        """)

    def _ensure_trade_log_columns(self, cur):
        cur.execute("PRAGMA table_info(trade_log)")
        cols = {row[1] for row in cur.fetchall()}
        if not cols:
            return
        col_defs = {
            "信号价格": "REAL",
            "滑点bps": "REAL",
            "手续费": "REAL",
            "手续费币种": "TEXT",
        }
        for name, col_type in col_defs.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE trade_log ADD COLUMN {name} {col_type}")

    def _ensure_binance_views(self, cur):
        cur.execute("DROP VIEW IF EXISTS 当前仓位")
        cur.execute("DROP VIEW IF EXISTS 仓位")
        cur.execute("DROP VIEW IF EXISTS 历史成交")
        cur.execute("DROP VIEW IF EXISTS 历史仓位")
        cur.execute("DROP VIEW IF EXISTS 历史委托")
        cur.execute("DROP VIEW IF EXISTS 成交记录")
        cur.execute("""
        CREATE VIEW IF NOT EXISTS 仓位 AS
        SELECT
            symbol AS 合约,
            方向 AS 方向,
            持仓编号 AS 持仓编号,
            时间戳 AS 更新时间戳,
            持仓数量 AS 持仓数量,
            持仓价格 AS 开仓均价,
            实时价格 AS 标记价格,
            ROUND(持仓数量 * 实时价格, 8) AS 名义价值,
            实时盈亏 AS 未实现盈亏,
            实时盈亏比例 AS 盈亏比例,
            止损价 AS 止损价,
            止盈价 AS 止盈价,
            峰值价 AS 峰值价,
            止盈阶段 AS 止盈阶段,
            状态 AS 状态
        FROM positions_live
        """)
        cur.execute("""
        CREATE VIEW IF NOT EXISTS 当前仓位 AS
        SELECT * FROM 仓位
        """)
        cur.execute("""
        CREATE VIEW IF NOT EXISTS 成交记录 AS
        SELECT
            时间戳 AS 成交时间戳,
            symbol AS 合约,
            方向 AS 方向,
            事件 AS 事件,
            交易类型 AS 成交类型,
            价格 AS 成交价格,
            数量 AS 成交数量,
            ROUND(数量 * 价格, 8) AS 成交额,
            信号价格 AS 信号价格,
            滑点bps AS 滑点基点,
            手续费 AS 手续费,
            手续费币种 AS 手续费币种,
            盈亏 AS 已实现盈亏,
            盈亏比例 AS 盈亏比例,
            订单编号 AS 订单编号,
            状态 AS 状态,
            原因 AS 原因,
            持仓编号 AS 持仓编号,
            进场价格 AS 进场价格,
            止损价 AS 止损价,
            峰值价 AS 峰值价,
            止盈阶段 AS 止盈阶段
        FROM trade_log
        WHERE 状态 IN ('OK', 'FILLED', 'SIMULATED', 'PARTIALLY_FILLED')
        """)
        cur.execute("""
        CREATE VIEW IF NOT EXISTS 历史成交 AS
        SELECT * FROM 成交记录
        """)
        cur.execute("""
        CREATE VIEW IF NOT EXISTS 历史仓位 AS
        WITH fills AS (
            SELECT *
            FROM trade_log
            WHERE 状态 IN ('OK', 'FILLED', 'SIMULATED', 'PARTIALLY_FILLED', 'SYNC')
        ),
        agg AS (
            SELECT
                持仓编号 AS 开仓时间戳,
                MIN(时间戳) AS 首次成交时间戳,
                MAX(symbol) AS 合约,
                MAX(方向) AS 方向,
                SUM(CASE WHEN 事件 IN ('ENTRY', 'ADD') THEN 数量 ELSE 0 END) AS 开仓数量,
                SUM(CASE WHEN 事件 IN ('EXIT', 'REDUCE') THEN 数量 ELSE 0 END) AS 平仓数量,
                MAX(CASE WHEN 事件 = 'EXIT' THEN 时间戳 END) AS 平仓时间戳,
                MAX(CASE WHEN 事件 = 'ENTRY' THEN 进场价格 END) AS 开仓均价,
                MAX(CASE WHEN 事件 = 'EXIT' THEN 价格 END) AS 平仓价格,
                SUM(CASE WHEN 事件 = 'EXIT' THEN 盈亏 ELSE 0 END) AS 已实现盈亏,
                SUM(CASE WHEN 事件 = 'EXIT' THEN 盈亏比例 ELSE 0 END) AS 盈亏比例
            FROM fills
            WHERE 持仓编号 IS NOT NULL
            GROUP BY 持仓编号
        )
        SELECT
            开仓时间戳,
            首次成交时间戳,
            平仓时间戳,
            CASE
                WHEN 平仓时间戳 IS NOT NULL THEN ROUND((julianday(平仓时间戳) - julianday(首次成交时间戳)) * 86400.0, 2)
                ELSE NULL
            END AS 持仓时长秒,
            合约,
            方向,
            开仓均价,
            平仓价格,
            开仓数量,
            平仓数量,
            已实现盈亏,
            盈亏比例,
            CASE WHEN 平仓数量 > 0 THEN 1 ELSE 0 END AS 是否已平仓,
            CASE WHEN 开仓数量 > 0 AND 平仓数量 >= 开仓数量 THEN 1 ELSE 0 END AS 是否完全平仓
        FROM agg
        """)
        cur.execute("""
        CREATE VIEW 历史委托 AS
        SELECT
            o.时间戳 AS 委托时间戳,
            o.symbol AS 合约,
            o.方向 AS 方向,
            o.类型 AS 委托类型,
            o.价格 AS 委托价格,
            o.数量 AS 委托数量,
            o.成交数量 AS 成交数量,
            o.均价 AS 成交均价,
            ROUND(o.成交数量 * o.均价, 8) AS 成交额,
            o.状态 AS 状态,
            o.订单编号 AS 订单编号,
            o.客户订单编号 AS 客户订单编号
        FROM orders o
        WHERE o.时间戳 = (
            SELECT MAX(o2.时间戳)
            FROM orders o2
        WHERE o2.symbol = o.symbol
              AND o2.订单编号 = o.订单编号
        )
        """)

    def _ensure_signal_proximity_table(self, cur):
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signal_proximity(
            symbol TEXT,
            方向 TEXT,
            时间戳 TEXT,
            实际涨跌幅 REAL,
            阈值 REAL,
            距离 REAL,
            起始价格 REAL,
            起始时间戳 TEXT,
            终止价格 REAL,
            终止时间戳 TEXT,
            PRIMARY KEY(symbol, 方向)
        )""")
        cur.execute("PRAGMA table_info(signal_proximity)")
        cols = {row[1] for row in cur.fetchall()}
        if cols and "symbol" not in cols:
            cur.execute("DROP TABLE IF EXISTS signal_proximity")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS signal_proximity(
                symbol TEXT,
                方向 TEXT,
                时间戳 TEXT,
                实际涨跌幅 REAL,
                阈值 REAL,
                距离 REAL,
                起始价格 REAL,
                起始时间戳 TEXT,
                终止价格 REAL,
                终止时间戳 TEXT,
                PRIMARY KEY(symbol, 方向)
            )""")
            cur.execute("PRAGMA table_info(signal_proximity)")
            cols = {row[1] for row in cur.fetchall()}
        col_defs = {
            "方向": "TEXT",
            "时间戳": "TEXT",
            "实际涨跌幅": "REAL",
            "阈值": "REAL",
            "距离": "REAL",
            "起始价格": "REAL",
            "起始时间戳": "TEXT",
            "终止价格": "REAL",
            "终止时间戳": "TEXT",
        }
        for name, col_type in col_defs.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE signal_proximity ADD COLUMN {name} {col_type}")
        cur.execute("DROP VIEW IF EXISTS 涨跌幅阈值监控")

    def _ensure_signal_debug_table(self, cur):
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signal_debug(
            编号 INTEGER PRIMARY KEY AUTOINCREMENT,
            时间戳 TEXT,
            symbol TEXT,
            方向 TEXT,
            事件 TEXT,
            原因 TEXT,
            触发模式 TEXT,
            实际涨跌幅 REAL,
            阈值 REAL,
            成交量信号 REAL,
            成交量阈值 REAL,
            成交量通过 INTEGER,
            成交量基线 REAL,
            成交量倍数 REAL,
            use_percentile INTEGER,
            布林命中 INTEGER,
            布林上轨 REAL,
            布林下轨 REAL,
            价差bps REAL,
            资金费率 REAL,
            open15 REAL,
            low15 REAL,
            high15 REAL,
            close15 REAL,
            last_price REAL,
            备注 TEXT
        )""")
        cur.execute("PRAGMA table_info(signal_debug)")
        cols = {row[1] for row in cur.fetchall()}
        required = {
            "时间戳": "TEXT",
            "symbol": "TEXT",
            "方向": "TEXT",
            "事件": "TEXT",
            "原因": "TEXT",
            "触发模式": "TEXT",
            "实际涨跌幅": "REAL",
            "阈值": "REAL",
            "成交量信号": "REAL",
            "成交量阈值": "REAL",
            "成交量通过": "INTEGER",
            "成交量基线": "REAL",
            "成交量倍数": "REAL",
            "use_percentile": "INTEGER",
            "布林命中": "INTEGER",
            "布林上轨": "REAL",
            "布林下轨": "REAL",
            "价差bps": "REAL",
            "资金费率": "REAL",
            "open15": "REAL",
            "low15": "REAL",
            "high15": "REAL",
            "close15": "REAL",
            "last_price": "REAL",
            "备注": "TEXT",
        }
        for name, col_type in required.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE signal_debug ADD COLUMN {name} {col_type}")

    def _ensure_error_log_table(self, cur):
        cur.execute("""
        CREATE TABLE IF NOT EXISTS error_log(
            编号 INTEGER PRIMARY KEY AUTOINCREMENT,
            时间戳 TEXT,
            阶段 TEXT,
            symbol TEXT,
            方向 TEXT,
            事件 TEXT,
            原因 TEXT,
            错误 TEXT,
            详情 TEXT,
            订单编号 INTEGER,
            持仓编号 INTEGER
        )""")
        cur.execute("PRAGMA table_info(error_log)")
        cols = {row[1] for row in cur.fetchall()}
        col_defs = {
            "时间戳": "TEXT",
            "阶段": "TEXT",
            "symbol": "TEXT",
            "方向": "TEXT",
            "事件": "TEXT",
            "原因": "TEXT",
            "错误": "TEXT",
            "详情": "TEXT",
            "订单编号": "INTEGER",
            "持仓编号": "INTEGER",
        }
        for name, col_type in col_defs.items():
            if name not in cols:
                cur.execute(f"ALTER TABLE error_log ADD COLUMN {name} {col_type}")

    def _clear_transient_tables(self, cur):
        cur.execute("DELETE FROM positions_live")
        cur.execute("DELETE FROM signal_proximity")

    async def insert_trigger(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            ts = _format_ts(row["ts"])
            deadline_ts = _format_ts(row["deadline_ts"])
            cur.execute("INSERT INTO trigger_log VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ts, row["symbol"], row["side"], row["r15"], row["vol15"], row["base_vol15"],
                         row["baseline"], row["close15"], row["trigger_reason"], row["entry_price"],
                         row["entry_reason"], deadline_ts))
            self.conn.commit()

    async def upsert_order(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            ts = _format_ts(row["ts"])
            cur.execute("INSERT INTO orders VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?)",
                        (ts, row["symbol"], row["client_order_id"], row["order_id"], row["side"], row["type"],
                         row["price"], row["qty"], row["status"], row["avg_price"], row["executed_qty"]))
            self.conn.commit()

    async def insert_entry_latency(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            signal_ts = _format_ts(row["signal_ts"])
            order_accept_ts = _format_ts(row["order_accept_ts"])
            fill_ts = _format_ts(row["fill_ts"])
            cur.execute("""
            INSERT INTO entry_latency(
                信号时间戳,
                挂单时间戳,
                成交时间戳,
                信号到挂单延迟ms,
                信号到成交延迟ms,
                symbol,
                方向,
                订单编号,
                信号价格,
                成交价格,
                滑点bps
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
                        (signal_ts, order_accept_ts, fill_ts, row["latency_order_ms"],
                         row["latency_fill_ms"], row["symbol"], row["side"], row["order_id"],
                         row["signal_price"], row["fill_price"], row["slippage_bps"]))
            self.conn.commit()

    async def upsert_position_live(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            position_id = _format_ts(row["position_id"])
            ts = _format_ts(row["ts"])
            cur.execute("""
            INSERT INTO positions_live VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, 方向) DO UPDATE SET
                持仓编号=excluded.持仓编号,
                时间戳=excluded.时间戳,
                持仓价格=excluded.持仓价格,
                持仓数量=excluded.持仓数量,
                实时价格=excluded.实时价格,
                实时盈亏=excluded.实时盈亏,
                实时盈亏比例=excluded.实时盈亏比例,
                止损价=excluded.止损价,
                止盈价=excluded.止盈价,
                峰值价=excluded.峰值价,
                止盈阶段=excluded.止盈阶段,
                状态=excluded.状态
            """,
                        (row["symbol"], row["side"], position_id, ts, row["entry_price"], row["qty"],
                         row["last_price"], row["pnl"], row["pnl_pct"], row["stop_price"], row["tp_price"],
                         row["peak_price"], row["tp_state"], row["status"]))
            self.conn.commit()

    async def upsert_signal_proximity(self, rows):
        if not rows:
            return
        async with self.lock:
            cur = self.conn.cursor()
            cur.executemany("""
            INSERT INTO signal_proximity(
                symbol,
                方向,
                时间戳,
                实际涨跌幅,
                阈值,
                距离,
                起始价格,
                起始时间戳,
                终止价格,
                终止时间戳
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, 方向) DO UPDATE SET
                时间戳=excluded.时间戳,
                实际涨跌幅=excluded.实际涨跌幅,
                阈值=excluded.阈值,
                距离=excluded.距离,
                起始价格=excluded.起始价格,
                起始时间戳=excluded.起始时间戳,
                终止价格=excluded.终止价格,
                终止时间戳=excluded.终止时间戳
            """, [
                (
                    row["symbol"],
                    row["side"],
                    _format_ts(row["ts"]),
                    row["r15"],
                    row["threshold"],
                    row["diff"],
                    row.get("start_price"),
                    _format_ts(row.get("start_ts")),
                    row.get("end_price"),
                    _format_ts(row.get("end_ts")),
                )
                for row in rows
            ])
            self.conn.commit()

    async def insert_trade_log(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            ts = _format_ts(row["ts"])
            position_id = _format_ts(row["position_id"])
            cur.execute("INSERT INTO trade_log VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ts, position_id, row["symbol"], row["side"], row["event"], row["trade_type"],
                         row["is_add_on"], row["qty"], row["price"], row["order_id"], row["status"], row["reason"],
                         row["entry_price"], row["stop_price"], row["peak_price"], row["tp_state"], row["pnl"],
                         row["pnl_pct"], row.get("signal_price"), row.get("slippage_bps"), row.get("fee"),
                         row.get("fee_asset")))
            self.conn.commit()

    async def update_trade_log_fee(self, *, order_id: int, fee: Optional[float] = None,
                                   fee_asset: Optional[str] = None) -> None:
        if not order_id:
            return
        if fee is None and fee_asset is None:
            return
        async with self.lock:
            cur = self.conn.cursor()
            if fee is not None and fee_asset is not None:
                cur.execute("UPDATE trade_log SET 手续费=?, 手续费币种=? WHERE 订单编号=?",
                            (fee, fee_asset, order_id))
            elif fee is not None:
                cur.execute("UPDATE trade_log SET 手续费=? WHERE 订单编号=?",
                            (fee, order_id))
            else:
                cur.execute("UPDATE trade_log SET 手续费币种=? WHERE 订单编号=?",
                            (fee_asset, order_id))
            self.conn.commit()

    async def insert_signal_debug(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            ts = _format_ts(row.get("ts"))
            cur.execute("""
            INSERT INTO signal_debug(
                时间戳,
                symbol,
                方向,
                事件,
                原因,
                触发模式,
                实际涨跌幅,
                阈值,
                成交量信号,
                成交量阈值,
                成交量通过,
                成交量基线,
                成交量倍数,
                use_percentile,
                布林命中,
                布林上轨,
                布林下轨,
                价差bps,
                资金费率,
                open15,
                low15,
                high15,
                close15,
                last_price,
                备注
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ts,
                row.get("symbol"),
                row.get("side"),
                row.get("event"),
                row.get("reason"),
                row.get("trigger_mode"),
                row.get("move_value"),
                row.get("abs_thr"),
                row.get("vol_signal"),
                row.get("vol_threshold"),
                row.get("vol_ok"),
                row.get("baseline"),
                row.get("vol_multiple"),
                row.get("use_percentile"),
                row.get("bb_hit"),
                row.get("bb_upper"),
                row.get("bb_lower"),
                row.get("spread_bps"),
                row.get("funding_rate"),
                row.get("open15"),
                row.get("low15"),
                row.get("high15"),
                row.get("close15"),
                row.get("last_price"),
                row.get("note"),
            ))
            self.conn.commit()

    async def insert_error_log(self, **row):
        async with self.lock:
            cur = self.conn.cursor()
            ts = _format_ts(row.get("ts"))
            detail = _format_detail(row.get("details"))
            cur.execute("INSERT INTO error_log VALUES(NULL,?,?,?,?,?,?,?,?,?,?)",
                        (ts, row.get("stage"), row.get("symbol"), row.get("side"),
                         row.get("event"), row.get("reason"), row.get("error"), detail,
                         row.get("order_id"), row.get("position_id")))
            self.conn.commit()

    async def delete_position_live(self, symbol: str, side: str):
        async with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM positions_live WHERE symbol=? AND 方向=?", (symbol, side))
            self.conn.commit()

    async def delete_signal_proximity(self, symbol: str, side: str):
        async with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM signal_proximity WHERE symbol=? AND 方向=?", (symbol, side))
            self.conn.commit()

    async def upsert_meta(self, key: str, value: str) -> None:
        if not key:
            return
        async with self.lock:
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO meta(键, 值) VALUES(?,?)", (key, value))
            self.conn.commit()
