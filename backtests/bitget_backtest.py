import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import yaml

from binance_anomaly_bot.binance_client import BinanceFuturesClient
from binance_anomaly_bot.universe import UniverseSelector

from backtests import run_backtest as rb

_UTC = timezone.utc


def _get_json(url: str, retries: int = 3, sleep_s: float = 1.0) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload
        except HTTPError as e:
            last_err = f"HTTP {e.code} {e.read().decode('utf-8')}"
        except Exception as e:
            last_err = str(e)
        if attempt < retries - 1:
            time.sleep(sleep_s * (2 ** attempt))
    raise RuntimeError(f"request failed: {url} err={last_err}")


def _bitget_get(path: str, params: dict) -> dict:
    base = "https://api.bitget.com"
    url = f"{base}{path}?{urlencode(params)}"
    payload = _get_json(url)
    if payload.get("code") != "00000":
        raise RuntimeError(f"bitget api error path={path} msg={payload.get('msg')}")
    return payload


def fetch_tickers(product_type: str) -> List[dict]:
    payload = _bitget_get("/api/v2/mix/market/tickers", {"productType": product_type})
    return payload.get("data") or []


def fetch_contracts(product_type: str) -> Dict[str, dict]:
    payload = _bitget_get("/api/v2/mix/market/contracts", {"productType": product_type})
    data = payload.get("data") or []
    return {row.get("symbol"): row for row in data if row.get("symbol")}


def fetch_history_candles(symbol: str, product_type: str, start_ms: int, end_ms: int) -> List[List[str]]:
    params = {
        "symbol": symbol,
        "productType": product_type,
        "granularity": "1m",
        "startTime": str(start_ms),
        "endTime": str(end_ms),
    }
    payload = _bitget_get("/api/v2/mix/market/history-candles", params)
    return payload.get("data") or []


def expand_1m_to_20s(candles: List[List[str]]) -> List[Tuple[int, float, float, float, float, float, float]]:
    rows = []
    for raw in candles:
        if len(raw) < 7:
            continue
        try:
            ts = int(float(raw[0]))
            o = float(raw[1])
            h = float(raw[2])
            l = float(raw[3])
            c = float(raw[4])
            v = float(raw[5])
            q = float(raw[6])
        except (TypeError, ValueError):
            continue
        step = (c - o) / 3.0
        o1, c1 = o, o + step
        o2, c2 = c1, o + 2 * step
        o3, c3 = c2, c
        v3 = v / 3.0
        q3 = q / 3.0
        highs = [max(o1, c1), max(o2, c2), max(o3, c3)]
        lows = [min(o1, c1), min(o2, c2), min(o3, c3)]
        highs[1] = max(highs[1], h)
        lows[1] = min(lows[1], l)
        rows.extend([
            (ts, o1, highs[0], lows[0], c1, v3, q3),
            (ts + 20000, o2, highs[1], lows[1], c2, v3, q3),
            (ts + 40000, o3, highs[2], lows[2], c3, v3, q3),
        ])
    return rows


def raw_1m_rows(candles: List[List[str]]) -> List[Tuple[int, float, float, float, float, float, float]]:
    rows = []
    for raw in candles:
        if len(raw) < 7:
            continue
        try:
            ts = int(float(raw[0]))
            o = float(raw[1])
            h = float(raw[2])
            l = float(raw[3])
            c = float(raw[4])
            v = float(raw[5])
            q = float(raw[6])
        except (TypeError, ValueError):
            continue
        rows.append((ts, o, h, l, c, v, q))
    return rows


def _date_windows(days: int) -> List[Tuple[datetime, datetime]]:
    end_dt = datetime.now(tz=_UTC)
    start_dt = end_dt - timedelta(days=days)
    windows = []
    cur = datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=_UTC)
    while cur <= end_dt:
        nxt = cur + timedelta(days=1)
        if nxt > end_dt:
            windows.append((cur, end_dt))
            break
        windows.append((cur, nxt))
        cur = nxt
    return windows


def _existing_symbols(existing_dir: str, existing_file: Optional[str], extra_files: Optional[List[str]] = None) -> set:
    symbols = set()
    if os.path.isdir(existing_dir):
        symbols.update({name for name in os.listdir(existing_dir) if name})
    if existing_file and os.path.exists(existing_file):
        with open(existing_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    symbols.add(s)
    for path in extra_files or []:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    symbols.add(s)
    return symbols


def select_symbols(tickers: List[dict], existing: set, min_usdt_volume: float, count: int) -> List[str]:
    filtered = []
    for row in tickers:
        symbol = row.get("symbol")
        if not symbol or symbol in existing:
            continue
        try:
            vol = float(row.get("usdtVolume", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if vol >= min_usdt_volume:
            filtered.append((symbol, vol))
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in filtered[:count]]


async def _fetch_binance_universe(config_path: str) -> set:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    class _Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    client = BinanceFuturesClient("", "", False, _Logger())
    try:
        selector = UniverseSelector(client, cfg, _Logger())
        universe = await selector.build_universe()
        return set(universe.keys())
    finally:
        await client.close()


def _interval_to_seconds(interval: str) -> int:
    key = str(interval).lower().strip()
    if key in ("1m", "60s", "60"):
        return 60
    if key in ("20s", "20"):
        return 20
    raise ValueError("interval must be 1m or 20s")


def _interval_subdir(interval_s: int) -> str:
    return "1m" if interval_s == 60 else "20s"


def write_daily_csv(out_dir: str, day: datetime, rows: List[Tuple[int, float, float, float, float, float, float]]) -> None:
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    day_str = day.strftime("%Y-%m-%d")
    path = os.path.join(out_dir, f"{day_str}.csv")
    rows.sort(key=lambda r: r[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("open_time,open,high,low,close,volume,quote_volume\n")
        for row in rows:
            f.write("{},{},{},{},{},{},{}\n".format(*row))


def download_and_aggregate(symbols: List[str], product_type: str, out_data_dir: str, days: int,
                           interval_s: int, sleep_s: float) -> None:
    windows = _date_windows(days)
    for symbol in symbols:
        out_dir = os.path.join(out_data_dir, symbol, _interval_subdir(interval_s))
        for start_dt, end_dt in windows:
            day_str = start_dt.strftime("%Y-%m-%d")
            path = os.path.join(out_dir, f"{day_str}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            candles = fetch_history_candles(symbol, product_type, start_ms, end_ms)
            if not candles:
                continue
            if interval_s == 20:
                rows = expand_1m_to_20s(candles)
            else:
                rows = raw_1m_rows(candles)
            write_daily_csv(out_dir, start_dt, rows)
            if sleep_s > 0:
                time.sleep(sleep_s)


def run_backtest_on_data(symbols: List[str], config_path: str, data_dir: str, out_dir: str,
                         use_long_strategy: bool, workers: int, interval_s: int) -> dict:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    cfg.setdefault("backtest", {"maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 5.0})
    cfg["backtest"]["data_interval_seconds"] = int(interval_s)
    long_cfg = None
    if use_long_strategy:
        long_cfg = rb._load_long_strategy_config(cfg)
    if long_cfg:
        accepted_df, all_df, portfolio_stats = rb.run_backtest_dual(
            cfg,
            long_cfg,
            symbols,
            None,
            None,
            data_dir,
            workers,
            use_symbol_range=True,
        )
    else:
        accepted_df, all_df, portfolio_stats = rb.run_backtest(
            cfg,
            symbols,
            None,
            None,
            data_dir,
            workers,
            use_symbol_range=True,
        )
    if accepted_df.empty:
        return {"trades": 0}
    os.makedirs(out_dir, exist_ok=True)
    accepted_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    if portfolio_stats and all_df is not None:
        all_df.to_csv(os.path.join(out_dir, "trades_all.csv"), index=False)
    summary = rb.summarize(accepted_df, portfolio_stats.get("initial_capital_usdt", 0.0) if portfolio_stats else 0.0)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30, help="number of symbols to download")
    ap.add_argument("--min_usdt_volume", type=float, default=100000.0)
    ap.add_argument("--days", type=int, default=2, help="lookback days for candles")
    ap.add_argument("--interval", default="1m", help="1m or 20s")
    ap.add_argument("--product_type", default="USDT-FUTURES")
    ap.add_argument("--existing_dir", default="data/raw/before_2026_02_01")
    ap.add_argument("--existing_file", default="configs/symbols/symbols_db_available.txt")
    ap.add_argument("--exclude_files", default="", help="comma-separated symbol lists to exclude")
    ap.add_argument("--exclude_universe", action="store_true", help="exclude Binance universe symbols")
    ap.add_argument("--data_out", default="data/raw/bitget")
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--results_out", default="artifacts/backtests/bitget_backtest")
    ap.add_argument("--use_long_strategy", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sleep_s", type=float, default=0.05, help="sleep between API calls")
    args = ap.parse_args()

    interval_s = _interval_to_seconds(args.interval)
    extra_files = [p.strip() for p in args.exclude_files.split(",") if p.strip()]
    existing = _existing_symbols(args.existing_dir, args.existing_file, extra_files)
    if args.exclude_universe:
        universe_syms = asyncio.run(_fetch_binance_universe(args.config))
        existing |= universe_syms
    tickers = fetch_tickers(args.product_type)
    symbols = select_symbols(tickers, existing, args.min_usdt_volume, args.count)
    if len(symbols) < args.count:
        raise SystemExit(f"only found {len(symbols)} symbols after filtering")

    contracts = fetch_contracts(args.product_type)
    os.makedirs(args.data_out, exist_ok=True)
    with open(os.path.join(args.data_out, "symbols_selected.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(symbols))
    info = {s: contracts.get(s, {}) for s in symbols}
    with open(os.path.join(args.data_out, "symbols_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=True, indent=2)

    download_and_aggregate(symbols, args.product_type, args.data_out, args.days, interval_s, args.sleep_s)
    summary = run_backtest_on_data(
        symbols,
        args.config,
        args.data_out,
        args.results_out,
        args.use_long_strategy,
        args.workers,
        interval_s,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
