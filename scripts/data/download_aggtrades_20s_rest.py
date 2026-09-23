import argparse
import datetime as dt
import os
import random
import time
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/aggTrades"


def _make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _date_range(start: dt.date, end: dt.date) -> List[str]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += dt.timedelta(days=1)
    return days


def _load_symbols(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip() and not line.startswith("#")}


def _fetch_symbols(min_quote_usdt: float, exclude: Set[str], limit: int, seed: int) -> List[str]:
    session = _make_session()
    try:
        ex = session.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
        symbols = []
        for s in ex.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            sym = s.get("symbol")
            if not sym or sym in exclude:
                continue
            symbols.append(sym)

        t24 = session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=30).json()
        qv = {x["symbol"]: float(x.get("quoteVolume", 0) or 0.0) for x in t24 if "symbol" in x}
        universe = [s for s in symbols if qv.get(s, 0.0) >= min_quote_usdt]
        if not universe:
            return []
        rng = random.Random(seed)
        rng.shuffle(universe)
        return universe[:limit]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []


def _existing_days(symbol: str, data_dir: Path, interval_s: int) -> Set[str]:
    sym_dir = data_dir / symbol / f"{interval_s}s"
    if not sym_dir.exists():
        return set()
    days = set()
    for p in sym_dir.glob("*.csv"):
        if p.stat().st_size > 0:
            days.add(p.stem)
    return days


def _fetch_batch(session: requests.Session, params: dict, retries: int = 5) -> List[dict]:
    for i in range(retries):
        try:
            resp = session.get(BASE_URL, params=params, timeout=15)
        except requests.RequestException:
            resp = None
        if resp is None:
            time.sleep(1 + i)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (418, 429):
            time.sleep(5 + i * 5)  # Back off more on rate limits
            continue
        time.sleep(1 + i)
    return []


def _aggregate_day(
    session: requests.Session,
    symbol: str,
    day: str,
    data_dir: Path,
    interval_s: int,
    sleep_s: float,
) -> Tuple[str, str, int, str]:
    out_csv = data_dir / symbol / f"{interval_s}s" / f"{day}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return symbol, day, 1, "" # Assume 1 bar to indicate success/skip

    try:
        start_dt = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
        end_dt = start_dt + dt.timedelta(days=1) - dt.timedelta(milliseconds=1)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        bars = 0
        interval_ms = interval_s * 1000

        cur_bucket = None
        open_p = high_p = low_p = close_p = 0.0
        volume = quote_volume = 0.0

        params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}

        start_time_perf = time.time()

        while True:
            try:
                data = _fetch_batch(session, params)
                if not data:
                    break

                for t in data:
                    ts = int(t.get("T", 0))
                    if ts < start_ms or ts > end_ms:
                        continue
                    price = float(t.get("p", 0.0))
                    qty = float(t.get("q", 0.0))
                    if price <= 0 or qty <= 0:
                        continue
                    bucket = ts - (ts % interval_ms)
                    if cur_bucket is None:
                        cur_bucket = bucket
                        open_p = high_p = low_p = close_p = price
                        volume = qty
                        quote_volume = price * qty
                        continue
                    if bucket != cur_bucket:
                        with out_csv.open("a", encoding="utf-8") as f:
                            if bars == 0:
                                f.write("open_time,open,high,low,close,volume,quote_volume\n")
                            # Reduced precision for storage optimization
                            f.write(
                                f"{cur_bucket},{open_p:.8f},{high_p:.6f},{low_p:.6f},{close_p:.6f},{volume:.4f},{quote_volume:.4f}\n"
                            )
                        bars += 1
                        cur_bucket = bucket
                        open_p = high_p = low_p = close_p = price
                        volume = qty
                        quote_volume = price * qty
                        continue
                    high_p = max(high_p, price)
                    low_p = min(low_p, price)
                    close_p = price
                    volume += qty
                    quote_volume += price * qty

                last = data[-1]
                last_id = int(last.get("a", 0))
                last_ts = int(last.get("T", 0))

                if last_ts >= end_ms:
                    break

                # Ensure we make progress if the last data point is still within the same millisecond
                next_start = max(last_ts + 1, start_ms)
                params = {"symbol": symbol, "startTime": next_start, "endTime": end_ms, "limit": 1000}

                if sleep_s > 0:
                    time.sleep(sleep_s)
            except Exception as e:
                # Local retry logic
                retry_ts = cur_bucket + interval_ms if cur_bucket else start_ms
                params = {"symbol": symbol, "startTime": retry_ts, "endTime": end_ms, "limit": 1000}
                time.sleep(1)
                continue

        if cur_bucket is not None:
            with out_csv.open("a", encoding="utf-8") as f:
                if bars == 0:
                    f.write("open_time,open,high,low,close,volume,quote_volume\n")
                f.write(
                    f"{cur_bucket},{open_p:.8f},{high_p:.6f},{low_p:.6f},{close_p:.6f},{volume:.4f},{quote_volume:.4f}\n"
                )
            bars += 1

        if bars == 0:
            if out_csv.exists():
                out_csv.unlink()
            return symbol, day, 0, "No data found for this day"

        return symbol, day, bars, ""
    except Exception as e:
        error_report = f"Error in _aggregate_day for {symbol} {day}:\n{traceback.format_exc()}"
        return symbol, day, 0, error_report


def _download_symbol_days(
    symbol: str,
    days: List[str],
    data_dir: Path,
    interval_s: int,
    sleep_s: float,
) -> Tuple[str, int, int, List[str]]:
    session = _make_session()
    ok_days = 0
    errors = []
    for day in days:
        _, _, bars, err = _aggregate_day(session, symbol, day, data_dir, interval_s, sleep_s)
        if bars > 0:
            ok_days += 1
        if err:
            errors.append(f"Day {day}: {err}")
    return symbol, ok_days, len(days), errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default="", help="Download a single symbol")
    ap.add_argument("--min_quote_usdt", type=float, default=5e5)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--exclude_symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--out_symbols_file", default="configs/symbols/symbols_new50_market.txt")
    ap.add_argument("--symbols_file", default="")
    ap.add_argument("--existing_symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--start", default="", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default="", help="End date YYYY-MM-DD")
    ap.add_argument("--data_dir", default="data/raw/before_2026_02_01")
    ap.add_argument("--interval_s", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sleep_s", type=float, default=0.05)
    ap.add_argument("--update_existing", action="store_true")
    args = ap.parse_args()

    # Date Range logic
    if args.end:
        end_date = dt.date.fromisoformat(args.end)
    else:
        end_date = dt.datetime.utcnow().date() - dt.timedelta(days=1)

    if args.start:
        start_date = dt.date.fromisoformat(args.start)
    elif args.months > 0:
        start_date = end_date - dt.timedelta(days=args.months * 30 - 1)
    else:
        start_date = end_date

    days = _date_range(start_date, end_date)

    data_dir = Path(args.data_dir)

    # Symbols logic
    if args.symbol:
        new_symbols = [args.symbol.upper()]
        print(f"Using single symbol: {new_symbols}")
    elif args.symbols_file:
        new_symbols = sorted(_load_symbols(Path(args.symbols_file)))
        if not new_symbols:
            raise SystemExit("symbols_file is empty.")
        print(f"Loaded {len(new_symbols)} symbols from file")
    else:
        exclude = _load_symbols(Path(args.exclude_symbols_file))
        new_symbols = _fetch_symbols(args.min_quote_usdt, exclude, args.limit, args.seed)
        if not new_symbols:
            raise SystemExit("No symbols found for the given filters.")
        out_symbols = Path(args.out_symbols_file)
        out_symbols.parent.mkdir(parents=True, exist_ok=True)
        out_symbols.write_text("\n".join(new_symbols) + "\n", encoding="utf-8")
        print(f"Selected {len(new_symbols)} symbols -> {out_symbols}")

    print(f"Date range: {start_date}..{end_date} ({len(days)} days)")

    tasks = []
    for sym in new_symbols:
        tasks.append((sym, days))

    if args.update_existing:
        existing = _load_symbols(Path(args.existing_symbols_file))
        for sym in sorted(existing):
            if args.symbol and sym != args.symbol.upper():
                continue
            have = _existing_days(sym, data_dir, args.interval_s)
            missing = [d for d in days if d not in have]
            if missing:
                tasks.append((sym, missing))

    total_tasks = len(tasks)
    total_days = sum(len(d) for _, d in tasks)
    print(f"Total tasks: {total_tasks} total_days: {total_days}")

    results = []
    completed = 0
    start_perf = time.time()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(_download_symbol_days, sym, dlist, data_dir, args.interval_s, args.sleep_s): (sym, len(dlist))
                for sym, dlist in tasks
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                completed += 1
                sym, ok_days, req_days, errs = res
                print(f"done {completed}/{total_tasks} {sym} ok_days={ok_days} requested_days={req_days}", flush=True)
                if errs:
                    for e in errs:
                        print(f"  ERROR: {e}")
    else:
        for sym, dlist in tasks:
            res = _download_symbol_days(sym, dlist, data_dir, args.interval_s, args.sleep_s)
            results.append(res)
            completed += 1
            sym, ok_days, req_days, errs = res
            print(f"done {completed}/{total_tasks} {sym} ok_days={ok_days} requested_days={req_days}", flush=True)
            if errs:
                for e in errs:
                    print(f"  ERROR: {e}")

    duration = time.time() - start_perf
    if total_days > 0:
        print(f"\nFinal Summary (Total Time: {duration:.2f}s, Avg: {duration/total_days:.2f}s/day)")

    for sym, ok_days, req_days, _ in sorted(results):
        print(f"{sym}: ok_days={ok_days} requested_days={req_days}")


if __name__ == "__main__":
    main()
