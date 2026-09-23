import argparse
import csv
import datetime as dt
import io
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import random

import requests

BASE_URL = "https://data.binance.vision"


def _date_range(start: dt.date, end: dt.date) -> List[str]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += dt.timedelta(days=1)
    return days


def _download_zip(url: str, out_path: Path, retries: int = 3, timeout: int = 30) -> str:
    for i in range(retries):
        try:
            resp = requests.get(url, stream=True, timeout=timeout, proxies={"http": None, "https": None})
        except requests.RequestException:
            resp = None
        if resp is None:
            if i < retries - 1:
                time.sleep(2 + i)
                continue
            return "error"
        if resp.status_code == 404:
            return "not_found"
        if resp.status_code != 200:
            if i < retries - 1:
                time.sleep(2 + i)
                continue
            return f"http_{resp.status_code}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return "ok"
    return "error"


def _aggregate_zip_to_20s(zip_path: Path, out_csv: Path, interval_s: int) -> int:
    interval_ms = interval_s * 1000
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not names:
            return 0
        name = names[0]
        with zf.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)
            with open(out_csv, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.writer(out_f)
                writer.writerow(["open_time", "open", "high", "low", "close", "volume", "quote_volume"])

                cur_bucket = None
                open_p = high_p = low_p = close_p = 0.0
                volume = quote_volume = 0.0

                for row in reader:
                    if not row:
                        continue
                    try:
                        ts = int(row[5])
                        price = float(row[1])
                        qty = float(row[2])
                    except (ValueError, IndexError):
                        continue

                    bucket = ts - (ts % interval_ms)
                    if cur_bucket is None:
                        cur_bucket = bucket
                        open_p = high_p = low_p = close_p = price
                        volume = qty
                        quote_volume = price * qty
                        continue

                    if bucket != cur_bucket:
                        writer.writerow([
                            cur_bucket,
                            f"{open_p:.8f}",
                            f"{high_p:.8f}",
                            f"{low_p:.8f}",
                            f"{close_p:.8f}",
                            f"{volume:.8f}",
                            f"{quote_volume:.8f}",
                        ])
                        rows += 1
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

                if cur_bucket is not None:
                    writer.writerow([
                        cur_bucket,
                        f"{open_p:.8f}",
                        f"{high_p:.8f}",
                        f"{low_p:.8f}",
                        f"{close_p:.8f}",
                        f"{volume:.8f}",
                        f"{quote_volume:.8f}",
                    ])
                    rows += 1
    return rows


def _download_symbol(
    symbol: str,
    days: List[str],
    data_dir: Path,
    interval_s: int,
    keep_zip: bool,
) -> Tuple[str, int, int]:
    ok_days = 0
    skipped = 0
    for day in days:
        out_csv = data_dir / symbol / f"{interval_s}s" / f"{day}.csv"
        if out_csv.exists() and out_csv.stat().st_size > 0:
            skipped += 1
            continue
        url = (
            f"{BASE_URL}/data/futures/um/daily/aggTrades/{symbol}/"
            f"{symbol}-aggTrades-{day}.zip"
        )
        zip_path = data_dir / symbol / "_tmp" / f"{symbol}-aggTrades-{day}.zip"
        status = _download_zip(url, zip_path)
        if status == "not_found":
            continue
        if status != "ok":
            continue
        try:
            rows = _aggregate_zip_to_20s(zip_path, out_csv, interval_s)
        finally:
            if not keep_zip:
                try:
                    zip_path.unlink(missing_ok=True)
                except TypeError:
                    if zip_path.exists():
                        zip_path.unlink()
        if rows <= 0:
            if out_csv.exists():
                out_csv.unlink()
            continue
        ok_days += 1
    tmp_dir = data_dir / symbol / "_tmp"
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    return symbol, ok_days, skipped


def _load_existing_symbols(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip() and not line.startswith("#")}


def _fetch_new_symbols(min_quote_usdt: float, exclude: set, limit: int, sample_mode: str, seed: int) -> List[str]:
    ex = requests.get(
        "https://fapi.binance.com/fapi/v1/exchangeInfo",
        timeout=30,
        proxies={"http": None, "https": None},
    ).json()
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

    t24 = requests.get(
        "https://fapi.binance.com/fapi/v1/ticker/24hr",
        timeout=30,
        proxies={"http": None, "https": None},
    ).json()
    qv = {x["symbol"]: float(x.get("quoteVolume", 0) or 0) for x in t24 if "symbol" in x}

    universe = [s for s in symbols if qv.get(s, 0.0) >= min_quote_usdt]
    if not universe:
        return []
    if sample_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(universe)
        return universe[:limit]
    universe = sorted(universe, key=lambda x: qv.get(x, 0.0), reverse=True)
    return universe[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_quote_usdt", type=float, default=5e5)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--exclude_symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--out_symbols_file", default="configs/symbols/symbols_new30_market.txt")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--data_dir", default="data/raw/after_2026_02_01")
    ap.add_argument("--interval_s", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep_zip", action="store_true")
    ap.add_argument("--sample_mode", choices=["top", "random"], default="top")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today() - dt.timedelta(days=1)
    start = dt.date.fromisoformat(args.start) if args.start else end - dt.timedelta(days=89)
    days = _date_range(start, end)

    exclude = _load_existing_symbols(Path(args.exclude_symbols_file))
    new_symbols = _fetch_new_symbols(args.min_quote_usdt, exclude, args.limit, args.sample_mode, args.seed)
    if not new_symbols:
        raise SystemExit("No symbols found for the given filters.")

    with open(args.out_symbols_file, "w", encoding="utf-8") as f:
        for s in new_symbols:
            f.write(s + "\n")

    data_dir = Path(args.data_dir)
    print(f"Selected {len(new_symbols)} symbols -> {args.out_symbols_file}")
    print(f"Date range: {start}..{end} ({len(days)} days)")

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(_download_symbol, sym, days, data_dir, args.interval_s, args.keep_zip)
                for sym in new_symbols
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for sym in new_symbols:
            results.append(_download_symbol(sym, days, data_dir, args.interval_s, args.keep_zip))

    for sym, ok_days, skipped in sorted(results):
        print(f"{sym}: ok_days={ok_days} skipped_existing={skipped}")


if __name__ == "__main__":
    main()
