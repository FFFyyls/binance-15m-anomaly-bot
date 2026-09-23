import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from backtests.run_backtest import run_backtest, summarize


def _parse_date(name: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(name)
    except ValueError:
        return None


def _find_date_bounds(data_dir: Path, symbols: List[str], interval_s: int) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    earliest = None
    latest = None
    for sym in symbols:
        path = data_dir / sym / f"{interval_s}s"
        if not path.exists():
            continue
        dates = []
        for f in path.iterdir():
            if f.is_file() and f.name.endswith(".csv"):
                d = _parse_date(f.stem)
                if d:
                    dates.append(d)
        if not dates:
            continue
        sym_min = min(dates)
        sym_max = max(dates)
        if earliest is None or sym_min < earliest:
            earliest = sym_min
        if latest is None or sym_max > latest:
            latest = sym_max
    return earliest, latest


def _meets_drawdown(mdd: float, limit: float) -> bool:
    return mdd >= -limit


def _pick_best(rows: List[Dict], dd_limit: float) -> Optional[Dict]:
    best = None
    for r in rows:
        if best is None:
            best = r
            continue
        cand_meet = _meets_drawdown(r["max_drawdown_naive"], dd_limit)
        best_meet = _meets_drawdown(best["max_drawdown_naive"], dd_limit)
        if cand_meet and not best_meet:
            best = r
            continue
        if not cand_meet and best_meet:
            continue
        if cand_meet and best_meet:
            if r["total_pnl_usdt"] > best["total_pnl_usdt"]:
                best = r
                continue
            pnl_tol = max(1e-6, abs(best["total_pnl_usdt"]) * 0.01)
            if abs(r["total_pnl_usdt"] - best["total_pnl_usdt"]) <= pnl_tol:
                if r["trades"] > best["trades"]:
                    best = r
            continue
        if r["max_drawdown_naive"] > best["max_drawdown_naive"]:
            best = r
            continue
        if r["max_drawdown_naive"] == best["max_drawdown_naive"] and r["total_pnl_usdt"] > best["total_pnl_usdt"]:
            best = r
    return best


def _parse_multipliers(raw: str) -> List[float]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _daily_drawdown(trades_df: pd.DataFrame, initial_capital: float) -> float:
    if trades_df.empty:
        return 0.0
    df = trades_df.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["exit_time"])
    if df.empty:
        return 0.0
    daily = df.set_index("exit_time").resample("D")["pnl_usdt"].sum()
    equity = float(initial_capital or 0.0)
    worst = 0.0
    for pnl in daily:
        if equity <= 0:
            day_ret = -1.0 if pnl < 0 else 0.0
        else:
            day_ret = pnl / equity
        if day_ret < worst:
            worst = day_ret
        equity += pnl
    return float(worst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--symbols_limit", type=int, default=60)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--interval_s", type=int, default=0)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--multipliers", default="")
    ap.add_argument("--multiplier_min", type=float, default=6.0)
    ap.add_argument("--multiplier_max", type=float, default=20.0)
    ap.add_argument("--multiplier_step", type=float, default=2.0)
    ap.add_argument("--fixed_thresholds", default="")
    ap.add_argument("--fixed_threshold_min", type=float, default=0.05)
    ap.add_argument("--fixed_threshold_max", type=float, default=0.15)
    ap.add_argument("--fixed_threshold_step", type=float, default=0.02)
    ap.add_argument("--cap_thresholds", default="")
    ap.add_argument("--cap_threshold_min", type=float, default=0.08)
    ap.add_argument("--cap_threshold_max", type=float, default=0.2)
    ap.add_argument("--cap_threshold_step", type=float, default=0.02)
    ap.add_argument("--lookback_windows", type=int, default=96)
    ap.add_argument("--update_minutes", type=int, default=60)
    ap.add_argument("--vol_window_minutes", type=int, default=15)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_drawdown", type=float, default=1.0)
    ap.add_argument("--max_daily_drawdown", type=float, default=0.7)
    ap.add_argument("--weight_pnl", type=float, default=0.7)
    ap.add_argument("--weight_trades", type=float, default=0.3)
    ap.add_argument("--flush_every", type=int, default=20)
    ap.add_argument("--leverage", type=int, default=5)
    ap.add_argument("--out", default="artifacts/backtests/dynamic_threshold_sweep")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    cfg.setdefault("signal", {})
    cfg.setdefault("backtest", {})
    cfg.setdefault("position", {})

    if args.interval_s > 0:
        interval_s = args.interval_s
    else:
        interval_s = int(cfg.get("backtest", {}).get("data_interval_seconds", 60))

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]
    if args.symbols_limit > 0:
        symbols = symbols[: args.symbols_limit]

    data_dir = Path(args.data)
    earliest, latest = _find_date_bounds(data_dir, symbols, interval_s)
    if earliest is None or latest is None:
        raise SystemExit("No data found for symbols.")
    if args.days and args.days > 0:
        start = earliest
        end = start + dt.timedelta(days=args.days - 1)
    else:
        start = earliest
        end = latest

    if args.multipliers.strip():
        multipliers = _parse_multipliers(args.multipliers)
    else:
        mult = args.multiplier_min
        multipliers = []
        while mult <= args.multiplier_max + 1e-9:
            multipliers.append(round(mult, 10))
            mult += args.multiplier_step

    if args.fixed_thresholds.strip():
        fixed_thresholds = _parse_multipliers(args.fixed_thresholds)
    else:
        val = args.fixed_threshold_min
        fixed_thresholds = []
        while val <= args.fixed_threshold_max + 1e-9:
            fixed_thresholds.append(round(val, 10))
            val += args.fixed_threshold_step

    if args.cap_thresholds.strip():
        cap_thresholds = _parse_multipliers(args.cap_thresholds)
    else:
        val = args.cap_threshold_min
        cap_thresholds = []
        while val <= args.cap_threshold_max + 1e-9:
            cap_thresholds.append(round(val, 10))
            val += args.cap_threshold_step

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.csv"
    if trials_path.exists():
        trials_path.unlink()
    rows = []
    buffer = []
    flush_every = max(1, int(args.flush_every))

    for mult in multipliers:
        for cap in cap_thresholds:
            for fixed_thr in fixed_thresholds:
                cfg_run = copy.deepcopy(cfg)
                cfg_run.setdefault("position", {})
                cfg_run.setdefault("backtest", {})
                cfg_run.setdefault("signal", {})
                cfg_run["position"]["leverage"] = int(args.leverage)
                cfg_run["backtest"]["leverage"] = int(args.leverage)
                cfg_run["signal"]["abs_r15_threshold"] = float(fixed_thr)
                cfg_run["signal"]["dynamic_abs_threshold"] = {
                    "enabled": True,
                    "multiplier": float(mult),
                    "lookback_windows": int(args.lookback_windows),
                    "update_minutes": int(args.update_minutes),
                    "vol_window_minutes": int(args.vol_window_minutes),
                    "max_threshold": float(cap),
                }
                accepted_df, all_df, _ = run_backtest(
                    cfg_run,
                    symbols,
                    start.isoformat(),
                    end.isoformat(),
                    args.data,
                    args.workers,
                )
                init_cap = 0.0
                if cfg_run.get("backtest", {}).get("use_portfolio_sizing", False):
                    init_cap = float(cfg_run.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
                summary = summarize(accepted_df, initial_capital=init_cap)
                daily_dd = _daily_drawdown(accepted_df, init_cap)
                summary.update({
                    "multiplier": float(mult),
                    "cap_threshold": float(cap),
                    "fixed_threshold": float(fixed_thr),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "trades": int(summary.get("trades", 0)),
                    "max_daily_drawdown": float(daily_dd),
                })
                rows.append(summary)
                buffer.append(summary)
                if len(buffer) >= flush_every:
                    df_out = pd.DataFrame(buffer)
                    df_out.to_csv(trials_path, index=False, mode="a", header=not trials_path.exists())
                    buffer = []

    if buffer:
        df_out = pd.DataFrame(buffer)
        df_out.to_csv(trials_path, index=False, mode="a", header=not trials_path.exists())
        buffer = []

    eligible = [
        r for r in rows
        if _meets_drawdown(r["max_drawdown_naive"], args.max_drawdown)
        and r.get("max_daily_drawdown", 0.0) >= -args.max_daily_drawdown
    ]

    if eligible:
        pnl_vals = [r["total_pnl_usdt"] for r in eligible]
        tr_vals = [r["trades"] for r in eligible]
        pnl_min, pnl_max = min(pnl_vals), max(pnl_vals)
        tr_min, tr_max = min(tr_vals), max(tr_vals)
        for r in eligible:
            pnl_norm = 0.0 if pnl_max == pnl_min else (r["total_pnl_usdt"] - pnl_min) / (pnl_max - pnl_min)
            tr_norm = 0.0 if tr_max == tr_min else (r["trades"] - tr_min) / (tr_max - tr_min)
            r["weighted_score"] = args.weight_pnl * pnl_norm + args.weight_trades * tr_norm

        best_weighted = max(eligible, key=lambda r: r.get("weighted_score", float("-inf")))
    else:
        best_weighted = None

    best = _pick_best(rows, args.max_drawdown)
    best_cfg = None
    if best_weighted is not None:
        best_cfg = copy.deepcopy(cfg)
        best_cfg["signal"]["abs_r15_threshold"] = float(best_weighted["fixed_threshold"])
        best_cfg["signal"]["dynamic_abs_threshold"] = {
            "enabled": True,
            "multiplier": float(best_weighted["multiplier"]),
            "lookback_windows": int(args.lookback_windows),
            "update_minutes": int(args.update_minutes),
            "vol_window_minutes": int(args.vol_window_minutes),
            "max_threshold": float(best_weighted["cap_threshold"]),
        }
        with open(out_dir / "best_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

        with open(out_dir / "best_summary.json", "w", encoding="utf-8") as f:
            json.dump(best_weighted, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "start": start.isoformat(),
        "end": end.isoformat(),
        "multipliers": multipliers,
        "best": best_weighted,
        "eligible_count": len(eligible),
        "out": str(out_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
