import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from backtests.run_backtest import run_backtest_dual, summarize


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


def _parse_list(val: str, cast) -> List[Any]:
    if not val:
        return []
    out = []
    for item in val.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(cast(item))
    return out


def _load_long_config(short_cfg: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    if path:
        return yaml.safe_load(open(path, "r", encoding="utf-8"))
    raw = short_cfg.get("long_strategy")
    if not raw:
        return None
    if isinstance(raw, dict) and raw.get("enabled") is False:
        return None
    if isinstance(raw, str):
        cfg_path = raw
    elif isinstance(raw, dict):
        cfg_path = raw.get("config_path", "")
    else:
        return None
    if not cfg_path:
        return None
    return yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))


def _with_bollinger(cfg: Dict[str, Any], window_minutes: int, num_std: float, mode: str) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["bollinger"] = {
        "enabled": True,
        "window_minutes": int(window_minutes),
        "num_std": float(num_std),
        "mode": str(mode),
    }
    return out


def _is_better(cand: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    if best is None:
        return True
    if cand.get("meets_constraints") and not best.get("meets_constraints"):
        return True
    if not cand.get("meets_constraints") and best.get("meets_constraints"):
        return False
    cand_pnl = float(cand.get("total_pnl_usdt", 0.0))
    best_pnl = float(best.get("total_pnl_usdt", 0.0))
    if cand_pnl > best_pnl:
        return True
    if abs(cand_pnl - best_pnl) <= max(1e-6, abs(best_pnl) * 0.01):
        cand_wr = float(cand.get("win_rate", 0.0))
        best_wr = float(best.get("win_rate", 0.0))
        if cand_wr > best_wr:
            return True
        if abs(cand_wr - best_wr) <= 1e-6:
            cand_tr = int(cand.get("trades", 0) or 0)
            best_tr = int(best.get("trades", 0) or 0)
            if cand_tr > best_tr:
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--short_config", required=True)
    ap.add_argument("--long_config", default="")
    ap.add_argument("--windows", default="20,30,60")
    ap.add_argument("--stds", default="2.0,2.5,3.0")
    ap.add_argument("--mode", default="filter")
    ap.add_argument("--baseline_window", type=int, default=20)
    ap.add_argument("--baseline_std", type=float, default=2.5)
    ap.add_argument("--min_trade_ratio", type=float, default=1.05)
    ap.add_argument("--max_daily_drawdown", type=float, default=0.45)
    ap.add_argument("--max_drawdown", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="artifacts/backtests/bollinger_filter_grid")
    args = ap.parse_args()

    short_base = yaml.safe_load(open(args.short_config, "r", encoding="utf-8"))
    long_base = _load_long_config(short_base, args.long_config)
    if long_base is None:
        raise SystemExit("ERROR: missing long config (provide --long_config or set long_strategy in short config)")

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]

    windows = _parse_list(args.windows, int)
    stds = _parse_list(args.stds, float)
    if not windows or not stds:
        raise SystemExit("ERROR: windows/stds are required")

    os.makedirs(args.out, exist_ok=True)
    trials_path = os.path.join(args.out, "trials.csv")
    if os.path.exists(trials_path):
        os.remove(trials_path)

    base_short = _with_bollinger(short_base, args.baseline_window, args.baseline_std, args.mode)
    base_long = _with_bollinger(long_base, args.baseline_window, args.baseline_std, args.mode)
    base_trades_df, _, _ = run_backtest_dual(base_short, base_long, symbols, args.start, args.end, args.data, args.workers)
    init_cap = 0.0
    if short_base.get("backtest", {}).get("use_portfolio_sizing", False):
        init_cap = float(short_base.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
    base_summary = summarize(base_trades_df, initial_capital=init_cap)
    base_summary["max_daily_drawdown"] = _daily_drawdown(base_trades_df, init_cap)
    base_trades = int(base_summary.get("trades", 0) or 0)

    rows: List[Dict[str, Any]] = []
    best = None
    best_short = None
    best_long = None

    for w in windows:
        for s in stds:
            short_cfg = _with_bollinger(short_base, w, s, args.mode)
            long_cfg = _with_bollinger(long_base, w, s, args.mode)
            accepted_df, _, _ = run_backtest_dual(
                short_cfg, long_cfg, symbols, args.start, args.end, args.data, args.workers
            )
            summary = summarize(accepted_df, initial_capital=init_cap)
            summary["window_minutes"] = int(w)
            summary["num_std"] = float(s)
            summary["max_daily_drawdown"] = _daily_drawdown(accepted_df, init_cap)
            trade_ratio = 0.0
            if base_trades > 0:
                trade_ratio = float(summary.get("trades", 0) or 0) / base_trades
            summary["trade_ratio"] = trade_ratio
            summary["meets_constraints"] = (
                trade_ratio >= args.min_trade_ratio
                and summary.get("max_daily_drawdown", 0.0) >= -args.max_daily_drawdown
                and summary.get("max_drawdown_naive", 0.0) >= -args.max_drawdown
            )
            rows.append(summary)
            if _is_better(summary, best):
                best = summary
                best_short = short_cfg
                best_long = long_cfg

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(trials_path, index=False)

    if best_short is None or best_long is None:
        print("No valid trials executed.")
        return

    best_dir = os.path.join(args.out, "best")
    os.makedirs(best_dir, exist_ok=True)
    best_trades, best_all, portfolio_stats = run_backtest_dual(
        best_short, best_long, symbols, args.start, args.end, args.data, args.workers
    )
    best_trades.to_csv(os.path.join(best_dir, "trades.csv"), index=False)
    if not best_all.empty:
        best_all.to_csv(os.path.join(best_dir, "trades_all.csv"), index=False)

    best_summary = summarize(best_trades, initial_capital=init_cap)
    best_summary["max_daily_drawdown"] = _daily_drawdown(best_trades, init_cap)
    best_summary.update(
        {
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "baseline": base_summary,
            "constraints": {
                "min_trade_ratio": args.min_trade_ratio,
                "max_daily_drawdown": args.max_daily_drawdown,
                "max_drawdown": args.max_drawdown,
            },
            "bollinger": {
                "mode": args.mode,
                "window_minutes": int(best.get("window_minutes")),
                "num_std": float(best.get("num_std")),
            },
            "assumptions": {
                "data": f"{int(short_base.get('backtest', {}).get('data_interval_seconds', 60))}s OHLCV",
                "fill_model": "limit filled if next bar touches (low/high)",
            },
            "backtest_params": short_base.get("backtest", {}),
            "portfolio": portfolio_stats,
        }
    )

    with open(os.path.join(best_dir, "best_config_short.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_short, f, sort_keys=False)
    with open(os.path.join(best_dir, "best_config_long.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_long, f, sort_keys=False)
    with open(os.path.join(best_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(best_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
