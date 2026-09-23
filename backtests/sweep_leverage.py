import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from backtests.run_backtest import run_backtest_dual, summarize, _load_long_strategy_config


def _meets_drawdown(summary: Dict[str, Any], limit: float) -> bool:
    return float(summary.get("max_drawdown_naive", 0.0)) >= -limit


def _is_better(cand: Dict[str, Any], best: Optional[Dict[str, Any]], dd_limit: float) -> bool:
    if best is None:
        return True
    cand_meet = _meets_drawdown(cand, dd_limit)
    best_meet = _meets_drawdown(best, dd_limit)
    cand_pnl = float(cand.get("total_pnl_usdt", 0.0))
    best_pnl = float(best.get("total_pnl_usdt", 0.0))
    cand_trades = int(cand.get("trades", 0) or 0)
    best_trades = int(best.get("trades", 0) or 0)
    cand_wr = float(cand.get("win_rate", 0.0))
    best_wr = float(best.get("win_rate", 0.0))
    cand_dd = float(cand.get("max_drawdown_naive", 0.0))
    best_dd = float(best.get("max_drawdown_naive", 0.0))

    if cand_meet and not best_meet:
        return True
    if not cand_meet and best_meet:
        return False
    if cand_meet and best_meet:
        if cand_pnl > best_pnl:
            return True
        pnl_tol = max(1e-6, abs(best_pnl) * 0.01)
        if abs(cand_pnl - best_pnl) <= pnl_tol:
            if cand_trades > best_trades:
                return True
            if cand_trades == best_trades and cand_wr > best_wr:
                return True
        return False

    if cand_dd > best_dd:
        return True
    if abs(cand_dd - best_dd) <= 1e-6 and cand_pnl > best_pnl:
        return True
    if abs(cand_dd - best_dd) <= 1e-6 and abs(cand_pnl - best_pnl) <= 1e-6 and cand_wr > best_wr:
        return True
    return False


def _apply_leverage(cfg: dict, leverage: int) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("position", {})["leverage"] = int(leverage)
    out.setdefault("backtest", {})["leverage"] = int(leverage)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--long_config", default="")
    ap.add_argument("--min_lev", type=int, default=1)
    ap.add_argument("--max_lev", type=int, default=12)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max_drawdown", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="artifacts/backtests/leverage_sweep")
    args = ap.parse_args()

    short_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    long_cfg = _load_long_strategy_config(short_cfg, args.long_config) if args.long_config else _load_long_strategy_config(short_cfg)
    if not long_cfg:
        raise SystemExit(
            "long_strategy config not found; provide --long_config or set it in the strategy config"
        )

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]

    os.makedirs(args.out, exist_ok=True)
    trials_path = os.path.join(args.out, "trials.csv")
    if os.path.exists(trials_path):
        os.remove(trials_path)

    best = None
    best_cfg = None
    rows: List[Dict[str, Any]] = []

    for lev in range(int(args.min_lev), int(args.max_lev) + 1, int(args.step)):
        short_run = _apply_leverage(short_cfg, lev)
        long_run = _apply_leverage(long_cfg, lev)
        accepted_df, all_df, portfolio_stats = run_backtest_dual(
            short_run, long_run, symbols, args.start, args.end, args.data, args.workers
        )
        init_cap = 0.0
        if short_run.get("backtest", {}).get("use_portfolio_sizing", False):
            init_cap = float(short_run.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
        summary = summarize(accepted_df, initial_capital=init_cap)
        summary["leverage"] = lev
        summary["meets_drawdown"] = _meets_drawdown(summary, args.max_drawdown)
        rows.append(summary)

        if _is_better(summary, best, args.max_drawdown):
            best = summary
            best_cfg = {"short": short_run, "long": long_run}

        if len(rows) >= 5:
            df_out = pd.DataFrame(rows)
            write_header = not os.path.exists(trials_path)
            df_out.to_csv(trials_path, index=False, mode="a", header=write_header)
            rows = []

    if rows:
        df_out = pd.DataFrame(rows)
        write_header = not os.path.exists(trials_path)
        df_out.to_csv(trials_path, index=False, mode="a", header=write_header)

    best_dir = os.path.join(args.out, "best")
    os.makedirs(best_dir, exist_ok=True)
    if best_cfg:
        with open(os.path.join(best_dir, "best_config_short.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(best_cfg["short"], f, allow_unicode=True, sort_keys=False)
        with open(os.path.join(best_dir, "best_config_long.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(best_cfg["long"], f, allow_unicode=True, sort_keys=False)
        with open(os.path.join(best_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)

    print(json.dumps(best or {}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
