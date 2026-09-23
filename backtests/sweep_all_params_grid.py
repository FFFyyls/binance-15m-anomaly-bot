import argparse
import copy
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from backtests.run_backtest import (
    _load_long_strategy_config,
    apply_portfolio_constraints,
    run_backtest_dual,
    run_symbol_dual,
    summarize,
)


def _load_symbols(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]


def _apply_params(cfg: Dict, params: Dict) -> Dict:
    out = copy.deepcopy(cfg)

    signal = out.setdefault("signal", {})
    base_thr = float(signal.get("abs_r15_threshold", 0.0))
    signal["abs_r15_threshold"] = base_thr * params["abs_mult"]

    vol = out.setdefault("volume", {})
    if vol.get("use_percentile_threshold", False):
        vol["percentile"] = float(params["percentile"])
        vol["percentile_lookback_windows"] = int(params["percentile_lookback_windows"])

    entry = out.setdefault("entry", {})
    entry["min_pullback_bps"] = float(params["min_pullback_bps"])
    entry["max_taker_slippage_bps"] = float(params["max_taker_slippage_bps"])

    risk = out.setdefault("risk", {})
    risk["hard_stop_pct"] = float(params["hard_stop_pct"])

    bb = out.setdefault("bollinger", {})
    if bb.get("enabled", False):
        bb["window_minutes"] = int(params["bb_window_minutes"])
        bb["num_std"] = float(params["bb_num_std"])

    tp = out.setdefault("takeprofit", {})
    tpc = tp.setdefault("trailing_ladder", {})
    trail_scale = float(params["trail_scale"])
    for key in ("p1", "p2", "p3", "lock2", "trail_pct"):
        base = float(tpc.get(key, 0.0))
        tpc[key] = max(0.0, base * trail_scale)

    partial_scale = float(params["partial_scale"])
    staged = tp.get("staged_partial", {})
    levels = staged.get("levels", [])
    for level in levels:
        base = float(level.get("r_multiple", 0.0))
        level["r_multiple"] = max(0.0, base * partial_scale)
    if "move_stop_to_breakeven_after_r" in staged:
        base = float(staged.get("move_stop_to_breakeven_after_r", 0.0))
        staged["move_stop_to_breakeven_after_r"] = max(0.0, base * partial_scale)

    return out


def _score(summary: Dict, baseline: Dict, weight_pnl: float, weight_trades: float,
           weight_win_rate: float, mdd_gate: float) -> Tuple[float, Dict]:
    base_pnl = float(baseline.get("total_pnl_usdt", 1.0) or 1.0)
    base_trades = float(baseline.get("trades", 1.0) or 1.0)
    base_win = float(baseline.get("win_rate", 1.0) or 1.0)

    pnl = float(summary.get("total_pnl_usdt", 0.0))
    trades = float(summary.get("trades", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))
    mdd = float(summary.get("max_drawdown_naive", 0.0))

    score = (
        weight_pnl * (pnl / base_pnl)
        + weight_trades * (trades / base_trades)
        + weight_win_rate * (win_rate / base_win)
    )
    gates = {
        "mdd_ok": mdd >= mdd_gate,
    }
    gates["pass"] = all(gates.values())
    return score, gates


def _run_backtest_dual_pool(short_cfg: dict, long_cfg: dict, symbols: List[str], start: str, end: str,
                            data: str, ex: ProcessPoolExecutor, use_symbol_range: bool,
                            keep_all: bool) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    tasks = [(sym, start or None, end or None, data, short_cfg, long_cfg, use_symbol_range) for sym in symbols]
    all_trades = []
    missing = []
    for fut in as_completed([ex.submit(run_symbol_dual, t) for t in tasks]):
        sym, tr, err = fut.result()
        if err:
            missing.append(sym)
            continue
        all_trades.extend(tr)

    if not all_trades:
        return pd.DataFrame(), pd.DataFrame(), {"missing": missing}

    tdf = pd.DataFrame([t.__dict__ for t in all_trades])
    portfolio_cfg = copy.deepcopy(short_cfg)
    if bool(long_cfg.get("backtest", {}).get("use_portfolio_sizing", False)):
        portfolio_cfg.setdefault("backtest", {})["use_portfolio_sizing"] = True
    accepted_df, all_df, portfolio_stats = apply_portfolio_constraints(tdf, portfolio_cfg, keep_all=keep_all)
    if portfolio_stats:
        portfolio_stats["missing"] = missing
    return accepted_df, all_df, portfolio_stats


def _sample_params(space: Dict[str, List], trials: int, seed: int) -> List[Dict]:
    random.seed(seed)
    keys = list(space.keys())
    seen = set()
    out = []
    attempts = 0
    max_attempts = trials * 50
    while len(out) < trials and attempts < max_attempts:
        attempts += 1
        combo = tuple(random.choice(space[k]) for k in keys)
        if combo in seen:
            continue
        seen.add(combo)
        out.append(dict(zip(keys, combo)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--long_config", default="")
    ap.add_argument("--symbols_file", required=True)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts/backtests/all_params_sweep")
    ap.add_argument("--baseline_summary", default="artifacts/backtests/live_params_backtest/summary.json")
    ap.add_argument("--use_symbol_range", action="store_true")
    ap.add_argument("--weight_pnl", type=float, default=0.425)
    ap.add_argument("--weight_trades", type=float, default=0.325)
    ap.add_argument("--weight_win_rate", type=float, default=0.25)
    ap.add_argument("--mdd_gate", type=float, default=-0.45)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    short_base = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    long_base = _load_long_strategy_config(short_base, args.long_config)
    if long_base is None:
        raise SystemExit("ERROR: missing long config (provide --long_config or set long_strategy in config).")

    if not os.path.exists(args.baseline_summary):
        raise SystemExit(f"ERROR: baseline summary not found: {args.baseline_summary}")
    baseline = json.load(open(args.baseline_summary, "r", encoding="utf-8"))

    symbols = _load_symbols(args.symbols_file)

    space = {
        "abs_mult": [0.6, 0.7, 0.8, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.2, 1.3, 1.4, 1.5],
        "percentile": [80, 82, 85, 88, 90, 92, 94, 96, 98, 99],
        "percentile_lookback_windows": [18, 24, 36, 48, 60, 72, 96, 120, 144, 168],
        "min_pullback_bps": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12],
        "max_taker_slippage_bps": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32],
        "hard_stop_pct": [0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.024, 0.028, 0.032],
        "trail_scale": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0],
        "partial_scale": [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
        "bb_window_minutes": [8, 10, 12, 15, 18, 20, 24, 30, 36, 40],
        "bb_num_std": [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6],
    }

    trials = _sample_params(space, args.trials, args.seed)
    top_k = max(1, int(args.top_k))

    out_csv = os.path.join(args.out, "trials.csv")
    fieldnames = [
        "trial",
        "trades",
        "total_pnl_usdt",
        "win_rate",
        "profit_factor",
        "max_drawdown_naive",
        "score",
        "pass",
        "mdd_ok",
        "abs_mult",
        "percentile",
        "percentile_lookback_windows",
        "min_pullback_bps",
        "max_taker_slippage_bps",
        "hard_stop_pct",
        "trail_scale",
        "partial_scale",
        "bb_window_minutes",
        "bb_num_std",
    ]

    top_results: List[Tuple[Dict, Dict, Dict, Dict, Dict]] = []

    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)

    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, params in enumerate(trials, start=1):
                short_cfg = _apply_params(short_base, params)
                long_cfg = _apply_params(long_base, params)

                if pool:
                    accepted_df, all_df, portfolio_stats = _run_backtest_dual_pool(
                        short_cfg,
                        long_cfg,
                        symbols,
                        args.start,
                        args.end,
                        args.data,
                        pool,
                        args.use_symbol_range,
                        keep_all=False,
                    )
                else:
                    accepted_df, all_df, portfolio_stats = run_backtest_dual(
                        short_cfg,
                        long_cfg,
                        symbols,
                        args.start or None,
                        args.end or None,
                        args.data,
                        args.workers,
                        use_symbol_range=args.use_symbol_range,
                        keep_all=False,
                    )
                init_cap = float(short_cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
                summary = summarize(accepted_df, initial_capital=init_cap)
                score, gates = _score(
                    summary,
                    baseline,
                    args.weight_pnl,
                    args.weight_trades,
                    args.weight_win_rate,
                    args.mdd_gate,
                )

                row = {
                    "trial": i,
                    "trades": summary["trades"],
                    "total_pnl_usdt": summary["total_pnl_usdt"],
                    "win_rate": summary["win_rate"],
                    "profit_factor": summary["profit_factor"],
                    "max_drawdown_naive": summary["max_drawdown_naive"],
                    "score": score,
                    "pass": gates["pass"],
                    "mdd_ok": gates["mdd_ok"],
                    **params,
                }
                writer.writerow(row)
                if gates["pass"]:
                    top_results.append((row, short_cfg, long_cfg, summary, portfolio_stats))
                    top_results.sort(key=lambda r: r[0]["score"], reverse=True)
                    if len(top_results) > top_k:
                        top_results = top_results[:top_k]
                del accepted_df
                del all_df
    finally:
        if pool:
            pool.shutdown(wait=True)

    if top_results:
        best_dir = os.path.join(args.out, "best")
        os.makedirs(best_dir, exist_ok=True)
        best = top_results[0][0]
        best_payload = top_results[0]
        with open(os.path.join(best_dir, "best_config_short.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(best_payload[1], f, sort_keys=False, allow_unicode=True)
        with open(os.path.join(best_dir, "best_config_long.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(best_payload[2], f, sort_keys=False, allow_unicode=True)
        with open(os.path.join(best_dir, "best_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best": best,
                    "summary": best_payload[3],
                    "portfolio": best_payload[4],
                    "baseline": baseline,
                    "weights": {
                        "pnl": args.weight_pnl,
                        "trades": args.weight_trades,
                        "win_rate": args.weight_win_rate,
                        "mdd_gate": args.mdd_gate,
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        top_dir = os.path.join(args.out, "top_k")
        os.makedirs(top_dir, exist_ok=True)
        for rank, (row, short_cfg, long_cfg, summary, portfolio_stats) in enumerate(top_results, start=1):
            with open(os.path.join(top_dir, f"config_short_rank{rank}.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(short_cfg, f, sort_keys=False, allow_unicode=True)
            with open(os.path.join(top_dir, f"config_long_rank{rank}.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(long_cfg, f, sort_keys=False, allow_unicode=True)
            with open(os.path.join(top_dir, f"summary_rank{rank}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rank": rank,
                        "score_row": row,
                        "summary": summary,
                        "portfolio": portfolio_stats,
                        "baseline": baseline,
                        "weights": {
                            "pnl": args.weight_pnl,
                            "trades": args.weight_trades,
                            "win_rate": args.weight_win_rate,
                            "mdd_gate": args.mdd_gate,
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )


if __name__ == "__main__":
    main()
