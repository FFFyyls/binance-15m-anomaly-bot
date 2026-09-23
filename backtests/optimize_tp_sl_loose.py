import argparse
import copy
import csv
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from backtests.run_backtest import _load_long_strategy_config, run_backtest_dual, summarize


def _load_symbols(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]


def _apply_tp_sl(base_cfg: Dict[str, Any], hard_stop: float, trail_scale: float, partial_scale: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("risk", {})["hard_stop_pct"] = float(hard_stop)

    tp_cfg = cfg.setdefault("takeprofit", {})
    tpc = tp_cfg.setdefault("trailing_ladder", {})
    for key in ("p1", "p2", "p3", "lock2", "trail_pct"):
        if key in tpc and tpc[key] is not None:
            tpc[key] = float(tpc[key]) * float(trail_scale)

    sp = tp_cfg.setdefault("staged_partial", {})
    levels = sp.get("levels", []) or []
    for lvl in levels:
        if "r_multiple" in lvl and lvl["r_multiple"] is not None:
            lvl["r_multiple"] = float(lvl["r_multiple"]) * float(partial_scale)
    if "move_stop_to_breakeven_after_r" in sp and sp["move_stop_to_breakeven_after_r"] is not None:
        sp["move_stop_to_breakeven_after_r"] = float(sp["move_stop_to_breakeven_after_r"]) * float(partial_scale)
    return cfg


def _score(summary: Dict[str, Any], base: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, float, float, float]:
    pnl = float(summary.get("total_pnl_usdt", 0.0) or 0.0)
    trades = float(summary.get("trades", 0.0) or 0.0)
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)

    pnl_ratio = pnl / base["pnl"] if base["pnl"] else 0.0
    trades_ratio = trades / base["trades"] if base["trades"] else 0.0
    win_ratio = win_rate / base["win_rate"] if base["win_rate"] else 0.0

    score = (
        weights["pnl"] * pnl_ratio
        + weights["trades"] * trades_ratio
        + weights["win_rate"] * win_ratio
    )
    return score, pnl_ratio, trades_ratio, win_ratio


def _loose_score(hard_stop: float, base_hard_stop: float, trail_scale: float, partial_scale: float) -> float:
    if base_hard_stop <= 0:
        return trail_scale + partial_scale
    return hard_stop / base_hard_stop + trail_scale + partial_scale


def _sample_broad(rng: random.Random, base_hard_stop: float, max_hard_stop: float,
                  trail_min: float, trail_max: float, partial_min: float, partial_max: float) -> Tuple[float, float, float]:
    hard_stop = rng.uniform(base_hard_stop, max_hard_stop)
    trail_scale = rng.uniform(trail_min, trail_max)
    partial_scale = rng.uniform(partial_min, partial_max)
    return hard_stop, trail_scale, partial_scale


def _sample_refine(rng: random.Random, base: Tuple[float, float, float],
                   base_hard_stop: float, max_hard_stop: float,
                   trail_min: float, trail_max: float, partial_min: float, partial_max: float) -> Tuple[float, float, float]:
    hard_stop, trail_scale, partial_scale = base
    hard_stop *= rng.uniform(0.85, 1.15)
    trail_scale *= rng.uniform(0.85, 1.15)
    partial_scale *= rng.uniform(0.85, 1.15)

    hard_stop = max(base_hard_stop, min(max_hard_stop, hard_stop))
    trail_scale = max(trail_min, min(trail_max, trail_scale))
    partial_scale = max(partial_min, min(partial_max, partial_scale))
    return hard_stop, trail_scale, partial_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--long_config", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_drawdown", type=float, default=0.45)
    ap.add_argument("--min_pnl_ratio", type=float, default=0.6)
    ap.add_argument("--min_win_rate_ratio", type=float, default=0.8)
    ap.add_argument("--min_trades_ratio", type=float, default=0.7)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    long_base = _load_long_strategy_config(base_cfg, args.long_config)
    if long_base is None:
        raise SystemExit("ERROR: missing long config (provide --long_config or set long_strategy in config)")

    symbols = _load_symbols(args.symbols_file)

    # Baseline
    base_trades_df, _, _ = run_backtest_dual(
        base_cfg,
        long_base,
        symbols,
        None,
        None,
        args.data,
        args.workers,
        use_symbol_range=True,
        keep_all=False,
    )
    init_cap = float(base_cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
    base_summary = summarize(base_trades_df, initial_capital=init_cap)
    base_metrics = {
        "pnl": float(base_summary.get("total_pnl_usdt", 0.0) or 0.0),
        "trades": float(base_summary.get("trades", 0.0) or 0.0),
        "win_rate": float(base_summary.get("win_rate", 0.0) or 0.0),
    }

    weights = {"pnl": 0.45, "trades": 0.30, "win_rate": 0.25}

    base_hard_stop = float(base_cfg.get("risk", {}).get("hard_stop_pct", 0.0) or 0.0)
    max_hard_stop = min(0.05, base_hard_stop * 3.0) if base_hard_stop > 0 else 0.05
    trail_min, trail_max = 1.0, 2.5
    partial_min, partial_max = 1.0, 2.5

    total_trials = max(1, int(args.trials))
    stage1 = int(round(total_trials * 0.6))
    stage2 = total_trials - stage1

    if not args.out:
        if args.resume:
            root = "artifacts/backtests"
            candidates = [
                os.path.join(root, p)
                for p in os.listdir(root)
                if p.startswith("tp_sl_loose_sweep_")
            ] if os.path.isdir(root) else []
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p))
                args.out = candidates[-1]
        if not args.out:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            args.out = f"artifacts/backtests/tp_sl_loose_sweep_{ts}"
    os.makedirs(args.out, exist_ok=True)

    trials_path = os.path.join(args.out, "trials.csv")
    fieldnames = [
        "trial",
        "stage",
        "trades",
        "total_pnl_usdt",
        "win_rate",
        "profit_factor",
        "max_drawdown_naive",
        "score",
        "pnl_ratio",
        "trades_ratio",
        "win_rate_ratio",
        "loose_score",
        "meets_constraints",
        "hard_stop_pct",
        "trail_scale",
        "partial_scale",
    ]

    rng = random.Random(args.seed)
    all_rows: List[Dict[str, Any]] = []

    def run_trial(trial_idx: int, stage: str, hard_stop: float, trail_scale: float, partial_scale: float) -> Dict[str, Any]:
        short_cfg = _apply_tp_sl(base_cfg, hard_stop, trail_scale, partial_scale)
        long_cfg = _apply_tp_sl(long_base, hard_stop, trail_scale, partial_scale)
        trades_df, _, _ = run_backtest_dual(
            short_cfg,
            long_cfg,
            symbols,
            None,
            None,
            args.data,
            args.workers,
            use_symbol_range=True,
            keep_all=False,
        )
        summary = summarize(trades_df, initial_capital=init_cap)
        score, pnl_ratio, trades_ratio, win_ratio = _score(summary, base_metrics, weights)
        mdd = float(summary.get("max_drawdown_naive", 0.0) or 0.0)
        meets = (
            mdd >= -args.max_drawdown
            and pnl_ratio >= args.min_pnl_ratio
            and trades_ratio >= args.min_trades_ratio
            and win_ratio >= args.min_win_rate_ratio
        )
        loose = _loose_score(hard_stop, base_hard_stop, trail_scale, partial_scale)
        row = {
            "trial": trial_idx,
            "stage": stage,
            "trades": int(summary.get("trades", 0) or 0),
            "total_pnl_usdt": float(summary.get("total_pnl_usdt", 0.0) or 0.0),
            "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
            "profit_factor": float(summary.get("profit_factor", 0.0) or 0.0),
            "max_drawdown_naive": mdd,
            "score": score,
            "pnl_ratio": pnl_ratio,
            "trades_ratio": trades_ratio,
            "win_rate_ratio": win_ratio,
            "loose_score": loose,
            "meets_constraints": meets,
            "hard_stop_pct": float(hard_stop),
            "trail_scale": float(trail_scale),
            "partial_scale": float(partial_scale),
        }
        return row, short_cfg, long_cfg

    start_idx = 1
    stage1_done = 0
    stage2_done = 0
    if args.resume and os.path.exists(trials_path):
        with open(trials_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["trial"] = int(float(row["trial"]))
                row["trades"] = int(float(row["trades"]))
                row["total_pnl_usdt"] = float(row["total_pnl_usdt"])
                row["win_rate"] = float(row["win_rate"])
                row["profit_factor"] = float(row["profit_factor"])
                row["max_drawdown_naive"] = float(row["max_drawdown_naive"])
                row["score"] = float(row["score"])
                row["pnl_ratio"] = float(row["pnl_ratio"])
                row["trades_ratio"] = float(row["trades_ratio"])
                row["win_rate_ratio"] = float(row["win_rate_ratio"])
                row["loose_score"] = float(row["loose_score"])
                row["meets_constraints"] = str(row["meets_constraints"]).lower() in ("1", "true", "yes")
                row["hard_stop_pct"] = float(row["hard_stop_pct"])
                row["trail_scale"] = float(row["trail_scale"])
                row["partial_scale"] = float(row["partial_scale"])
                all_rows.append(row)
        if all_rows:
            start_idx = max(r["trial"] for r in all_rows) + 1
            stage1_done = sum(1 for r in all_rows if r.get("stage") == "broad")
            stage2_done = sum(1 for r in all_rows if r.get("stage") == "refine")

    write_header = not os.path.exists(trials_path) or not args.resume
    with open(trials_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i in range(start_idx, stage1 + 1):
            params = _sample_broad(rng, base_hard_stop, max_hard_stop, trail_min, trail_max, partial_min, partial_max)
            row, short_cfg, long_cfg = run_trial(i, "broad", *params)
            writer.writerow(row)
            all_rows.append(row)
            print(
                f"[{i}/{total_trials}] stage=broad score={row['score']:.4f} pnl_ratio={row['pnl_ratio']:.2f} "
                f"trades_ratio={row['trades_ratio']:.2f} win_ratio={row['win_rate_ratio']:.2f} "
                f"mdd={row['max_drawdown_naive']:.3f} meets={row['meets_constraints']}"
            )

        # refine
        if stage2 > 0 and stage2_done < stage2:
            stage1_rows = [r for r in all_rows if r.get("stage") == "broad"]
            meets_rows = [r for r in all_rows if r.get("meets_constraints")]
            seed_pool = meets_rows or stage1_rows
            seed_pool.sort(key=lambda r: (r["score"], r["loose_score"]), reverse=True)
            seeds = [(r["hard_stop_pct"], r["trail_scale"], r["partial_scale"]) for r in seed_pool[:10]]
            for j in range(stage2_done + 1, stage2 + 1):
                base = seeds[(j - 1) % len(seeds)]
                params = _sample_refine(rng, base, base_hard_stop, max_hard_stop, trail_min, trail_max, partial_min, partial_max)
                idx = stage1 + j
                row, short_cfg, long_cfg = run_trial(idx, "refine", *params)
                writer.writerow(row)
                all_rows.append(row)
                print(
                    f"[{idx}/{total_trials}] stage=refine score={row['score']:.4f} pnl_ratio={row['pnl_ratio']:.2f} "
                    f"trades_ratio={row['trades_ratio']:.2f} win_ratio={row['win_rate_ratio']:.2f} "
                    f"mdd={row['max_drawdown_naive']:.3f} meets={row['meets_constraints']}"
                )

    # Select final top 5 from all meeting constraints, prefer score then looseness.
    top_results = [r for r in all_rows if r.get("meets_constraints")]
    top_results.sort(key=lambda r: (r["score"], r["loose_score"]), reverse=True)
    top5 = top_results[:5]

    top_dir = os.path.join(args.out, "top_k")
    os.makedirs(top_dir, exist_ok=True)
    for rank, row in enumerate(top5, start=1):
        short_cfg = _apply_tp_sl(base_cfg, row["hard_stop_pct"], row["trail_scale"], row["partial_scale"])
        long_cfg = _apply_tp_sl(long_base, row["hard_stop_pct"], row["trail_scale"], row["partial_scale"])
        with open(os.path.join(top_dir, f"config_short_rank{rank}.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(short_cfg, f, sort_keys=False, allow_unicode=True)
        with open(os.path.join(top_dir, f"config_long_rank{rank}.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(long_cfg, f, sort_keys=False, allow_unicode=True)
        summary = {
            "rank": rank,
            "score_row": row,
            "baseline": base_summary,
            "constraints": {
                "max_drawdown": args.max_drawdown,
                "min_pnl_ratio": args.min_pnl_ratio,
                "min_trades_ratio": args.min_trades_ratio,
                "min_win_rate_ratio": args.min_win_rate_ratio,
            },
            "weights": weights,
            "symbols_file": args.symbols_file,
            "data": args.data,
            "assumptions": {
                "use_symbol_range": True,
                "fill_model": "limit filled if next bar touches (low/high)",
                "data_interval_seconds": int(base_cfg.get("backtest", {}).get("data_interval_seconds", 60)),
            },
        }
        with open(os.path.join(top_dir, f"summary_rank{rank}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "summary_top5.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "top5": [
                    {k: v for k, v in row.items() if k not in ("short_cfg", "long_cfg")}
                    for row in top5
                ],
                "baseline": base_summary,
                "constraints": {
                    "max_drawdown": args.max_drawdown,
                    "min_pnl_ratio": args.min_pnl_ratio,
                    "min_trades_ratio": args.min_trades_ratio,
                    "min_win_rate_ratio": args.min_win_rate_ratio,
                },
                "weights": weights,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Done. Results in {args.out}")


if __name__ == "__main__":
    main()
