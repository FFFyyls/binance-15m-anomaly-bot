import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from backtests.run_backtest import run_backtest, summarize


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


def _sample_takeprofit(cfg: Dict[str, Any], rng: np.random.Generator) -> None:
    tp = cfg.setdefault("takeprofit", {})
    tp["mode"] = "hybrid"

    tpc = tp.setdefault("trailing_ladder", {})
    p1 = float(rng.uniform(0.02, 0.06))
    p2 = float(rng.uniform(p1 + 0.01, p1 + 0.08))
    p3 = float(rng.uniform(p2 + 0.01, p2 + 0.10))
    lock2 = float(rng.uniform(0.0, min(0.03, p2)))
    trail = float(rng.uniform(0.015, 0.08))
    tpc["p1"] = p1
    tpc["p2"] = p2
    tpc["p3"] = p3
    tpc["lock2"] = lock2
    tpc["trail_pct"] = trail

    sp = tp.setdefault("staged_partial", {})
    levels_n = int(rng.integers(2, 4))
    r1 = float(rng.uniform(0.8, 1.8))
    r2 = float(rng.uniform(r1 + 0.4, r1 + 2.0))
    f1 = float(rng.uniform(0.2, 0.5))
    f2 = float(rng.uniform(0.2, 0.5))
    levels = [
        {"r_multiple": r1, "fraction": f1},
        {"r_multiple": r2, "fraction": f2},
    ]
    if levels_n == 3:
        r3 = float(rng.uniform(r2 + 0.4, r2 + 2.0))
        f3 = float(rng.uniform(0.1, 0.4))
        levels.append({"r_multiple": r3, "fraction": f3})
    levels = sorted(levels, key=lambda x: float(x.get("r_multiple", 0.0)))
    sp["levels"] = levels
    sp["move_stop_to_breakeven_after_r"] = float(rng.uniform(0.5, 1.8))


def _collect_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tp = cfg.get("takeprofit", {})
    tpc = tp.get("trailing_ladder", {})
    sp = tp.get("staged_partial", {})
    levels = sp.get("levels", [])
    out = {
        "tp_mode": tp.get("mode"),
        "p1": tpc.get("p1"),
        "p2": tpc.get("p2"),
        "p3": tpc.get("p3"),
        "lock2": tpc.get("lock2"),
        "trail_pct": tpc.get("trail_pct"),
        "tp_levels": len(levels),
        "move_be_r": sp.get("move_stop_to_breakeven_after_r"),
    }
    for idx, lvl in enumerate(levels, 1):
        out[f"r{idx}"] = lvl.get("r_multiple")
        out[f"f{idx}"] = lvl.get("fraction")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_top60.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--out", default="artifacts/backtests/takeprofit_sweep")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_drawdown", type=float, default=0.3)
    ap.add_argument("--flush_every", type=int, default=5)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    base_cfg.setdefault("takeprofit", {})
    base_cfg.setdefault("backtest", {})

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]

    os.makedirs(args.out, exist_ok=True)
    trials_path = os.path.join(args.out, "trials.csv")
    if os.path.exists(trials_path):
        os.remove(trials_path)

    rng = np.random.default_rng(args.seed)
    best = None
    best_cfg = None
    buffer: List[Dict[str, Any]] = []
    flush_every = max(1, int(args.flush_every))

    for i in range(1, args.trials + 1):
        cfg = copy.deepcopy(base_cfg)
        _sample_takeprofit(cfg, rng)

        accepted_df, all_df, _ = run_backtest(cfg, symbols, args.start, args.end, args.data, args.workers)
        init_cap = 0.0
        if cfg.get("backtest", {}).get("use_portfolio_sizing", False):
            init_cap = float(cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
        summary = summarize(accepted_df, initial_capital=init_cap)
        summary["trial"] = i
        summary["meets_drawdown"] = _meets_drawdown(summary, args.max_drawdown)
        summary.update(_collect_params(cfg))
        buffer.append(summary)

        if _is_better(summary, best, args.max_drawdown):
            best = summary
            best_cfg = cfg

        if i % flush_every == 0 or i == args.trials:
            df_out = pd.DataFrame(buffer)
            if not df_out.empty:
                write_header = not os.path.exists(trials_path)
                df_out.to_csv(trials_path, index=False, mode="a", header=write_header)
            buffer = []

    if best_cfg is None:
        print("No valid trials executed.")
        return

    best_dir = os.path.join(args.out, "best")
    os.makedirs(best_dir, exist_ok=True)
    accepted_df, all_df, portfolio_stats = run_backtest(best_cfg, symbols, args.start, args.end, args.data, args.workers)
    accepted_df.to_csv(os.path.join(best_dir, "trades.csv"), index=False)
    if not all_df.empty:
        all_df.to_csv(os.path.join(best_dir, "trades_all.csv"), index=False)

    init_cap = 0.0
    if best_cfg.get("backtest", {}).get("use_portfolio_sizing", False):
        init_cap = float(best_cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
    summary = summarize(accepted_df, initial_capital=init_cap)
    summary.update(
        {
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "meets_drawdown": _meets_drawdown(summary, args.max_drawdown),
            "assumptions": {
                "data": f"{int(best_cfg.get('backtest', {}).get('data_interval_seconds', 60))}s OHLCV",
                "fill_model": "limit filled if next bar touches (low/high)",
            },
            "backtest_params": best_cfg.get("backtest", {}),
        }
    )
    if portfolio_stats:
        summary["portfolio"] = portfolio_stats

    with open(os.path.join(best_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(best_dir, "best_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
