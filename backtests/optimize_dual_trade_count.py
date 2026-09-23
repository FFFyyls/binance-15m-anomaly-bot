import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from backtests.run_backtest import run_backtest_dual, summarize


def _clamp(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return float(value)


def _scale_value(base: float, rng: np.random.Generator, low: float, high: float,
                 min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    scaled = base * float(rng.uniform(low, high))
    return _clamp(scaled, min_v, max_v)


def _sample_params(base_cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    window_choices = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    cfg["signal"]["window_minutes"] = int(rng.choice(window_choices))

    base_thr = float(cfg["signal"].get("abs_r15_threshold", 0.05))
    cfg["signal"]["abs_r15_threshold"] = _scale_value(base_thr, rng, 0.65, 1.15, 0.02, 0.3)
    cfg["signal"]["use_abs_threshold"] = True

    vol_cfg = cfg.get("volume", {})
    base_lb = float(vol_cfg.get("baseline_lookback_windows", vol_cfg.get("baseline_lookback_15m", 96)))
    vol_cfg["baseline_lookback_windows"] = int(round(_scale_value(base_lb, rng, 0.7, 1.3, 24, 260)))

    if vol_cfg.get("use_percentile_threshold", False):
        base_pct = float(vol_cfg.get("percentile", 95.0))
        vol_cfg["percentile"] = _scale_value(base_pct, rng, 0.98, 1.02, 88.0, 99.5)
        base_plb = float(vol_cfg.get("percentile_lookback_windows", 96))
        vol_cfg["percentile_lookback_windows"] = int(round(_scale_value(base_plb, rng, 0.7, 1.3, 24, 260)))
    else:
        base_mult = float(vol_cfg.get("multiple", 6.0))
        vol_cfg["multiple"] = _scale_value(base_mult, rng, 0.7, 1.2, 2.0, 25.0)
    cfg["volume"] = vol_cfg

    # Risk
    base_stop = float(cfg["risk"].get("hard_stop_pct", 0.02))
    cfg["risk"]["hard_stop_pct"] = _scale_value(base_stop, rng, 0.8, 1.2, 0.005, 0.05)

    # Entry
    base_pullback = float(cfg["entry"].get("min_pullback_bps", 0.0))
    cfg["entry"]["min_pullback_bps"] = int(round(_scale_value(max(base_pullback, 0.1), rng, 0.5, 1.5, 0.0, 15.0)))
    base_slip = float(cfg["entry"].get("max_taker_slippage_bps", 8.0))
    cfg["entry"]["max_taker_slippage_bps"] = _scale_value(base_slip, rng, 0.8, 1.2, 4.0, 25.0)

    # Time exit
    time_exit = cfg.get("time_exit", {})
    base_p1 = float(time_exit.get("profit_1", 0.01))
    base_p2 = float(time_exit.get("profit_2", 0.02))
    time_exit["profit_1"] = _scale_value(base_p1, rng, 0.8, 1.2, 0.005, 0.05)
    time_exit["profit_2"] = _scale_value(base_p2, rng, 0.8, 1.2, 0.01, 0.08)
    if time_exit["profit_2"] <= time_exit["profit_1"]:
        time_exit["profit_2"] = min(0.1, time_exit["profit_1"] * 1.5)
    base_h1 = float(time_exit.get("hold_1_minutes", 12))
    base_h2 = float(time_exit.get("hold_2_minutes", 20))
    time_exit["hold_1_minutes"] = int(round(_scale_value(base_h1, rng, 0.7, 1.3, 5, 60)))
    time_exit["hold_2_minutes"] = int(round(_scale_value(base_h2, rng, 0.7, 1.3, 10, 120)))
    if time_exit["hold_2_minutes"] <= time_exit["hold_1_minutes"]:
        time_exit["hold_2_minutes"] = time_exit["hold_1_minutes"] + 5
    cfg["time_exit"] = time_exit

    # Take profit (hybrid: staged partial + trailing ladder)
    tp_cfg = cfg.get("takeprofit", {})
    tpc = tp_cfg.get("trailing_ladder", {})
    base_p1 = float(tpc.get("p1", 0.02))
    base_p2 = float(tpc.get("p2", 0.04))
    base_p3 = float(tpc.get("p3", 0.08))
    new_p1 = _scale_value(base_p1, rng, 0.8, 1.2, 0.01, 0.08)
    new_p2 = _scale_value(base_p2, rng, 0.8, 1.2, 0.02, 0.12)
    new_p3 = _scale_value(base_p3, rng, 0.8, 1.2, 0.04, 0.2)
    p1, p2, p3 = sorted([new_p1, new_p2, new_p3])
    tpc["p1"] = p1
    tpc["p2"] = p2
    tpc["p3"] = p3
    base_lock2 = float(tpc.get("lock2", 0.01))
    tpc["lock2"] = _scale_value(base_lock2, rng, 0.8, 1.2, 0.0, 0.05)
    base_trail = float(tpc.get("trail_pct", 0.02))
    tpc["trail_pct"] = _scale_value(base_trail, rng, 0.8, 1.2, 0.005, 0.05)
    tp_cfg["trailing_ladder"] = tpc

    sp = tp_cfg.get("staged_partial", {})
    base_levels = list(sp.get("levels", []) or [])
    new_levels = []
    total_frac = 0.0
    for lvl in base_levels:
        base_r = float(lvl.get("r_multiple", 1.0))
        base_f = float(lvl.get("fraction", 0.2))
        r_mult = _scale_value(base_r, rng, 0.8, 1.2, 0.5, 4.0)
        frac = _scale_value(base_f, rng, 0.8, 1.2, 0.05, 0.6)
        new_levels.append({"r_multiple": r_mult, "fraction": frac})
        total_frac += frac
    if total_frac > 0.85:
        scale = 0.85 / total_frac
        for lvl in new_levels:
            lvl["fraction"] = float(lvl["fraction"]) * scale
    sp["levels"] = new_levels
    base_be = float(sp.get("move_stop_to_breakeven_after_r", 1.0))
    sp["move_stop_to_breakeven_after_r"] = _scale_value(base_be, rng, 0.8, 1.2, 0.3, 4.0)
    tp_cfg["staged_partial"] = sp
    cfg["takeprofit"] = tp_cfg

    # Bollinger (keep mode as-is, allow window/std to vary)
    bb_cfg = cfg.get("bollinger", {})
    if bb_cfg.get("enabled", False):
        bb_cfg["window_minutes"] = int(rng.choice([10, 15, 20, 30]))
        base_std = float(bb_cfg.get("num_std", 2.5))
        bb_cfg["num_std"] = _scale_value(base_std, rng, 0.85, 1.15, 2.0, 3.5)
        cfg["bollinger"] = bb_cfg

    return cfg


def _collect_params(prefix: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    tp = cfg.get("takeprofit", {})
    tpc = tp.get("trailing_ladder", {})
    sp = tp.get("staged_partial", {})
    levels = list(sp.get("levels", []) or [])
    lvl1 = levels[0] if len(levels) > 0 else {}
    lvl2 = levels[1] if len(levels) > 1 else {}
    bb = cfg.get("bollinger", {})

    return {
        f"{prefix}_window_minutes": cfg["signal"].get("window_minutes"),
        f"{prefix}_abs_r15_threshold": cfg["signal"].get("abs_r15_threshold"),
        f"{prefix}_vol_percentile": cfg.get("volume", {}).get("percentile"),
        f"{prefix}_vol_percentile_lookback": cfg.get("volume", {}).get("percentile_lookback_windows"),
        f"{prefix}_baseline_lookback": cfg.get("volume", {}).get("baseline_lookback_windows"),
        f"{prefix}_hard_stop_pct": cfg.get("risk", {}).get("hard_stop_pct"),
        f"{prefix}_pullback_bps": cfg.get("entry", {}).get("min_pullback_bps"),
        f"{prefix}_max_taker_slip_bps": cfg.get("entry", {}).get("max_taker_slippage_bps"),
        f"{prefix}_tp_p1": tpc.get("p1"),
        f"{prefix}_tp_p2": tpc.get("p2"),
        f"{prefix}_tp_p3": tpc.get("p3"),
        f"{prefix}_tp_lock2": tpc.get("lock2"),
        f"{prefix}_tp_trail_pct": tpc.get("trail_pct"),
        f"{prefix}_tp_r1": lvl1.get("r_multiple"),
        f"{prefix}_tp_f1": lvl1.get("fraction"),
        f"{prefix}_tp_r2": lvl2.get("r_multiple"),
        f"{prefix}_tp_f2": lvl2.get("fraction"),
        f"{prefix}_tp_move_be_r": sp.get("move_stop_to_breakeven_after_r"),
        f"{prefix}_time_profit_1": cfg.get("time_exit", {}).get("profit_1"),
        f"{prefix}_time_profit_2": cfg.get("time_exit", {}).get("profit_2"),
        f"{prefix}_time_hold_1": cfg.get("time_exit", {}).get("hold_1_minutes"),
        f"{prefix}_time_hold_2": cfg.get("time_exit", {}).get("hold_2_minutes"),
        f"{prefix}_bb_window_minutes": bb.get("window_minutes"),
        f"{prefix}_bb_num_std": bb.get("num_std"),
        f"{prefix}_bb_mode": bb.get("mode"),
    }


def _meets_constraints(summary: Dict[str, Any], min_pnl: float, min_wr: float) -> bool:
    return float(summary.get("total_pnl_usdt", 0.0)) >= min_pnl and float(summary.get("win_rate", 0.0)) >= min_wr


def _is_better(cand: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    if best is None:
        return True
    cand_tr = int(cand.get("trades", 0) or 0)
    best_tr = int(best.get("trades", 0) or 0)
    if cand_tr > best_tr:
        return True
    if cand_tr < best_tr:
        return False
    cand_pnl = float(cand.get("total_pnl_usdt", 0.0))
    best_pnl = float(best.get("total_pnl_usdt", 0.0))
    if cand_pnl > best_pnl:
        return True
    if cand_pnl < best_pnl:
        return False
    cand_wr = float(cand.get("win_rate", 0.0))
    best_wr = float(best.get("win_rate", 0.0))
    if cand_wr > best_wr:
        return True
    cand_pf = float(cand.get("profit_factor", 0.0))
    best_pf = float(best.get("profit_factor", 0.0))
    if cand_pf > best_pf:
        return True
    cand_dd = float(cand.get("max_drawdown_naive", 0.0))
    best_dd = float(best.get("max_drawdown_naive", 0.0))
    return cand_dd > best_dd


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--short_config", default="configs/strategy.yaml")
    ap.add_argument("--long_config", default="")
    ap.add_argument("--baseline_summary", default="artifacts/backtests/baseline/summary.json")
    ap.add_argument("--out", default="artifacts/backtests/trade_count_optimization")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min_pnl_ratio", type=float, default=0.9)
    ap.add_argument("--max_win_rate_drop", type=float, default=0.03)
    ap.add_argument("--flush_every", type=int, default=5)
    args = ap.parse_args()

    short_base = yaml.safe_load(open(args.short_config, "r", encoding="utf-8"))
    long_base = _load_long_config(short_base, args.long_config)
    if long_base is None:
        raise SystemExit("ERROR: missing long config (provide --long_config or set long_strategy in short config)")

    with open(args.baseline_summary, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    base_pnl = float(baseline.get("total_pnl_usdt", 0.0))
    base_wr = float(baseline.get("win_rate", 0.0))
    min_pnl = base_pnl * args.min_pnl_ratio
    min_wr = base_wr - args.max_win_rate_drop

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]

    os.makedirs(args.out, exist_ok=True)
    trials_path = os.path.join(args.out, "trials.csv")
    if os.path.exists(trials_path):
        os.remove(trials_path)

    rng = np.random.default_rng(args.seed)
    best = None
    best_short = None
    best_long = None
    rows: List[Dict[str, Any]] = []

    for i in range(1, args.trials + 1):
        short_cfg = _sample_params(short_base, rng)
        long_cfg = _sample_params(long_base, rng)
        accepted_df, all_df, _ = run_backtest_dual(
            short_cfg, long_cfg, symbols, args.start, args.end, args.data, args.workers
        )

        init_cap = 0.0
        if short_cfg.get("backtest", {}).get("use_portfolio_sizing", False):
            init_cap = float(short_cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)

        summary = summarize(accepted_df, initial_capital=init_cap)
        summary["trial"] = i
        summary["meets_constraints"] = _meets_constraints(summary, min_pnl, min_wr)
        summary.update(_collect_params("short", short_cfg))
        summary.update(_collect_params("long", long_cfg))
        rows.append(summary)

        if summary["meets_constraints"] and _is_better(summary, best):
            best = summary
            best_short = short_cfg
            best_long = long_cfg

        if i % args.flush_every == 0 or i == args.trials:
            df_out = pd.DataFrame(rows)
            if not df_out.empty:
                write_header = not os.path.exists(trials_path)
                df_out.to_csv(trials_path, index=False, mode="a", header=write_header)
            rows = []

    if best_short is None or best_long is None:
        print("No trials met the constraints.")
        return

    best_dir = os.path.join(args.out, "best")
    os.makedirs(best_dir, exist_ok=True)
    accepted_df, all_df, portfolio_stats = run_backtest_dual(
        best_short, best_long, symbols, args.start, args.end, args.data, args.workers
    )
    accepted_df.to_csv(os.path.join(best_dir, "trades.csv"), index=False)
    if not all_df.empty:
        all_df.to_csv(os.path.join(best_dir, "trades_all.csv"), index=False)

    init_cap = 0.0
    if best_short.get("backtest", {}).get("use_portfolio_sizing", False):
        init_cap = float(best_short.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
    summary = summarize(accepted_df, initial_capital=init_cap)
    summary.update(
        {
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "baseline": {
                "total_pnl_usdt": base_pnl,
                "win_rate": base_wr,
            },
            "constraints": {
                "min_pnl_ratio": args.min_pnl_ratio,
                "max_win_rate_drop": args.max_win_rate_drop,
            },
            "assumptions": {
                "data": f"{int(best_short.get('backtest', {}).get('data_interval_seconds', 60))}s OHLCV",
                "fill_model": "limit filled if next bar touches (low/high)",
            },
            "backtest_params": best_short.get("backtest", {}),
            "portfolio": portfolio_stats,
        }
    )

    with open(os.path.join(best_dir, "best_config_short.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_short, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(best_dir, "best_config_long.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_long, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(best_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
