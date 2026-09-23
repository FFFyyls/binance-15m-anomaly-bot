import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

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

    # Signal
    base_thr = float(cfg["signal"].get("abs_r15_threshold", 0.05))
    cfg["signal"]["abs_r15_threshold"] = _scale_value(base_thr, rng, 0.7, 1.3, 0.02, 0.3)
    cfg["signal"]["use_abs_threshold"] = True

    # Volume filters
    base_mult = float(cfg["volume"].get("multiple", 6.0))
    cfg["volume"]["multiple"] = _scale_value(base_mult, rng, 0.7, 1.3, 2.0, 25.0)

    base_lb = float(cfg["volume"].get("baseline_lookback_windows", 96))
    cfg["volume"]["baseline_lookback_windows"] = int(round(_scale_value(base_lb, rng, 0.7, 1.3, 24, 260)))

    if cfg["volume"].get("use_percentile_threshold", False):
        base_pct = float(cfg["volume"].get("percentile", 95.0))
        cfg["volume"]["percentile"] = _scale_value(base_pct, rng, 0.97, 1.03, 85.0, 99.5)
        base_plb = float(cfg["volume"].get("percentile_lookback_windows", 96))
        cfg["volume"]["percentile_lookback_windows"] = int(round(_scale_value(base_plb, rng, 0.7, 1.3, 24, 260)))

    # Risk
    base_stop = float(cfg["risk"].get("hard_stop_pct", 0.02))
    cfg["risk"]["hard_stop_pct"] = _scale_value(base_stop, rng, 0.7, 1.3, 0.005, 0.05)

    # Entry
    base_pullback = float(cfg["entry"].get("min_pullback_bps", 0.0))
    cfg["entry"]["min_pullback_bps"] = int(round(_scale_value(max(base_pullback, 0.1), rng, 0.5, 1.5, 0.0, 15.0)))
    base_slip = float(cfg["entry"].get("max_taker_slippage_bps", 8.0))
    cfg["entry"]["max_taker_slippage_bps"] = _scale_value(base_slip, rng, 0.7, 1.3, 4.0, 20.0)

    # Time exit
    time_exit = cfg.get("time_exit", {})
    base_p1 = float(time_exit.get("profit_1", 0.01))
    base_p2 = float(time_exit.get("profit_2", 0.02))
    time_exit["profit_1"] = _scale_value(base_p1, rng, 0.7, 1.3, 0.005, 0.05)
    time_exit["profit_2"] = _scale_value(base_p2, rng, 0.7, 1.3, 0.01, 0.08)
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
    new_p1 = _scale_value(base_p1, rng, 0.7, 1.3, 0.01, 0.08)
    new_p2 = _scale_value(base_p2, rng, 0.7, 1.3, 0.02, 0.12)
    new_p3 = _scale_value(base_p3, rng, 0.7, 1.3, 0.04, 0.2)
    p1, p2, p3 = sorted([new_p1, new_p2, new_p3])
    tpc["p1"] = p1
    tpc["p2"] = p2
    tpc["p3"] = p3
    base_lock2 = float(tpc.get("lock2", 0.01))
    tpc["lock2"] = _scale_value(base_lock2, rng, 0.7, 1.3, 0.0, 0.05)
    base_trail = float(tpc.get("trail_pct", 0.02))
    tpc["trail_pct"] = _scale_value(base_trail, rng, 0.7, 1.3, 0.005, 0.05)
    tp_cfg["trailing_ladder"] = tpc

    sp = tp_cfg.get("staged_partial", {})
    base_levels = list(sp.get("levels", []) or [])
    new_levels = []
    total_frac = 0.0
    for lvl in base_levels:
        base_r = float(lvl.get("r_multiple", 1.0))
        base_f = float(lvl.get("fraction", 0.2))
        r_mult = _scale_value(base_r, rng, 0.7, 1.3, 0.5, 4.0)
        frac = _scale_value(base_f, rng, 0.7, 1.3, 0.05, 0.6)
        new_levels.append({"r_multiple": r_mult, "fraction": frac})
        total_frac += frac
    if total_frac > 0.85:
        scale = 0.85 / total_frac
        for lvl in new_levels:
            lvl["fraction"] = float(lvl["fraction"]) * scale
    sp["levels"] = new_levels
    base_be = float(sp.get("move_stop_to_breakeven_after_r", 1.0))
    sp["move_stop_to_breakeven_after_r"] = _scale_value(base_be, rng, 0.7, 1.3, 0.3, 4.0)
    tp_cfg["staged_partial"] = sp
    cfg["takeprofit"] = tp_cfg
    return cfg


def _collect_params(prefix: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    tp = cfg.get("takeprofit", {})
    tpc = tp.get("trailing_ladder", {})
    sp = tp.get("staged_partial", {})
    levels = list(sp.get("levels", []) or [])
    lvl1 = levels[0] if len(levels) > 0 else {}
    lvl2 = levels[1] if len(levels) > 1 else {}

    return {
        f"{prefix}_abs_r15_threshold": cfg["signal"].get("abs_r15_threshold"),
        f"{prefix}_vol_multiple": cfg["volume"].get("multiple"),
        f"{prefix}_vol_percentile": cfg["volume"].get("percentile"),
        f"{prefix}_vol_percentile_lookback": cfg["volume"].get("percentile_lookback_windows"),
        f"{prefix}_baseline_lookback": cfg["volume"].get("baseline_lookback_windows"),
        f"{prefix}_hard_stop_pct": cfg["risk"].get("hard_stop_pct"),
        f"{prefix}_pullback_bps": cfg["entry"].get("min_pullback_bps"),
        f"{prefix}_max_taker_slip_bps": cfg["entry"].get("max_taker_slippage_bps"),
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
    }


def _meets_limits(summary: Dict[str, Any], max_dd: float, max_daily_dd: float) -> bool:
    return float(summary.get("max_drawdown_naive", 0.0)) >= -max_dd and float(
        summary.get("max_daily_drawdown", 0.0)
    ) >= -max_daily_dd


def _is_better(cand: Dict[str, Any], best: Optional[Dict[str, Any]],
               max_dd: float, max_daily_dd: float) -> bool:
    if best is None:
        return True

    cand_ok = _meets_limits(cand, max_dd, max_daily_dd)
    best_ok = _meets_limits(best, max_dd, max_daily_dd)
    if cand_ok and not best_ok:
        return True
    if not cand_ok and best_ok:
        return False

    cand_pnl = float(cand.get("total_pnl_usdt", 0.0))
    best_pnl = float(best.get("total_pnl_usdt", 0.0))
    cand_wr = float(cand.get("win_rate", 0.0))
    best_wr = float(best.get("win_rate", 0.0))
    cand_dd = float(cand.get("max_daily_drawdown", 0.0))
    best_dd = float(best.get("max_daily_drawdown", 0.0))
    cand_trades = int(cand.get("trades", 0) or 0)
    best_trades = int(best.get("trades", 0) or 0)

    if cand_pnl > best_pnl:
        return True
    pnl_tol = max(1e-6, abs(best_pnl) * 0.05)
    if abs(cand_pnl - best_pnl) <= pnl_tol:
        if cand_wr > best_wr:
            return True
        if abs(cand_wr - best_wr) <= 1e-6:
            if cand_dd > best_dd:
                return True
            if abs(cand_dd - best_dd) <= 1e-6 and cand_trades > best_trades:
                return True
    return False


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
    ap.add_argument("--out", default="artifacts/backtests/dual_optimization")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_drawdown", type=float, default=1.0)
    ap.add_argument("--max_daily_drawdown", type=float, default=0.45)
    ap.add_argument("--flush_every", type=int, default=5)
    args = ap.parse_args()

    short_base = yaml.safe_load(open(args.short_config, "r", encoding="utf-8"))
    long_base = _load_long_config(short_base, args.long_config)
    if long_base is None:
        raise SystemExit("ERROR: missing long config (provide --long_config or set long_strategy in short config)")

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
        summary["max_daily_drawdown"] = _daily_drawdown(accepted_df, init_cap)
        summary["meets_daily_drawdown"] = summary["max_daily_drawdown"] >= -args.max_daily_drawdown
        summary["meets_drawdown"] = summary["max_drawdown_naive"] >= -args.max_drawdown
        summary.update(_collect_params("short", short_cfg))
        summary.update(_collect_params("long", long_cfg))
        rows.append(summary)

        if _is_better(summary, best, args.max_drawdown, args.max_daily_drawdown):
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
        print("No valid trials executed.")
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
    summary["max_daily_drawdown"] = _daily_drawdown(accepted_df, init_cap)
    summary.update(
        {
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "meets_daily_drawdown": summary["max_daily_drawdown"] >= -args.max_daily_drawdown,
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
