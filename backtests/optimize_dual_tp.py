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


def _partial_rate(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    if "reason" not in trades_df.columns:
        return 0.0
    return float(trades_df["reason"].astype(str).str.contains("PARTIAL").mean())


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


def _normalize_levels(levels: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not levels:
        return levels
    # sort by r_multiple ascending
    levels = sorted(levels, key=lambda x: float(x.get("r_multiple", 0.0)))
    total_frac = sum(float(lvl.get("fraction", 0.0)) for lvl in levels)
    if total_frac > 0.85:
        scale = 0.85 / total_frac
        for lvl in levels:
            lvl["fraction"] = float(lvl.get("fraction", 0.0)) * scale
    return levels


def _sample_tp_sl_params(base_cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # Risk: tighten/loosen hard stop moderately
    base_stop = float(cfg["risk"].get("hard_stop_pct", 0.02))
    cfg["risk"]["hard_stop_pct"] = _scale_value(base_stop, rng, 0.7, 1.1, 0.005, 0.05)

    # Time exit: small nudges to avoid changing trade count too much
    time_exit = cfg.get("time_exit", {})
    base_p1 = float(time_exit.get("profit_1", 0.01))
    base_p2 = float(time_exit.get("profit_2", 0.02))
    time_exit["profit_1"] = _scale_value(base_p1, rng, 0.85, 1.15, 0.005, 0.05)
    time_exit["profit_2"] = _scale_value(base_p2, rng, 0.85, 1.15, 0.01, 0.08)
    if time_exit["profit_2"] <= time_exit["profit_1"]:
        time_exit["profit_2"] = min(0.1, time_exit["profit_1"] * 1.4)
    base_h1 = float(time_exit.get("hold_1_minutes", 12))
    base_h2 = float(time_exit.get("hold_2_minutes", 20))
    time_exit["hold_1_minutes"] = int(round(_scale_value(base_h1, rng, 0.85, 1.15, 5, 60)))
    time_exit["hold_2_minutes"] = int(round(_scale_value(base_h2, rng, 0.85, 1.15, 10, 120)))
    if time_exit["hold_2_minutes"] <= time_exit["hold_1_minutes"]:
        time_exit["hold_2_minutes"] = time_exit["hold_1_minutes"] + 5
    cfg["time_exit"] = time_exit

    # Take profit: staged partial + trailing ladder
    tp_cfg = cfg.get("takeprofit", {})
    tpc = tp_cfg.get("trailing_ladder", {})

    base_p1 = float(tpc.get("p1", 0.02))
    base_p2 = float(tpc.get("p2", 0.04))
    base_p3 = float(tpc.get("p3", 0.08))
    new_p1 = _scale_value(base_p1, rng, 0.75, 1.15, 0.01, 0.08)
    new_p2 = _scale_value(base_p2, rng, 0.75, 1.15, 0.02, 0.12)
    new_p3 = _scale_value(base_p3, rng, 0.75, 1.15, 0.04, 0.2)
    p1, p2, p3 = sorted([new_p1, new_p2, new_p3])
    tpc["p1"] = p1
    tpc["p2"] = p2
    tpc["p3"] = p3
    base_lock2 = float(tpc.get("lock2", 0.01))
    tpc["lock2"] = _scale_value(base_lock2, rng, 0.7, 1.2, 0.0, 0.05)
    base_trail = float(tpc.get("trail_pct", 0.02))
    tpc["trail_pct"] = _scale_value(base_trail, rng, 0.7, 1.2, 0.005, 0.05)
    tp_cfg["trailing_ladder"] = tpc

    sp = tp_cfg.get("staged_partial", {})
    base_levels = list(sp.get("levels", []) or [])
    new_levels: List[Dict[str, float]] = []
    for lvl in base_levels:
        base_r = float(lvl.get("r_multiple", 1.0))
        base_f = float(lvl.get("fraction", 0.2))
        # lower r_multiple to increase partial trigger rate
        r_mult = _scale_value(base_r, rng, 0.6, 1.05, 0.4, 4.0)
        frac = _scale_value(base_f, rng, 0.85, 1.2, 0.05, 0.7)
        new_levels.append({"r_multiple": r_mult, "fraction": frac})
    sp["levels"] = _normalize_levels(new_levels)
    base_be = float(sp.get("move_stop_to_breakeven_after_r", 1.0))
    sp["move_stop_to_breakeven_after_r"] = _scale_value(base_be, rng, 0.7, 1.2, 0.3, 4.0)
    tp_cfg["staged_partial"] = sp

    cfg["takeprofit"] = tp_cfg
    return cfg


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


def _summarize_with_extras(accepted_df: pd.DataFrame, init_cap: float) -> Dict[str, Any]:
    summary = summarize(accepted_df, initial_capital=init_cap)
    summary["max_daily_drawdown"] = _daily_drawdown(accepted_df, init_cap)
    summary["partial_rate"] = _partial_rate(accepted_df)
    return summary


def _is_better(cand: Dict[str, Any], best: Optional[Dict[str, Any]]) -> bool:
    if best is None:
        return True
    if cand.get("meets_constraints") and not best.get("meets_constraints"):
        return True
    if not cand.get("meets_constraints") and best.get("meets_constraints"):
        return False
    if cand.get("total_pnl_usdt", 0.0) > best.get("total_pnl_usdt", 0.0):
        return True
    if cand.get("total_pnl_usdt", 0.0) == best.get("total_pnl_usdt", 0.0):
        if cand.get("partial_rate", 0.0) > best.get("partial_rate", 0.0):
            return True
        if cand.get("max_drawdown_naive", 0.0) > best.get("max_drawdown_naive", 0.0):
            return True
        if cand.get("win_rate", 0.0) > best.get("win_rate", 0.0):
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
    ap.add_argument("--out", default="artifacts/backtests/dual_tp_optimization")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trade_band", type=float, default=0.15)
    ap.add_argument("--min_partial_rate_delta", type=float, default=0.03)
    ap.add_argument("--min_drawdown_improve", type=float, default=0.05)
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

    # Baseline
    accepted_df, all_df, _ = run_backtest_dual(
        short_base, long_base, symbols, args.start, args.end, args.data, args.workers
    )
    init_cap = 0.0
    if short_base.get("backtest", {}).get("use_portfolio_sizing", False):
        init_cap = float(short_base.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
    base_summary = _summarize_with_extras(accepted_df, init_cap)
    base_trades = int(base_summary.get("trades", 0) or 0)
    base_partial = float(base_summary.get("partial_rate", 0.0))
    base_dd = float(base_summary.get("max_drawdown_naive", 0.0))

    rng = np.random.default_rng(args.seed)
    best = None
    best_short = None
    best_long = None
    rows: List[Dict[str, Any]] = []

    for i in range(1, args.trials + 1):
        short_cfg = _sample_tp_sl_params(short_base, rng)
        long_cfg = _sample_tp_sl_params(long_base, rng)
        accepted_df, all_df, _ = run_backtest_dual(
            short_cfg, long_cfg, symbols, args.start, args.end, args.data, args.workers
        )
        summary = _summarize_with_extras(accepted_df, init_cap)
        summary["trial"] = i

        trades = int(summary.get("trades", 0) or 0)
        trade_dev = abs(trades - base_trades) / max(1, base_trades)
        summary["trade_dev"] = trade_dev
        summary["partial_rate_delta"] = float(summary.get("partial_rate", 0.0)) - base_partial
        summary["drawdown_improve"] = float(summary.get("max_drawdown_naive", 0.0)) - base_dd

        summary["meets_constraints"] = (
            trade_dev <= args.trade_band
            and summary["partial_rate"] >= base_partial + args.min_partial_rate_delta
            and summary["max_drawdown_naive"] >= base_dd + args.min_drawdown_improve
            and summary["max_daily_drawdown"] >= -args.max_daily_drawdown
        )

        rows.append(summary)
        if _is_better(summary, best):
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

    summary = _summarize_with_extras(accepted_df, init_cap)
    summary.update(
        {
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "baseline": {
                "trades": base_trades,
                "partial_rate": base_partial,
                "max_drawdown_naive": base_dd,
            },
            "constraints": {
                "trade_band": args.trade_band,
                "min_partial_rate_delta": args.min_partial_rate_delta,
                "min_drawdown_improve": args.min_drawdown_improve,
                "max_daily_drawdown": args.max_daily_drawdown,
            },
            "meets_constraints": bool(best.get("meets_constraints")),
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
