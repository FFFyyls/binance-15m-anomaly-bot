import argparse
import copy
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from backtests.run_backtest import run_backtest, summarize


def _meets_drawdown(summary: Dict[str, Any], limit: float) -> bool:
    return float(summary.get("max_drawdown_naive", 0.0)) >= -limit


def _is_better(cand: Dict[str, Any], best: Dict[str, Any], dd_limit: float) -> bool:
    if best is None:
        return True
    cand_meet = _meets_drawdown(cand, dd_limit)
    best_meet = _meets_drawdown(best, dd_limit)
    cand_pnl = float(cand.get("total_pnl_usdt", 0.0))
    best_pnl = float(best.get("total_pnl_usdt", 0.0))
    cand_wr = float(cand.get("win_rate", 0.0))
    best_wr = float(best.get("win_rate", 0.0))
    cand_dd = float(cand.get("max_drawdown_naive", 0.0))
    best_dd = float(best.get("max_drawdown_naive", 0.0))
    cand_trades = int(cand.get("trades", 0) or 0)
    best_trades = int(best.get("trades", 0) or 0)

    if cand_meet and not best_meet:
        return True
    if not cand_meet and best_meet:
        return False
    if cand_meet and best_meet:
        if cand_pnl > best_pnl:
            return True
        pnl_tol = max(1e-6, abs(best_pnl) * 0.01)
        if abs(cand_pnl - best_pnl) <= pnl_tol:
            if cand_wr > best_wr:
                return True
            if abs(cand_wr - best_wr) <= 1e-6:
                if cand_trades > best_trades:
                    return True
                if cand_trades == best_trades and cand_dd > best_dd:
                    return True
        return False

    # neither meets drawdown: prefer smaller drawdown, then higher pnl, then win rate
    if cand_dd > best_dd:
        return True
    if abs(cand_dd - best_dd) < 1e-6 and cand_pnl > best_pnl:
        return True
    if abs(cand_dd - best_dd) < 1e-6 and abs(cand_pnl - best_pnl) < 1e-6 and cand_wr > best_wr:
        return True
    return False


def _sample_params(base_cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    cfg["signal"]["window_minutes"] = int(rng.choice([5, 10, 15, 20, 30]))
    cfg["signal"]["trigger_mode"] = str(rng.choice(["range", "open"]))
    cfg["signal"]["abs_r15_threshold"] = float(rng.uniform(0.03, 0.2))
    cfg["signal"]["use_abs_threshold"] = True

    cfg["volume"]["multiple"] = float(rng.uniform(1.5, 12.0))
    cfg["volume"]["baseline_lookback_windows"] = int(rng.integers(24, 241))
    cfg["volume"]["baseline_method"] = str(rng.choice(["median", "mean"]))
    cfg["volume"]["use_quote_volume"] = bool(rng.integers(0, 2))
    cfg["volume"]["use_percentile_threshold"] = bool(rng.integers(0, 2))
    cfg["volume"]["percentile"] = float(rng.uniform(80.0, 98.0))
    cfg["volume"]["percentile_lookback_windows"] = int(rng.integers(24, 241))

    cfg["risk"]["hard_stop_pct"] = float(rng.uniform(0.005, 0.05))
    cfg["risk"]["atr_enabled"] = bool(rng.integers(0, 2))
    cfg["risk"]["atr_period"] = int(rng.integers(10, 31))
    cfg["risk"]["atr_multiplier"] = float(rng.uniform(0.5, 3.0))

    p1 = float(rng.uniform(0.005, 0.04))
    p2 = float(rng.uniform(p1 + 0.003, p1 + 0.05))
    p3 = float(rng.uniform(p2 + 0.003, p2 + 0.08))
    lock2 = float(rng.uniform(0.0, min(0.03, p2)))
    trail = float(rng.uniform(0.003, 0.06))
    cfg["takeprofit"]["trailing_ladder"]["p1"] = p1
    cfg["takeprofit"]["trailing_ladder"]["p2"] = p2
    cfg["takeprofit"]["trailing_ladder"]["p3"] = p3
    cfg["takeprofit"]["trailing_ladder"]["lock2"] = lock2
    cfg["takeprofit"]["trailing_ladder"]["trail_pct"] = trail

    cfg["entry"]["min_pullback_bps"] = int(rng.integers(0, 151))
    cfg["entry"]["max_taker_slippage_bps"] = float(rng.uniform(1.0, 20.0))
    allow_long = bool(rng.integers(0, 2))
    allow_short = bool(rng.integers(0, 2))
    if not allow_long and not allow_short:
        allow_long = True
    cfg["entry"]["allow_long"] = allow_long
    cfg["entry"]["allow_short"] = allow_short
    cfg["entry"]["symmetric_close_rule"] = bool(allow_short)
    cfg["entry"]["strict_below_close_for_all"] = False

    cfg.setdefault("backtest", {})
    cfg["backtest"]["entry_offset_bps"] = int(rng.integers(0, 61))
    cfg["backtest"]["max_fill_pct_of_bar"] = float(rng.uniform(0.01, 0.2))
    cfg["backtest"]["min_fill_notional_usdt"] = float(rng.uniform(0.0, 5.0))
    cfg["backtest"]["allow_partial_fill"] = bool(rng.integers(0, 2))
    cfg["backtest"]["time_exit_price"] = str(rng.choice(["close", "open", "mid"]))
    cfg["backtest"]["liquidation_buffer_pct"] = float(rng.uniform(0.0, 0.02))

    cfg["position"]["position_pct_of_equity"] = float(rng.uniform(0.003, 0.1))
    cfg["backtest"]["position_pct_of_equity"] = cfg["position"]["position_pct_of_equity"]
    cfg["position"]["notional_usdt_per_trade"] = float(rng.uniform(5.0, 30.0))
    cfg["backtest"]["leverage"] = int(rng.integers(1, 11))
    cfg["backtest"]["max_concurrent_positions"] = int(rng.integers(2, 21))

    cfg.setdefault("time_exit", {})
    time_p1 = float(rng.uniform(0.005, 0.02))
    time_p2 = float(rng.uniform(time_p1 + 0.002, time_p1 + 0.03))
    time_h1 = int(rng.integers(6, 25))
    time_h2 = int(rng.integers(time_h1 + 4, time_h1 + 31))
    cfg["time_exit"]["profit_1"] = time_p1
    cfg["time_exit"]["profit_2"] = time_p2
    cfg["time_exit"]["hold_1_minutes"] = time_h1
    cfg["time_exit"]["hold_2_minutes"] = time_h2

    cfg.setdefault("trend_filter", {})
    cfg["trend_filter"]["enabled"] = bool(rng.integers(0, 2))
    cfg["trend_filter"]["window_minutes"] = 60
    cfg["trend_filter"]["ma_window"] = int(rng.integers(10, 41))

    cfg.setdefault("aux_filters", {})
    trend_on = bool(rng.integers(0, 2))
    cfg["aux_filters"]["trend_ma_enabled"] = trend_on
    cfg["aux_filters"]["trend_ma_bars"] = int(rng.integers(30, 181)) if trend_on else 0
    if rng.random() < 0.5:
        min_range = 0.0
    else:
        min_range = float(rng.uniform(0.003, 0.06))
    if rng.random() < 0.5:
        max_range = 0.0
    else:
        max_floor = max(min_range + 0.01, 0.02)
        max_range = float(rng.uniform(max_floor, 0.2))
    cfg["aux_filters"]["min_range_pct"] = min_range
    cfg["aux_filters"]["max_range_pct"] = max_range

    return cfg


def _collect_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    aux_cfg = cfg.get("aux_filters", {})
    time_cfg = cfg.get("time_exit", {})
    trend_cfg = cfg.get("trend_filter", {})
    return {
        "window_minutes": cfg["signal"].get("window_minutes"),
        "trigger_mode": cfg["signal"].get("trigger_mode"),
        "abs_r15_threshold": cfg["signal"]["abs_r15_threshold"],
        "volume_multiple": cfg["volume"]["multiple"],
        "baseline_lookback_windows": cfg["volume"].get("baseline_lookback_windows"),
        "baseline_method": cfg["volume"].get("baseline_method"),
        "use_quote_volume": cfg["volume"].get("use_quote_volume"),
        "use_vol_percentile": cfg["volume"].get("use_percentile_threshold"),
        "volume_percentile": cfg["volume"].get("percentile"),
        "volume_percentile_lookback": cfg["volume"].get("percentile_lookback_windows"),
        "hard_stop_pct": cfg["risk"]["hard_stop_pct"],
        "atr_enabled": cfg["risk"].get("atr_enabled"),
        "atr_period": cfg["risk"].get("atr_period"),
        "atr_multiplier": cfg["risk"].get("atr_multiplier"),
        "p1": cfg["takeprofit"]["trailing_ladder"]["p1"],
        "p2": cfg["takeprofit"]["trailing_ladder"]["p2"],
        "p3": cfg["takeprofit"]["trailing_ladder"]["p3"],
        "lock2": cfg["takeprofit"]["trailing_ladder"]["lock2"],
        "trail_pct": cfg["takeprofit"]["trailing_ladder"]["trail_pct"],
        "min_pullback_bps": cfg["entry"]["min_pullback_bps"],
        "max_taker_slippage_bps": cfg["entry"]["max_taker_slippage_bps"],
        "allow_long": cfg["entry"].get("allow_long", True),
        "allow_short": cfg["entry"].get("allow_short", True),
        "entry_offset_bps": cfg["backtest"]["entry_offset_bps"],
        "max_fill_pct_of_bar": cfg["backtest"]["max_fill_pct_of_bar"],
        "min_fill_notional_usdt": cfg["backtest"].get("min_fill_notional_usdt", 0.0),
        "allow_partial_fill": cfg["backtest"].get("allow_partial_fill", True),
        "time_exit_price": cfg["backtest"].get("time_exit_price", "close"),
        "liquidation_buffer_pct": cfg["backtest"].get("liquidation_buffer_pct", 0.0),
        "position_pct_of_equity": cfg["position"]["position_pct_of_equity"],
        "notional_usdt_per_trade": cfg["position"].get("notional_usdt_per_trade"),
        "leverage": cfg["backtest"]["leverage"],
        "max_concurrent_positions": cfg["backtest"]["max_concurrent_positions"],
        "time_exit_p1": time_cfg.get("profit_1"),
        "time_exit_p2": time_cfg.get("profit_2"),
        "time_exit_h1": time_cfg.get("hold_1_minutes"),
        "time_exit_h2": time_cfg.get("hold_2_minutes"),
        "trend_filter_enabled": trend_cfg.get("enabled"),
        "trend_filter_window": trend_cfg.get("window_minutes"),
        "trend_filter_ma_window": trend_cfg.get("ma_window"),
        "trend_ma_enabled": aux_cfg.get("trend_ma_enabled"),
        "trend_ma_bars": aux_cfg.get("trend_ma_bars"),
        "min_range_pct": aux_cfg.get("min_range_pct"),
        "max_range_pct": aux_cfg.get("max_range_pct"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols_file", default="configs/symbols/symbols_all_data_auto.txt")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--out", default="artifacts/backtests/parameter_optimization")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_drawdown", type=float, default=0.3)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    base_cfg.setdefault("backtest", {})
    base_cfg.setdefault("signal", {})
    base_cfg.setdefault("volume", {})
    base_cfg.setdefault("risk", {})
    base_cfg.setdefault("entry", {})
    base_cfg.setdefault("position", {})
    base_cfg.setdefault("time_exit", {})
    base_cfg.setdefault("trend_filter", {})
    base_cfg.setdefault("aux_filters", {})
    base_cfg.setdefault("takeprofit", {})
    base_cfg.setdefault("takeprofit", {}).setdefault("trailing_ladder", {})
    base_cfg["signal"].setdefault("trigger_mode", "range")

    with open(args.symbols_file, "r", encoding="utf-8") as f:
        symbols = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]

    os.makedirs(args.out, exist_ok=True)
    trials_path = os.path.join(args.out, "trials.csv")
    if os.path.exists(trials_path):
        os.remove(trials_path)

    rng = np.random.default_rng(args.seed)
    best = None
    best_cfg = None
    rows: List[Dict[str, Any]] = []

    for i in range(1, args.trials + 1):
        cfg = _sample_params(base_cfg, rng)
        accepted_df, all_df, _ = run_backtest(cfg, symbols, args.start, args.end, args.data, args.workers)
        init_cap = 0.0
        if cfg.get("backtest", {}).get("use_portfolio_sizing", False):
            init_cap = float(cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
        summary = summarize(accepted_df, initial_capital=init_cap)
        summary["trial"] = i
        summary["meets_drawdown"] = _meets_drawdown(summary, args.max_drawdown)
        summary.update(_collect_params(cfg))
        rows.append(summary)

        if _is_better(summary, best, args.max_drawdown):
            best = summary
            best_cfg = cfg

        if i % 5 == 0 or i == args.trials:
            df_out = pd.DataFrame(rows)
            if not df_out.empty:
                write_header = not os.path.exists(trials_path)
                df_out.to_csv(trials_path, index=False, mode="a", header=write_header)
            rows = []

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

    with open(os.path.join(best_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(best_dir, "best_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, allow_unicode=True, sort_keys=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
