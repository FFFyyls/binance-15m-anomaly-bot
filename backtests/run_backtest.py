import argparse
import copy
import json
import os
import re
from collections import OrderedDict
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    qty: float
    pnl_usdt: float
    reason: str


def _read_csv(path: str, usecols: List[str], dtype: dict) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype)
    except ValueError:
        df = pd.read_csv(path, usecols=usecols, dtype=str, on_bad_lines="skip")
        if "open_time" in df.columns:
            df = df[df["open_time"] != "open_time"]
        for c in usecols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["open_time"])


def _bar_subdirs(bar_seconds: int) -> List[str]:
    subdirs = [f"{bar_seconds}s"]
    if bar_seconds == 60:
        subdirs.insert(0, "1m")
    return subdirs


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.csv$")
_BAR_CACHE: "OrderedDict[Tuple[str, str, str, str, int], pd.DataFrame]" = OrderedDict()
try:
    _BAR_CACHE_MAX = int(os.getenv("BACKTEST_BAR_CACHE_SIZE", "8") or 8)
except (TypeError, ValueError):
    _BAR_CACHE_MAX = 8


def _cache_get(key):
    if _BAR_CACHE_MAX <= 0:
        return None
    if key not in _BAR_CACHE:
        return None
    val = _BAR_CACHE.pop(key)
    _BAR_CACHE[key] = val
    return val


def _cache_put(key, val):
    if _BAR_CACHE_MAX <= 0:
        return
    _BAR_CACHE[key] = val
    if len(_BAR_CACHE) > _BAR_CACHE_MAX:
        _BAR_CACHE.popitem(last=False)


@lru_cache(maxsize=4096)
def _symbol_date_range(symbol: str, data_dir: str, bar_seconds: int) -> Optional[Tuple[str, str]]:
    subdirs = _bar_subdirs(bar_seconds)
    dates: List[str] = []
    for subdir in subdirs:
        root = os.path.join(data_dir, symbol, subdir)
        if not os.path.isdir(root):
            continue
        try:
            for name in os.listdir(root):
                if _DATE_RE.match(name):
                    dates.append(name[:-4])
        except FileNotFoundError:
            continue
    if not dates:
        return None
    return min(dates), max(dates)


def load_bars(symbol: str, start: str, end: str, data_dir: str, bar_seconds: int) -> pd.DataFrame:
    cache_key = (symbol, start, end, data_dir, bar_seconds)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    frames = []
    usecols = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    dtype = {
        "open_time": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_volume": "float64",
    }
    subdirs = _bar_subdirs(bar_seconds)
    for day in days:
        for subdir in subdirs:
            path = os.path.join(data_dir, symbol, subdir, f"{day}.csv")
            if os.path.exists(path):
                frames.append(_read_csv(path, usecols, dtype))
                break
    if not frames:
        raise FileNotFoundError(f"No data for {symbol} in {data_dir} within {start}..{end}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("ts").set_index("ts")
    df = df[["open", "high", "low", "close", "volume", "quote_volume"]].dropna()
    _cache_put(cache_key, df)
    return df


def resample_window(df1m: pd.DataFrame, window_minutes: int, vol_col: str) -> pd.DataFrame:
    rule = f"{window_minutes}min"
    o = df1m["open"].resample(rule).first()
    h = df1m["high"].resample(rule).max()
    l = df1m["low"].resample(rule).min()
    c = df1m["close"].resample(rule).last()
    vv = df1m[vol_col].resample(rule).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, vol_col: vv}).dropna()
    out["ret"] = out["close"].pct_change()
    return out


def _taker_slippage_bps(close15: float, limit_px: float, mid_px: float, side: str) -> Tuple[bool, float]:
    if close15 <= 0 or mid_px <= 0:
        return False, 0.0
    if side == "BUY":
        if limit_px < mid_px:
            return False, 0.0
        bps = (mid_px / close15 - 1.0) * 10000.0
        return True, max(0.0, bps)
    if limit_px > mid_px:
        return False, 0.0
    bps = (1.0 - mid_px / close15) * 10000.0
    return True, max(0.0, bps)


def backtest_symbol(symbol: str, df1m: pd.DataFrame, cfg: dict) -> List[Trade]:
    signal = cfg["signal"]
    abs_thr = float(signal["abs_r15_threshold"])
    use_abs = bool(signal.get("use_abs_threshold", True))
    window_minutes = int(signal.get("window_minutes", 15))
    trigger_mode = str(signal.get("trigger_mode", "range")).lower()
    dyn_cfg = signal.get("dynamic_abs_threshold", {})
    dyn_enabled = bool(dyn_cfg.get("enabled", False))
    dyn_multiplier = float(dyn_cfg.get("multiplier", 1.0))
    dyn_lookback_windows = int(dyn_cfg.get("lookback_windows", 96))
    dyn_update_minutes = int(dyn_cfg.get("update_minutes", 60))
    dyn_vol_window_minutes = int(dyn_cfg.get("vol_window_minutes", 15))
    dyn_max_threshold = float(dyn_cfg.get("max_threshold", 0.0))
    bb_cfg = cfg.get("bollinger", {})
    bb_enabled = bool(bb_cfg.get("enabled", False))
    bb_window_minutes = int(bb_cfg.get("window_minutes", 20))
    bb_num_std = float(bb_cfg.get("num_std", 2.5))
    bb_mode = str(bb_cfg.get("mode", "or")).lower()
    bar_seconds = int(cfg.get("backtest", {}).get("data_interval_seconds", 60))
    bars_per_window = int(window_minutes * 60 / bar_seconds)
    if bars_per_window <= 0 or window_minutes * 60 % bar_seconds != 0:
        raise ValueError("window_minutes must be divisible by data_interval_seconds")
    bb_window_bars = 0
    if bb_enabled:
        bb_window_bars = int(bb_window_minutes * 60 / bar_seconds)
        if bb_window_bars <= 1 or bb_window_minutes * 60 % bar_seconds != 0:
            raise ValueError("bollinger window_minutes must be divisible by data_interval_seconds")

    volume_cfg = cfg["volume"]
    vol_mult = float(volume_cfg["multiple"])
    baseline_lookback = int(volume_cfg.get("baseline_lookback_windows", volume_cfg.get("baseline_lookback_15m", 96)))
    baseline_method = str(volume_cfg.get("baseline_method", "median")).lower()
    use_quote = bool(volume_cfg.get("use_quote_volume", True))
    vol_col = "quote_volume" if use_quote else "volume"
    use_vol_percentile = bool(volume_cfg.get("use_percentile_threshold", False))
    vol_percentile = float(volume_cfg.get("percentile", 90.0))
    vol_lookback = int(volume_cfg.get("percentile_lookback_windows", baseline_lookback))

    risk_cfg = cfg["risk"]
    hard_stop = float(risk_cfg["hard_stop_pct"])
    atr_enabled = bool(risk_cfg.get("atr_enabled", False))
    atr_period = int(risk_cfg.get("atr_period", 14))
    atr_multiplier = float(risk_cfg.get("atr_multiplier", 1.5))
    use_mark_price_for_risk = bool(risk_cfg.get("use_mark_price_for_risk", False))

    bt = cfg.get("backtest", {})
    maker_fee = float(bt.get("maker_fee", 0.0002))
    taker_fee = float(bt.get("taker_fee", 0.0004))
    slippage_bps = float(bt.get("slippage_bps", 5.0))
    slippage_bps_base = float(bt.get("slippage_bps_base", slippage_bps))
    slippage_bps_k = float(bt.get("slippage_bps_k", 0.0))
    slippage_bps_min = float(bt.get("slippage_bps_min", 0.0))
    slippage_bps_max = float(bt.get("slippage_bps_max", 1e9))
    slippage_bps_vol_factor = float(bt.get("slippage_bps_vol_factor", 0.0))
    latency_seconds = float(bt.get("latency_seconds", 0.0))
    funding_rate_8h = float(bt.get("funding_rate_8h", 0.0001))
    entry_offset_bps = float(bt.get("entry_offset_bps", 0.0))
    max_fill_pct = float(bt.get("max_fill_pct_of_bar", 1.0))
    min_fill_notional = float(bt.get("min_fill_notional_usdt", 0.0))
    allow_partial = bool(bt.get("allow_partial_fill", True))
    leverage = float(bt.get("leverage", cfg.get("position", {}).get("leverage", 1)))
    liq_buffer = float(bt.get("liquidation_buffer_pct", 0.0))
    use_mark_price_for_risk = bool(bt.get("use_mark_price_for_risk", use_mark_price_for_risk))
    max_stop_pct = 0.0
    if leverage > 0:
        max_stop_pct = max(0.0, 1.0 / leverage - liq_buffer)
    effective_hard_stop = hard_stop
    if max_stop_pct > 0:
        effective_hard_stop = min(hard_stop, max_stop_pct)
    entry_offset = entry_offset_bps / 10000.0
    entry_tol = slippage_bps_base / 10000.0

    entry_cfg = cfg["entry"]
    symmetric = bool(entry_cfg["symmetric_close_rule"])
    strict = bool(entry_cfg["strict_below_close_for_all"])
    pullback = float(entry_cfg["min_pullback_bps"]) / 10000.0
    allow_long = bool(entry_cfg.get("allow_long", True))
    allow_short = bool(entry_cfg.get("allow_short", True))
    max_taker_slippage_bps = float(entry_cfg.get("max_taker_slippage_bps", slippage_bps_base))
    notional = float(cfg["position"]["notional_usdt_per_trade"])

    time_exit_price = str(bt.get("time_exit_price", "close")).lower()
    time_exit_cfg = cfg.get("time_exit", {})
    time_exit_p1 = float(time_exit_cfg.get("profit_1", 0.01))
    time_exit_p2 = float(time_exit_cfg.get("profit_2", 0.02))
    time_exit_h1 = int(time_exit_cfg.get("hold_1_minutes", 12))
    time_exit_h2 = int(time_exit_cfg.get("hold_2_minutes", 20))

    aux_cfg = cfg.get("aux_filters", {})
    trend_ma_enabled = bool(aux_cfg.get("trend_ma_enabled", False))
    trend_ma_bars = int(aux_cfg.get("trend_ma_bars", 0) or 0)
    min_range_pct = float(aux_cfg.get("min_range_pct", 0.0) or 0.0)
    max_range_pct = float(aux_cfg.get("max_range_pct", 0.0) or 0.0)

    trend_cfg = cfg.get("trend_filter", {})
    trend_enabled = bool(trend_cfg.get("enabled", False))
    trend_minutes = int(trend_cfg.get("window_minutes", 60))
    trend_ma_window = int(trend_cfg.get("ma_window", 20))

    tp_cfg = cfg.get("takeprofit", {})
    tp_mode = str(tp_cfg.get("mode", "trailing_ladder")).lower()
    tp_partial_cfg = tp_cfg.get("staged_partial", {})
    tp_levels = list(tp_partial_cfg.get("levels", []) or [])
    tp_move_be_r = float(tp_partial_cfg.get("move_stop_to_breakeven_after_r", 1.0))
    tpc = tp_cfg.get("trailing_ladder", {})
    p1 = float(tpc.get("p1", 0.0))
    p2 = float(tpc.get("p2", 0.0))
    lock2 = float(tpc.get("lock2", 0.0))
    p3 = float(tpc.get("p3", 0.0))
    trail = float(tpc.get("trail_pct", 0.0))
    tp_partial_modes = {"staged_partial", "hybrid", "staged_partial_trailing", "staged_partial+trailing"}
    tp_trailing_modes = {"trailing_ladder", "hybrid", "staged_partial_trailing", "staged_partial+trailing"}
    use_partial = tp_mode in tp_partial_modes
    use_trailing = tp_mode in tp_trailing_modes
    if tp_mode not in tp_partial_modes | tp_trailing_modes:
        use_trailing = True

    dfw = resample_window(df1m, window_minutes, vol_col)
    if baseline_method == "mean":
        baseline = dfw[vol_col].rolling(baseline_lookback).mean()
    else:
        baseline = dfw[vol_col].rolling(baseline_lookback).median()
    dfw["baseline_qv"] = baseline.shift(1)
    if use_vol_percentile:
        q = min(max(vol_percentile / 100.0, 0.0), 1.0)
        dfw["vol_threshold"] = dfw[vol_col].rolling(vol_lookback).quantile(q).shift(1)
    dyn_thr_series = None
    if dyn_enabled:
        if dyn_vol_window_minutes == window_minutes:
            dyn_dfw = dfw
        else:
            dyn_dfw = resample_window(df1m, dyn_vol_window_minutes, vol_col)
        vol15 = dyn_dfw["ret"].abs()
        dyn_vol = vol15.rolling(dyn_lookback_windows).mean().shift(1)
        if dyn_update_minutes > 0:
            rule = f"{dyn_update_minutes}min"
            dyn_vol = dyn_vol.resample(rule).last()
        dyn_thr_series = dyn_vol * dyn_multiplier

    df = df1m.copy()
    df["volN"] = df[vol_col].rolling(bars_per_window).sum()
    df["openN"] = df["open"].shift(bars_per_window - 1)
    df["lowN"] = df["low"].rolling(bars_per_window).min()
    df["highN"] = df["high"].rolling(bars_per_window).max()
    if trend_ma_enabled and trend_ma_bars > 0:
        df["trend_ma"] = df["close"].rolling(trend_ma_bars).mean()
    if bb_enabled:
        df["bb_mid"] = df["close"].rolling(bb_window_bars).mean()
        df["bb_std"] = df["close"].rolling(bb_window_bars).std(ddof=0)
        df["bb_upper"] = df["bb_mid"] + bb_num_std * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - bb_num_std * df["bb_std"]
    if atr_enabled and atr_period > 0:
        df["atr"] = (df["high"] - df["low"]).rolling(atr_period).mean()
        df["atr"] = df["atr"].ffill()
    df = df.dropna()

    trend_series = None
    if trend_enabled and trend_minutes > 0 and trend_ma_window > 0:
        df_trend = resample_window(df1m, trend_minutes, vol_col)
        df_trend["trend_ma"] = df_trend["close"].rolling(trend_ma_window).mean()
        trend_series = df_trend["trend_ma"].reindex(df.index, method="ffill")

    align_cols = ["baseline_qv"]
    if "vol_threshold" in dfw.columns:
        align_cols.append("vol_threshold")
    aligned = dfw[align_cols].reindex(df.index, method="ffill")
    dyn_thr = dyn_thr_series.reindex(df.index, method="ffill") if dyn_thr_series is not None else None

    idx = df.index.to_numpy()
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    bar_vol = df[vol_col].to_numpy()
    volN = df["volN"].to_numpy()
    bN = aligned["baseline_qv"].to_numpy()
    vol_thr = aligned["vol_threshold"].to_numpy() if "vol_threshold" in aligned.columns else None
    openN = df["openN"].to_numpy()
    lowN = df["lowN"].to_numpy()
    highN = df["highN"].to_numpy()
    bb_upper = df["bb_upper"].to_numpy() if bb_enabled else None
    bb_lower = df["bb_lower"].to_numpy() if bb_enabled else None
    trend_ma = df["trend_ma"].to_numpy() if "trend_ma" in df.columns else None
    atr_arr = df["atr"].to_numpy() if "atr" in df.columns else None
    trend_ma_1h = trend_series.to_numpy() if trend_series is not None else None
    dyn_thr_arr = dyn_thr.to_numpy() if dyn_thr is not None else None

    trades: List[Trade] = []
    in_pos = False
    side = ""
    entry_px = qty = 0.0
    stop_dyn = peak = 0.0
    entry_time = ""
    entry_fee = 0.0
    entry_i = 0
    entry_qty = 0.0
    entry_fee_per_qty = 0.0
    pnl_accum = 0.0
    exit_notional = 0.0
    exit_qty_sum = 0.0
    tp_state = 0
    partial_hit = False
    last_ts_loop: Optional[pd.Timestamp] = None

    def apply_funding(current_ts: pd.Timestamp, prev_ts: Optional[pd.Timestamp], position_qty: float, current_px: float, side: str) -> float:
        if prev_ts is None or position_qty <= 0:
            return 0.0
        # Funding cycles at 0, 8, 16 UTC
        funding_hours = [0, 8, 16]
        funding_paid = 0.0
        for h in funding_hours:
            # Check if we crossed a funding hour
            # Create a timestamp for the funding hour on the current day
            funding_ts = current_ts.normalize() + pd.Timedelta(hours=h)
            if prev_ts < funding_ts <= current_ts:
                notional = position_qty * current_px
                rate = funding_rate_8h if side == "BUY" else -funding_rate_8h
                funding_paid += notional * rate
        return funding_paid

    def dyn_slippage_bps(h: float, l: float, ref_px: float, trade_notional: float = 0.0, bar_volume_usdt: float = 0.0) -> float:
        if ref_px <= 0:
            return slippage_bps_base
        range_pct = max(0.0, (h - l) / ref_px)
        bps = slippage_bps_base + slippage_bps_k * (range_pct * 10000.0)
        if slippage_bps_vol_factor > 0 and bar_volume_usdt > 0:
            bps += slippage_bps_vol_factor * (trade_notional / bar_volume_usdt) * 10000.0
        return min(max(bps, slippage_bps_min), slippage_bps_max)

    def bar_quote_notional(i: int) -> float:
        if vol_col == "quote_volume":
            return float(bar_vol[i])
        return float(bar_vol[i]) * float(close[i])


    for i in range(0, len(idx) - 1):
        curr_ts = pd.Timestamp(idx[i])
        t_next = pd.Timestamp(idx[i + 1])

        if in_pos:
            funding = apply_funding(curr_ts, last_ts_loop, qty, float(close[i]), side)
            pnl_accum -= funding

        last_ts_loop = curr_ts

        if not in_pos:
            v15 = float(volN[i])
            b15 = float(bN[i])
            open15 = float(openN[i])
            low15 = float(lowN[i])
            high15 = float(highN[i])

            if not use_vol_percentile:
                if np.isnan(b15) or b15 <= 0:
                    continue
            if trigger_mode in ("range", "legacy", "lowhigh"):
                if np.isnan(low15) or low15 <= 0 or np.isnan(high15) or high15 <= 0:
                    continue
            else:
                if np.isnan(open15) or open15 <= 0:
                    continue

            if use_vol_percentile:
                if vol_thr is None:
                    continue
                vthr = float(vol_thr[i])
                if np.isnan(vthr) or vthr <= 0:
                    continue
                if v15 < vthr:
                    continue
            else:
                if v15 < vol_mult * b15:
                    continue

            thr = abs_thr
            if dyn_thr_arr is not None:
                dyn_thr = float(dyn_thr_arr[i])
                if np.isnan(dyn_thr) or dyn_thr <= 0:
                    thr = abs_thr
                elif dyn_max_threshold > 0 and dyn_thr > dyn_max_threshold:
                    thr = abs_thr
                else:
                    thr = dyn_thr
            mid = float(close[i])
            if (min_range_pct > 0 or max_range_pct > 0) and low15 > 0 and high15 > 0:
                range_pct = high15 / low15 - 1.0
                if min_range_pct > 0 and range_pct < min_range_pct:
                    continue
                if max_range_pct > 0 and range_pct > max_range_pct:
                    continue
            if trigger_mode in ("range", "legacy", "lowhigh"):
                move_long = (mid / low15) - 1.0
                move_short = (high15 / mid) - 1.0
                hit_long = use_abs and move_long >= thr
                hit_short = use_abs and move_short >= thr
            else:
                move = (mid / open15) - 1.0
                hit_long = use_abs and move >= thr
                hit_short = use_abs and move <= -thr
            if bb_enabled and bb_upper is not None and bb_lower is not None:
                upper = float(bb_upper[i])
                lower = float(bb_lower[i])
                bb_hit_long = not np.isnan(upper) and mid > upper
                bb_hit_short = not np.isnan(lower) and mid < lower
                if bb_mode in ("filter", "and"):
                    hit_long = hit_long and bb_hit_long
                    hit_short = hit_short and bb_hit_short
                elif bb_mode in ("only", "bb_only"):
                    hit_long = bb_hit_long
                    hit_short = bb_hit_short
                else:
                    hit_long = hit_long or bb_hit_long
                    hit_short = hit_short or bb_hit_short
            if not (hit_long or hit_short):
                continue

            if hit_long and hit_short:
                if trigger_mode in ("range", "legacy", "lowhigh"):
                    side = "BUY" if move_long >= move_short else "SELL"
                else:
                    side = "BUY" if move >= 0 else "SELL"
            elif hit_long:
                side = "BUY"
            else:
                side = "SELL"
            close15 = mid
            if trend_ma_1h is not None:
                ma_1h = float(trend_ma_1h[i])
                if np.isnan(ma_1h) or ma_1h <= 0:
                    continue
                if side == "BUY" and mid < ma_1h:
                    continue
                if side == "SELL" and mid > ma_1h:
                    continue
            if trend_ma is not None:
                ma = float(trend_ma[i])
                if not np.isnan(ma) and ma > 0:
                    if side == "BUY" and mid < ma:
                        continue
                    if side == "SELL" and mid > ma:
                        continue

            if side == "BUY":
                if not allow_long:
                    continue
                base = mid * (1.0 - entry_offset)
                cap = close15 * (1.0 + entry_tol)
                limit_px = min(base, cap)
                if pullback > 0:
                    limit_px = min(limit_px, close15 * (1.0 - pullback))
                if limit_px <= 0:
                    continue
                marketable, taker_bps = _taker_slippage_bps(close15, limit_px, mid, side)
                if marketable and taker_bps > max_taker_slippage_bps:
                    continue

                # Latency check: can we fill in i+1?
                if latency_seconds > 0:
                    fill_i = i + 1
                    while fill_i < len(idx) and (pd.Timestamp(idx[fill_i]) - curr_ts).total_seconds() < latency_seconds:
                        fill_i += 1
                    if fill_i >= len(idx):
                        continue
                else:
                    fill_i = i + 1

                if float(low[fill_i]) <= limit_px:
                    bar_vol_usdt = float(bar_vol[fill_i]) * float(close[fill_i]) if vol_col != "quote_volume" else float(bar_vol[fill_i])
                    bps = dyn_slippage_bps(float(high[fill_i]), float(low[fill_i]), float(close[fill_i]), notional, bar_vol_usdt)
                    slp = bps / 10000.0
                    entry_px = limit_px * (1.0 + slp)
                    max_fill = max_fill_pct * bar_quote_notional(fill_i)
                    fill_notional = min(notional, max_fill)
                    if not allow_partial and fill_notional < notional:
                        continue
                    if fill_notional < min_fill_notional:
                        continue
                    qty = fill_notional / entry_px
                    entry_fee = maker_fee * fill_notional
                    peak = entry_px
                    atr_val = float(atr_arr[fill_i]) if atr_arr is not None else 0.0
                    stop_dist = entry_px * effective_hard_stop
                    if atr_enabled and atr_val > 0:
                        stop_dist = atr_multiplier * atr_val
                        if max_stop_pct > 0:
                            stop_dist = min(stop_dist, entry_px * max_stop_pct)
                    stop_dyn = entry_px - stop_dist
                    in_pos = True
                    entry_time = pd.Timestamp(idx[fill_i]).isoformat()
                    entry_i = fill_i
                    entry_qty = qty
                    entry_fee_per_qty = entry_fee / max(1e-12, qty)
                    pnl_accum = 0.0
                    exit_notional = 0.0
                    exit_qty_sum = 0.0
                    tp_state = 0
                    partial_hit = False
            else:
                if strict:
                    continue
                if not allow_short:
                    continue
                if symmetric:
                    base = mid * (1.0 + entry_offset)
                    floor = close15 * (1.0 - entry_tol)
                    limit_px = max(base, floor)
                    if pullback > 0:
                        limit_px = max(limit_px, close15 * (1.0 + pullback))
                    if limit_px <= 0:
                        continue
                    marketable, taker_bps = _taker_slippage_bps(close15, limit_px, mid, side)
                    if marketable and taker_bps > max_taker_slippage_bps:
                        continue

                    # Latency check
                    if latency_seconds > 0:
                        fill_i = i + 1
                        while fill_i < len(idx) and (pd.Timestamp(idx[fill_i]) - curr_ts).total_seconds() < latency_seconds:
                            fill_i += 1
                        if fill_i >= len(idx):
                            continue
                    else:
                        fill_i = i + 1

                    if float(high[fill_i]) >= limit_px:
                        bar_vol_usdt = float(bar_vol[fill_i]) * float(close[fill_i]) if vol_col != "quote_volume" else float(bar_vol[fill_i])
                        bps = dyn_slippage_bps(float(high[fill_i]), float(low[fill_i]), float(close[fill_i]), notional, bar_vol_usdt)
                        slp = bps / 10000.0
                        entry_px = limit_px * (1.0 - slp)
                        max_fill = max_fill_pct * bar_quote_notional(fill_i)
                        fill_notional = min(notional, max_fill)
                        if not allow_partial and fill_notional < notional:
                            continue
                        if fill_notional < min_fill_notional:
                            continue
                        qty = fill_notional / entry_px
                        entry_fee = maker_fee * fill_notional
                        peak = entry_px
                        atr_val = float(atr_arr[fill_i]) if atr_arr is not None else 0.0
                        stop_dist = entry_px * effective_hard_stop
                        if atr_enabled and atr_val > 0:
                            stop_dist = atr_multiplier * atr_val
                            if max_stop_pct > 0:
                                stop_dist = min(stop_dist, entry_px * max_stop_pct)
                        stop_dyn = entry_px + stop_dist
                        in_pos = True
                        entry_time = pd.Timestamp(idx[fill_i]).isoformat()
                        entry_i = fill_i
                        entry_qty = qty
                        entry_fee_per_qty = entry_fee / max(1e-12, qty)
                        pnl_accum = 0.0
                        exit_notional = 0.0
                        exit_qty_sum = 0.0
                        tp_state = 0
                        partial_hit = False
                else:
                    continue
            continue

        high_next = float(high[i + 1])
        low_next = float(low[i + 1])
        close_next = float(close[i + 1])
        bar_vol_usdt_next = float(bar_vol[i + 1]) * float(close[i + 1]) if vol_col != "quote_volume" else float(bar_vol[i + 1])

        if side == "BUY":
            # 1. Update peak first? No, use previous peak for SL check to be conservative.
            # However, if price goes UP then DOWN in same bar, we might hit TP then SL.
            # If price goes DOWN then UP, we might hit SL first.
            # Conservative: check SL first.

            sl_hit = low_next <= stop_dyn

            # 2. Check TP levels (Partial)
            if use_partial and tp_levels:
                if tp_move_be_r > 0:
                    be_px = entry_px * (1.0 + tp_move_be_r * hard_stop)
                    if high_next >= be_px:
                        stop_dyn = max(stop_dyn, entry_px)

                while tp_state < len(tp_levels):
                    lvl = tp_levels[tp_state]
                    r_mult = float(lvl.get("r_multiple", 0.0) or 0.0)
                    frac = float(lvl.get("fraction", 0.0) or 0.0)
                    if r_mult <= 0 or frac <= 0:
                        tp_state += 1
                        continue
                    tgt = entry_px * (1.0 + r_mult * hard_stop)
                    if high_next < tgt:
                        break

                    # Partial TP Hit!
                    qty_exit = min(qty, entry_qty * frac)
                    if qty_exit > 1e-12:
                        bps = dyn_slippage_bps(high_next, low_next, close_next, qty_exit * tgt, bar_vol_usdt_next)
                        slp = bps / 10000.0
                        exit_px = tgt * (1.0 - slp)
                        fee = taker_fee * (qty_exit * exit_px)
                        entry_fee_piece = entry_fee_per_qty * qty_exit
                        pnl_piece = (exit_px - entry_px) * qty_exit - entry_fee_piece - fee
                        pnl_accum += pnl_piece
                        exit_notional += qty_exit * exit_px
                        exit_qty_sum += qty_exit
                        qty -= qty_exit
                        partial_hit = True
                    tp_state += 1
                    if qty <= 1e-12:
                        avg_exit = exit_notional / exit_qty_sum if exit_qty_sum > 0 else exit_px
                        trades.append(Trade(symbol, side, entry_time, entry_px, t_next.isoformat(), avg_exit, entry_qty, float(pnl_accum), "TP_PARTIAL"))
                        in_pos = False
                        break

            if not in_pos: continue

            # 3. Check Trailing / Hard Stop
            if use_trailing:
                prof = max(peak, high_next) / entry_px - 1.0
                if prof >= p1:
                    stop_dyn = max(stop_dyn, entry_px)
                if prof >= p2:
                    stop_dyn = max(stop_dyn, entry_px * (1.0 + lock2))
                if prof >= p3:
                    # trail from peak
                    stop_dyn = max(stop_dyn, max(peak, high_next) * (1.0 - trail))

            if low_next <= stop_dyn:
                bps = dyn_slippage_bps(high_next, low_next, close_next, qty * stop_dyn, bar_vol_usdt_next)
                slp = bps / 10000.0
                exit_px = stop_dyn * (1.0 - slp)
                fee = taker_fee * (qty * exit_px)
                entry_fee_piece = entry_fee_per_qty * qty
                pnl = (exit_px - entry_px) * qty - entry_fee_piece - fee
                pnl_total = pnl_accum + pnl
                exit_notional += qty * exit_px
                exit_qty_sum += qty
                avg_exit = exit_notional / exit_qty_sum if exit_qty_sum > 0 else exit_px
                reason = "STOP/TP+PARTIAL" if partial_hit else "STOP/TP"
                trades.append(Trade(symbol, side, entry_time, entry_px, t_next.isoformat(), avg_exit, entry_qty, float(pnl_total), reason))
                in_pos = False

            if in_pos:
                peak = max(peak, high_next)
        else:
            # Short position exit checks
            # 1. Conservative SL check
            sl_hit = high_next >= stop_dyn

            # 2. Check TP levels
            if use_partial and tp_levels:
                if tp_move_be_r > 0:
                    be_px = entry_px * (1.0 - tp_move_be_r * hard_stop)
                    if low_next <= be_px:
                        stop_dyn = min(stop_dyn, entry_px)

                while tp_state < len(tp_levels):
                    lvl = tp_levels[tp_state]
                    r_mult = float(lvl.get("r_multiple", 0.0) or 0.0)
                    frac = float(lvl.get("fraction", 0.0) or 0.0)
                    if r_mult <= 0 or frac <= 0:
                        tp_state += 1
                        continue
                    tgt = entry_px * (1.0 - r_mult * hard_stop)
                    if low_next > tgt:
                        break

                    qty_exit = min(qty, entry_qty * frac)
                    if qty_exit > 1e-12:
                        bps = dyn_slippage_bps(high_next, low_next, close_next, qty_exit * tgt, bar_vol_usdt_next)
                        slp = bps / 10000.0
                        exit_px = tgt * (1.0 + slp)
                        fee = taker_fee * (qty_exit * exit_px)
                        entry_fee_piece = entry_fee_per_qty * qty_exit
                        pnl_piece = (entry_px - exit_px) * qty_exit - entry_fee_piece - fee
                        pnl_accum += pnl_piece
                        exit_notional += qty_exit * exit_px
                        exit_qty_sum += qty_exit
                        qty -= qty_exit
                        partial_hit = True
                    tp_state += 1
                    if qty <= 1e-12:
                        avg_exit = exit_notional / exit_qty_sum if exit_qty_sum > 0 else exit_px
                        trades.append(Trade(symbol, side, entry_time, entry_px, t_next.isoformat(), avg_exit, entry_qty, float(pnl_accum), "TP_PARTIAL"))
                        in_pos = False
                        break

            if not in_pos: continue

            # 3. Trailing / Hard Stop
            if use_trailing:
                prof = entry_px / min(peak, low_next) - 1.0
                if prof >= p1:
                    stop_dyn = min(stop_dyn, entry_px)
                if prof >= p2:
                    stop_dyn = min(stop_dyn, entry_px * (1.0 - lock2))
                if prof >= p3:
                    stop_dyn = min(stop_dyn, min(peak, low_next) * (1.0 + trail))

            if high_next >= stop_dyn:
                bps = dyn_slippage_bps(high_next, low_next, close_next, qty * stop_dyn, bar_vol_usdt_next)
                slp = bps / 10000.0
                exit_px = stop_dyn * (1.0 + slp)
                fee = taker_fee * (qty * exit_px)
                entry_fee_piece = entry_fee_per_qty * qty
                pnl = (entry_px - exit_px) * qty - entry_fee_piece - fee
                pnl_total = pnl_accum + pnl
                exit_notional += qty * exit_px
                exit_qty_sum += qty
                avg_exit = exit_notional / exit_qty_sum if exit_qty_sum > 0 else exit_px
                reason = "STOP/TP+PARTIAL" if partial_hit else "STOP/TP"
                trades.append(Trade(symbol, side, entry_time, entry_px, t_next.isoformat(), avg_exit, entry_qty, float(pnl_total), reason))
                in_pos = False

            if in_pos:
                peak = min(peak, low_next)

        hold_bars = 0
        px_cur = float(close[i + 1])
        if entry_px > 0 and px_cur > 0 and bar_seconds > 0:
            if side == "BUY":
                prof = px_cur / entry_px - 1.0
            else:
                prof = entry_px / px_cur - 1.0
            if prof < time_exit_p1:
                hold_bars = int(round(time_exit_h1 * 60 / bar_seconds))
            elif prof < time_exit_p2:
                hold_bars = int(round(time_exit_h2 * 60 / bar_seconds))
        if in_pos and hold_bars > 0 and (i + 1 - entry_i) >= hold_bars:
            if time_exit_price == "open":
                exit_px_raw = float(open_[i + 1])
            elif time_exit_price == "mid":
                exit_px_raw = (float(high[i + 1]) + float(low[i + 1])) / 2.0
            else:
                exit_px_raw = float(close[i + 1])

            bps = dyn_slippage_bps(high_next, low_next, close_next, qty * exit_px_raw, bar_vol_usdt_next)
            slp = bps / 10000.0
            exit_px = exit_px_raw * (1.0 - slp) if side == "BUY" else exit_px_raw * (1.0 + slp)
            fee = taker_fee * (qty * exit_px)
            entry_fee_piece = entry_fee_per_qty * qty
            pnl = (exit_px - entry_px) * qty - entry_fee_piece - fee if side == "BUY" else (entry_px - exit_px) * qty - entry_fee_piece - fee
            pnl_total = pnl_accum + pnl
            exit_notional += qty * exit_px
            exit_qty_sum += qty
            avg_exit = exit_notional / exit_qty_sum if exit_qty_sum > 0 else exit_px
            reason = "TIME_EXIT+PARTIAL" if partial_hit else "TIME_EXIT"
            trades.append(Trade(symbol, side, entry_time, entry_px, t_next.isoformat(), avg_exit, entry_qty, float(pnl_total), reason))
            in_pos = False

    return trades


def apply_portfolio_constraints(
    tdf: pd.DataFrame,
    cfg: dict,
    keep_all: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], dict]:
    bt = cfg.get("backtest", {})
    use_portfolio = bool(bt.get("use_portfolio_sizing", False))
    if not use_portfolio:
        return tdf.copy(), tdf.copy() if keep_all else None, {}

    init_cap = float(bt.get("initial_capital_usdt", 0))
    pos_pct = float(bt.get("position_pct_of_equity", 0))
    leverage = float(bt.get("leverage", cfg.get("position", {}).get("leverage", 1)))
    max_conc = int(bt.get("max_concurrent_positions", 0))
    if init_cap <= 0 or pos_pct <= 0 or max_conc <= 0:
        return tdf.copy(), tdf.copy() if keep_all else None, {}

    if not keep_all:
        entry_t = pd.to_datetime(tdf["entry_time"], utc=True, errors="coerce")
        exit_t = pd.to_datetime(tdf["exit_time"], utc=True, errors="coerce")
        order = entry_t.argsort(kind="mergesort").to_numpy()

        pnl_raw = pd.to_numeric(tdf["pnl_usdt"], errors="coerce").to_numpy(dtype=float)
        qty_raw = pd.to_numeric(tdf["qty"], errors="coerce").to_numpy(dtype=float)
        entry_px = pd.to_numeric(tdf["entry_price"], errors="coerce").to_numpy(dtype=float)
        notional_raw = qty_raw * entry_px
        notional_raw = np.where(np.isfinite(notional_raw), notional_raw, 0.0)
        denom = np.where(notional_raw > 0, notional_raw, 1.0)
        pnl_pct = pnl_raw / denom

        accepted = np.zeros(len(tdf), dtype=bool)
        scaled_pnl = np.zeros(len(tdf), dtype=float)
        scaled_qty = np.zeros(len(tdf), dtype=float)
        scaled_notional = np.zeros(len(tdf), dtype=float)

        equity = init_cap
        open_positions: List[Tuple[pd.Timestamp, float]] = []

        for idx in order:
            et = entry_t.iloc[idx]
            xt = exit_t.iloc[idx]
            if pd.isna(et) or pd.isna(xt):
                continue

            still_open: List[Tuple[pd.Timestamp, float]] = []
            for exit_time, pnl in open_positions:
                if exit_time <= et:
                    equity += pnl
                else:
                    still_open.append((exit_time, pnl))
            open_positions = still_open

            if len(open_positions) >= max_conc:
                continue

            notional = equity * pos_pct * leverage
            if notional <= 0:
                continue

            pnl = pnl_pct[idx] * notional
            qty = notional / max(1e-12, float(entry_px[idx]))

            accepted[idx] = True
            scaled_pnl[idx] = pnl
            scaled_qty[idx] = qty
            scaled_notional[idx] = notional
            open_positions.append((xt, pnl))

        for exit_time, pnl in open_positions:
            equity += pnl

        out_df = tdf.loc[accepted].copy()
        out_df["pnl_usdt"] = scaled_pnl[accepted]
        out_df["qty"] = scaled_qty[accepted]
        out_df["notional_usdt"] = scaled_notional[accepted]
        stats = {
            "initial_capital_usdt": init_cap,
            "final_capital_usdt": float(equity),
            "accepted_trades": int(accepted.sum()),
            "skipped_trades": int(len(accepted) - accepted.sum()),
            "max_concurrent_positions": max_conc,
            "position_pct_of_equity": pos_pct,
            "leverage": leverage,
        }
        return out_df, None, stats

    all_df = tdf.copy()
    all_df["pnl_usdt_raw"] = all_df["pnl_usdt"]
    all_df["qty_raw"] = all_df["qty"]
    all_df["notional_usdt_raw"] = all_df["qty_raw"] * all_df["entry_price"]
    all_df["notional_usdt_raw"] = all_df["notional_usdt_raw"].replace([0, float("inf"), float("-inf")], 0.0)
    all_df["notional_usdt_raw"] = all_df["notional_usdt_raw"].fillna(0.0)

    entry_t = pd.to_datetime(all_df["entry_time"], utc=True, errors="coerce")
    exit_t = pd.to_datetime(all_df["exit_time"], utc=True, errors="coerce")
    order = entry_t.argsort(kind="mergesort").to_numpy()

    denom = all_df["notional_usdt_raw"].to_numpy()
    denom = np.where(denom > 0, denom, 1.0)
    pnl_pct = (all_df["pnl_usdt_raw"] / denom).to_numpy()
    entry_px = all_df["entry_price"].to_numpy()

    accepted = np.zeros(len(all_df), dtype=bool)
    scaled_pnl = np.zeros(len(all_df), dtype=float)
    scaled_qty = np.zeros(len(all_df), dtype=float)
    scaled_notional = np.zeros(len(all_df), dtype=float)

    equity = init_cap
    open_positions: List[Tuple[pd.Timestamp, float]] = []

    for idx in order:
        et = entry_t.iloc[idx]
        xt = exit_t.iloc[idx]
        if pd.isna(et) or pd.isna(xt):
            continue

        still_open: List[Tuple[pd.Timestamp, float]] = []
        for exit_time, pnl in open_positions:
            if exit_time <= et:
                equity += pnl
            else:
                still_open.append((exit_time, pnl))
        open_positions = still_open

        if len(open_positions) >= max_conc:
            continue

        notional = equity * pos_pct * leverage
        if notional <= 0:
            continue

        pnl = pnl_pct[idx] * notional
        qty = notional / max(1e-12, float(entry_px[idx]))

        accepted[idx] = True
        scaled_pnl[idx] = pnl
        scaled_qty[idx] = qty
        scaled_notional[idx] = notional
        open_positions.append((xt, pnl))

    for exit_time, pnl in open_positions:
        equity += pnl

    all_df["accepted"] = accepted
    all_df.loc[accepted, "pnl_usdt"] = scaled_pnl[accepted]
    all_df.loc[accepted, "qty"] = scaled_qty[accepted]
    all_df.loc[accepted, "notional_usdt"] = scaled_notional[accepted]

    out_df = all_df[all_df["accepted"]].copy()
    stats = {
        "initial_capital_usdt": init_cap,
        "final_capital_usdt": float(equity),
        "accepted_trades": int(accepted.sum()),
        "skipped_trades": int(len(accepted) - accepted.sum()),
        "max_concurrent_positions": max_conc,
        "position_pct_of_equity": pos_pct,
        "leverage": leverage,
    }
    return out_df, all_df, stats


def run_backtest(cfg: dict, symbols: List[str], start: Optional[str], end: Optional[str], data_dir: str,
                 workers: int = 1, use_symbol_range: bool = False,
                 keep_all: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    all_trades: List[Trade] = []
    missing = []

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [(sym, start, end, data_dir, cfg, use_symbol_range) for sym in symbols]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_symbol, t) for t in tasks]
            for fut in as_completed(futures):
                sym, tr, err = fut.result()
                if err:
                    missing.append(sym)
                    continue
                all_trades.extend(tr)
    else:
        for sym in symbols:
            try:
                bar_seconds = int(cfg.get("backtest", {}).get("data_interval_seconds", 60))
                if use_symbol_range:
                    rng = _symbol_date_range(sym, data_dir, bar_seconds)
                    if rng is None:
                        missing.append(sym)
                        continue
                    s_start, s_end = rng
                else:
                    if not start or not end:
                        raise ValueError("Missing --start/--end (or use --use_symbol_range)")
                    s_start, s_end = start, end
                df1m = load_bars(sym, s_start, s_end, data_dir, bar_seconds)
            except FileNotFoundError:
                missing.append(sym)
                continue
            tr = backtest_symbol(sym, df1m, cfg)
            all_trades.extend(tr)

    if not all_trades:
        return pd.DataFrame(), pd.DataFrame(), {"missing": missing}

    tdf = pd.DataFrame([t.__dict__ for t in all_trades])
    accepted_df, all_df, portfolio_stats = apply_portfolio_constraints(tdf, cfg, keep_all=keep_all)
    if portfolio_stats:
        portfolio_stats["missing"] = missing
    return accepted_df, all_df, portfolio_stats


def run_backtest_dual(short_cfg: dict, long_cfg: dict, symbols: List[str], start: Optional[str], end: Optional[str],
                      data_dir: str, workers: int = 1, use_symbol_range: bool = False,
                      keep_all: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    all_trades: List[Trade] = []
    missing = []

    short_interval = int(short_cfg.get("backtest", {}).get("data_interval_seconds", 60))
    long_interval = int(long_cfg.get("backtest", {}).get("data_interval_seconds", 60))
    if short_interval != long_interval:
        raise ValueError("short and long configs must use the same data_interval_seconds")

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [(sym, start, end, data_dir, short_cfg, long_cfg, use_symbol_range) for sym in symbols]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_symbol_dual, t) for t in tasks]
            for fut in as_completed(futures):
                sym, tr, err = fut.result()
                if err:
                    missing.append(sym)
                    continue
                all_trades.extend(tr)
    else:
        for sym in symbols:
            try:
                if use_symbol_range:
                    rng = _symbol_date_range(sym, data_dir, short_interval)
                    if rng is None:
                        missing.append(sym)
                        continue
                    s_start, s_end = rng
                else:
                    if not start or not end:
                        raise ValueError("Missing --start/--end (or use --use_symbol_range)")
                    s_start, s_end = start, end
                df1m = load_bars(sym, s_start, s_end, data_dir, short_interval)
            except FileNotFoundError:
                missing.append(sym)
                continue
            all_trades.extend(backtest_symbol(sym, df1m, short_cfg))
            all_trades.extend(backtest_symbol(sym, df1m, long_cfg))

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


def _merge_cfg(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in ("enabled", "config_path"):
            continue
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_cfg(merged.get(key, {}), val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def _load_long_strategy_config(cfg: dict, path: str = "") -> Optional[dict]:
    if path:
        return yaml.safe_load(open(path, "r", encoding="utf-8"))
    raw = cfg.get("long_strategy")
    if not raw:
        return None
    if isinstance(raw, dict) and raw.get("enabled") is False:
        return None
    cfg_path = ""
    if isinstance(raw, str):
        cfg_path = raw
    elif isinstance(raw, dict):
        cfg_path = raw.get("config_path", "")
    else:
        return None
    if cfg_path:
        return yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    if isinstance(raw, dict):
        overrides = raw.get("overrides", raw)
        if isinstance(overrides, dict):
            return _merge_cfg(cfg, overrides)
    return None


def run_symbol(args: Tuple[str, Optional[str], Optional[str], str, dict, bool]) -> Tuple[str, List[Trade], Optional[str]]:
    symbol, start, end, data_dir, cfg, use_symbol_range = args
    try:
        bar_seconds = int(cfg.get("backtest", {}).get("data_interval_seconds", 60))
        if use_symbol_range:
            rng = _symbol_date_range(symbol, data_dir, bar_seconds)
            if rng is None:
                return symbol, [], f"No data for {symbol} in {data_dir}"
            start, end = rng
        if not start or not end:
            return symbol, [], "Missing --start/--end (or use --use_symbol_range)"
        df1m = load_bars(symbol, start, end, data_dir, bar_seconds)
    except FileNotFoundError as e:
        return symbol, [], str(e)
    tr = backtest_symbol(symbol, df1m, cfg)
    return symbol, tr, None


def run_symbol_dual(args: Tuple[str, Optional[str], Optional[str], str, dict, dict, bool]) -> Tuple[str, List[Trade], Optional[str]]:
    symbol, start, end, data_dir, short_cfg, long_cfg, use_symbol_range = args
    try:
        bar_seconds = int(short_cfg.get("backtest", {}).get("data_interval_seconds", 60))
        if use_symbol_range:
            rng = _symbol_date_range(symbol, data_dir, bar_seconds)
            if rng is None:
                return symbol, [], f"No data for {symbol} in {data_dir}"
            start, end = rng
        if not start or not end:
            return symbol, [], "Missing --start/--end (or use --use_symbol_range)"
        df1m = load_bars(symbol, start, end, data_dir, bar_seconds)
    except FileNotFoundError as e:
        return symbol, [], str(e)
    trades = []
    trades.extend(backtest_symbol(symbol, df1m, short_cfg))
    trades.extend(backtest_symbol(symbol, df1m, long_cfg))
    return symbol, trades, None


def summarize(trades_df: pd.DataFrame, initial_capital: float = 0.0) -> Dict:
    if trades_df.empty:
        return {
            "trades": 0,
            "total_pnl_usdt": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_naive": 0.0,
        }
    pnl = float(trades_df["pnl_usdt"].sum())
    win_rate = float((trades_df["pnl_usdt"] > 0).mean())
    gross_win = float(trades_df.loc[trades_df["pnl_usdt"] > 0, "pnl_usdt"].sum())
    gross_loss = float(-trades_df.loc[trades_df["pnl_usdt"] < 0, "pnl_usdt"].sum())
    profit_factor = float(gross_win / max(1e-9, gross_loss))

    eq = trades_df["pnl_usdt"].cumsum() + float(initial_capital or 0.0)
    peak = eq.cummax()
    mdd = float((eq / peak - 1.0).min())

    return {
        "trades": int(len(trades_df)),
        "total_pnl_usdt": pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_naive": mdd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbols")
    ap.add_argument("--symbols_file", default="", help="Path to symbols.txt (one per line)")
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)
    ap.add_argument("--data", default="data/raw/before_2026_02_01")
    ap.add_argument("--config", default="configs/strategy.yaml")
    ap.add_argument("--long_config", default="", help="Optional config path for long-only strategy")
    ap.add_argument("--use_long_strategy", action="store_true", help="Use long_strategy from config")
    ap.add_argument("--use_symbol_range", action="store_true", help="Use each symbol's available date range (ignores --start/--end).")
    ap.add_argument("--out", default="artifacts/backtests/run")
    ap.add_argument("--workers", type=int, default=1, help="parallel workers for symbols")
    ap.add_argument("--latency_s", type=float, default=None, help="simulation latency in seconds")
    ap.add_argument("--funding_8h", type=float, default=None, help="8h funding rate")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    bt_cfg = cfg.setdefault("backtest", {"maker_fee": 0.0002, "taker_fee": 0.0004, "slippage_bps": 5.0, "entry_offset_bps": 0.0})

    if args.latency_s is not None:
        bt_cfg["latency_seconds"] = args.latency_s
    if args.funding_8h is not None:
        bt_cfg["funding_rate_8h"] = args.funding_8h

    if not args.use_symbol_range and (not args.start or not args.end):
        raise SystemExit("ERROR: need --start and --end unless --use_symbol_range is set")

    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            syms = [line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        if not args.symbols:
            raise SystemExit("ERROR: need --symbols or --symbols_file")
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    long_cfg = None
    if args.long_config:
        long_cfg = _load_long_strategy_config(cfg, args.long_config)
    elif args.use_long_strategy:
        long_cfg = _load_long_strategy_config(cfg)
    if long_cfg:
        accepted_df, all_df, portfolio_stats = run_backtest_dual(
            cfg,
            long_cfg,
            syms,
            args.start,
            args.end,
            args.data,
            args.workers,
            use_symbol_range=args.use_symbol_range,
        )
    else:
        accepted_df, all_df, portfolio_stats = run_backtest(
            cfg,
            syms,
            args.start,
            args.end,
            args.data,
            args.workers,
            use_symbol_range=args.use_symbol_range,
        )
    if accepted_df.empty:
        print("No trades. Consider lowering thresholds / expanding symbols / longer date range.")
        return

    os.makedirs(args.out, exist_ok=True)
    accepted_df.to_csv(os.path.join(args.out, "trades.csv"), index=False)
    if portfolio_stats:
        all_df.to_csv(os.path.join(args.out, "trades_all.csv"), index=False)

    init_cap = 0.0
    if cfg.get("backtest", {}).get("use_portfolio_sizing", False):
        init_cap = float(cfg.get("backtest", {}).get("initial_capital_usdt", 0.0) or 0.0)
    summary = summarize(accepted_df, initial_capital=init_cap)
    start_val = "per_symbol" if args.use_symbol_range else args.start
    end_val = "per_symbol" if args.use_symbol_range else args.end
    summary.update(
        {
            "symbols": syms,
            "start": start_val,
            "end": end_val,
            "assumptions": {
                "data": f"{int(cfg.get('backtest', {}).get('data_interval_seconds', 60))}s OHLCV",
                "fill_model": "limit filled if next bar touches (low/high)",
                "symbol_range": "per_symbol_available" if args.use_symbol_range else "fixed",
            },
            "backtest_params": cfg.get("backtest", {}),
        }
    )
    if portfolio_stats:
        summary["portfolio"] = portfolio_stats

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
