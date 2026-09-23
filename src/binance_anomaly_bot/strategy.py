import asyncio
import copy
import math
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from .binance_client import BinanceFuturesClient
from .market_ws import MarketWs
from .user_ws import UserWs, OrderUpdate, AccountPositionUpdate
from .universe import SymbolSpec, UniverseSelector
from .utils import utc_ms, round_step, round_step_up, round_tick, format_by_step

@dataclass
class Trigger:
    symbol: str
    side: str            # "BUY" or "SELL"
    close15: float
    r15: float
    vol15: float
    baseline: float
    ts: int
    deadline_ts: int

@dataclass
class PositionState:
    symbol: str
    side: str
    qty: float
    entry_price: float
    peak_price: float
    stop_price: float
    tp_state: int
    open_ts: int

@dataclass
class StopOrderState:
    order_id: int
    stop_price: float
    orig_qty: float
    executed_qty: float
    status: str
    created_ts: int
    client_order_id: str = ""

    def remaining_qty(self) -> float:
        return max(0.0, self.orig_qty - self.executed_qty)

@dataclass
class TpOrderState:
    order_id: int
    level: int
    price: float
    orig_qty: float
    executed_qty: float
    status: str
    created_ts: int

    def remaining_qty(self) -> float:
        return max(0.0, self.orig_qty - self.executed_qty)

class StrategyEngine:
    def __init__(self, client: BinanceFuturesClient, market_ws: MarketWs, user_ws: UserWs,
                 store, universe: Dict[str, SymbolSpec], config: dict, dry_run: bool, logger,
                 position_mode: str = "ONEWAY"):
        self.client = client
        self.market_ws = market_ws
        self.user_ws = user_ws
        self.store = store
        self.universe = universe
        self._monitor_symbols: Set[str] = set(universe.keys())
        self.config = config
        self.dry_run = dry_run
        self.logger = logger

        self._long_cfg: Optional[dict] = None
        self._use_vol_percentile_long = False
        self._long_use_quote_volume = None

        self.baseline_quote: Dict[str, float] = {}
        self.last_close15: Dict[str, float] = {}
        self._baseline_partial_logged: Set[str] = set()
        self._baseline_insufficient_logged: Set[str] = set()

        self.triggers: asyncio.Queue[Trigger] = asyncio.Queue()
        self.positions: Dict[Tuple[str, str], PositionState] = {}
        self._stop_orders: Dict[Tuple[str, str], StopOrderState] = {}
        self._stop_order_ids: Dict[int, Tuple[str, str]] = {}
        self._stop_order_client_ids: Dict[str, Tuple[str, str]] = {}
        self._tp_orders: Dict[Tuple[str, str, int], TpOrderState] = {}
        self._tp_order_ids: Dict[int, Tuple[str, str, int]] = {}
        self._last_px_cache: Dict[str, Tuple[float, int]] = {}
        self._equity_cache = 0.0
        self._equity_ts = 0.0
        self._baseline_refresh_ts: Dict[str, float] = {}
        engine_cfg = self.config.get("engine", {})
        self._use_aggtrades = bool(engine_cfg.get("use_aggtrades_ws", False))
        self._bar_interval_s = int(engine_cfg.get("bar_interval_s", 20))
        self._bb_kline_cache_ttl_s = float(engine_cfg.get("bb_kline_cache_ttl_s", 30.0) or 0.0)
        self._bb_kline_cache: Dict[str, Tuple[float, List[float]]] = {}
        self._heartbeat_interval_s = int(engine_cfg.get("heartbeat_interval_s", 1800))
        self._roll_add_on_interval_s = max(float(self._bar_interval_s), 10.0)
        self._roll_add_on_ts: Dict[str, float] = {}
        uni_cfg = self.config.get("universe", {})
        self._universe_refresh_interval_s = float(uni_cfg.get("refresh_interval_s", 3600) or 0.0)
        self._universe_refresh_lock = asyncio.Lock()
        self._trigger_workers = max(1, int(engine_cfg.get("trigger_workers", 1) or 1))
        self._reject_summary_interval_s = 600
        self._reject_stats = {
            "since_ms": utc_ms(),
            "total_since_ms": utc_ms(),
            "reason_counts": Counter(),
            "symbol_counts": Counter(),
            "reason_symbol_counts": defaultdict(Counter),
            "examples": {},
            "total_reason_counts": Counter(),
            "total_symbol_counts": Counter(),
            "total_examples": {},
        }
        self._trigger_symbol_locks: Dict[str, asyncio.Lock] = {}
        self._signal_meta_by_order: Dict[int, Tuple[int, float, int, bool]] = {}
        self._order_waiters: Dict[int, asyncio.Queue] = {}
        self._order_update_cache: Dict[int, OrderUpdate] = {}
        self._order_fee_cache: Dict[int, dict] = {}
        self._threshold_proximity_best: Dict[Tuple[str, str], float] = {}
        self._bb_snapshot: Dict[Tuple[str, str], dict] = {}
        risk_cfg = self.config.get("risk", {})
        self._risk_loop_interval_s = float(risk_cfg.get("loop_interval_s", 1.0) or 1.0)
        self._max_stale_price_ms = int(float(risk_cfg.get("max_stale_price_s", 5.0) or 5.0) * 1000)
        self._force_stop_pct = max(0.0, float(risk_cfg.get("force_stop_pct", 0.005) or 0.005))
        bt_cfg = self.config.get("backtest", {})
        self._entry_tol_bps = float(bt_cfg.get("slippage_bps_base", bt_cfg.get("slippage_bps", 0.0)) or 0.0)
        self._time_exit_price = str(bt_cfg.get("time_exit_price", "close")).lower()
        entry_cfg = self.config.get("entry", {})
        self._max_taker_slippage_bps = float(entry_cfg.get("max_taker_slippage_bps", self._entry_tol_bps) or 0.0)
        direction_cfg = self.config.get("direction", {})
        self._enable_long = bool(direction_cfg.get("enable_long", True))
        self._enable_short = bool(direction_cfg.get("enable_short", True))
        self._position_mode = str(position_mode or "ONEWAY").upper()
        self._base_leverage = int(self.config.get("position", {}).get("leverage", 1) or 1)
        self._min_leverage = 5
        self._leverage_overrides: Dict[str, int] = {}
        self._leverage_initialized: Set[str] = set()
        vol_cfg = self.config.get("volume", {})
        self._baseline_method = str(vol_cfg.get("baseline_method", "median")).lower()
        self._use_quote_volume = bool(vol_cfg.get("use_quote_volume", True))
        self._use_vol_percentile = bool(vol_cfg.get("use_percentile_threshold", False))
        self._vol_percentile = float(vol_cfg.get("percentile", 90.0))
        lookback = int(vol_cfg.get("percentile_lookback_windows", 0) or 0)
        if lookback <= 0:
            lookback = int(vol_cfg.get("baseline_lookback_windows", vol_cfg.get("baseline_lookback_15m", 0)) or 0)
        self._vol_percentile_lookback = max(1, lookback)
        self._vol_percentile_update_s = float(vol_cfg.get("percentile_update_s", engine_cfg.get("baseline_refresh_s", 900)))
        self._vol_window_hist: Dict[str, deque] = {}
        self._vol_threshold: Dict[str, float] = {}
        self._vol_threshold_refresh_ts: Dict[str, float] = {}
        self._load_long_strategy_config()
        self._entry_offset_bps = float(entry_cfg.get("entry_offset_bps", self.config.get("backtest", {}).get("entry_offset_bps", 0.0)))
        sig_cfg = self.config.get("signal", {})
        self._use_abs_threshold = bool(sig_cfg.get("use_abs_threshold", True))
        pos_cfg = self.config.get("position", {})
        self._use_internal_equity = bool(pos_cfg.get("use_internal_equity", False))
        self._initial_equity = float(pos_cfg.get("initial_equity_usdt", 0.0) or 0.0)
        self._min_equity = float(pos_cfg.get("min_equity_usdt", 0.0) or 0.0)
        self._position_sync_interval_s = float(pos_cfg.get("position_sync_s", 10.0) or 10.0)
        self._use_account_updates = bool(pos_cfg.get("use_account_update_sync", True))
        self._account_update_wait_s = float(pos_cfg.get("account_update_wait_s", 30.0) or 30.0)
        self._position_sync_ts: Dict[str, float] = {}
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._halt_trading = False
        self._daily_date = None
        self._daily_pnl = 0.0
        self._ws_unhealthy_since = 0.0
        time_cfg = self.config.get("time_exit", {})
        self._time_exit_p1 = float(time_cfg.get("profit_1", 0.01))
        self._time_exit_p2 = float(time_cfg.get("profit_2", 0.02))
        self._time_exit_h1 = int(time_cfg.get("hold_1_minutes", 12))
        self._time_exit_h2 = int(time_cfg.get("hold_2_minutes", 20))
        if self._use_internal_equity:
            self.logger.info(
                "event=internal_equity_enabled initial=%.4f min_equity=%.4f",
                self._initial_equity,
                self._min_equity,
            )

    def _merge_cfg(self, base: dict, override: dict) -> dict:
        merged = copy.deepcopy(base)
        for key, val in override.items():
            if key in ("enabled", "config_path"):
                continue
            if isinstance(val, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_cfg(merged.get(key, {}), val)
            else:
                merged[key] = copy.deepcopy(val)
        return merged

    def _load_long_strategy_config(self) -> None:
        raw = self.config.get("long_strategy")
        if not raw:
            return
        if isinstance(raw, dict) and raw.get("enabled") is False:
            return
        cfg = None
        if isinstance(raw, str):
            cfg_path = raw
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
            except Exception as e:
                self.logger.warning("event=long_strategy_load_failed path=%s err=%s", cfg_path, e)
                return
        elif isinstance(raw, dict):
            cfg_path = raw.get("config_path")
            if cfg_path:
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                except Exception as e:
                    self.logger.warning("event=long_strategy_load_failed path=%s err=%s", cfg_path, e)
                    return
            else:
                cfg = self._merge_cfg(self.config, raw.get("overrides", raw))
        if not isinstance(cfg, dict):
            return
        self._long_cfg = cfg
        vol_cfg = cfg.get("volume", {})
        self._use_vol_percentile_long = bool(vol_cfg.get("use_percentile_threshold", False))
        self._long_use_quote_volume = bool(vol_cfg.get("use_quote_volume", True))

        base_window = int(self.config.get("signal", {}).get("window_minutes", 0) or 0)
        long_window = int(cfg.get("signal", {}).get("window_minutes", 0) or 0)
        if base_window and long_window and base_window != long_window:
            self.logger.warning(
                "event=long_strategy_window_mismatch base=%d long=%d",
                base_window,
                long_window,
            )
        cfg.setdefault("entry", {})

        if self._use_vol_percentile_long:
            long_pct = float(vol_cfg.get("percentile", self._vol_percentile))
            long_lb = int(vol_cfg.get("percentile_lookback_windows", self._vol_percentile_lookback))
            if abs(long_pct - self._vol_percentile) > 1e-6 or long_lb != self._vol_percentile_lookback:
                self.logger.warning(
                    "event=long_strategy_percentile_mismatch base_pct=%.4f long_pct=%.4f base_lb=%d long_lb=%d",
                    self._vol_percentile,
                    long_pct,
                    self._vol_percentile_lookback,
                    long_lb,
                )
        if self._long_use_quote_volume is not None and self._long_use_quote_volume != self._use_quote_volume:
            self.logger.warning(
                "event=long_strategy_volume_unit_mismatch base_quote=%s long_quote=%s",
                self._use_quote_volume,
                self._long_use_quote_volume,
            )

    def _cfg_for_side(self, side: str) -> dict:
        if side == "BUY" and self._long_cfg:
            return self._long_cfg
        return self.config

    def _side_enabled(self, side: str) -> bool:
        if side == "BUY":
            return self._enable_long
        return self._enable_short

    def _side_active_for_cfg(self, cfg: dict, side: str) -> bool:
        if not self._side_enabled(side):
            return False
        if side == "BUY":
            if self._long_cfg:
                return cfg is self._long_cfg
            return True
        if self._long_cfg:
            return cfg is self.config
        return True

    def _max_bb_window_minutes(self) -> int:
        max_window = 0
        cfgs = (self.config, self._long_cfg) if self._long_cfg else (self.config,)
        for cfg in cfgs:
            if not cfg:
                continue
            bb_cfg = cfg.get("bollinger", {})
            if bool(bb_cfg.get("enabled", False)):
                window = int(bb_cfg.get("window_minutes", 0) or 0)
                if window > max_window:
                    max_window = window
        return max_window

    async def run(self):
        lev = int(self.config["position"]["leverage"])
        await self._set_leverage_bulk(lev)
        await self._load_baselines()
        await self._prewarm_volume_percentiles()
        if not self.dry_run:
            if self._use_account_updates:
                await self._wait_for_account_update()
            else:
                await self._sync_positions_full()

        tasks = [
            asyncio.create_task(self._kline_poll_loop()),
            asyncio.create_task(self._order_update_consumer()),
            asyncio.create_task(self._risk_loop()),
            asyncio.create_task(self._reject_summary_loop()),
        ]
        if self._trigger_workers > 1:
            self.logger.info("event=trigger_workers count=%d", self._trigger_workers)
        for worker_id in range(self._trigger_workers):
            tasks.append(asyncio.create_task(self._trigger_consumer(worker_id)))
        if self._use_account_updates:
            tasks.append(asyncio.create_task(self._account_update_consumer()))
        if self._position_sync_interval_s > 0 and not self._use_account_updates:
            tasks.append(asyncio.create_task(self._position_sync_loop()))
        if self._heartbeat_interval_s > 0:
            tasks.append(asyncio.create_task(self._heartbeat_loop()))
        if self._universe_refresh_interval_s > 0:
            tasks.append(asyncio.create_task(self._universe_refresh_loop()))
        await asyncio.gather(*tasks)

    async def _set_leverage_bulk(self, lev: int):
        # avoid blasting leverage changes; set a subset up-front, rest on-demand
        if self.dry_run:
            return
        for sym in sorted(self._monitor_symbols)[:50]:
            try:
                r = await self.client.change_leverage(sym, lev)
                if r.status == 200:
                    self._leverage_initialized.add(sym)
                else:
                    await self._log_error(
                        "position_manage",
                        event="leverage_bulk_rejected",
                        symbol=sym,
                        reason="api_rejected",
                        details={
                            "leverage": lev,
                            "status": r.status,
                            "response": r.data,
                        },
                    )
            except Exception as e:
                await self._log_error(
                    "position_manage",
                    event="leverage_bulk_failed",
                    symbol=sym,
                    error=e,
                    details={
                        "leverage": lev,
                        "traceback": traceback.format_exc(),
                    },
                )

    async def _load_baselines(self):
        lookback = self._baseline_lookback()
        lim = max(lookback, 1) + 2
        for sym in sorted(self._monitor_symbols):
            try:
                kl = await self._fetch_klines_with_retry(
                    sym,
                    "15m",
                    limit=lim,
                    stage="signal_acquire",
                    event="baseline_klines_failed",
                )
                if not kl:
                    continue
                kl_len = len(kl)
                if kl_len < 2:
                    if sym not in self._baseline_insufficient_logged:
                        self.logger.warning(
                            "event=baseline_klines_insufficient symbol=%s klines=%d",
                            sym,
                            kl_len,
                        )
                        self._baseline_insufficient_logged.add(sym)
                    continue
                try:
                    closes = [float(x[4]) for x in kl]
                    qvols = [float(x[7]) for x in kl]  # quoteVolume
                    bvols = [float(x[5]) for x in kl]  # base volume
                except Exception as e:
                    await self._log_error(
                        "signal_acquire",
                        event="baseline_klines_parse_failed",
                        symbol=sym,
                        error=e,
                        details={
                            "lookback": lookback,
                            "limit": lim,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    continue

                vols = qvols if self._use_quote_volume else bvols
                lookback_use = min(lookback, kl_len - 1)
                if lookback_use <= 0:
                    continue
                q = vols[-(lookback_use + 1):-1]
                self.baseline_quote[sym] = self._calc_baseline(q)

                if lookback_use < lookback and sym not in self._baseline_partial_logged:
                    self.logger.info(
                        "event=baseline_klines_partial symbol=%s lookback=%d used=%d klines=%d",
                        sym,
                        lookback,
                        lookback_use,
                        kl_len,
                    )
                    self._baseline_partial_logged.add(sym)
                if kl_len >= 2:
                    self.last_close15[sym] = float(kl[-2][4])  # last completed 15m close
            except Exception as e:
                await self._log_error(
                    "signal_acquire",
                    event="baseline_load_failed",
                    symbol=sym,
                    error=e,
                    details={
                        "lookback": lookback,
                        "traceback": traceback.format_exc(),
                    },
                )
                continue
        self.logger.info("event=baselines_loaded symbols=%d", len(self.baseline_quote))

    async def _prewarm_volume_percentiles(self) -> None:
        if not (self._use_vol_percentile or self._use_vol_percentile_long):
            return
        window_minutes = int(self.config.get("signal", {}).get("window_minutes", 0) or 0)
        if window_minutes <= 0:
            return
        lookback = int(self._vol_percentile_lookback or 0)
        if lookback <= 0:
            return
        bars_needed = window_minutes + lookback - 1
        sem = asyncio.Semaphore(int(self.config.get("engine", {}).get("rest_concurrency", 20)))
        symbols = list(self._monitor_symbols)
        self.logger.info(
            "event=vol_percentile_prewarm_start symbols=%d window=%d lookback=%d bars=%d",
            len(symbols),
            window_minutes,
            lookback,
            bars_needed,
        )

        async def _prewarm_symbol(sym: str) -> bool:
            async with sem:
                try:
                    kl = await self._fetch_klines_with_retry(
                        sym,
                        "1m",
                        limit=bars_needed,
                        stage="signal_calc",
                        event="vol_percentile_klines_failed",
                    )
                    if not kl or len(kl) < window_minutes:
                        return False
                    base_vol = [float(x[5]) for x in kl]
                    quote_vol = [float(x[7]) for x in kl]
                    vols = quote_vol if self._use_quote_volume else base_vol
                    if len(vols) < window_minutes:
                        return False
                    window = window_minutes
                    rolling = sum(vols[:window])
                    samples = [rolling]
                    for v_out, v_in in zip(vols, vols[window:]):
                        rolling += v_in - v_out
                        samples.append(rolling)
                    if len(samples) < lookback:
                        return False
                    tail = samples[-lookback:]
                    dq = deque(tail, maxlen=lookback)
                    self._vol_window_hist[sym] = dq
                    try:
                        thr = float(np.percentile(list(dq), self._vol_percentile))
                    except Exception as e:
                        await self._log_error(
                            "signal_calc",
                            event="vol_percentile_calc_failed",
                            symbol=sym,
                            error=e,
                            details={"traceback": traceback.format_exc()},
                        )
                        return False
                    if thr > 0:
                        self._vol_threshold[sym] = thr
                        self._vol_threshold_refresh_ts[sym] = time.time()
                        return True
                    return False
                except Exception as e:
                    self.logger.warning("event=vol_percentile_prewarm_failed symbol=%s err=%s", sym, e)
                    await self._log_error(
                        "signal_calc",
                        event="vol_percentile_prewarm_failed",
                        symbol=sym,
                        error=e,
                        details={
                            "window_minutes": window_minutes,
                            "lookback": lookback,
                            "bars_needed": bars_needed,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    return False

        tasks = [asyncio.create_task(_prewarm_symbol(sym)) for sym in symbols]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            ok = sum(1 for r in results if r is True)
            skipped = len(results) - ok
            self.logger.info("event=vol_percentile_prewarm_done ok=%d skipped=%d", ok, skipped)

    async def _kline_poll_loop(self):
        interval_s = int(self.config["engine"]["kline_poll_interval_s"])
        sem = asyncio.Semaphore(int(self.config["engine"]["rest_concurrency"]))
        while True:
            t0 = time.time()
            symbols = list(self._monitor_symbols)
            await asyncio.gather(
                *(self._poll_symbol(sym, sem) for sym in symbols),
                return_exceptions=True,
            )
            dt = time.time() - t0
            await asyncio.sleep(max(1.0, interval_s - dt))

    async def _poll_symbol(self, sym: str, sem: asyncio.Semaphore):
        if sym not in self._monitor_symbols:
            return
        async with sem:
            await self._maybe_refresh_baseline(sym)
            window_minutes = int(self.config["signal"]["window_minutes"])
            if self._use_aggtrades:
                bars_per_window = int(window_minutes * 60 / self._bar_interval_s)
                if bars_per_window <= 0:
                    return
                bb_window_minutes = self._max_bb_window_minutes()
                bars_needed = bars_per_window
                if bb_window_minutes > 0 and self._bar_interval_s > 0:
                    total_s = bb_window_minutes * 60
                    if total_s % self._bar_interval_s == 0:
                        bars_needed = max(bars_needed, int(total_s / self._bar_interval_s))
                agg_bars = self.market_ws.get_recent_bars_with_current(sym, bars_needed)
                if len(agg_bars) < bars_per_window:
                    return
                signal_bars = agg_bars[-bars_per_window:]
                open15 = signal_bars[0].open if signal_bars else 0.0
                open15_ts = signal_bars[0].open_time if signal_bars else 0
                low_bar = min(signal_bars, key=lambda b: b.low)
                high_bar = max(signal_bars, key=lambda b: b.high)
                low15 = low_bar.low
                low15_ts = low_bar.open_time
                high15 = high_bar.high
                high15_ts = high_bar.open_time
                vol15_q = sum(b.quote_volume for b in signal_bars)
                vol15_b = sum(b.volume for b in signal_bars)
            else:
                agg_bars = None
                limit = max(2, window_minutes + 1)
                kl = await self._fetch_klines_with_retry(
                    sym,
                    "1m",
                    limit=limit,
                    stage="signal_acquire",
                    event="klines_fetch_failed",
                )
                if not kl:
                    return
                try:
                    open_times = [int(x[0]) for x in kl]
                    close_times = [int(x[6]) for x in kl]
                    opens = [float(x[1]) for x in kl]
                    closes = [float(x[4]) for x in kl]
                    lows = [float(x[3]) for x in kl]
                    highs = [float(x[2]) for x in kl]
                    base_vol = [float(x[5]) for x in kl]
                    quote_vol = [float(x[7]) for x in kl]
                except Exception as e:
                    await self._log_error(
                        "signal_calc",
                        event="klines_parse_failed",
                        symbol=sym,
                        error=e,
                        details={
                            "interval": "1m",
                            "limit": limit,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    return
                if len(closes) < window_minutes:
                    return
                open15 = opens[-window_minutes] if len(opens) >= window_minutes else 0.0
                open15_ts = open_times[-window_minutes] if len(open_times) >= window_minutes else 0
                low15 = float("inf")
                low15_ts = 0
                high15 = 0.0
                high15_ts = 0
                for idx in range(len(lows) - window_minutes, len(lows)):
                    low_val = lows[idx]
                    if low_val < low15:
                        low15 = low_val
                        low15_ts = open_times[idx]
                    high_val = highs[idx]
                    if high_val > high15:
                        high15 = high_val
                        high15_ts = open_times[idx]
                vol15_q = sum(quote_vol[-window_minutes:])
                vol15_b = sum(base_vol[-window_minutes:])

            # dynamic spread filter using WS bookTicker
            b = self.market_ws.books.get(sym)
            if b:
                mid = (b.bid + b.ask) / 2.0
                spread_bps = (b.ask - b.bid) / mid * 10000.0 if mid > 0 else 9999.0
                if spread_bps > float(self.config["universe"]["max_spread_bps"]):
                    return

            # funding filter
            m = self.market_ws.marks.get(sym)
            if m and abs(m.funding_rate) > float(self.config["position"]["max_abs_funding_rate"]):
                return

            if self._use_aggtrades:
                last_price, last_ts = self._latest_trade_snapshot(sym)
                if last_price <= 0:
                    last_price = signal_bars[-1].close
                    last_ts = signal_bars[-1].close_time
            else:
                last_price = closes[-1]
                last_ts = close_times[-1] if close_times else 0
            close15 = last_price
            vol15_signal = vol15_q if self._use_quote_volume else vol15_b
            if self._use_vol_percentile or self._use_vol_percentile_long:
                self._update_volume_percentile(sym, vol15_signal)
            try:
                await self._update_threshold_proximity(
                    sym,
                    open15,
                    low15,
                    high15,
                    last_price,
                    open15_ts,
                    low15_ts,
                    high15_ts,
                    last_ts,
                )
            except Exception as e:
                await self._log_error(
                    "signal_calc",
                    event="threshold_proximity_failed",
                    symbol=sym,
                    error=e,
                    details={
                        "open15": open15,
                        "low15": low15,
                        "high15": high15,
                        "last_price": last_price,
                        "traceback": traceback.format_exc(),
                    },
                )
            try:
                await self._maybe_emit_trigger(
                    sym,
                    self.config,
                    vol15_q,
                    vol15_b,
                    open15,
                    low15,
                    high15,
                    last_price,
                    close15,
                    b,
                    agg_bars=agg_bars,
                )
            except Exception as e:
                await self._log_error(
                    "signal_calc",
                    event="emit_trigger_failed",
                    symbol=sym,
                    error=e,
                    details={
                        "open15": open15,
                        "low15": low15,
                        "high15": high15,
                        "close15": close15,
                        "last_price": last_price,
                        "vol15_q": vol15_q,
                        "vol15_b": vol15_b,
                        "baseline": self.baseline_quote.get(sym, 0.0),
                        "traceback": traceback.format_exc(),
                    },
                )
            if self._long_cfg:
                try:
                    await self._maybe_emit_trigger(
                        sym,
                        self._long_cfg,
                        vol15_q,
                        vol15_b,
                        open15,
                        low15,
                        high15,
                        last_price,
                        close15,
                        b,
                        agg_bars=agg_bars,
                    )
                except Exception as e:
                    await self._log_error(
                        "signal_calc",
                        event="emit_trigger_failed_long",
                        symbol=sym,
                        error=e,
                        details={
                            "open15": open15,
                            "low15": low15,
                            "high15": high15,
                            "close15": close15,
                            "last_price": last_price,
                            "vol15_q": vol15_q,
                            "vol15_b": vol15_b,
                            "baseline": self.baseline_quote.get(sym, 0.0),
                            "traceback": traceback.format_exc(),
                        },
                    )

    async def _universe_refresh_loop(self):
        while True:
            await asyncio.sleep(self._universe_refresh_interval_s)
            try:
                await self._refresh_universe()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning("event=universe_refresh_failed err=%s", e)
                await self._log_error(
                    "signal_acquire",
                    event="universe_refresh_failed",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )

    async def _refresh_universe(self) -> None:
        async with self._universe_refresh_lock:
            selector = UniverseSelector(client=self.client, config=self.config, logger=self.logger)
            new_universe = await selector.build_universe()
            await self._apply_universe_update(new_universe)

    async def _apply_universe_update(self, new_universe: Dict[str, SymbolSpec]) -> None:
        new_symbols = set(new_universe.keys())
        prev_symbols = set(self._monitor_symbols)
        added = new_symbols - prev_symbols
        removed = prev_symbols - new_symbols
        if not added and not removed:
            self.logger.info("event=universe_refresh_noop symbols=%d", len(new_symbols))
            return

        self._monitor_symbols = set(new_symbols)
        for sym, spec in new_universe.items():
            self.universe[sym] = spec

        if removed:
            await self._drop_symbol_state(removed)

        active_symbols = {ps.symbol for ps in self.positions.values()}
        for sym in removed:
            if sym in active_symbols:
                continue
            self.universe.pop(sym, None)

        try:
            await self.market_ws.update_agg_trade_symbols(sorted(self._monitor_symbols))
        except Exception as e:
            self.logger.warning("event=market_ws_symbols_update_failed err=%s", e)
            await self._log_error(
                "signal_acquire",
                event="market_ws_symbols_update_failed",
                error=e,
                details={
                    "symbols": len(self._monitor_symbols),
                    "traceback": traceback.format_exc(),
                },
            )

        self.logger.info(
            "event=universe_refreshed total=%d added=%d removed=%d",
            len(self._monitor_symbols),
            len(added),
            len(removed),
        )
        if added:
            self.logger.info("event=universe_symbols_added symbols=%s", ",".join(sorted(added)))
        if removed:
            self.logger.info("event=universe_symbols_removed symbols=%s", ",".join(sorted(removed)))

    async def _drop_symbol_state(self, symbols: Set[str]) -> None:
        if not symbols:
            return
        self.market_ws.remove_symbol_data(symbols)
        for sym in symbols:
            self.baseline_quote.pop(sym, None)
            self.last_close15.pop(sym, None)
            self._baseline_refresh_ts.pop(sym, None)
            self._baseline_partial_logged.discard(sym)
            self._baseline_insufficient_logged.discard(sym)
            self._vol_window_hist.pop(sym, None)
            self._vol_threshold.pop(sym, None)
            self._vol_threshold_refresh_ts.pop(sym, None)
            self._roll_add_on_ts.pop(sym, None)
            self._bb_kline_cache.pop(sym, None)
            self._bb_snapshot.pop((sym, "BUY"), None)
            self._bb_snapshot.pop((sym, "SELL"), None)
            self._trigger_symbol_locks.pop(sym, None)
            self._position_sync_ts.pop(sym, None)
            self._leverage_overrides.pop(sym, None)
            self._leverage_initialized.discard(sym)
            self._last_px_cache.pop(sym, None)
            self._threshold_proximity_best.pop((sym, "BUY"), None)
            self._threshold_proximity_best.pop((sym, "SELL"), None)
            for side in ("BUY", "SELL"):
                try:
                    await self.store.delete_signal_proximity(sym, side)
                except Exception as e:
                    self.logger.warning(
                        "event=signal_proximity_cleanup_failed symbol=%s side=%s err=%s",
                        sym,
                        side,
                        e,
                    )
                    await self._log_error(
                        "signal_calc",
                        event="signal_proximity_cleanup_failed",
                        symbol=sym,
                        side=side,
                        error=e,
                        details={"traceback": traceback.format_exc()},
                    )

    async def _fetch_klines_with_retry(self, sym: str, interval: str, limit: int, *,
                                       stage: str, event: str,
                                       retries: int = 1, retry_delay_s: float = 0.2):
        for attempt in range(retries + 1):
            try:
                kl = await self.client.klines(sym, interval, limit=limit)
                if not isinstance(kl, list):
                    raise TypeError(f"klines_type={type(kl).__name__}")
                if not kl:
                    raise ValueError("klines_empty")
                return kl
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(retry_delay_s)
                    continue
                await self._log_error(
                    stage,
                    event=event,
                    symbol=sym,
                    error=e,
                    details={
                        "interval": interval,
                        "limit": limit,
                        "attempts": retries + 1,
                        "traceback": traceback.format_exc(),
                    },
                )
                return None
        return None

    async def _fetch_bb_closes_1m(self, sym: str, window_minutes: int):
        if window_minutes <= 0:
            return []
        if self._bb_kline_cache_ttl_s > 0:
            cached = self._bb_kline_cache.get(sym)
            if cached and (time.time() - cached[0]) < self._bb_kline_cache_ttl_s:
                return cached[1]
        limit = max(2, window_minutes + 1)
        kl = await self._fetch_klines_with_retry(
            sym,
            "1m",
            limit=limit,
            stage="signal_filter",
            event="bb_closes_fetch_failed",
        )
        if not kl:
            self.logger.warning("event=bb_closes_fetch_failed symbol=%s", sym)
            return []
        closes = []
        for row in kl:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            try:
                closes.append(float(row[4]))
            except (TypeError, ValueError):
                continue
        if len(closes) < window_minutes:
            return []
        closes = closes[-window_minutes:]
        if self._bb_kline_cache_ttl_s > 0:
            self._bb_kline_cache[sym] = (time.time(), closes)
        return closes

    async def _maybe_emit_trigger(
        self,
        sym: str,
        cfg: dict,
        vol15_q: float,
        vol15_b: float,
        open15: float,
        low15: float,
        high15: float,
        last_price: float,
        close15: float,
        book,
        agg_bars: Optional[List] = None,
    ) -> None:
        sig_cfg = cfg.get("signal", {})
        trigger_mode = str(sig_cfg.get("trigger_mode", "range")).lower()
        use_abs = bool(sig_cfg.get("use_abs_threshold", True))
        abs_thr = float(sig_cfg.get("abs_r15_threshold", 0.0))
        allow_long = self._side_active_for_cfg(cfg, "BUY")
        allow_short = self._side_active_for_cfg(cfg, "SELL")
        if not allow_long and not allow_short:
            return

        if trigger_mode in ("range", "legacy", "lowhigh"):
            if low15 <= 0 or high15 <= 0:
                return
        else:
            if open15 <= 0 or not use_abs:
                return

        mid = float(last_price)
        if trigger_mode in ("range", "legacy", "lowhigh"):
            move_long = self._move_from_low(low15, mid)
            move_short = self._move_from_high(high15, mid)
        else:
            move = (mid / open15) - 1.0
            move_long = move
            move_short = -move
        hit_long_pre = (use_abs and move_long >= abs_thr) and allow_long
        hit_short_pre = (use_abs and move_short >= abs_thr) and allow_short

        vol_cfg = cfg.get("volume", {})
        use_quote = bool(vol_cfg.get("use_quote_volume", True))
        vol_signal = vol15_q if use_quote else vol15_b
        baseline = self.baseline_quote.get(sym, 0.0)
        vol_multiple = float(vol_cfg.get("multiple", 0.0) or 0.0)
        use_vol_percentile = bool(vol_cfg.get("use_percentile_threshold", False))
        vol_threshold = self._vol_threshold.get(sym, 0.0) if use_vol_percentile else (
            vol_multiple * baseline if baseline > 0 else 0.0
        )
        vol_ok = self._volume_trigger_ok(sym, vol_signal, cfg)

        spread_bps = None
        if book:
            book_mid = (book.bid + book.ask) / 2.0
            if book_mid > 0:
                spread_bps = (book.ask - book.bid) / book_mid * 10000.0
        funding_rate = None
        mark = self.market_ws.marks.get(sym)
        if mark:
            funding_rate = mark.funding_rate

        debug_common = dict(
            ts=utc_ms(),
            symbol=sym,
            trigger_mode=trigger_mode,
            abs_thr=abs_thr,
            vol_signal=vol_signal,
            vol_threshold=vol_threshold,
            vol_ok=1 if vol_ok else 0,
            baseline=baseline,
            vol_multiple=vol_multiple,
            use_percentile=1 if use_vol_percentile else 0,
            spread_bps=spread_bps,
            funding_rate=funding_rate,
            open15=open15,
            low15=low15,
            high15=high15,
            close15=close15,
            last_price=last_price,
        )

        if hit_long_pre or hit_short_pre:
            reason = "volume_not_met" if not vol_ok else None
            if hit_long_pre:
                await self._log_signal_debug(
                    **debug_common,
                    side="BUY",
                    event="THRESHOLD_TOUCH",
                    reason=reason,
                    move_value=move_long,
                )
            if hit_short_pre:
                await self._log_signal_debug(
                    **debug_common,
                    side="SELL",
                    event="THRESHOLD_TOUCH",
                    reason=reason,
                    move_value=move_short,
                )
        if not vol_ok:
            return

        hit_long = hit_long_pre
        hit_short = hit_short_pre

        bb_cfg = cfg.get("bollinger", {})
        bb_hit_long = None
        bb_hit_short = None
        bb_upper = None
        bb_lower = None
        bb_note = None
        if bool(bb_cfg.get("enabled", False)):
            bb_mode = str(bb_cfg.get("mode", "or")).lower()
            bb_window_minutes = int(bb_cfg.get("window_minutes", 20))
            bb_num_std = float(bb_cfg.get("num_std", 2.5))
            bb_hit_long = False
            bb_hit_short = False
            bb_has_info = False
            closes = []
            if bb_window_minutes > 0:
                if self._use_aggtrades:
                    if self._bar_interval_s > 0 and (bb_window_minutes * 60) % self._bar_interval_s == 0:
                        bars_needed = int(bb_window_minutes * 60 / self._bar_interval_s)
                        if bars_needed > 1:
                            if agg_bars and len(agg_bars) >= bars_needed:
                                closes = [b.close for b in agg_bars[-bars_needed:]]
                            elif agg_bars is None:
                                bars = self.market_ws.get_recent_bars_with_current(sym, bars_needed)
                                if len(bars) >= bars_needed:
                                    closes = [b.close for b in bars[-bars_needed:]]
                    if not closes and (hit_long_pre or hit_short_pre or bb_mode in ("only", "bb_only", "or")):
                        closes = await self._fetch_bb_closes_1m(sym, bb_window_minutes)
                        if closes:
                            bb_note = "bb_fallback_1m"
                        else:
                            bb_note = "bb_data_insufficient"
                else:
                    closes = await self._fetch_bb_closes_1m(sym, bb_window_minutes)
                    if not closes:
                        bb_note = "bb_data_insufficient"
            if len(closes) >= 2:
                mean = sum(closes) / len(closes)
                var = sum((c - mean) ** 2 for c in closes) / len(closes)
                std = math.sqrt(var)
                bb_upper = mean + bb_num_std * std
                bb_lower = mean - bb_num_std * std
                bb_hit_long = bb_upper > 0 and mid > bb_upper
                bb_hit_short = bb_lower > 0 and mid < bb_lower
                bb_has_info = True
            if bb_has_info:
                snap_ts = debug_common["ts"]
                snap_common = {
                    "ts": snap_ts,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_note": bb_note,
                }
                if allow_long:
                    self._bb_snapshot[(sym, "BUY")] = {
                        **snap_common,
                        "bb_hit": 1 if bb_hit_long else 0,
                    }
                if allow_short:
                    self._bb_snapshot[(sym, "SELL")] = {
                        **snap_common,
                        "bb_hit": 1 if bb_hit_short else 0,
                    }
            if bb_has_info:
                if bb_mode in ("filter", "and"):
                    hit_long = hit_long and bb_hit_long
                    hit_short = hit_short and bb_hit_short
                elif bb_mode in ("only", "bb_only"):
                    hit_long = bb_hit_long
                    hit_short = bb_hit_short
                else:
                    hit_long = hit_long or bb_hit_long
                    hit_short = hit_short or bb_hit_short

            if hit_long_pre and not hit_long:
                await self._log_signal_debug(
                    **debug_common,
                    side="BUY",
                    event="BB_FILTER",
                    reason="bollinger_filter",
                    move_value=move_long,
                    bb_hit=1 if bb_hit_long else 0,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    note=bb_note,
                )
            if hit_short_pre and not hit_short:
                await self._log_signal_debug(
                    **debug_common,
                    side="SELL",
                    event="BB_FILTER",
                    reason="bollinger_filter",
                    move_value=move_short,
                    bb_hit=1 if bb_hit_short else 0,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    note=bb_note,
                )

        entry_cfg = cfg.get("entry", {})
        if hit_long and not self._can_open_side(sym, "BUY"):
            await self._log_signal_debug(
                **debug_common,
                side="BUY",
                event="NO_ORDER",
                reason="position_blocked",
                move_value=move_long,
                bb_hit=1 if bb_hit_long else 0,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
            )
            hit_long = False
        if hit_short and not self._can_open_side(sym, "SELL"):
            await self._log_signal_debug(
                **debug_common,
                side="SELL",
                event="NO_ORDER",
                reason="position_blocked",
                move_value=move_short,
                bb_hit=1 if bb_hit_short else 0,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
            )
            hit_short = False
        if not (hit_long or hit_short):
            return
        if hit_long and hit_short:
            side = "BUY" if move_long >= move_short else "SELL"
        elif hit_long:
            side = "BUY"
        else:
            side = "SELL"

        move_pct = move_long if side == "BUY" else move_short
        deadline_s = float(entry_cfg.get("signal_to_order_deadline_s", 0) or 0.0)
        deadline = utc_ms() + int(deadline_s * 1000)
        trig = Trigger(sym, side, float(close15), move_pct, vol_signal, self.baseline_quote.get(sym, 0.0), utc_ms(), deadline)
        await self.triggers.put(trig)
        trigger_reason = "low_move_up" if side == "BUY" else "high_move_down"

        entry_price = 0.0
        entry_reason = ""
        if book:
            px, entry_reason = self._compute_entry_price(trig, book, cfg)
            entry_price = float(px or 0.0)

        await self._safe_store(
            self.store.insert_trigger(
                ts=trig.ts,
                symbol=sym,
                side=side,
                r15=move_pct,
                vol15=vol_signal,
                base_vol15=vol15_b,
                baseline=trig.baseline,
                close15=close15,
                trigger_reason=trigger_reason,
                entry_price=entry_price,
                entry_reason=entry_reason,
                deadline_ts=deadline,
            ),
            stage="signal_calc",
            event="insert_trigger_failed",
            symbol=sym,
            side=side,
        )
        await self._reset_threshold_proximity(sym, side)

    def _latest_trade_snapshot(self, sym: str) -> Tuple[float, int]:
        now_ms = utc_ms()
        last = self.market_ws.last_trades.get(sym)
        if last and (now_ms - last[1]) <= self._max_stale_price_ms:
            return float(last[0]), int(last[1])
        b = self.market_ws.books.get(sym)
        if b and b.bid > 0 and b.ask > 0 and (now_ms - b.ts) <= self._max_stale_price_ms:
            return float((b.bid + b.ask) / 2.0), int(b.ts)
        bars = self.market_ws.get_recent_bars(sym, 1)
        if bars and bars[-1].close > 0:
            return float(bars[-1].close), int(bars[-1].close_time)
        return 0.0, 0

    def _move_from_low(self, low_px: float, last_px: float) -> float:
        if low_px <= 0 or last_px <= 0:
            return 0.0
        return (last_px / low_px) - 1.0

    def _move_from_high(self, high_px: float, last_px: float) -> float:
        if high_px <= 0 or last_px <= 0:
            return 0.0
        return (high_px / last_px) - 1.0

    def _calc_move_for_side(self, trigger_mode: str, side: str, open15: float,
                            low15: float, high15: float, last_price: float) -> Optional[float]:
        if trigger_mode in ("range", "legacy", "lowhigh"):
            if low15 <= 0 or high15 <= 0 or last_price <= 0:
                return None
            if side == "BUY":
                return self._move_from_low(low15, last_price)
            return self._move_from_high(high15, last_price)
        if open15 <= 0 or last_price <= 0:
            return None
        move = (last_price / open15) - 1.0
        return move if side == "BUY" else -move

    def _stop_slippage_bps(self, mark_price: float, book, side: str) -> Optional[float]:
        if mark_price <= 0 or not book:
            return None
        if side == "BUY":
            exec_px = float(book.bid)
        else:
            exec_px = float(book.ask)
        if exec_px <= 0:
            return None
        return abs(exec_px / mark_price - 1.0) * 10000.0

    def _calc_move_endpoints(self, trigger_mode: str, side: str, open15: float, low15: float,
                             high15: float, last_price: float, open15_ts: int, low15_ts: int,
                             high15_ts: int, last_ts: int) -> Optional[Tuple[float, int, float, int]]:
        if trigger_mode in ("range", "legacy", "lowhigh"):
            if side == "BUY":
                start_price = low15
                start_ts = low15_ts
            else:
                start_price = high15
                start_ts = high15_ts
        else:
            start_price = open15
            start_ts = open15_ts
        end_price = last_price
        end_ts = last_ts
        if start_price <= 0 or end_price <= 0:
            return None
        if end_ts <= 0:
            end_ts = utc_ms()
        if start_ts <= 0:
            start_ts = end_ts
        return start_price, start_ts, end_price, end_ts

    async def _update_threshold_proximity(self, sym: str, open15: float, low15: float,
                                          high15: float, last_price: float, open15_ts: int,
                                          low15_ts: int, high15_ts: int, last_ts: int) -> None:
        rows = []
        now_ms = utc_ms()
        for side in ("BUY", "SELL"):
            cfg = self._cfg_for_side(side)
            if not self._side_enabled(side):
                continue
            sig_cfg = cfg.get("signal", {})
            if not bool(sig_cfg.get("use_abs_threshold", True)):
                continue
            abs_thr = float(sig_cfg.get("abs_r15_threshold", 0.0) or 0.0)
            if abs_thr <= 0:
                continue
            trigger_mode = str(sig_cfg.get("trigger_mode", "range")).lower()
            move_value = self._calc_move_for_side(trigger_mode, side, open15, low15, high15, last_price)
            if move_value is None:
                continue
            endpoints = self._calc_move_endpoints(
                trigger_mode,
                side,
                open15,
                low15,
                high15,
                last_price,
                open15_ts,
                low15_ts,
                high15_ts,
                last_ts,
            )
            if endpoints is None:
                continue
            start_price, start_ts, end_price, end_ts = endpoints
            key = (sym, side)
            best = self._threshold_proximity_best.get(key)
            if best is not None and move_value <= best:
                continue
            self._threshold_proximity_best[key] = move_value
            diff = abs(abs_thr - move_value)
            rows.append({
                "ts": now_ms,
                "symbol": sym,
                "side": side,
                "r15": float(move_value),
                "threshold": abs_thr,
                "diff": float(diff),
                "start_price": float(start_price),
                "start_ts": int(start_ts),
                "end_price": float(end_price),
                "end_ts": int(end_ts),
            })
        if not rows:
            return
        try:
            await self.store.upsert_signal_proximity(rows)
        except Exception as e:
            self.logger.warning("event=signal_proximity_update_failed symbol=%s err=%s", sym, e)
            await self._log_error(
                "signal_calc",
                event="signal_proximity_update_failed",
                symbol=sym,
                error=e,
                details={"traceback": traceback.format_exc()},
            )

    async def _log_signal_debug(self, **row) -> None:
        try:
            await self.store.insert_signal_debug(**row)
        except Exception as e:
            self.logger.warning("event=signal_debug_log_failed symbol=%s err=%s", row.get("symbol"), e)

    async def _safe_store(self, coro, *, stage: str, event: str, symbol: Optional[str] = None,
                          side: Optional[str] = None, order_id: Optional[int] = None,
                          position_id: Optional[int] = None, details: Optional[dict] = None) -> None:
        try:
            await coro
        except Exception as e:
            self.logger.warning("event=store_op_failed stage=%s event=%s err=%s", stage, event, e)
            await self._log_error(
                stage,
                event=event,
                symbol=symbol,
                side=side,
                order_id=order_id,
                position_id=position_id,
                error=e,
                details=details,
            )

    async def _log_error(self, stage: str, *, event: str = "", reason: str = "",
                         symbol: Optional[str] = None, side: Optional[str] = None,
                         order_id: Optional[int] = None, position_id: Optional[int] = None,
                         error: Optional[Exception] = None, details: Optional[dict] = None) -> None:
        payload = dict(details or {})
        err_text = None
        if error is not None:
            err_text = f"{type(error).__name__}: {error}"
            payload.setdefault("error_type", type(error).__name__)
            payload.setdefault("error_message", str(error))
        if not payload:
            payload = None
        try:
            await self.store.insert_error_log(
                ts=utc_ms(),
                stage=stage,
                symbol=symbol,
                side=side,
                event=event,
                reason=reason,
                error=err_text,
                details=payload,
                order_id=order_id,
                position_id=position_id,
            )
        except Exception as e:
            self.logger.warning("event=error_log_failed stage=%s err=%s", stage, e)

    def _schedule_error_log(self, stage: str, **kwargs) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(self._log_error(stage, **kwargs))

    async def _reset_threshold_proximity(self, sym: str, side: str) -> None:
        self._threshold_proximity_best.pop((sym, side), None)
        try:
            await self.store.delete_signal_proximity(sym, side)
        except Exception as e:
            self.logger.warning(
                "event=signal_proximity_reset_failed symbol=%s side=%s err=%s",
                sym,
                side,
                e,
            )
            await self._log_error(
                "signal_calc",
                event="signal_proximity_reset_failed",
                symbol=sym,
                side=side,
                error=e,
                details={"traceback": traceback.format_exc()},
            )

    def _can_open_side(self, sym: str, side: str) -> bool:
        if self._position_mode != "HEDGE":
            if self._has_position(sym) and not self._has_position(sym, side):
                return False
        else:
            if self._has_both_sides(sym):
                return False
        return True

    def _update_volume_percentile(self, sym: str, vol_signal: float) -> None:
        if not self._use_vol_percentile or vol_signal <= 0:
            return
        dq = self._vol_window_hist.get(sym)
        if dq is None:
            dq = deque(maxlen=self._vol_percentile_lookback)
            self._vol_window_hist[sym] = dq
        dq.append(float(vol_signal))
        now = time.time()
        last = self._vol_threshold_refresh_ts.get(sym, 0.0)
        if (now - last) < self._vol_percentile_update_s:
            return
        if len(dq) < self._vol_percentile_lookback:
            return
        try:
            thr = float(np.percentile(list(dq), self._vol_percentile))
        except Exception as e:
            self._schedule_error_log(
                "signal_calc",
                event="vol_percentile_calc_failed",
                symbol=sym,
                error=e,
                details={
                    "vol_signal": vol_signal,
                    "percentile": self._vol_percentile,
                    "lookback": self._vol_percentile_lookback,
                    "traceback": traceback.format_exc(),
                },
            )
            return
        if thr > 0:
            self._vol_threshold[sym] = thr
            self._vol_threshold_refresh_ts[sym] = now

    def _volume_trigger_ok(self, sym: str, vol_signal: float, cfg: dict) -> bool:
        if vol_signal <= 0:
            return False
        vol_cfg = cfg.get("volume", {})
        use_vol_percentile = bool(vol_cfg.get("use_percentile_threshold", False))
        if use_vol_percentile:
            thr = self._vol_threshold.get(sym, 0.0)
            if thr <= 0:
                return False
            return vol_signal >= thr
        baseline = self.baseline_quote.get(sym, 0.0)
        if baseline <= 0:
            return False
        return vol_signal >= float(vol_cfg.get("multiple", 0.0)) * baseline

    def _is_trigger(self, sym: str, move_pct: float) -> bool:
        if self._halt_trading:
            return False
        if move_pct == 0:
            return False
        if not self._use_abs_threshold:
            return False
        abs_thr = float(self.config["signal"]["abs_r15_threshold"])
        return abs(move_pct) >= abs_thr

    async def _trigger_consumer(self, worker_id: int = 0):
        while True:
            trig = await self.triggers.get()
            lock = self._trigger_symbol_locks.setdefault(trig.symbol, asyncio.Lock())
            try:
                async with lock:
                    await self._handle_trigger(trig)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning("event=trigger_worker_error worker=%d err=%s", worker_id, e)
                await self._log_error(
                    "order_request",
                    event="trigger_worker_error",
                    error=e,
                    details={
                        "worker_id": worker_id,
                        "traceback": traceback.format_exc(),
                    },
                )

    async def _handle_trigger(self, trig: Trigger):
        if self._halt_trading:
            self._record_reject("halt_trading", trig.symbol, trig.side)
            return
        if trig.symbol not in self._monitor_symbols:
            self._record_reject("symbol_removed", trig.symbol, trig.side)
            return
        cfg = self._cfg_for_side(trig.side)
        same_side_pos = self._get_position(trig.symbol, trig.side)
        if not same_side_pos and not self.dry_run:
            if self._use_account_updates:
                if not self.user_ws.account_ready.is_set():
                    self._record_reject("account_update_not_ready", trig.symbol, trig.side)
                    return
            else:
                await self._sync_positions_for_symbol(trig.symbol)
                same_side_pos = self._get_position(trig.symbol, trig.side)
        active_symbols = {ps.symbol for ps in self.positions.values()}
        max_conc = int(cfg.get("position", {}).get("max_concurrent_positions", self.config["position"]["max_concurrent_positions"]))
        if (not same_side_pos and trig.symbol not in active_symbols and
                len(active_symbols) >= max_conc):
            self._record_reject(
                "max_concurrent_positions",
                trig.symbol,
                trig.side,
                f"symbols={len(active_symbols)}",
            )
            return
        if self._position_mode != "HEDGE" and self._has_position(trig.symbol) and not same_side_pos:
            self._record_reject("position_exists", trig.symbol, trig.side)
            return
        if not self._side_enabled(trig.side):
            self._record_reject("direction_disabled", trig.symbol, trig.side)
            return

        while utc_ms() < trig.deadline_ts:
            same_side_pos = self._get_position(trig.symbol, trig.side)
            if same_side_pos:
                px = self._price_for_add_on(trig.symbol)
                prof = self._profit_pct(same_side_pos, px) if px > 0 else None
                if prof is None or prof < 0.03:
                    extra = f"pnl_pct={prof:.4f} px={px:.8f}" if prof is not None else "pnl_pct=na"
                    self._record_reject("add_on_profit_not_met", trig.symbol, trig.side, extra)
                    return
            b = self.market_ws.books.get(trig.symbol)
            if not b:
                await asyncio.sleep(0.05)
                continue

            price, reject_reason = self._compute_entry_price(trig, b, cfg)
            if price is None:
                self._record_reject(reject_reason or "entry_price_not_eligible", trig.symbol, trig.side)
                await asyncio.sleep(0.2)
                continue

            spec = self.universe[trig.symbol]
            min_notional_cfg = float(cfg.get("position", {}).get("min_notional_usdt", 0.0) or 0.0)
            min_notional = max(float(spec.min_notional or 0.0), min_notional_cfg)
            notional = await self._compute_notional(cfg)
            if notional <= 0:
                self._record_reject("notional_zero_or_halt", trig.symbol, trig.side)
                return
            if min_notional > 0 and notional < min_notional:
                notional = min_notional
            notional = self._apply_execution_caps(trig.symbol, notional, trig.side, cfg)
            if notional <= 0:
                self._record_reject("execution_caps", trig.symbol, trig.side)
                return
            price = self._apply_dynamic_slippage(trig.symbol, price, trig.side, cfg)
            qty_raw = notional / price
            qty = round_step(qty_raw, spec.step_size)
            qty_rounding = ROUND_DOWN
            if qty < spec.min_qty:
                self._record_reject(
                    "qty_below_min_qty",
                    trig.symbol,
                    trig.side,
                    f"qty={qty:.6f} min_qty={spec.min_qty:.6f}",
                )
                return
            cap = self._get_bar_notional_cap(trig.symbol, cfg)
            max_qty = None
            if cap > 0:
                max_qty = round_step(cap / price, spec.step_size)
                if max_qty < spec.min_qty:
                    self._record_reject(
                        "bar_cap_below_min_qty",
                        trig.symbol,
                        trig.side,
                        f"cap_qty={max_qty:.6f} min_qty={spec.min_qty:.6f}",
                    )
                    return
                qty = min(qty, max_qty)
            px = round_tick(price, spec.tick_size)
            order_notional = qty * px
            if min_notional > 0 and order_notional < min_notional:
                qty_min = round_step_up(min_notional / max(px, 1e-12), spec.step_size)
                if qty_min < spec.min_qty:
                    self._record_reject(
                        "min_notional_below_min_qty",
                        trig.symbol,
                        trig.side,
                        f"qty_min={qty_min:.6f} min_qty={spec.min_qty:.6f}",
                    )
                    return
                if max_qty is not None and qty_min > max_qty:
                    self._record_reject(
                        "min_notional_above_bar_cap",
                        trig.symbol,
                        trig.side,
                        f"qty_min={qty_min:.6f} cap_qty={max_qty:.6f}",
                    )
                    return
                qty = qty_min
                order_notional = qty * px
                qty_rounding = ROUND_UP
            if not self._preflight_order(trig.symbol, trig.side, qty, px, spec, order_notional, cfg):
                return
            qty_str = format_by_step(qty, spec.step_size, rounding=qty_rounding)
            px_str = format_by_step(px, spec.tick_size, rounding=ROUND_DOWN)
            pos_side = self._position_side_for_order(trig.side)

            maker_only = bool(cfg.get("entry", {}).get("maker_only", False))
            marketable, taker_slippage_bps = self._taker_slippage_bps(trig, b, px)
            is_taker = bool(marketable)
            max_taker_slippage_bps = float(cfg.get("entry", {}).get("max_taker_slippage_bps", self._max_taker_slippage_bps))
            if marketable and taker_slippage_bps > max_taker_slippage_bps:
                self._record_reject(
                    "taker_slippage_exceeds",
                    trig.symbol,
                    trig.side,
                    f"bps={taker_slippage_bps:.2f} max={max_taker_slippage_bps:.2f}",
                )
                return
            tif = "GTC" if (marketable and maker_only) else ("GTX" if maker_only else "GTC")
            client_id = f"EV15M_{trig.symbol}_{trig.ts}"

            if self.dry_run:
                self.logger.info(
                    "event=entry_simulated symbol=%s side=%s qty=%.6f price=%.8f tif=%s last_px=%.8f move=%.3f%%",
                    trig.symbol,
                    trig.side,
                    qty,
                    px,
                    tif,
                    trig.close15,
                    trig.r15 * 100,
                )
                fill = b.bid if trig.side == "BUY" else b.ask
                await self._on_filled(
                    trig.symbol,
                    trig.side,
                    qty,
                    fill,
                    order_id=None,
                    signal_ts=trig.ts,
                    signal_price=trig.close15,
                    order_accept_ts=utc_ms(),
                    is_taker=is_taker,
                )
                return

            await self._ensure_symbol_leverage(trig.symbol)

            order_params = dict(
                symbol=trig.symbol,
                side=trig.side,
                type="LIMIT",
                timeInForce=tif,
                quantity=qty_str,
                price=px_str,
                newClientOrderId=client_id,
                newOrderRespType="ACK",
            )
            if pos_side:
                order_params["positionSide"] = pos_side
            order_id = await self._submit_limit_order(trig.symbol, trig.side, order_params)
            if not order_id:
                return
            self._signal_meta_by_order[int(order_id)] = (trig.ts, trig.close15, utc_ms(), is_taker)
            await self._safe_store(
                self.store.upsert_order(
                    ts=utc_ms(),
                    symbol=trig.symbol,
                    client_order_id=client_id,
                    order_id=order_id,
                    side=trig.side,
                    type="LIMIT",
                    price=px,
                    qty=qty,
                    status="NEW",
                    avg_price=0.0,
                    executed_qty=0.0,
                ),
                stage="order_submit",
                event="order_upsert_failed",
                symbol=trig.symbol,
                side=trig.side,
                order_id=int(order_id),
            )

            ttl_s = int(cfg.get("entry", {}).get("order_ttl_s", self.config["entry"]["order_ttl_s"]))
            filled = await self._wait_fill_or_timeout(trig.symbol, order_id, ttl_s)
            if filled:
                return
            await self._log_error(
                "order_fill",
                event="order_timeout",
                symbol=trig.symbol,
                side=trig.side,
                order_id=order_id,
                reason="order_timeout",
                details={
                    "ttl_s": ttl_s,
                    "qty": qty,
                    "price": px,
                },
            )
            self._record_reject(
                "order_timeout",
                trig.symbol,
                trig.side,
                f"order_id={order_id} ttl_s={ttl_s}",
            )
            await self._log_trade_event(
                symbol=trig.symbol,
                side=trig.side,
                event="ENTRY",
                qty=qty,
                price=px,
                status="TIMEOUT",
                reason="order_timeout",
                order_id=order_id,
                entry_price=px,
            )
            await self.client.cancel_order(symbol=trig.symbol, orderId=order_id)
            self._signal_meta_by_order.pop(int(order_id), None)
            return
        self._record_reject("signal_deadline_expired", trig.symbol, trig.side)

    def _apply_execution_caps(self, symbol: str, notional: float, side: str, cfg: dict) -> float:
        exec_cfg = cfg.get("execution", {})
        max_fill_pct = float(exec_cfg.get("max_fill_pct_of_bar", 1.0))
        min_fill = float(exec_cfg.get("min_fill_notional_usdt", 0.0))
        allow_partial = bool(exec_cfg.get("allow_partial_fill", True))
        require_bar = bool(exec_cfg.get("require_bar_for_execution", False))

        bars = self.market_ws.get_recent_bars(symbol, 1)
        if not bars:
            if require_bar:
                self._record_reject("no_recent_bar", symbol, side)
                return 0.0
            return notional
        bar = bars[-1]
        max_notional = max_fill_pct * float(bar.quote_volume)
        if max_notional <= 0:
            self._record_reject("bar_quote_volume_zero", symbol, side)
            return 0.0
        if notional > max_notional:
            if not allow_partial:
                self._record_reject(
                    "notional_above_bar_cap_no_partial",
                    symbol,
                    side,
                    f"notional={notional:.6f} cap={max_notional:.6f}",
                )
                return 0.0
            notional = max_notional
        if notional < min_fill:
            self._record_reject(
                "notional_below_min_fill",
                symbol,
                side,
                f"notional={notional:.6f} min_fill={min_fill:.6f}",
            )
            return 0.0
        return notional

    def _get_bar_notional_cap(self, symbol: str, cfg: dict) -> float:
        exec_cfg = cfg.get("execution", {})
        max_fill_pct = float(exec_cfg.get("max_fill_pct_of_bar", 1.0))
        bars = self.market_ws.get_recent_bars(symbol, 1)
        if not bars:
            return 0.0
        bar = bars[-1]
        return max_fill_pct * float(bar.quote_volume)

    def _apply_dynamic_slippage(self, symbol: str, price: float, side: str, cfg: dict) -> float:
        exec_cfg = cfg.get("execution", {})
        base = float(exec_cfg.get("slippage_bps_base", 0.0))
        k = float(exec_cfg.get("slippage_bps_k", 0.0))
        bps_min = float(exec_cfg.get("slippage_bps_min", 0.0))
        bps_max = float(exec_cfg.get("slippage_bps_max", 1e9))
        toward_market = bool(exec_cfg.get("slippage_towards_market", False))
        bars = self.market_ws.get_recent_bars(symbol, 1)
        if not bars:
            return price
        bar = bars[-1]
        if bar.close <= 0:
            return price
        range_bps = (bar.high - bar.low) / bar.close * 10000.0
        bps = base + k * range_bps
        bps = min(max(bps, bps_min), bps_max)
        slip = bps / 10000.0
        if toward_market:
            adj = price * (1.0 + slip) if side == "BUY" else price * (1.0 - slip)
        else:
            adj = price * (1.0 - slip) if side == "BUY" else price * (1.0 + slip)

        if not toward_market:
            return adj

        if not bool(self.config.get("entry", {}).get("maker_only", False)):
            return adj

        b = self.market_ws.books.get(symbol)
        if not b:
            return adj
        spec = self.universe.get(symbol)
        tick = spec.tick_size if spec else 0.0
        if side == "BUY":
            limit = b.ask - tick if tick > 0 else b.ask
            if limit > 0:
                return min(adj, limit)
            return adj
        limit = b.bid + tick if tick > 0 else b.bid
        if limit > 0:
            return max(adj, limit)
        return adj

    async def _maybe_refresh_baseline(self, sym: str) -> None:
        refresh_s = float(self.config["engine"].get("baseline_refresh_s", 900))
        now = time.time()
        last = self._baseline_refresh_ts.get(sym, 0.0)
        if (now - last) < refresh_s:
            return
        self._baseline_refresh_ts[sym] = now
        try:
            lookback = self._baseline_lookback()
            lim = max(lookback, 1) + 2
            kl = await self._fetch_klines_with_retry(
                sym,
                "15m",
                limit=lim,
                stage="signal_acquire",
                event="baseline_refresh_failed",
            )
            if not kl:
                return
            kl_len = len(kl)
            if kl_len < 2:
                if sym not in self._baseline_insufficient_logged:
                    self.logger.warning(
                        "event=baseline_refresh_insufficient symbol=%s klines=%d",
                        sym,
                        kl_len,
                    )
                    self._baseline_insufficient_logged.add(sym)
                return
            try:
                closes = [float(x[4]) for x in kl]
                qvols = [float(x[7]) for x in kl]
                bvols = [float(x[5]) for x in kl]
            except Exception as e:
                await self._log_error(
                    "signal_acquire",
                    event="baseline_refresh_parse_failed",
                    symbol=sym,
                    error=e,
                    details={
                        "lookback": lookback,
                        "limit": lim,
                        "traceback": traceback.format_exc(),
                    },
                )
                return

            vols = qvols if self._use_quote_volume else bvols
            lookback_use = min(lookback, kl_len - 1)
            if lookback_use <= 0:
                return
            q = vols[-(lookback_use + 1):-1]
            self.baseline_quote[sym] = self._calc_baseline(q)

            if lookback_use < lookback and sym not in self._baseline_partial_logged:
                self.logger.info(
                    "event=baseline_refresh_partial symbol=%s lookback=%d used=%d klines=%d",
                    sym,
                    lookback,
                    lookback_use,
                    kl_len,
                )
                self._baseline_partial_logged.add(sym)
            if len(closes) >= 2:
                self.last_close15[sym] = float(closes[-2])
        except Exception as e:
            await self._log_error(
                "signal_acquire",
                event="baseline_refresh_failed",
                symbol=sym,
                error=e,
                details={
                    "lookback": self._baseline_lookback(),
                    "traceback": traceback.format_exc(),
                },
            )
            return

    async def _compute_notional(self, cfg: Optional[dict] = None) -> float:
        cfg = cfg or self.config
        pos_cfg = cfg["position"]
        if self._use_internal_equity:
            eq = self._initial_equity + self._realized_pnl + self._unrealized_pnl
            if eq <= self._min_equity:
                self._halt_trading = True
                return 0.0
            pct = float(pos_cfg.get("position_pct_of_equity", 0.0))
            lev = float(pos_cfg.get("leverage", 1.0))
            notional = eq * pct * lev
            min_notional = float(pos_cfg.get("min_notional_usdt", 0.0))
            return max(notional, min_notional) if min_notional > 0 else notional
        use_equity = bool(pos_cfg.get("use_equity_sizing", False))
        if not use_equity:
            return float(pos_cfg["notional_usdt_per_trade"])

        eq = await self._get_equity()
        if eq <= 0:
            return 0.0
        pct = float(pos_cfg.get("position_pct_of_equity", 0.0))
        lev = float(pos_cfg.get("leverage", 1.0))
        notional = eq * pct * lev
        min_notional = float(pos_cfg.get("min_notional_usdt", 0.0))
        return max(notional, min_notional) if min_notional > 0 else notional

    async def _get_equity(self) -> float:
        ttl_s = float(self.config["position"].get("equity_refresh_s", 10.0))
        now = time.time()
        if (now - self._equity_ts) < ttl_s and self._equity_cache > 0:
            return self._equity_cache
        try:
            r = await self.client.account_info()
            if r.status != 200:
                self.logger.warning("event=account_info_failed err=%s", self._format_api_error(r.data))
                await self._log_error(
                    "position_manage",
                    event="account_info_failed",
                    error=None,
                    details={
                        "status": r.status,
                        "response": r.data,
                    },
                )
                return self._equity_cache
            data = r.data
            eq = float(data.get("availableBalance", 0) or data.get("totalWalletBalance", 0) or 0)
            if eq > 0:
                self._equity_cache = eq
                self._equity_ts = now
            return self._equity_cache
        except Exception as e:
            self.logger.warning("event=account_info_error err=%s", e)
            await self._log_error(
                "position_manage",
                event="account_info_error",
                error=e,
                details={"traceback": traceback.format_exc()},
            )
            return self._equity_cache

    def _compute_entry_price(self, trig: Trigger, b, cfg: Optional[dict] = None) -> Tuple[Optional[float], str]:
        cfg = cfg or self._cfg_for_side(trig.side)
        close15 = trig.close15
        entry_cfg = cfg.get("entry", {})
        min_pullback = float(entry_cfg.get("min_pullback_bps", 0.0)) / 10000.0
        strict = bool(entry_cfg.get("strict_below_close_for_all", False))
        symmetric = bool(entry_cfg.get("symmetric_close_rule", True))
        bt_cfg = cfg.get("backtest", {})
        entry_offset_bps = float(bt_cfg.get("entry_offset_bps", self._entry_offset_bps) or 0.0)
        tol_bps = float(bt_cfg.get("slippage_bps_base", bt_cfg.get("slippage_bps", self._entry_tol_bps * 10000.0)) or 0.0)
        entry_offset = entry_offset_bps / 10000.0
        tol = tol_bps / 10000.0
        if close15 <= 0:
            return None, "entry_price_close_invalid"
        mid = (b.bid + b.ask) / 2.0
        if mid <= 0:
            return None, "entry_price_no_book"

        if trig.side == "BUY":
            base = mid * (1.0 - entry_offset)
            cap = close15 * (1.0 + tol)
            px = min(base, cap)
            if min_pullback > 0:
                px = min(px, close15 * (1.0 - min_pullback))
            return (px, "") if px > 0 else (None, "entry_price_non_positive")

        # SELL / short
        if strict:
            return None, "entry_price_strict_rule"
        if not symmetric:
            return None, "entry_price_symmetric_disabled"
        base = mid * (1.0 + entry_offset)
        floor = close15 * (1.0 - tol)
        px = max(base, floor)
        if min_pullback > 0:
            px = max(px, close15 * (1.0 + min_pullback))
        return (px, "") if px > 0 else (None, "entry_price_non_positive")

    def _preflight_order(self, symbol: str, side: str, qty: float, price: float,
                         spec: SymbolSpec, notional: float, cfg: dict) -> bool:
        if qty <= 0 or price <= 0 or notional <= 0:
            self._record_reject("non_positive_qty_price_notional", symbol, side)
            return False
        if qty < spec.min_qty:
            self._record_reject(
                "qty_below_min_qty",
                symbol,
                side,
                f"qty={qty:.6f} min_qty={spec.min_qty:.6f}",
            )
            return False
        min_notional_cfg = float(cfg.get("position", {}).get("min_notional_usdt", 0.0) or 0.0)
        min_notional = max(float(spec.min_notional or 0.0), min_notional_cfg)
        if min_notional > 0 and notional < min_notional:
            self._record_reject(
                "notional_below_min_notional",
                symbol,
                side,
                f"notional={notional:.6f} min_notional={min_notional:.6f}",
            )
            return False
        return True

    def _record_reject(self, reason: str, symbol: str, side: str, extra: str = "") -> None:
        stats = self._reject_stats
        stats["reason_counts"][reason] += 1
        stats["symbol_counts"][symbol] += 1
        stats["reason_symbol_counts"][reason][symbol] += 1
        if extra and reason not in stats["examples"]:
            stats["examples"][reason] = f"{symbol} {side} {extra}".strip()
        stats["total_reason_counts"][reason] += 1
        stats["total_symbol_counts"][symbol] += 1
        if extra and reason not in stats["total_examples"]:
            stats["total_examples"][reason] = f"{symbol} {side} {extra}".strip()
        snap = self._bb_snapshot.get((symbol, side))
        bb_hit = snap.get("bb_hit") if snap else None
        bb_upper = snap.get("bb_upper") if snap else None
        bb_lower = snap.get("bb_lower") if snap else None
        note = extra or (snap.get("bb_note") if snap else None)
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self.store.insert_signal_debug(
                ts=utc_ms(),
                symbol=symbol,
                side=side,
                event="ORDER_REJECT",
                reason=reason,
                bb_hit=bb_hit,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                note=note,
            ))
        except RuntimeError:
            pass
        except Exception as e:
            self.logger.warning("event=signal_debug_reject_log_failed symbol=%s err=%s", symbol, e)

    def _fee_rate(self, is_taker: bool) -> float:
        bt_cfg = self.config.get("backtest", {})
        key = "taker_fee" if is_taker else "maker_fee"
        return float(bt_cfg.get(key, 0.0) or 0.0)

    def _estimate_fee(self, qty: float, price: float, is_taker: Optional[bool]) -> Optional[float]:
        if is_taker is None or qty <= 0 or price <= 0:
            return None
        rate = self._fee_rate(bool(is_taker))
        if rate <= 0:
            return None
        return qty * price * rate

    def _update_order_fee_cache(self, upd: OrderUpdate) -> Tuple[Optional[float], Optional[str]]:
        order_id = int(upd.order_id or 0)
        if order_id <= 0:
            return (None, None)
        if upd.executed_qty <= 0:
            return (None, None)
        fee_val = upd.fee
        fee_asset = (upd.fee_asset or "").strip() if isinstance(upd.fee_asset, str) else upd.fee_asset
        if fee_val == 0:
            fee_val = None
        if fee_val is None and not fee_asset:
            return (None, None)
        entry = self._order_fee_cache.get(order_id)
        if entry is None:
            entry = {"fee": 0.0, "fee_asset": None, "has_fee": False}
            self._order_fee_cache[order_id] = entry
        if fee_val is not None:
            entry["fee"] += float(fee_val)
            entry["has_fee"] = True
        if fee_asset:
            if entry["fee_asset"] and entry["fee_asset"] != fee_asset:
                self.logger.warning(
                    "event=order_fee_asset_mismatch order_id=%s prev=%s new=%s",
                    order_id,
                    entry["fee_asset"],
                    fee_asset,
                )
            entry["fee_asset"] = fee_asset
        fee_out = entry["fee"] if entry.get("has_fee") else None
        return (fee_out, entry.get("fee_asset"))

    async def _log_trade_event(self, *, symbol: str, side: str, event: str, qty: float, price: float,
                               status: str, reason: str, order_id: Optional[int] = None,
                               position_id: int = 0, entry_price: Optional[float] = None,
                               stop_price: Optional[float] = None, peak_price: Optional[float] = None,
                               tp_state: Optional[int] = None, pnl: Optional[float] = None,
                               pnl_pct: Optional[float] = None, trade_type: Optional[str] = None,
                               is_add_on: Optional[bool] = None, signal_price: Optional[float] = None,
                               slippage_bps: Optional[float] = None, fee: Optional[float] = None,
                               fee_asset: Optional[str] = None) -> None:
        if entry_price is None:
            entry_price = price
        if trade_type is None:
            trade_type = self._trade_type_from_reason(event, reason)
        if is_add_on is None:
            is_add_on = (event == "ADD")
        await self._safe_store(
            self.store.insert_trade_log(
                ts=utc_ms(),
                position_id=position_id,
                symbol=symbol,
                side=side,
                event=event,
                trade_type=trade_type,
                is_add_on=1 if is_add_on else 0,
                qty=qty,
                price=price,
                order_id=order_id or 0,
                status=status,
                reason=reason,
                entry_price=entry_price if entry_price is not None else 0.0,
                stop_price=stop_price if stop_price is not None else 0.0,
                peak_price=peak_price if peak_price is not None else 0.0,
                tp_state=tp_state if tp_state is not None else 0,
                pnl=pnl if pnl is not None else 0.0,
                pnl_pct=pnl_pct if pnl_pct is not None else 0.0,
                signal_price=signal_price,
                slippage_bps=slippage_bps,
                fee=fee,
                fee_asset=fee_asset,
            ),
            stage="trade_log",
            event="trade_log_failed",
            symbol=symbol,
            side=side,
            order_id=order_id,
            position_id=position_id,
        )

    def _pos_key(self, symbol: str, side: str) -> Tuple[str, str]:
        return (symbol, side)

    def _get_position(self, symbol: str, side: str) -> Optional[PositionState]:
        return self.positions.get(self._pos_key(symbol, side))

    def _has_position(self, symbol: str, side: Optional[str] = None) -> bool:
        if side is not None:
            return self._pos_key(symbol, side) in self.positions
        return any(ps.symbol == symbol for ps in self.positions.values())

    def _has_both_sides(self, symbol: str) -> bool:
        return self._has_position(symbol, "BUY") and self._has_position(symbol, "SELL")

    def _stop_order_active(self, state: Optional[StopOrderState]) -> bool:
        if not state:
            return False
        return state.status not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED")

    def _tp_order_active(self, state: Optional[TpOrderState]) -> bool:
        if not state:
            return False
        return state.status not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED")

    def _format_stop_price(self, price: float, tick_size: float, stop_side: str) -> Tuple[float, str]:
        rounding = ROUND_UP if stop_side == "SELL" else ROUND_DOWN
        price_str = format_by_step(price, tick_size, rounding=rounding)
        try:
            price_val = float(price_str)
        except (TypeError, ValueError):
            price_val = float(price)
        return price_val, price_str

    def _price_for_add_on(self, symbol: str) -> float:
        b = self.market_ws.books.get(symbol)
        m = self.market_ws.marks.get(symbol)
        now_ms = utc_ms()
        px = 0.0
        if self.config["risk"]["use_mark_price_for_risk"]:
            if m and m.mark > 0 and (now_ms - m.ts) <= self._max_stale_price_ms:
                px = m.mark
            elif b and b.bid > 0 and b.ask > 0 and (now_ms - b.ts) <= self._max_stale_price_ms:
                px = (b.bid + b.ask) / 2.0
        else:
            if b and b.bid > 0 and b.ask > 0 and (now_ms - b.ts) <= self._max_stale_price_ms:
                px = (b.bid + b.ask) / 2.0
            elif m and m.mark > 0 and (now_ms - m.ts) <= self._max_stale_price_ms:
                px = m.mark
        if px <= 0:
            cached = self._last_px_cache.get(symbol)
            if cached and (now_ms - cached[1]) <= self._max_stale_price_ms:
                px = cached[0]
        return float(px or 0.0)

    def _profit_pct(self, ps: PositionState, px: float) -> Optional[float]:
        if ps.entry_price <= 0 or px <= 0:
            return None
        if ps.side == "BUY":
            return px / ps.entry_price - 1.0
        return ps.entry_price / px - 1.0

    def _next_tp_price(self, ps: PositionState, px: float, cfg: Optional[dict] = None) -> float:
        cfg = cfg or self._cfg_for_side(ps.side)
        mode = cfg["takeprofit"]["mode"]
        hard_stop = float(cfg["risk"]["hard_stop_pct"])
        if mode == "staged_partial":
            levels = cfg["takeprofit"]["staged_partial"]["levels"]
            if ps.tp_state < len(levels):
                r_mult = float(levels[ps.tp_state]["r_multiple"])
                tgt = ps.entry_price * (1.0 + r_mult * hard_stop) if ps.side == "BUY" else ps.entry_price * (1.0 - r_mult * hard_stop)
                return float(tgt)
            return 0.0
        tp_cfg = cfg["takeprofit"]["trailing_ladder"]
        p1, p2, p3 = float(tp_cfg["p1"]), float(tp_cfg["p2"]), float(tp_cfg["p3"])
        prof = self._profit_pct(ps, px) or 0.0
        if prof < p1:
            return ps.entry_price * (1.0 + p1) if ps.side == "BUY" else ps.entry_price * (1.0 - p1)
        if prof < p2:
            return ps.entry_price * (1.0 + p2) if ps.side == "BUY" else ps.entry_price * (1.0 - p2)
        if prof < p3:
            return ps.entry_price * (1.0 + p3) if ps.side == "BUY" else ps.entry_price * (1.0 - p3)
        return 0.0

    def _dynamic_time_exit_ms(self, ps: PositionState, px: float) -> int:
        prof = self._profit_pct(ps, px)
        if prof is None:
            return 0
        cfg = self._cfg_for_side(ps.side)
        time_cfg = cfg.get("time_exit", {})
        time_exit_p1 = float(time_cfg.get("profit_1", self._time_exit_p1))
        time_exit_p2 = float(time_cfg.get("profit_2", self._time_exit_p2))
        time_exit_h1 = int(time_cfg.get("hold_1_minutes", self._time_exit_h1))
        time_exit_h2 = int(time_cfg.get("hold_2_minutes", self._time_exit_h2))
        if time_exit_h1 <= 0 or time_exit_h2 <= 0:
            return 0
        if prof < time_exit_p1:
            return int(time_exit_h1 * 60 * 1000)
        if prof < time_exit_p2:
            return int(time_exit_h2 * 60 * 1000)
        return 0

    async def _maybe_roll_add_on(self, symbol: str) -> None:
        now = time.time()
        last = self._roll_add_on_ts.get(symbol, 0.0)
        if (now - last) < self._roll_add_on_interval_s:
            return
        self._roll_add_on_ts[symbol] = now
        sem = asyncio.Semaphore(1)
        await self._poll_symbol(symbol, sem)

    async def _upsert_live_position(self, ps: PositionState, px: float, status: str = "OPEN") -> None:
        pnl = self._calc_pnl(ps, ps.qty, px) if px > 0 else 0.0
        pnl_pct = (self._profit_pct(ps, px) or 0.0) if px > 0 else 0.0
        tp_price = self._next_tp_price(ps, px) if px > 0 else 0.0
        await self._safe_store(
            self.store.upsert_position_live(
                symbol=ps.symbol,
                side=ps.side,
                position_id=ps.open_ts,
                ts=utc_ms(),
                entry_price=ps.entry_price,
                qty=ps.qty,
                last_price=px,
                pnl=pnl,
                pnl_pct=pnl_pct,
                stop_price=ps.stop_price,
                tp_price=tp_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
                status=status,
            ),
            stage="position",
            event="position_upsert_failed",
            symbol=ps.symbol,
            side=ps.side,
            position_id=ps.open_ts,
        )

    async def _remove_position(self, ps: PositionState) -> None:
        await self._clear_stop_order(ps.symbol, ps.side, reason="position_closed")
        await self._clear_tp_orders(ps.symbol, ps.side, reason="position_closed")
        self.positions.pop(self._pos_key(ps.symbol, ps.side), None)
        await self._safe_store(
            self.store.delete_position_live(ps.symbol, ps.side),
            stage="position",
            event="position_delete_failed",
            symbol=ps.symbol,
            side=ps.side,
            position_id=ps.open_ts,
        )
        if not self._has_position(ps.symbol):
            await self._restore_symbol_leverage(ps.symbol)

    def _is_leverage_limit_error(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        code = data.get("code")
        msg = str(data.get("msg", "")).lower()
        return code == -2027 or "maximum allowable position" in msg

    def _is_reduce_only_error(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        code = data.get("code")
        msg = str(data.get("msg", "")).lower()
        return code == -1106 and "reduceonly" in msg

    def _is_post_only_reject(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        code = data.get("code")
        msg = str(data.get("msg", "")).lower()
        if code == -5022:
            return True
        return "post only" in msg and "reject" in msg

    def _is_order_not_found(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        code = data.get("code")
        msg = str(data.get("msg", "")).lower()
        return code == -2011 or "unknown order" in msg

    def _format_api_error(self, data) -> str:
        if isinstance(data, dict):
            code = data.get("code", "")
            msg = data.get("msg", "")
            if code != "" or msg:
                return f"code={code} msg={msg}"
        return f"raw={data}"

    def _current_leverage(self, symbol: str) -> int:
        return int(self._leverage_overrides.get(symbol, self._base_leverage))

    async def _ensure_symbol_leverage(self, symbol: str) -> None:
        if self.dry_run or symbol in self._leverage_initialized:
            return
        lev = int(self._base_leverage)
        try:
            r = await self.client.change_leverage(symbol, lev)
        except Exception as e:
            self.logger.warning("event=leverage_ensure_failed symbol=%s lev=%d err=%s", symbol, lev, e)
            await self._log_error(
                "position_manage",
                event="leverage_ensure_failed",
                symbol=symbol,
                error=e,
                details={
                    "leverage": lev,
                    "traceback": traceback.format_exc(),
                },
            )
            return
        if r.status != 200:
            self.logger.warning(
                "event=leverage_ensure_rejected symbol=%s lev=%d err=%s",
                symbol,
                lev,
                self._format_api_error(r.data),
            )
            await self._log_error(
                "position_manage",
                event="leverage_ensure_rejected",
                symbol=symbol,
                reason="api_rejected",
                details={
                    "leverage": lev,
                    "status": r.status,
                    "response": r.data,
                },
            )
            return
        self._leverage_initialized.add(symbol)

    async def _position_size_on_exchange(self, symbol: str, side: str) -> float:
        if self.dry_run:
            return 0.0
        try:
            r = await self.client.position_risk(symbol=symbol)
        except Exception as e:
            self.logger.warning("event=position_risk_failed symbol=%s err=%s", symbol, e)
            await self._log_error(
                "position_manage",
                event="position_risk_failed",
                symbol=symbol,
                error=e,
                details={"traceback": traceback.format_exc()},
            )
            return 0.0
        if r.status != 200:
            self.logger.warning(
                "event=position_risk_rejected symbol=%s err=%s",
                symbol,
                self._format_api_error(r.data),
            )
            await self._log_error(
                "position_manage",
                event="position_risk_rejected",
                symbol=symbol,
                reason="api_rejected",
                details={
                    "status": r.status,
                    "response": r.data,
                },
            )
            return 0.0
        data = r.data
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return 0.0
        want_side = "LONG" if side == "BUY" else "SHORT"
        for entry in data:
            if entry.get("symbol") != symbol:
                continue
            pos_side = str(entry.get("positionSide", "")).upper()
            try:
                amt = float(entry.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            if pos_side:
                if pos_side == want_side:
                    return abs(amt)
                if pos_side == "BOTH":
                    if side == "BUY" and amt > 0:
                        return abs(amt)
                    if side == "SELL" and amt < 0:
                        return abs(amt)
                continue
            if side == "BUY" and amt > 0:
                return abs(amt)
            if side == "SELL" and amt < 0:
                return abs(amt)
        return 0.0

    async def _sync_positions_for_symbol(self, symbol: str) -> None:
        if self.dry_run:
            return
        now = time.time()
        last = self._position_sync_ts.get(symbol, 0.0)
        if (now - last) < self._position_sync_interval_s:
            return
        self._position_sync_ts[symbol] = now
        try:
            r = await self.client.position_risk(symbol=symbol)
        except Exception as e:
            self.logger.warning("event=position_sync_failed symbol=%s err=%s", symbol, e)
            await self._log_error(
                "position_manage",
                event="position_sync_failed",
                symbol=symbol,
                error=e,
                details={"traceback": traceback.format_exc()},
            )
            return
        if r.status != 200:
            self.logger.warning(
                "event=position_sync_rejected symbol=%s err=%s",
                symbol,
                self._format_api_error(r.data),
            )
            await self._log_error(
                "position_manage",
                event="position_sync_rejected",
                symbol=symbol,
                reason="api_rejected",
                details={
                    "status": r.status,
                    "response": r.data,
                },
            )
            return
        data = r.data
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return

        seen: Set[str] = set()
        for entry in data:
            if entry.get("symbol") != symbol:
                continue
            pos_side = str(entry.get("positionSide", "")).upper()
            try:
                amt = float(entry.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            try:
                entry_px = float(entry.get("entryPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_px = 0.0
            if pos_side in ("LONG", "SHORT"):
                side = "BUY" if pos_side == "LONG" else "SELL"
                seen.add(side)
                await self._sync_position_from_exchange(symbol, side, amt, entry_px)
                continue
            if amt == 0:
                continue
            side = "BUY" if amt > 0 else "SELL"
            seen.add(side)
            await self._sync_position_from_exchange(symbol, side, amt, entry_px)

        for side in ("BUY", "SELL"):
            if side in seen:
                continue
            local = self._get_position(symbol, side)
            if local:
                self.logger.warning(
                    "event=position_sync_exchange_empty symbol=%s side=%s action=clear_local",
                    symbol,
                    side,
                )
                await self._remove_position(local)

    async def _sync_position_from_exchange(self, symbol: str, side: str, amt: float, entry_px: float) -> None:
        qty = abs(float(amt or 0.0))
        local = self._get_position(symbol, side)
        if qty <= 0:
            if local:
                self.logger.warning(
                    "event=position_sync_qty_zero symbol=%s side=%s action=clear_local",
                    symbol,
                    side,
                )
                await self._remove_position(local)
            return
        if entry_px <= 0:
            entry_px = local.entry_price if local else 0.0
        cfg = self._cfg_for_side(side)
        hard_stop = float(cfg["risk"]["hard_stop_pct"])
        lev = float(cfg["position"]["leverage"])
        buffer_pct = float(cfg["risk"].get("liquidation_buffer_pct", 0.0))
        max_stop_pct = 0.0
        if lev > 0:
            max_stop_pct = 1.0 / lev - buffer_pct
        eff_stop = hard_stop if max_stop_pct <= 0 else min(hard_stop, max_stop_pct)
        new_stop = entry_px * (1.0 - eff_stop) if side == "BUY" else entry_px * (1.0 + eff_stop)

        if local:
            local.qty = qty
            if entry_px > 0:
                local.entry_price = entry_px
            if side == "BUY":
                local.stop_price = max(local.stop_price, new_stop)
                local.peak_price = max(local.peak_price, entry_px)
            else:
                local.stop_price = min(local.stop_price, new_stop)
                local.peak_price = min(local.peak_price, entry_px)
            px = self._price_for_add_on(symbol) or entry_px
            await self._upsert_live_position(local, px, status="SYNC")
            await self._ensure_stop_order(local, px, reason="position_sync")
            await self._ensure_tp_orders(local, reason="position_sync")
            return

        ps = PositionState(symbol, side, qty, entry_px, entry_px, new_stop, 0, utc_ms())
        self.positions[self._pos_key(symbol, side)] = ps
        await self._upsert_live_position(ps, entry_px, status="SYNC")
        await self._ensure_stop_order(ps, entry_px, reason="position_sync")
        await self._ensure_tp_orders(ps, reason="position_sync")

    async def _sync_positions_full(self) -> bool:
        if self.dry_run:
            return True
        try:
            r = await self.client.position_risk()
        except Exception as e:
            self.logger.warning("event=position_sync_full_failed err=%s", e)
            await self._log_error(
                "position_manage",
                event="position_sync_full_failed",
                error=e,
                details={"traceback": traceback.format_exc()},
            )
            return False
        if r.status != 200:
            self.logger.warning("event=position_sync_full_rejected err=%s", self._format_api_error(r.data))
            await self._log_error(
                "position_manage",
                event="position_sync_full_rejected",
                reason="api_rejected",
                details={
                    "status": r.status,
                    "response": r.data,
                },
            )
            return False
        data = r.data
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return False

        seen: Set[Tuple[str, str]] = set()
        for entry in data:
            sym = entry.get("symbol")
            if not sym or sym not in self.universe:
                continue
            pos_side = str(entry.get("positionSide", "")).upper()
            try:
                amt = float(entry.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                amt = 0.0
            if amt == 0:
                continue
            try:
                entry_px = float(entry.get("entryPrice", 0.0) or 0.0)
            except (TypeError, ValueError):
                entry_px = 0.0
            if pos_side in ("LONG", "SHORT"):
                side = "BUY" if pos_side == "LONG" else "SELL"
                seen.add((sym, side))
                await self._sync_position_from_exchange(sym, side, amt, entry_px)
                continue
            side = "BUY" if amt > 0 else "SELL"
            seen.add((sym, side))
            await self._sync_position_from_exchange(sym, side, amt, entry_px)

        for (sym, side), ps in list(self.positions.items()):
            if sym not in self.universe:
                continue
            if (sym, side) in seen:
                continue
            await self._remove_position(ps)
        return True

    async def _wait_for_account_update(self, timeout_s: Optional[float] = None) -> bool:
        if not self._use_account_updates or self.dry_run:
            return True
        ready = getattr(self.user_ws, "account_ready", None)
        if ready is None:
            return True
        wait_s = self._account_update_wait_s if timeout_s is None else float(timeout_s)
        try:
            await asyncio.wait_for(ready.wait(), timeout=wait_s)
            self.logger.info("event=account_update_ready")
            return True
        except asyncio.TimeoutError:
            self.logger.warning(
                "event=account_update_wait_timeout wait_s=%.1f",
                wait_s,
            )
        ok = await self._sync_positions_full()
        if ok:
            ready.set()
            self.logger.info("event=account_update_fallback_sync_ok")
            return True
        self.logger.error("event=account_update_fallback_sync_failed")
        return False

    async def _account_update_consumer(self) -> None:
        while True:
            try:
                updates = await self.user_ws.account_updates.get()
                await self._apply_account_update(updates)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._log_error(
                    "position_manage",
                    event="account_update_consumer_error",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )

    async def _apply_account_update(self, updates: List[AccountPositionUpdate]) -> None:
        if not updates:
            return
        for upd in updates:
            sym = upd.symbol
            if sym not in self.universe and not self._has_position(sym):
                continue
            pos_side = str(upd.position_side or "").upper()
            amt = float(upd.position_amt or 0.0)
            entry_px = float(upd.entry_price or 0.0)
            if pos_side in ("LONG", "SHORT"):
                side = "BUY" if pos_side == "LONG" else "SELL"
                if amt == 0.0:
                    local = self._get_position(sym, side)
                    if local:
                        await self._remove_position(local)
                    continue
                await self._sync_position_from_exchange(sym, side, amt, entry_px)
                continue
            if amt == 0.0:
                for side in ("BUY", "SELL"):
                    local = self._get_position(sym, side)
                    if local:
                        await self._remove_position(local)
                continue
            side = "BUY" if amt > 0 else "SELL"
            await self._sync_position_from_exchange(sym, side, amt, entry_px)

    async def _position_sync_loop(self) -> None:
        while True:
            await asyncio.sleep(self._position_sync_interval_s)
            try:
                await self._sync_positions_full()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._log_error(
                    "position_manage",
                    event="position_sync_loop_error",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )
    async def _sync_position_after_failure(self, ps: PositionState, event: str,
                                           reason: str, price: float) -> bool:
        if self.dry_run:
            return False
        exch_qty = await self._position_size_on_exchange(ps.symbol, ps.side)
        spec = self.universe[ps.symbol]
        exch_qty = round_step(exch_qty, spec.step_size)
        if exch_qty <= 0:
            self.logger.warning(
                "event=position_sync_after_failure_no_position action=%s symbol=%s",
                event,
                ps.symbol,
            )
            await self._log_trade_event(
                symbol=ps.symbol,
                side=ps.side,
                event=event,
                qty=ps.qty,
                price=price,
                status="SKIP",
                reason=f"{reason}_NO_POSITION_ON_EXCHANGE",
                position_id=ps.open_ts,
                entry_price=ps.entry_price,
                stop_price=ps.stop_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
            )
            await self._remove_position(ps)
            return True
        if abs(exch_qty - ps.qty) >= spec.step_size:
            old_qty = ps.qty
            ps.qty = exch_qty
            self.logger.warning(
                "event=position_sync_after_failure_qty action=%s symbol=%s qty_from=%.6f qty_to=%.6f",
                event,
                ps.symbol,
                old_qty,
                ps.qty,
            )
            await self._log_trade_event(
                symbol=ps.symbol,
                side=ps.side,
                event=event,
                qty=ps.qty,
                price=price,
                status="SYNC",
                reason=f"{reason}_SYNC_QTY {old_qty:.6f}->{ps.qty:.6f}",
                position_id=ps.open_ts,
                entry_price=ps.entry_price,
                stop_price=ps.stop_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
            )
            await self._ensure_tp_orders(ps, reason="position_sync_after_failure")
        return False

    async def _set_symbol_leverage(self, symbol: str, leverage: int) -> bool:
        if self.dry_run:
            self._leverage_overrides[symbol] = leverage
            return True
        try:
            r = await self.client.change_leverage(symbol, leverage)
        except Exception as e:
            self.logger.warning("event=change_leverage_failed symbol=%s lev=%d err=%s", symbol, leverage, e)
            await self._log_error(
                "position_manage",
                event="change_leverage_failed",
                symbol=symbol,
                error=e,
                details={
                    "leverage": leverage,
                    "traceback": traceback.format_exc(),
                },
            )
            return False
        if r.status != 200:
            self.logger.warning(
                "event=change_leverage_rejected symbol=%s lev=%d err=%s",
                symbol,
                leverage,
                self._format_api_error(r.data),
            )
            await self._log_error(
                "position_manage",
                event="change_leverage_rejected",
                symbol=symbol,
                reason="api_rejected",
                details={
                    "leverage": leverage,
                    "status": r.status,
                    "response": r.data,
                },
            )
            return False
        self._leverage_overrides[symbol] = leverage
        return True

    async def _restore_symbol_leverage(self, symbol: str) -> None:
        if self._has_position(symbol):
            return
        target = int(self._base_leverage)
        current = int(self._leverage_overrides.get(symbol, target))
        if current == target:
            self._leverage_overrides.pop(symbol, None)
            return
        if self.dry_run:
            self._leverage_overrides.pop(symbol, None)
            return
        try:
            r = await self.client.change_leverage(symbol, target)
        except Exception as e:
            self.logger.warning("event=restore_leverage_failed symbol=%s lev=%d err=%s", symbol, target, e)
            await self._log_error(
                "position_manage",
                event="restore_leverage_failed",
                symbol=symbol,
                error=e,
                details={
                    "leverage": target,
                    "traceback": traceback.format_exc(),
                },
            )
            return
        if r.status != 200:
            self.logger.warning(
                "event=restore_leverage_rejected symbol=%s lev=%d err=%s",
                symbol,
                target,
                self._format_api_error(r.data),
            )
            await self._log_error(
                "position_manage",
                event="restore_leverage_rejected",
                symbol=symbol,
                reason="api_rejected",
                details={
                    "leverage": target,
                    "status": r.status,
                    "response": r.data,
                },
            )
            return
        self._leverage_overrides.pop(symbol, None)

    async def _submit_limit_order(self, symbol: str, side: str, order_params: dict) -> Optional[int]:
        resp = await self.client.new_order(**order_params)
        if resp.status == 200:
            return int(resp.data.get("orderId", 0))
        await self._log_error(
            "order_request",
            event="entry_order_rejected",
            symbol=symbol,
            side=side,
            reason="leverage_limit" if self._is_leverage_limit_error(resp.data) else "api_rejected",
            details={
                "status": resp.status,
                "response": resp.data,
                "order_params": order_params,
            },
        )
        if not self._is_leverage_limit_error(resp.data):
            self._record_reject("order_rejected", symbol, side, str(resp.data))
            self.logger.warning(
                "event=entry_rejected symbol=%s side=%s qty=%s price=%s err=%s",
                symbol,
                side,
                order_params.get("quantity"),
                order_params.get("price"),
                self._format_api_error(resp.data),
            )
            await self._log_trade_event(
                symbol=symbol,
                side=side,
                event="ENTRY",
                qty=float(order_params.get("quantity") or 0.0),
                price=float(order_params.get("price") or 0.0),
                status="REJECTED",
                reason=str(resp.data),
            )
            return None

        lev = self._current_leverage(symbol)
        while lev > self._min_leverage:
            lev -= 1
            if lev < self._min_leverage:
                break
            if not await self._set_symbol_leverage(symbol, lev):
                break
            self.logger.warning("event=entry_rejected_leverage_retry symbol=%s lev=%d", symbol, lev)
            resp = await self.client.new_order(**order_params)
            if resp.status == 200:
                self.logger.info("event=entry_accepted_after_leverage_reduce symbol=%s lev=%d", symbol, lev)
                return int(resp.data.get("orderId", 0))
            if not self._is_leverage_limit_error(resp.data):
                await self._log_error(
                    "order_request",
                    event="entry_order_rejected_retry",
                    symbol=symbol,
                    side=side,
                    reason="api_rejected",
                    details={
                        "status": resp.status,
                        "response": resp.data,
                        "order_params": order_params,
                        "leverage": lev,
                    },
                )
                self._record_reject("order_rejected", symbol, side, str(resp.data))
                self.logger.warning(
                    "event=entry_rejected symbol=%s side=%s qty=%s price=%s err=%s",
                    symbol,
                    side,
                    order_params.get("quantity"),
                    order_params.get("price"),
                    self._format_api_error(resp.data),
                )
                return None

        self._record_reject("order_rejected", symbol, side, str(resp.data))
        await self._log_error(
            "order_request",
            event="entry_order_rejected_final",
            symbol=symbol,
            side=side,
            reason="leverage_limit" if self._is_leverage_limit_error(resp.data) else "api_rejected",
            details={
                "status": resp.status,
                "response": resp.data,
                "order_params": order_params,
            },
        )
        self.logger.warning(
            "event=entry_rejected symbol=%s side=%s qty=%s price=%s err=%s",
            symbol,
            side,
            order_params.get("quantity"),
            order_params.get("price"),
            self._format_api_error(resp.data),
        )
        await self._log_trade_event(
            symbol=symbol,
            side=side,
            event="ENTRY",
            qty=float(order_params.get("quantity") or 0.0),
            price=float(order_params.get("price") or 0.0),
            status="REJECTED",
            reason=str(resp.data),
        )
        return None

    async def _reject_summary_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reject_summary_interval_s)
            stats = self._reject_stats
            total = sum(stats["reason_counts"].values())
            total_all = sum(stats["total_reason_counts"].values())
            since_ms = stats["since_ms"]
            if total <= 0:
                self.logger.info("event=trade_block_summary scope=last_10m total=0")
                stats["since_ms"] = utc_ms()
                continue

            top_reasons = stats["reason_counts"].most_common(6)
            top_symbols = stats["symbol_counts"].most_common(6)
            top_reasons_all = stats["total_reason_counts"].most_common(6)
            top_symbols_all = stats["total_symbol_counts"].most_common(6)
            reason_str = ", ".join([f"{k}={v}" for k, v in top_reasons])
            symbol_str = ", ".join([f"{k}={v}" for k, v in top_symbols])
            reason_str_all = ", ".join([f"{k}={v}" for k, v in top_reasons_all])
            symbol_str_all = ", ".join([f"{k}={v}" for k, v in top_symbols_all])
            self.logger.info(
                "event=trade_block_summary scope=last_10m total=%d since_ms=%d reasons=[%s] symbols=[%s]",
                total,
                since_ms,
                reason_str,
                symbol_str,
            )
            if stats["examples"]:
                sample = "; ".join([f"{k}: {v}" for k, v in list(stats["examples"].items())[:6]])
                self.logger.info("event=trade_block_samples scope=last_10m samples=%s", sample)
            self.logger.info(
                "event=trade_block_summary scope=cumulative total=%d since_ms=%d reasons=[%s] symbols=[%s]",
                total_all,
                stats["total_since_ms"],
                reason_str_all,
                symbol_str_all,
            )
            if stats["total_examples"]:
                sample = "; ".join([f"{k}: {v}" for k, v in list(stats["total_examples"].items())[:6]])
                self.logger.info("event=trade_block_samples scope=cumulative samples=%s", sample)

            stats["since_ms"] = utc_ms()
            stats["reason_counts"].clear()
            stats["symbol_counts"].clear()
            stats["reason_symbol_counts"].clear()
            stats["examples"].clear()

    def _position_side_for_order(self, side: str) -> Optional[str]:
        if self._position_mode != "HEDGE":
            return None
        return "LONG" if side == "BUY" else "SHORT"

    def _position_side_for_position(self, side: str) -> Optional[str]:
        if self._position_mode != "HEDGE":
            return None
        return "LONG" if side == "BUY" else "SHORT"

    def _taker_slippage_bps(self, trig: Trigger, b, limit_px: float) -> Tuple[bool, float]:
        if trig.close15 <= 0:
            return False, 0.0
        if trig.side == "BUY":
            if b.ask <= 0 or limit_px < b.ask:
                return False, 0.0
            bps = (b.ask / trig.close15 - 1.0) * 10000.0
            return True, max(0.0, bps)
        if b.bid <= 0 or limit_px > b.bid:
            return False, 0.0
        bps = (1.0 - b.bid / trig.close15) * 10000.0
        return True, max(0.0, bps)

    def _calc_baseline(self, vols) -> float:
        if not vols:
            return 0.0
        if self._baseline_method == "mean":
            return sum(vols) / len(vols)
        q_sorted = sorted(vols)
        return q_sorted[len(q_sorted) // 2]

    def _baseline_lookback(self) -> int:
        vol_cfg = self.config.get("volume", {})
        lookback = int(vol_cfg.get("baseline_lookback_windows", 0) or 0)
        if lookback > 0:
            return lookback
        return int(vol_cfg.get("baseline_lookback_15m", 0) or 0)

    def _time_exit_ref_price(self, symbol: str, last_px: float) -> float:
        mode = self._time_exit_price
        bars = self.market_ws.get_recent_bars(symbol, 1)
        if bars:
            bar = bars[-1]
            if mode == "open":
                ref = bar.open
            elif mode == "mid":
                ref = (bar.high + bar.low) / 2.0
            else:
                ref = bar.close
            if ref > 0:
                return float(ref)

        if last_px > 0:
            return float(last_px)

        b = self.market_ws.books.get(symbol)
        if b and b.bid > 0 and b.ask > 0:
            return float((b.bid + b.ask) / 2.0)

        m = self.market_ws.marks.get(symbol)
        if m and m.mark > 0:
            return float(m.mark)
        return 0.0

    async def _wait_fill_or_timeout(self, symbol: str, order_id: int, ttl_s: int) -> bool:
        deadline = time.time() + ttl_s
        partial_qty = 0.0
        partial_avg = 0.0
        partial_side = ""
        partial_update_time = 0
        order_id_int = int(order_id)
        waiter = self._order_waiters.get(order_id_int)
        if waiter is None:
            waiter = asyncio.Queue(maxsize=50)
            self._order_waiters[order_id_int] = waiter

        async def _apply_update(upd: OrderUpdate) -> Optional[bool]:
            nonlocal partial_qty, partial_avg, partial_side, partial_update_time
            if upd.executed_qty > 0:
                partial_qty = float(upd.executed_qty)
                partial_avg = float(upd.avg_price or upd.price)
                partial_side = upd.side
                partial_update_time = int(upd.update_time or 0)
            if upd.status == "FILLED" and upd.executed_qty > 0:
                await self._on_filled(
                    symbol,
                    upd.side,
                    upd.executed_qty,
                    upd.avg_price or upd.price,
                    order_id=upd.order_id,
                    fill_ts=upd.update_time,
                )
                return True
            if upd.status == "CANCELED" and partial_qty > 0:
                fill_ts = upd.update_time or partial_update_time
                await self._on_filled(
                    symbol,
                    partial_side,
                    partial_qty,
                    partial_avg,
                    order_id=upd.order_id,
                    fill_ts=fill_ts,
                )
                return True
            return None

        try:
            cached = self._order_update_cache.get(order_id_int)
            if cached:
                done = await _apply_update(cached)
                if done:
                    return True
            while time.time() < deadline:
                try:
                    upd = await asyncio.wait_for(waiter.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                if upd.symbol != symbol or upd.order_id != order_id_int:
                    continue
                done = await _apply_update(upd)
                if done:
                    return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._log_error(
                "order_fill",
                event="wait_fill_error",
                symbol=symbol,
                order_id=order_id,
                error=e,
                details={"traceback": traceback.format_exc()},
            )
            raise
        finally:
            self._order_waiters.pop(order_id_int, None)
            self._order_update_cache.pop(order_id_int, None)
            self._order_fee_cache.pop(order_id_int, None)
        if partial_qty > 0:
            if not self.dry_run:
                await self.client.cancel_order(symbol=symbol, orderId=order_id)
            fill_ts = partial_update_time or utc_ms()
            await self._on_filled(
                symbol,
                partial_side,
                partial_qty,
                partial_avg,
                order_id=order_id,
                fill_ts=fill_ts,
            )
            return True
        return False

    async def _on_filled(self, symbol: str, side: str, qty: float, fill_price: float,
                         order_id: Optional[int] = None, signal_ts: Optional[int] = None,
                         signal_price: Optional[float] = None, fill_ts: Optional[int] = None,
                         order_accept_ts: Optional[int] = None, is_taker: Optional[bool] = None):
        now_ms = int(fill_ts or utc_ms())
        if (signal_ts is None or signal_price is None or order_accept_ts is None) and order_id is not None:
            meta = self._signal_meta_by_order.pop(int(order_id), None)
            if meta:
                if len(meta) == 4:
                    signal_ts, signal_price, order_accept_ts, is_taker = meta
                else:
                    signal_ts, signal_price, order_accept_ts = meta
        if order_accept_ts is None:
            order_accept_ts = now_ms
        slippage_bps = None
        if signal_ts is not None:
            latency_order_ms = max(0, int(order_accept_ts) - int(signal_ts))
            latency_fill_ms = max(0, now_ms - int(signal_ts))
            order_label = order_id if order_id is not None else "SIM"
            self.logger.info(
                "event=entry_latency symbol=%s side=%s order_id=%s latency_ms=%d",
                symbol,
                side,
                order_label,
                latency_fill_ms,
            )
            if signal_price and signal_price > 0 and fill_price > 0:
                if side == "BUY":
                    slippage_bps = (fill_price / signal_price - 1.0) * 10000.0
                else:
                    slippage_bps = (signal_price / fill_price - 1.0) * 10000.0
            await self._safe_store(
                self.store.insert_entry_latency(
                    signal_ts=int(signal_ts),
                    fill_ts=now_ms,
                    latency_order_ms=int(latency_order_ms),
                    latency_fill_ms=int(latency_fill_ms),
                    symbol=symbol,
                    side=side,
                    order_id=int(order_id or 0),
                    signal_price=float(signal_price or 0.0),
                    fill_price=float(fill_price),
                    slippage_bps=slippage_bps,
                    order_accept_ts=int(order_accept_ts),
                ),
                stage="order_fill",
                event="entry_latency_failed",
                symbol=symbol,
                side=side,
                order_id=order_id,
            )
        fee = None
        fee_asset = None
        if order_id:
            cached_fee = self._order_fee_cache.get(int(order_id))
            if cached_fee:
                if cached_fee.get("has_fee"):
                    fee = float(cached_fee.get("fee", 0.0))
                fee_asset = cached_fee.get("fee_asset")
        if fee is None:
            fee = self._estimate_fee(qty, fill_price, is_taker)
        cfg = self._cfg_for_side(side)
        hard_stop = float(cfg["risk"]["hard_stop_pct"])
        lev = float(cfg["position"]["leverage"])
        buffer_pct = float(cfg["risk"].get("liquidation_buffer_pct", 0.0))
        max_stop_pct = 0.0
        if lev > 0:
            max_stop_pct = 1.0 / lev - buffer_pct
        eff_stop = hard_stop if max_stop_pct <= 0 else min(hard_stop, max_stop_pct)
        stop = fill_price * (1.0 - eff_stop) if side == "BUY" else fill_price * (1.0 + eff_stop)

        existing = self._get_position(symbol, side)
        if existing:
            new_qty = existing.qty + qty
            if new_qty <= 0:
                return
            avg_entry = (existing.entry_price * existing.qty + fill_price * qty) / new_qty
            new_stop = avg_entry * (1.0 - eff_stop) if side == "BUY" else avg_entry * (1.0 + eff_stop)
            if side == "BUY":
                existing.stop_price = max(existing.stop_price, new_stop)
                existing.peak_price = max(existing.peak_price, fill_price)
            else:
                existing.stop_price = min(existing.stop_price, new_stop)
                existing.peak_price = min(existing.peak_price, fill_price)
            existing.entry_price = avg_entry
            existing.qty = new_qty
            await self._log_trade_event(
                symbol=symbol,
                side=side,
                event="ADD",
                qty=qty,
                price=fill_price,
                status="SIMULATED" if self.dry_run else "FILLED",
                reason="ADD_ON",
                order_id=order_id,
                position_id=existing.open_ts,
                entry_price=existing.entry_price,
                stop_price=existing.stop_price,
                peak_price=existing.peak_price,
                tp_state=existing.tp_state,
                pnl=0.0,
                pnl_pct=0.0,
                signal_price=signal_price,
                slippage_bps=slippage_bps,
                fee=fee,
                fee_asset=fee_asset,
            )
            await self._upsert_live_position(existing, fill_price)
            self.logger.info(
                "event=position_add symbol=%s side=%s qty_add=%.6f avg=%.8f stop=%.8f",
                symbol,
                side,
                qty,
                existing.entry_price,
                existing.stop_price,
            )
            await self._ensure_stop_order(existing, fill_price, reason="entry_add")
            await self._ensure_tp_orders(existing, reason="entry_add")
            return

        ps = PositionState(symbol, side, qty, fill_price, fill_price, stop, 0, utc_ms())
        self.positions[self._pos_key(symbol, side)] = ps
        await self._upsert_live_position(ps, fill_price)
        await self._log_trade_event(
            symbol=symbol,
            side=side,
            event="ENTRY",
            qty=qty,
            price=fill_price,
            status="SIMULATED" if self.dry_run else "FILLED",
            reason="ENTRY",
            order_id=order_id,
            position_id=ps.open_ts,
            entry_price=fill_price,
            stop_price=ps.stop_price,
            peak_price=ps.peak_price,
            tp_state=ps.tp_state,
            pnl=0.0,
            pnl_pct=0.0,
            signal_price=signal_price,
            slippage_bps=slippage_bps,
            fee=fee,
            fee_asset=fee_asset,
        )
        self.logger.info(
            "event=position_open symbol=%s side=%s qty=%.6f entry=%.8f stop=%.8f",
            symbol,
            side,
            qty,
            fill_price,
            stop,
        )
        await self._ensure_stop_order(ps, fill_price, reason="entry_open")
        await self._ensure_tp_orders(ps, reason="entry_open")

    async def _order_update_consumer(self):
        while True:
            try:
                upd = await self.user_ws.order_updates.get()
                order_id_int = int(upd.order_id)
                self._order_update_cache[order_id_int] = upd
                fee_val, fee_asset = self._update_order_fee_cache(upd)
                if fee_val is not None or fee_asset is not None:
                    await self._safe_store(
                        self.store.update_trade_log_fee(order_id=order_id_int, fee=fee_val, fee_asset=fee_asset),
                        stage="trade_log",
                        event="trade_log_fee_update_failed",
                        symbol=upd.symbol,
                        side=upd.side,
                        order_id=order_id_int,
                    )
                waiter = self._order_waiters.get(order_id_int)
                if waiter:
                    try:
                        waiter.put_nowait(upd)
                    except asyncio.QueueFull:
                        try:
                            waiter.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            waiter.put_nowait(upd)
                        except asyncio.QueueFull:
                            pass
                try:
                    await self.store.upsert_order(
                        ts=utc_ms(),
                        symbol=upd.symbol,
                        client_order_id=upd.client_order_id,
                        order_id=upd.order_id,
                        side=upd.side,
                        type=upd.order_type,
                        price=upd.price,
                        qty=upd.orig_qty,
                        status=upd.status,
                        avg_price=upd.avg_price,
                        executed_qty=upd.executed_qty,
                    )
                except Exception as e:
                    await self._log_error(
                        "order_fill",
                        event="order_update_store_failed",
                        symbol=upd.symbol,
                        side=upd.side,
                        order_id=upd.order_id,
                        error=e,
                        details={
                            "status": upd.status,
                            "traceback": traceback.format_exc(),
                        },
                    )
                await self._handle_stop_order_update(upd)
                await self._handle_tp_order_update(upd)
                if upd.status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") and order_id_int not in self._order_waiters:
                    self._order_fee_cache.pop(order_id_int, None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._log_error(
                    "order_fill",
                    event="order_update_consumer_error",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )

    async def _cancel_stop_order(self, symbol: str, order_id: int, reason: str = "") -> bool:
        if order_id <= 0:
            return False
        if self.dry_run:
            return True
        try:
            r = await self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            self.logger.warning(
                "event=stop_order_cancel_failed symbol=%s order_id=%s reason=%s err=%s",
                symbol,
                order_id,
                reason,
                e,
            )
            await self._log_error(
                "stop_loss",
                event="stop_order_cancel_failed",
                symbol=symbol,
                reason=reason,
                error=e,
                details={"order_id": order_id, "traceback": traceback.format_exc()},
            )
            return False
        if r.status == 200:
            return True
        if self._is_order_not_found(r.data):
            return True
        try:
            r = await self.client.cancel_algo_order(symbol=symbol, algoId=order_id)
        except Exception:
            r = None
        if r is not None:
            if r.status == 200 or self._is_order_not_found(r.data):
                return True
        self.logger.warning(
            "event=stop_order_cancel_rejected symbol=%s order_id=%s reason=%s err=%s",
            symbol,
            order_id,
            reason,
            self._format_api_error(r.data),
        )
        await self._log_error(
            "stop_loss",
            event="stop_order_cancel_rejected",
            symbol=symbol,
            reason=reason,
            details={
                "order_id": order_id,
                "status": r.status,
                "response": r.data,
            },
        )
        return False

    async def _clear_stop_order(self, symbol: str, side: str, reason: str = "") -> None:
        key = self._pos_key(symbol, side)
        state = self._stop_orders.get(key)
        if not state:
            return
        self._stop_order_ids.pop(state.order_id, None)
        if state.client_order_id:
            self._stop_order_client_ids.pop(state.client_order_id, None)
        if self._stop_order_active(state):
            await self._cancel_stop_order(symbol, state.order_id, reason=reason or "clear_stop")
        self._stop_orders.pop(key, None)

    async def _cancel_tp_order(self, symbol: str, order_id: int, reason: str = "") -> bool:
        if order_id <= 0:
            return False
        if self.dry_run:
            return True
        try:
            r = await self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            self.logger.warning(
                "event=tp_order_cancel_failed symbol=%s order_id=%s reason=%s err=%s",
                symbol,
                order_id,
                reason,
                e,
            )
            await self._log_error(
                "take_profit",
                event="tp_order_cancel_failed",
                symbol=symbol,
                reason=reason,
                error=e,
                details={"order_id": order_id, "traceback": traceback.format_exc()},
            )
            return False
        if r.status == 200:
            return True
        if self._is_order_not_found(r.data):
            return True
        self.logger.warning(
            "event=tp_order_cancel_rejected symbol=%s order_id=%s reason=%s err=%s",
            symbol,
            order_id,
            reason,
            self._format_api_error(r.data),
        )
        await self._log_error(
            "take_profit",
            event="tp_order_cancel_rejected",
            symbol=symbol,
            reason=reason,
            details={
                "order_id": order_id,
                "status": r.status,
                "response": r.data,
            },
        )
        return False

    async def _clear_tp_orders(self, symbol: str, side: str, reason: str = "") -> None:
        keys = [k for k in self._tp_orders if k[0] == symbol and k[1] == side]
        for key in keys:
            state = self._tp_orders.get(key)
            if not state:
                continue
            self._tp_order_ids.pop(state.order_id, None)
            if self._tp_order_active(state):
                await self._cancel_tp_order(symbol, state.order_id, reason=reason or "clear_tp")
            self._tp_orders.pop(key, None)

    async def _ensure_stop_order(self, ps: PositionState, px: float, reason: str = "") -> bool:
        if ps.qty <= 0 or ps.stop_price <= 0:
            return False
        spec = self.universe.get(ps.symbol)
        if not spec:
            return False
        stop_side = "SELL" if ps.side == "BUY" else "BUY"
        stop_price, _ = self._format_stop_price(ps.stop_price, spec.tick_size, stop_side)
        if stop_price <= 0:
            return False
        qty = round_step(ps.qty, spec.step_size)
        if qty <= 0 or qty < spec.min_qty:
            return False
        key = self._pos_key(ps.symbol, ps.side)
        state = self._stop_orders.get(key)
        if state and not self._stop_order_active(state):
            self._stop_order_ids.pop(state.order_id, None)
            if state.client_order_id:
                self._stop_order_client_ids.pop(state.client_order_id, None)
            self._stop_orders.pop(key, None)
            state = None
        if state:
            remaining_qty = state.remaining_qty()
            same_price = abs(state.stop_price - stop_price) <= (spec.tick_size * 0.5)
            same_qty = abs(remaining_qty - qty) <= (spec.step_size * 0.5)
            if same_price and same_qty:
                return True
            if state.order_id > 0:
                self._stop_order_ids.pop(state.order_id, None)
            if state.client_order_id:
                self._stop_order_client_ids.pop(state.client_order_id, None)

        self._stop_orders[key] = StopOrderState(
            order_id=0,
            stop_price=stop_price,
            orig_qty=qty,
            executed_qty=0.0,
            status="LOCAL",
            created_ts=utc_ms(),
            client_order_id="",
        )
        return True

    def _tp_preplace_enabled(self, cfg: dict) -> bool:
        if not isinstance(cfg, dict):
            return False
        tp_cfg = cfg.get("takeprofit", {}) or {}
        enabled = bool(tp_cfg.get("preplace_orders", False))
        return enabled and not self.dry_run

    def _tp_maker_offset_ticks(self, cfg: dict) -> float:
        tp_cfg = cfg.get("takeprofit", {}) or {}
        try:
            return float(tp_cfg.get("maker_offset_ticks", 1) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _tp_post_only_retry_settings(self, cfg: dict) -> Tuple[int, float]:
        tp_cfg = cfg.get("takeprofit", {}) or {}
        try:
            retries = int(tp_cfg.get("post_only_retry_times", 2) or 0)
        except (TypeError, ValueError):
            retries = 0
        try:
            tick_step = float(tp_cfg.get("post_only_retry_ticks", 1) or 0.0)
        except (TypeError, ValueError):
            tick_step = 0.0
        return max(0, retries), max(0.0, tick_step)

    def _tp_limit_price(self, symbol: str, side: str, target_px: float, tick: float, offset_ticks: float) -> float:
        if target_px <= 0:
            return 0.0
        b = self.market_ws.books.get(symbol)
        if not b or b.bid <= 0 or b.ask <= 0 or tick <= 0 or offset_ticks <= 0:
            return target_px
        if side == "SELL":
            return max(target_px, b.ask + tick * offset_ticks)
        bid_px = b.bid - tick * offset_ticks
        if bid_px <= 0:
            return min(target_px, b.bid)
        return min(target_px, bid_px)

    async def _ensure_tp_orders(self, ps: PositionState, reason: str = "") -> None:
        cfg = self._cfg_for_side(ps.side)
        if not self._tp_preplace_enabled(cfg):
            return
        tp_cfg = cfg.get("takeprofit", {}) or {}
        mode = str(tp_cfg.get("mode", "trailing_ladder")).lower()
        if mode not in ("staged_partial", "hybrid", "staged_partial_trailing", "staged_partial+trailing"):
            return
        levels = (tp_cfg.get("staged_partial", {}) or {}).get("levels", [])
        if not levels:
            return
        spec = self.universe.get(ps.symbol)
        if not spec:
            return
        hard_stop = float(cfg.get("risk", {}).get("hard_stop_pct", 0.0) or 0.0)
        if hard_stop <= 0:
            return

        remaining = ps.qty
        level_qtys: List[float] = []
        for lvl in levels:
            try:
                frac = float(lvl.get("fraction", 0.0) or 0.0)
            except (TypeError, ValueError):
                frac = 0.0
            qty = remaining * frac
            remaining -= qty
            level_qtys.append(qty)

        offset_ticks = self._tp_maker_offset_ticks(cfg)
        retry_times, retry_tick_step = self._tp_post_only_retry_settings(cfg)
        for idx, lvl in enumerate(levels):
            if idx < ps.tp_state:
                continue
            try:
                r_mult = float(lvl.get("r_multiple", 0.0) or 0.0)
            except (TypeError, ValueError):
                r_mult = 0.0
            if r_mult <= 0:
                continue
            target_px = ps.entry_price * (1.0 + r_mult * hard_stop) if ps.side == "BUY" else (
                ps.entry_price * (1.0 - r_mult * hard_stop)
            )
            limit_px_base = self._tp_limit_price(
                ps.symbol,
                "SELL" if ps.side == "BUY" else "BUY",
                target_px,
                spec.tick_size,
                offset_ticks,
            )
            qty_raw = level_qtys[idx] if idx < len(level_qtys) else 0.0
            qty = round_step(qty_raw, spec.step_size)
            key = (ps.symbol, ps.side, idx)
            existing = self._tp_orders.get(key)
            if qty <= 0 or qty < spec.min_qty or limit_px_base <= 0:
                if existing and self._tp_order_active(existing):
                    await self._cancel_tp_order(ps.symbol, existing.order_id, reason=reason or "tp_qty_invalid")
                if existing:
                    self._tp_order_ids.pop(existing.order_id, None)
                    self._tp_orders.pop(key, None)
                continue
            if existing and self._tp_order_active(existing):
                same_price = abs(existing.price - limit_px_base) <= (spec.tick_size * 0.5)
                same_qty = abs(existing.remaining_qty() - qty) <= (spec.step_size * 0.5)
                if same_price and same_qty:
                    continue
                canceled = await self._cancel_tp_order(ps.symbol, existing.order_id, reason=reason or "replace_tp")
                if not canceled:
                    continue
                self._tp_order_ids.pop(existing.order_id, None)
                self._tp_orders.pop(key, None)

            side = "SELL" if ps.side == "BUY" else "BUY"
            order_id = 0
            for attempt in range(retry_times + 1):
                attempt_offset = offset_ticks + attempt * retry_tick_step
                limit_px = self._tp_limit_price(
                    ps.symbol,
                    side,
                    target_px,
                    spec.tick_size,
                    attempt_offset,
                )
                order_params = dict(
                    symbol=ps.symbol,
                    side=side,
                    type="LIMIT",
                    timeInForce="GTX",
                    quantity=format_by_step(qty, spec.step_size, rounding=ROUND_DOWN),
                    price=format_by_step(limit_px, spec.tick_size, rounding=ROUND_UP if side == "SELL" else ROUND_DOWN),
                    newClientOrderId=f"TP{idx+1}_{ps.symbol}_{utc_ms()}",
                    newOrderRespType="ACK",
                )
                if self._position_mode != "HEDGE":
                    order_params["reduceOnly"] = "true"
                pos_side = self._position_side_for_position(ps.side)
                if pos_side:
                    order_params["positionSide"] = pos_side
                r = await self.client.new_order(**order_params)
                if r.status == 200:
                    order_id = int(r.data.get("orderId", 0) or 0)
                    if order_id > 0:
                        break
                if self._is_post_only_reject(r.data) and attempt < retry_times:
                    self.logger.warning(
                        "event=tp_order_retry symbol=%s side=%s price=%.8f qty=%.6f attempt=%d/%d err=%s",
                        ps.symbol,
                        ps.side,
                        limit_px,
                        qty,
                        attempt + 1,
                        retry_times + 1,
                        self._format_api_error(r.data),
                    )
                    await asyncio.sleep(0)
                    continue
                self.logger.warning(
                    "event=tp_order_rejected symbol=%s side=%s price=%.8f qty=%.6f err=%s",
                    ps.symbol,
                    ps.side,
                    limit_px,
                    qty,
                    self._format_api_error(r.data),
                )
                await self._log_error(
                    "take_profit",
                    event="tp_order_rejected",
                    symbol=ps.symbol,
                    side=ps.side,
                    reason=reason or "tp_order_rejected",
                    details={
                        "status": r.status,
                        "response": r.data,
                        "order_params": order_params,
                    },
                )
                break
            if order_id <= 0:
                continue
            if order_id <= 0:
                self.logger.warning(
                    "event=tp_order_rejected symbol=%s side=%s price=%.8f qty=%.6f err=missing_order_id",
                    ps.symbol,
                    ps.side,
                    limit_px,
                    qty,
                )
                continue
            state = TpOrderState(
                order_id=order_id,
                level=idx,
                price=limit_px,
                orig_qty=qty,
                executed_qty=0.0,
                status="NEW",
                created_ts=utc_ms(),
            )
            self._tp_orders[key] = state
            self._tp_order_ids[order_id] = key

    async def _maybe_force_stop(self, ps: PositionState, px: float) -> bool:
        if px <= 0 or ps.stop_price <= 0 or self._force_stop_pct <= 0:
            return False
        state = self._stop_orders.get(self._pos_key(ps.symbol, ps.side))
        if not self._stop_order_active(state):
            return False
        stop_price = state.stop_price if state else ps.stop_price
        if ps.side == "BUY":
            if px <= stop_price * (1.0 - self._force_stop_pct):
                await self._close_position_market(ps, reason="STOP_FORCE", exit_price=px)
                return True
        else:
            if px >= stop_price * (1.0 + self._force_stop_pct):
                await self._close_position_market(ps, reason="STOP_FORCE", exit_price=px)
                return True
        return False

    async def _apply_stop_fill(self, ps: PositionState, qty_exec: float,
                               fill_price: float, order_id: int, status: str) -> None:
        if qty_exec <= 0:
            return
        qty_before = ps.qty
        qty_exec = min(qty_exec, qty_before)
        px = float(fill_price or 0.0) or ps.stop_price
        pnl = None
        pnl_pct = None
        if px > 0:
            pnl = self._calc_pnl(ps, qty_exec, px)
            pnl_pct = self._calc_pnl_pct(ps, px)
        if pnl is not None:
            self._apply_realized_pnl(pnl)
        ps.qty -= qty_exec
        reason = "STOP_MAKER"
        event = "EXIT" if ps.qty <= 0 else "REDUCE"
        await self._log_trade_event(
            symbol=ps.symbol,
            side=ps.side,
            event=event,
            qty=qty_exec,
            price=px,
            status=status or "OK",
            reason=reason,
            order_id=order_id,
            position_id=ps.open_ts,
            entry_price=ps.entry_price,
            stop_price=ps.stop_price,
            peak_price=ps.peak_price,
            tp_state=ps.tp_state,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        if ps.qty <= 0:
            await self._remove_position(ps)
            return
        await self._upsert_live_position(ps, px)
        await self._ensure_tp_orders(ps, reason="stop_fill")

    async def _handle_stop_order_update(self, upd: OrderUpdate) -> None:
        order_id_int = int(upd.order_id)
        key = self._stop_order_ids.get(order_id_int)
        if not key and upd.client_order_id:
            key = self._stop_order_client_ids.get(upd.client_order_id)
            if key:
                self._stop_order_ids[order_id_int] = key
        if not key:
            return
        state = self._stop_orders.get(key)
        if not state:
            self._stop_order_ids.pop(order_id_int, None)
            return
        prev_exec = state.executed_qty
        if upd.orig_qty > 0:
            state.orig_qty = float(upd.orig_qty)
        if upd.price > 0:
            state.stop_price = float(upd.price)
        state.status = upd.status
        new_exec = max(prev_exec, float(upd.executed_qty))
        state.executed_qty = new_exec
        if new_exec > prev_exec:
            ps = self._get_position(*key)
            if ps:
                fill_px = upd.avg_price if upd.avg_price > 0 else upd.price
                await self._apply_stop_fill(ps, new_exec - prev_exec, fill_px, order_id_int, upd.status)
        if upd.status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") or state.remaining_qty() <= 0:
            self._stop_order_ids.pop(order_id_int, None)
            if state.client_order_id:
                self._stop_order_client_ids.pop(state.client_order_id, None)
            self._stop_orders.pop(key, None)

    async def _apply_tp_fill(self, ps: PositionState, qty_exec: float,
                             fill_price: float, order_id: int, status: str, level: int) -> None:
        if qty_exec <= 0:
            return
        qty_before = ps.qty
        qty_exec = min(qty_exec, qty_before)
        px = float(fill_price or 0.0)
        pnl = None
        pnl_pct = None
        if px > 0:
            pnl = self._calc_pnl(ps, qty_exec, px)
            pnl_pct = self._calc_pnl_pct(ps, px)
        if pnl is not None:
            self._apply_realized_pnl(pnl)
        ps.qty -= qty_exec
        reason = f"TP{level + 1}"
        event = "EXIT" if ps.qty <= 0 else "REDUCE"
        await self._log_trade_event(
            symbol=ps.symbol,
            side=ps.side,
            event=event,
            qty=qty_exec,
            price=px,
            status=status or "OK",
            reason=reason,
            order_id=order_id,
            position_id=ps.open_ts,
            entry_price=ps.entry_price,
            stop_price=ps.stop_price,
            peak_price=ps.peak_price,
            tp_state=ps.tp_state,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        if ps.qty <= 0:
            await self._remove_position(ps)
            return
        await self._upsert_live_position(ps, px)
        await self._ensure_stop_order(ps, px, reason="tp_fill")

    async def _handle_tp_order_update(self, upd: OrderUpdate) -> None:
        order_id_int = int(upd.order_id)
        key = self._tp_order_ids.get(order_id_int)
        if not key:
            return
        state = self._tp_orders.get(key)
        if not state:
            self._tp_order_ids.pop(order_id_int, None)
            return
        prev_exec = state.executed_qty
        if upd.orig_qty > 0:
            state.orig_qty = float(upd.orig_qty)
        if upd.price > 0:
            state.price = float(upd.price)
        state.status = upd.status
        new_exec = max(prev_exec, float(upd.executed_qty))
        state.executed_qty = new_exec
        if new_exec > prev_exec:
            ps = self._get_position(key[0], key[1])
            if ps:
                fill_px = upd.avg_price if upd.avg_price > 0 else upd.price
                await self._apply_tp_fill(ps, new_exec - prev_exec, fill_px, order_id_int, upd.status, state.level)
                if state.executed_qty >= state.orig_qty and ps.tp_state <= state.level:
                    ps.tp_state = state.level + 1
        if upd.status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") or state.remaining_qty() <= 0:
            self._tp_order_ids.pop(order_id_int, None)
            self._tp_orders.pop(key, None)

    async def _risk_loop(self):
        while True:
            await asyncio.sleep(self._risk_loop_interval_s)
            try:
                total_unrealized = 0.0
                last_px: Dict[str, float] = {}
                now_ms = utc_ms()
                for _, ps in list(self.positions.items()):
                    sym = ps.symbol
                    m = self.market_ws.marks.get(sym)
                    b = self.market_ws.books.get(sym)
                    px = 0.0
                    if self.config["risk"]["use_mark_price_for_risk"]:
                        if m and m.mark > 0:
                            px = m.mark
                        elif b and b.bid > 0 and b.ask > 0:
                            px = (b.bid + b.ask) / 2.0
                    else:
                        if b and b.bid > 0 and b.ask > 0:
                            px = (b.bid + b.ask) / 2.0
                        elif m and m.mark > 0:
                            px = m.mark
                    if px <= 0:
                        cached = self._last_px_cache.get(sym)
                        if cached and (now_ms - cached[1]) <= self._max_stale_price_ms:
                            px = cached[0]
                        else:
                            continue
                    self._last_px_cache[sym] = (px, now_ms)
                    last_px[sym] = px
                    if ps.side == "BUY":
                        total_unrealized += (px - ps.entry_price) * ps.qty
                    else:
                        total_unrealized += (ps.entry_price - px) * ps.qty

                    if ps.side == "BUY":
                        ps.peak_price = max(ps.peak_price, px)
                    else:
                        ps.peak_price = min(ps.peak_price, px)

                    await self._apply_takeprofit(ps, px)
                    if self._pos_key(sym, ps.side) not in self.positions:
                        continue
                    await self._ensure_stop_order(ps, px, reason="risk_loop")
                    mark_px = m.mark if (m and m.mark > 0) else px
                    if mark_px > 0:
                        stop_triggered = (
                            (ps.side == "BUY" and mark_px <= ps.stop_price) or
                            (ps.side == "SELL" and mark_px >= ps.stop_price)
                        )
                        if stop_triggered:
                            max_bps = float(self.config.get("risk", {}).get(
                                "stop_slippage_bps_max",
                                self._max_taker_slippage_bps,
                            ) or 0.0)
                            slippage_bps = self._stop_slippage_bps(mark_px, b, ps.side)
                            if max_bps > 0 and slippage_bps is not None and slippage_bps > max_bps:
                                self.logger.warning(
                                    "event=stop_slippage_exceeds symbol=%s side=%s mark=%.8f stop=%.8f bps=%.2f max=%.2f",
                                    ps.symbol,
                                    ps.side,
                                    mark_px,
                                    ps.stop_price,
                                    slippage_bps,
                                    max_bps,
                                )
                                await self._log_error(
                                    "stop_loss",
                                    event="stop_slippage_exceeds",
                                    symbol=ps.symbol,
                                    side=ps.side,
                                    reason="local_stop_slippage",
                                    details={
                                        "mark_price": mark_px,
                                        "stop_price": ps.stop_price,
                                        "slippage_bps": slippage_bps,
                                        "max_slippage_bps": max_bps,
                                    },
                                )
                                continue
                            await self._close_position_market(ps, reason="STOP_LOCAL", exit_price=mark_px)
                            continue
                    if await self._maybe_force_stop(ps, px):
                        continue
                    prof = self._profit_pct(ps, px)
                    if prof is not None and prof >= 0.03:
                        await self._maybe_roll_add_on(sym)
                    max_hold_ms = self._dynamic_time_exit_ms(ps, px)
                    if max_hold_ms > 0 and (now_ms - ps.open_ts) >= max_hold_ms:
                        exit_px = self._time_exit_ref_price(sym, px)
                        await self._close_position_market(ps, reason="TIME_EXIT", exit_price=exit_px)
                        continue
                    await self._upsert_live_position(ps, px)

                self._unrealized_pnl = total_unrealized
                self._roll_daily_pnl()
                await self._enforce_min_equity(last_px)
                await self._enforce_daily_loss(last_px)
                await self._enforce_ws_health(last_px)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._log_error(
                    "position_manage",
                    event="risk_loop_error",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )
                continue

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(self._heartbeat_interval_s)
            try:
                pos_cnt = len(self.positions)
                trig_cnt = self.triggers.qsize()
                if self._use_internal_equity:
                    eq = self._initial_equity + self._realized_pnl + self._unrealized_pnl
                    self.logger.info(
                        "event=heartbeat positions=%d triggers=%d equity=%.4f realized=%.4f unrealized=%.4f halt=%s",
                        pos_cnt,
                        trig_cnt,
                        eq,
                        self._realized_pnl,
                        self._unrealized_pnl,
                        self._halt_trading,
                    )
                else:
                    eq = await self._get_equity()
                    if eq > 0:
                        self.logger.info(
                            "event=heartbeat positions=%d triggers=%d equity=%.4f halt=%s",
                            pos_cnt,
                            trig_cnt,
                            eq,
                            self._halt_trading,
                        )
                    else:
                        self.logger.info(
                            "event=heartbeat positions=%d triggers=%d halt=%s",
                            pos_cnt,
                            trig_cnt,
                            self._halt_trading,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning("event=heartbeat_error err=%s", e)

    async def _apply_takeprofit(self, ps: PositionState, px: float):
        cfg = self._cfg_for_side(ps.side)
        mode = str(cfg.get("takeprofit", {}).get("mode", "trailing_ladder")).lower()
        if mode == "staged_partial":
            await self._tp_staged_partial(ps, px, cfg)
            return
        if mode in ("hybrid", "staged_partial_trailing", "staged_partial+trailing"):
            await self._tp_staged_partial(ps, px, cfg)
            await self._tp_trailing_ladder(ps, px, cfg)
            return
        if mode == "trailing_ladder":
            await self._tp_trailing_ladder(ps, px, cfg)
            return
        await self._tp_trailing_ladder(ps, px, cfg)

    async def _tp_staged_partial(self, ps: PositionState, px: float, cfg: dict):
        hard_stop = float(cfg["risk"]["hard_stop_pct"])
        levels = cfg["takeprofit"]["staged_partial"]["levels"]
        move_be_r = float(cfg["takeprofit"]["staged_partial"]["move_stop_to_breakeven_after_r"])
        prof = (px / ps.entry_price - 1.0) if ps.side == "BUY" else (ps.entry_price / px - 1.0)
        R = hard_stop

        if prof >= move_be_r * R:
            ps.stop_price = max(ps.stop_price, ps.entry_price) if ps.side == "BUY" else min(ps.stop_price, ps.entry_price)

        if self._tp_preplace_enabled(cfg):
            return

        while ps.tp_state < len(levels):
            lvl = levels[ps.tp_state]
            if prof >= float(lvl["r_multiple"]) * R:
                frac = float(lvl["fraction"])
                qty_exit = ps.qty * frac
                await self._reduce_position_market(ps, qty_exit, reason=f"TP{ps.tp_state+1}", exit_price=px)
                ps.tp_state += 1
                if ps.qty <= 0:
                    await self._remove_position(ps)
                    return
            else:
                break

    async def _tp_trailing_ladder(self, ps: PositionState, px: float, cfg: dict):
        tp_cfg = cfg["takeprofit"]["trailing_ladder"]
        p1, p2, lock2, p3, trail = float(tp_cfg["p1"]), float(tp_cfg["p2"]), float(tp_cfg["lock2"]), float(tp_cfg["p3"]), float(tp_cfg["trail_pct"])
        prof = (px / ps.entry_price - 1.0) if ps.side == "BUY" else (ps.entry_price / px - 1.0)

        if prof >= p1:
            ps.stop_price = max(ps.stop_price, ps.entry_price) if ps.side == "BUY" else min(ps.stop_price, ps.entry_price)
        if prof >= p2:
            ps.stop_price = max(ps.stop_price, ps.entry_price * (1.0 + lock2)) if ps.side == "BUY" else min(ps.stop_price, ps.entry_price * (1.0 - lock2))
        if prof >= p3:
            new_stop = ps.peak_price * (1.0 - trail) if ps.side == "BUY" else ps.peak_price * (1.0 + trail)
            ps.stop_price = max(ps.stop_price, new_stop) if ps.side == "BUY" else min(ps.stop_price, new_stop)

    async def _reduce_position_market(self, ps: PositionState, qty_exit: float, reason: str,
                                      exit_price: Optional[float] = None):
        if qty_exit <= 0:
            return
        is_tp = str(reason or "").upper().startswith("TP")
        cfg = self._cfg_for_side(ps.side)
        spec = self.universe[ps.symbol]
        qty_exec = round_step(qty_exit, spec.step_size)
        if qty_exec <= 0:
            self._record_reject("reduce_qty_rounds_to_zero", ps.symbol, ps.side)
            return
        if ps.qty - qty_exec < spec.min_qty:
            await self._close_position_market(ps, reason=f"{reason}_FINAL", exit_price=exit_price)
            return
        if qty_exec < spec.min_qty:
            self._record_reject(
                "reduce_qty_below_min_qty",
                ps.symbol,
                ps.side,
                f"qty={qty_exec:.6f} min_qty={spec.min_qty:.6f}",
            )
            return
        px = float(exit_price or 0.0)
        if px <= 0:
            cached = self._last_px_cache.get(ps.symbol)
            if cached:
                px = cached[0]
        pnl = None
        pnl_pct = None
        if px > 0:
            pnl = self._calc_pnl(ps, qty_exec, px)
            pnl_pct = self._calc_pnl_pct(ps, px)
        if self.dry_run:
            if pnl is not None:
                self._apply_realized_pnl(pnl)
            ps.qty -= qty_exec
            self.logger.info(
                "event=reduce_simulated symbol=%s side=%s qty=%.6f reason=%s remaining=%.6f",
                ps.symbol,
                ps.side,
                qty_exec,
                reason,
                ps.qty,
            )
            fee = self._estimate_fee(qty_exec, px, True)
            await self._log_trade_event(
                symbol=ps.symbol,
                side=ps.side,
                event="REDUCE",
                qty=qty_exec,
                price=px,
                status="SIMULATED",
                reason=reason,
                position_id=ps.open_ts,
                entry_price=ps.entry_price,
                stop_price=ps.stop_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fee=fee,
            )
            if ps.qty > 0:
                await self._ensure_stop_order(ps, px, reason="reduce_simulated")
            return

        side = "SELL" if ps.side == "BUY" else "BUY"
        stage = self._error_stage_for_exit_reason(reason)
        pos_side = self._position_side_for_position(ps.side)
        order_params = dict(
            symbol=ps.symbol,
            side=side,
            type="MARKET",
            quantity=format_by_step(qty_exec, spec.step_size, rounding=ROUND_DOWN),
            newOrderRespType="ACK",
        )
        if is_tp:
            b = self.market_ws.books.get(ps.symbol)
            limit_px = float(px or 0.0)
            if b and b.bid > 0 and b.ask > 0:
                offset_ticks = float(cfg.get("takeprofit", {}).get("maker_offset_ticks", 1) or 0.0)
                tick = spec.tick_size
                if tick > 0 and offset_ticks > 0:
                    if side == "SELL":
                        limit_px = max(limit_px, b.ask + tick * offset_ticks)
                    else:
                        bid_px = b.bid - tick * offset_ticks
                        if bid_px > 0:
                            limit_px = min(limit_px, bid_px)
                        else:
                            limit_px = min(limit_px, b.bid)
                else:
                    if side == "SELL":
                        limit_px = max(limit_px, b.ask)
                    else:
                        limit_px = min(limit_px, b.bid)
            rounding = ROUND_UP if side == "SELL" else ROUND_DOWN
            if limit_px > 0:
                order_params["type"] = "LIMIT"
                order_params["timeInForce"] = "GTX"
                order_params["price"] = format_by_step(limit_px, spec.tick_size, rounding=rounding)
        if self._position_mode != "HEDGE":
            order_params["reduceOnly"] = "true"
        if pos_side:
            order_params["positionSide"] = pos_side
        r = await self.client.new_order(**order_params)
        if r.status != 200:
            await self._log_error(
                stage,
                event="reduce_order_rejected",
                symbol=ps.symbol,
                side=ps.side,
                reason=reason,
                details={
                    "status": r.status,
                    "response": r.data,
                    "order_params": order_params,
                },
            )
            if self._is_reduce_only_error(r.data) and "reduceOnly" in order_params:
                exch_qty = await self._position_size_on_exchange(ps.symbol, ps.side)
                if exch_qty <= 0:
                    self.logger.warning(
                        "event=reduce_failed_no_position symbol=%s side=%s reason=reduceonly_reject",
                        ps.symbol,
                        ps.side,
                    )
                    await self._log_trade_event(
                        symbol=ps.symbol,
                        side=ps.side,
                        event="REDUCE",
                        qty=qty_exec,
                        price=px,
                        status="SKIP",
                        reason="reduceonly_no_position",
                        position_id=ps.open_ts,
                        entry_price=ps.entry_price,
                        stop_price=ps.stop_price,
                        peak_price=ps.peak_price,
                        tp_state=ps.tp_state,
                    )
                    await self._remove_position(ps)
                    return
                qty_retry = round_step(min(qty_exec, exch_qty), spec.step_size)
                if qty_retry <= 0:
                    self.logger.warning(
                        "event=reduce_retry_skipped symbol=%s side=%s reason=qty_rounds_to_zero",
                        ps.symbol,
                        ps.side,
                    )
                    await self._log_trade_event(
                        symbol=ps.symbol,
                        side=ps.side,
                        event="REDUCE",
                        qty=qty_retry,
                        price=px,
                        status="SKIP",
                        reason="reduceonly_qty_zero",
                        position_id=ps.open_ts,
                        entry_price=ps.entry_price,
                        stop_price=ps.stop_price,
                        peak_price=ps.peak_price,
                        tp_state=ps.tp_state,
                    )
                    return
                retry_params = dict(order_params)
                retry_params.pop("reduceOnly", None)
                retry_params["quantity"] = format_by_step(qty_retry, spec.step_size, rounding=ROUND_DOWN)
                r2 = await self.client.new_order(**retry_params)
                if r2.status == 200:
                    order_id = int(r2.data.get("orderId", 0) or 0)
                    if px > 0:
                        pnl = self._calc_pnl(ps, qty_retry, px)
                        self._apply_realized_pnl(pnl)
                    ps.qty -= qty_retry
                    self.logger.info(
                        "event=reduce_ok symbol=%s side=%s qty=%.6f reason=%s remaining=%.6f reduce_only=false",
                        ps.symbol,
                        ps.side,
                        qty_retry,
                        reason,
                        ps.qty,
                    )
                    fee = self._estimate_fee(qty_retry, px, True)
                    await self._log_trade_event(
                        symbol=ps.symbol,
                        side=ps.side,
                        event="REDUCE",
                        qty=qty_retry,
                        price=px,
                        status="OK",
                        reason=f"{reason}_NO_REDUCEONLY",
                        order_id=order_id,
                        position_id=ps.open_ts,
                        entry_price=ps.entry_price,
                        stop_price=ps.stop_price,
                        peak_price=ps.peak_price,
                        tp_state=ps.tp_state,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        fee=fee,
                    )
                    if ps.qty > 0:
                        await self._ensure_stop_order(ps, px, reason="reduce_retry")
                    if ps.qty <= 0:
                        await self._remove_position(ps)
                    return
                await self._log_error(
                    stage,
                    event="reduce_order_retry_failed",
                    symbol=ps.symbol,
                    side=ps.side,
                    reason=reason,
                    details={
                        "status": r2.status,
                        "response": r2.data,
                        "order_params": retry_params,
                    },
                )
                self.logger.warning(
                    "event=reduce_failed symbol=%s side=%s qty=%.6f reason=%s err=%s reduce_only=false",
                    ps.symbol,
                    ps.side,
                    qty_retry,
                    reason,
                    self._format_api_error(r2.data),
                )
                await self._log_trade_event(
                    symbol=ps.symbol,
                    side=ps.side,
                    event="REDUCE",
                    qty=qty_retry,
                    price=px,
                    status="FAILED",
                    reason=f"{reason} {r.data} retry_no_reduceonly {r2.data}",
                    position_id=ps.open_ts,
                    entry_price=ps.entry_price,
                    stop_price=ps.stop_price,
                    peak_price=ps.peak_price,
                    tp_state=ps.tp_state,
                )
                await self._sync_position_after_failure(ps, "REDUCE", reason, px)
                return
            self.logger.warning(
                "event=reduce_failed symbol=%s side=%s qty=%.6f reason=%s err=%s",
                ps.symbol,
                ps.side,
                qty_exec,
                reason,
                self._format_api_error(r.data),
            )
            await self._log_trade_event(
                symbol=ps.symbol,
                side=ps.side,
                event="REDUCE",
                qty=qty_exec,
                price=px,
                status="FAILED",
                reason=f"{reason} {r.data}",
                position_id=ps.open_ts,
                entry_price=ps.entry_price,
                stop_price=ps.stop_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
            )
            await self._sync_position_after_failure(ps, "REDUCE", reason, px)
            return
        order_id = int(r.data.get("orderId", 0) or 0)
        if pnl is not None:
            self._apply_realized_pnl(pnl)
        ps.qty -= qty_exec
        self.logger.info(
            "event=reduce_ok symbol=%s side=%s qty=%.6f reason=%s remaining=%.6f",
            ps.symbol,
            ps.side,
            qty_exec,
            reason,
            ps.qty,
        )
        fee = self._estimate_fee(qty_exec, px, True)
        await self._log_trade_event(
            symbol=ps.symbol,
            side=ps.side,
            event="REDUCE",
            qty=qty_exec,
            price=px,
            status="OK",
            reason=reason,
            order_id=order_id,
            position_id=ps.open_ts,
            entry_price=ps.entry_price,
            stop_price=ps.stop_price,
            peak_price=ps.peak_price,
            tp_state=ps.tp_state,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee=fee,
        )
        if ps.qty > 0:
            await self._ensure_stop_order(ps, px, reason="reduce_ok")
            await self._ensure_tp_orders(ps, reason="reduce_ok")

    async def _close_position_market(self, ps: PositionState, reason: str,
                                     exit_price: Optional[float] = None):
        px = float(exit_price or 0.0)
        if px <= 0:
            cached = self._last_px_cache.get(ps.symbol)
            if cached:
                px = cached[0]
        pnl = None
        pnl_pct = None
        if px > 0:
            pnl = self._calc_pnl(ps, ps.qty, px)
            pnl_pct = self._calc_pnl_pct(ps, px)
        if self.dry_run:
            if pnl is not None:
                self._apply_realized_pnl(pnl)
            self.logger.info(
                "event=close_simulated symbol=%s side=%s qty=%.6f reason=%s",
                ps.symbol,
                ps.side,
                ps.qty,
                reason,
            )
            fee = self._estimate_fee(ps.qty, px, True)
            await self._log_trade_event(
                symbol=ps.symbol,
                side=ps.side,
                event="EXIT",
                qty=ps.qty,
                price=px,
                status="SIMULATED",
                reason=reason,
                position_id=ps.open_ts,
                entry_price=ps.entry_price,
                stop_price=ps.stop_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fee=fee,
            )
            await self._remove_position(ps)
            return
        side = "SELL" if ps.side == "BUY" else "BUY"
        qty = ps.qty
        stage = self._error_stage_for_exit_reason(reason)
        pos_side = self._position_side_for_position(ps.side)
        order_params = dict(
            symbol=ps.symbol,
            side=side,
            type="MARKET",
            quantity=format_by_step(qty, self.universe[ps.symbol].step_size, rounding=ROUND_DOWN),
            newOrderRespType="ACK",
        )
        if self._position_mode != "HEDGE":
            order_params["reduceOnly"] = "true"
        if pos_side:
            order_params["positionSide"] = pos_side
        r = await self.client.new_order(**order_params)
        if r.status != 200:
            await self._log_error(
                stage,
                event="close_order_rejected",
                symbol=ps.symbol,
                side=ps.side,
                reason=reason,
                details={
                    "status": r.status,
                    "response": r.data,
                    "order_params": order_params,
                },
            )
            if self._is_reduce_only_error(r.data) and "reduceOnly" in order_params:
                exch_qty = await self._position_size_on_exchange(ps.symbol, ps.side)
                if exch_qty <= 0:
                    self.logger.warning(
                        "event=close_failed_no_position symbol=%s side=%s reason=reduceonly_reject",
                        ps.symbol,
                        ps.side,
                    )
                    await self._log_trade_event(
                        symbol=ps.symbol,
                        side=ps.side,
                        event="EXIT",
                        qty=qty,
                        price=px,
                        status="SKIP",
                        reason="reduceonly_no_position",
                        position_id=ps.open_ts,
                        entry_price=ps.entry_price,
                        stop_price=ps.stop_price,
                        peak_price=ps.peak_price,
                        tp_state=ps.tp_state,
                    )
                    await self._remove_position(ps)
                    return
                spec = self.universe[ps.symbol]
                qty_retry = round_step(min(qty, exch_qty), spec.step_size)
                if qty_retry <= 0:
                    self.logger.warning(
                        "event=close_retry_skipped symbol=%s side=%s reason=qty_rounds_to_zero",
                        ps.symbol,
                        ps.side,
                    )
                    await self._log_trade_event(
                        symbol=ps.symbol,
                        side=ps.side,
                        event="EXIT",
                        qty=qty_retry,
                        price=px,
                        status="SKIP",
                        reason="reduceonly_qty_zero",
                        position_id=ps.open_ts,
                        entry_price=ps.entry_price,
                        stop_price=ps.stop_price,
                        peak_price=ps.peak_price,
                        tp_state=ps.tp_state,
                    )
                    return
                retry_params = dict(order_params)
                retry_params.pop("reduceOnly", None)
                retry_params["quantity"] = format_by_step(qty_retry, spec.step_size, rounding=ROUND_DOWN)
                r2 = await self.client.new_order(**retry_params)
                if r2.status == 200:
                    order_id = int(r2.data.get("orderId", 0) or 0)
                    if px > 0:
                        pnl = self._calc_pnl(ps, qty_retry, px)
                        self._apply_realized_pnl(pnl)
                    self.logger.info(
                        "event=position_closed symbol=%s side=%s qty=%.6f reason=%s reduce_only=false",
                        ps.symbol,
                        ps.side,
                        qty_retry,
                        reason,
                    )
                    fee = self._estimate_fee(qty_retry, px, True)
                    await self._log_trade_event(
                        symbol=ps.symbol,
                        side=ps.side,
                        event="EXIT",
                        qty=qty_retry,
                        price=px,
                        status="OK",
                        reason=f"{reason}_NO_REDUCEONLY",
                        order_id=order_id,
                        position_id=ps.open_ts,
                        entry_price=ps.entry_price,
                        stop_price=ps.stop_price,
                        peak_price=ps.peak_price,
                        tp_state=ps.tp_state,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        fee=fee,
                    )
                    await self._remove_position(ps)
                    return
                await self._log_error(
                    stage,
                    event="close_order_retry_failed",
                    symbol=ps.symbol,
                    side=ps.side,
                    reason=reason,
                    details={
                        "status": r2.status,
                        "response": r2.data,
                        "order_params": retry_params,
                    },
                )
                self.logger.warning(
                    "event=close_failed symbol=%s side=%s qty=%.6f reason=%s err=%s reduce_only=false",
                    ps.symbol,
                    ps.side,
                    qty_retry,
                    reason,
                    self._format_api_error(r2.data),
                )
                await self._log_trade_event(
                    symbol=ps.symbol,
                    side=ps.side,
                    event="EXIT",
                    qty=qty_retry,
                    price=px,
                    status="FAILED",
                    reason=f"{reason} {r.data} retry_no_reduceonly {r2.data}",
                    position_id=ps.open_ts,
                    entry_price=ps.entry_price,
                    stop_price=ps.stop_price,
                    peak_price=ps.peak_price,
                    tp_state=ps.tp_state,
                )
                await self._sync_position_after_failure(ps, "EXIT", reason, px)
                return
            self.logger.warning(
                "event=close_failed symbol=%s side=%s qty=%.6f reason=%s err=%s",
                ps.symbol,
                ps.side,
                qty,
                reason,
                self._format_api_error(r.data),
            )
            await self._log_trade_event(
                symbol=ps.symbol,
                side=ps.side,
                event="EXIT",
                qty=qty,
                price=px,
                status="FAILED",
                reason=f"{reason} {r.data}",
                position_id=ps.open_ts,
                entry_price=ps.entry_price,
                stop_price=ps.stop_price,
                peak_price=ps.peak_price,
                tp_state=ps.tp_state,
            )
            await self._sync_position_after_failure(ps, "EXIT", reason, px)
            return
        order_id = int(r.data.get("orderId", 0) or 0)
        if pnl is not None:
            self._apply_realized_pnl(pnl)
        self.logger.info(
            "event=position_closed symbol=%s side=%s qty=%.6f reason=%s",
            ps.symbol,
            ps.side,
            qty,
            reason,
        )
        fee = self._estimate_fee(qty, px, True)
        await self._log_trade_event(
            symbol=ps.symbol,
            side=ps.side,
            event="EXIT",
            qty=qty,
            price=px,
            status="OK",
            reason=reason,
            order_id=order_id,
            position_id=ps.open_ts,
            entry_price=ps.entry_price,
            stop_price=ps.stop_price,
            peak_price=ps.peak_price,
            tp_state=ps.tp_state,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee=fee,
        )
        await self._remove_position(ps)

    def _calc_pnl(self, ps: PositionState, qty_exit: float, exit_price: float) -> float:
        if ps.side == "BUY":
            return (exit_price - ps.entry_price) * qty_exit
        return (ps.entry_price - exit_price) * qty_exit

    def _trade_type_from_reason(self, event: str, reason: str) -> str:
        r = (reason or "").upper()
        e = (event or "").upper()
        if e == "ADD":
            return "ADD"
        if e == "ENTRY":
            return "ENTRY"
        if e == "REDUCE":
            if r.startswith("TP"):
                return "TP"
            return "REDUCE"
        if e == "EXIT":
            if r.startswith("STOP"):
                return "STOP"
            if r.startswith("TIME_EXIT"):
                return "TIME_EXIT"
            if r.startswith("MIN_EQUITY") or r.startswith("DAILY_LOSS") or r.startswith("WS_DISCONNECT"):
                return "RISK"
            return "EXIT"
        return e or "UNKNOWN"

    def _error_stage_for_exit_reason(self, reason: str) -> str:
        r = (reason or "").upper()
        if r.startswith("TP"):
            return "take_profit"
        if r.startswith("STOP"):
            return "stop_loss"
        return "position_manage"

    def _calc_pnl_pct(self, ps: PositionState, exit_price: float) -> Optional[float]:
        if ps.entry_price <= 0 or exit_price <= 0:
            return None
        if ps.side == "BUY":
            return exit_price / ps.entry_price - 1.0
        return ps.entry_price / exit_price - 1.0

    def _apply_realized_pnl(self, pnl: float) -> None:
        if pnl == 0:
            return
        self._realized_pnl += pnl
        self._update_daily_pnl(pnl)

    def _roll_daily_pnl(self) -> None:
        now = datetime.now(timezone.utc).date()
        if self._daily_date != now:
            self._daily_date = now
            self._daily_pnl = 0.0

    def _update_daily_pnl(self, pnl: float) -> None:
        self._roll_daily_pnl()
        self._daily_pnl += pnl

    async def _enforce_min_equity(self, last_px: Dict[str, float]) -> None:
        if not self._use_internal_equity:
            return
        total_equity = self._initial_equity + self._realized_pnl + self._unrealized_pnl
        if total_equity > self._min_equity or self._halt_trading:
            return
        self._halt_trading = True
        self.logger.error(
            "event=equity_min_reached equity=%.4f min_equity=%.4f action=close_all_halt",
            total_equity,
            self._min_equity,
        )
        for _, ps in list(self.positions.items()):
            px = last_px.get(ps.symbol)
            await self._close_position_market(ps, reason="MIN_EQUITY", exit_price=px)

    async def _enforce_daily_loss(self, last_px: Dict[str, float]) -> None:
        max_daily = float(self.config["risk"].get("max_daily_loss_usdt", 0.0) or 0.0)
        if max_daily <= 0 or self._halt_trading:
            return
        if self._daily_pnl > -max_daily:
            return
        self._halt_trading = True
        self.logger.error(
            "event=max_daily_loss_reached daily_pnl=%.4f max_daily_loss=%.4f action=close_all_halt",
            self._daily_pnl,
            max_daily,
        )
        for _, ps in list(self.positions.items()):
            px = last_px.get(ps.symbol)
            await self._close_position_market(ps, reason="DAILY_LOSS", exit_price=px)

    async def _enforce_ws_health(self, last_px: Dict[str, float]) -> None:
        risk_cfg = self.config.get("risk", {})
        if not bool(risk_cfg.get("kill_switch_on_ws_disconnect", False)) or self._halt_trading:
            return
        stale_s = float(risk_cfg.get("ws_stale_seconds", 30.0))
        healthy = self.market_ws.is_healthy() and self.user_ws.is_connected()
        now = time.time()
        if healthy:
            self._ws_unhealthy_since = 0.0
            return
        if self._ws_unhealthy_since == 0.0:
            self._ws_unhealthy_since = now
            return
        if (now - self._ws_unhealthy_since) < stale_s:
            return
        self._halt_trading = True
        self.logger.error(
            "event=ws_unhealthy duration_s=%.1f threshold_s=%.1f action=close_all_halt",
            now - self._ws_unhealthy_since,
            stale_s,
        )
        for _, ps in list(self.positions.items()):
            px = last_px.get(ps.symbol)
            await self._close_position_market(ps, reason="WS_DISCONNECT", exit_price=px)
