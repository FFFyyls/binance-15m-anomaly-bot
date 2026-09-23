import asyncio
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import orjson
import websockets

from .utils import utc_ms
@dataclass
class Book:
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    ts: int

@dataclass
class Mark:
    mark: float
    funding_rate: float
    next_funding_time: int
    ts: int

@dataclass
class AggBar:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    close_time: int

class MarketWs:
    """
    USDⓈ-M Futures Market Streams
    Base URL: wss://fstream.binance.com
    Subscribed streams:
      - !bookTicker (all symbols best bid/ask)
      - !markPrice@arr@1s (all symbols mark price + funding rate)
    """
    def __init__(self, logger, use_testnet: bool = False,
                 agg_trade_symbols: Optional[List[str]] = None,
                 agg_trade_interval_s: int = 20,
                 enable_agg_trades: bool = False,
                 agg_trade_bars_maxlen: int = 2000,
                 agg_trade_chunk_size: int = 200,
                 proxy_url: Optional[str] = None,
                 store=None,
                 force_direct: bool = True):
        self.logger = logger
        self.use_testnet = use_testnet
        self.base = os.getenv("BINANCE_WS_BASE", "").strip() or (
            "wss://stream.binancefuture.com" if use_testnet else "wss://fstream.binance.com"
        )
        self.books: Dict[str, Book] = {}
        self.marks: Dict[str, Mark] = {}
        self.enable_agg_trades = enable_agg_trades
        self.agg_trade_symbols = [s.lower() for s in (agg_trade_symbols or [])]
        self.agg_trade_interval_s = agg_trade_interval_s
        self.agg_bars: Dict[str, Deque[AggBar]] = {}
        self._agg_bar_state: Dict[str, AggBar] = {}
        self.last_trades: Dict[str, Tuple[float, int]] = {}
        self._agg_bars_maxlen = agg_trade_bars_maxlen
        self._agg_trade_chunk_size = agg_trade_chunk_size
        self._agg_trade_stream_ids: Set[str] = set()
        self._agg_trade_tasks: Dict[str, asyncio.Task] = {}
        self._connected_streams: Set[str] = set()
        self.last_disconnect_ts = 0.0
        self.proxy_url = None if force_direct else (proxy_url or "")
        self.store = store
        self._stop = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._base_tasks: List[asyncio.Task] = []
        self._running = False

    def _schedule_error_log(self, stage: str, *, event: str, symbol: Optional[str] = None,
                            reason: str = "", error: Optional[Exception] = None,
                            details: Optional[dict] = None) -> None:
        if not self.store:
            return
        payload = dict(details or {})
        err_text = None
        if error is not None:
            err_text = f"{type(error).__name__}: {error}"
            payload.setdefault("error_type", type(error).__name__)
            payload.setdefault("error_message", str(error))
        if not payload:
            payload = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.create_task(self.store.insert_error_log(
            ts=utc_ms(),
            stage=stage,
            symbol=symbol,
            event=event,
            reason=reason,
            error=err_text,
            details=payload,
        ))

    async def run(self):
        self._running = True
        if self.proxy_url:
            self.logger.info("event=market_ws_proxy_enabled")
        self.logger.info("event=market_ws_base url=%s", self.base)
        self._base_tasks = [
            asyncio.create_task(self._consume_stream("!bookTicker", self._handle_book)),
            asyncio.create_task(self._consume_stream("!markPrice@arr@1s", self._handle_mark_arr)),
        ]
        self._tasks = list(self._base_tasks)
        if self.enable_agg_trades and self.agg_trade_symbols:
            self._start_agg_trade_tasks(self.agg_trade_symbols)
        await asyncio.gather(*self._tasks)

    async def close(self):
        self._stop.set()
        await self._stop_agg_trade_tasks()
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._running = False

    async def update_agg_trade_symbols(self, symbols: List[str]) -> None:
        symbols_norm = [s.lower() for s in symbols]
        if set(symbols_norm) == set(self.agg_trade_symbols):
            return
        self.agg_trade_symbols = symbols_norm
        if not self.enable_agg_trades or not self._running:
            return
        await self._stop_agg_trade_tasks()
        if self.agg_trade_symbols:
            self._start_agg_trade_tasks(self.agg_trade_symbols)

    def remove_symbol_data(self, symbols: Iterable[str]) -> None:
        for sym in symbols:
            self.agg_bars.pop(sym, None)
            self._agg_bar_state.pop(sym, None)
            self.last_trades.pop(sym, None)

    async def _consume_stream(self, stream_name: str, handler):
        url = f"{self.base}/ws/{stream_name}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=None,
                    max_size=2**23,
                    proxy=self.proxy_url if self.proxy_url else None,
                ) as ws:
                    self.logger.info("event=market_ws_connected stream=%s", stream_name)
                    self._connected_streams.add(stream_name)
                    async for msg in ws:
                        self.last_disconnect_ts = 0.0
                        handler(msg)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._connected_streams.discard(stream_name)
                self.last_disconnect_ts = time.time()
                self.logger.warning("event=market_ws_error stream=%s err=%s action=reconnect_s2", stream_name, e)
                self._schedule_error_log(
                    "signal_acquire",
                    event="market_ws_error",
                    reason=stream_name,
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )
                await asyncio.sleep(2)

    async def _consume_agg_trades(self, symbols: List[str], stream_id: str):
        stream = "/".join([f"{s}@aggTrade" for s in symbols])
        url = f"{self.base}/stream?streams={stream}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=None,
                    max_size=2**23,
                    proxy=self.proxy_url if self.proxy_url else None,
                ) as ws:
                    self.logger.info(
                        "event=market_ws_connected stream=%s symbols=%d",
                        stream_id,
                        len(symbols),
                    )
                    self._connected_streams.add(stream_id)
                    async for msg in ws:
                        self.last_disconnect_ts = 0.0
                        self._handle_agg_trade(msg)
            except asyncio.CancelledError:
                self._connected_streams.discard(stream_id)
                return
            except Exception as e:
                self._connected_streams.discard(stream_id)
                self.last_disconnect_ts = time.time()
                self.logger.warning("event=market_ws_error stream=%s err=%s action=reconnect_s2", stream_id, e)
                self._schedule_error_log(
                    "signal_aggregate",
                    event="market_ws_error",
                    reason=stream_id,
                    error=e,
                    details={
                        "symbols": len(symbols),
                        "traceback": traceback.format_exc(),
                    },
                )
                await asyncio.sleep(2)

    def _handle_book(self, raw: str):
        try:
            j = orjson.loads(raw)
            if j.get("e") != "bookTicker":
                return
            s = j["s"]
            self.books[s] = Book(
                bid=float(j["b"]), bid_qty=float(j["B"]),
                ask=float(j["a"]), ask_qty=float(j["A"]),
                ts=int(j["E"]),
            )
        except Exception as e:
            self._schedule_error_log(
                "signal_acquire",
                event="market_ws_parse_failed",
                reason="bookTicker",
                error=e,
                details={"raw": raw[:512], "traceback": traceback.format_exc()},
            )
            return

    def _handle_mark_arr(self, raw: str):
        try:
            arr = orjson.loads(raw)
            if not isinstance(arr, list):
                return
            for j in arr:
                if j.get("e") != "markPriceUpdate":
                    continue
                s = j["s"]
                self.marks[s] = Mark(
                    mark=float(j["p"]),
                    funding_rate=float(j.get("r", 0.0)),
                    next_funding_time=int(j.get("T", 0)),
                    ts=int(j["E"]),
                )
        except Exception as e:
            self._schedule_error_log(
                "signal_acquire",
                event="market_ws_parse_failed",
                reason="markPrice",
                error=e,
                details={"raw": raw[:512], "traceback": traceback.format_exc()},
            )
            return

    def _handle_agg_trade(self, raw: str):
        try:
            j = orjson.loads(raw)
            data = j.get("data", j)
            if data.get("e") != "aggTrade":
                return
            sym = data["s"]
            price = float(data["p"])
            qty = float(data["q"])
            ts = int(data["T"])
            self.last_trades[sym] = (price, ts)
            interval_ms = self.agg_trade_interval_s * 1000
            bucket = ts - (ts % interval_ms)

            state = self._agg_bar_state.get(sym)
            if state is None or state.open_time != bucket:
                if state is not None:
                    self._store_bar(sym, state)
                state = AggBar(
                    open_time=bucket,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=qty,
                    quote_volume=price * qty,
                    close_time=bucket + interval_ms - 1,
                )
                self._agg_bar_state[sym] = state
                return

            state.high = max(state.high, price)
            state.low = min(state.low, price)
            state.close = price
            state.volume += qty
            state.quote_volume += price * qty
        except Exception as e:
            self._schedule_error_log(
                "signal_aggregate",
                event="market_ws_parse_failed",
                reason="aggTrade",
                error=e,
                details={"raw": raw[:512], "traceback": traceback.format_exc()},
            )
            return

    def _store_bar(self, sym: str, bar: AggBar) -> None:
        dq = self.agg_bars.get(sym)
        if dq is None:
            dq = deque(maxlen=self._agg_bars_maxlen)
            self.agg_bars[sym] = dq
        dq.append(bar)

    def get_recent_bars(self, symbol: str, n: int) -> List[AggBar]:
        dq = self.agg_bars.get(symbol)
        if not dq:
            return []
        return list(dq)[-n:]

    def get_recent_bars_with_current(self, symbol: str, n: int) -> List[AggBar]:
        if n <= 0:
            return []
        dq = self.agg_bars.get(symbol)
        bars = list(dq) if dq else []
        state = self._agg_bar_state.get(symbol)
        if state:
            if not bars or state.open_time > bars[-1].open_time:
                bars.append(state)
            elif state.open_time == bars[-1].open_time:
                bars[-1] = state
        return bars[-n:]

    def is_healthy(self) -> bool:
        required = {"!bookTicker", "!markPrice@arr@1s"}
        if self.enable_agg_trades and self.agg_trade_symbols:
            required |= self._agg_trade_stream_ids
        return required.issubset(self._connected_streams)

    def _chunk(self, items: List[str], size: int) -> List[List[str]]:
        if size <= 0:
            return [items]
        return [items[i:i + size] for i in range(0, len(items), size)]

    def _start_agg_trade_tasks(self, symbols: List[str]) -> None:
        self._agg_trade_stream_ids.clear()
        self._agg_trade_tasks.clear()
        chunks = list(self._chunk(symbols, self._agg_trade_chunk_size))
        for idx, chunk in enumerate(chunks, start=1):
            stream_id = f"aggTrade-{idx}"
            self._agg_trade_stream_ids.add(stream_id)
            task = asyncio.create_task(self._consume_agg_trades(chunk, stream_id))
            self._agg_trade_tasks[stream_id] = task
            self._tasks.append(task)

    async def _stop_agg_trade_tasks(self) -> None:
        if not self._agg_trade_tasks:
            return
        tasks = list(self._agg_trade_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream_id in list(self._agg_trade_stream_ids):
            self._connected_streams.discard(stream_id)
        self._agg_trade_tasks.clear()
        self._agg_trade_stream_ids.clear()
        task_set = set(tasks)
        self._tasks = [t for t in self._tasks if t not in task_set]
