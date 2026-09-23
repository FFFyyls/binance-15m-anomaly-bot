import asyncio
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

import orjson
import websockets

from .binance_client import BinanceFuturesClient
from .utils import utc_ms

@dataclass
class OrderUpdate:
    symbol: str
    order_id: int
    status: str
    side: str
    order_type: str
    price: float
    orig_qty: float
    executed_qty: float
    avg_price: float
    update_time: int
    client_order_id: str
    fee: Optional[float]
    fee_asset: Optional[str]

@dataclass
class AccountPositionUpdate:
    symbol: str
    position_side: str
    position_amt: float
    entry_price: float
    update_time: int

class UserWs:
    """
    User Data Stream:
      - listenKey valid 60 minutes (needs keepalive)
      - WS endpoint: wss://fstream.binance.com/ws/<listenKey>
    """
    def __init__(self, client: BinanceFuturesClient, logger, use_testnet: bool = False,
                 proxy_url: Optional[str] = None, store=None, force_direct: bool = True):
        self.client = client
        self.logger = logger
        self.use_testnet = use_testnet
        self.base = os.getenv("BINANCE_WS_BASE", "").strip() or (
            "wss://stream.binancefuture.com" if use_testnet else "wss://fstream.binance.com"
        )
        self.listen_key: Optional[str] = None
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self.proxy_url = None if force_direct else (proxy_url or "")
        self.store = store
        self.connected = False
        self.last_disconnect_ts = 0.0
        self._listen_key_refresh = asyncio.Event()
        self._listen_key_lock = asyncio.Lock()

        self.order_updates: asyncio.Queue[OrderUpdate] = asyncio.Queue()
        self.account_updates: asyncio.Queue[List[AccountPositionUpdate]] = asyncio.Queue()
        self.account_ready = asyncio.Event()

    def _schedule_error_log(self, stage: str, *, event: str, reason: str = "",
                            error: Optional[Exception] = None, details: Optional[dict] = None) -> None:
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
            event=event,
            reason=reason,
            error=err_text,
            details=payload,
        ))

    async def run(self):
        if self.proxy_url:
            self.logger.info("event=user_ws_proxy_enabled")
        self.logger.info("event=user_ws_base url=%s", self.base)
        try:
            self.listen_key = await self.client.listen_key()
        except Exception as e:
            self._schedule_error_log(
                "order_fill",
                event="listen_key_failed",
                error=e,
                details={"traceback": traceback.format_exc()},
            )
            raise
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        self._task = asyncio.create_task(self._consume())
        await asyncio.gather(self._task, self._keepalive_task)

    async def close(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
        if self._keepalive_task:
            self._keepalive_task.cancel()
        await asyncio.gather(*(t for t in [self._task, self._keepalive_task] if t), return_exceptions=True)

    def _is_listen_key_invalid(self, data) -> bool:
        if not isinstance(data, dict):
            return False
        code = data.get("code")
        msg = str(data.get("msg", "")).lower()
        if code == -1125:
            return True
        return "listenkey" in msg and "does not exist" in msg

    async def _refresh_listen_key(self, reason: str) -> None:
        async with self._listen_key_lock:
            try:
                self.listen_key = await self.client.listen_key()
            except Exception as e:
                self.logger.warning("event=listenkey_refresh_failed reason=%s err=%s", reason, e)
                self._schedule_error_log(
                    "order_fill",
                    event="listenkey_refresh_failed",
                    reason=reason,
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )
                return
            self._listen_key_refresh.set()
            self.logger.warning("event=listenkey_refreshed reason=%s", reason)

    async def _keepalive_loop(self):
        while not self._stop.is_set():
            try:
                await asyncio.sleep(30 * 60)
                r = await self.client.keepalive_listen_key()
                if r.status == 200:
                    self.logger.info("event=user_ws_keepalive_sent")
                elif self._is_listen_key_invalid(r.data):
                    await self._refresh_listen_key("keepalive_invalid")
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.logger.warning("event=user_ws_keepalive_error err=%s", e)
                self._schedule_error_log(
                    "order_fill",
                    event="user_ws_keepalive_error",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )

    async def _consume(self):
        while not self._stop.is_set():
            try:
                if not self.listen_key:
                    await asyncio.sleep(1)
                    continue
                url = f"{self.base}/ws/{self.listen_key}"
                async with websockets.connect(
                    url,
                    ping_interval=None,
                    max_size=2**23,
                    proxy=self.proxy_url if self.proxy_url else None,
                ) as ws:
                    self.logger.info("event=user_ws_connected")
                    self.connected = True
                    self._listen_key_refresh.clear()
                    while not self._stop.is_set():
                        if self._listen_key_refresh.is_set():
                            await ws.close()
                            break
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue
                        self.last_disconnect_ts = 0.0
                        self._handle(msg)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.connected = False
                self.last_disconnect_ts = time.time()
                self.logger.warning("event=user_ws_error err=%s action=reconnect_s2", e)
                self._schedule_error_log(
                    "order_fill",
                    event="user_ws_error",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )
                await asyncio.sleep(2)

    def _handle(self, raw: str):
        try:
            j = orjson.loads(raw)
        except Exception as e:
            self._schedule_error_log(
                "order_fill",
                event="user_ws_parse_failed",
                reason="json_decode",
                error=e,
                details={"raw": raw[:512], "traceback": traceback.format_exc()},
            )
            return
        if j.get("e") == "ACCOUNT_UPDATE":
            a = j.get("a", {})
            positions = a.get("P", []) if isinstance(a, dict) else []
            updates: List[AccountPositionUpdate] = []
            update_time = int(j.get("E", 0) or 0)
            for p in positions:
                if not isinstance(p, dict):
                    continue
                sym = p.get("s")
                if not sym:
                    continue
                try:
                    amt = float(p.get("pa", 0.0) or 0.0)
                    entry = float(p.get("ep", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                pos_side = str(p.get("ps", "") or "").upper() or "BOTH"
                updates.append(AccountPositionUpdate(
                    symbol=sym,
                    position_side=pos_side,
                    position_amt=amt,
                    entry_price=entry,
                    update_time=update_time,
                ))
            if updates:
                try:
                    self.account_updates.put_nowait(updates)
                except asyncio.QueueFull:
                    try:
                        self.account_updates.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self.account_updates.put_nowait(updates)
                    except asyncio.QueueFull:
                        pass
            self.account_ready.set()
            return
        if j.get("e") == "listenKeyExpired":
            self.logger.warning("event=listenkey_expired")
            asyncio.create_task(self._refresh_listen_key("listenkey_expired"))
            return
        if j.get("e") == "ORDER_TRADE_UPDATE":
            o = j.get("o", {})
            try:
                fee_val = None
                fee_raw = o.get("n")
                if fee_raw not in (None, ""):
                    try:
                        fee_val = float(fee_raw)
                    except (TypeError, ValueError):
                        fee_val = None
                fee_asset = o.get("N") or None
                if isinstance(fee_asset, str) and fee_asset.strip() == "":
                    fee_asset = None
                upd = OrderUpdate(
                    symbol=o["s"],
                    order_id=int(o["i"]),
                    status=o["X"],
                    side=o["S"],
                    order_type=o["o"],
                    price=float(o.get("p", "0") or 0),
                    orig_qty=float(o.get("q", "0") or 0),
                    executed_qty=float(o.get("z", "0") or 0),
                    avg_price=float(o.get("ap", "0") or 0),
                    update_time=int(o.get("T", j.get("E", 0)) or 0),
                    client_order_id=o.get("c", ""),
                    fee=fee_val,
                    fee_asset=fee_asset,
                )
                self.order_updates.put_nowait(upd)
            except Exception as e:
                self._schedule_error_log(
                    "order_fill",
                    event="user_ws_parse_failed",
                    reason="order_update",
                    error=e,
                    details={"traceback": traceback.format_exc()},
                )
                return

    def is_connected(self) -> bool:
        return self.connected
