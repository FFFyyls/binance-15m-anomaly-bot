import asyncio
import time
import aiohttp
import orjson
from urllib.parse import urlencode
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .utils import sign_hmac_sha256, utc_ms

@dataclass
class RestResponse:
    status: int
    data: Any
    headers: Dict[str, str]

class BinanceFuturesClient:
    """
    Minimal USDⓈ-M Futures REST client (HMAC).
    Prod REST base: https://fapi.binance.com
    """
    def __init__(self, api_key: str, api_secret: str, use_testnet: bool, logger,
                 rest_min_interval_s: float = 0.1, rest_max_retries: int = 2,
                 rest_retry_base_s: float = 0.5, rest_max_retry_s: float = 5.0):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.logger = logger
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), trust_env=False)
        self._time_offset_ms = 0
        self._rest_min_interval_s = max(0.0, float(rest_min_interval_s or 0.0))
        self._rest_max_retries = max(0, int(rest_max_retries or 0))
        self._rest_retry_base_s = max(0.0, float(rest_retry_base_s or 0.0))
        self._rest_max_retry_s = max(0.5, float(rest_max_retry_s or 0.5))
        self._rest_lock = asyncio.Lock()
        self._rest_last_ts = 0.0
        self.base_url = "https://fapi.binance.com"
        if use_testnet:
            # Widely used USDⓈ-M futures testnet domain.
            self.base_url = "https://demo-fapi.binance.com"

    async def close(self):
        await self.session.close()

    async def sync_time(self):
        srv = await self.public_get("/fapi/v1/time", params={})
        server_time = int(srv["serverTime"])
        local = utc_ms()
        self._time_offset_ms = server_time - local
        self.logger.info("event=time_sync offset_ms=%d", self._time_offset_ms)

    def _ts(self) -> int:
        return utc_ms() + self._time_offset_ms

    async def _rate_limit(self) -> None:
        if self._rest_min_interval_s <= 0:
            return
        async with self._rest_lock:
            now = time.monotonic()
            wait_s = self._rest_min_interval_s - (now - self._rest_last_ts)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._rest_last_ts = time.monotonic()

    async def _sleep_backoff(self, headers: Dict[str, str], attempt: int) -> None:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        wait_s = 0.0
        if retry_after:
            try:
                wait_s = float(retry_after)
            except (TypeError, ValueError):
                wait_s = 0.0
        if wait_s <= 0:
            wait_s = min(self._rest_max_retry_s, self._rest_retry_base_s * (2 ** attempt))
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    async def public_get(self, path: str, params: Dict[str, Any]) -> Any:
        url = self.base_url + path
        last_err = None
        for attempt in range(self._rest_max_retries + 1):
            await self._rate_limit()
            try:
                async with self.session.get(url, params=params) as resp:
                    if resp.status in (418, 429):
                        if attempt < self._rest_max_retries:
                            await self._sleep_backoff(dict(resp.headers), attempt)
                            continue
                        last_err = RuntimeError(f"rate_limited status={resp.status}")
                        break
                    data = await resp.json(loads=orjson.loads)
                    if resp.status >= 400:
                        raise RuntimeError(f"public_get_failed status={resp.status} data={data}")
                    return data
            except Exception as e:
                last_err = e
                if attempt < self._rest_max_retries:
                    await asyncio.sleep(min(self._rest_max_retry_s, self._rest_retry_base_s * (2 ** attempt)))
                    continue
                break
        raise RuntimeError(f"public_get_failed path={path} err={last_err}")

    async def signed(self, method: str, path: str, params: Dict[str, Any]) -> RestResponse:
        url = self.base_url + path
        params = dict(params)
        params["timestamp"] = self._ts()
        ordered = sorted(params.items())
        query = urlencode(ordered, doseq=True)
        signature = sign_hmac_sha256(self.api_secret, query)
        ordered.append(("signature", signature))
        query = urlencode(ordered, doseq=True)

        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{url}?{query}"
        m = method.upper()
        for attempt in range(self._rest_max_retries + 1):
            await self._rate_limit()
            if m == "GET":
                async with self.session.get(url, headers=headers) as resp:
                    data = await self._read_response(resp)
                    if resp.status in (418, 429) and attempt < self._rest_max_retries:
                        await self._sleep_backoff(dict(resp.headers), attempt)
                        continue
                    return RestResponse(resp.status, data, dict(resp.headers))
            if m == "POST":
                async with self.session.post(url, headers=headers) as resp:
                    data = await self._read_response(resp)
                    if resp.status in (418, 429) and attempt < self._rest_max_retries:
                        await self._sleep_backoff(dict(resp.headers), attempt)
                        continue
                    return RestResponse(resp.status, data, dict(resp.headers))
            if m == "DELETE":
                async with self.session.delete(url, headers=headers) as resp:
                    data = await self._read_response(resp)
                    if resp.status in (418, 429) and attempt < self._rest_max_retries:
                        await self._sleep_backoff(dict(resp.headers), attempt)
                        continue
                    return RestResponse(resp.status, data, dict(resp.headers))
            if m == "PUT":
                async with self.session.put(url, headers=headers) as resp:
                    data = await self._read_response(resp)
                    if resp.status in (418, 429) and attempt < self._rest_max_retries:
                        await self._sleep_backoff(dict(resp.headers), attempt)
                        continue
                    return RestResponse(resp.status, data, dict(resp.headers))
        raise ValueError(f"Unsupported method={method}")

    async def _read_response(self, resp: aiohttp.ClientResponse) -> Any:
        try:
            return await resp.json(loads=orjson.loads)
        except aiohttp.ContentTypeError:
            text = await resp.text()
            return {
                "_raw": text,
                "_content_type": resp.headers.get("Content-Type", ""),
            }

    # --- Public endpoints ---
    async def exchange_info(self) -> Any:
        return await self.public_get("/fapi/v1/exchangeInfo", {})

    async def ticker_24hr(self) -> Any:
        return await self.public_get("/fapi/v1/ticker/24hr", {})

    async def klines(self, symbol: str, interval: str, limit: int = 200) -> Any:
        return await self.public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    # --- Trade endpoints ---
    async def new_order(self, **params) -> RestResponse:
        return await self.signed("POST", "/fapi/v1/order", params)

    async def cancel_order(self, **params) -> RestResponse:
        return await self.signed("DELETE", "/fapi/v1/order", params)

    # --- Algo order endpoints ---
    async def new_algo_order(self, **params) -> RestResponse:
        return await self.signed("POST", "/fapi/v1/algo/order", params)

    async def cancel_algo_order(self, **params) -> RestResponse:
        return await self.signed("DELETE", "/fapi/v1/algo/order", params)

    async def open_orders(self, symbol: Optional[str] = None) -> RestResponse:
        p: Dict[str, Any] = {}
        if symbol:
            p["symbol"] = symbol
        return await self.signed("GET", "/fapi/v1/openOrders", p)

    async def position_risk(self, symbol: Optional[str] = None) -> RestResponse:
        p: Dict[str, Any] = {}
        if symbol:
            p["symbol"] = symbol
        return await self.signed("GET", "/fapi/v2/positionRisk", p)

    async def change_leverage(self, symbol: str, leverage: int) -> RestResponse:
        return await self.signed("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    async def account_info(self) -> RestResponse:
        return await self.signed("GET", "/fapi/v2/account", {})

    async def position_side_dual(self) -> RestResponse:
        return await self.signed("GET", "/fapi/v1/positionSide/dual", {})

    # --- User data stream ---
    async def listen_key(self) -> str:
        r = await self.signed("POST", "/fapi/v1/listenKey", {})
        if r.status != 200:
            raise RuntimeError(f"listenKey error status={r.status} data={r.data}")
        return r.data["listenKey"]

    async def keepalive_listen_key(self) -> RestResponse:
        r = await self.signed("PUT", "/fapi/v1/listenKey", {})
        if r.status != 200:
            self.logger.warning("event=listenkey_keepalive_failed err=%s", r.data)
        return r
