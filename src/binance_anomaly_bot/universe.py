import asyncio
from dataclasses import dataclass
from typing import Dict, Set, Tuple

import os

from .binance_client import BinanceFuturesClient

@dataclass
class SymbolSpec:
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float

class UniverseSelector:
    def __init__(self, client: BinanceFuturesClient, config: dict, logger):
        self.client = client
        self.config = config
        self.logger = logger

    async def _fetch_exchange_info(self, retries: int = 1, retry_delay_s: float = 0.5):
        for attempt in range(retries + 1):
            try:
                ex = await self.client.exchange_info()
                if not isinstance(ex, dict):
                    raise TypeError(f"exchange_info_type={type(ex).__name__}")
                return ex
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(retry_delay_s)
                    continue
                self.logger.warning("event=exchange_info_failed err=%s", e)
                return None

    async def _fetch_ticker_24hr(self, retries: int = 1, retry_delay_s: float = 0.5):
        for attempt in range(retries + 1):
            try:
                t24 = await self.client.ticker_24hr()
                if not isinstance(t24, list):
                    raise TypeError(f"ticker_24hr_type={type(t24).__name__}")
                if not t24:
                    raise ValueError("ticker_24hr_empty")
                return t24
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(retry_delay_s)
                    continue
                self.logger.warning("event=ticker_24hr_failed err=%s", e)
                return None

    async def build_universe(self) -> Dict[str, SymbolSpec]:
        ignore_whitelist = bool(self.config["universe"].get("ignore_whitelist", False))
        whitelist = set() if ignore_whitelist else self._load_whitelist()
        ex = await self._fetch_exchange_info()
        if not ex:
            raise RuntimeError("exchange_info_unavailable")
        specs = []
        for s in ex.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            sym = s["symbol"]
            if sym in set(self.config["universe"].get("blacklist", [])):
                continue
            if whitelist and sym not in whitelist:
                continue

            tick = step = min_qty = 0.0
            min_notional = 0.0
            for f in s.get("filters", []):
                if f.get("filterType") == "PRICE_FILTER":
                    tick = float(f["tickSize"])
                if f.get("filterType") == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = float(f.get("notional", f.get("minNotional", 0.0)) or 0.0)

            specs.append(SymbolSpec(sym, tick, step, min_qty, min_notional))

        t24 = await self._fetch_ticker_24hr()
        if not t24:
            raise RuntimeError("ticker_24hr_unavailable")
        qv = {x["symbol"]: float(x.get("quoteVolume", 0.0) or 0.0) for x in t24 if "symbol" in x}
        min_q = float(self.config["universe"]["min_24h_quote_usdt"])
        top_n = int(self.config["universe"].get("top_n_24h_quote", 0) or 0)
        ranked: list[Tuple[float, SymbolSpec]] = [
            (qv.get(sp.symbol, 0.0), sp) for sp in specs if qv.get(sp.symbol, 0.0) >= min_q
        ]
        ranked.sort(key=lambda x: x[0], reverse=True)
        if top_n > 0:
            ranked = ranked[:top_n]
        universe = {sp.symbol: sp for _, sp in ranked}
        self.logger.info(
            "event=universe_built base=%d after_24h_quote_filter=%d min_quote=%s top_n=%d",
            len(specs),
            len(ranked),
            min_q,
            top_n if top_n > 0 else len(ranked),
        )
        return universe

    def _load_whitelist(self) -> Set[str]:
        path = str(self.config["universe"].get("whitelist_symbols_file", "") or "").strip()
        if not path:
            return set()

        # Resolve repository-managed symbol lists from the standard config directory.
        potential_path = os.path.join("configs/symbols", path)
        if os.path.exists(potential_path):
            path = potential_path

        if not os.path.exists(path):
            self.logger.warning("event=whitelist_missing path=%s", path)
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().upper() for line in f if line.strip() and not line.strip().startswith("#")}
