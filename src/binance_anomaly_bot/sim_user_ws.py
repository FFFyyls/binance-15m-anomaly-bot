import asyncio


class SimUserWs:
    def __init__(self, logger=None) -> None:
        self.logger = logger
        self.order_updates: asyncio.Queue = asyncio.Queue()
        self.account_updates: asyncio.Queue = asyncio.Queue()
        self.account_ready = asyncio.Event()
        self.account_ready.set()
        self._stop = asyncio.Event()

    def is_connected(self) -> bool:
        return True

    async def run(self) -> None:
        await self._stop.wait()

    async def close(self) -> None:
        self._stop.set()
