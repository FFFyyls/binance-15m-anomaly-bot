import asyncio
import argparse
import math
import os
import signal as os_signal
import time
from pathlib import Path

import aiohttp
import websockets

import yaml
from dotenv import load_dotenv

from binance_anomaly_bot.binance_client import BinanceFuturesClient
from binance_anomaly_bot.market_ws import MarketWs
from binance_anomaly_bot.user_ws import UserWs
from binance_anomaly_bot.universe import UniverseSelector
from binance_anomaly_bot.strategy import StrategyEngine
from binance_anomaly_bot.storage import SqliteStore
from binance_anomaly_bot.utils import setup_logger

def _force_direct_env() -> None:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        if key in os.environ:
            os.environ[key] = ""
    os.environ["NO_PROXY"] = "*"

DEFAULT_CONFIG_PATH = Path("configs/strategy.yaml")
DEFAULT_DATABASE_PATH = Path("data/runtime/bot.sqlite3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Binance USDT-M anomaly trading bot.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--reset-database",
        action="store_true",
        help="Delete the selected runtime database before startup.",
    )
    return parser


def prepare_database_path(path: Path, *, reset: bool = False) -> Path:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and database_path.exists():
        database_path.unlink()
    return database_path


def resolve_runtime_mode(environ=None) -> tuple[bool, bool]:
    source = os.environ if environ is None else environ
    dry_run_raw = source.get("DRY_RUN", "1")
    use_testnet_raw = source.get("USE_TESTNET", "1")

    for name, value in (("DRY_RUN", dry_run_raw), ("USE_TESTNET", use_testnet_raw)):
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be 0 or 1")

    dry_run = dry_run_raw == "1"
    use_testnet = use_testnet_raw == "1"
    if not dry_run and not use_testnet:
        confirmation = source.get("CONFIRM_LIVE_TRADING", "")
        if confirmation != "I_UNDERSTAND":
            raise RuntimeError(
                "Live production trading requires CONFIRM_LIVE_TRADING=I_UNDERSTAND"
            )
    return dry_run, use_testnet

async def _check_rest(logger, url: str, attempts: int, min_success: int) -> bool:
    ok = 0
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        for i in range(1, attempts + 1):
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        await resp.json()
                        ok += 1
                    else:
                        logger.warning(
                            "event=startup_rest_check_non_200 attempt=%d/%d status=%s",
                            i,
                            attempts,
                            resp.status,
                        )
            except Exception as e:
                logger.warning("event=startup_rest_check_failed attempt=%d/%d err=%s", i, attempts, e)
            await asyncio.sleep(1.0)
    logger.info("event=startup_rest_check_ok ok=%d/%d url=%s", ok, attempts, url)
    return ok >= min_success

async def _check_ws(logger, url: str, attempts: int, min_success: int) -> bool:
    ok = 0
    for i in range(1, attempts + 1):
        try:
            async with websockets.connect(
                url,
                ping_interval=None,
                max_size=2**23,
                open_timeout=8,
                close_timeout=5,
            ) as ws:
                await asyncio.wait_for(ws.recv(), timeout=8)
                ok += 1
        except Exception as e:
            logger.warning("event=startup_ws_check_failed attempt=%d/%d err=%s", i, attempts, e)
        await asyncio.sleep(1.0)
    logger.info("event=startup_ws_check_ok ok=%d/%d url=%s", ok, attempts, url)
    return ok >= min_success

async def preflight_network_check(logger, use_testnet: bool) -> None:
    if os.getenv("SKIP_STARTUP_NET_CHECK", "0") == "1":
        logger.info("event=startup_net_check_skipped reason=env")
        return
    rest_base = "https://demo-fapi.binance.com" if use_testnet else "https://fapi.binance.com"
    rest_url = f"{rest_base}/fapi/v1/time"
    ws_base = os.getenv("BINANCE_WS_BASE", "").strip() or (
        "wss://stream.binancefuture.com" if use_testnet else "wss://fstream.binance.com"
    )
    ws_url = f"{ws_base}/ws/!markPrice@arr@1s"
    attempts = 3
    min_success = 2
    rest_ok = await _check_rest(logger, rest_url, attempts, min_success)
    ws_ok = await _check_ws(logger, ws_url, attempts, min_success)
    if not rest_ok or not ws_ok:
        raise RuntimeError("Startup network check failed; verify REST/WS connectivity and proxy.")

async def _wait_for_condition(check, timeout_s: float, interval_s: float = 0.2) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if check():
            return True
        await asyncio.sleep(interval_s)
    return False

async def startup_self_check(logger, market_ws: MarketWs, user_ws: UserWs,
                             engine: StrategyEngine, config: dict, dry_run: bool) -> None:
    attempts = 3
    timeout_s = 12.0
    retry_delay_s = 3.0
    use_aggtrades = bool(config.get("engine", {}).get("use_aggtrades_ws", False))
    use_account_updates = bool(config.get("position", {}).get("use_account_update_sync", True))

    last_errors = []
    for attempt in range(1, attempts + 1):
        logger.info("event=startup_self_check_attempt attempt=%d/%d", attempt, attempts)
        errors = []

        books_ok = await _wait_for_condition(lambda: bool(market_ws.books), timeout_s)
        if not books_ok:
            errors.append("market_ws_no_book")
        marks_ok = await _wait_for_condition(lambda: bool(market_ws.marks), timeout_s)
        if not marks_ok:
            errors.append("market_ws_no_mark")
        if use_aggtrades:
            agg_ok = await _wait_for_condition(
                lambda: bool(market_ws.last_trades) or any(market_ws.agg_bars.values()),
                timeout_s,
            )
            if not agg_ok:
                errors.append("market_ws_no_aggtrade")

        user_ok = await _wait_for_condition(user_ws.is_connected, timeout_s)
        if not user_ok:
            errors.append("user_ws_not_connected")

        if not dry_run and use_account_updates:
            ready_ok = await engine._wait_for_account_update(timeout_s=timeout_s)
            if not ready_ok:
                errors.append("account_update_not_ready")

        if not errors:
            logger.info("event=startup_self_check_ok")
            return

        last_errors = errors
        logger.warning("event=startup_self_check_failed attempt=%d errors=%s", attempt, ",".join(errors))
        if attempt < attempts:
            await asyncio.sleep(retry_delay_s)

    raise RuntimeError(f"Startup self-check failed: {', '.join(last_errors)}")

async def main(argv=None):
    args = build_parser().parse_args(argv)
    load_dotenv()
    _force_direct_env()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    log = setup_logger(config["engine"]["log_level"])

    dry_run, use_testnet = resolve_runtime_mode()

    await preflight_network_check(log, use_testnet)

    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET in .env")

    rest_cfg = config.get("rest", {}) or {}
    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=use_testnet,
        logger=log,
        rest_min_interval_s=float(rest_cfg.get("min_interval_s", 0.1) or 0.1),
        rest_max_retries=int(rest_cfg.get("max_retries", 2) or 2),
        rest_retry_base_s=float(rest_cfg.get("retry_base_s", 0.5) or 0.5),
        rest_max_retry_s=float(rest_cfg.get("max_retry_s", 5.0) or 5.0),
    )
    await client.sync_time()

    db_path = prepare_database_path(args.database, reset=args.reset_database)
    if args.reset_database:
        log.warning("Reset runtime database before startup: %s", db_path)
    store = SqliteStore(str(db_path))
    await store.init()

    selector = UniverseSelector(client=client, config=config, logger=log)
    universe = await selector.build_universe()
    log.info("Universe size=%d", len(universe))
    symbols = sorted(universe.keys())
    log.info("event=monitor_symbols count=%d", len(symbols))
    if symbols:
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            chunk = ",".join(symbols[i:i + batch_size])
            log.info("event=monitor_symbols_batch index=%d symbols=%s", (i // batch_size) + 1, chunk)
    try:
        position_mode = await validate_account(client, config, log, dry_run)
    except Exception:
        await client.close()
        raise

    bar_interval_s = int(config["engine"].get("bar_interval_s", 20))
    signal_window_minutes = float(config.get("signal", {}).get("window_minutes", 0) or 0)
    bars_per_window = int(signal_window_minutes * 60 / bar_interval_s) if bar_interval_s > 0 else 0
    default_bars_keep = max(bars_per_window * 4, bars_per_window + 2)
    try:
        bars_keep = int(config["engine"].get("agg_trade_bars_maxlen", 0) or 0)
    except (TypeError, ValueError):
        bars_keep = 0
    try:
        keep_minutes = float(config["engine"].get("agg_trade_keep_minutes", 0) or 0)
    except (TypeError, ValueError):
        keep_minutes = 0.0
    if bars_keep <= 0:
        if keep_minutes <= 0:
            bb_window_minutes = float(config.get("bollinger", {}).get("window_minutes", 0) or 0)
            target_window = max(signal_window_minutes, bb_window_minutes)
            if target_window > 0:
                keep_minutes = target_window + 0.5
        if keep_minutes > 0 and bar_interval_s > 0:
            bars_keep = math.ceil(keep_minutes * 60 / bar_interval_s)
        else:
            bars_keep = default_bars_keep
    if bars_keep < (bars_per_window + 2):
        bars_keep = bars_per_window + 2

    market_ws = MarketWs(
        logger=log,
        use_testnet=use_testnet,
        agg_trade_symbols=list(universe.keys()),
        agg_trade_interval_s=bar_interval_s,
        enable_agg_trades=bool(config["engine"].get("use_aggtrades_ws", False)),
        agg_trade_chunk_size=int(config["engine"].get("agg_trade_chunk_size", 200)),
        agg_trade_bars_maxlen=bars_keep,
        store=store,
    )
    user_ws = UserWs(client=client, logger=log, use_testnet=use_testnet, store=store)

    stop_event = asyncio.Event()
    def _stop(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    tasks = [
        asyncio.create_task(market_ws.run()),
        asyncio.create_task(user_ws.run()),
    ]

    engine = StrategyEngine(
        client=client,
        market_ws=market_ws,
        user_ws=user_ws,
        store=store,
        universe=universe,
        config=config,
        dry_run=dry_run,
        logger=log,
        position_mode=position_mode,
    )
    try:
        await startup_self_check(log, market_ws, user_ws, engine, config, dry_run)
    except Exception as e:
        log.error("event=startup_self_check_abort err=%s", e)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await user_ws.close()
        await market_ws.close()
        await client.close()
        raise

    tasks.append(asyncio.create_task(engine.run()))

    await stop_event.wait()
    log.warning("Shutdown requested. Cancelling tasks...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await user_ws.close()
    await market_ws.close()
    await client.close()

async def validate_account(client: BinanceFuturesClient, config: dict, logger, dry_run: bool) -> str:
    try:
        r = await client.account_info()
    except Exception as e:
        raise RuntimeError(f"account_info failed: {e}") from e
    if r.status != 200:
        raise RuntimeError(f"account_info error: {r.data}")

    data = r.data
    if not data.get("canTrade", True):
        raise RuntimeError("Account cannot trade (canTrade=false)")

    avail = float(data.get("availableBalance", 0) or 0)
    total = float(data.get("totalWalletBalance", 0) or 0)
    min_notional = float(config["position"].get("min_notional_usdt", 0))
    if total <= 0:
        logger.warning("event=account_balance_zero")
    if min_notional and avail < min_notional:
        logger.warning("event=balance_below_min_notional available=%.4f min_notional=%.4f", avail, min_notional)

    if dry_run:
        logger.info("event=dry_run_enabled")
    position_mode = "ONEWAY"
    try:
        r = await client.position_side_dual()
        if r.status == 200 and "dualSidePosition" in r.data:
            position_mode = "HEDGE" if bool(r.data["dualSidePosition"]) else "ONEWAY"
            logger.info("event=position_mode mode=%s", position_mode)
        else:
            logger.warning("event=position_mode_query_failed err=%s", r.data)
    except Exception as e:
        logger.warning("event=position_mode_query_error err=%s", e)
    return position_mode

def run(argv=None) -> None:
    asyncio.run(main(argv))


if __name__ == "__main__":
    run()
