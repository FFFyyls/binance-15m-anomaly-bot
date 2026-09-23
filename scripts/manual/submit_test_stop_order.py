#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dotenv import load_dotenv

from binance_anomaly_bot.binance_client import BinanceFuturesClient
from binance_anomaly_bot.utils import setup_logger, utc_ms


def validate_order_environment(*, use_testnet: bool, confirm_live_order: bool) -> None:
    if not use_testnet and not confirm_live_order:
        raise RuntimeError(
            "Refusing to submit a production order without --confirm-live-order."
        )


async def _run_test(args: argparse.Namespace) -> int:
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET in .env")

    use_testnet = os.getenv("USE_TESTNET", "1") == "1"
    validate_order_environment(
        use_testnet=use_testnet,
        confirm_live_order=args.confirm_live_order,
    )
    logger = setup_logger("INFO")
    client = None
    try:
        client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            use_testnet=use_testnet,
            logger=logger,
        )
        await client.sync_time()

        client_id = args.client_id or f"ALGOSL_{args.symbol}_{utc_ms()}"
        params = dict(
            symbol=args.symbol,
            side=args.side,
            orderType="STOP_MARKET",
            quantity=str(args.quantity),
            stopPrice=str(args.stop_price),
            newClientOrderId=client_id,
            newOrderRespType="ACK",
            workingType=args.working_type,
            priceProtect="true" if args.price_protect else "false",
        )
        if args.position_side:
            params["positionSide"] = args.position_side
        if args.reduce_only:
            params["reduceOnly"] = "true"

        resp = await client.new_algo_order(**params)
        print(f"status={resp.status}")
        print(resp.data)
        return 0 if resp.status == 200 else 1
    finally:
        if client is not None:
            await client.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Binance Algo STOP_MARKET order.")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL"], help="Order side")
    parser.add_argument("--quantity", required=True, type=float, help="Order quantity")
    parser.add_argument("--stop-price", required=True, type=float, help="Stop trigger price")
    parser.add_argument("--position-side", default="", choices=["", "LONG", "SHORT", "BOTH"])
    parser.add_argument("--working-type", default="MARK_PRICE", choices=["MARK_PRICE", "CONTRACT_PRICE"])
    parser.add_argument("--price-protect", dest="price_protect", action="store_true")
    parser.add_argument("--no-price-protect", dest="price_protect", action="store_false")
    parser.set_defaults(price_protect=True)
    parser.add_argument("--reduce-only", dest="reduce_only", action="store_true")
    parser.add_argument("--no-reduce-only", dest="reduce_only", action="store_false")
    parser.set_defaults(reduce_only=True)
    parser.add_argument("--client-id", default="", help="Override client order id")
    parser.add_argument(
        "--confirm-live-order",
        action="store_true",
        help="Required when USE_TESTNET=0; acknowledges that a real order will be sent.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raise SystemExit(asyncio.run(_run_test(args)))


if __name__ == "__main__":
    main()
