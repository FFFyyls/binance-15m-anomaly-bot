import argparse
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_quote_usdt", type=float, default=2e7)
    ap.add_argument("--blacklist", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--out", default="configs/symbols/symbols.txt")
    args = ap.parse_args()

    blacklist = set([s.strip().upper() for s in args.blacklist.split(",") if s.strip()])

    session = requests.Session()
    session.trust_env = False
    ex = session.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
    symbols = []
    for s in ex.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        sym = s.get("symbol")
        if sym and sym not in blacklist:
            symbols.append(sym)

    t24 = session.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=30).json()
    qv = {x["symbol"]: float(x.get("quoteVolume", 0) or 0) for x in t24 if "symbol" in x}

    universe = [s for s in symbols if qv.get(s, 0.0) >= args.min_quote_usdt]
    universe = sorted(universe, key=lambda x: qv.get(x, 0.0), reverse=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in universe:
            f.write(s + "\n")

    print(f"Saved {len(universe)} symbols -> {args.out}")
    if universe[:10]:
        print("Top10:", ", ".join(universe[:10]))

if __name__ == "__main__":
    main()
