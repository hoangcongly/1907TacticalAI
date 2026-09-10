#!/usr/bin/env python3
"""Tải universe nhiều cặp cho nghiên cứu cross-sectional (relative value)."""
import argparse, logging, sys, time
import pandas as pd
from aegis.data.ingestion.binance_history import download_klines, download_funding_rates, save_klines
from aegis.data.ingestion.binance_rest import BinanceFuturesREST

# Các perp USDT thanh khoản tốt, lịch sử dài (đa số niêm yết 2020-2021).
DEFAULT_UNIVERSE = [
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT","SOLUSDT","DOGEUSDT","DOTUSDT",
    "AVAXUSDT","LINKUSDT","LTCUSDT","TRXUSDT","ATOMUSDT","XLMUSDT","ETCUSDT","FILUSDT",
    "NEARUSDT","UNIUSDT","AAVEUSDT","ALGOUSDT","VETUSDT","ICPUSDT","SANDUSDT","EOSUSDT",
]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--days", type=float, default=2200)
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    client = BinanceFuturesREST.public_mainnet()
    ok, fail = [], []
    for i, sym in enumerate(a.symbols, 1):
        try:
            bars = download_klines(sym, a.interval, days=a.days, client=client)
            save_klines(bars, sym, a.interval)
            fr = download_funding_rates(sym, days=a.days, client=client)
            pd.DataFrame({"funding_time": list(fr), "rate": list(fr.values())}).to_parquet(
                f"data/binance/{sym}_funding.parquet", index=False)
            span = (bars.timestamp_ms.iloc[-1]-bars.timestamp_ms.iloc[0])/86400000
            print(f"[{i:2}/{len(a.symbols)}] {sym:<12} {len(bars):>6} nến  {span:>6.0f} ngày  {len(fr):>5} mốc funding")
            ok.append(sym)
        except Exception as exc:
            print(f"[{i:2}/{len(a.symbols)}] {sym:<12} LỖI: {type(exc).__name__}")
            fail.append(sym)
        time.sleep(0.15)
    print(f"\n✅ {len(ok)} cặp tải xong" + (f" | ❌ {len(fail)}: {fail}" if fail else ""))
    return 0

if __name__ == "__main__":
    sys.exit(main())
