#!/usr/bin/env python3
"""Đo spread sổ lệnh THẬT cho toàn bộ perp USDⓈ-M — một lệnh gọi, lưu ra artifacts."""
import json, sys
from aegis.data.ingestion.binance_rest import BinanceFuturesREST

def main() -> int:
    rows = BinanceFuturesREST.public_mainnet()._request(
        "GET", "/fapi/v1/ticker/bookTicker", signed=False)
    out = {}
    for r in rows:
        try:
            bid, ask = float(r["bidPrice"]), float(r["askPrice"])
        except (KeyError, ValueError, TypeError):
            continue
        if bid <= 0 or ask <= bid:
            continue
        out[r["symbol"]] = (ask - bid) / (0.5 * (bid + ask)) * 1e4
    json.dump(out, open("artifacts/live_spreads_bps.json", "w"), indent=1)
    vals = sorted(out.values())
    print(f"{len(out)} cặp | spread toàn phần bp: trung vị {vals[len(vals)//2]:.2f} "
          f"| p90 {vals[int(len(vals)*0.9)]:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
