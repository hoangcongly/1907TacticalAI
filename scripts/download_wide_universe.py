#!/usr/bin/env python3
"""
Tải universe RỘNG khung 1h — nền tảng cho nâng cấp tần suất + độ rộng (breadth).

VÌ SAO: định luật cơ bản của quản lý chủ động IR = IC x sqrt(breadth).
- breadth tăng theo SỐ TÀI SẢN được xếp hạng (không phải số vị thế nắm giữ)
- breadth cũng tăng theo SỐ LẦN ra quyết định mỗi năm (tần suất tái cân bằng)
Hệ thống cũ: 59 cặp, quyết định 1 lần/ngày. Đây là hai trần bị tự đặt ra.

Tải 1h làm NGUỒN DUY NHẤT; 4h/8h/12h sinh ra bằng resample nhân quả để mọi khung
đều nhất quán về OFI và quote_volume.
"""
import argparse, json, logging, pathlib, sys, time
import pandas as pd

from aegis.data.ingestion.binance_history import download_klines, download_funding_rates, save_klines
from aegis.data.ingestion.binance_rest import BinanceFuturesREST

ROOT = pathlib.Path("data/binance")


def liquid_universe(client: BinanceFuturesREST, top_n: int, min_qv_usd: float) -> list:
    """Xếp hạng perp USDT theo khối lượng quote 24h; loại cặp đã ngừng giao dịch."""
    info = client.exchange_info()
    live = {
        s["symbol"] for s in info["symbols"]
        if s.get("contractType") == "PERPETUAL"
        and s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
    }
    tick = client._request("GET", "/fapi/v1/ticker/24hr", signed=False)
    rows = [(t["symbol"], float(t["quoteVolume"])) for t in tick if t["symbol"] in live]
    rows.sort(key=lambda x: -x[1])
    keep = [s for s, qv in rows if qv >= min_qv_usd][:top_n]
    return keep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=float, default=2400)
    ap.add_argument("--top", type=int, default=160)
    ap.add_argument("--min-qv", type=float, default=20_000_000.0, help="Khối lượng quote 24h tối thiểu (USD)")
    ap.add_argument("--out", default="artifacts/universe_wide.json")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    ROOT.mkdir(parents=True, exist_ok=True)
    client = BinanceFuturesREST.public_mainnet()

    syms = liquid_universe(client, a.top, a.min_qv)
    print(f"Universe: {len(syms)} cặp perp USDT thanh khoản >= ${a.min_qv/1e6:.0f}M/24h", flush=True)
    json.dump(syms, open(a.out, "w"), indent=1)

    ok, fail = [], []
    for i, sym in enumerate(syms, 1):
        kpath = ROOT / f"{sym}_{a.interval}.parquet"
        fpath = ROOT / f"{sym}_funding.parquet"
        try:
            if not kpath.is_file():
                bars = download_klines(sym, a.interval, days=a.days, client=client)
                save_klines(bars, sym, a.interval, root=str(ROOT))
            else:
                bars = pd.read_parquet(kpath)
            if not fpath.is_file():
                fr = download_funding_rates(sym, days=a.days, client=client)
                pd.DataFrame({"funding_time": list(fr), "rate": list(fr.values())}).to_parquet(fpath, index=False)
            span = (bars.timestamp_ms.iloc[-1] - bars.timestamp_ms.iloc[0]) / 86_400_000
            print(f"[{i:3}/{len(syms)}] {sym:<16} {len(bars):>6} nến {span:>6.0f} ngày", flush=True)
            ok.append(sym)
        except Exception as exc:
            print(f"[{i:3}/{len(syms)}] {sym:<16} LỖI {type(exc).__name__}: {exc}", flush=True)
            fail.append(sym)
        time.sleep(0.08)

    print(f"\nXONG: {len(ok)} cặp" + (f" | lỗi {len(fail)}: {fail}" if fail else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
