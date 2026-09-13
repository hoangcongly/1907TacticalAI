#!/usr/bin/env python3
"""
Tải nến SPOT cho mọi cặp perp có thị trường spot tương ứng — nền tảng cho basis trade.

Basis trade cần CẢ HAI chân: short perp + long spot cùng tài sản. Số cặp có đủ hai
chân chính là độ rộng thực sự của lớp chiến lược này — và độ rộng quyết định chất
lượng chọn funding. 20 cặp là quá hẹp; toàn bộ universe mới cho phép chọn lọc.
"""
import argparse, json, pathlib, sys, time
import pandas as pd, requests

SPOT = "https://api.binance.com"
ROOT = pathlib.Path("data/binance_spot")


def spot_symbols() -> set:
    info = requests.get(f"{SPOT}/api/v3/exchangeInfo", timeout=30).json()
    return {s["symbol"] for s in info["symbols"]
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}


def fetch(symbol: str, days: float, interval: str = "1h") -> pd.DataFrame:
    rows, end = [], int(time.time() * 1000)
    cur = end - int(days * 86_400_000)
    while cur < end:
        r = requests.get(f"{SPOT}/api/v3/klines",
                         params={"symbol": symbol, "interval": interval,
                                 "startTime": cur, "limit": 1000}, timeout=25)
        if r.status_code == 429:
            time.sleep(10); continue
        if r.status_code != 200:
            break
        d = r.json()
        if not d:
            break
        rows += d
        if len(d) < 1000:
            break
        cur = int(d[-1][0]) + 3_600_000
        time.sleep(0.06)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    df = df.astype({"timestamp_ms": "int64"})
    for c in df.columns[1:]:
        df[c] = df[c].astype("float64")
    return df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="artifacts/universe_wide.json")
    ap.add_argument("--days", type=float, default=1400)
    ap.add_argument("--min-bars", type=int, default=3000)
    ap.add_argument("--out", default="artifacts/basis_universe.json")
    a = ap.parse_args(argv)

    ROOT.mkdir(parents=True, exist_ok=True)
    perps = json.load(open(a.universe))
    have_spot = spot_symbols()
    cand = [s for s in perps if s in have_spot]
    print(f"{len(perps)} perp | {len(cand)} có thị trường spot", flush=True)

    ok = []
    for i, s in enumerate(cand, 1):
        p = ROOT / f"{s}_1h.parquet"
        if p.is_file():
            if len(pd.read_parquet(p)) >= a.min_bars:
                ok.append(s)
            continue
        try:
            df = fetch(s, a.days)
            if len(df) < a.min_bars:
                print(f"[{i:3}/{len(cand)}] {s:<16} chỉ {len(df)} nến — bỏ", flush=True)
                continue
            df.to_parquet(p, index=False)
            ok.append(s)
            print(f"[{i:3}/{len(cand)}] {s:<16} {len(df)} nến spot", flush=True)
        except Exception as exc:
            print(f"[{i:3}/{len(cand)}] {s:<16} LỖI {type(exc).__name__}", flush=True)
        time.sleep(0.05)

    json.dump(sorted(ok), open(a.out, "w"), indent=1)
    print(f"\nXONG: {len(ok)} cặp có ĐỦ CẢ perp lẫn spot -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
