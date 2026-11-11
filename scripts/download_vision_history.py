#!/usr/bin/env python3
"""Tải nến 1h và funding của MỌI perp USDT từng niêm yết, gồm cả cặp đã huỷ.

    python scripts/download_vision_history.py --workers 16

Ghi `data/binance/<SYMBOL>_1h.parquet`, `<SYMBOL>_funding.parquet` (cùng định dạng với
`download_wide_universe.py`) và `artifacts/universe_all_listed.json`. Chạy lại an toàn:
cặp đã có file sẽ được bỏ qua.
"""

import argparse
import concurrent.futures
import json
import pathlib
import sys

sys.path.insert(0, "src")

from aegis.data.ingestion.binance_vision import (
    download_symbol,
    list_symbols,
)  # noqa: E402

ROOT = pathlib.Path("data/binance")


def fetch_one(symbol: str, interval: str) -> tuple[str, int, int, int]:
    bars_path = ROOT / f"{symbol}_{interval}.parquet"
    funding_path = ROOT / f"{symbol}_funding.parquet"
    if bars_path.is_file() and funding_path.is_file():
        import pandas as pd

        bars = pd.read_parquet(bars_path, columns=["timestamp_ms"])
        return (
            symbol,
            len(bars),
            int(bars.timestamp_ms.iloc[0]),
            int(bars.timestamp_ms.iloc[-1]),
        )
    bars, funding = download_symbol(symbol, interval)
    if not len(bars):
        return symbol, 0, 0, 0
    if not len(funding):
        import pandas as pd

        funding = pd.DataFrame(
            {
                "funding_time": pd.Series(dtype="int64"),
                "rate": pd.Series(dtype="float64"),
            }
        )
    funding.to_parquet(funding_path, index=False)
    bars.to_parquet(
        bars_path, index=False
    )  # ghi sau cùng: có file nến nghĩa là đã xong
    return (
        symbol,
        len(bars),
        int(bars.timestamp_ms.iloc[0]),
        int(bars.timestamp_ms.iloc[-1]),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=0, help="chỉ tải N cặp đầu (thử nhanh)"
    )
    parser.add_argument("--out", default="artifacts/universe_all_listed.json")
    args = parser.parse_args(argv)

    ROOT.mkdir(parents=True, exist_ok=True)
    symbols = list_symbols()
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"{len(symbols)} perp USDT từng niêm yết", flush=True)

    done, failed = {}, []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(fetch_one, symbol, args.interval): symbol for symbol in symbols
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            symbol = futures[future]
            try:
                _, bars, first, last = future.result()
                if bars:
                    done[symbol] = {"bars": bars, "first_ms": first, "last_ms": last}
            except Exception as error:
                failed.append(f"{symbol}: {type(error).__name__} {error}")
            if count % 25 == 0 or count == len(symbols):
                print(f"  {count}/{len(symbols)} xong, {len(failed)} lỗi", flush=True)

    json.dump(sorted(done), open(args.out, "w"), indent=1)
    json.dump(done, open(ROOT / "_vision_manifest.json", "w"), indent=1)
    print(f"XONG: {len(done)} cặp có dữ liệu -> {args.out}", flush=True)
    for line in failed[:20]:
        print("LỖI", line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
