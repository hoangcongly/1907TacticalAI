#!/usr/bin/env python3
"""
Tải DỮ LIỆU VỊ THẾ lịch sử từ kho dump Binance — nguồn tín hiệu mà thư viện đang thiếu.

VÌ SAO ĐÁNG TẢI: 26 tín hiệu hiện có đều dựng từ GIÁ, KHỐI LƯỢNG và FUNDING. Không cái
nào nhìn thấy VỊ THẾ — ai đang cầm gì, đòn bẩy đang tích tụ ở đâu. Với perpetual futures,
vị thế là biến trạng thái quan trọng nhất: các đợt squeeze và thanh lý dây chuyền không
bắt nguồn từ giá mà từ việc quá nhiều người đứng cùng một bên.

Endpoint `/futures/data/openInterestHist` chỉ cho 30 ngày — không backtest được. Nhưng
kho dump `data.binance.vision/data/futures/um/daily/metrics` có ĐẦY ĐỦ lịch sử từ 2021,
5 phút một điểm, miễn phí, không cần khoá. Sáu trường:

    sum_open_interest                  OI theo đơn vị cơ sở
    sum_open_interest_value            OI theo USD
    count_toptrader_long_short_ratio   tỷ lệ long/short theo SỐ TÀI KHOẢN của top trader
    sum_toptrader_long_short_ratio     tỷ lệ long/short theo VỊ THẾ của top trader
    count_long_short_ratio             tỷ lệ long/short của toàn bộ tài khoản
    sum_taker_long_short_vol_ratio     tỷ lệ khối lượng taker mua/bán

Hai cột `toptrader` là thứ quý nhất: chúng tách vị thế của nhóm vốn lớn khỏi đám đông,
và chênh lệch giữa `count_` với `sum_` cho biết bên nào đang cầm vị thế LỚN hơn.

THIẾT KẾ:
* Tổng hợp về khung mục tiêu NGAY khi tải, không lưu 5 phút — 5m cho 127 cặp x 5 năm
  là hàng chục GB và ta không dùng tới.
* Checkpoint theo CẶP: cặp nào xong thì ghi parquet luôn. Chạy lại chỉ tải phần thiếu.
* Ngày thiếu (cặp chưa niêm yết, hoặc Binance không có file) được bỏ qua im lặng —
  404 ở đây là thông tin bình thường, không phải lỗi.

    python scripts/download_metrics.py --symbols 40 --days 700    # thử nghiệm
    python scripts/download_metrics.py                            # toàn bộ
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import io
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import List, Optional

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
OUT_ROOT = pathlib.Path("data/binance_metrics")

NUMERIC = ["sum_open_interest", "sum_open_interest_value",
           "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
           "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def _fetch_day(symbol: str, day: str, timeout: float) -> Optional[pd.DataFrame]:
    """Một file ngày. Trả None nếu không có (404) — không có KHÔNG phải lỗi."""
    url = f"{BASE}/{symbol}/{symbol}-metrics-{day}.zip"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]))
    if df.empty:
        return None
    # ⚠️ ĐỔI ĐƠN VỊ TƯỜNG MINH, KHÔNG chia tay.
    #
    # pandas 3.0 trả `datetime64[us]` (micro giây) cho `to_datetime`, còn pandas 2.x
    # trả `datetime64[ns]`. Viết `.astype("int64") // 10**6` là ngầm giả định nano
    # giây — trên pandas 3 nó cho ra GIÂY, và mọi mốc thời gian rơi về năm 1970 mà
    # không có lỗi nào được ném ra. Đã dính: 400 ngày dữ liệu gộp lại thành "4 nến,
    # 1970-01".
    #
    # `.astype("datetime64[ms]")` nói thẳng đơn vị mình muốn, nên nó đúng ở mọi phiên
    # bản pandas và sẽ vẫn đúng khi pandas đổi mặc định lần nữa.
    df["timestamp_ms"] = pd.to_datetime(df["create_time"]).astype("datetime64[ms]").astype("int64")
    return df[["timestamp_ms"] + [c for c in NUMERIC if c in df.columns]]


def _aggregate(df: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    """
    Tổng hợp 5 phút lên khung mục tiêu.

    Mọi trường ở đây là MỨC TỒN (stock), không phải LƯU LƯỢNG (flow): open interest tại
    một thời điểm là số hợp đồng đang mở, không phải số hợp đồng đã mở trong kỳ. Nên
    quy tắc tổng hợp là LẤY GIÁ TRỊ CUỐI KỲ, không phải tổng. Lấy tổng sẽ biến một mức
    tồn ổn định thành một chuỗi tăng tuyến tính — sai hoàn toàn về bản chất.
    """
    d = df.copy()
    d["bucket"] = (d["timestamp_ms"] // interval_ms) * interval_ms
    g = d.groupby("bucket", sort=True)
    out = pd.DataFrame({"timestamp_ms": g["timestamp_ms"].first().index.to_numpy()})
    for c in NUMERIC:
        if c in d.columns:
            out[c] = g[c].last().to_numpy()
    return out


def _download_symbol(symbol: str, days: List[str], interval_ms: int,
                     timeout: float, workers: int) -> Optional[pd.DataFrame]:
    frames = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_day, symbol, d, timeout): d for d in days}
        for f in cf.as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = None
            if r is not None and not r.empty:
                frames.append(r)
    if not frames:
        return None
    df = pd.concat(frames).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
    return _aggregate(df, interval_ms)


def main(argv=None) -> int:
    from aegis.data.panel_v2 import INTERVAL_MS

    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="artifacts/universe_wide.json")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--days", type=int, default=2100, help="số ngày lùi về quá khứ")
    ap.add_argument("--symbols", type=int, default=0, help="giới hạn số cặp (0 = tất cả)")
    ap.add_argument("--workers", type=int, default=24, help="số kết nối song song mỗi cặp")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--force", action="store_true", help="tải lại cả cặp đã có")
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args(argv)

    interval_ms = INTERVAL_MS[a.interval]
    out_root = pathlib.Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    syms = json.load(open(a.universe))
    if a.symbols:
        syms = syms[:a.symbols]

    end = dt.date.today()
    days = [(end - dt.timedelta(days=i)).isoformat() for i in range(1, a.days + 1)]

    print(f"tải metrics: {len(syms)} cặp x {len(days)} ngày -> khung {a.interval}")
    print(f"đích: {out_root}  |  {a.workers} kết nối song song\n", flush=True)

    t0 = time.time()
    done = skipped = empty = 0
    for i, sym in enumerate(syms, 1):
        path = out_root / f"{sym}_{a.interval}.parquet"
        if path.is_file() and not a.force:
            skipped += 1
            continue
        df = _download_symbol(sym, days, interval_ms, a.timeout, a.workers)
        if df is None or df.empty:
            empty += 1
            print(f"  [{i:>3}/{len(syms)}] {sym:<16} không có dữ liệu", flush=True)
            continue
        df.to_parquet(path, index=False)
        done += 1
        el = time.time() - t0
        eta = el / max(done, 1) * (len(syms) - i)
        print(f"  [{i:>3}/{len(syms)}] {sym:<16} {len(df):>6} nến "
              f"{pd.to_datetime(df.timestamp_ms.min(), unit='ms'):%Y-%m} -> "
              f"{pd.to_datetime(df.timestamp_ms.max(), unit='ms'):%Y-%m}"
              f"   ({el/60:.1f}p trôi qua, còn ~{eta/60:.0f}p)", flush=True)

    manifest = {"interval": a.interval, "days_requested": a.days,
                "n_symbols_written": done, "n_skipped": skipped, "n_empty": empty,
                "fields": NUMERIC, "source": BASE,
                "generated_ms": int(time.time() * 1000)}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nxong: {done} cặp mới, {skipped} bỏ qua (đã có), {empty} không có dữ liệu")
    print(f"-> {out_root}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
