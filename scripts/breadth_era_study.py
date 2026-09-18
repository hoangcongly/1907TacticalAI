#!/usr/bin/env python3
"""
ĐỘ RỘNG DANH MỤC, CHẤM RIÊNG TRÊN KỶ NGUYÊN ĐỘ RỘNG CAO.

VÌ SAO SCRIPT NÀY TỒN TẠI: `n_positions = 12` và việc ĐÓNG các hướng cần độ rộng
(hồi quy mặt cắt ngang, LambdaRank, danh mục rộng) đều dựa trên một câu: "universe
chỉ có ~41 tài sản/kỳ". Con số 41 đó là TRUNG VỊ CỦA TOÀN LỊCH SỬ 2020-2026, bị kéo
xuống bởi những năm đầu khi perp crypto còn rất ít cặp:

    2020:  20 cặp/kỳ      2024:  78 cặp/kỳ
    2022:  41 cặp/kỳ      2025: 111 cặp/kỳ
    2023:  51 cặp/kỳ      2026: 127 cặp/kỳ   <- HIỆN TẠI

Hệ thống sẽ giao dịch trong MÔI TRƯỜNG 127 CẶP, không phải môi trường 41 cặp. Chấm
một cấu hình cần độ rộng bằng trung bình toàn lịch sử là bắt nó gánh những năm mà nó
KHÔNG THỂ chạy được — rồi kết luận nó không chạy được.

Định luật cơ bản của quản lý chủ động: IR = IC x sqrt(breadth). Từ 41 lên 127 cặp là
hệ số sqrt(127/41) = 1,76x trên IR, nếu IC giữ nguyên.

THỨ TỰ NHÂN QUẢ (bài học F35): `run_v3(mask=...)` chạy tầng gộp thích ứng trên TOÀN
lưới rồi mới CẮT lợi suất về kỷ nguyên cần chấm. Cắt dữ liệu TRƯỚC sẽ khởi động lại
tầng gộp ở đầu kỷ nguyên — một handicap mà live không bao giờ gặp.

    python scripts/breadth_era_study.py
    python scripts/breadth_era_study.py --min-breadth 100 --grid 12,20,30,40
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)

OUT_CSV = "artifacts/breadth_era_study.csv"


def _stats(r, ppy):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) < 20 or r.std(ddof=1) < 1e-15:
        return dict(sharpe=np.nan, ann=np.nan, vol=np.nan, kelly=np.nan, n=len(r))
    ann = float(r.mean() * ppy)
    vol = float(r.std(ddof=1) * np.sqrt(ppy))
    return dict(sharpe=ann / vol, ann=ann, vol=vol,
                kelly=ann / vol**2 if vol > 0 else np.nan, n=len(r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="8,12,16,20,24,30,40,50")
    ap.add_argument("--eras", default="0,60,100",
                    help="ngưỡng số cặp/kỳ tối thiểu định nghĩa mỗi kỷ nguyên")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args(argv)
    grid = [int(x) for x in a.grid.split(",")]
    eras = [int(x) for x in a.eras.split(",")]

    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    sig = combined_signal(data, V3)
    ppy = V3.periods_per_year
    idx = data.close.index
    breadth = data.close.notna().sum(axis=1)

    # Chạy MỘT lần cho mỗi n, trên toàn lưới. Các kỷ nguyên chỉ là mặt nạ chấm điểm.
    runs = {}
    for n in grid:
        runs[n] = run_v3(data, V3, n_positions=n, sig=sig).returns.dropna()
        print(f"  n={n:>3} xong ({len(runs[n])} kỳ)", flush=True)

    rows = []
    for thr in eras:
        ok = breadth >= thr
        first = idx[ok][0] if ok.any() else None
        if first is None:
            continue
        lab = (f"TOÀN BỘ" if thr == 0
               else f">= {thr} cặp/kỳ (từ {pd.to_datetime(first, unit='ms'):%Y-%m})")
        print("\n" + "=" * 100)
        print(f"KỶ NGUYÊN: {lab}")
        print("=" * 100)
        print(f"{'n':>4}{'kỳ':>7}{'Sharpe':>9}{'ann':>8}{'vol':>8}{'Kelly':>8}"
              f"{'trung vị fold':>15}{'%dương':>8}{'tệ nhất':>9}{'VND/tuần':>11}")
        print("-" * 100)
        for n in grid:
            r = runs[n]
            keep = pd.Series(r.index).map(lambda t: bool(breadth.get(t, 0) >= thr)).values
            rr = r[keep]
            st = _stats(rr, ppy)
            if not np.isfinite(st["sharpe"]):
                continue
            b = np.linspace(0, len(rr), a.folds + 1).astype(int)
            fs = []
            for i in range(a.folds):
                s = _stats(rr.iloc[b[i]:b[i + 1]], ppy)["sharpe"]
                fs.append(s)
            med = float(np.nanmedian(fs))
            pos = float(np.mean([x > 0 for x in fs if np.isfinite(x)])) if any(
                np.isfinite(fs)) else np.nan
            worst = float(np.nanmin(fs))
            lev = st["kelly"] / 2.0
            g = st["ann"] * lev - (st["vol"] * lev) ** 2 / 2.0
            vnd = g / 52 * 1_000_000
            mark = "  <- đang chạy" if n == 12 else ""
            print(f"{n:>4}{st['n']:>7}{st['sharpe']:>9.2f}{st['ann']*100:>7.0f}%"
                  f"{st['vol']*100:>7.0f}%{st['kelly']:>7.2f}x{med:>15.2f}"
                  f"{pos*100:>7.0f}%{worst:>9.2f}{vnd:>11,.0f}{mark}")
            rows.append(dict(era_min_breadth=thr, era_label=lab, n_positions=n,
                             periods=st["n"], sharpe=st["sharpe"], ann=st["ann"],
                             vol=st["vol"], kelly=st["kelly"], median_fold=med,
                             frac_pos=pos, worst_fold=worst, vnd_week=vnd))

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("\n" + "=" * 100)
    print(f"-> {a.out}")
    base = df[(df.era_min_breadth == 0) & (df.n_positions == 12)]
    if len(base):
        b0 = base.iloc[0]
        print(f"\nMỐC SO SÁNH (toàn bộ, n=12): Sharpe {b0.sharpe:.2f}, "
              f"{b0.vnd_week:,.0f} VND/tuần")
    hi = df[df.era_min_breadth == max(eras)]
    if len(hi):
        best = hi.sort_values("vnd_week").iloc[-1]
        cur = hi[hi.n_positions == 12]
        if len(cur):
            c = cur.iloc[0]
            print(f"KỶ NGUYÊN RỘNG, n=12    : Sharpe {c.sharpe:.2f}, {c.vnd_week:,.0f} VND/tuần")
        print(f"KỶ NGUYÊN RỘNG, TỐT NHẤT: n={best.n_positions}, Sharpe {best.sharpe:.2f}, "
              f"{best.vnd_week:,.0f} VND/tuần")
    print("\n⚠️  Kỷ nguyên độ rộng cao ngắn hơn -> ít kỳ hơn -> sai số lớn hơn. Đây là")
    print("    bằng chứng ĐỊNH HƯỚNG, phải xác nhận bằng giao dịch giấy tiến về phía trước.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
