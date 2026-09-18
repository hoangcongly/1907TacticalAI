#!/usr/bin/env python3
"""
CHU KỲ TÁI CÂN BẰNG nên là bao lâu — và nó đổi đòn bẩy khả thi thế nào.

CÂU HỎI: `period_hours = 72` đến từ đâu? Nó được chốt cùng v3 và chưa ai quét riêng
nó TRÊN v3. `lab_grid.json` có quét chu kỳ nhưng trên `StrategyV2` (top_frac, 4 tín
hiệu, beta_neutral=True) và CHỈ trên train — kết quả ở đó (96h -> Sharpe 1,92) là
manh mối, không phải kết luận chuyển thẳng sang được.

VÌ SAO CÂU HỎI NÀY ĐÁNG TIỀN: chi phí tỉ lệ thuận với TẦN SUẤT. Lưới cũ cho thấy chu
kỳ 6h ăn 27,3%/năm tiền phí và kéo Sharpe xuống 0,03 — tức giao dịch nhanh hơn không
phải "nhiều cơ hội hơn", nó là "trả phí nhiều hơn cho cùng một tín hiệu". Chiều ngược
lại chưa được đo trên v3: chậm hơn 72h thì sao?

PHƯƠNG PHÁP — giống `breadth_study.py`, và vì cùng lý do: không chấm theo train, cũng
không chấm theo holdout (holdout đã dùng 2 lần cho v3, chạm lần ba là tự chấm bài
mình). Cắt TOÀN dòng thời gian thành K giai đoạn con liên tiếp rồi chấm theo ĐỘ ỔN
ĐỊNH: trung vị Sharpe, tỷ lệ giai đoạn dương, giai đoạn tệ nhất.

    python scripts/rebalance_period_study.py
    python scripts/rebalance_period_study.py --grid 12,18,24,30 --folds 8
"""
import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aegis.research.strategy_v3 import V3, V3_VINTAGE_MS, load_v3_data, run_v3

OUT_CSV = "artifacts/rebalance_period_study.csv"


def _sharpe(r, ppy):
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    if len(r) < 10 or r.std(ddof=1) < 1e-15:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="6,9,12,18,24,30,36",
                    help="số nến 4h mỗi kỳ (18 = 72h, cấu hình đang chạy)")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args(argv)
    grid = [int(x) for x in a.grid.split(",")]

    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    bar_h = 4.0

    series, ppys = {}, {}
    for rb in grid:
        cfg = replace(V3, rebalance_every=rb)
        # `periods_per_year` phải đi theo chu kỳ, nếu không Sharpe bị quy đổi sai.
        ppy = 365.0 * 24.0 / (bar_h * rb)
        out = run_v3(data, cfg)          # KHÔNG dùng lại `sig`: lưới mốc đổi theo rb
        series[rb] = out.returns.dropna()
        ppys[rb] = ppy
        print(f"  {bar_h*rb:>5.0f}h ({rb:>2} nến) xong — {len(series[rb])} kỳ", flush=True)

    print("\n" + "=" * 104)
    print(f"SHARPE QUA {a.folds} GIAI ĐOẠN CON CỦA TOÀN DÒNG THỜI GIAN")
    print("=" * 104)
    print(f"{'chu kỳ':>8}{'kỳ/năm':>8}", end="")
    print(f"{'trung vị':>11}{'%dương':>8}{'tệ nhất':>9}{'Sharpe toàn':>13}"
          f"{'ann':>8}{'vol':>8}{'Kelly':>8}{'VND/tuần':>11}")
    print("-" * 104)

    rows = []
    for rb in grid:
        r, ppy = series[rb], ppys[rb]
        b = np.linspace(0, len(r), a.folds + 1).astype(int)
        fs = [_sharpe(r.iloc[b[i]:b[i + 1]], ppy) for i in range(a.folds)]
        med = float(np.nanmedian(fs))
        pos = float(np.mean([s > 0 for s in fs if np.isfinite(s)]))
        worst = float(np.nanmin(fs))
        full = _sharpe(r, ppy)
        vol = float(r.std(ddof=1) * np.sqrt(ppy))
        ann = float(r.mean() * ppy)
        kelly = ann / vol**2 if vol > 0 else np.nan
        # Nửa Kelly là mức thực hành; tăng trưởng tuần ở mức đó.
        lev = kelly / 2.0
        g_week = (ann * lev - (vol * lev) ** 2 / 2.0) / 52.0
        vnd = g_week * 1_000_000
        mark = "  <- đang chạy" if rb == V3.rebalance_every else ""
        print(f"{bar_h*rb:>7.0f}h{ppy:>8.0f}{med:>11.2f}{pos*100:>7.0f}%{worst:>9.2f}"
              f"{full:>13.2f}{ann*100:>7.0f}%{vol*100:>7.0f}%{kelly:>7.2f}x{vnd:>11,.0f}{mark}")
        rows.append({"rebalance_every": rb, "period_hours": bar_h * rb,
                     "periods_per_year": ppy, "sharpe_median_fold": med,
                     "frac_folds_positive": pos, "sharpe_worst_fold": worst,
                     "sharpe_full": full, "ann_return": ann, "ann_vol": vol,
                     "kelly_lev": kelly, "vnd_per_week_half_kelly": vnd,
                     **{f"fold_{i}": fs[i] for i in range(a.folds)}})

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("-" * 104)
    print(f"-> {a.out}")

    cur = df[df.rebalance_every == V3.rebalance_every]
    # Chọn theo ĐỘ ỔN ĐỊNH: phải 100% giai đoạn dương, rồi mới xét trung vị.
    stable = df[df.frac_folds_positive >= 1.0]
    best = (stable if len(stable) else df).sort_values("sharpe_median_fold").iloc[-1]
    if len(cur):
        c = cur.iloc[0]
        print(f"\nĐANG CHẠY {c.period_hours:.0f}h: trung vị {c.sharpe_median_fold:.2f}, "
              f"{c.frac_folds_positive*100:.0f}% dương, tệ nhất {c.sharpe_worst_fold:.2f}, "
              f"{c.vnd_per_week_half_kelly:,.0f} VND/tuần")
    print(f"TỐT NHẤT  {best.period_hours:.0f}h: trung vị {best.sharpe_median_fold:.2f}, "
          f"{best.frac_folds_positive*100:.0f}% dương, tệ nhất {best.sharpe_worst_fold:.2f}, "
          f"{best.vnd_per_week_half_kelly:,.0f} VND/tuần")
    print("\n⚠️  Đây là bằng chứng ĐỘ ỔN ĐỊNH trên toàn dòng thời gian, KHÔNG phải kiểm")
    print("    định sạch. Holdout đã dùng 2 lần cho v3; đổi tham số rồi chấm lại trên")
    print("    holdout là lần thứ ba, tức tự chấm bài mình. Muốn đổi thật thì phải đi")
    print("    qua giao dịch giấy tiến về phía trước.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
