#!/usr/bin/env python3
"""
ĐỘ RỘNG DANH MỤC có mua được Sharpe không — và nó tương tác thế nào với vốn nhỏ.

CÂU HỎI: `n_positions = 12` đến từ đâu? Không phải từ nghiên cứu. Nó đến từ số học
vốn: $38 x đòn bẩy 2x / min notional $5 = 15 vị thế, làm tròn xuống cho an toàn. Mọi
so sánh trong repo đều CỐ ĐỊNH con số đó (đúng, khi đang so tín hiệu), nên chưa ai đo
bản thân nó.

VÌ SAO CÂU HỎI NÀY KHÔNG TẦM THƯỜNG VỚI TÀI KHOẢN NHỎ: đòn bẩy và độ rộng bị KHOÁ VÀO
NHAU. Muốn `n` vị thế thì gross phải >= n * $5 / $38. Nên chọn n = 40 KHÔNG phải chọn
"đa dạng hơn" — nó là chọn "đa dạng hơn VÀ buộc phải chạy >= 5,26x". Hai thứ đó đi
cùng nhau và phải được đánh giá cùng nhau.

PHƯƠNG PHÁP — vì sao không dùng train hay holdout:
Đo thử trước cho thấy train chọn n = 12 còn holdout chọn n = 40. Khi hai tập bất đồng
thì không tập nào đáng tin, và chọn theo holdout là tự chấm bài mình (holdout đã dùng
2 lần cho v3). Nên script này KHÔNG chọn theo tập nào cả: nó cắt TOÀN dòng thời gian
thành K giai đoạn con liên tiếp và chấm theo ĐỘ ỔN ĐỊNH — trung vị Sharpe qua các giai
đoạn, tỷ lệ giai đoạn dương, và giai đoạn tệ nhất. Một cấu hình thắng nhờ đúng một giai
đoạn may mắn sẽ lộ ra ngay.

    python scripts/breadth_study.py
    python scripts/breadth_study.py --folds 10 --grid 8,12,16,20,24,30,40
"""
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)

OUT_CSV = "artifacts/breadth_study.csv"


def _sharpe(r, ppy):
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    if len(r) < 10 or r.std(ddof=1) < 1e-15:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="8,10,12,14,16,20,24,30,40")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args(argv)
    grid = [int(x) for x in a.grid.split(",")]

    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    sig = combined_signal(data, V3)        # đắt, tính MỘT lần cho cả lưới
    ppy = V3.periods_per_year

    series = {}
    for n in grid:
        series[n] = run_v3(data, V3, n_positions=n, sig=sig).returns.dropna()
        print(f"  n={n:>3} xong ({len(series[n])} kỳ)", flush=True)

    idx = series[grid[0]].index
    bounds = np.linspace(0, len(idx), a.folds + 1).astype(int)
    labels = [f"{pd.to_datetime(idx[bounds[i]], unit='ms'):%y/%m}" for i in range(a.folds)]

    print("\n" + "=" * 104)
    print(f"SHARPE QUA {a.folds} GIAI ĐOẠN CON CỦA TOÀN DÒNG THỜI GIAN "
          f"({pd.to_datetime(idx[0], unit='ms'):%Y-%m} -> {pd.to_datetime(idx[-1], unit='ms'):%Y-%m})")
    print("=" * 104)
    print(f"{'n':>4}{'lev tối thiểu':>14}", end="")
    for L in labels:
        print(f"{L:>8}", end="")
    print(f"{'trung vị':>10}{'%dương':>8}{'tệ nhất':>9}")
    print("-" * 104)

    rows = []
    for n in grid:
        r = series[n]
        fold_sh = [_sharpe(r.iloc[bounds[i]:bounds[i + 1]], ppy) for i in range(a.folds)]
        med = float(np.nanmedian(fold_sh))
        pos = float(np.mean([s > 0 for s in fold_sh if np.isfinite(s)]))
        worst = float(np.nanmin(fold_sh))
        lev_min = n * V3.min_notional_usd / V3.capital_usd

        print(f"{n:>4}{lev_min:>13.2f}x", end="")
        for s in fold_sh:
            print(f"{s:>8.2f}", end="")
        print(f"{med:>10.2f}{pos*100:>7.0f}%{worst:>9.2f}")

        full_sh = _sharpe(r, ppy)
        vol = float(r.std(ddof=1) * np.sqrt(ppy))
        ann = float(r.mean() * ppy)
        rows.append({"n_positions": n, "lev_min": lev_min, "sharpe_full": full_sh,
                     "ann_return": ann, "ann_vol": vol, "sharpe_median_fold": med,
                     "frac_folds_positive": pos, "sharpe_worst_fold": worst,
                     **{f"fold_{i}": fold_sh[i] for i in range(a.folds)}})

    df = pd.DataFrame(rows)

    # ---- Đòn bẩy KHẢ THI và tốc độ tăng trưởng thật sự đạt được ----------
    print("\n" + "=" * 104)
    print("ĐÒN BẨY KHẢ THI — độ rộng và đòn bẩy bị khoá vào nhau khi vốn nhỏ")
    print("=" * 104)
    print("Gross phải >= n*$5/$38 để mọi vị thế vượt min notional. Nhưng gross cũng không")
    print("nên vượt Kelly (S/sigma), vì quá đó thì tăng đòn bẩy LÀM GIẢM tăng trưởng.")
    print(f"\n{'n':>4}{'lev tối thiểu':>14}{'lev Kelly':>11}{'khả thi?':>10}"
          f"{'lev dùng':>10}{'tăng trưởng/tuần':>18}{'VND/tuần':>11}")
    print("-" * 104)
    for _, r in df.iterrows():
        S, vol = r["sharpe_median_fold"], r["ann_vol"]
        kelly = S / vol if vol > 1e-9 else 0.0
        feasible = kelly >= r["lev_min"]
        # Dùng nửa Kelly (chuẩn thực hành) nhưng không dưới sàn đặt lệnh.
        lev_use = max(r["lev_min"], 0.5 * kelly) if feasible else r["lev_min"]
        g_ann = lev_use * S * vol - 0.5 * (lev_use * vol) ** 2     # log-growth
        g_week = np.expm1(g_ann / 52)
        print(f"{r['n_positions']:>4.0f}{r['lev_min']:>13.2f}x{kelly:>10.2f}x"
              f"{('OK' if feasible else 'KHÔNG'):>10}{lev_use:>9.2f}x"
              f"{g_week*100:>17.2f}%{g_week*1_000_000:>11,.0f}")
        df.loc[df["n_positions"] == r["n_positions"], ["kelly_lev", "lev_used", "growth_week"]] = \
            [kelly, lev_use, g_week]

    print("\nĐọc bảng: 'khả thi = KHÔNG' nghĩa là chỉ riêng việc đặt đủ n vị thế đã buộc")
    print("phải chạy QUÁ Kelly — tức cấu hình đó tự huỷ trước khi bàn tới chất lượng tín hiệu.")

    df.to_csv(a.out, index=False)
    print(f"\n-> {a.out}")

    best = df.loc[df["growth_week"].idxmax()]
    cur = df[df["n_positions"] == V3.n_positions]
    print(f"\nHIỆN TẠI n={V3.n_positions}: trung vị Sharpe {float(cur['sharpe_median_fold'].iloc[0]):.2f}, "
          f"tăng trưởng {float(cur['growth_week'].iloc[0])*100:.2f}%/tuần")
    print(f"TỐT NHẤT  n={best['n_positions']:.0f}: trung vị Sharpe {best['sharpe_median_fold']:.2f}, "
          f"tăng trưởng {best['growth_week']*100:.2f}%/tuần "
          f"({best['growth_week']*1_000_000:,.0f} VND/tuần ở đòn bẩy {best['lev_used']:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
