#!/usr/bin/env python3
"""
n_positions > 50 có tốt hơn không — câu hỏi bảng độ rộng cũ KHÔNG trả lời.

VÌ SAO SCRIPT NÀY TỒN TẠI: bảng trong CLAUDE.md quét tới n=50 rồi dừng, và 50 được
chốt vì RÀNG BUỘC VỐN chứ không phải vì nó là đỉnh: $38 x 7,89 / 50 = $6,00, vừa
khít min_notional $5 x an toàn 1,2. Nhưng quan hệ đo được là ĐƠN ĐIỆU TĂNG theo độ
rộng (giao cắt ở ~60 cặp, khoảng cách NỚI RỘNG tới 120 cặp), nên "50 là tốt nhất"
chưa từng được kiểm — chỉ có "50 là lớn nhất mà $38 mua nổi".

Hai câu hỏi khác nhau, và trộn chúng lại là đúng lỗi đã sinh ra con số 41 trong
ghi chú cũ: lấy một ràng buộc rồi dùng nó như một kết luận.

Trọng số ĐỀU (`max_weight = 1/n`) đi theo n, vì đó là lựa chọn CẤU TRÚC chứ không
phải tham số dò — giữ 0,02 cố định khi n đổi sẽ đo lẫn hai thứ.

    python scripts/breadth_beyond_50.py
"""
import numpy as np
import pandas as pd

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)
from aegis.risk.portfolio import PortfolioSpec

GRID = (30, 40, 50, 60, 70, 80, 100)
MAKER = 0.39                      # mức THẬT đang đạt, không phải giả định 0,50
CAPITAL_USD = 38.0
MIN_NOTIONAL, SAFETY = 5.0, 1.2


def _stats(r: pd.Series, ppy: float):
    mu, sd = r.mean() * ppy, r.std() * np.sqrt(ppy)
    return (mu / sd if sd > 0 else np.nan), mu, sd


def main() -> int:
    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    sig = combined_signal(data, V3)
    ppy = 365.0 * 24.0 / V3.period_hours
    hi = data.close.notna().sum(axis=1) >= 100

    print("=" * 96)
    print(f"ĐỘ RỘNG DANH MỤC VƯỢT 50 — kỷ nguyên >=100 cặp, maker {MAKER} (mức thật)")
    print("=" * 96)
    print(f"{'n':>4}{'max_w':>8}{'Sharpe':>9}{'ann%':>8}{'vol%':>7}{'Kelly':>8}"
          f"{'g(Kelly)%':>11}{'trung vị fold':>15}{'%fold+':>8}{'$/vị thế @$38':>15}")
    print("-" * 96)

    for n in GRID:
        spec = PortfolioSpec(mode="zscore_riskparity", n_positions=n,
                             max_weight=1.0 / n, beta_neutral=False, vol_window=60)
        r = run_v3(data, V3, maker_ratio=MAKER, portfolio=spec, sig=sig).returns.dropna()
        r = r[r.index.isin(hi.index[hi.values])]
        s, mu, sd = _stats(r, ppy)
        kelly = mu / (sd ** 2) if sd > 0 else np.nan
        g = mu * kelly - (sd * kelly) ** 2 / 2          # g = muL - (sigma L)^2/2

        folds = np.array_split(r.values, 8)             # ổn định qua 8 đoạn con
        fs = [_stats(pd.Series(f), ppy)[0] for f in folds if len(f) > 5]
        # Đòn bẩy vừa khít cho vốn $38: mỗi vị thế đúng min_notional x an toàn.
        lev_fit = n * MIN_NOTIONAL * SAFETY / CAPITAL_USD
        per_pos = CAPITAL_USD * lev_fit / n

        print(f"{n:>4}{1/n:>8.3f}{s:>9.2f}{mu*100:>8.1f}{sd*100:>7.1f}{kelly:>8.2f}"
              f"{g*100:>11.0f}{np.median(fs):>15.2f}{np.mean(np.array(fs) > 0)*100:>7.0f}%"
              f"{per_pos:>10.2f} @{lev_fit:>4.1f}x")

    print("-" * 96)
    print("g(Kelly) = S²/2 — trần tăng trưởng BỀN VỮNG, đã trừ hình phạt bậc hai của rủi ro.")
    print(f"Cột cuối: đòn bẩy cần để mỗi vị thế đạt ${MIN_NOTIONAL:g} x {SAFETY:g} ở vốn"
          f" ${CAPITAL_USD:g}. n càng lớn càng cần đòn bẩy cao — đó là CÁI GIÁ THẬT của độ rộng")
    print("ở vốn nhỏ, và là lý do n=50 được chốt. Với vốn lớn hơn, ràng buộc này biến mất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
