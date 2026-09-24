#!/usr/bin/env python3
"""
Tỷ lệ maker đáng bao nhiêu tiền — đo trên CẤU HÌNH ĐANG CHẠY, không phải cấu hình cũ.

VÌ SAO SCRIPT NÀY TỒN TẠI: ghi chú 14/09 định giá "maker 0,378 -> 0,85 đáng +7,7%
lợi nhuận". Con số đó đo ở `n_positions=12`, đòn bẩy **2,0x**. Hệ thống nay chạy
`strategy_v3_wide.json` ở **7,89x**.

Chi phí là TURNOVER x đơn giá, và turnover tỷ lệ THUẬN với đòn bẩy. Nên cùng một
cải thiện tỷ lệ maker đáng gấp ~4 lần ở cấu hình hiện tại. Dùng lại con số 7,7% là
định giá thấp bản vá F45 đúng 4 lần — và vì thế xếp sai thứ tự ưu tiên công việc.

Thêm một điều bảng cũ không nói: `strategy_v3_wide.json` ghi `maker_ratio_assumed:
0.50`, còn lượt 19/09 đo được **0,39**. Live đang ĐẮT HƠN mô hình, tức Sharpe đã
kiểm định vốn đã lạc quan hơn thực tế — trước cả khi bàn tới việc cải thiện.

    python scripts/maker_ratio_study.py
"""
import numpy as np
import pandas as pd

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)
from aegis.risk.portfolio import PortfolioSpec

#: Đúng khối `portfolio` của `artifacts/strategy_v3_wide.json`.
WIDE = PortfolioSpec(mode="zscore_riskparity", n_positions=50,
                     max_weight=0.02, beta_neutral=False, vol_window=60)

#: 0,39 = đo thật ở lượt 19/09. 0,50 = giả định trong artifact. 0,85 = mục tiêu.
RATIOS = (0.39, 0.50, 0.60, 0.70, 0.85)

CAPITAL_USD = 38.0
VND_PER_USD = 26_300.0

#: Lợi suất do `run_v3` trả về là ở GROSS 1,0x. Live nhân lên bằng đòn bẩy, và CHI
#: PHÍ cũng nhân lên cùng hệ số đó — nên chênh lệch do tỷ lệ maker cũng tỷ lệ thuận
#: với đòn bẩy. Bỏ quên hệ số này là lý do ghi chú cũ định giá thấp bản vá F45:
#: "+7,7%" đo ở 2,0x, còn hệ thống nay chạy 7,89x.
LEVERAGES = (2.0, 5.0, 7.89)


def _stats(r: pd.Series, ppy: float) -> tuple:
    mu, sd = r.mean() * ppy, r.std() * np.sqrt(ppy)
    return (mu / sd if sd > 0 else np.nan), mu, sd


def main() -> int:
    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    sig = combined_signal(data, V3)                 # tốn nhất — tính MỘT lần
    ppy = 365.0 * 24.0 / V3.period_hours

    # Kỷ nguyên độ rộng cao: cấu hình wide được chốt ở đây, chấm nó trên toàn
    # lịch sử là bắt nó gánh những năm mà nó không thể tồn tại (xem CLAUDE.md).
    hi = data.close.notna().sum(axis=1) >= 100

    rows = []
    for m in RATIOS:
        res = run_v3(data, V3, maker_ratio=m, portfolio=WIDE, sig=sig)
        r = res.returns.dropna()
        s_all, a_all, _ = _stats(r, ppy)
        r_hi = r[r.index.isin(hi.index[hi.values])]
        s_hi, a_hi, _ = _stats(r_hi, ppy)
        rows.append((m, s_all, a_all, s_hi, a_hi))

    print("=" * 78)
    print("TỶ LỆ MAKER -> LỢI NHUẬN, trên cấu hình n=50 đang chạy")
    print("=" * 78)
    print(f"{'maker':>7}{'Sharpe (toàn)':>15}{'ann% (toàn)':>13}"
          f"{'Sharpe (>=100)':>16}{'ann% (>=100)':>14}")
    print("-" * 78)
    for m, s_a, a_a, s_h, a_h in rows:
        tag = "  <- live 19/09" if m == 0.39 else ("  <- artifact" if m == 0.50 else "")
        print(f"{m:>7.2f}{s_a:>15.3f}{a_a*100:>13.1f}{s_h:>16.3f}{a_h*100:>14.1f}{tag}")
    print("-" * 78)

    base = next(r for r in rows if r[0] == 0.39)
    print("\nGIÁ TRỊ BẢN VÁ F45 — so với 0,39 đang thật sự đạt được (kỷ nguyên >=100 cặp)")
    print("Sharpe không phụ thuộc đòn bẩy; lợi nhuận thì CÓ, vì chi phí cũng nhân lên.")
    hdr = "".join(f"{f'+ann% @{L:g}x':>14}" for L in LEVERAGES)
    print(f"{'maker':>7}{'+Sharpe':>10}{'+ann% @1x':>12}{hdr}")
    print("-" * (29 + 14 * len(LEVERAGES)))
    for m, s_a, a_a, s_h, a_h in rows:
        if m == 0.39:
            continue
        d = a_h - base[4]
        lev = "".join(f"{d*L*100:>+14.1f}" for L in LEVERAGES)
        print(f"{m:>7.2f}{s_h - base[3]:>+10.3f}{d*100:>+12.1f}{lev}")
    print("-" * (29 + 14 * len(LEVERAGES)))

    d85 = next(r for r in rows if r[0] == 0.85)[4] - base[4]
    print(f"\nQuy ra tiền ở vốn ${CAPITAL_USD:g} (maker 0,39 -> 0,85):")
    for L in LEVERAGES:
        vnd = CAPITAL_USD * d85 * L / 52.0 * VND_PER_USD
        print(f"   đòn bẩy {L:>4g}x : {d85*L*100:>+6.1f}%/năm  = {vnd:>+9,.0f} VND/tuần")
    print("\nGhi chú 14/09 định giá '+7,7% lợi nhuận' — đo ở n=12 / 2,0x. Cùng một cải")
    print("thiện ở cấu hình n=50 / 7,89x đáng GẤP ~4 LẦN, vì turnover nhân theo đòn bẩy.")

    # Tăng trưởng bền vững ở đòn bẩy Kelly: g = S^2/2 (xem CLAUDE.md).
    print(f"\nTrần tăng trưởng bền vững S²/2:  "
          f"maker 0,39 -> {base[3]**2/2*100:.0f}%/năm  |  "
          f"maker 0,85 -> {d85 and next(r for r in rows if r[0]==0.85)[3]**2/2*100:.0f}%/năm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
