#!/usr/bin/env python3
"""
Mỗi vị thế nên có cỡ KHÁC NHAU hay BẰNG NHAU — đo, không tranh luận.

CÂU HỎI: hệ thống có nên phân biệt cỡ (tức "đòn bẩy") giữa các lệnh không?

ĐIỀU CẦN BIẾT TRƯỚC: với ký quỹ CHÉO trên perp USDⓈ-M, "đòn bẩy từng lệnh" KHÔNG
phải một đại lượng rủi ro độc lập. Cài `leverage=20x` cho một cặp chỉ đổi bậc ký quỹ
ban đầu của cặp đó; rủi ro danh mục vẫn là TỔNG NOTIONAL / VỐN. Thứ thật sự phân biệt
được giữa các lệnh là TRỌNG SỐ (notional), và đó chính là việc `PortfolioSpec.mode`
đang làm.

Năm chế độ, và chúng trả lời đúng câu hỏi trên:
  rank_binary       — MỌI vị thế bằng nhau (không phân biệt gì)
  rank_riskparity   — phân biệt theo BIẾN ĐỘNG (cặp động mạnh thì nhỏ lại)
  zscore            — phân biệt theo ĐỘ MẠNH TÍN HIỆU
  zscore_riskparity — phân biệt theo CẢ HAI  (đang chạy)
  mvo               — tối ưu hoá theo ma trận hiệp phương sai đầy đủ

⚠️ `max_weight` KẸP PHẲNG sự phân biệt. Ở n=50, trần 0,02 = đúng 1/50 = trọng số đều,
nên mọi chế độ đều bị ép về gần bằng nhau (live đo dải 0,0148-0,0202, chênh 1,37 lần).
Vì thế phải quét ở CẢ trần chặt lẫn trần lỏng — nếu không ta chỉ đang đo cái trần.

Đo cả ĐUÔI TRÁI chứ không chỉ Sharpe: đuôi trái mới là thứ quyết định đòn bẩy an toàn.

    python scripts/sizing_mode_study.py
"""
import numpy as np

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)
from aegis.risk.portfolio import PortfolioSpec

MODES = ("rank_binary", "rank_riskparity", "zscore", "zscore_riskparity")
CAPS = (0.02, 0.05, 0.10)          # 0,02 = 1/50 (đều) ... 0,10 = cho phân biệt bung ra
N = 50
MAKER = 0.39


def main() -> int:
    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    sig = combined_signal(data, V3)
    ppy = 365.0 * 24.0 / V3.period_hours
    hi = data.close.notna().sum(axis=1) >= 100
    hi_idx = set(hi.index[hi.values])

    print("=" * 92)
    print(f"CỠ VỊ THẾ: PHÂN BIỆT hay BẰNG NHAU — n={N}, kỷ nguyên >=100 cặp, maker {MAKER}")
    print("=" * 92)
    print(f"{'chế độ':<20}{'trần w':>8}{'Sharpe':>9}{'ann%':>8}{'vol%':>7}"
          f"{'kỳ tệ nhất%':>13}{'cháy từ':>10}{'L* an toàn':>12}")
    print("-" * 92)

    for mode in MODES:
        for cap in CAPS:
            spec = PortfolioSpec(mode=mode, n_positions=N, max_weight=cap,
                                 beta_neutral=False, vol_window=60)
            try:
                r = run_v3(data, V3, maker_ratio=MAKER, portfolio=spec,
                           sig=sig).returns.dropna()
            except Exception as exc:
                print(f"{mode:<20}{cap:>8.3f}  lỗi: {exc}")
                continue
            r = r[r.index.isin(hi_idx)].values
            mu, sd = r.mean() * ppy, r.std() * np.sqrt(ppy)
            worst = r.min()
            # Đòn bẩy chịu được cú sốc 5,19 sigma (cú tệ nhất lịch sử, quy theo sigma
            # của chính chuỗi này) mà chỉ mất tối đa 60% vốn — không phải cháy.
            shock = 5.19 * r.std()
            l_safe = 0.60 / shock if shock > 0 else np.nan
            print(f"{mode:<20}{cap:>8.3f}{mu/sd:>9.2f}{mu*100:>8.1f}{sd*100:>7.1f}"
                  f"{worst*100:>13.2f}{-1/worst:>9.1f}x{l_safe:>11.1f}x")
        print()

    print("-" * 92)
    print("'cháy từ' = đòn bẩy mà tại đó kỳ tệ nhất ĐÃ TỪNG xảy ra quét sạch vốn.")
    print("'L* an toàn' = đòn bẩy chịu được cú sốc 5,19σ (cú tệ nhất lịch sử) mà mất <= 60%.")

    # ---- Dạng phân biệt THỨ BA: theo CHI PHÍ giao dịch ----------------------
    # Đây là dạng duy nhất có cơ chế rõ: cặp spread rộng ăn mất lợi nhuận thật, và
    # chi phí đã đo đáng tới 21%/năm ở 7,89x (`maker_ratio_study.py`). Nếu có chỗ nào
    # để phân biệt giữa các lệnh thì đây là chỗ hợp lý nhất.
    bps = data.per_symbol_bps.reindex(sig.columns)
    print()
    print("=" * 92)
    print("PHÂN BIỆT THEO CHI PHÍ — loại bớt cặp đắt khỏi bể chọn")
    print("=" * 92)
    print(f"chi phí/cặp (bp, một chiều): trung vị {bps.median():.2f} | "
          f"p25 {bps.quantile(.25):.2f} | p75 {bps.quantile(.75):.2f} | max {bps.max():.2f}")
    print(f"{'bể chọn':<34}{'n cặp':>7}{'Sharpe':>9}{'ann%':>8}{'vol%':>7}{'kỳ tệ nhất%':>13}")
    print("-" * 92)
    spec = PortfolioSpec(mode="zscore_riskparity", n_positions=N,
                         max_weight=0.02, beta_neutral=False, vol_window=60)
    for q, lab in ((1.00, "TOÀN BỘ (đang chạy)"), (0.90, "bỏ 10% đắt nhất"),
                   (0.75, "bỏ 25% đắt nhất"), (0.50, "chỉ 50% rẻ nhất"),
                   (0.25, "chỉ 25% rẻ nhất")):
        s2 = sig.copy()
        if q < 1.0:
            keep = set(bps[bps <= bps.quantile(q)].index)
            drop = [c for c in sig.columns if c not in keep]
            if drop:
                s2[drop] = np.nan
            nkeep = len(keep)
        else:
            nkeep = len(sig.columns)
        r = run_v3(data, V3, maker_ratio=MAKER, portfolio=spec, sig=s2).returns.dropna()
        r = r[r.index.isin(hi_idx)].values
        mu, sd = r.mean() * ppy, r.std() * np.sqrt(ppy)
        print(f"{lab:<34}{nkeep:>7}{mu/sd:>9.2f}{mu*100:>8.1f}{sd*100:>7.1f}"
              f"{r.min()*100:>13.2f}")
    print("-" * 92)
    print("Cặp ĐẮT chính là cặp NHỎ và KÉM HIỆU QUẢ — tức nơi alpha sống. Tiết kiệm vài bp")
    print("không bù nổi phần alpha mất đi. `IR = IC x sqrt(độ rộng)`: mọi phép TẬP TRUNG")
    print("— theo tín hiệu, theo biến động, hay theo chi phí — đều bán đi chính nguồn edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
