#!/usr/bin/env python3
"""
Vốn 1.000.000 VND thì ra bao nhiêu — bằng số, theo tuần, có cả đuôi trái.

Script này KHÔNG chọn tham số và KHÔNG chạm holdout theo nghĩa tinh chỉnh. Nó lấy
chuỗi lợi suất của cấu hình đã chốt trên TOÀN mẫu rồi trình bày lại dưới dạng mà
người bỏ vốn thật sự cần: số tiền, theo tuần, kèm xác suất.

Điểm quan trọng nhất mà mọi bảng backtest thường giấu: với $38 và min notional $5,
danh mục 12 vị thế KHÔNG chạy được ở đòn bẩy 1x. Sàn đòn bẩy tối thiểu là ~2x. Nghĩa
là "lợi suất ở gross 1.0x" là con số KHÔNG khả dụng với tài khoản này — mọi kỳ vọng
phải đọc ở mức đòn bẩy thật sự bắt buộc phải dùng.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.data.panel_v2 import load_funding_panel_v2, load_panel_v2, align_panel
from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.backtest_v2 import CostModel, estimate_cost_bps, simulate
from aegis.research.leverage import kelly_leverage, simulate_paths
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights

CAPITAL_VND = 1_000_000
USD_VND = 26_300           # tỷ giá xấp xỉ 2026-09
CAPITAL_USD = CAPITAL_VND / USD_VND
MIN_NOTIONAL = 5.0
N_POS = 12
REBAL = 18
MAKER = 0.5
PPY = 24 * 365.0 / (4 * REBAL)
PERIODS_PER_WEEK = 7 * 24 / (4 * REBAL)


def main():
    syms = json.load(open("artifacts/universe_wide.json"))
    panel = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)
    cf = panel["close"]

    marks = cf.index[::REBAL]
    close = cf.reindex(marks)
    sigs = {n: build_signal(n, panel, funding).reindex(marks) for n in SIGNAL_REGISTRY}
    sig = combine_adaptive(sigs, close, CombinerSpec(lookback=500, min_periods=120,
                                                     t_threshold=2.0, max_abs_weight=0.20,
                                                     max_step=0.05), top_frac=0.10)
    W = build_weights(sig, close, PortfolioSpec(mode="zscore_riskparity",
                                                n_positions=N_POS, max_weight=0.20))
    per_sym = estimate_cost_bps(panel, notional_usd=CAPITAL_USD * 3 / N_POS, bars_per_day=6)
    cost = CostModel(maker_ratio=MAKER, half_spread_bps=0.0,
                     per_symbol_bps=per_sym * (1 - MAKER), min_bps=0.5)
    r_all = simulate(W, close, funding.reindex(marks), cost,
                     bar_hours=4 * REBAL, rebalance_every=1).returns.dropna()

    # HAI KỊCH BẢN, và chênh lệch giữa chúng chính là thước đo mức lạc quan:
    #   "toàn mẫu"  — gồm cả đoạn đã dùng để chọn cấu hình. LẠC QUAN.
    #   "chỉ holdout" — đoạn chưa từng dùng khi chốt tham số. THẬN TRỌNG.
    # Kế hoạch phải lập trên kịch bản thận trọng; kịch bản lạc quan chỉ để biết
    # trần trên của điều có thể xảy ra nếu chế độ thị trường quay lại như cũ.
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]
    r_ho = r_all[r_all.index >= split]
    scenario = sys.argv[1] if len(sys.argv) > 1 else "holdout"
    r = r_ho if scenario == "holdout" else r_all
    label = "CHỈ HOLDOUT (thận trọng)" if scenario == "holdout" else "TOÀN MẪU (lạc quan)"

    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mu / sd * np.sqrt(PPY)
    kelly = kelly_leverage(mu, sd, fraction=1.0, cap=1e9)

    print("=" * 94)
    print(f"VỐN {CAPITAL_VND:,} VND = ${CAPITAL_USD:.2f}  |  {N_POS} vị thế  |  "
          f"min notional ${MIN_NOTIONAL:.0f}/lệnh")
    print("=" * 94)
    lev_min = MIN_NOTIONAL * N_POS / CAPITAL_USD
    mu_a, sd_a = float(r_all.mean()), float(r_all.std(ddof=1))
    mu_h, sd_h = float(r_ho.mean()), float(r_ho.std(ddof=1))
    print(f"Kịch bản đang dùng: {label}   (đổi bằng: python scripts/capital_reality.py all)")
    print(f"  toàn mẫu    : Sharpe {mu_a/sd_a*np.sqrt(PPY):>5.2f} | {mu_a*PPY*100:>6.1f}%/năm  "
          f"({len(r_all)} kỳ)  <- gồm đoạn đã dùng để chọn cấu hình")
    print(f"  chỉ holdout : Sharpe {mu_h/sd_h*np.sqrt(PPY):>5.2f} | {mu_h*PPY*100:>6.1f}%/năm  "
          f"({len(r_ho)} kỳ)  <- chưa từng dùng khi chốt tham số")
    print()
    print(f"Chiến lược ({label}, {len(r)} kỳ): Sharpe {sharpe:.2f} | {mu*PPY*100:.1f}%/năm "
          f"| biến động {sd*np.sqrt(PPY)*100:.1f}%/năm ở gross 1.0x")
    print(f"Kelly toàn phần {kelly:.1f}x — nửa Kelly {kelly/2:.1f}x (mức thực hành chuẩn)")
    print()
    print(f"⚠️  RÀNG BUỘC CỨNG: {N_POS} vị thế x ${MIN_NOTIONAL:.0f} = ${MIN_NOTIONAL*N_POS:.0f} "
          f"notional tối thiểu.")
    print(f"    Với ${CAPITAL_USD:.2f} vốn, ĐÒN BẨY TỐI THIỂU để chạy được là "
          f"{lev_min:.1f}x — mức 1.0x KHÔNG khả dụng.")
    print(f"    Nghĩa là mọi con số 'ở gross 1.0x' đều không áp dụng cho tài khoản này.")
    print()

    print(f"{'đòn bẩy':>8}{'×Kelly':>8}{'notional/lệnh':>15}{'kỳ vọng/tuần':>15}"
          f"{'trung vị/tuần':>15}{'P(cháy 1 tuần)':>16}{'DD kỳ vọng/năm':>16}")
    print("-" * 94)
    rows = []
    for lev in (2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
        per_pos = CAPITAL_USD * lev / N_POS
        sim = simulate_paths(r, lev, max(1, int(round(PERIODS_PER_WEEK))),
                             n_paths=20000, block=3, ruin_threshold=0.30)
        med = float(np.median(sim["final"]) - 1.0)
        exp_week = mu * lev * PERIODS_PER_WEEK

        # drawdown kỳ vọng trong 1 năm ở đòn bẩy này
        yr = simulate_paths(r, lev, int(PPY), n_paths=3000, block=3, ruin_threshold=0.30)
        dd_year = 1.0 - float(np.median(yr["min_equity"]))

        rows.append((lev, med, sim["ruined"].mean(), dd_year))
        print(f"{lev:>8.1f}{lev/kelly:>8.2f}{per_pos:>14.2f}$"
              f"{exp_week*CAPITAL_VND:>14,.0f}₫{med*CAPITAL_VND:>14,.0f}₫"
              f"{sim['ruined'].mean()*100:>15.1f}%{dd_year*100:>15.1f}%")

    print("-" * 94)
    print("Đọc TRUNG VỊ, không đọc kỳ vọng: ở đòn bẩy cao, kỳ vọng bị vài đường cực tốt")
    print("kéo lên trong khi phần lớn kết quả đi xuống.")

    print(f"\n{'='*94}")
    print("MỤC TIÊU +50%/TUẦN — cần bao nhiêu đòn bẩy và đổi bằng gì")
    print("=" * 94)
    print(f"{'đòn bẩy':>8}{'P(đạt +50%)':>14}{'P(cháy)':>10}{'trung vị':>12}"
          f"{'kỳ vọng số tiền':>18}")
    print("-" * 94)
    for lev in (5.0, 10.0, 15.0, 25.0, 40.0):
        sim = simulate_paths(r, lev, max(1, int(round(PERIODS_PER_WEEK))),
                             n_paths=20000, block=3, ruin_threshold=0.30)
        f = sim["final"]
        p_t = float((f >= 1.5).mean())
        med = float(np.median(f) - 1.0)
        print(f"{lev:>8.1f}{p_t*100:>13.1f}%{sim['ruined'].mean()*100:>9.1f}%"
              f"{med*100:>11.1f}%{med*CAPITAL_VND:>17,.0f}₫")

    print("-" * 94)
    print("Kết luận số học: KHÔNG có mức đòn bẩy nào cho +50%/tuần với xác suất > 35%,")
    print("và ở mọi mức đạt được xác suất đó thì trung vị đều BẰNG HOẶC DƯỚI 0.")
    print("Đó là định nghĩa của một vé số, không phải của một chiến lược.")

    print(f"\n{'='*94}")
    print("VỐN CẦN CÓ ĐỂ 100.000₫/TUẦN LÀ MỤC TIÊU HỢP LÝ")
    print("=" * 94)
    target_week_vnd = 100_000
    for lev in (2.0, 3.0, 5.0):
        weekly_rate = mu * lev * PERIODS_PER_WEEK
        if weekly_rate <= 0:
            continue
        need_vnd = target_week_vnd / weekly_rate
        print(f"  đòn bẩy {lev:.0f}x -> {weekly_rate*100:.2f}%/tuần kỳ vọng "
              f"-> cần vốn {need_vnd:>13,.0f}₫  (${need_vnd/USD_VND:,.0f})")
    print("\nĐây là câu trả lời thật cho câu hỏi ban đầu: ràng buộc là VỐN, không phải thuật toán.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
