#!/usr/bin/env python3
"""
ĐÒN BẨY NÊN LÀ BAO NHIÊU — trả lời bằng compounding THẬT, không bằng xấp xỉ bậc hai.

VÌ SAO KHÔNG DÙNG `g = muL - (sigma*L)^2/2`: công thức đó là khai triển Taylor bậc 2
của `E[log(1 + L*r)]`. Nó giả định (a) lợi suất gần Gauss và (b) tái cân bằng liên
tục. Với perp crypto ở lưới 72h thì CẢ HAI đều sai, và sai theo hướng LẠC QUAN — nó
bỏ qua đuôi trái, nơi `log(1 + L*r)` phân kỳ về âm vô cùng.

BA ĐIỀU SCRIPT NÀY ĐO MÀ CÔNG THỨC KHÔNG THẤY:

1. TĂNG TRƯỞNG THẬT `E[log(1 + L*r)]` trên chính chuỗi lịch sử — đuôi dày vào thẳng
   phép tính thay vì bị giả định đi.

2. THANH LÝ GIỮA KỲ. Trong 72h hệ thống giữ KHỐI LƯỢNG cố định, nên equity đi tuyến
   tính theo lợi suất TÍCH LUỸ trong kỳ: `E_t = E_0 (1 + L*C_t)`. Cháy khi
   `C_t <= mm - 1/L`. Lưới 72h chỉ thấy điểm đầu và cuối kỳ nên hoàn toàn mù với cú
   sụt ở giữa — đúng lỗi đã ghi trong CLAUDE.md (maxDD thật 45,5% chứ không phải 36,9%).

3. SAI SỐ ƯỚC LƯỢNG. `L* = mu/sigma^2` là hàm của `mu`, đại lượng ước lượng TỆ NHẤT
   trong tài chính. Đã đo trong repo này: tương quan Sharpe(1 năm qua) với Sharpe(quý
   tới) = -0,015. Không dự báo được Sharpe thì không dự báo được Kelly. Bootstrap khối
   cho ra PHÂN PHỐI của L*, và kiểm walk-forward cho biết chọn L từ quá khứ có sống
   sót ra tương lai không.

    python scripts/leverage_study.py
"""
import dataclasses

import numpy as np
import pandas as pd

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, WIDE_CONFIG_FILE, config_from_json, load_v3_data, run_v3, run_v3_fine,
)

MAKER = 0.39                     # mức THẬT đo ở lượt 19/09, không phải giả định 0,50
GRID = np.arange(1.0, 14.1, 0.5)
MM = 0.01                        # ký quỹ duy trì ~1% (Binance, vị thế nhỏ)
BARS_PER_PERIOD = 18             # 18 nến 4h = 72h
RNG = np.random.default_rng(20260922)


def g_exact(r: np.ndarray, L: float, ppy: float) -> float:
    """E[log(1 + L*r)] * kỳ/năm. Trả -inf nếu có kỳ làm cháy tài khoản."""
    x = 1.0 + L * r
    if (x <= 0).any():
        return -np.inf
    return float(np.log(x).mean() * ppy)


def path_risk(fine: np.ndarray, L: float) -> tuple:
    """
    Sụt giảm và xác suất cháy khi giữ KHỐI LƯỢNG cố định trong mỗi kỳ 72h.

    Trong kỳ, gross cố định = L * E_0, nên E_t = E_0 (1 + L * C_t) với C_t là lợi
    suất tích luỹ từ đầu kỳ. Cháy khi vốn còn lại không đủ ký quỹ duy trì.
    """
    n_per = len(fine) // BARS_PER_PERIOD
    equity, peak, maxdd, ruined = 1.0, 1.0, 0.0, 0
    for p in range(n_per):
        seg = fine[p * BARS_PER_PERIOD:(p + 1) * BARS_PER_PERIOD]
        c = np.cumsum(seg)                       # lợi suất tích luỹ trong kỳ
        path = equity * (1.0 + L * c)
        if (1.0 + L * c <= MM * L).any():        # chạm ký quỹ duy trì
            ruined += 1
            equity = equity * max(1e-12, 1.0 + L * c[0])
            maxdd = 1.0
            continue
        peak = max(peak, path.max())
        maxdd = max(maxdd, 1.0 - path.min() / peak)
        equity = path[-1]
    return maxdd, ruined / max(1, n_per)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=WIDE_CONFIG_FILE,
                    help="file cấu hình daemon chạy [F51]")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="chi phí một chiều ĐO THẬT (vd 15.7 từ replay 24/09)")
    a = ap.parse_args(argv)

    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    # [FIX F51] Bảng "4-6x" trong CLAUDE.md (22/09) đo bằng `WIDE` + tầng gộp V3
    # (`max_step=0,05`) và chi phí mô hình ~4,5bp. Daemon chạy `max_step=0,025`, và
    # replay đo chi phí 15,7bp. Nay mặc định đo ĐÚNG cấu hình daemon; chi phí qua cờ.
    # MỘT lô, tường minh: chia lô giảm phương sai do chọn giờ, nên đòn bẩy đo ở một lô
    # là cận THẬN TRỌNG. `run_v3` từ chối cấu hình chia lô thay vì âm thầm bỏ qua.
    cfg = dataclasses.replace(config_from_json(a.config), maker_ratio=MAKER, n_tranches=1)

    per = run_v3(data, cfg, maker_ratio=MAKER, cost_bps=a.cost_bps).returns.dropna()
    fin = run_v3_fine(data, cfg, maker_ratio=MAKER, cost_bps=a.cost_bps).returns.dropna()
    ppy = 365.0 * 24.0 / V3.period_hours

    hi = data.close.notna().sum(axis=1) >= 100
    hi60 = data.close.notna().sum(axis=1) >= 60      # đủ dài để walk-forward có chỗ
    hi_idx = set(hi.index[hi.values])
    per_hi = per[per.index.isin(hi_idx)]
    fin_hi = fin[fin.index.isin(hi_idx)]

    for lab, r, f in (("TOÀN LỊCH SỬ", per.values, fin.values),
                      ("KỶ NGUYÊN >=100 CẶP", per_hi.values, fin_hi.values)):
        mu, sd = r.mean() * ppy, r.std() * np.sqrt(ppy)
        print("=" * 86)
        print(f"{lab} — {len(r)} kỳ 72h, {len(f)} nến 4h | Sharpe {mu/sd:.2f} "
              f"| mu {mu*100:.1f}% | sigma {sd*100:.1f}%")
        print("=" * 86)
        print(f"{'L':>5}{'g XẤP XỈ%':>12}{'g THẬT%':>11}{'chênh':>9}"
              f"{'maxDD%':>9}{'P(cháy)/kỳ':>13}{'kỳ tệ nhất%':>13}")
        print("-" * 86)
        best_L, best_g = None, -np.inf
        for L in GRID:
            gq = (mu * L - (sd * L) ** 2 / 2) * 100
            ge = g_exact(r, L, ppy)
            dd, pru = path_risk(f, L)
            worst = r.min() * L * 100
            if ge > best_g:
                best_g, best_L = ge, L
            ges = "CHÁY" if not np.isfinite(ge) else f"{ge*100:>10.0f}"
            gap = "" if not np.isfinite(ge) else f"{ge*100-gq:>+9.0f}"
            if L % 1 == 0 or L == best_L:
                print(f"{L:>5.1f}{gq:>12.0f}{ges:>11}{gap:>9}"
                      f"{dd*100:>9.0f}{pru*100:>12.1f}%{worst:>13.1f}")
        print("-" * 86)
        print(f"  L* theo g THẬT = {best_L:.1f}x  (g = {best_g*100:.0f}%/năm)   |   "
              f"Kelly xấp xỉ = {mu/sd**2:.2f}x")

        # --- bootstrap khối: L* biến động bao nhiêu vì SAI SỐ ƯỚC LƯỢNG? ---
        Ls = []
        nb, blk = 400, 8
        for _ in range(nb):
            idx = []
            while len(idx) < len(r):
                s = RNG.integers(0, max(1, len(r) - blk))
                idx.extend(range(s, s + blk))
            br = r[np.array(idx[:len(r)])]
            gs = [g_exact(br, L, ppy) for L in GRID]
            Ls.append(GRID[int(np.nanargmax(gs))])
        Ls = np.array(Ls)
        print(f"  L* qua {nb} lần bootstrap khối: trung vị {np.median(Ls):.1f}x | "
              f"KTC 90% [{np.percentile(Ls,5):.1f} .. {np.percentile(Ls,95):.1f}]x")
        for L in (5.0, 7.89):
            # KHÔNG được lọc `isfinite` ở đây. -inf nghĩa là CHÁY TÀI KHOẢN, và vứt
            # nó đi rồi lấy trung vị phần còn lại là bỏ đúng đuôi trái mà toàn bộ câu
            # hỏi về đòn bẩy xoay quanh. Cháy được BÁO RIÊNG như một xác suất.
            gs = np.array([g_exact(r[RNG.integers(0, len(r), len(r))], L, ppy)
                           for _ in range(300)])
            ruin = ~np.isfinite(gs)
            ok = gs[~ruin]
            print(f"     g({L:g}x): P(CHÁY) = {ruin.mean()*100:>4.0f}%"
                  f" | trong các lần KHÔNG cháy: trung vị {np.median(ok)*100:>5.0f}%/năm,"
                  f" KTC 90% [{np.percentile(ok,5)*100:>5.0f} .. {np.percentile(ok,95)*100:>5.0f}]%,"
                  f" P(g<0) = {(ok<0).mean()*100:.0f}%")
        worst = r.min()
        print(f"  Kỳ 72h TỆ NHẤT ở gross 1,0x: {worst*100:.2f}%  ->  cháy sạch vốn từ "
              f"đòn bẩy {-1/worst:.2f}x trở lên (chưa tính ký quỹ duy trì).")
        print()

    # --- kiểm WALK-FORWARD: chọn L từ quá khứ có sống ra tương lai không? ---
    print("=" * 86)
    print("WALK-FORWARD — chọn L trên 1 năm quá khứ, chấm trên quý kế tiếp")
    print("=" * 86)
    for lab, series, W, F in (("TOÀN LỊCH SỬ", per.values, 122, 30),
                              ("KỶ NGUYÊN >=60 CẶP", per[per.index.isin(
                                  set(hi60.index[hi60.values]))].values, 61, 20)):
        r = series
        rows = []
        for i in range(W, len(r) - F, 3):
            past, fut = r[i - W:i], r[i:i + F]
            gs = [g_exact(past, L, ppy) for L in GRID]
            Lp = GRID[int(np.nanargmax(gs))]
            rows.append((Lp, g_exact(fut, Lp, ppy), g_exact(fut, 5.0, ppy),
                         g_exact(fut, 7.89, ppy)))
        a = np.array(rows, dtype=float)
        if not len(a):
            continue
        print(f"\n  {lab} — {len(a)} cửa sổ. L* chọn từ quá khứ: trung vị "
              f"{np.median(a[:,0]):.1f}x, dải [{a[:,0].min():.1f} .. {a[:,0].max():.1f}]x")
        print(f"{'chiến lược chọn L':>28}{'P(cháy)':>10}{'g TB%':>10}"
              f"{'trung vị%':>12}{'%cửa sổ âm':>13}")
        print("  " + "-" * 71)
        for nm, col in (("L* tối ưu hoá theo quá khứ", 1), ("cố định 5,0x", 2),
                        ("cố định 7,89x", 3)):
            c = a[:, col]
            ruin = ~np.isfinite(c)
            ok = c[~ruin]
            print(f"{nm:>28}{ruin.mean()*100:>9.0f}%{ok.mean()*100:>9.0f}%"
                  f"{np.median(ok)*100:>11.0f}%{(ok<0).mean()*100:>12.0f}%")
        print("  " + "-" * 71)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
