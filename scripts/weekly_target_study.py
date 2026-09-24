#!/usr/bin/env python3
"""
MỤC TIÊU "+20% MỖI TUẦN" — trả lời bằng số trước khi ai đó vặn đòn bẩy để đuổi theo nó.

Câu hỏi được đặt ra (24/09/2026): "bảo đảm hệ thống sinh lời ít nhất 20% trong 1 tuần,
bằng mọi cách". Script này không tranh luận, nó đo bốn thứ:

  1. SHARPE CẦN CÓ để +X%/tuần là BỀN VỮNG. Tăng trưởng log tối đa ở đòn bẩy Kelly là
     S²/2 mỗi năm, nên +20%/tuần (log 9,48/năm) đòi S = 4,35. Không có đòn bẩy nào
     thay được Sharpe: đòn bẩy nhân lợi nhuận tuyến tính nhưng nhân rủi ro bậc hai.
  2. TRẦN DẠNG ĐÓNG của P(đạt đích trong 7 ngày) — dùng lại `goal_dp` của repo — và
     phần nào của xác suất đó đến từ EDGE, phần nào chỉ là HÌNH HỌC của trò chơi.
  3. ĐÒN BẨY CỐ ĐỊNH, có thanh lý GIỮA tuần (bước 4h), rồi GỘP 52 tuần: một tuần may
     không nói gì về cái ví nhận được sau một năm.
  4. CHÍNH SÁCH DP TỐI ƯU (`solve_goal_dp`) với đích +20% — cách tốt nhất có thể để
     chạm đích trong MỘT tuần — chấm lại với edge = 0, và xác suất đạt đích MỌI tuần.

DỮ LIỆU: lợi suất 4h TỔNG HỢP, hiệu chỉnh theo thống kê ĐÃ ĐO trong repo (bảng ở
`BASES`): Student-t(3) nhân chế độ biến động theo tuần, chuẩn hoá về đúng Sharpe/vol.
Kiểm tra hiệu chỉnh được in ra: kỳ 72h tệ nhất trong 799 kỳ phải quanh −5,2σ như đo
thật. Lý do không dùng chuỗi thật: `returns_v3_fine.csv` là của cấu hình n=12, không
phải cấu hình wide đang chạy. Có dữ liệu thật thì đối chiếu bằng
`python scripts/goal_plan.py --target 0.20 --max-lev 20`.

KẾT QUẢ ĐO 24/09/2026 (ghi vào CLAUDE.md): +20%/tuần là mục tiêu KHÔNG THỂ BẢO ĐẢM.
Mọi cách đẩy P(đạt trong một tuần) lên cao đều là BÁN BẢO HIỂM: edge = 0 vẫn cho gần
hết xác suất đó, và đòi hỏi nó MỌI tuần làm xác suất tụt về 0 theo cấp số nhân.

    python scripts/weekly_target_study.py
    python scripts/weekly_target_study.py --target 0.05 --max-lev 10
"""
import argparse

import numpy as np

from aegis.research.goal_dp import (
    GoalSpec, _bootstrap_paths, constant_leverage_ceiling, dynamic_leverage_ceiling,
    solve_goal_dp,
)

PPY = 24 * 365 / 4.0             # số nến 4h mỗi năm
BARS_WEEK = 42                   # 7 ngày x 6 nến 4h
RUIN = 0.05                      # còn 5% vốn = coi như thanh lý sạch (ký quỹ chéo)

#: Cơ sở hiệu chỉnh — đều là số ĐÃ ĐO, không phải giả định:
#:   "lạc quan"  : kỷ nguyên >=100 cặp, n=50 đều, maker 0,39 (`breadth_beyond_50.py`).
#:                 Là IN-SAMPLE / độ ổn định giai đoạn con — cận TRÊN.
#:   "thận trọng": toàn lịch sử 799 kỳ 72h, σ kỳ 2,58% (`leverage_study.py`).
BASES = {
    "lạc quan (>=100 cặp)": (2.23, 0.226),
    "thận trọng (toàn lịch sử)": (1.34, 0.0258 * np.sqrt(365 / 3)),
}


def synth(n: int, sharpe: float, vol: float, seed: int,
          nu: float = 3.0, vol_of_vol: float = 0.45) -> np.ndarray:
    """Lợi suất 4h đuôi dày có cụm biến động, đúng Sharpe/vol năm đã cho."""
    rng = np.random.default_rng(seed)
    t = rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    regime = np.repeat(np.exp(vol_of_vol * rng.standard_normal(n // BARS_WEEK + 1)
                              - vol_of_vol ** 2), BARS_WEEK)[:n]
    z = t * regime
    z = (z - z.mean()) / z.std()
    return sharpe * vol / PPY + z * vol / np.sqrt(PPY)


def _weeks(r: np.ndarray) -> np.ndarray:
    return r[: len(r) // BARS_WEEK * BARS_WEEK].reshape(-1, BARS_WEEK)


def _tail_check(r: np.ndarray, seed: int = 0) -> float:
    """Trung vị của (kỳ 72h tệ nhất trong 799 kỳ) tính bằng σ — đo thật: −5,19σ."""
    r72 = r[: len(r) // 18 * 18].reshape(-1, 18).sum(1)
    rng = np.random.default_rng(seed)
    return float(np.median([rng.choice(r72, 799).min() / r72.std() for _ in range(200)]))


def section_required(target: float) -> None:
    print("\n1. SHARPE CẦN CÓ để lợi suất tuần là BỀN VỮNG (tăng trưởng Kelly = S²/2 /năm)")
    print(f"   {'mỗi tuần':>9}{'log/năm':>9}{'Sharpe cần':>12}{'nhân vốn/năm':>15}")
    for w in sorted({0.01, 0.02, 0.05, 0.10, target}):
        g = np.log1p(w) * 52
        print(f"   {w:>9.0%}{g:>9.2f}{np.sqrt(2 * g):>12.2f}{(1 + w) ** 52:>15,.0f}x")
    print("   Đã đo: holdout 1,10 | toàn lịch sử 1,34 | kỷ nguyên >=100 cặp 2,23 (in-sample).")


def section_ceiling(target: float, days: float) -> None:
    T = days / 365.0
    print(f"\n2. TRẦN DẠNG ĐÓNG — P(+{target:.0%} trong {days:.0f} ngày)")
    # P không phụ thuộc vol, chỉ L* phụ thuộc — nên holdout đi với vol CỦA NÓ (n=12).
    rows = [("edge = 0 (tung đồng xu)", 0.0, 0.226), ("holdout 4h (n=12)", 1.10, 0.635)]
    rows += [(k, s, v) for k, (s, v) in BASES.items()]
    for name, s, v in rows:
        c = constant_leverage_ceiling(s, v, T, target)
        d = dynamic_leverage_ceiling(s, T, target)
        print(f"   {name:<27} S={s:4.2f}  cố định {c['p_max_constant']:6.1%} tại "
              f"L*={c['leverage_star']:5.1f}x | động không chặn {d['p_max_dynamic']:6.1%}")
    print(f"   edge = 0, động không chặn = 1/(1+g) = {1 / (1 + target):.1%}: phần còn lại là MẤT")
    print("   TRẮNG (bất đẳng thức martingale). Xác suất cao ở đây là do trò chơi, không do edge.")


def section_fixed(r: np.ndarray, levs, target: float, seed: int = 5) -> None:
    W = _weeks(r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(W), (20000, 52))
    print(f"   {'L':>6}{'P(>=đích)':>11}{'P(cháy)/tuần':>14}{'trung vị tuần':>15}"
          f"{'TB tuần':>9}{'  sau 52 tuần: trung vị':>24}{'P(cháy/năm)':>13}")
    for L in levs:
        eq = np.cumprod(np.maximum(1 + L * W, 0.0), axis=1)     # thanh lý GIỮA tuần
        ruin = eq.min(1) <= RUIN
        fin = np.where(ruin, 0.0, eq[:, -1])
        dead_y = ruin[idx].any(1)
        year = np.where(dead_y, 0.0, np.prod(eq[:, -1][idx], axis=1))
        med_y = float(np.median(year))
        per_wk = med_y ** (1 / 52) - 1 if med_y > 0 else -1.0
        print(f"   {L:>6.2f}{np.mean(fin >= 1 + target):>11.1%}{ruin.mean():>14.1%}"
              f"{np.median(fin) - 1:>+15.1%}{fin.mean() - 1:>+9.1%}"
              f"{f'x{med_y:.2f} ({per_wk:+.2%}/tuần)':>24}{dead_y.mean():>13.1%}")


def _run_policy(pol, spec, r_oos: np.ndarray, seed: int = 11) -> np.ndarray:
    """Chạy chính sách DP trên đường block-bootstrap; trả về vốn cuối tuần (1 = hoà)."""
    paths = _bootstrap_paths(r_oos, spec.horizon_periods, 40000, 6, seed)
    xg, Lg = pol.wealth_grid, pol.leverage_grid
    x = np.zeros(len(paths))
    j = np.zeros(len(paths), dtype=np.int64)
    dead = np.zeros(len(paths), dtype=bool)
    for t in range(spec.horizon_periods):
        a = pol.policy[spec.horizon_periods - t - 1][
            np.clip(np.searchsorted(xg, x), 0, len(xg) - 1), j]
        Lt = np.where(dead, 0.0, Lg[a])
        g = 1 + Lt * paths[:, t] - spec.cost_per_leverage_turn * np.abs(Lt - Lg[j])
        x = np.where(dead, x, x + np.log(np.maximum(g, 1e-3)))
        dead |= x <= spec.log_floor()
        j = a
    return np.where(dead, spec.ruin_floor, np.exp(x))


def section_dp(sharpe: float, vol: float, max_lev: float, target: float) -> None:
    tr = synth(200_000, sharpe, vol, 11)            # giải chính sách trên chuỗi NÀY
    oos = synth(200_000, sharpe, vol, 22)           # chấm trên chuỗi KHÁC
    print(f"   {'sàn cháy':>9}{'P(đạt)':>8}{'P(đạt) edge=0':>15}"
          f"{'P(chạm sàn)':>13}{'P(mất>=50%)':>13}"
          f"{'TB':>8}{'TB edge=0':>11}{'đạt MỌI tuần: 4 tuần':>22}{'12 tuần':>9}{'52 tuần':>9}")
    for floor in (0.70, 0.05):
        spec = GoalSpec(target_return=target, horizon_periods=BARS_WEEK, ruin_floor=floor,
                        max_leverage=max_lev, wealth_headroom=0.10)
        pol = solve_goal_dp(tr, spec)
        f = _run_policy(pol, spec, oos)
        f0 = _run_policy(pol, spec, oos - oos.mean())
        p = float(np.mean(f >= 1 + target - 1e-9))
        print(f"   {'mất ' + format(1 - floor, '.0%'):>9}{p:>8.1%}"
              f"{np.mean(f0 >= 1 + target - 1e-9):>15.1%}{np.mean(f <= floor + 1e-9):>13.1%}"
              f"{np.mean(f <= 0.5):>13.1%}"
              f"{f.mean() - 1:>+8.1%}{f0.mean() - 1:>+11.1%}"
              f"{p ** 4:>22.2%}{p ** 12:>9.3%}{p ** 52:>9.0e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.20, help="lợi suất tuần mục tiêu")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--max-lev", type=float, default=20.0)
    a = ap.parse_args(argv)

    print("=" * 96)
    print(f"MỤC TIÊU +{a.target:.0%} MỖI {a.days:.0f} NGÀY — cần gì, và đuổi theo nó thì mất gì")
    print("=" * 96)
    section_required(a.target)
    section_ceiling(a.target, a.days)

    levs = (2.0, 5.0, 7.89, 10.0, 15.0, 20.0)
    for name, (s, v) in BASES.items():
        r = synth(400_000, s, v, 1)
        print(f"\n3. ĐÒN BẨY CỐ ĐỊNH — cơ sở {name}: Sharpe {s:.2f}, vol {v:.1%}, "
              f"kỳ 72h tệ nhất/799 = {_tail_check(r):.2f}σ (đo thật −5,19σ)")
        section_fixed(r, levs, a.target)

    s, v = BASES["lạc quan (>=100 cặp)"]
    print(f"\n4. CHÍNH SÁCH DP TỐI ƯU (`solve_goal_dp`), đích +{a.target:.0%}, "
          f"trần {a.max_lev:.0f}x,"
          f" cơ sở LẠC QUAN — cách tốt nhất để chạm đích trong MỘT tuần")
    section_dp(s, v, a.max_lev, a.target)
    print("\n   Đọc: P(đạt) cao chủ yếu do HÌNH HỌC (cột edge=0 gần bằng), kỳ vọng edge=0 ≈ 0 hoặc")
    print("   âm, và tuần thua mất rất nặng. Đòi đạt MỌI tuần thì xác suất nhân lên theo mũ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
