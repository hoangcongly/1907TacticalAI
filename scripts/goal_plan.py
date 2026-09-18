#!/usr/bin/env python3
"""
MỤC TIÊU 1.000.000 VND -> +50.000 VND TRONG 7 NGÀY — trả lời bằng số, không bằng ý kiến.

Script này không cố thuyết phục rằng mục tiêu khả thi hay bất khả thi. Nó tính bốn thứ
và để người bỏ vốn quyết định:

  1. TRẦN — xác suất tốt nhất có thể đạt, dạng đóng, không cần mô phỏng.
  2. CHÍNH SÁCH — quy hoạch động giải trên train, chấm trên holdout.
  3. NGUỒN GỐC CỦA XÁC SUẤT — bao nhiêu đến từ edge, bao nhiêu chỉ đến từ hình học
     (đích +5% gần hơn ngưỡng cháy -30% rất nhiều). Đây là phần dễ tự lừa nhất.
  4. RÀNG BUỘC VỐN — $38 có mua nổi số vị thế mà chiến lược cần không.

    python scripts/goal_plan.py
    python scripts/goal_plan.py --target 0.05 --days 7 --floor 0.70 --max-lev 5
"""
import argparse
import json
import sys
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.research.goal_dp import (
    GoalSpec, constant_leverage_ceiling, dynamic_leverage_ceiling,
    evaluate_policy, solve_goal_dp,
)
from aegis.research.strategy_v3 import V3

FINE_CSV = "artifacts/returns_v3_fine.csv"
OUT_JSON = "artifacts/goal_plan.json"
VND_PER_USD = 26_300.0          # xấp xỉ; chỉ dùng để quy đổi cho dễ đọc


def _sharpe(r, ppy):
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ppy))


def _tbl(res: pd.DataFrame, title: str):
    print(f"\n{title}")
    print(f"{'chính sách':<16}{'P(đạt)':>9}{'P(cháy)':>9}{'P(lỗ)':>8}{'trung vị':>10}"
          f"{'trung bình':>11}{'p05':>9}{'p95':>9}{'L tb':>7}")
    print("-" * 88)
    for _, r in res.iterrows():
        print(f"{r['policy']:<16}{r['p_target']*100:>8.1f}%{r['p_ruin']*100:>8.1f}%"
              f"{r['p_loss']*100:>7.1f}%{r['median']*100:>9.2f}%{r['mean']*100:>10.2f}%"
              f"{r['p05']*100:>8.1f}%{r['p95']*100:>8.1f}%{r['avg_leverage']:>7.2f}"
              f"{r['avg_leverage_on']:>10.2f}{r['frac_flat']*100:>7.0f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.05, help="lợi suất mục tiêu (0.05 = +5%%)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--floor", type=float, default=0.70, help="ngưỡng cháy (0.70 = mất 30%%)")
    ap.add_argument("--max-lev", type=float, default=5.0)
    ap.add_argument("--capital-vnd", type=float, default=1_000_000.0)
    ap.add_argument("--paths", type=int, default=40000)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args(argv)

    bar_h = V3.bar_hours
    ppy = 24 * 365.0 / bar_h
    horizon = int(round(a.days * 24 / bar_h))

    df = pd.read_csv(FINE_CSV)
    tr = df[df.segment == "train"]["net_return"].dropna().to_numpy()
    ho = df[df.segment == "holdout"]["net_return"].dropna().to_numpy()
    s_tr, s_ho = _sharpe(tr, ppy), _sharpe(ho, ppy)
    vol_ho = float(np.std(ho, ddof=1) * np.sqrt(ppy))
    cap_usd = a.capital_vnd / VND_PER_USD

    print("=" * 90)
    print(f"MỤC TIÊU: {a.capital_vnd:,.0f} VND  ->  +{a.capital_vnd*a.target:,.0f} VND "
          f"(+{a.target:.0%}) trong {a.days:.0f} ngày")
    print(f"Vốn ≈ ${cap_usd:.2f} | chân trời = {horizon} nến {bar_h:.0f}h | "
          f"ngưỡng cháy = mất {(1-a.floor):.0%}")
    print("=" * 90)
    print(f"Chiến lược v3 đo trên lưới {bar_h:.0f}h (độ phân giải sàn thật sự nhìn thấy):")
    print(f"  train   : Sharpe {s_tr:.2f}  ({len(tr):>5} nến)")
    print(f"  holdout : Sharpe {s_ho:.2f}  ({len(ho):>5} nến)   <- dùng con số NÀY để kỳ vọng")

    # ---- 1. TRẦN (dạng đóng) ----------------------------------------------
    T_years = a.days / 365.0
    c = constant_leverage_ceiling(s_ho, vol_ho, T_years, a.target)
    d = dynamic_leverage_ceiling(s_ho, T_years, a.target)
    print(f"\n{'─'*90}\n1. TRẦN — xác suất tốt nhất có thể, tính bằng công thức\n{'─'*90}")
    print(f"  P = Phi( S*sqrt(T) - sqrt(2*ln(1+g)) )")
    print(f"      phần do chiến lược đóng góp   S*sqrt(T)      = {c['edge_term']:+.4f}")
    print(f"      cái giá của mục tiêu          sqrt(2ln(1+g)) = {c['goal_cost_term']:+.4f}  "
          f"(KHÔNG phụ thuộc chiến lược)")
    print(f"      z = {c['z_score']:+.4f}")
    print(f"\n  Trần với ĐÒN BẨY CỐ ĐỊNH   : {c['p_max_constant']*100:5.1f}%  "
          f"(đạt tại L* = {c['leverage_star']:.2f}x)")
    print(f"  Trần với ĐIỀU KHIỂN ĐỘNG   : {d['p_max_dynamic']*100:5.1f}%  "
          f"(cần đòn bẩy không chặn — KHÔNG đạt được, chỉ để biết dư địa)")
    print(f"\n  => Tăng đòn bẩy KHÔNG BAO GIỜ vượt {c['p_max_constant']*100:.1f}%. "
          f"Khoảng cách tới {d['p_max_dynamic']*100:.0f}% là phần thưởng của việc điều khiển động.")

    # ---- 2. CHÍNH SÁCH -----------------------------------------------------
    # Sàn đòn bẩy ĐẶT ĐƯỢC LỆNH — ràng buộc của sàn, không phải sở thích rủi ro.
    lev_floor = GoalSpec.executable_floor(cap_usd, V3.n_positions, V3.min_notional_usd)
    print(f"\n  Gross tối thiểu để {V3.n_positions} vị thế đều vượt min notional "
          f"${V3.min_notional_usd:.0f}: {lev_floor:.2f}x")
    print(f"  => lưới hành động = {{0}} hợp [{lev_floor:.2f}, {a.max_lev:.1f}] "
          f"(đứng ngoài đặt được; 0,25x thì không)")
    spec = GoalSpec(target_return=a.target, horizon_periods=horizon,
                    ruin_floor=a.floor, max_leverage=a.max_lev,
                    min_executable_leverage=lev_floor)
    pol = solve_goal_dp(tr, spec)
    print(f"\n{'─'*90}\n2. CHÍNH SÁCH TỐI ƯU — giải trên TRAIN, chấm trên HOLDOUT\n{'─'*90}")
    print(f"  DP tự nói (in-sample, i.i.d. trên train): {pol.p_success(1.0, horizon, 0.0)*100:.1f}% "
          f"— con số này LẠC QUAN, đừng dùng")

    ks = [horizon, horizon*3//4, horizon//2, horizon//4, 3, 1]
    print(f"\n  Đòn bẩy gộp nên đặt (đang giữ 2.0x):")
    print(f"  {'vốn':>8}" + "".join(f"{'còn '+str(k):>9}" for k in ks))
    print("  " + "-" * (8 + 9*len(ks)))
    for wr in (0.85, 0.95, 1.00, 1.02, 1.0+a.target-0.001, 1.0+a.target+0.01):
        print(f"  {wr:>8.3f}" + "".join(f"{pol.leverage_for(wr, k, 2.0):>9.2f}" for k in ks))
    print(f"\n  Đọc bảng: HẠ đòn bẩy khi đã gần đích (không còn gì để được, chỉ còn để mất),")
    print(f"  TĂNG khi sắp hết giờ mà chưa tới. Không luật nào ở đây được viết tay — quy nạp")
    print(f"  lùi tự suy ra cả hai từ đúng một mục tiêu: tối đa P(đạt đích).")

    levs = [1.0, 2.0, 3.0, 4.0, a.max_lev]
    res = evaluate_policy(ho, pol, spec, constant_leverages=levs, n_paths=a.paths, block=6)
    _tbl(res, f"  CHẤM TRÊN HOLDOUT ({a.paths:,} đường block bootstrap, khối 6 nến = 24h):")

    dp = res.iloc[-1]
    best_const = res.iloc[:-1].loc[res.iloc[:-1]["p_target"].idxmax()]
    print(f"\n  DP {dp['p_target']*100:.1f}% so với tốt nhất của đòn bẩy cố định "
          f"{best_const['p_target']*100:.1f}% ({best_const['policy']})"
          f"  ->  +{(dp['p_target']-best_const['p_target'])*100:.1f} điểm phần trăm,"
          f" ở đòn bẩy TRUNG BÌNH THẤP HƠN ({dp['avg_leverage']:.2f}x so với "
          f"{best_const['avg_leverage']:.2f}x).")

    # ---- 3. XÁC SUẤT NÀY ĐẾN TỪ ĐÂU ---------------------------------------
    print(f"\n{'─'*90}\n3. XÁC SUẤT ĐẾN TỪ ĐÂU — phần dễ tự lừa nhất\n{'─'*90}")
    print("  Đích +5% gần hơn ngưỡng cháy -30% rất nhiều. Ngay cả một trò chơi CÔNG BẰNG")
    print("  (không edge) cũng cho xác suất chạm đích trước khi cháy khá cao — đó là hình")
    print("  học của bài toán, không phải chất lượng của chiến lược. Phải tách hai phần đó.")
    zero = ho - ho.mean()                       # giữ nguyên biến động & cụm, bỏ hết edge
    res0 = evaluate_policy(zero, pol, spec, constant_leverages=[2.0], n_paths=a.paths, block=6)
    _tbl(res0, "  STRESS: CÙNG chính sách, CÙNG biến động, nhưng EDGE = 0 (đã trừ trung bình):")
    dp0 = res0.iloc[-1]
    edge_part = (dp["p_target"] - dp0["p_target"]) * 100
    print(f"\n  => Trong {dp['p_target']*100:.1f}%, có {dp0['p_target']*100:.1f} điểm là HÌNH HỌC")
    print(f"     (đích gần, sàn xa) và chỉ {edge_part:.1f} điểm là do EDGE của chiến lược.")
    print(f"     Nói cách khác: phần lớn xác suất đến từ việc đặt mục tiêu KHIÊM TỐN, không")
    print(f"     phải từ việc mô hình giỏi. Đó là tin tốt — nó bền hơn nhiều so với edge.")
    print(f"     Nhưng hãy nhìn cột 'trung bình' ở kịch bản edge=0: {dp0['mean']*100:+.2f}%.")
    print(f"     Xác suất thắng cao KHÔNG có nghĩa kỳ vọng dương. Nó chỉ có nghĩa là thắng")
    print(f"     nhỏ thường xuyên và thua lớn hiếm khi — đúng hình dạng của việc bán bảo hiểm.")

    # ---- 4. RÀNG BUỘC VỐN --------------------------------------------------
    print(f"\n{'─'*90}\n4. RÀNG BUỘC VỐN — ${cap_usd:.2f} có mua nổi chiến lược này không\n{'─'*90}")
    print(f"  {'đòn bẩy':>9}{'notional':>11}{'/vị thế':>10}{'đủ min $5?':>12}{'vị thế mua nổi':>17}")
    print("  " + "-" * 59)
    for L in (1, 2, 3, 4, 5):
        notion = cap_usd * L
        per = notion / V3.n_positions
        afford = int(notion // V3.min_notional_usd)
        ok = "OK" if per >= V3.min_notional_usd else "KHÔNG ĐỦ"
        print(f"  {L:>9}{notion:>10.0f}${per:>9.2f}${ok:>12}{afford:>17}")
    per_on = cap_usd * dp["avg_leverage_on"] / V3.n_positions
    print(f"\n  Chiến lược cần {V3.n_positions} vị thế để trung lập.")
    print(f"  DP đứng NGOÀI thị trường {dp['frac_flat']:.0%} thời gian; khi CÓ vị thế nó chạy")
    print(f"  trung bình {dp['avg_leverage_on']:.2f}x = ${per_on:.2f}/vị thế — ", end="")
    print(f"ĐỦ (min ${V3.min_notional_usd:.0f})." if per_on >= V3.min_notional_usd
          else f"KHÔNG ĐỦ min notional ${V3.min_notional_usd:.0f}.")
    print(f"  (Đừng đọc cột 'L tb' {dp['avg_leverage']:.2f}x như một trạng thái — nó là trung bình")
    print(f"   trộn giữa 0x và các mức >= {lev_floor:.2f}x, và bản thân nó không đặt được lệnh.)")

    # ---- 5. ĐÁNH ĐỔI NGƯỠNG CHÁY ------------------------------------------
    print(f"\n{'─'*90}\n5. ĐÁNH ĐỔI — chọn ngưỡng cháy là quyết định của người bỏ vốn\n{'─'*90}")
    print(f"  {'cháy khi mất':>13}{'trần L':>9}{'P(đạt)':>9}{'P(cháy)':>9}{'trung vị':>10}{'trung bình':>11}")
    print("  " + "-" * 61)
    frontier = []
    for floor in (0.85, 0.80, 0.70, 0.60):
        for mx in (3.0, 5.0):
            sp = replace(spec, ruin_floor=floor, max_leverage=mx)
            pl = solve_goal_dp(tr, sp)
            rr = evaluate_policy(ho, pl, sp, constant_leverages=[],
                                 n_paths=a.paths, block=6).iloc[-1]   # dòng cuối = DP
            print(f"  {1-floor:>12.0%}{mx:>9.1f}{rr['p_target']*100:>8.1f}%{rr['p_ruin']*100:>8.1f}%"
                  f"{rr['median']*100:>9.2f}%{rr['mean']*100:>10.2f}%")
            frontier.append({"ruin_floor": floor, "max_leverage": mx,
                             **{k: float(rr[k]) for k in
                                ("p_target", "p_ruin", "p_loss", "median", "mean", "avg_leverage")}})

    # ---- 6. VỐN CẦN CHO MỤC TIÊU BỀN VỮNG --------------------------------
    print(f"\n{'─'*90}\n6. ĐẢO NGƯỢC CÂU HỎI — cần bao nhiêu VỐN để mục tiêu thành BỀN VỮNG\n{'─'*90}")
    print("  Mục 2-5 trả lời 'xác suất đạt đích TRONG MỘT TUẦN'. Câu hỏi khác, quan trọng")
    print("  hơn, là 'đạt đều đặn thì cần gì'. Tốc độ tăng trưởng bền vững tối đa ở đòn bẩy")
    print("  Kelly là S²/2 mỗi năm — nó KHÔNG phụ thuộc vốn. Vốn chỉ nhân số tiền.")
    print(f"\n  {'Sharpe':>8}{'tuần':>9}{'VND/tuần @1tr':>16}{'vốn cần cho mục tiêu':>23}")
    print("  " + "-" * 56)
    target_vnd = a.capital_vnd * a.target
    for S in (s_ho, s_ho + 0.2, s_ho + 0.4, 1.5, 1.8, 2.25):
        wk = float(np.expm1(S ** 2 / 2 / 52))
        tag = "  <- hiện tại" if abs(S - s_ho) < 1e-9 else ""
        print(f"  {S:>8.2f}{wk*100:>8.2f}%{wk*1_000_000:>15,.0f}{target_vnd/wk:>22,.0f}{tag}")

    print(f"\n  Hai đường tới {target_vnd:,.0f} VND/tuần:")
    need_cap = target_vnd / float(np.expm1(s_ho ** 2 / 2 / 52))
    need_sharpe = float(np.sqrt(2 * 52 * np.log1p(target_vnd / a.capital_vnd)))
    print(f"    A. nâng Sharpe {s_ho:.2f} -> {need_sharpe:.2f}  (x{need_sharpe/s_ho:.2f})")
    print(f"    B. nâng vốn {a.capital_vnd:,.0f} -> {need_cap:,.0f} VND  (x{need_cap/a.capital_vnd:.2f})")

    print(f"\n  Và vốn lớn hơn NỚI một ràng buộc cứng, không chỉ nhân số tiền:")
    print(f"  {'vốn':>13}{'sàn đòn bẩy':>14}{'nửa Kelly':>12}{'chạy được ở mức tối ưu?':>26}")
    print("  " + "-" * 65)
    half_kelly = s_ho / max(vol_ho, 1e-9) / 2.0
    cap_rows = []
    for mult in (1, 2, 3, 5):
        cv = a.capital_vnd * mult
        floor = V3.n_positions * V3.min_notional_usd / (cv / VND_PER_USD)
        ok = floor <= half_kelly
        print(f"  {cv:>13,.0f}{floor:>13.2f}x{half_kelly:>11.2f}x"
              f"{('CÓ' if ok else f'KHÔNG — bị ép {floor:.2f}x'):>26}")
        cap_rows.append({"capital_vnd": cv, "lev_floor": floor,
                         "half_kelly": half_kelly, "feasible": bool(ok)})
    print(f"\n  Ở vốn nhỏ, min notional ÉP đòn bẩy lên trên mức tối ưu — tài khoản không")
    print(f"  được phép thận trọng. Đó là một cách vốn nhỏ tự làm hại mình mà không ai")
    print(f"  thấy trong bảng lợi suất.")

    out = {
        "capital_required_vnd": need_cap,
        "sharpe_required": need_sharpe,
        "capital_feasibility": cap_rows,
        "executable_leverage_floor": lev_floor,
        "target_return": a.target, "days": a.days, "horizon_periods": horizon,
        "capital_vnd": a.capital_vnd, "capital_usd": cap_usd,
        "sharpe_train_fine": s_tr, "sharpe_holdout_fine": s_ho, "ann_vol_holdout": vol_ho,
        "ceiling_constant": c, "ceiling_dynamic": d,
        "dp_in_sample_p": float(pol.p_success(1.0, horizon, 0.0)),
        "holdout_eval": json.loads(res.to_json(orient="records")),
        "zero_edge_stress": json.loads(res0.to_json(orient="records")),
        "frontier": frontier,
        "note": "DP giải trên train, chấm trên holdout. Xác suất chủ yếu đến từ hình học "
                "mục tiêu (đích gần, sàn xa), không phải từ edge — xem mục 3.",
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
