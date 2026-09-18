#!/usr/bin/env python3
"""
Giải chính sách đòn bẩy theo mục tiêu MỘT LẦN và ghi ra đĩa cho đường chạy live dùng.

Live tuyệt đối không được tự giải DP mỗi lượt: chính sách sẽ trôi theo dữ liệu mới mà
không ai duyệt, và khi mổ xẻ sự cố sẽ không ai trả lời được "hôm đó hệ thống dùng chính
sách nào". Giải ở đây, ghi ra `artifacts/goal_policy.npz`, rồi live chỉ tra bảng.

Chính sách được giải trên tập TRAIN. Holdout dùng để CHẤM (`scripts/goal_plan.py`),
không dùng để chọn — nếu giải trên holdout thì con số holdout thành con số train.

    python scripts/build_goal_policy.py
    python scripts/build_goal_policy.py --target 0.05 --days 7 --floor 0.70 --max-lev 5
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.research.goal_dp import GoalSpec, evaluate_policy, solve_goal_dp
from aegis.research.strategy_v3 import V3
from aegis.risk.goal_overlay import DEFAULT_POLICY_PATH, save_policy

FINE_CSV = "artifacts/returns_v3_fine.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--floor", type=float, default=0.70)
    ap.add_argument("--max-lev", type=float, default=5.0)
    ap.add_argument("--capital-usd", type=float, default=None,
                    help="mặc định lấy từ StrategyV3Config (1.000.000 VND ~ $38)")
    ap.add_argument("--out", default=DEFAULT_POLICY_PATH)
    a = ap.parse_args(argv)

    cap = a.capital_usd if a.capital_usd is not None else V3.capital_usd
    horizon = int(round(a.days * 24 / V3.bar_hours))
    lev_floor = GoalSpec.executable_floor(cap, V3.n_positions, V3.min_notional_usd)

    df = pd.read_csv(FINE_CSV)
    tr = df[df.segment == "train"]["net_return"].dropna().to_numpy()
    ho = df[df.segment == "holdout"]["net_return"].dropna().to_numpy()

    spec = GoalSpec(target_return=a.target, horizon_periods=horizon,
                    ruin_floor=a.floor, max_leverage=a.max_lev,
                    min_executable_leverage=lev_floor)
    print(f"giải DP: mục tiêu +{a.target:.0%} trong {a.days:.0f} ngày "
          f"({horizon} kỳ {V3.bar_hours:.0f}h), sàn cháy {a.floor:.2f}, "
          f"lưới đòn bẩy {{0}} ∪ [{lev_floor:.2f}, {a.max_lev:.1f}]", flush=True)

    pol = solve_goal_dp(tr, spec)
    path = save_policy(pol, a.out)

    res = evaluate_policy(ho, pol, spec, constant_leverages=[V3.gross_leverage], n_paths=20000)
    dp = res.iloc[-1]
    base = res.iloc[0]
    print(f"\nCHẤM trên holdout (20.000 đường):")
    print(f"  {base['policy']:<16} P(đạt)={base['p_target']:6.1%}  P(cháy)={base['p_ruin']:5.1%}")
    print(f"  {dp['policy']:<16} P(đạt)={dp['p_target']:6.1%}  P(cháy)={dp['p_ruin']:5.1%}"
          f"  (ngoài thị trường {dp['frac_flat']:.0%} thời gian, "
          f"{dp['avg_leverage_on']:.2f}x khi vào)")
    print(f"\n-> {path}")
    print(f"-> {path.replace('.npz', '.meta.json')}")
    print(f"\nBật ở live bằng cách đặt goal_overlay trong LiveConfig "
          f"(mặc định derisk_only=True — chỉ được GIẢM rủi ro).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
