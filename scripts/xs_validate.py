#!/usr/bin/env python3
"""
KIỂM ĐỊNH HOLDOUT — CHẠY ĐÚNG MỘT LẦN.

Cấu hình đã chốt hoàn toàn trên tập train. Script này mở phần dữ liệu chưa từng
được nhìn. Dù kết quả thế nào cũng KHÔNG được quay lại chỉnh rồi chạy lại — làm vậy
là biến holdout thành train và mất sạch giá trị kiểm định.
"""
import json, sys, warnings
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from xs_develop import build_all, stats, INTERVAL, REBAL, TOP_FRAC, COST
from aegis.research.cross_sectional import backtest_cross_sectional

# ---- CẤU HÌNH CHỐT (quyết định 100% trên train, không được đổi) ----
SIGNALS = ["funding_carry", "momentum_90", "ofi_flow", "funding_mom"]
COMBINE = "equal_weight"


def run(part):
    sigs, close, funding = build_all(part)
    rets = {n: backtest_cross_sectional(sigs[n], close, funding, INTERVAL, REBAL,
                                        top_frac=TOP_FRAC, cost_rate=COST).returns
            for n in SIGNALS}
    R = pd.DataFrame(rets).dropna()
    return R, R.mean(axis=1)


def main():
    print("CẤU HÌNH CHỐT:", ", ".join(SIGNALS))
    print(f"  gộp={COMBINE} | top_frac={TOP_FRAC} | tái cân bằng={REBAL} nến ({REBAL*4/24:.0f} ngày)")
    print(f"  khung={INTERVAL} | phí=maker {COST}\n")

    R_tr, eq_tr = run("train")
    R_ho, eq_ho = run("holdout")
    s_tr, s_ho = stats(eq_tr), stats(eq_ho)

    print("=" * 68)
    print(f"{'':<14}{'kỳ':>7}{'ann':>9}{'vol':>8}{'sharpe':>8}{'t-stat':>8}{'maxDD':>8}")
    print("-" * 68)
    for label, s in [("TRAIN", s_tr), ("HOLDOUT", s_ho)]:
        print(f"{label:<14}{s['n']:>7.0f}{s['ann']*100:>8.1f}%{s['vol']*100:>7.1f}%"
              f"{s['sharpe']:>8.2f}{s['t']:>8.2f}{s['dd']*100:>7.1f}%")
    print("=" * 68)

    decay = s_ho["sharpe"] / s_tr["sharpe"] if s_tr["sharpe"] > 0 else 0.0
    print(f"\nSharpe giữ lại được: {decay*100:.0f}% so với train")

    print("\n=== TỪNG TÍN HIỆU TRÊN HOLDOUT ===")
    print(f"{'tín hiệu':<16}{'ann':>9}{'sharpe':>8}{'t-stat':>8}")
    print("-" * 41)
    for c in R_ho.columns:
        s = stats(R_ho[c])
        print(f"{c:<16}{s['ann']*100:>8.1f}%{s['sharpe']:>8.2f}{s['t']:>8.2f}")

    print("\n=== HOLDOUT THEO NĂM ===")
    yr = pd.to_datetime(eq_ho.index, unit="ms").year
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() < 30:
            continue
        print(f"  {y}: {(np.prod(1+eq_ho[m].to_numpy())-1)*100:+7.1f}%   ({m.sum()} kỳ)")

    print("\n" + "=" * 68)
    if s_ho["t"] > 2 and s_ho["ann"] > 0:
        verdict = "✅ ĐẠT — edge sống sót trên dữ liệu chưa từng nhìn"
    elif s_ho["ann"] > 0:
        verdict = f"🟡 DƯƠNG nhưng t={s_ho['t']:.2f} < 2 — chưa đủ mạnh để khẳng định"
    else:
        verdict = "❌ THẤT BẠI — edge không sống sót ngoài mẫu"
    print("KẾT LUẬN:", verdict)

    json.dump({"signals": SIGNALS, "combine": COMBINE, "top_frac": TOP_FRAC,
               "rebalance_every": REBAL, "interval": INTERVAL, "cost_rate": COST,
               "train": {k: float(v) for k, v in s_tr.items()},
               "holdout": {k: float(v) for k, v in s_ho.items()}},
              open("artifacts/strategy_validated.json", "w"), indent=2)
    print("kết quả -> artifacts/strategy_validated.json")


if __name__ == "__main__":
    main()
