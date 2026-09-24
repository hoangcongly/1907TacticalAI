#!/usr/bin/env python3
"""
PHÁN QUYẾT CUỐI: chiến lược có vượt được nhiễu chọn lọc không?

Đây là bước sau `scripts/trial_sweep.py`. Sweep đo `variance_of_srs`; script này
lắp nó vào Deflated Sharpe Ratio để trả lời câu hỏi mà Sharpe trần không trả lời
được: "Sharpe 1.284 trên holdout có LỚN HƠN thứ mà 250 lần thử tự sinh ra do may
rủi hay không?"

VÌ SAO KHÔNG NHÌN SHARPE TRẦN: thử 250 cấu hình rồi chọn cái tốt nhất thì cái tốt
nhất LUÔN trông đẹp, kể cả khi mọi cấu hình đều vô dụng. Với 250 lần thử, Sharpe
cao nhất kỳ vọng do may rủi đã là một số dương đáng kể. DSR trừ đúng phần đó ra.

MỘT ĐIỀU SCRIPT NÀY LÀM MÀ BẢN TRƯỚC KHÔNG: `variance_of_srs` tự nó là một ƯỚC
LƯỢNG từ n lần thử, nên nó có sai số lấy mẫu. Báo cáo một DSR duy nhất từ một ước
lượng điểm là giấu mất phần bất định đó. Ở đây ta dựng khoảng tin cậy chi-bình
phương cho phương sai rồi tính DSR ở cả hai đầu — nếu kết luận đổi chiều trong
khoảng đó thì kết luận chưa đứng được.

    python scripts/dsr_report.py
    python scripts/dsr_report.py --trials 500   # xem độ nhạy theo số lần thử
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from aegis.validation.dsr import (
    compute_deflated_sharpe_ratio, compute_probabilistic_sharpe_ratio,
)

RETURNS = "artifacts/returns_v3.csv"
META = "artifacts/returns_v3.meta.json"
TRIALS = "artifacts/trial_sharpes.json"
APPROVAL = 0.95


def _seg(df: pd.DataFrame, name: str):
    r = df[df.segment == name]["net_return"].values.astype(float)
    return r[np.isfinite(r)]


def _fmt(label: str, dsr: float) -> str:
    mark = "✅ ĐẠT" if dsr >= APPROVAL else "❌ CHƯA ĐẠT"
    return f"{label:<34}{dsr:>8.4f}   {mark}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None,
                    help="Ghi đè số lần thử; mặc định lấy num_trials_declared.")
    a = ap.parse_args(argv)

    if not pathlib.Path(TRIALS).exists():
        print(f"THIẾU {TRIALS} — chạy `python scripts/trial_sweep.py` trước.")
        return 1

    df = pd.read_csv(RETURNS)
    meta = json.load(open(META))
    tr = json.load(open(TRIALS))
    ppy = meta["periods_per_year"]
    n_trials = a.trials or meta["config"]["num_trials_declared"]

    var_srs = tr["variance_of_srs_per_period"]
    n_tr = tr["n_trials_ok"]

    print("=" * 78)
    print("PHÁN QUYẾT DEFLATED SHARPE — dùng variance_of_srs ĐO ĐƯỢC")
    print("=" * 78)
    print(f"  Nguồn phương sai   : {n_tr} lần thử thật (seed {tr['seed']}, "
          f"vintage {tr['vintage_ms']})")
    print(f"  variance_of_srs    : {var_srs:.6f}  (theo kỳ)")
    st = tr["sharpe_train_annual"]
    print(f"  Sharpe train các lần thử (năm hoá): "
          f"TB {st['mean']:.3f} | đlc {st['std']:.3f} | "
          f"[{st['min']:.3f} .. {st['max']:.3f}]")
    print(f"  V3 đã chốt         : train {tr['v3_baseline']['sharpe_train']:.3f} | "
          f"holdout {tr['v3_baseline']['sharpe_holdout']:.3f}")
    print(f"  Số lần thử khai báo: {n_trials}")

    # -- khoảng tin cậy 95% cho phương sai (chi-bình phương) -----------------
    dfree = n_tr - 1
    lo = dfree * var_srs / stats.chi2.ppf(0.975, dfree)
    hi = dfree * var_srs / stats.chi2.ppf(0.025, dfree)
    print(f"  KTC 95% của var_srs: [{lo:.6f} .. {hi:.6f}]")

    for seg in ("train", "holdout"):
        r = _seg(df, seg)
        sr_p = r.mean() / r.std(ddof=1)
        sk = float(pd.Series(r).skew())
        ku = float(pd.Series(r).kurt() + 3.0)
        psr = compute_probabilistic_sharpe_ratio(sr_p, 0.0, len(r), sk, ku)

        print("\n" + "-" * 78)
        print(f"{seg.upper()}  (n={len(r)} kỳ ≈ {len(r)*72/24/365:.1f} năm)")
        print("-" * 78)
        print(f"  Sharpe năm hoá     : {sr_p*np.sqrt(ppy):.4f}")
        print(f"  t-stat             : {sr_p*np.sqrt(len(r)):.4f}")
        print(f"  PSR (SR > 0)       : {psr:.4f}   "
              f"<- xác suất edge THẬT SỰ dương")

        base = compute_deflated_sharpe_ratio(
            sr_estimated=sr_p, variance_of_srs=var_srs, sample_length=len(r),
            num_trials=n_trials, skewness=sk, kurtosis=ku)
        thr = base["sr_expected_max"] * np.sqrt(ppy)
        print(f"  Ngưỡng do {n_trials} lần thử: {thr:.4f} (năm hoá)  "
              f"<- Sharpe may rủi kỳ vọng")
        print()
        print(f"  {'DSR (var_srs đo được)':<32}{base['dsr']:>8.4f}   "
              f"{'✅ ĐẠT' if base['dsr'] >= APPROVAL else '❌ CHƯA ĐẠT'}")
        for lbl, v in (("DSR ở cận DƯỚI KTC var", lo), ("DSR ở cận TRÊN KTC var", hi)):
            d = compute_deflated_sharpe_ratio(
                sr_estimated=sr_p, variance_of_srs=v, sample_length=len(r),
                num_trials=n_trials, skewness=sk, kurtosis=ku)["dsr"]
            print(f"  {_fmt(lbl, d)}")

        verdict = "VƯỢT" if sr_p * np.sqrt(ppy) > thr else "KHÔNG VƯỢT"
        print(f"\n  => Sharpe {sr_p*np.sqrt(ppy):.3f} {verdict} ngưỡng may rủi {thr:.3f}")

    print("\n" + "=" * 78)
    print("ĐỌC KẾT QUẢ")
    print("=" * 78)
    print("  PSR cao + DSR thấp  = edge CÓ THẬT nhưng chưa đủ lớn để tách khỏi")
    print("                        lợi thế do thử-rồi-chọn. Không phải 'vô dụng'.")
    print("  DSR đổi chiều trong KTC = số lần thử còn quá ít để kết luận đứng vững.")
    print("  Cách nâng DSR: giảm số cấu hình thử, hoặc kéo dài holdout —")
    print("                 KHÔNG phải bằng cách thử thêm cấu hình.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
