#!/usr/bin/env python3
"""
CHỌN CẤU HÌNH THEO ĐỘ ỔN ĐỊNH, KHÔNG THEO ĐỈNH CAO NHẤT.

Sau khi thử hàng trăm cấu hình trên tập train, cấu hình có Sharpe cao nhất gần như
chắc chắn là cấu hình MAY MẮN NHẤT, không phải cấu hình tốt nhất. Với 100 phép thử
trên nhiễu thuần tuý, Sharpe lớn nhất kỳ vọng đã vào khoảng 2.5.

Cách chọn ở đây, theo đúng chuẩn thực hành:

  1. Chia train thành nhiều GIAI ĐOẠN CON liên tiếp.
  2. Chấm mỗi cấu hình bằng Sharpe TRUNG VỊ qua các giai đoạn và bằng SỐ GIAI ĐOẠN
     DƯƠNG, chứ không bằng Sharpe toàn mẫu.
  3. Ưu tiên cấu hình nằm giữa một VÙNG PHẲNG của lưới tham số: hàng xóm của nó
     cũng phải tốt. Đỉnh nhọn cô độc là dấu hiệu overfit, không phải dấu hiệu edge.

Holdout vẫn KHÔNG được chạm ở script này.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.data.panel_v2 import load_panel_v2, load_funding_panel_v2, align_panel
from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.backtest_v2 import CostModel, estimate_cost_bps, simulate
from aegis.research.cross_sectional import (
    combine_signals_zscore, signal_funding_carry, signal_funding_momentum,
    signal_momentum, signal_ofi,
)
from aegis.research.signal_library import SIGNAL_REGISTRY, build_all_families, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights

UNIVERSE = "artifacts/universe_wide.json"
N_POS = 12
MAKER = 0.5
N_FOLDS = 5


def sharpe(r, ppy):
    r = r.dropna()
    if len(r) < 15:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 1e-15 else np.nan


def main():
    syms = json.load(open(UNIVERSE))
    panel = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]
    mask = panel["close"].index < split
    per_sym = estimate_cost_bps(panel, notional_usd=20.0, bars_per_day=6)
    cost = CostModel(maker_ratio=MAKER, half_spread_bps=0.0,
                     per_symbol_bps=per_sym * (1 - MAKER), min_bps=1.0)
    close_full = panel["close"]

    # --- các ứng viên tín hiệu ---
    print("dựng tín hiệu...", flush=True)
    old4 = combine_signals_zscore({
        "funding_carry": signal_funding_carry(funding, 6),
        "momentum_90": signal_momentum(close_full, 90),
        "ofi_flow": signal_ofi(panel["ofi"], 6),
        "funding_mom": signal_funding_momentum(funding, 42),
    }, min_coverage=3)
    fam_full = build_all_families(panel, funding)
    all_sig_full = {n: build_signal(n, panel, funding) for n in SIGNAL_REGISTRY}

    rows = []
    for rebal in (12, 18, 24, 36):
        marks = close_full.index[::rebal]
        marks = marks[np.isin(marks, close_full.index[mask])]
        if len(marks) < 300:
            continue
        close = close_full.reindex(marks)
        fund = funding.reindex(marks)
        ppy = 24 * 365.0 / (4 * rebal)

        fam = {k: v.reindex(marks) for k, v in fam_full.items()}
        allsig = {k: v.reindex(marks) for k, v in all_sig_full.items()}
        cs = CombinerSpec(lookback=500, min_periods=90, t_threshold=1.5)

        candidates = {
            "4 tín hiệu cũ": old4.reindex(marks),
            "5 họ, gộp đều": combine_signals_zscore(fam, min_coverage=2),
            "5 họ, thích ứng": combine_adaptive(fam, close, cs, top_frac=0.10),
            "26 tín hiệu, thích ứng": combine_adaptive(
                allsig, close, CombinerSpec(lookback=500, min_periods=120,
                                            t_threshold=2.0, max_abs_weight=0.20,
                                            max_step=0.05), top_frac=0.10),
        }

        for mode in ("rank_binary", "zscore_riskparity"):
            spec = PortfolioSpec(mode=mode, n_positions=N_POS,
                                 max_weight=1.0 if mode == "rank_binary" else 0.20,
                                 beta_neutral=False)
            for name, sig in candidates.items():
                try:
                    W = build_weights(sig, close, spec)
                    res = simulate(W, close, fund, cost, bar_hours=4 * rebal, rebalance_every=1)
                except Exception as exc:
                    print(f"  lỗi {name}/{mode}/{rebal}: {type(exc).__name__}")
                    continue
                r = res.returns.dropna()
                if len(r) < 200:
                    continue
                folds = np.array_split(r, N_FOLDS)
                fs = [sharpe(pd.Series(f), ppy) for f in folds]
                fs = [x for x in fs if np.isfinite(x)]
                s = res.stats(ppy)
                rows.append({
                    "tín hiệu": name, "trọng số": mode, "rebal": rebal,
                    "sharpe_toàn": s["sharpe"], "ann": s["ann_return"] * 100,
                    "maxDD": s["max_dd"] * 100,
                    "sharpe_trung_vị_fold": float(np.median(fs)) if fs else np.nan,
                    "sharpe_min_fold": float(np.min(fs)) if fs else np.nan,
                    "fold_dương": int(np.sum(np.array(fs) > 0)),
                    "n_fold": len(fs),
                })
        print(f"  xong rebal={rebal}", flush=True)

    df = pd.DataFrame(rows)
    df["điểm"] = df["sharpe_trung_vị_fold"] * (df["fold_dương"] / df["n_fold"])

    print("\n" + "=" * 112)
    print(f"ĐỘ ỔN ĐỊNH QUA {N_FOLDS} GIAI ĐOẠN CON CỦA TRAIN ({N_POS} vị thế, phí thật, maker {MAKER:.0%})")
    print("=" * 112)
    show = df.sort_values("điểm", ascending=False)
    print(f"{'tín hiệu':<24}{'trọng số':<20}{'rebal':>6}{'Sharpe toàn':>12}"
          f"{'trung vị fold':>14}{'min fold':>10}{'fold+':>7}{'ann%':>8}{'maxDD%':>8}{'điểm':>7}")
    print("-" * 112)
    for _, r in show.iterrows():
        print(f"{r['tín hiệu']:<24}{r['trọng số']:<20}{int(r['rebal']):>6}{r['sharpe_toàn']:>12.2f}"
              f"{r['sharpe_trung_vị_fold']:>14.2f}{r['sharpe_min_fold']:>10.2f}"
              f"{int(r['fold_dương']):>4}/{int(r['n_fold'])}{r['ann']:>8.1f}{r['maxDD']:>8.1f}"
              f"{r['điểm']:>7.2f}")

    print("\nTỔNG HỢP THEO TÍN HIỆU (trung bình qua mọi rebal & chế độ trọng số):")
    print(df.groupby("tín hiệu")[["sharpe_toàn", "sharpe_trung_vị_fold", "sharpe_min_fold", "điểm"]]
          .mean().round(2).sort_values("điểm", ascending=False).to_string())
    print("\nTỔNG HỢP THEO CHU KỲ TÁI CÂN BẰNG:")
    print(df.groupby("rebal")[["sharpe_toàn", "sharpe_trung_vị_fold", "điểm"]].mean().round(2).to_string())
    print("\nTỔNG HỢP THEO CHẾ ĐỘ TRỌNG SỐ:")
    print(df.groupby("trọng số")[["sharpe_toàn", "sharpe_trung_vị_fold", "điểm"]].mean().round(2).to_string())

    df.to_csv("artifacts/stability.csv", index=False)
    print("\n-> artifacts/stability.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
