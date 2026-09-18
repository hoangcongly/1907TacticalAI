#!/usr/bin/env python3
"""
HỌ TÍN HIỆU VỊ THẾ có đáng đưa vào chiến lược không — chấm trên TRAIN, không đụng holdout.

26 tín hiệu hiện có đều dựng từ GIÁ, KHỐI LƯỢNG và FUNDING. Không cái nào nhìn thấy ai
đang cầm gì. Họ `positioning` (10 tín hiệu) lấp chỗ đó bằng open interest và tỷ lệ
long/short từ kho dump Binance — xem `scripts/download_metrics.py`.

Script này trả lời ba câu, theo thứ tự, và DỪNG nếu câu trước không đạt:

  1. Từng tín hiệu có chênh lệch decile đáng kể không? Ngưỡng thực dụng: |t| >= 2,0 —
     chính là ngưỡng mà `adaptive_combiner` dùng để cấp trọng số. Dưới ngưỡng đó, thêm
     vào cũng chỉ là thêm một cột trọng số 0.
  2. Họ này có TRỰC GIAO với 5 họ cũ không? Một tín hiệu mạnh nhưng trùng lặp chỉ làm
     tăng turnover. Tương quan hạng mặt cắt ngang trả lời.
  3. Thêm vào thì chiến lược đầu-cuối tốt lên hay xấu đi, đo theo ĐỘ ỔN ĐỊNH qua các
     giai đoạn con — không theo Sharpe toàn cục.

KỶ LUẬT: mọi quyết định chỉ nhìn dữ liệu TRƯỚC `holdout_split.json`. Holdout in ra ở
cuối CHỈ để xem, sau khi mọi lựa chọn đã chốt — và nó đã được dùng 2 lần cho v3 nên
con số đó không còn là bằng chứng, chỉ là một lần nhìn.

    python scripts/positioning_study.py
"""
import argparse
import json
import sys
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.research.ic_analysis import decile_spread
from aegis.research.signal_library import SIGNAL_REGISTRY, build_family, build_signal
from aegis.research.strategy_v3 import (
    V3, V3_SIGNALS, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)

OUT = "artifacts/positioning_study.json"


def _folds(r, ppy, k=8):
    b = np.linspace(0, len(r), k + 1).astype(int)
    out = []
    for i in range(k):
        x = r.iloc[b[i]:b[i + 1]].dropna()
        out.append(np.nan if len(x) < 10 or x.std(ddof=1) < 1e-15
                   else float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-threshold", type=float, default=2.0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)

    pos_names = [n for n, s in SIGNAL_REGISTRY.items() if s.family == "positioning"]
    all_names = tuple(V3_SIGNALS) + tuple(pos_names)
    cfg_ext = replace(V3, signals=all_names)

    data = load_v3_data(cfg_ext, end_ms=V3_VINTAGE_MS, with_metrics=True)
    ppy = V3.periods_per_year
    marks = data.close.index[::V3.rebalance_every]
    close_m = data.close.reindex(marks)
    split = data.split_ts
    train_mask = marks < split

    cov = data.panel["sum_open_interest_value"].notna()
    n_cov = int(cov.any().sum())
    first = cov.any(axis=1)
    t0 = marks[marks >= (first[first].index[0] if first.any() else marks[0])]
    print(f"\ndữ liệu vị thế: {n_cov}/{data.close.shape[1]} cặp, "
          f"từ {pd.to_datetime(t0[0], unit='ms'):%Y-%m}")
    n_train = int(((marks >= t0[0]) & train_mask).sum())
    print(f"cửa sổ TRAIN dùng được: {n_train} kỳ "
          f"({pd.to_datetime(t0[0], unit='ms'):%Y-%m} -> {pd.to_datetime(split, unit='ms'):%Y-%m})")

    # ---- 1. Từng tín hiệu ---------------------------------------------------
    print("\n" + "=" * 86)
    print(f"1. CHÊNH LỆCH DECILE TỪNG TÍN HIỆU — chỉ TRAIN (ngưỡng |t| >= {a.t_threshold})")
    print("=" * 86)
    print(f"{'tín hiệu':<22}{'họ':<15}{'kỳ':>6}{'chênh decile':>14}{'t-stat':>9}{'Sharpe/năm':>12}")
    print("-" * 86)

    rows = []
    for n in pos_names + ["carry_level", "mom_slow", "ofi_mid"]:
        sig = data.signals[n] if n in data.signals else build_signal(n, data.panel, data.funding)
        sm = sig.reindex(marks)
        sm = sm[sm.index < split]
        r = decile_spread(sm, close_m, horizon=1, top_frac=V3.top_frac)
        sh = r.get("sharpe_per_period", 0.0) * np.sqrt(ppy)
        fam = SIGNAL_REGISTRY[n].family
        tag = "" if fam == "positioning" else "   <- mốc so sánh"
        print(f"{n:<22}{fam:<15}{r['n']:>6.0f}{r['mean']*100:>13.3f}%"
              f"{r.get('t_stat', 0):>9.2f}{sh:>12.2f}{tag}")
        rows.append({"name": n, "family": fam, **{k: float(v) for k, v in r.items()}})

    strong = [x for x in rows if x["family"] == "positioning"
              and abs(x.get("t_stat", 0)) >= a.t_threshold]
    print(f"\n=> {len(strong)}/{len(pos_names)} tín hiệu vị thế vượt |t| >= {a.t_threshold}"
          + (f": {', '.join(x['name'] for x in strong)}" if strong else ""))

    # ---- 2. Trực giao -------------------------------------------------------
    print("\n" + "=" * 86)
    print("2. TRỰC GIAO — tương quan hạng mặt cắt ngang với các họ cũ (trung vị qua thời gian)")
    print("=" * 86)
    fam_pos = build_family("positioning", data.panel, data.funding).reindex(marks)
    fam_pos = fam_pos[fam_pos.index < split]
    corrs = {}
    for fam in ("carry", "momentum", "flow", "volatility", "microstructure"):
        f = build_family(fam, data.panel, data.funding).reindex(fam_pos.index)
        cs = [fam_pos.loc[t].corr(f.loc[t], method="spearman") for t in fam_pos.index
              if fam_pos.loc[t].notna().sum() > 10 and f.loc[t].notna().sum() > 10]
        corrs[fam] = float(np.nanmedian(cs)) if cs else float("nan")
        print(f"  positioning vs {fam:<16} {corrs[fam]:+.3f}")
    mx = max(abs(v) for v in corrs.values() if np.isfinite(v))
    print(f"\n=> |tương quan| lớn nhất {mx:.3f} — "
          f"{'TRỰC GIAO, bổ trợ được' if mx < 0.30 else 'TRÙNG LẶP, ít giá trị thêm'}")

    # ---- 3. Chiến lược đầu-cuối --------------------------------------------
    print("\n" + "=" * 86)
    print("3. CHIẾN LƯỢC ĐẦU-CUỐI — 26 tín hiệu so với 26+10, chấm theo ĐỘ ỔN ĐỊNH")
    print("=" * 86)
    print(f"{'bộ tín hiệu':<26}{'kỳ':>6}{'sharpe':>8}{'ann':>8}{'tr.vị fold':>11}"
          f"{'%dương':>8}{'tệ nhất':>9}{'turnover':>10}")
    print("-" * 86)

    results = {}
    for label, names in (("26 (đã kiểm định)", V3_SIGNALS), ("26 + 10 vị thế", all_names)):
        cfg = replace(V3, signals=tuple(names))
        sig = combined_signal(data, cfg)
        res = run_v3(data, cfg, sig=sig)
        r = res.returns.dropna()
        # CHỈ tính trên cửa sổ có dữ liệu vị thế, nếu không ta so hai thứ khác nhau.
        r = r[(r.index >= t0[0]) & (r.index < split)]
        f = _folds(r, ppy, k=6)
        results[label] = {
            "n": len(r), "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(ppy)),
            "ann": float(r.mean() * ppy), "fold_median": float(np.nanmedian(f)),
            "frac_pos": float(np.mean([x > 0 for x in f if np.isfinite(x)])),
            "worst": float(np.nanmin(f)),
            "turnover": float(res.turnover.reindex(r.index).dropna().mean()),
        }
        v = results[label]
        print(f"{label:<26}{v['n']:>6}{v['sharpe']:>8.2f}{v['ann']*100:>7.1f}%"
              f"{v['fold_median']:>11.2f}{v['frac_pos']*100:>7.0f}%{v['worst']:>9.2f}"
              f"{v['turnover']:>10.3f}")

    base, ext = results["26 (đã kiểm định)"], results["26 + 10 vị thế"]
    d_sh = ext["sharpe"] - base["sharpe"]
    d_med = ext["fold_median"] - base["fold_median"]
    print(f"\n=> chênh Sharpe {d_sh:+.3f} | chênh trung vị fold {d_med:+.3f} | "
          f"turnover {(ext['turnover']/max(base['turnover'],1e-9)-1)*100:+.1f}%")
    verdict = ("ĐÁNG THÊM" if d_med > 0.10 and d_sh > 0.03
               else "KHÔNG ĐÁNG — giữ 26 tín hiệu")
    print(f"=> KẾT LUẬN (trên TRAIN): {verdict}")

    out = {"n_symbols_with_metrics": n_cov, "train_periods": n_train,
           "signals": rows, "orthogonality": corrs, "end_to_end": results,
           "verdict": verdict, "t_threshold": a.t_threshold}
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
