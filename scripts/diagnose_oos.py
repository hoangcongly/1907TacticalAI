#!/usr/bin/env python3
"""
CHẨN ĐOÁN: vì sao Sharpe rơi từ 1.96 (train) xuống 0.71 (holdout)?

Có bốn giả thuyết, và chúng dẫn tới hai kết luận HOÀN TOÀN KHÁC NHAU về việc nên
làm gì tiếp:

  H1. OVERFIT SIÊU THAM SỐ — ta chọn rebal/lookback/ngưỡng theo train.
      -> Nếu đúng: phân phối kết quả holdout qua LƯỚI siêu tham số sẽ tệ đều,
         và cấu hình được chọn không nổi bật hơn các cấu hình khác.

  H2. ĐỔI CHẾ ĐỘ THỊ TRƯỜNG — giai đoạn holdout khác về bản chất.
      -> Nếu đúng: MỌI cấu hình, kể cả hệ thống cũ, đều rơi cùng mức trên holdout.

  H3. UNIVERSE RỘNG PHẢN TÁC DỤNG NGOÀI MẪU — các cặp mới niêm yết nhiễu hơn.
      -> Nếu đúng: universe hẹp giữ được Sharpe tốt hơn trên holdout.

  H4. HOLDOUT QUÁ NGẮN — 219 kỳ, sai số chuẩn của Sharpe ~ sqrt(1/219)*sqrt(PPY).
      -> Nếu đúng: khoảng tin cậy của Sharpe holdout đủ rộng để chứa cả 1.96.

Script này KHÔNG chọn cấu hình. Nó chỉ đo phân phối để biết nên tin gì.
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
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal, build_all_families
from aegis.risk.portfolio import PortfolioSpec, build_weights

MAKER = 0.5
N_POS = 12


def sharpe_ci(r, ppy):
    """Sharpe + khoảng tin cậy 95% (Lo 2002, xấp xỉ chuẩn)."""
    r = r.dropna()
    n = len(r)
    if n < 20:
        return np.nan, np.nan, np.nan
    sd = r.std(ddof=1)
    if sd <= 1e-15:
        return np.nan, np.nan, np.nan
    sr_p = float(r.mean() / sd)
    se = np.sqrt((1 + 0.5 * sr_p ** 2) / n)
    ann = np.sqrt(ppy)
    return sr_p * ann, (sr_p - 1.96 * se) * ann, (sr_p + 1.96 * se) * ann


def main():
    cache = {}

    def data(uf):
        if uf in cache:
            return cache[uf]
        syms = json.load(open(uf))
        panel = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.15)
        funding = load_funding_panel_v2(syms, panel["close"].index)
        panel, funding = align_panel(panel, funding)
        sigs = {n: build_signal(n, panel, funding) for n in SIGNAL_REGISTRY}
        fam = build_all_families(panel, funding)
        old = combine_signals_zscore({
            "funding_carry": signal_funding_carry(funding, 6),
            "momentum_90": signal_momentum(panel["close"], 90),
            "ofi_flow": signal_ofi(panel["ofi"], 6),
            "funding_mom": signal_funding_momentum(funding, 42),
        }, min_coverage=3)
        per = estimate_cost_bps(panel, notional_usd=10.0, bars_per_day=6)
        cache[uf] = (panel, funding, sigs, fam, old, per)
        return cache[uf]

    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]

    def evaluate(uf, which, rebal, lookback, thr, mode, part):
        """
        [SỬA LỖI ĐO LƯỜNG] Tính nhân quả trên TOÀN dòng thời gian rồi mới cắt lợi
        suất. Cắt lưới trước khiến tầng gộp thích ứng khởi động lại ở đầu holdout —
        một handicap mà live không bao giờ gặp. Sai lệch đo được: 0.71 so với 1.28.
        """
        panel, funding, sigs, fam, old, per = data(uf)
        cf = panel["close"]
        mask = cf.index < split if part == "train" else cf.index >= split
        marks = cf.index[::rebal]
        if len(marks) < 60:
            return None
        close = cf.reindex(marks)
        ppy = 24 * 365.0 / (4 * rebal)

        if which == "old4":
            sig = old.reindex(marks)
        elif which == "fam_eq":
            sig = combine_signals_zscore({k: v.reindex(marks) for k, v in fam.items()}, min_coverage=2)
        elif which == "fam_ad":
            sig = combine_adaptive({k: v.reindex(marks) for k, v in fam.items()}, close,
                                   CombinerSpec(lookback=lookback, min_periods=120,
                                                t_threshold=thr, max_abs_weight=0.20,
                                                max_step=0.05), top_frac=0.10)
        else:
            sig = combine_adaptive({k: v.reindex(marks) for k, v in sigs.items()}, close,
                                   CombinerSpec(lookback=lookback, min_periods=120,
                                                t_threshold=thr, max_abs_weight=0.20,
                                                max_step=0.05), top_frac=0.10)

        spec = PortfolioSpec(mode=mode, n_positions=N_POS,
                             max_weight=1.0 if mode == "rank_binary" else 0.20,
                             beta_neutral=False)
        cost = CostModel(maker_ratio=MAKER, half_spread_bps=0.0,
                         per_symbol_bps=per * (1 - MAKER), min_bps=0.5)
        W = build_weights(sig, close, spec)
        res = simulate(W, close, funding.reindex(marks), cost,
                       bar_hours=4 * rebal, rebalance_every=1)

        keep = np.isin(res.returns.index, cf.index[mask])
        for f in ("returns", "gross_returns", "cost_drag", "funding_pnl",
                  "turnover", "n_positions", "net_exposure", "gross_exposure"):
            setattr(res, f, getattr(res, f)[keep])
        if res.returns.dropna().shape[0] < 40:
            return None

        sr, lo, hi = sharpe_ci(res.returns, ppy)
        s = res.stats(ppy)
        return {"sharpe": sr, "lo": lo, "hi": hi, "ann": s["ann_return"] * 100,
                "dd": s["max_dd"] * 100, "n": s["n"]}

    # ---------- H4: khoảng tin cậy ----------
    print("=" * 100)
    print("H4 — HOLDOUT CÓ ĐỦ DÀI ĐỂ KẾT LUẬN KHÔNG?")
    print("=" * 100)
    r = evaluate("artifacts/universe_wide.json", "all_ad", 18, 500, 2.0, "zscore_riskparity", "holdout")
    t = evaluate("artifacts/universe_wide.json", "all_ad", 18, 500, 2.0, "zscore_riskparity", "train")
    print(f"  TRAIN   sharpe {t['sharpe']:.2f}  KTC95 [{t['lo']:.2f}, {t['hi']:.2f}]  ({t['n']:.0f} kỳ)")
    print(f"  HOLDOUT sharpe {r['sharpe']:.2f}  KTC95 [{r['lo']:.2f}, {r['hi']:.2f}]  ({r['n']:.0f} kỳ)")
    overlap = r["hi"] >= t["lo"]
    print(f"  Hai khoảng {'CÓ' if overlap else 'KHÔNG'} chồng lấn -> "
          f"{'chênh lệch CHƯA có ý nghĩa thống kê' if overlap else 'chênh lệch là THẬT'}")

    # ---------- H1: phân phối qua lưới siêu tham số ----------
    print("\n" + "=" * 100)
    print("H1 — PHÂN PHỐI KẾT QUẢ HOLDOUT QUA LƯỚI SIÊU THAM SỐ")
    print("(nếu chỉ vài cấu hình sống, đó là may mắn; nếu cả lưới dương, đó là edge)")
    print("=" * 100)
    rows = []
    for rebal in (12, 18, 24, 36):
        for lookback in (250, 500):
            for thr in (1.0, 2.0):
                for mode in ("rank_binary", "zscore_riskparity"):
                    o = evaluate("artifacts/universe_wide.json", "all_ad", rebal, lookback,
                                 thr, mode, "holdout")
                    if o:
                        rows.append({"rebal": rebal, "lookback": lookback, "thr": thr,
                                     "mode": mode, **o})
    g = pd.DataFrame(rows)
    if not g.empty:
        print(f"  {len(g)} cấu hình | Sharpe holdout: trung vị {g['sharpe'].median():.2f}  "
              f"trung bình {g['sharpe'].mean():.2f}  min {g['sharpe'].min():.2f}  "
              f"max {g['sharpe'].max():.2f}")
        print(f"  tỷ lệ cấu hình có Sharpe > 0: {(g['sharpe']>0).mean()*100:.0f}%  |  "
              f"> 0.5: {(g['sharpe']>0.5).mean()*100:.0f}%  |  > 1.0: {(g['sharpe']>1).mean()*100:.0f}%")
        print(f"  lợi suất năm trung vị: {g['ann'].median():.1f}%")
        print("\n  theo chu kỳ tái cân bằng:")
        print(g.groupby("rebal")[["sharpe", "ann"]].median().round(2).to_string())

    # ---------- H2 + H3 ----------
    print("\n" + "=" * 100)
    print("H2/H3 — SO TRAIN vs HOLDOUT CHO TỪNG BỘ TÍN HIỆU VÀ TỪNG UNIVERSE")
    print("=" * 100)
    print(f"{'universe':<8}{'tín hiệu':<12}{'rebal':>6}{'sharpe train':>14}"
          f"{'sharpe holdout':>16}{'giữ lại':>10}{'ann OOS':>10}")
    print("-" * 78)
    for uf, tag in (("artifacts/universe_filtered.json", "hẹp"),
                    ("artifacts/universe_wide.json", "rộng")):
        for which in ("old4", "fam_eq", "fam_ad", "all_ad"):
            for rebal in (18, 24):
                a = evaluate(uf, which, rebal, 500, 2.0, "zscore_riskparity", "train")
                b = evaluate(uf, which, rebal, 500, 2.0, "zscore_riskparity", "holdout")
                if not a or not b:
                    continue
                keep = b["sharpe"] / a["sharpe"] * 100 if a["sharpe"] > 0 else 0.0
                print(f"{tag:<8}{which:<12}{rebal:>6}{a['sharpe']:>14.2f}"
                      f"{b['sharpe']:>16.2f}{keep:>9.0f}%{b['ann']:>9.1f}%")

    if not g.empty:
        g.to_csv("artifacts/oos_grid.csv", index=False)
        print("\n-> artifacts/oos_grid.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
