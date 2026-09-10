#!/usr/bin/env python3
"""
BẢNG QUY KẾT — mỗi dòng thêm ĐÚNG MỘT thay đổi, SỐ VỊ THẾ GIỮ CỐ ĐỊNH ~12.

Vì sao phải cố định số vị thế: lần đo đầu tiên của đợt nâng cấp này cho thấy trọng
số liên tục nâng Sharpe từ 1.40 lên 1.85. Kiểm tra lại thì phát hiện chế độ liên tục
đang nắm TOÀN BỘ 59 cặp thay vì 12 — gần như toàn bộ phần tăng đến từ đa dạng hoá
do nắm nhiều vị thế hơn, không phải từ cách đánh trọng số. Với vốn $38 thì 59 vị thế
là bất khả thi (min notional $5/lệnh), nên con số đó vô nghĩa với tài khoản này.

Bài học đã đưa vào code (`risk/portfolio.py`): tập tài sản LUÔN chọn theo thứ hạng
trước, mọi chế độ trọng số chỉ khác nhau ở cách đánh trọng số TRONG tập đó.

Chạy: python scripts/attribution.py
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
from aegis.research.signal_library import build_all_families
from aegis.risk.portfolio import PortfolioSpec, build_weights

REBAL = 24                 # 24 nến 4h = 96h giữa hai lần tái cân bằng
TARGET_POSITIONS = 12      # cố định cho mọi dòng — ràng buộc thật của vốn $38
MAKER_RATIO = 0.5
PPY = 24 * 365.0 / (4 * REBAL)


def load(universe_file):
    syms = json.load(open(universe_file))
    panel = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]
    return panel, funding, panel["close"].index < split


def spec_of(mode, **kw):
    """PortfolioSpec với SỐ VỊ THẾ cố định — ràng buộc thật của tài khoản $38."""
    return PortfolioSpec(mode=mode, n_positions=TARGET_POSITIONS, top_frac=0.10, **kw)


def marks_of(panel, mask):
    idx = panel["close"].index
    m = idx[::REBAL]
    return m[np.isin(m, idx[mask])]


def old_signals(panel, funding):
    c = panel["close"]
    return combine_signals_zscore({
        "funding_carry": signal_funding_carry(funding, 6),
        "momentum_90": signal_momentum(c, 90),
        "ofi_flow": signal_ofi(panel["ofi"], 6),
        "funding_mom": signal_funding_momentum(funding, 42),
    }, min_coverage=3)


def evaluate(panel, funding, mask, sig_full, spec, cost, label, base=None):
    marks = marks_of(panel, mask)
    close = panel["close"].reindex(marks)
    W = build_weights(sig_full.reindex(marks), close, spec)
    res = simulate(W, close, funding.reindex(marks), cost,
                   bar_hours=4 * REBAL, rebalance_every=1)
    s = res.stats(PPY)
    d = f"{s['sharpe'] - base:+7.2f}" if base is not None else ""
    print(f"{label:<44}{s['avg_positions']:>5.0f}{s['ann_return']*100:>8.1f}%"
          f"{s['ann_vol']*100:>7.1f}%{s['sharpe']:>8.2f}{s['t_stat']:>7.2f}"
          f"{s['max_dd']*100:>7.1f}%{s['cost_drag_ann']*100:>8.1f}%{d:>8}")
    return s


def main():
    print("=" * 106)
    print(f"QUY KẾT TỪNG NÂNG CẤP — tập TRAIN, tái cân bằng {REBAL} nến 4h ({4*REBAL}h), "
          f"~{TARGET_POSITIONS} vị thế, phí thật")
    print("=" * 106)
    print(f"{'cấu hình':<44}{'vịthế':>5}{'ann':>9}{'vol':>7}{'sharpe':>8}"
          f"{'t':>7}{'maxDD':>8}{'phí/năm':>9}{'Δ':>8}")
    print("-" * 106)

    # ---- A: hệ thống cũ, universe hẹp, phí lạc quan như nghiên cứu cũ giả định ----
    pn, fn, mn = load("artifacts/universe_filtered.json")
    old_n = old_signals(pn, fn)
    cheap = CostModel(taker_fee_bps=1.0, maker_fee_bps=1.0, maker_ratio=1.0,
                      half_spread_bps=0.0, min_bps=1.0)
    spec_bin = spec_of("rank_binary", max_weight=1.0, beta_neutral=False)
    sA = evaluate(pn, fn, mn, old_n, spec_bin, cheap, "A. hệ thống cũ (phí 1bp GIẢ ĐỊNH)")

    per_n = estimate_cost_bps(pn, notional_usd=20.0, bars_per_day=6)
    real_n = CostModel(maker_ratio=MAKER_RATIO, half_spread_bps=0.0,
                       per_symbol_bps=per_n * (1 - MAKER_RATIO), min_bps=1.0)
    sB = evaluate(pn, fn, mn, old_n, spec_bin, real_n, "B. + chi phí ĐO THẬT theo từng cặp", sA["sharpe"])
    base = sB["sharpe"]

    # ---- C: universe rộng ----
    pw, fw, mw = load("artifacts/universe_wide.json")
    old_w = old_signals(pw, fw)
    per_w = estimate_cost_bps(pw, notional_usd=20.0, bars_per_day=6)
    real_w = CostModel(maker_ratio=MAKER_RATIO, half_spread_bps=0.0,
                       per_symbol_bps=per_w * (1 - MAKER_RATIO), min_bps=1.0)
    spec_bin_w = spec_of("rank_binary", max_weight=1.0, beta_neutral=False)
    n_narrow = int(pn["close"][mn].notna().sum(axis=1).median())
    n_wide = int(pw["close"][mw].notna().sum(axis=1).median())
    sC = evaluate(pw, fw, mw, old_w, spec_bin_w,
                  real_w, f"C. + universe rộng ({n_narrow}->{n_wide} cặp xếp hạng)", base)

    # ---- D: thư viện tín hiệu theo họ, gộp ĐỀU ----
    marks_w = marks_of(pw, mw)
    fam_full = build_all_families(pw, fw)
    fam = {k: v.reindex(marks_w) for k, v in fam_full.items()}
    eq_sig = combine_signals_zscore(fam, min_coverage=2)
    sD = evaluate(pw, fw, mw, eq_sig, spec_bin_w, real_w, "D. + thư viện 26 tín hiệu / 5 họ (gộp đều)", base)

    # ---- E: gộp THÍCH ỨNG ----
    ad_sig = combine_adaptive(fam, pw["close"].reindex(marks_w),
                              CombinerSpec(lookback=500, min_periods=90, t_threshold=1.5),
                              top_frac=0.10)
    sE = evaluate(pw, fw, mw, ad_sig, spec_bin_w, real_w, "E. + gộp THÍCH ỨNG (tự học dấu/trọng số)", base)

    # ---- F: trọng số theo rủi ro trong tập đã chọn ----
    spec_rp = spec_of("zscore_riskparity", max_weight=0.20, beta_neutral=False)
    sF = evaluate(pw, fw, mw, ad_sig, spec_rp, real_w, "F. + trọng số liên tục & chia đều rủi ro", base)

    # ---- G: khớp lệnh thụ động ----
    maker = 0.85
    real_mk = CostModel(maker_ratio=maker, half_spread_bps=0.0,
                        per_symbol_bps=per_w * (1 - maker), min_bps=0.5)
    best_spec = spec_rp if sF["sharpe"] >= sE["sharpe"] else spec_bin_w
    sG = evaluate(pw, fw, mw, ad_sig, best_spec, real_mk,
                  f"G. + khớp thụ động {maker:.0%} (thay vì {MAKER_RATIO:.0%})", base)

    print("-" * 106)
    print(f"Δ tính từ dòng B (cùng giả định chi phí thật). Dòng A->B là mức mà giả định "
          f"phí 1bp cũ đã thổi phồng.")
    print(f"TỔNG: Sharpe {base:.2f} -> {sG['sharpe']:.2f}  "
          f"({(sG['sharpe']/base-1)*100:+.0f}%) | lợi suất {sB['ann_return']*100:.1f}% -> "
          f"{sG['ann_return']*100:.1f}%/năm ở gross 1.0x")

    json.dump({k: v for k, v in zip(
        ["A_old_cheap", "B_real_cost", "C_wide", "D_library", "E_adaptive", "F_riskparity", "G_maker"],
        [sA, sB, sC, sD, sE, sF, sG])},
        open("artifacts/attribution.json", "w"), indent=2, default=float)
    print("-> artifacts/attribution.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
