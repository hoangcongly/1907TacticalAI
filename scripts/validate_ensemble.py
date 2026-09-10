#!/usr/bin/env python3
"""
Đánh giá TỔ HỢP nhiều cấu hình, so với từng thành phần riêng lẻ.

Thành phần được chọn để KHÁC NHAU ở chiều có ý nghĩa, và mỗi lựa chọn đều có lý do
nêu trước — không phải quét lưới rồi giữ cái thắng:

  - hai chu kỳ tái cân bằng (48h và 72h): bắt tín hiệu ở hai tốc độ suy giảm khác nhau
  - hai bộ tín hiệu (4 tín hiệu gốc, và 26 tín hiệu gộp thích ứng): một bộ đơn giản
    ít bậc tự do, một bộ rộng có khả năng thích ứng
  - hai cách đánh trọng số (nhị phân, liên tục chia đều rủi ro)

CẢNH BÁO VỀ GIÁ TRỊ CHỨNG CỨ: holdout đã bị dùng một lần cho cấu hình v3 và một lần
nữa cho lưới chẩn đoán. Con số holdout in ra ở đây là CHỈ BÁO, không phải kiểm định
sạch. Kiểm định thật của tổ hợp này chỉ có thể đến từ giao dịch giấy tiến về phía
trước trên dữ liệu chưa tồn tại.
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
from aegis.research.ensemble import EnsembleMember, ensemble_weights
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights, realized_vol

UNIVERSE = "artifacts/universe_wide.json"
N_POS = 12
MAKER = 0.5
GRID_REBAL = 12          # lưới chung: 48h
MEMBER_REBALS = (12, 18)  # 48h và 72h


def build_members(panel, funding, sigs_full, old_full, mask):
    cf = panel["close"]
    members = []
    for rebal in MEMBER_REBALS:
        marks = cf.index[::rebal]
        marks = marks[np.isin(marks, cf.index[mask])]
        if len(marks) < 60:
            continue
        close = cf.reindex(marks)
        ad = combine_adaptive({k: v.reindex(marks) for k, v in sigs_full.items()}, close,
                              CombinerSpec(lookback=500, min_periods=120, t_threshold=2.0,
                                           max_abs_weight=0.20, max_step=0.05),
                              top_frac=0.10)
        old = old_full.reindex(marks)
        for sig_name, sig in (("26ad", ad), ("old4", old)):
            for mode in ("rank_binary", "zscore_riskparity"):
                members.append(EnsembleMember(
                    name=f"{sig_name}/{mode}/r{rebal}", signal=sig,
                    spec=PortfolioSpec(mode=mode, n_positions=N_POS,
                                       max_weight=1.0 if mode == "rank_binary" else 0.20,
                                       beta_neutral=False)))
    return members


def main():
    syms = json.load(open(UNIVERSE))
    panel = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)
    cf = panel["close"]
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]

    print(f"dựng tín hiệu trên {cf.shape[1]} cặp...", flush=True)
    sigs_full = {n: build_signal(n, panel, funding) for n in SIGNAL_REGISTRY}
    old_full = combine_signals_zscore({
        "funding_carry": signal_funding_carry(funding, 6),
        "momentum_90": signal_momentum(cf, 90),
        "ofi_flow": signal_ofi(panel["ofi"], 6),
        "funding_mom": signal_funding_momentum(funding, 42),
    }, min_coverage=3)
    per_sym = estimate_cost_bps(panel, notional_usd=10.0, bars_per_day=6)
    cost = CostModel(maker_ratio=MAKER, half_spread_bps=0.0,
                     per_symbol_bps=per_sym * (1 - MAKER), min_bps=0.5)

    for part in ("train", "holdout"):
        mask = cf.index < split if part == "train" else cf.index >= split
        grid = cf.index[::GRID_REBAL]
        grid = grid[np.isin(grid, cf.index[mask])]
        ppy = 24 * 365.0 / (4 * GRID_REBAL)
        members = build_members(panel, funding, sigs_full, old_full, mask)

        print(f"\n{'='*92}")
        print(f"{part.upper()}  —  lưới chung {4*GRID_REBAL}h, {len(members)} thành phần, "
              f"{N_POS} vị thế")
        print("=" * 92)
        print(f"{'thành phần':<26}{'ann':>9}{'vol':>8}{'sharpe':>9}{'t':>7}{'maxDD':>9}{'turn':>8}")
        print("-" * 92)

        singles = []
        for m in members:
            close_m = cf.reindex(m.signal.index)
            W = build_weights(m.signal, close_m, m.spec)
            rb = int(round(len(cf.index[mask]) / max(len(m.signal.index), 1)))
            res = simulate(W, close_m, funding.reindex(m.signal.index), cost,
                           bar_hours=4 * (12 if "r12" in m.name else 18), rebalance_every=1)
            p = 24 * 365.0 / (4 * (12 if "r12" in m.name else 18))
            s = res.stats(p)
            singles.append(s["sharpe"])
            print(f"{m.name:<26}{s['ann_return']*100:>8.1f}%{s['ann_vol']*100:>7.1f}%"
                  f"{s['sharpe']:>9.2f}{s['t_stat']:>7.2f}{s['max_dd']*100:>8.1f}%"
                  f"{s['avg_turnover']:>8.2f}")

        vol_grid = realized_vol(cf.reindex(grid), 60)
        W_ens = ensemble_weights(members, cf, grid, max_weight=0.20,
                                 max_positions=N_POS, vol=vol_grid)
        res = simulate(W_ens, cf.reindex(grid), funding.reindex(grid), cost,
                       bar_hours=4 * GRID_REBAL, rebalance_every=1)
        s = res.stats(ppy)
        print("-" * 92)
        print(f"{'TỔ HỢP':<26}{s['ann_return']*100:>8.1f}%{s['ann_vol']*100:>7.1f}%"
              f"{s['sharpe']:>9.2f}{s['t_stat']:>7.2f}{s['max_dd']*100:>8.1f}%"
              f"{s['avg_turnover']:>8.2f}")
        print(f"{'':<26}(thành phần: trung vị {np.median(singles):.2f}, "
              f"min {np.min(singles):.2f}, max {np.max(singles):.2f}, "
              f"vị thế trung bình {s['avg_positions']:.0f})")

        if part == "holdout":
            json.dump({"ensemble_holdout": {k: float(v) for k, v in s.items()},
                       "member_sharpes": {m.name: float(x) for m, x in zip(members, singles)},
                       "grid_rebalance": GRID_REBAL, "n_positions": N_POS,
                       "maker_ratio": MAKER,
                       "evidence_note": "Holdout đã bị dùng trước đó; con số này là CHỈ BÁO, "
                                        "không phải kiểm định ngoài mẫu sạch."},
                      open("artifacts/ensemble_result.json", "w"), indent=2, ensure_ascii=False)
    print("\n-> artifacts/ensemble_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
