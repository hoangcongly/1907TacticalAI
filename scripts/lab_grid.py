#!/usr/bin/env python3
"""
Lưới thử nghiệm ĐỘ RỘNG x TẦN SUẤT trên tập TRAIN, qua đúng đường đi của StrategyV2.

Hai đòn bẩy còn lại sau khi đã sửa xong tầng dựng danh mục:
  - ĐỘ RỘNG: xếp hạng trên 59 cặp hay 152 cặp?
  - TẦN SUẤT: nến 4h/1h, tái cân bằng mỗi bao nhiêu nến?

Định luật cơ bản nói cả hai đều nâng IR theo căn bậc hai. Chi phí thì tăng TUYẾN
TÍNH theo tần suất. Lưới này tìm chỗ hai đường cắt nhau — chứ không đoán.
"""
import argparse
import itertools
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.data.panel_v2 import load_panel_v2, load_funding_panel_v2, align_panel, interval_hours
from aegis.research.backtest_v2 import CostModel, estimate_cost_bps
from aegis.research.strategy_v2 import StrategyV2, StrategyV2Config
from aegis.risk.portfolio import PortfolioSpec

_CACHE = {}


def get_data(universe_file, interval, source):
    key = (universe_file, interval, source)
    if key in _CACHE:
        return _CACHE[key]
    syms = json.load(open(universe_file))
    panel = load_panel_v2(syms, interval=interval, source_interval=source, min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]
    mask = panel["close"].index < split
    _CACHE[key] = (panel, funding, mask)
    return _CACHE[key]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*",
                    default=["artifacts/universe_filtered.json", "artifacts/universe_wide.json"])
    ap.add_argument("--intervals", nargs="*", default=["4h", "1h"])
    ap.add_argument("--rebalances", nargs="*", type=int, default=[3, 6, 12, 24])
    ap.add_argument("--top-frac", type=float, default=0.10, dest="top_frac")
    ap.add_argument("--maker-ratio", type=float, default=0.5, dest="maker_ratio")
    ap.add_argument("--out", default="artifacts/lab_grid.json")
    a = ap.parse_args(argv)

    rows = []
    print(f"{'universe':<10}{'khung':>6}{'rebal':>7}{'giờ/kỳ':>8}{'cặp':>6}"
          f"{'ann':>9}{'vol':>7}{'sharpe':>8}{'t':>7}{'maxDD':>8}{'turn':>7}{'phí/năm':>9}{'bp':>6}")
    print("-" * 98)

    for uf, interval in itertools.product(a.universes, a.intervals):
        try:
            panel, funding, mask = get_data(uf, interval, "1h")
        except Exception as exc:
            print(f"{uf} {interval}: bỏ qua ({type(exc).__name__}: {exc})")
            continue

        bars_per_day = max(1, int(round(24 / interval_hours(interval))))
        per_sym = estimate_cost_bps(panel, notional_usd=20.0, bars_per_day=bars_per_day)

        for rebal in a.rebalances:
            period_h = interval_hours(interval) * rebal
            if period_h < 4 or period_h > 96:
                continue
            cfg = StrategyV2Config(
                universe_file=uf, interval=interval, rebalance_every=rebal,
                maker_ratio=a.maker_ratio,
                portfolio=PortfolioSpec(mode="zscore_riskparity", top_frac=a.top_frac,
                                        beta_neutral=True, max_weight=0.20),
            )
            cost = CostModel(maker_ratio=a.maker_ratio, half_spread_bps=0.0,
                             per_symbol_bps=per_sym * (1 - a.maker_ratio), min_bps=1.0)
            try:
                out = StrategyV2(cfg).backtest(panel, funding, mask, cost, with_leverage=False)
            except Exception as exc:
                print(f"  {interval} rebal={rebal}: lỗi {type(exc).__name__}: {exc}")
                continue

            s = out["unlevered"]
            if s.get("n", 0) < 100:
                continue
            tag = "hẹp" if "filtered" in uf else "rộng"
            print(f"{tag:<10}{interval:>6}{rebal:>7}{period_h:>8.0f}{panel['close'].shape[1]:>6}"
                  f"{s['ann_return']*100:>8.1f}%{s['ann_vol']*100:>6.1f}%{s['sharpe']:>8.2f}"
                  f"{s['t_stat']:>7.2f}{s['max_dd']*100:>7.1f}%{s['avg_turnover']:>7.2f}"
                  f"{s['cost_drag_ann']*100:>8.1f}%{out['cost_bps_mean']:>6.1f}")
            s.update({"universe": tag, "interval": interval, "rebalance": rebal,
                      "period_hours": period_h, "n_symbols": int(panel["close"].shape[1])})
            rows.append(s)

    if rows:
        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        print(f"\nTOP 5 theo Sharpe:")
        print(df[["universe", "interval", "rebalance", "ann_return", "sharpe",
                  "t_stat", "max_dd", "cost_drag_ann"]].head(5).round(3).to_string(index=False))
        json.dump(rows, open(a.out, "w"), indent=2, default=float)
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
