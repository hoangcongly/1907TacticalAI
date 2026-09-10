#!/usr/bin/env python3
"""Quét tín hiệu cross-sectional trên universe đa tài sản."""
import sys, warnings
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from aegis.data.panel import load_panel, load_funding_panel
from aegis.research.cross_sectional import (
    backtest_cross_sectional, signal_funding_carry, signal_low_vol,
    signal_momentum, signal_ofi, signal_reversal,
)
from download_universe import DEFAULT_UNIVERSE

INTERVAL, REBAL = "4h", 6          # tái cân bằng mỗi 6 nến 4h = 1 ngày
PERIODS_PER_YEAR = 365.0

def main():
    panel = load_panel(DEFAULT_UNIVERSE, INTERVAL)
    close, ofi = panel["close"], panel.get("ofi")
    funding = load_funding_panel(DEFAULT_UNIVERSE, close.index)
    print(f"Universe {close.shape[1]} cặp x {close.shape[0]} nến {INTERVAL} "
          f"({close.shape[0]*4/24:.0f} ngày), tái cân bằng mỗi {REBAL} nến\n")

    signals = {
        "funding_carry":  signal_funding_carry(funding, smooth=6),
        "momentum_30":    signal_momentum(close, lookback=30),
        "momentum_90":    signal_momentum(close, lookback=90),
        "reversal_6":     signal_reversal(close, lookback=6),
        "low_vol":        signal_low_vol(close, lookback=60),
        "ofi_flow":       signal_ofi(ofi, smooth=6) if ofi is not None else None,
    }

    print(f"{'tín hiệu':<16}{'phí':<8}{'n':>6}{'ann.ret':>9}{'ann.vol':>9}"
          f"{'sharpe':>8}{'t-stat':>8}{'maxDD':>8}{'turn':>7}")
    print("-" * 79)

    rows = []
    for cost, cname in [(0.0004, "taker"), (0.0001, "maker")]:
        for name, sig in signals.items():
            if sig is None:
                continue
            res = backtest_cross_sectional(
                signal=sig, close=close, funding=funding, interval=INTERVAL,
                rebalance_every=REBAL, top_frac=0.25, cost_rate=cost, neutral=True,
            )
            s = res.stats(PERIODS_PER_YEAR)
            if s.get("viable") == 0.0:
                continue
            s.update(signal=name, cost=cname)
            rows.append(s)
            flag = " *" if abs(s["t_stat"]) > 2 else ""
            print(f"{name:<16}{cname:<8}{s['n']:>6.0f}{s['ann_return']*100:>8.1f}%"
                  f"{s['ann_vol']*100:>8.1f}%{s['sharpe']:>8.2f}{s['t_stat']:>8.2f}"
                  f"{s['max_dd']*100:>7.1f}%{s['avg_turnover']:>7.2f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/xs_research.csv", index=False)
    sig_rows = df[df.t_stat.abs() > 2]
    print("\n" + "=" * 79)
    if len(sig_rows):
        print(f"✅ {len(sig_rows)} tín hiệu có |t-stat| > 2:")
        for _, r in sig_rows.sort_values("t_stat", ascending=False).iterrows():
            print(f"   {r['signal']:<16}{r['cost']:<8} ann {r['ann_return']*100:+6.1f}%  "
                  f"sharpe {r['sharpe']:5.2f}  t={r['t_stat']:5.2f}")
    else:
        print("❌ Không tín hiệu nào đạt |t-stat| > 2")
    print("\nchi tiết -> artifacts/xs_research.csv")

if __name__ == "__main__":
    main()
