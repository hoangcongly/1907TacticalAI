#!/usr/bin/env python3
"""Phát triển chiến lược cross-sectional — CHỈ CHẠY TRÊN TẬP TRAIN."""
import json, sys, warnings
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from aegis.data.panel import load_panel, load_funding_panel
from aegis.research.cross_sectional import *
from aegis.research.holdout import TimeSplit

INTERVAL, REBAL, TOP_FRAC, COST = "4h", 6, 0.10, 0.0001
PPY = 365.0


def build_all(part="train"):
    U = json.load(open("artifacts/universe70.json"))
    panel = load_panel(U, INTERVAL, min_coverage=0.15)
    close_full = panel["close"]
    funding_full = load_funding_panel(U, close_full.index).reindex(columns=close_full.columns)

    sp = json.load(open("artifacts/holdout_split.json"))
    split = TimeSplit(sp["split_ts"], sp["train_frac"])

    # Tín hiệu tính trên TOÀN chuỗi (cần lịch sử để có lookback), rồi mới CẮT.
    # Cắt trước sẽ làm hỏng lookback ở đầu holdout — nhưng tín hiệu nhân quả nên
    # giá trị tại t không phụ thuộc tương lai, cắt sau là hợp lệ.
    sigs = {
        "funding_carry":  signal_funding_carry(funding_full, 6),
        "momentum_90":    signal_momentum(close_full, 90),
        "ofi_flow":       signal_ofi(panel["ofi"], 6),
        "risk_adj_mom":   signal_risk_adjusted_momentum(close_full, 90),
        "lt_reversal":    signal_long_term_reversal(close_full, 360, 90),
        "neg_skew":       signal_negative_skew(close_full, 90),
        "funding_mom":    signal_funding_momentum(funding_full, 42),
        "idio_vol":       signal_idiosyncratic_vol(close_full, 60),
    }
    cut = split.train if part == "train" else split.holdout
    return {k: cut(v) for k, v in sigs.items()}, cut(close_full), cut(funding_full)


def stats(r, ppy=PPY):
    r = r.dropna().to_numpy()
    if len(r) < 30:
        return None
    mu, sd = r.mean(), r.std(ddof=1)
    eq = np.cumprod(1 + r)
    dd = float((1 - eq / np.maximum.accumulate(eq)).max())
    return {"ann": mu * ppy, "vol": sd * np.sqrt(ppy),
            "sharpe": mu / sd * np.sqrt(ppy) if sd > 1e-15 else 0.0,
            "t": mu / (sd / np.sqrt(len(r))) if sd > 1e-15 else 0.0,
            "dd": dd, "n": len(r)}


def main():
    sigs, close, funding = build_all("train")
    print(f"=== TẬP TRAIN: {close.shape[0]} nến x {close.shape[1]} cặp ===")
    print(f"(holdout vẫn NIÊM PHONG — không chạm tới)\n")

    rets = {}
    print(f"{'tín hiệu':<16}{'ann':>8}{'vol':>8}{'sharpe':>8}{'t-stat':>8}{'maxDD':>8}")
    print("-" * 56)
    for name, sig in sigs.items():
        r = backtest_cross_sectional(sig, close, funding, INTERVAL, REBAL,
                                     top_frac=TOP_FRAC, cost_rate=COST).returns
        s = stats(r)
        if s is None:
            continue
        rets[name] = r
        flag = " *" if abs(s["t"]) > 2 else ""
        print(f"{name:<16}{s['ann']*100:>7.1f}%{s['vol']*100:>7.1f}%"
              f"{s['sharpe']:>8.2f}{s['t']:>8.2f}{s['dd']*100:>7.1f}%{flag}")

    # Giữ lại tín hiệu có t-stat > 1.5 trên TRAIN (ngưỡng chọn lọc, không phải kết luận)
    keep = {k: v for k, v in rets.items() if abs(stats(v)["t"]) > 1.5 and stats(v)["ann"] > 0}
    print(f"\nGiữ lại {len(keep)} tín hiệu có t>1.5 và lợi nhuận dương: {list(keep)}")

    R = pd.DataFrame(keep).dropna()
    print("\n=== TƯƠNG QUAN ===")
    print(R.corr().round(2).to_string())

    print("\n=== CÁCH GỘP ===")
    print(f"{'phương pháp':<26}{'ann':>8}{'vol':>8}{'sharpe':>8}{'t-stat':>8}{'maxDD':>8}")
    print("-" * 66)
    variants = {}
    eq = R.mean(axis=1); variants["chia đều vốn"] = eq
    iv = combine_inverse_vol({k: R[k] for k in R.columns}); variants["nghịch đảo biến động"] = iv
    for tgt in (0.15, 0.20, 0.30):
        variants[f"inv-vol + voltarget {tgt:.0%}"] = volatility_target(iv, tgt, max_leverage=3.0)

    best_name, best_s = None, None
    for name, r in variants.items():
        s = stats(r)
        if s is None:
            continue
        print(f"{name:<26}{s['ann']*100:>7.1f}%{s['vol']*100:>7.1f}%"
              f"{s['sharpe']:>8.2f}{s['t']:>8.2f}{s['dd']*100:>7.1f}%")
        if best_s is None or s["sharpe"] > best_s["sharpe"]:
            best_name, best_s = name, s

    print(f"\n>>> TỐT NHẤT TRÊN TRAIN: {best_name} (sharpe {best_s['sharpe']:.2f})")
    json.dump({"signals": list(keep), "combine": best_name,
               "top_frac": TOP_FRAC, "rebalance_every": REBAL,
               "interval": INTERVAL, "cost_rate": COST,
               "train_sharpe": best_s["sharpe"], "train_ann": best_s["ann"]},
              open("artifacts/strategy_candidate.json", "w"), indent=2)
    print("ứng viên -> artifacts/strategy_candidate.json  (CHƯA kiểm định holdout)")


if __name__ == "__main__":
    main()
