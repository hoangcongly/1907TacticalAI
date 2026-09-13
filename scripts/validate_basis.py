#!/usr/bin/env python3
"""
KIỂM ĐỊNH BASIS TRADE — trước khi tin, phải trả lời bốn câu.

  1. Có phải một đỉnh nhọn may mắn không?   -> lưới tham số, tìm VÙNG PHẲNG
  2. Edge còn sống hay đã chết?             -> phân rã theo năm
  3. Chi phí xấu tới đâu thì chết?          -> stress chi phí
  4. Vốn $38 có chạy được không?            -> ràng buộc min notional HAI chân

Câu 2 đặc biệt quan trọng với lớp chiến lược này: tài liệu ghi nhận Sharpe carry
crypto rơi từ 6.45 (2020-2025) xuống 4.06 (từ 2024) rồi ÂM trong 2025. Nếu edge đã
chết thì mọi con số trung bình toàn mẫu đều vô nghĩa.

    python scripts/validate_basis.py
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.data.panel_v2 import (
    align_panel, load_funding_panel_v2, load_panel_v2, resample_bars,
)
from aegis.research.basis_trade import BasisSpec, backtest_basis, basis_series

SPOT_ROOT = pathlib.Path("data/binance_spot")
PPY = 365 * 6          # nến 4h


def load_all(universe_file: str):
    syms = json.load(open(universe_file))
    perp = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.10)
    funding = load_funding_panel_v2(syms, perp["close"].index)
    perp, funding = align_panel(perp, funding)

    spot = {}
    for s in perp["close"].columns:
        f = SPOT_ROOT / f"{s}_1h.parquet"
        if not f.is_file():
            continue
        raw = pd.read_parquet(f).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
        d = resample_bars(raw, "1h", "4h")
        spot[s] = d.set_index("timestamp_ms")["close"]

    S = pd.DataFrame(spot).reindex(perp["close"].index)
    keep = [c for c in perp["close"].columns
            if c in S.columns and S[c].notna().sum() > 2000]
    return perp["close"][keep], S[keep], funding[keep]


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 100:
        return {}
    eq = (1 + r).cumprod()
    dd = float((1 - eq / eq.cummax()).max())
    sd = r.std(ddof=1)
    return {"ann": float(r.mean() * PPY), "vol": float(sd * np.sqrt(PPY)),
            "sharpe": float(r.mean() / sd * np.sqrt(PPY)) if sd > 0 else 0.0,
            "dd": dd, "n": len(r)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="artifacts/basis_universe.json")
    ap.add_argument("--capital", type=float, default=38.0)
    a = ap.parse_args(argv)

    P, S, F = load_all(a.universe)
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]
    print(f"{P.shape[1]} cặp có đủ perp + spot | {P.shape[0]} nến 4h\n")

    b = basis_series(P, S).stack()
    print("=" * 84)
    print("BASIS (perp/spot - 1) — nguồn rủi ro thật của lớp chiến lược này")
    print("=" * 84)
    print(f"  trung vị {b.median()*100:+.3f}%  độ lệch {b.std()*100:.3f}%  "
          f"p1 {b.quantile(.01)*100:+.2f}%  p99 {b.quantile(.99)*100:+.2f}%")

    # ---------- 1. Lưới tham số ----------
    print("\n" + "=" * 84)
    print("1. LƯỚI THAM SỐ — tìm vùng phẳng, không tìm đỉnh")
    print("=" * 84)
    print(f"{'vịthế':>6}{'rebal(ngày)':>13}{'cửa sổ':>8}"
          f"{'  | TRAIN  ann   sharpe   maxDD':<32}{'| HOLDOUT  ann   sharpe   maxDD':<32}")
    print("-" * 92)

    rows = []
    for npos in (3, 5, 8, 12):
        for rb in (252, 504, 1008):
            for fw in (6, 12, 42):
                spec = BasisSpec(n_positions=npos, rebalance_bars=rb, funding_window=fw)
                df = backtest_basis(P, S, F, spec)
                tr, ho = stats(df["total"][df.index < split]), stats(df["total"][df.index >= split])
                if not tr or not ho:
                    continue
                rows.append({"npos": npos, "rebal": rb, "fw": fw,
                             "tr_sharpe": tr["sharpe"], "ho_sharpe": ho["sharpe"],
                             "tr_ann": tr["ann"], "ho_ann": ho["ann"],
                             "tr_dd": tr["dd"], "ho_dd": ho["dd"]})
                print(f"{npos:>6}{rb*4/24:>13.0f}{fw:>8}"
                      f"  | {tr['ann']*100:>7.1f}% {tr['sharpe']:>7.2f} {tr['dd']*100:>6.1f}%"
                      f"  | {ho['ann']*100:>7.1f}% {ho['sharpe']:>7.2f} {ho['dd']*100:>6.1f}%")

    g = pd.DataFrame(rows)
    if g.empty:
        print("Không đủ dữ liệu."); return 1
    print("-" * 92)
    print(f"HOLDOUT qua {len(g)} cấu hình: trung vị Sharpe {g['ho_sharpe'].median():.2f} | "
          f"min {g['ho_sharpe'].min():.2f} | max {g['ho_sharpe'].max():.2f} | "
          f"dương {(g['ho_sharpe']>0).mean()*100:.0f}%")
    print(f"lợi suất năm holdout: trung vị {g['ho_ann'].median()*100:.1f}%  "
          f"drawdown trung vị {g['ho_dd'].median()*100:.2f}%")

    # ---------- 2. Theo năm ----------
    best = g.sort_values("tr_sharpe", ascending=False).iloc[0]
    spec = BasisSpec(n_positions=int(best["npos"]), rebalance_bars=int(best["rebal"]),
                     funding_window=int(best["fw"]))
    df = backtest_basis(P, S, F, spec)
    print("\n" + "=" * 84)
    print(f"2. THEO NĂM (cấu hình chốt trên train: {int(best['npos'])} vị thế, "
          f"{int(best['rebal'])*4/24:.0f} ngày/lượt) — edge còn sống không?")
    print("=" * 84)
    yr = pd.to_datetime(df.index, unit="ms").year
    print(f"{'năm':<7}{'tổng':>9}{'funding':>10}{'basis':>9}{'phí':>8}{'sharpe':>9}{'kỳ':>7}")
    print("-" * 60)
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() < 100:
            continue
        x = df[m]
        s = stats(x["total"])
        print(f"{y:<7}{x['total'].sum()*100:>8.1f}%{x['funding'].sum()*100:>9.1f}%"
              f"{x['basis'].sum()*100:>8.1f}%{x['fee'].sum()*100:>7.1f}%"
              f"{s.get('sharpe',0):>9.2f}{m.sum():>7}")

    # ---------- 3. Stress chi phí ----------
    print("\n" + "=" * 84)
    print("3. STRESS CHI PHÍ — mức phí một vòng nào thì chết")
    print("=" * 84)
    print(f"{'bp/vòng':>9}{'ann (holdout)':>16}{'sharpe':>9}")
    print("-" * 36)
    for bps in (12, 24, 40, 60, 100):
        d = backtest_basis(P, S, F, BasisSpec(
            n_positions=spec.n_positions, rebalance_bars=spec.rebalance_bars,
            funding_window=spec.funding_window, round_trip_bps=bps))
        s = stats(d["total"][d.index >= split])
        print(f"{bps:>9}{s.get('ann',0)*100:>15.1f}%{s.get('sharpe',0):>9.2f}")

    # ---------- 4. Ràng buộc vốn ----------
    print("\n" + "=" * 84)
    print(f"4. VỐN ${a.capital:.0f} CÓ CHẠY ĐƯỢC KHÔNG?")
    print("=" * 84)
    print("Basis trade cần HAI chân cùng lúc: long spot + short perp.")
    print("  - chân spot  : min notional Binance ~$5, KHÔNG dùng được đòn bẩy")
    print("  - chân perp  : min notional $5")
    print(f"  - mỗi vị thế cần >= $10 vốn thật (chưa tính đệm an toàn)\n")
    for npos in (1, 2, 3, 5, 8):
        need = npos * 10.0
        ok = "OK" if need <= a.capital else f"THIẾU (cần ${need:.0f})"
        print(f"  {npos} vị thế -> cần ${need:>5.0f}   {ok}")
    print(f"\n=> Với ${a.capital:.0f}: tối đa {int(a.capital // 10)} vị thế basis.")
    print("   Lưới trên cho thấy ít vị thế hơn = biến động cao hơn và Sharpe thấp hơn.")

    json.dump({"n_symbols": int(P.shape[1]), "grid": rows,
               "chosen": {"n_positions": int(best["npos"]),
                          "rebalance_bars": int(best["rebal"]),
                          "funding_window": int(best["fw"])}},
              open("artifacts/basis_validation.json", "w"), indent=2, default=float)
    print("\n-> artifacts/basis_validation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
