#!/usr/bin/env python3
"""
ÍT LỆNH HƠN, ÍT ĐỒNG HƠN ở vốn vài triệu VND — cách nào giữ được lời, cách nào không.

VÌ SAO PHẢI ĐO LẠI: mọi kết luận "đóng" về chu kỳ tái cân bằng, vùng đệm, số vị thế đều
đo ở chi phí MÔ HÌNH ~4,5bp/chiều. Chi phí thật là 15,7bp (replay 24/09), tức mỗi đơn vị
turnover đắt gấp ~3,5 lần — đánh đổi "ít lệnh hơn đổi lấy tín hiệu cũ hơn" có thể lật.
Và backtest cũ tái cân bằng TOÀN PHẦN mỗi kỳ, trong khi live bỏ mọi điều chỉnh < 20% vị
thế (dải không giao dịch) và mọi lệnh mở/tăng < $5 — nó chưa từng đo đúng số lệnh live đặt.

ĐIỀU CẦN NÓI THẲNG TRƯỚC: phí Binance tính theo TỶ LỆ (bp), không có phí cố định mỗi lệnh.
Một lệnh $10 và một lệnh $1.000 trả cùng một tỷ lệ phí, nên lệnh nhỏ KHÔNG tự nó làm lời
ít đi theo phần trăm. Ở vốn nhỏ, thứ thật sự gắn với cỡ lệnh chỉ là: min notional $5 (ép
đòn bẩy sàn 6n/vốn, bỏ vị thế và lệnh nhỏ — F54/F55) và làm tròn khối lượng. Script đo
chúng đúng như live.

Hai giai đoạn, trên `run_v3_fine` (nến 4h), cấu hình V3 12 vị thế, chi phí thật, vốn cố định:
  A. GIẢM LỆNH ở n=12, 2x: chu kỳ {72h, 144h} × dải không giao dịch {0 (backtest cũ),
     0,2 (live), 0,35, 0,5, 1,0 (chỉ vào/ra)} × làm mượt tín hiệu {0, 1, 2 kỳ}.
  B. ÍT ĐỒNG HƠN: n ∈ {4, 6, 8, 10, 12, 16, 20} × đòn bẩy, với cách giảm lệnh chọn ở A.
Mỗi cấu hình: Sharpe, lệnh/tuần, $/lệnh, turnover, rồi bootstrap khối 72h ra 1 năm CÓ
ngắt mạch như live (kill 30% hấp thụ).

═══════════════════════════════════════════════════════════════════════════════
LUẬT CHỌN — CHỐT TRƯỚC KHI CHẠY
═══════════════════════════════════════════════════════════════════════════════
Ràng buộc: P(cháy 1 năm, không ngắt mạch) <= 2% ở CẢ HAI cơ sở; |net|/gross <= 2%.
Chỉ tiêu: vốn cuối năm TRUNG VỊ có ngắt mạch như live, cơ sở THẬN TRỌNG (toàn lịch sử).
Chọn: trong các cấu hình đạt ràng buộc và có trung vị >= 95% cấu hình tốt nhất, lấy cấu
hình ÍT LỆNH/TUẦN nhất. Tức là: ít lệnh hơn được chấp nhận nếu và chỉ nếu nó tốn không
quá 5% tiền lời. Cơ sở lạc quan (kỷ nguyên >=100 cặp) chỉ để LOẠI.
Làm mượt tín hiệu chưa có ở live: nếu thắng, script báo và KHÔNG ghi cấu hình với nó.

    python scripts/fewer_trades_study.py                        # 3 triệu VND, 15,7bp
    python scripts/fewer_trades_study.py --capital-vnd 5000000
    python scripts/fewer_trades_study.py --synthetic            # CHỈ kiểm tra đường chạy
"""
import argparse
import copy
import json
import sys
import zlib
from dataclasses import replace

import numpy as np
import pandas as pd

from aegis.research.small_capital import (
    block_bootstrap_paths, drop_below_min_notional, drop_stats, evaluate_paths, min_leverage,
)
from aegis.research.strategy_v3 import (
    combined_signal, config_from_json, load_v3_data, run_v3_fine,
)
from aegis.risk.portfolio import build_weights

BASE_CONFIG = "artifacts/strategy_v3.json"
OUT_CONFIG = "artifacts/strategy_v3_lean.json"
OUT_CSV = "artifacts/fewer_trades_study.csv"
VND_PER_USD = 26_300.0
MIN_ORDER_USD = 5.0 * 1.2
MAKER = 0.39
BARS_YEAR = 2190
BLOCK = 18
HIGH_MIN_NOTIONAL = ("ETHUSDT", "LTCUSDT", "LINKUSDT", "ETCUSDT", "BCHUSDT", "BTCUSDT")
PERIODS = (18, 36)                      # nến 4h: 72h, 144h
BANDS = (0.0, 0.2, 0.35, 0.5, 1.0)
SMOOTH = (0, 1, 2)
N_GRID = (4, 6, 8, 10, 12, 16, 20)
LEV_GRID = (1.0, 1.5, 2.0, 2.5, 3.0)
STAGE_A_LEV = 2.0
MAX_P_RUIN = 0.02
MAX_NET = 0.02
TOLERANCE = 0.95
BASES = ("toàn lịch sử", "kỷ nguyên >=100")
LIVE_DEFAULT = (18, 0.2, 0)             # cấu hình live hiện tại: 72h, dải 0,2, không làm mượt


def measure(res, hi_idx, cap_usd, lev, paths_n, label, period_bars) -> list:
    """Một cấu hình -> một dòng mỗi cơ sở: Sharpe, lệnh/tuần, $/lệnh, turnover, bootstrap."""
    r = res.returns.dropna()
    orders = res.meta["orders"].reindex(r.index).fillna(0.0)
    turn = res.turnover.reindex(r.index).fillna(0.0)
    rows = []
    for bname, keep in zip(BASES, (np.ones(len(r), bool), r.index.isin(hi_idx))):
        rv = r[keep].to_numpy()
        if len(rv) < BLOCK * 20:
            continue
        weeks = keep.sum() / 42.0
        n_ord = float(orders[keep].sum())
        paths = block_bootstrap_paths(rv, BARS_YEAR, paths_n,
                                      zlib.crc32(f"{label}|{bname}".encode()), BLOCK)
        rows.append({
            "basis": bname,
            "sharpe": float(rv.mean() / rv.std(ddof=1) * np.sqrt(BARS_YEAR)),
            "orders_week": n_ord / weeks,
            "usd_per_order": float(turn[keep].sum()) * cap_usd * lev / max(n_ord, 1.0),
            "turnover_year": float(turn[keep].sum()) / weeks * 52,
            "net_abs": float(res.net_exposure.reindex(r.index)[keep].abs().mean()
                             / res.gross_exposure.reindex(r.index)[keep].replace(0, np.nan)
                             .mean()),
            **evaluate_paths(paths, lev, period_bars)})
    return rows


def widen(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    vals = ["sharpe", "orders_week", "usd_per_order", "turnover_year", "net_abs",
            "median_live", "median_raw", "p_ruin", "p_kill", "p_loss_live"]
    w = df.pivot_table(index=keys, columns="basis", values=vals)
    w.columns = [f"{v}|{b}" for v, b in w.columns]
    for v in vals:
        for b in BASES:
            if f"{v}|{b}" not in w.columns:
                w[f"{v}|{b}"] = np.nan
    return w.reset_index()


def choose(w: pd.DataFrame) -> pd.DataFrame:
    """Luật chốt trước: đạt ràng buộc, >= 95% trung vị tốt nhất, rồi ÍT LỆNH nhất."""
    c, o = BASES
    ok = w[(w[f"p_ruin|{c}"] <= MAX_P_RUIN) & (w[f"p_ruin|{o}"].fillna(0) <= MAX_P_RUIN)
           & (w[f"net_abs|{c}"] <= MAX_NET)]
    if ok.empty:
        return ok
    best = ok[f"median_live|{c}"].max()
    near = ok[ok[f"median_live|{c}"] >= TOLERANCE * best]
    return near.sort_values([f"orders_week|{c}", f"median_live|{c}"], ascending=[True, False])


def weekly(x: float) -> float:
    return (x ** (1 / 52) - 1) if x > 0 else -1.0


def show(frame: pd.DataFrame, cols: list, capital_vnd: float) -> None:
    c, o = BASES
    head = "".join(f"{k:>7}" for k in cols)
    print(f"{head}{'Sharpe':>8}{'lệnh/tuần':>10}{'$/lệnh':>8}{'turn/năm':>9}{'x/năm':>7}"
          f"{'VND/tuần':>10}{'kill':>6}{'cháy':>6}{'x lạc q':>8}")
    print("-" * (7 * len(cols) + 72))
    for _, b in frame.iterrows():
        x = b[f"median_live|{c}"]
        cells = "".join(f"{b[k]:>7.2f}" if isinstance(b[k], float) else f"{b[k]:>7}" for k in cols)
        print(f"{cells}{b[f'sharpe|{c}']:>8.2f}{b[f'orders_week|{c}']:>10.1f}"
              f"{b[f'usd_per_order|{c}']:>8.1f}{b[f'turnover_year|{c}']:>9.0f}{x:>7.2f}"
              f"{weekly(x) * capital_vnd:>10,.0f}{b[f'p_kill|{c}'] * 100:>5.0f}%"
              f"{b[f'p_ruin|{c}'] * 100:>5.1f}%{b[f'median_live|{o}']:>8.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital-vnd", type=float, default=3_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=15.7)
    ap.add_argument("--paths", type=int, default=3000)
    ap.add_argument("--synthetic", action="store_true",
                    help="panel giả lập — CHỈ kiểm tra đường chạy, KHÔNG phải kết quả")
    a = ap.parse_args(argv)
    cap = a.capital_vnd / VND_PER_USD

    cfg0 = replace(config_from_json(BASE_CONFIG), n_tranches=1)
    if a.synthetic:
        print("!" * 100)
        print("!! CHẾ ĐỘ TỔNG HỢP — số liệu GIẢ, chỉ chứng minh script chạy hết đường.")
        print("!" * 100)
        sys.path.insert(0, "scripts")
        from residual_study import synthetic_data
        data = synthetic_data(n_bars=6 * 365 * 2, n_assets=80)
    else:
        data = load_v3_data(cfg0)
    hi = data.close.notna().sum(axis=1) >= 100
    hi_idx = set(hi.index[hi.values])

    sigs = {}
    for p in PERIODS:
        s = combined_signal(data, replace(cfg0, rebalance_every=p))
        drop = [c for c in s.columns if c in HIGH_MIN_NOTIONAL] if cap < 500 else []
        if drop:
            s = s.copy()
            s[drop] = np.nan
        sigs[p] = s

    # ------------------------------------------------------------ A. GIẢM LỆNH ở n=12
    rows = []
    for p in PERIODS:
        cfg = replace(cfg0, rebalance_every=p)
        for band in BANDS:
            for sm in SMOOTH:
                res = run_v3_fine(data, cfg, maker_ratio=MAKER, cost_bps=a.cost_bps, sig=sigs[p],
                                  capital_usd=cap, leverage=STAGE_A_LEV, no_trade_band=band,
                                  smooth_halflife=sm or None)
                for row in measure(res, hi_idx, cap, STAGE_A_LEV, a.paths,
                                   f"A|{p}|{band}|{sm}", p):
                    rows.append({"stage": "A", "hours": p * 4, "band": band, "smooth": sm,
                                 "n": cfg.n_positions, "L": STAGE_A_LEV, **row})
                print(f"  A: {p * 4}h, dải {band}, làm mượt {sm}", flush=True)
    dfa = pd.DataFrame(rows)
    wa = widen(dfa, ["hours", "band", "smooth"])
    c = BASES[0]
    print("\n" + "=" * 120)
    print(f"A. GIẢM LỆNH — V3 12 vị thế, {STAGE_A_LEV}x, vốn {a.capital_vnd:,.0f} VND (~${cap:.0f}), "
          f"chi phí {a.cost_bps}bp/chiều, có ngắt mạch như live")
    print("=" * 120)
    show(wa.sort_values(f"median_live|{c}", ascending=False), ["hours", "band", "smooth"],
         a.capital_vnd)
    pick_a = choose(wa)
    live_now = wa[(wa["hours"] == LIVE_DEFAULT[0] * 4) & (wa["band"] == LIVE_DEFAULT[1])
                  & (wa["smooth"] == LIVE_DEFAULT[2])]
    deployable = pick_a[pick_a["smooth"] == 0]
    if pick_a.empty:
        print("\n=> A: không cấu hình nào đạt ràng buộc.")
        return 0
    ba = pick_a.iloc[0]
    print(f"\n=> A chọn (luật chốt trước): {int(ba['hours'])}h, dải {ba['band']}, làm mượt "
          f"{int(ba['smooth'])} — {ba[f'orders_week|{c}']:.1f} lệnh/tuần so với live hiện tại "
          f"{live_now[f'orders_week|{c}'].iloc[0]:.1f}")
    if ba["smooth"] and not deployable.empty:
        ba = deployable.iloc[0]
        print(f"   ⚠️ làm mượt chưa có ở live -> dùng cấu hình triển khai được tốt nhất: "
              f"{int(ba['hours'])}h, dải {ba['band']}")

    # ------------------------------------------------------------ B. ÍT ĐỒNG HƠN
    p_b, band_b = int(ba["hours"]) // 4, float(ba["band"])
    rows = []
    for n in N_GRID:
        lmin = min_leverage(n, cap, MIN_ORDER_USD)
        port = replace(cfg0.portfolio, n_positions=n)
        cfg = replace(cfg0, rebalance_every=p_b, n_positions=n, portfolio=port)
        w0 = build_weights(sigs[p_b], data.close.reindex(sigs[p_b].index), port)
        for L in sorted({round(lmin, 2)} | {x for x in LEV_GRID if x >= lmin}):
            ds = drop_stats(w0, drop_below_min_notional(w0, cap, L, cfg.min_notional_usd))
            res = run_v3_fine(data, cfg, maker_ratio=MAKER, cost_bps=a.cost_bps, sig=sigs[p_b],
                              capital_usd=cap, leverage=L, no_trade_band=band_b)
            for row in measure(res, hi_idx, cap, L, a.paths, f"B|{n}|{L}", p_b):
                rows.append({"stage": "B", "n": n, "L": L, "L_min": lmin,
                             "gross_kept": ds["gross_kept"], **row})
        print(f"  B: n={n} (sàn {lmin:.2f}x)", flush=True)
    dfb = pd.DataFrame(rows)
    wb = widen(dfb, ["n", "L"])
    print("\n" + "=" * 120)
    print(f"B. ÍT ĐỒNG HƠN — {p_b * 4}h, dải {band_b}, vốn {a.capital_vnd:,.0f} VND")
    print("=" * 120)
    show(wb.sort_values(["n", "L"]), ["n", "L"], a.capital_vnd)
    pick_b = choose(wb)
    if pick_b.empty:
        print("\n=> B: không cấu hình nào đạt ràng buộc.")
    else:
        bb = pick_b.iloc[0]
        best_n = wb.loc[wb[f"median_live|{c}"].idxmax()]
        print(f"\n=> B chọn: n={int(bb['n'])}, {bb['L']:.2f}x — {bb[f'orders_week|{c}']:.1f} "
              f"lệnh/tuần, ${bb[f'usd_per_order|{c}']:.0f}/lệnh, trung vị x{bb[f'median_live|{c}']:.2f}"
              f" (tốt nhất bất kể số lệnh: n={int(best_n['n'])}, {best_n['L']:.2f}x, "
              f"x{best_n[f'median_live|{c}']:.2f})")
        if not a.synthetic:
            base = json.load(open(BASE_CONFIG))
            out = copy.deepcopy(base)
            out["config"].update(n_positions=int(bb["n"]), rebalance_every=p_b,
                                 period_hours=p_b * 4, no_trade_band=band_b)
            for k in ("train", "holdout", "deflated_sharpe", "kelly_leverage_full",
                      "leverage_table", "required_sharpe_for_50pct_week"):
                out.pop(k, None)
            out["holdout_uses"] = "KHÔNG áp dụng — chọn bởi fewer_trades_study, chưa kiểm định sạch."
            out["num_trials_declared"] = int(base.get("num_trials_declared", 250)) \
                + len(wa) + len(wb)
            out["recommended_gross_leverage"] = float(bb["L"])
            out["capital_vnd"] = a.capital_vnd
            out["derived_from"] = BASE_CONFIG
            out["note"] = (f"CẤU HÌNH ÍT LỆNH — scripts/fewer_trades_study.py, luật chốt trước, vốn "
                           f"{a.capital_vnd:,.0f} VND. Chạy daemon với --leverage {bb['L']:.2f}.")
            with open(OUT_CONFIG, "w") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"   đã ghi {OUT_CONFIG}. Đổi daemon sang file này là việc của NGƯỜI VẬN HÀNH.")

    if not a.synthetic:
        pd.concat([dfa, dfb]).to_csv(OUT_CSV, index=False)
        print(f"đã ghi {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
