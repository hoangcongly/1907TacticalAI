#!/usr/bin/env python3
"""
MỌI CÁCH giảm số đồng, giảm số lệnh, tăng lời ở vốn vài triệu VND — đo trên DỮ LIỆU THẬT.

Dữ liệu: BINANCE USDⓈ-M — đúng sàn và đúng rổ live (`data/binance`, rổ
`artifacts/universe_wide.json`, 127 cặp). Chạy trên máy có `data/` (máy Mac), hoặc trong
container sau khi mở Network access cho `data.binance.vision` và `fapi.binance.com` rồi tải
bằng `scripts/download_wide_universe.py`.

Mọi cấu hình giao dịch NHƯ LIVE: bỏ vị thế và lệnh mở/tăng dưới $5 ở đúng vốn × đòn bẩy
(F54/F55), dải không giao dịch, cân lại trung lập (F42); chi phí thật 15,7bp/chiều; chỉ
đồng có khối lượng >= $5 triệu/ngày (trung bình 30 ngày trước đó, như bộ lọc live). Tất cả
chấm trên CÙNG một cửa sổ (từ `EVAL_START`, khi tầng gộp đã khởi động xong ở mọi chu kỳ).

DANH MỤC ĐÒN BẨY (mỗi nhóm đổi MỘT thứ so với gốc: V3 12 đồng, 72h, dải 0,2, 1,5x):
  1. SỐ ĐỒNG / CÁCH CHỌN ĐỒNG
     1a số vị thế n ∈ {4..50}
     1b cách chia vốn: V3 (zscore, trần 0,2) / đều tuyệt đối / zscore trần 1/n — ở n=6 và 12
     1c chỉ đồng thanh khoản nhất: top 20 / 40 / 80 theo khối lượng
     1d long k đồng tốt nhất + short BTC phòng hộ (k+1 đồng); 1e ngược lại
  2. SỐ LỆNH
     2a chu kỳ tái cân bằng 24–216h        2b dải không giao dịch 0–1,0
     2c vùng đệm thứ hạng 1,5x / 2x         2d làm mượt tín hiệu (bán rã 1/2/4 kỳ)
     2e cân một phần (đi 50% quãng đường mỗi kỳ, Gârleanu-Pedersen)
  3. CHIẾN LƯỢC ÍT ĐỒNG KHÁC: xu hướng BTC; xu hướng 5 đồng lớn; cặp ETH/BTC
  4. VÒNG 2: tổ hợp các lựa chọn tốt nhất; 5. ĐÒN BẨY × ngắt mạch; 6. VỐN; 7. CHI PHÍ

LUẬT CHỌN — CHỐT TRƯỚC: chỉ tiêu là VND/tuần trung vị (bootstrap 1 năm, CÓ ngắt mạch như
live). Chọn cấu hình có VND/tuần cao nhất; trong các cấu hình đạt >= 95% mức đó, lấy cấu
hình ÍT LỆNH/TUẦN nhất. Mỗi cấu hình có P(hơn gốc) từ bootstrap khối GHÉP CẶP — dưới 0,8
là trong nhiễu, không được coi là cải thiện.

⚠️ Cửa sổ đánh giá ~16 tháng. Quét nhiều cấu hình trên một cửa sổ ngắn là chọn lọc nhiều
lần: đọc P(hơn gốc), đừng đọc riêng con số lớn nhất.

    python scripts/all_levers_study.py                          # Binance, 3 triệu VND
    python scripts/all_levers_study.py --capital-vnd 5000000
    python scripts/all_levers_study.py --eval-start 2025-02-01  # chỉ kỷ nguyên >=100 cặp
"""
import argparse
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from aegis.research.backtest_v2 import CostModel, simulate_marked_to_market
from aegis.research.small_capital import (
    block_bootstrap_paths, drop_below_min_notional, evaluate_paths,
)
from aegis.research.strategy_v3 import combined_signal, config_from_json, load_v3_data
from aegis.research.trade_band import smooth_signal
from aegis.risk.portfolio import build_weights

DATA_ROOT = "data/binance"
UNIVERSE = "artifacts/universe_wide.json"
BASE_CONFIG = "artifacts/strategy_v3.json"
OUT_CSV = "artifacts/all_levers_study.csv"
EVAL_START = "2021-07-01"          # sau 1 năm khởi động tầng gộp; --eval-start 2025-02-01 = kỷ nguyên >=100 cặp
VND_PER_USD = 26_300.0
BARS_YEAR = 2190
BLOCK = 18
MIN_NOTIONAL = 5.0
MIN_DV_DAY = 5e6
MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")


# ---------------------------------------------------------------------------- dữ liệu
def load(data_root: str, universe: str):
    cfg0 = replace(config_from_json(BASE_CONFIG), n_tranches=1, universe_file=universe)
    data = load_v3_data(cfg0, data_root=data_root)
    close = data.close
    dv = (close * data.panel["volume"]).rolling(180, min_periods=90).sum() / 30.0
    return cfg0, data, dv


def liquid_mask(dv_marks: pd.DataFrame, topk=None) -> pd.DataFrame:
    ok = dv_marks >= MIN_DV_DAY
    if topk:
        ok &= dv_marks.where(ok).rank(axis=1, ascending=False) <= topk
    return ok


# ---------------------------------------------------------------------------- chạy
class Runner:
    def __init__(self, cfg0, data, dv, capital_usd, cost_bps, paths, eval_start=EVAL_START):
        self.cfg0, self.data, self.dv = cfg0, data, dv
        self.cap, self.cost, self.paths = capital_usd, cost_bps, paths
        self.sigs = {}
        self.eval_ms = int(pd.Timestamp(eval_start, tz="UTC").value // 10**6)
        self.base = None

    def sig(self, period: int) -> pd.DataFrame:
        if period not in self.sigs:
            self.sigs[period] = combined_signal(self.data, replace(self.cfg0, rebalance_every=period))
        return self.sigs[period]

    def weights_v3(self, period=18, n=12, mode="zscore_riskparity", max_w=0.20, topk=None,
                   exit_ratio=None, smooth=0) -> pd.DataFrame:
        s = self.sig(period)
        s = s.where(liquid_mask(self.dv.reindex(s.index), topk))
        if smooth:
            s = smooth_signal(s, smooth)
        port = replace(self.cfg0.portfolio, mode=mode, n_positions=n,
                       max_weight=1.0 if mode == "rank_binary" else max_w, top_frac=0.10,
                       exit_frac=0.10 * exit_ratio if exit_ratio else None)
        return build_weights(s, self.data.close.reindex(s.index), port)

    def weights_hedged(self, period=18, k=3, side="long") -> pd.DataFrame:
        s = self.sig(period)
        s = s.where(liquid_mask(self.dv.reindex(s.index))).drop(columns=["BTCUSDT"], errors="ignore")
        W = pd.DataFrame(0.0, index=s.index, columns=self.data.close.columns)
        for ts, row in s.iterrows():
            row = row.dropna()
            if len(row) < 2 * k:
                continue
            pick = row.nlargest(k).index if side == "long" else row.nsmallest(k).index
            sgn = 1.0 if side == "long" else -1.0
            W.loc[ts, pick] = sgn * 0.5 / k
            W.loc[ts, "BTCUSDT"] = -sgn * 0.5
        return W

    def weights_trend(self, coins, period=42, pair=False) -> pd.DataFrame:
        close = self.data.close
        marks = close.index[::period]
        W = pd.DataFrame(0.0, index=marks, columns=close.columns)
        if pair:
            ratio = close["ETHUSDT"] / close["BTCUSDT"]
            s = np.sign(ratio / ratio.shift(180) - 1.0).reindex(marks).fillna(0.0)
            W["ETHUSDT"], W["BTCUSDT"] = 0.5 * s, -0.5 * s
            return W
        for c in coins:
            p = close[c]
            s = (np.sign(p / p.shift(180) - 1.0) + np.sign(p / p.shift(540) - 1.0)) / 2.0
            W[c] = s.reindex(marks).fillna(0.0) / len(coins)
        return W

    def run(self, label, group, W, lev=1.5, band=0.2, partial=1.0, cost=None, cap=None) -> dict:
        cap = cap or self.cap
        cost = self.cost if cost is None else cost
        W = drop_below_min_notional(W, cap, lev, MIN_NOTIONAL)
        res = simulate_marked_to_market(
            W, self.data.close, self.data.funding, CostModel(flat_bps=cost), bar_hours=4.0,
            no_trade_band=band, min_trade=MIN_NOTIONAL / (cap * lev), partial=partial)
        keep = res.returns.index >= self.eval_ms
        r = res.returns[keep]
        rv = r.to_numpy()
        weeks = keep.sum() / 42.0
        orders = float(res.meta["orders"][keep].sum())
        turn = float(res.turnover[keep].sum())
        held = res.n_positions[keep]
        eq = np.cumprod(1.0 + lev * rv)
        seed = zlib.crc32(label.encode())
        paths = block_bootstrap_paths(rv, BARS_YEAR, self.paths, seed, BLOCK)
        ev = evaluate_paths(paths, lev, BLOCK)
        x = ev["median_live"]
        out = {"group": group, "label": label, "L": lev, "band": band,
               "sharpe": float(rv.mean() / rv.std(ddof=1) * np.sqrt(BARS_YEAR)),
               "ann_1x": float(rv.mean() * BARS_YEAR), "vol_1x": float(rv.std() * np.sqrt(BARS_YEAR)),
               "maxdd_L": float((eq / np.maximum.accumulate(eq) - 1).min()),
               "ret_window_L": float(eq[-1] - 1.0),
               "orders_week": orders / weeks, "coins": float(held[held > 0].mean()),
               "usd_order": turn * cap * lev / max(orders, 1.0),
               "vnd_week": ((x ** (1 / 52) - 1) if x > 0 else -1.0) * cap * VND_PER_USD,
               **ev, "_r": r}
        if self.base is not None:
            out["p_better"] = paired_p(self.base["_r"], r)
        return out


def paired_p(r0: pd.Series, r1: pd.Series, n=1000, seed=7) -> float:
    """P(Sharpe cấu hình > Sharpe gốc), bootstrap khối GHÉP CẶP trên cùng các nến."""
    a = pd.concat([r0, r1], axis=1).dropna().to_numpy()
    if len(a) < BLOCK * 10:
        return np.nan
    rng = np.random.default_rng(seed)
    nb = len(a) // BLOCK
    wins = 0
    for _ in range(n):
        st = rng.integers(0, len(a) - BLOCK, nb)
        idx = (st[:, None] + np.arange(BLOCK)).ravel()
        s = a[idx]
        sh = s.mean(0) / s.std(0, ddof=1)
        wins += sh[1] > sh[0]
    return wins / n


def show(rows, title):
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)
    print(f"{'cấu hình':<34}{'Sharpe':>7}{'lời/năm1x':>10}{'đồng':>6}{'lệnh/tuần':>10}{'$/lệnh':>8}"
          f"{'VND/tuần':>10}{'P(kill)':>8}{'lỗ 1n':>7}{'cửa sổ':>8}{'maxDD':>7}{'P(hơn gốc)':>11}")
    print("-" * 132)
    for o in rows:
        pb = o.get("p_better", np.nan)
        print(f"{o['label']:<34}{o['sharpe']:>7.2f}{o['ann_1x'] * 100:>9.0f}%{o['coins']:>6.1f}"
              f"{o['orders_week']:>10.1f}{o['usd_order']:>8.1f}{o['vnd_week']:>10,.0f}"
              f"{o['p_kill'] * 100:>7.0f}%{o['p_loss_live'] * 100:>6.0f}%"
              f"{o['ret_window_L'] * 100:>7.0f}%{o['maxdd_L'] * 100:>6.0f}%"
              f"{'' if np.isnan(pb) else f'{pb:.2f}':>11}")


def pick(rows):
    """Luật chốt trước: >= 95% VND/tuần tốt nhất, rồi ít lệnh nhất."""
    best = max(o["vnd_week"] for o in rows)
    near = [o for o in rows if o["vnd_week"] >= 0.95 * best] if best > 0 else rows
    return min(near, key=lambda o: (o["orders_week"], -o["vnd_week"]))


# ---------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--universe", default=UNIVERSE)
    ap.add_argument("--eval-start", default=EVAL_START)
    ap.add_argument("--capital-vnd", type=float, default=3_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=15.7)
    ap.add_argument("--paths", type=int, default=2000)
    a = ap.parse_args(argv)
    if not Path(a.data_root).exists():
        sys.exit(f"thiếu {a.data_root}: chạy trên máy có dữ liệu Binance, hoặc tải bằng "
                 f"scripts/download_wide_universe.py")
    cap = a.capital_vnd / VND_PER_USD
    cfg0, data, dv = load(a.data_root, a.universe)
    R = Runner(cfg0, data, dv, cap, a.cost_bps, a.paths, a.eval_start)
    close = data.close
    print(f"dữ liệu: {close.shape[1]} cặp, {pd.to_datetime(close.index[0], unit='ms').date()} -> "
          f"{pd.to_datetime(close.index[-1], unit='ms').date()}, chấm từ {a.eval_start}; "
          f"cặp đạt thanh khoản trung vị/kỳ: "
          f"{int((dv >= MIN_DV_DAY).sum(axis=1)[close.index >= R.eval_ms].median())}", flush=True)

    allrows = []
    base = R.run("GỐC: V3 12 đồng 72h dải0,2 1,5x", "gốc", R.weights_v3())
    R.base = base
    allrows.append(base)

    def grp(title, items):
        rows = [base] + [R.run(lab, title, W, **kw) for lab, W, kw in items]
        show(rows, title)
        allrows.extend(rows[1:])
        return rows

    g1 = grp("1a. SỐ VỊ THẾ", [(f"n={n}", R.weights_v3(n=n), {})
                               for n in (4, 6, 8, 10, 16, 20, 30, 40, 50)])
    grp("1b. CÁCH CHIA VỐN", [
        (f"n={n} đều tuyệt đối", R.weights_v3(n=n, mode="rank_binary"), {}) for n in (6, 12)] + [
        (f"n={n} zscore trần 1/n", R.weights_v3(n=n, max_w=1.0 / n), {}) for n in (6, 12)] + [
        ("n=6 V3", R.weights_v3(n=6), {})])
    grp("1c. CHỈ ĐỒNG THANH KHOẢN NHẤT", [(f"top {k} khối lượng", R.weights_v3(topk=k), {})
                                               for k in (20, 40, 80)])
    grp("1d/1e. LONG/SHORT VÀI ĐỒNG + PHÒNG HỘ BTC", [
        (f"long {k} + short BTC", R.weights_hedged(k=k, side="long"), {}) for k in (3, 6)] + [
        (f"short {k} + long BTC", R.weights_hedged(k=k, side="short"), {}) for k in (3, 6)])
    g2a = grp("2a. CHU KỲ TÁI CÂN BẰNG", [(f"{p * 4}h", R.weights_v3(period=p), {})
                                          for p in (6, 12, 24, 36, 54)])
    g2b = grp("2b. DẢI KHÔNG GIAO DỊCH", [(f"dải {b}", R.weights_v3(), {"band": b})
                                          for b in (0.0, 0.35, 0.5, 1.0)])
    grp("2c. VÙNG ĐỆM THỨ HẠNG", [(f"vùng đệm {x}x", R.weights_v3(exit_ratio=x), {})
                                        for x in (1.5, 2.0)])
    g2d = grp("2d. LÀM MƯỢT TÍN HIỆU", [(f"bán rã {h} kỳ", R.weights_v3(smooth=h), {})
                                        for h in (1, 2, 4)])
    grp("2e. CÂN MỘT PHẦN", [("đi 50% quãng đường", R.weights_v3(), {"partial": 0.5})])
    grp("3. CHIẾN LƯỢC ÍT ĐỒNG KHÁC (1 tuần/lần)", [
        ("xu hướng BTC", R.weights_trend(["BTCUSDT"]), {}),
        ("xu hướng 5 đồng lớn", R.weights_trend([c for c in MAJORS if c in close]), {}),
        ("cặp ETH/BTC", R.weights_trend([], pair=True), {})])

    # ------------------------------------------------------------ 4. vòng 2: tổ hợp
    def top2(rows, key):
        ok = sorted(rows, key=lambda o: -o["vnd_week"])[:2]
        return [key(o) for o in ok]

    ns = sorted(set(top2(g1, lambda o: 12 if o is base else int(o["label"][2:]))))
    periods = sorted(set(top2(g2a, lambda o: 18 if o is base else int(o["label"][:-1]) // 4)))
    bands = sorted(set(top2(g2b, lambda o: o["band"])))
    smooths = sorted(set(top2(g2d, lambda o: 0 if o is base else int(o["label"].split()[2]))))
    combo = []
    for n in ns:
        for p in periods:
            for b in bands:
                for h in smooths:
                    lab = f"n={n} {p * 4}h dải{b} mượt{h}"
                    combo.append((lab, R.weights_v3(n=n, period=p, smooth=h), {"band": b}))
    g4 = grp("4. VÒNG 2 — TỔ HỢP HAI LỰA CHỌN TỐT NHẤT MỖI NHÓM", combo)
    final = pick(g4)
    print(f"\n=> CHỌN (luật chốt trước): {final['label']}")

    # ------------------------------------------------------------ 5-7: đòn bẩy, vốn, chi phí
    fl = final["label"].split()
    n_f, p_f = int(fl[0][2:]), int(fl[1][:-1]) // 4
    b_f, h_f = float(fl[2][3:]), int(fl[3][4:])
    Wf = R.weights_v3(n=n_f, period=p_f, smooth=h_f)
    grp("5. ĐÒN BẨY (cấu hình đã chọn)", [(f"{final['label']} @{L}x", Wf, {"band": b_f, "lev": L})
                                             for L in (0.5, 1.0, 2.0, 2.5, 3.0)])
    grp("6. VỐN (1,5x)", [(f"vốn {v / 1e6:.0f} triệu", Wf,
                                {"band": b_f, "cap": v / VND_PER_USD})
                               for v in (1e6, 2e6, 5e6, 10e6)])
    grp("7. CHI PHÍ MỖI CHIỀU (1,5x)", [(f"chi phí {c}bp", Wf, {"band": b_f, "cost": c})
                                            for c in (4.5, 8.0, 25.0)])

    df = pd.DataFrame([{k: v for k, v in o.items() if k != "_r"} for o in allrows])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nđã ghi {OUT_CSV} ({len(df)} cấu hình)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
