#!/usr/bin/env python3
"""
CHIA LÔ TÁI CÂN BẰNG có đáng bật không — đo trên dữ liệu thật, cấu hình daemon thật.

BỐI CẢNH (replay 24/09/2026): cùng hệ thống, cùng 28 ngày, chỉ khác giờ tái cân bằng,
kết quả chạy từ −14% tới +231%; giờ của daemon rơi vào nhóm 5% xấu nhất. Chia lô
(Hoffstein, Faber & Braun 2020) thay MỘT lần rút thăm giờ bằng TRUNG BÌNH K lô lệch pha.

Script này đo trên lưới 4h (đường vốn thật, có sụt giảm trong kỳ):
  * MỘT LÔ ở cả 18 pha (18 giờ bắt đầu cách nhau 4h) — phân tán do chọn giờ.
  * CHIA LÔ K = 3 / 6 / 18, ở mọi pha còn lại của chính nó — phần may rủi còn sót.
Mỗi cấu hình: Sharpe kỷ nguyên >=100 cặp, lợi suất N ngày gần nhất ở đòn bẩy L,
sụt giảm N ngày, chi phí/năm (để thấy chia lô tốn thêm bao nhiêu phí).

═══════════════════════════════════════════════════════════════════════════════
LUẬT QUYẾT ĐỊNH — CHỐT TRƯỚC KHI CHẠY
═══════════════════════════════════════════════════════════════════════════════
Bật `n_tranches = K` khi đủ CẢ HAI:
  1. KHÔNG MẤT EDGE: Sharpe (kỷ nguyên >=100) của K >= trung vị Sharpe một lô − 0,05.
     Chia lô không tạo alpha; nó chỉ được phép tốn rất ít (phí đặt lại phần trôi giá).
  2. GIẢM MAY RỦI: khoảng (p90 − p10) của lợi suất N ngày qua các pha giảm >= 50% so với
     một lô.
Chọn K NHỎ NHẤT thoả cả hai — ít lần tái cân bằng hơn là ít rủi ro vận hành hơn (máy
Mac ngủ). ⚠️ Kỷ nguyên >=100 chồng lên holdout: đây là bằng chứng ĐỘ ỔN ĐỊNH; bật trên
testnet chính là phép kiểm tiến về phía trước.

    python scripts/tranche_study.py --cost-bps 15.7
    python scripts/tranche_study.py --days 28 --leverage 5 --cost-bps 15.7
    python scripts/tranche_study.py --synthetic      # CHỈ kiểm tra đường chạy
"""
import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from aegis.research.strategy_v3 import (
    WIDE_CONFIG_FILE, config_from_json, load_v3_data, run_v3_tranched,
)

MAKER = 0.39
BARS_PER_YEAR = 24 * 365 / 4.0
DAY_MS = 86_400_000
OUT_CSV = "artifacts/tranche_study.csv"
MAX_SHARPE_LOSS = 0.05
MIN_SPREAD_CUT = 0.50


def _compound(r: np.ndarray, lev: float) -> float:
    path = np.cumprod(np.maximum(1.0 + lev * r, 0.0))
    return float(path[-1] - 1.0) if len(path) else 0.0


def _maxdd(r: np.ndarray, lev: float) -> float:
    path = np.cumprod(np.maximum(1.0 + lev * r, 0.0))
    if not len(path):
        return 0.0
    return float((path / np.maximum.accumulate(path) - 1.0).min())


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else np.nan


def measure(res, hi_idx, lo_ms: int, lev: float) -> dict:
    r = res.returns.dropna()
    hi = r[r.index.isin(hi_idx)].to_numpy()
    rec = r[r.index >= lo_ms].to_numpy()
    cost = res.cost_drag.reindex(r.index)
    return {"sharpe_hi": _sharpe(hi), "ret_recent": _compound(rec, lev),
            "dd_recent": _maxdd(rec, lev),
            "cost_ann": float(cost[cost.index.isin(hi_idx)].mean() * BARS_PER_YEAR)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=28.0)
    ap.add_argument("--leverage", type=float, default=5.0)
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="chi phí một chiều ĐO THẬT (vd 15.7); bỏ trống = mô hình")
    ap.add_argument("--config", default=WIDE_CONFIG_FILE)
    ap.add_argument("--synthetic", action="store_true",
                    help="panel giả lập — CHỈ kiểm tra đường chạy, KHÔNG phải kết quả")
    a = ap.parse_args(argv)

    cfg = replace(config_from_json(a.config), n_tranches=1)
    step = cfg.rebalance_every
    if a.synthetic:
        print("!" * 100)
        print("!! CHẾ ĐỘ TỔNG HỢP — số liệu GIẢ, chỉ chứng minh script chạy hết đường.")
        print("!" * 100)
        sys.path.insert(0, "scripts")
        from residual_study import synthetic_data
        data = synthetic_data(n_bars=6 * 365 * 2, n_assets=80)
    else:
        data = load_v3_data(cfg)

    idx = data.index
    lo_ms = int(idx[-1]) - int(a.days * DAY_MS)
    hi = data.close.notna().sum(axis=1) >= 100
    hi_idx = set(hi.index[hi.values]) if hi.any() else set(idx)
    cache: dict = {}                       # trọng số từng pha: tầng gộp chạy đúng `step` lần

    def run(k, phase):
        res = run_v3_tranched(data, cfg, n_tranches=k, phase=phase, maker_ratio=MAKER,
                              cost_bps=a.cost_bps, cache=cache)
        return {"k": k, "phase": phase, **measure(res, hi_idx, lo_ms, a.leverage)}

    rows = []
    for p in range(step):
        rows.append(run(1, p))
        print(f"  một lô, pha {p:>2}: {rows[-1]['ret_recent']*100:+7.1f}%", flush=True)
    for k in [k for k in (3, 6, step) if step % k == 0 and k > 1]:
        for p in range(step // k):
            rows.append(run(k, p))
    df = pd.DataFrame(rows)

    print("\n" + "=" * 104)
    print(f"CHIA LÔ vs MỘT LÔ — cấu hình {a.config}, maker {MAKER}, chi phí "
          f"{'mô hình' if a.cost_bps is None else f'{a.cost_bps}bp/chiều'}, "
          f"{a.days:.0f} ngày gần nhất ở {a.leverage:g}x")
    print("=" * 104)
    print(f"{'':<16}{'số pha':>7}{'Sharpe tv':>10}{'Sharpe min':>11}"
          f"{f'{a.days:.0f}n tv':>10}{'p10':>9}{'p90':>9}{'tệ nhất':>9}{'tốt nhất':>10}"
          f"{'p90-p10':>9}{'phí/năm':>9}")
    print("-" * 104)
    summary = {}
    for k, g in df.groupby("k"):
        rr = g["ret_recent"].to_numpy()
        spread = np.percentile(rr, 90) - np.percentile(rr, 10)
        summary[k] = {"sharpe_med": g["sharpe_hi"].median(), "spread": spread}
        lab = "MỘT LÔ" if k == 1 else f"chia {k} lô"
        print(f"{lab:<16}{len(g):>7}{g['sharpe_hi'].median():>10.2f}"
              f"{g['sharpe_hi'].min():>11.2f}"
              f"{np.median(rr)*100:>9.1f}%{np.percentile(rr, 10)*100:>8.1f}%"
              f"{np.percentile(rr, 90)*100:>8.1f}%{rr.min()*100:>8.1f}%{rr.max()*100:>9.1f}%"
              f"{spread*100:>8.1f}%{g['cost_ann'].median()*100:>8.1f}%")

    print("\n" + "=" * 104)
    print(f"PHÁN QUYẾT (luật chốt trước): Sharpe >= trung vị một lô − {MAX_SHARPE_LOSS}, "
          f"và p90−p10 giảm >= {MIN_SPREAD_CUT:.0%}")
    print("=" * 104)
    base = summary[1]
    chosen = None
    for k in sorted(k for k in summary if k > 1):
        s = summary[k]
        c1 = s["sharpe_med"] >= base["sharpe_med"] - MAX_SHARPE_LOSS
        c2 = s["spread"] <= base["spread"] * (1 - MIN_SPREAD_CUT)
        ok = c1 and c2
        chosen = chosen or (k if ok else None)
        print(f"  chia {k:>2} lô: Sharpe {'✓' if c1 else '✗'}  giảm may rủi {'✓' if c2 else '✗'}"
              f"  =>  {'ĐẠT' if ok else 'không đạt'}")
    if chosen:
        print(f"\n  => BẬT n_tranches = {chosen}: thêm \"n_tranches\": {chosen} vào khối "
              f"\"config\" của {a.config}, rồi NẠP LẠI daemon.")
    else:
        print("\n  => KHÔNG bật: không K nào đạt cả hai điều kiện.")

    if not a.synthetic:
        df.to_csv(OUT_CSV, index=False)
        print(f"\nđã ghi {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
