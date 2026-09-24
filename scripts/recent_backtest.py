#!/usr/bin/env python3
"""
LỜI LỖ CỦA HỆ THỐNG ĐANG CHẠY trong N ngày gần nhất — backtest công bằng, đặt cạnh ví thật.

Ba điều kiện để phép đo này CÔNG BẰNG, và script cưỡng chế cả ba:

  1. ĐÚNG CẤU HÌNH LIVE. Cấu hình dựng bằng `config_from_json(strategy_v3_wide.json)` —
     cùng file, cùng phép ánh xạ daemon dùng. [F51] Mọi nghiên cứu wide trước đây dùng
     `max_step=0,05` của V3 trong khi daemon chạy `0,025`; script này đo CẢ HAI để
     biết khoảng lệch đó đáng bao nhiêu.
  2. TÍNH TOÀN DÒNG THỜI GIAN RỒI CẮT SAU (F35). Tầng gộp thích ứng phụ thuộc đường đi;
     cắt dữ liệu về 4 tuần rồi mới chạy là khởi động lại nó từ con số 0.
  3. CHI PHÍ THẬT: maker 0,39 đo ở lượt 19/09 (không phải 0,50 giả định trong artifact),
     trượt giá theo từng cặp `estimate_cost_bps`. Đường vốn đo trên nến 4h
     (`run_v3_fine`) để thấy cả sụt giảm TRONG kỳ, không chỉ điểm đầu-cuối 72h.

Rồi đặt cạnh equity THẬT trong `artifacts/execution_log.jsonl` cùng khoảng thời gian.
⚠️ So sánh đó chỉ XẤP XỈ: backtest tái cân bằng ở mốc 72h cố định, còn live tái cân bằng
khi máy thức (khoảng cách thật 3,2-3,6 ngày), nên hai bên không cầm cùng một danh mục ở
mọi thời điểm. Chênh lệch lớn và CÙNG CHIỀU qua nhiều khoảng mới là tín hiệu.

⚠️ Vài tuần KHÔNG kiểm định được edge (cần ~0,8-2,2 năm cho t = 2). Thứ script này trả
lời được là: "live có đang đi theo mô phỏng không", và "mô phỏng nói gì về tháng qua".

    python scripts/recent_backtest.py                       # 28 ngày, 5,0x như daemon
    python scripts/recent_backtest.py --days 14 --leverage 5
    python scripts/recent_backtest.py --cost-bps 15.7        # chi phí đo thật (replay 24/09)
    python scripts/recent_backtest.py --cost-bps 15.7 --tranches 3   # hệ thống chia 3 lô
    python scripts/recent_backtest.py --synthetic           # CHỈ kiểm tra đường chạy
"""
import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from aegis.research.strategy_v3 import (
    WIDE_CONFIG_FILE, config_from_json, load_v3_data, run_v3, run_v3_fine,
)

MAKER = 0.39
EXEC_LOG = "artifacts/execution_log.jsonl"
OUT_CSV = "artifacts/recent_backtest.csv"
BAR_MS = 4 * 3_600_000
DAY_MS = 86_400_000
VND_CAPITAL = 1_000_000


def compound(r: np.ndarray, lev: float) -> float:
    """Vốn cuối / vốn đầu - 1 ở đòn bẩy `lev`, chặn ở -100% (cháy là hết, không âm)."""
    path = np.cumprod(np.maximum(1.0 + lev * r, 0.0))
    return float(path[-1] - 1.0) if len(path) else 0.0


def max_drawdown(r: np.ndarray, lev: float) -> float:
    path = np.cumprod(np.maximum(1.0 + lev * r, 0.0))
    if not len(path):
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], path]))[1:]
    return float((path / peak - 1.0).min())


def live_snapshots(lo_ms: int, hi_ms: int) -> pd.DataFrame:
    try:
        rows = [json.loads(x) for x in open(EXEC_LOG, encoding="utf-8") if x.strip()]
    except FileNotFoundError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df.timestamp_ms >= lo_ms) & (df.timestamp_ms <= hi_ms)].copy()
    df["gross_lev"] = df["gross_notional"] / df["equity"]
    return df.sort_values("timestamp_ms").reset_index(drop=True)


def _sharpe(r: np.ndarray, ppy: float) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=28.0)
    ap.add_argument("--leverage", type=float, default=5.0,
                    help="đòn bẩy gộp daemon chạy (cờ --leverage của run_daily.py)")
    ap.add_argument("--maker", type=float, default=MAKER)
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="chi phí một chiều ĐO THẬT (vd 15.7 từ replay 24/09); "
                         "bỏ trống = mô hình phí+spread (~4,5bp)")
    ap.add_argument("--config", default=WIDE_CONFIG_FILE)
    ap.add_argument("--tranches", type=int, default=None,
                    help="số lô tái cân bằng lệch pha (ghi đè n_tranches của cấu hình); "
                         "3 = ba lô lệch 24h, chống timing luck")
    ap.add_argument("--synthetic", action="store_true",
                    help="panel giả lập — CHỈ kiểm tra đường chạy, KHÔNG phải kết quả")
    a = ap.parse_args(argv)

    cfg = config_from_json(a.config)
    if a.tranches is not None:
        cfg = replace(cfg, n_tranches=a.tranches)
    cfg_v3step = replace(cfg, combiner=replace(cfg.combiner, max_step=0.05))

    if a.synthetic:
        print("!" * 100)
        print("!! CHẾ ĐỘ TỔNG HỢP — số liệu GIẢ, chỉ chứng minh script chạy hết đường.")
        print("!! KHÔNG ghi artifact.")
        print("!" * 100)
        sys.path.insert(0, "scripts")
        from residual_study import synthetic_data
        data = synthetic_data(n_bars=6 * 365 * 2)
    else:
        data = load_v3_data(cfg)                  # KHÔNG ghim mốc: muốn dữ liệu MỚI NHẤT

    idx = data.index
    end_ms = int(idx[-1])
    lo_ms = end_ms - int(a.days * DAY_MS)
    mask = np.asarray(idx >= lo_ms)
    age_h = (time.time() * 1000 - (end_ms + BAR_MS)) / 3_600_000

    print("=" * 100)
    print(f"LỜI LỖ HỆ THỐNG ĐANG CHẠY — {a.days:.0f} ngày gần nhất | cấu hình {a.config}")
    print("=" * 100)
    print(f"  n={cfg.n_positions}, max_w={cfg.portfolio.max_weight}, "
          f"max_step={cfg.combiner.max_step}, tái cân bằng {cfg.period_hours:.0f}h, "
          f"{cfg.n_tranches} lô, maker {a.maker}, đòn bẩy {a.leverage}x, chi phí "
          f"{'mô hình' if a.cost_bps is None else f'{a.cost_bps}bp/chiều (đo thật)'}")
    print(f"  dữ liệu tới {pd.Timestamp(end_ms + BAR_MS, unit='ms')} UTC", end="")
    if not a.synthetic:
        stale = "  ⚠️ CŨ — cập nhật dữ liệu trước" if age_h > 12 else ""
        print(f"  ({age_h:.1f}h trước){stale}")
    else:
        print()

    # ---- 1. Đường vốn 4h, TOÀN dòng thời gian rồi cắt ------------------------------
    fine = run_v3_fine(data, cfg, mask=mask, maker_ratio=a.maker, cost_bps=a.cost_bps)
    r = fine.returns.dropna()
    rv = r.to_numpy()
    tot1, totL = compound(rv, 1.0), compound(rv, a.leverage)
    print(f"\n1. BACKTEST {a.days:.0f} NGÀY (nến 4h, đã trừ phí + trượt giá + funding)")
    print(f"   {'':<22}{'1x':>10}{f'{a.leverage:g}x':>10}")
    print(f"   {'lợi suất cộng dồn':<22}{tot1*100:>9.2f}%{totL*100:>9.2f}%")
    print(f"   {'sụt giảm tối đa':<22}{max_drawdown(rv, 1.0)*100:>9.2f}%"
          f"{max_drawdown(rv, a.leverage)*100:>9.2f}%")
    print(f"   {'chi phí (1x, cộng dồn)':<22}{fine.cost_drag.sum()*100:>9.2f}%"
          f"{fine.cost_drag.sum()*a.leverage*100:>9.2f}%")
    print(f"   {'funding (1x, cộng dồn)':<22}{fine.funding_pnl.sum()*100:>9.2f}%"
          f"{fine.funding_pnl.sum()*a.leverage*100:>9.2f}%")
    print(f"   => trên vốn {VND_CAPITAL:,.0f} VND ở {a.leverage:g}x: {totL*VND_CAPITAL:+,.0f} VND")

    print(f"\n   theo tuần ({a.leverage:g}x):")
    wk = pd.Series(rv, index=r.index)
    t0 = int(r.index[0])
    for k in range(int(np.ceil(a.days / 7))):
        part = wk[(wk.index >= t0 + k * 7 * DAY_MS) & (wk.index < t0 + (k + 1) * 7 * DAY_MS)]
        if len(part):
            print(f"     {pd.Timestamp(int(part.index[0]), unit='ms'):%d/%m} -> "
                  f"{pd.Timestamp(int(part.index[-1]) + BAR_MS, unit='ms'):%d/%m}: "
                  f"{compound(part.to_numpy(), a.leverage)*100:+7.2f}%")

    # ---- 2. Cạnh ví thật -----------------------------------------------------------
    live = live_snapshots(lo_ms, end_ms + BAR_MS)
    print("\n2. ĐẶT CẠNH EQUITY THẬT (execution_log) — XẤP XỈ, xem docstring")
    if len(live) < 2:
        print("   (không đủ ảnh chụp equity trong khoảng này)")
    else:
        print(f"   {'từ':<17}{'tới':<17}{'gross live':>11}{'LIVE':>9}{'BACKTEST @gross live':>22}")
        for i in range(len(live) - 1):
            a0, a1 = live.iloc[i], live.iloc[i + 1]
            t_a, t_b = int(a0.timestamp_ms), int(a1.timestamp_ms)
            seg = wk[(wk.index >= t_a) & (wk.index + BAR_MS <= t_b)].to_numpy()
            bt = compound(seg, float(a0.gross_lev)) if len(seg) else np.nan
            bt_txt = f"{bt*100:.2f}%" if np.isfinite(bt) else "— (< 1 nến 4h)"
            print(f"   {pd.Timestamp(t_a, unit='ms'):%d/%m %H:%M}      "
                  f"{pd.Timestamp(t_b, unit='ms'):%d/%m %H:%M}      {a0.gross_lev:>8.2f}x"
                  f"{(a1.equity / a0.equity - 1)*100:>8.2f}%{bt_txt:>22}")

    # ---- 3. F51: max_step live (0,025) so với giá trị đã được đo (0,05) ----------------
    ppy = cfg.periods_per_year
    hi = data.close.notna().sum(axis=1) >= 100
    hi_idx = set(hi.index[hi.values])
    print(f"\n3. [F51] max_step {cfg.combiner.max_step} (DAEMON ĐANG CHẠY) so với 0,05 "
          "(mọi Sharpe wide đã báo cáo)")
    print(f"   {'':<22}{'Sharpe >=100 cặp':>18}{'Sharpe toàn bộ':>16}"
          f"{f'{a.days:.0f} ngày @1x':>14}")
    variants = ((f"max_step {cfg.combiner.max_step} (live)", replace(cfg, n_tranches=1)),
                ("max_step 0,05", replace(cfg_v3step, n_tranches=1)))
    for lab, c in variants:
        rr = run_v3(data, c, maker_ratio=a.maker, cost_bps=a.cost_bps).returns.dropna()
        s_hi = _sharpe(rr[rr.index.isin(hi_idx)].to_numpy(), ppy)
        s_all = _sharpe(rr.to_numpy(), ppy)
        rec = rr[rr.index >= lo_ms].to_numpy()
        print(f"   {lab:<22}{s_hi:>18.2f}{s_all:>16.2f}{compound(rec, 1.0)*100:>13.2f}%")

    if not a.synthetic:
        out = pd.DataFrame({"ts": r.index, "ret_1x": rv,
                            "cost": fine.cost_drag.reindex(r.index).values,
                            "funding": fine.funding_pnl.reindex(r.index).values})
        out.to_csv(OUT_CSV, index=False)
        print(f"\nđã ghi {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
