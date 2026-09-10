#!/usr/bin/env python3
"""
Bộ săn edge có kiểm soát đa kiểm định.

Quét một lưới giả thuyết qua CPCV trung thực (fit từng fold, purge phủ trọn thời
gian nắm giữ), rồi CHIẾT KHẤU kết quả bằng Deflated Sharpe Ratio với num_trials =
đúng số cấu hình đã thử. Không có bước này, thử 12 cấu hình rồi chọn cái tốt nhất
chính là quá khớp có tổ chức.

    python scripts/edge_hunt.py

Thiết kế: thời gian nắm giữ giữ CỐ ĐỊNH theo đồng hồ (~5 ngày) khi đổi khung thời
gian — nếu không sẽ lẫn lộn biến "khung nến" với biến "thời gian gồng lệnh".
"""
import argparse
import itertools
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

from aegis.core.config_loader import load_canonical_config
from aegis.data.ingestion.binance_history import load_klines
from aegis.features.signal_pipeline import AegisSignalEngine
from aegis.pipelines.cpcv_pipeline import CPCVPipeline
from aegis.validation.dsr import compute_deflated_sharpe_ratio

# Số nến tương ứng ~5 ngày nắm giữ, theo từng khung.
BARS_PER_5D = {"1h": 120, "4h": 30, "1d": 5}
BARS_PER_YEAR = {"1h": 8760, "4h": 2190, "1d": 365}


def build_signal_bars(symbol: str, interval: str) -> pd.DataFrame:
    """Dựng (và cache) signal bars từ nến thô đã tải."""
    cache = pathlib.Path(f"data/binance/{symbol}_{interval}_signal.parquet")
    if cache.is_file():
        return pd.read_parquet(cache)

    raw = load_klines(symbol, interval)
    bars = AegisSignalEngine(symbol=symbol, warmup_window=50).process_ohlcv_to_signal_bars(raw)
    bars.to_parquet(cache, index=False)
    return bars


def run_config(bars: pd.DataFrame, interval: str, direction: int, c_trade: float) -> dict:
    """Chạy CPCV cho một cấu hình, trả về thống kê giao dịch."""
    hold = BARS_PER_5D[interval]

    cfg = load_canonical_config()
    cfg.update({
        "fast_mode": True,
        "primary_direction": direction,
        "c_trade": c_trade,
        "taker_fee_rate": c_trade,
        "t_max_live_follow": hold,
        "t_max_live_fade": max(4, hold // 3),
        "fade_enabled": False,   # cô lập biến: chiều lệnh do primary_direction quyết
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CPCVPipeline(n_groups=6, n_test_groups=2, embargo_bars=24, config=cfg).run(
            bars, num_bins=5
        )

    recs = res["clean_records"]
    m = res["metrics"]
    if len(recs) < 30:
        return {"n_trades": len(recs), "viable": False, "n_fits": m["n_fits"]}

    # ------------------------------------------------------------------
    # GỘP VỀ SỰ KIỆN DUY NHẤT — bước bắt buộc, nếu không sẽ tự lừa mình
    # ------------------------------------------------------------------
    # CPCV C(6,2)=15 fold, mỗi sự kiện nằm trong test fold ~5 lần, nên `clean_records`
    # đếm TRÙNG mỗi sự kiện ~3.5-5 lần. Dùng thẳng số đó làm cỡ mẫu sẽ thổi phồng
    # ý nghĩa thống kê lên ~2x và biến nhiễu thành "edge".
    raw_n = len(recs)
    per_event = {}
    for rec in recs:
        per_event.setdefault(int(rec["entry_idx"]), []).append(float(rec["realized_return"]))
    r = np.array([float(np.mean(v)) for v in per_event.values()], dtype=np.float64)
    wins, losses = r[r > 0], r[r <= 0]
    p = len(wins) / len(r)
    payoff = (wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan

    std = r.std(ddof=1)
    # Tần suất tính trên SỰ KIỆN DUY NHẤT, không phải bản ghi thô.
    years = BARS_PER_YEAR[interval]
    trades_per_year = len(r) / (len(bars) / years) if len(bars) > 0 else 0.0
    sharpe = (r.mean() / std * np.sqrt(trades_per_year)) if std > 1e-12 else 0.0

    # t-stat cho giả thuyết H0: lợi suất kỳ vọng = 0. Đây mới là thước đo
    # "có edge hay không", chứ không phải Sharpe hay lợi suất cộng dồn.
    t_stat = r.mean() / (std / np.sqrt(len(r))) if std > 1e-12 else 0.0

    return {
        "viable": True,
        "n_fits": m["n_fits"],
        "n_starved": m["n_folds_starved"],
        "n_raw_records": raw_n,
        "n_trades": len(r),
        "dup_factor": raw_n / max(len(r), 1),
        "mean_ret": float(r.mean()),
        "win_rate": float(p),
        "payoff": float(payoff),
        "payoff_needed": float((1 - p) / p) if p > 0 else np.inf,
        "sharpe": float(sharpe),
        "t_stat": float(t_stat),
        "trades_per_year": float(trades_per_year),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Săn edge có chiết khấu đa kiểm định")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--intervals", nargs="+", default=["1h", "4h", "1d"])
    args = parser.parse_args(argv)

    directions = [(1, "follow"), (-1, "mean-rev")]
    costs = [(0.0004, "taker"), (0.0001, "maker")]

    grid = list(itertools.product(args.intervals, directions, costs))
    print(f"Săn edge trên {args.symbol}: {len(grid)} cấu hình\n")
    print(f"{'khung':<6}{'chiều':<10}{'phí':<7}{'sựkiện':>8}{'win':>7}{'payoff':>8}"
          f"{'cần':>7}{'ret/lệnh':>11}{'sharpe':>8}{'t-stat':>8}")
    print("-" * 80)

    results = []
    for interval in args.intervals:
        bars = build_signal_bars(args.symbol, interval)
        for (direction, dname), (c_trade, cname) in itertools.product(directions, costs):
            stats = run_config(bars, interval, direction, c_trade)
            if not stats["viable"]:
                print(f"{interval:<6}{dname:<10}{cname:<7}{stats['n_trades']:>7}"
                      f"   (quá ít lệnh, fit {stats['n_fits']} fold)")
                continue
            stats.update(interval=interval, direction=dname, cost=cname)
            results.append(stats)
            flag = " *" if abs(stats["t_stat"]) > 2 else ""
            print(f"{interval:<6}{dname:<10}{cname:<7}{stats['n_trades']:>8}"
                  f"{stats['win_rate']*100:>6.1f}%{stats['payoff']:>8.2f}"
                  f"{stats['payoff_needed']:>7.2f}{stats['mean_ret']*100:>10.4f}%"
                  f"{stats['sharpe']:>8.2f}{stats['t_stat']:>8.2f}{flag}")

    if not results:
        print("\nKhông cấu hình nào đủ số lệnh để kết luận.")
        return 1

    # ------------------------------------------------------------------
    # CHIẾT KHẤU ĐA KIỂM ĐỊNH — bước không được bỏ qua
    # ------------------------------------------------------------------
    sharpes = np.array([r["sharpe"] for r in results], dtype=np.float64)
    best = max(results, key=lambda r: r["sharpe"])
    var_srs = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 1e-12

    print("\n" + "=" * 72)
    print(f"TỐT NHẤT: {best['interval']} / {best['direction']} / {best['cost']}")
    print(f"  ret/lệnh {best['mean_ret']*100:+.4f}%  |  sharpe {best['sharpe']:.2f}  "
          f"|  t-stat {best['t_stat']:.2f}  |  {best['n_trades']} sự kiện duy nhất "
          f"(gộp từ {best['n_raw_records']} bản ghi, hệ số lặp {best['dup_factor']:.2f}x)")
    if abs(best["t_stat"]) <= 2:
        print(f"\n  ⚠️  t-stat {best['t_stat']:.2f} < 2: lợi suất KHÔNG phân biệt được với 0.")

    if best["sharpe"] > 0:
        dsr = compute_deflated_sharpe_ratio(
            sr_estimated=best["sharpe"],
            variance_of_srs=max(var_srs, 1e-12),
            sample_length=best["n_trades"],
            num_trials=len(results),
        )
        print(f"\n  DSR (chiết khấu {len(results)} lần thử) = {dsr['dsr']:.4f}")
        print(f"  SR kỳ vọng tối đa do may rủi  = {dsr['sr_expected_max']:.2f}")
        print(f"  => {'✅ VƯỢT ngưỡng 0.95' if dsr['dsr'] >= 0.95 else '❌ KHÔNG vượt ngưỡng 0.95 — chưa đủ bằng chứng có edge'}")
    else:
        print("\n  ❌ Không cấu hình nào có Sharpe dương. Không cần chiết khấu DSR —")
        print("     không có gì để chiết khấu. Chiến lược này không có edge trên dữ liệu đã thử.")

    pd.DataFrame(results).to_csv("artifacts/edge_hunt_results.csv", index=False)
    print("\nchi tiết -> artifacts/edge_hunt_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
