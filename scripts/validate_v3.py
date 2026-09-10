#!/usr/bin/env python3
"""
KIỂM ĐỊNH CUỐI — CHẠY ĐÚNG MỘT LẦN TRÊN HOLDOUT.

Cấu hình dưới đây đã được chốt HOÀN TOÀN trên tập train qua `scripts/stability.py`,
và được chọn theo ĐỘ ỔN ĐỊNH qua 5 giai đoạn con chứ không theo Sharpe cao nhất.
Từ thời điểm script này chạy, mọi việc quay lại chỉnh tham số rồi chạy lại đều biến
holdout thành train và xoá sạch giá trị của con số nhận được.

Script trả lời bốn câu, theo thứ tự:
  1. Edge có sống sót ngoài mẫu không? (holdout)
  2. Sau khi chiết khấu ~250 phép thử, còn lại gì? (Deflated Sharpe Ratio)
  3. Chi phí phải xấu tới đâu thì chiến lược chết? (stress chi phí)
  4. Với vốn thật và mục tiêu thật, xác suất là bao nhiêu? (bảng đòn bẩy)
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
from aegis.research.leverage import (
    LeverageSpec, apply_leverage, kelly_leverage, required_sharpe_for_target,
    target_probability_table,
)
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights
from aegis.validation.dsr import compute_deflated_sharpe_ratio

# ============================ CẤU HÌNH CHỐT ============================
UNIVERSE = "artifacts/universe_wide.json"
INTERVAL = "4h"
REBAL = 18                 # 18 nến 4h = 72h giữa hai lần tái cân bằng
N_POSITIONS = 12           # 6 long + 6 short
MAKER_RATIO = 0.50         # giả định BI QUAN; testnet đo 0.378, mục tiêu 0.85
COMBINER = CombinerSpec(lookback=500, min_periods=120, t_threshold=2.0,
                        max_abs_weight=0.20, max_step=0.05)
PORTFOLIO = PortfolioSpec(mode="zscore_riskparity", n_positions=N_POSITIONS,
                          max_weight=0.20, beta_neutral=False, vol_window=60)
# Số cấu hình đã thử trong toàn đợt nghiên cứu này — đếm bảo thủ, làm tròn LÊN.
NUM_TRIALS = 250
PPY = 24 * 365.0 / (4 * REBAL)
CAPITAL_USD = 38.0         # 1.000.000 VND
MIN_NOTIONAL = 5.0
# =======================================================================


def stats_line(label, s, width=30):
    print(f"{label:<{width}}{s['n']:>7.0f}{s['ann_return']*100:>9.1f}%{s['ann_vol']*100:>8.1f}%"
          f"{s['sharpe']:>8.2f}{s['t_stat']:>8.2f}{s['max_dd']*100:>8.1f}%"
          f"{s['hit_rate']*100:>7.1f}%{s['avg_turnover']:>7.2f}")


def main():
    print("nạp dữ liệu...", flush=True)
    syms = json.load(open(UNIVERSE))
    panel = load_panel_v2(syms, INTERVAL, source_interval="1h", min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)
    close_full = panel["close"]
    split = json.load(open("artifacts/holdout_split.json"))["split_ts"]

    # Tín hiệu tính trên TOÀN chuỗi (cần warm-up) rồi mới cắt — hợp lệ vì nhân quả.
    print(f"dựng {len(SIGNAL_REGISTRY)} tín hiệu trên {close_full.shape[1]} cặp...", flush=True)
    sigs_full = {n: build_signal(n, panel, funding) for n in SIGNAL_REGISTRY}

    per_sym = estimate_cost_bps(panel, notional_usd=CAPITAL_USD * 3 / N_POSITIONS,
                                bars_per_day=6)

    def run(mask, maker_ratio=MAKER_RATIO):
        """
        Tính NHÂN QUẢ trên TOÀN dòng thời gian rồi mới cắt lợi suất về đoạn cần đo.

        [SỬA LỖI ĐO LƯỜNG] Bản đầu làm ngược: cắt lưới về holdout TRƯỚC rồi mới tính.
        Hệ quả là tầng gộp thích ứng khởi động lại từ con số 0 ở đầu holdout — 120 kỳ
        đầu chạy bằng trọng số đều, và cửa sổ học chỉ còn <=219 mốc thay vì 500. Đó
        KHÔNG phải điều live sẽ trải qua: live luôn có toàn bộ lịch sử trong tay.
        Sai lệch do lỗi này rất lớn: Sharpe holdout 0.71 (sai) so với 1.28 (đúng).

        Cắt sau là hợp lệ và đã được kiểm chứng: đổi TOÀN BỘ dữ liệu sau một mốc bất
        kỳ không làm đổi một chữ số nào của lợi suất trước mốc đó (sai khác = 0.0).
        """
        marks = close_full.index[::REBAL]
        close = close_full.reindex(marks)
        sigs = {k: v.reindex(marks) for k, v in sigs_full.items()}
        sig = combine_adaptive(sigs, close, COMBINER, top_frac=0.10)
        W = build_weights(sig, close, PORTFOLIO)
        cost = CostModel(maker_ratio=maker_ratio, half_spread_bps=0.0,
                         per_symbol_bps=per_sym * (1 - maker_ratio), min_bps=0.5)
        res = simulate(W, close, funding.reindex(marks), cost,
                       bar_hours=4 * REBAL, rebalance_every=1)

        keep = np.isin(res.returns.index, close_full.index[mask])
        res.returns = res.returns[keep]
        res.gross_returns = res.gross_returns[keep]
        res.cost_drag = res.cost_drag[keep]
        res.funding_pnl = res.funding_pnl[keep]
        res.turnover = res.turnover[keep]
        res.n_positions = res.n_positions[keep]
        res.net_exposure = res.net_exposure[keep]
        res.gross_exposure = res.gross_exposure[keep]
        return res

    idx = close_full.index
    res_tr = run(idx < split)
    res_ho = run(idx >= split)
    s_tr, s_ho = res_tr.stats(PPY), res_ho.stats(PPY)

    print("\n" + "=" * 96)
    print(f"CẤU HÌNH CHỐT: universe rộng ({close_full.shape[1]} cặp) | {INTERVAL} | "
          f"tái cân bằng {REBAL} nến ({4*REBAL}h)")
    print(f"  {len(SIGNAL_REGISTRY)} tín hiệu / 5 họ, gộp thích ứng (lookback {COMBINER.lookback}, "
          f"ngưỡng t {COMBINER.t_threshold})")
    print(f"  {N_POSITIONS} vị thế, trọng số liên tục chia đều rủi ro, trần {PORTFOLIO.max_weight:.0%}/cặp")
    print(f"  phí: maker {MAKER_RATIO:.0%} | trượt giá trung vị {per_sym.median():.2f}bp/cặp")
    print("=" * 96)
    print(f"{'':<30}{'kỳ':>7}{'ann':>10}{'vol':>8}{'sharpe':>8}{'t-stat':>8}"
          f"{'maxDD':>9}{'thắng':>8}{'turn':>7}")
    print("-" * 96)
    stats_line("TRAIN (đã dùng để chốt)", s_tr)
    stats_line("HOLDOUT (chưa từng nhìn)", s_ho)
    print("-" * 96)
    keep = s_ho["sharpe"] / s_tr["sharpe"] if s_tr["sharpe"] > 0 else 0.0
    print(f"Sharpe giữ lại ngoài mẫu: {keep*100:.0f}%  "
          f"(suy giảm 30-50% là bình thường và lành mạnh)")

    # ---- 2. Deflated Sharpe ----
    r_ho = res_ho.returns.dropna()
    sr_p = float(r_ho.mean() / r_ho.std(ddof=1))
    # variance_of_srs = phương sai Sharpe QUA CÁC PHÉP THỬ, lấy từ lưới đã chạy.
    try:
        grid = pd.read_csv("artifacts/stability.csv")
        var_srs = float((grid["sharpe_toàn"] / np.sqrt(PPY)).var(ddof=1))
    except Exception:
        var_srs = float((0.5 / np.sqrt(PPY)) ** 2)

    dsr = compute_deflated_sharpe_ratio(
        sr_estimated=sr_p, variance_of_srs=var_srs, sample_length=int(len(r_ho)),
        num_trials=NUM_TRIALS, skewness=float(r_ho.skew()),
        kurtosis=float(r_ho.kurtosis() + 3.0))

    print(f"\nDEFLATED SHARPE (chiết khấu {NUM_TRIALS} phép thử, tính trên HOLDOUT):")
    print(f"  Sharpe/kỳ đo được          {sr_p:.4f}   (quy năm {sr_p*np.sqrt(PPY):.2f})")
    print(f"  ngưỡng do phép thử bội     {dsr['sr_expected_max']:.4f}   "
          f"(quy năm {dsr['sr_expected_max']*np.sqrt(PPY):.2f})")
    print(f"  DSR                        {dsr['dsr']:.4f}   "
          f"{'✅ VƯỢT 0.95' if dsr['is_approved'] else '⚠️ DƯỚI 0.95'}")

    # ---- 3. Stress chi phí ----
    print(f"\nSTRESS CHI PHÍ (holdout) — chiến lược sống được tới mức nào:")
    print(f"{'tỷ lệ maker':>13}{'bp/chiều xấp xỉ':>18}{'ann':>9}{'sharpe':>9}{'maxDD':>9}")
    print("-" * 58)
    for mr in (1.0, 0.85, 0.5, 0.378, 0.0):
        s = run(idx >= split, maker_ratio=mr).stats(PPY)
        cm = CostModel(maker_ratio=mr, half_spread_bps=0.0, min_bps=0.5)
        bps = cm.base_bps() + float((per_sym * (1 - mr)).median())
        print(f"{mr:>13.0%}{bps:>18.2f}{s['ann_return']*100:>8.1f}%{s['sharpe']:>9.2f}"
              f"{s['max_dd']*100:>8.1f}%")

    # ---- 4. Theo năm ----
    print(f"\nHOLDOUT THEO NĂM:")
    yr = pd.to_datetime(r_ho.index, unit="ms").year
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() < 10:
            continue
        rr = r_ho[m]
        print(f"  {y}: {(np.prod(1+rr.to_numpy())-1)*100:+8.1f}%   "
              f"({m.sum():>3} kỳ, sharpe {rr.mean()/rr.std(ddof=1)*np.sqrt(PPY):>5.2f})")

    # ---- 5. Đòn bẩy & mục tiêu ----
    r_all = pd.concat([res_tr.returns.dropna(), r_ho])
    mu, sd = float(r_all.mean()), float(r_all.std(ddof=1))
    kelly = kelly_leverage(mu, sd, fraction=1.0, cap=1e9)
    periods_per_week = 7 * 24 / (4 * REBAL)

    print("\n" + "=" * 96)
    print("MỤC TIÊU +50% MỖI TUẦN — TRẢ LỜI BẰNG SỐ")
    print("=" * 96)
    print(f"Chiến lược (toàn mẫu): Sharpe {mu/sd*np.sqrt(PPY):.2f} | "
          f"lợi suất {mu*PPY*100:.1f}%/năm | biến động {sd*np.sqrt(PPY)*100:.1f}%/năm ở gross 1.0x")
    print(f"Một tuần = {periods_per_week:.1f} kỳ tái cân bằng. "
          f"Đòn bẩy Kelly toàn phần = {kelly:.1f}x (nửa Kelly = {kelly/2:.1f}x)")
    print(f"\nMô phỏng block bootstrap 20.000 đường, 1 tuần, 'cháy' = mất 70% vốn:")
    tbl = target_probability_table(
        r_all, target_return=0.50, n_periods=max(1, int(round(periods_per_week))),
        leverages=[1, 2, 3, 5, kelly / 2, kelly, 10, 15, 25, 40],
        n_paths=20000, block=3, ruin_threshold=0.30, periods_per_year=PPY)
    print(f"{'đòn bẩy':>9}{'x Kelly':>9}{'P(đạt +50%)':>13}{'P(cháy)':>10}{'P(lỗ)':>9}"
          f"{'trung vị':>11}{'p5':>9}{'p95':>10}{'vol năm':>10}")
    print("-" * 92)
    for _, r in tbl.iterrows():
        print(f"{r['leverage']:>9.1f}{r['kelly_multiple']:>9.2f}{r['p_target']*100:>12.1f}%"
              f"{r['p_ruin']*100:>9.1f}%{r['p_loss']*100:>8.1f}%{r['median']*100:>10.1f}%"
              f"{r['p05']*100:>8.1f}%{r['p95']*100:>9.1f}%{r['ann_vol_implied']*100:>9.0f}%")

    req = required_sharpe_for_target(0.50, max(1, int(round(periods_per_week))), PPY,
                                     confidence=0.50, max_ann_vol=0.60)
    print(f"\nĐảo ngược câu hỏi: để đạt +50%/tuần với xác suất 50% ở mức biến động 60%/năm,")
    print(f"cần Sharpe năm = {req['required_ann_sharpe']:.0f} "
          f"(tương đương {req['required_ann_return']*100:,.0f}%/năm).")
    print(f"  {req['note']}")

    # ---- 6. Ràng buộc vốn ----
    print(f"\nRÀNG BUỘC VỐN THẬT (${CAPITAL_USD:.0f} = 1.000.000 VND):")
    for lev in (1, 3, 5, 10):
        per_pos = CAPITAL_USD * lev / N_POSITIONS
        ok = "OK" if per_pos >= MIN_NOTIONAL else f"KHÔNG ĐỦ (cần ≥ ${MIN_NOTIONAL})"
        print(f"  đòn bẩy {lev:>2}x -> ${CAPITAL_USD*lev:>6.0f} notional / {N_POSITIONS} vị thế "
              f"= ${per_pos:>5.2f}/lệnh   {ok}")

    # ---- lưu ----
    out = {
        "config": {
            "universe_file": UNIVERSE, "n_symbols": int(close_full.shape[1]),
            "interval": INTERVAL, "rebalance_every": REBAL,
            "period_hours": 4 * REBAL, "n_positions": N_POSITIONS,
            "n_signals": len(SIGNAL_REGISTRY),
            "combiner": {"lookback": COMBINER.lookback, "min_periods": COMBINER.min_periods,
                         "t_threshold": COMBINER.t_threshold,
                         "max_abs_weight": COMBINER.max_abs_weight,
                         "max_step": COMBINER.max_step},
            "portfolio": {"mode": PORTFOLIO.mode, "max_weight": PORTFOLIO.max_weight,
                          "vol_window": PORTFOLIO.vol_window,
                          "beta_neutral": PORTFOLIO.beta_neutral},
            "maker_ratio_assumed": MAKER_RATIO,
            "slippage_bps_median": float(per_sym.median()),
        },
        "train": {k: float(v) for k, v in s_tr.items()},
        "holdout": {k: float(v) for k, v in s_ho.items()},
        "deflated_sharpe": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in dsr.items()},
        "num_trials_declared": NUM_TRIALS,
        "kelly_leverage_full": float(kelly),
        "leverage_table": json.loads(tbl.to_json(orient="records")),
        "required_sharpe_for_50pct_week": req,
        "holdout_uses": 2,
        "note": "Lần 2 — chạy lại sau khi SỬA LỖI ĐO LƯỜNG (tầng gộp bị khởi động lại ở "
                "đầu holdout). Không đổi một tham số chiến lược nào. Không được tinh chỉnh rồi chạy lại.",
    }
    json.dump(out, open("artifacts/strategy_v3.json", "w"), indent=2, ensure_ascii=False)
    print("\n-> artifacts/strategy_v3.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
