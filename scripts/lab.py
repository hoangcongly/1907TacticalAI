#!/usr/bin/env python3
"""
PHÒNG THÍ NGHIỆM NÂNG CẤP — đo từng thay đổi một, CHỈ TRÊN TẬP TRAIN.

Nguyên tắc: mỗi dòng trong bảng kết quả chỉ khác dòng trên nó ĐÚNG MỘT THAY ĐỔI.
Nhờ vậy cột "delta Sharpe" đọc được thành quy kết nguyên nhân. Chạy 10 thay đổi
cùng lúc rồi thấy Sharpe tăng thì không biết cái nào có tác dụng, cái nào có hại.

Holdout KHÔNG được chạm ở đây. Kiểm định cuối dùng `scripts/xs_validate_v2.py`.

    python scripts/lab.py --stage portfolio    # nâng cấp dựng danh mục
    python scripts/lab.py --stage signals      # mở rộng thư viện tín hiệu
    python scripts/lab.py --stage frequency    # chọn khung + chu kỳ tái cân bằng
    python scripts/lab.py --stage all
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.data.panel_v2 import load_panel_v2, load_funding_panel_v2, align_panel, interval_hours
from aegis.research.backtest_v2 import CostModel, estimate_cost_bps, simulate
from aegis.research.cross_sectional import combine_signals_zscore
from aegis.research.ic_analysis import (
    effective_breadth, ic_decay_curve, implied_sharpe, information_coefficient,
    ic_correlation_matrix, signal_autocorrelation,
)
from aegis.research.signal_library import (
    FAMILIES, SIGNAL_REGISTRY, build_all_families, build_family, build_signal,
)
from aegis.risk.portfolio import PortfolioSpec, build_weights, realized_vol, rolling_beta

ARTIFACTS = pathlib.Path("artifacts")
HOURS_PER_YEAR = 24 * 365.0


# ---------------------------------------------------------------------------
def load_universe(path: str) -> list:
    return json.load(open(path))


def load_data(universe_file: str, interval: str, part: str = "train", source: str = "1h"):
    """Nạp panel + funding, cắt theo mốc chia train/holdout đã niêm phong."""
    syms = load_universe(universe_file)
    panel = load_panel_v2(syms, interval=interval, source_interval=source, min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)

    split = json.load(open(ARTIFACTS / "holdout_split.json"))["split_ts"]
    idx = panel["close"].index
    mask = idx < split if part == "train" else idx >= split
    if part == "all":
        mask = np.ones(len(idx), dtype=bool)

    # Tín hiệu cần lịch sử warm-up: tính trên TOÀN chuỗi rồi mới cắt (tín hiệu nhân
    # quả nên cắt sau là hợp lệ và tránh mất 700 nến đầu holdout).
    return panel, funding, mask


def cut(df, mask):
    return df.loc[mask] if isinstance(df, (pd.DataFrame, pd.Series)) else df


def stats_of(res, periods_per_year):
    return res.stats(periods_per_year)


def fmt_row(label, s, base_sharpe=None):
    delta = ""
    if base_sharpe is not None and base_sharpe:
        delta = f"{(s['sharpe'] - base_sharpe):+7.2f}"
    return (f"{label:<34}{s['ann_return']*100:>8.1f}%{s['ann_vol']*100:>7.1f}%"
            f"{s['sharpe']:>8.2f}{s['t_stat']:>8.2f}{s['max_dd']*100:>7.1f}%"
            f"{s['avg_turnover']:>8.2f}{s['cost_drag_ann']*100:>8.1f}%{delta:>8}")


HEADER = (f"{'cấu hình':<34}{'ann':>9}{'vol':>7}{'sharpe':>8}{'t-stat':>8}"
          f"{'maxDD':>8}{'turn':>8}{'phí/năm':>9}{'ΔSharpe':>8}")


# ---------------------------------------------------------------------------
def stage_portfolio(args):
    """Đo riêng phần DỰNG DANH MỤC: cùng tín hiệu, chỉ đổi cách biến thành vị thế."""
    panel, funding, mask = load_data(args.universe, args.interval, "train", args.source)
    close_full = panel["close"]

    # Tín hiệu của hệ thống HIỆN TẠI, dựng lại bằng thư viện mới để so sánh công bằng.
    from aegis.research.cross_sectional import (
        signal_funding_carry, signal_momentum, signal_ofi, signal_funding_momentum,
    )
    bars_per_4h = int(round(4 / interval_hours(args.interval)))
    sigs = {
        "funding_carry": signal_funding_carry(funding, 6 * bars_per_4h),
        "momentum_90": signal_momentum(close_full, 90 * bars_per_4h),
        "ofi_flow": signal_ofi(panel["ofi"], 6 * bars_per_4h),
        "funding_mom": signal_funding_momentum(funding, 42 * bars_per_4h),
    }
    comb_full = combine_signals_zscore(sigs, min_coverage=3)

    rebal = args.rebalance
    marks = close_full.index[::rebal]
    marks = marks[mask.repeat(1)[::rebal]] if False else marks[np.isin(marks, close_full.index[mask])]

    comb = comb_full.reindex(marks)
    close = close_full.reindex(marks)
    fund = funding.reindex(marks)
    period_hours = interval_hours(args.interval) * rebal
    ppy = HOURS_PER_YEAR / period_hours

    # Chi phí: mức "giả định lạc quan" cũ so với mức ĐO ĐƯỢC theo từng cặp.
    flat_optimistic = CostModel(taker_fee_bps=1.0, maker_fee_bps=1.0, maker_ratio=1.0,
                                half_spread_bps=0.0, min_bps=1.0)
    per_symbol = estimate_cost_bps(panel, notional_usd=args.notional)
    realistic = CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=args.maker_ratio,
                          half_spread_bps=0.0, per_symbol_bps=per_symbol * (1 - args.maker_ratio),
                          min_bps=1.0)

    print(f"\n{'='*104}")
    print(f"GIAI ĐOẠN 1 — DỰNG DANH MỤC  (train, {args.interval}, tái cân bằng {rebal} nến "
          f"= {period_hours:.0f}h, {close.shape[1]} cặp)")
    print(f"chi phí thực tế ước lượng: trung vị {per_symbol.median():.1f}bp trượt giá + phí sàn "
          f"→ tổng {realistic.base_bps() + (per_symbol*(1-args.maker_ratio)).median():.1f}bp/chiều")
    print("=" * 104)
    print(HEADER)
    print("-" * 104)

    def run(spec, cost, label, base=None):
        W = build_weights(comb, close, spec)
        res = simulate(W, close, fund, cost, bar_hours=period_hours, rebalance_every=1)
        s = stats_of(res, ppy)
        print(fmt_row(label, s, base))
        return s

    base_spec = PortfolioSpec(mode="rank_binary", top_frac=args.top_frac, max_weight=1.0)
    s0 = run(base_spec, flat_optimistic, "0. GỐC (nhị phân, phí 1bp giả định)")
    b = s0["sharpe"]

    s1 = run(base_spec, realistic, "1. + chi phí thật theo từng cặp", b)
    b1 = s1["sharpe"]

    results = {"baseline_optimistic": s0, "baseline_realistic": s1}

    for label, spec in [
        ("2. + chia đều RỦI RO (inv-vol)", PortfolioSpec(mode="rank_riskparity", top_frac=args.top_frac)),
        ("3. + trọng số liên tục z-score", PortfolioSpec(mode="zscore", top_frac=args.top_frac)),
        ("4. + liên tục & inv-vol", PortfolioSpec(mode="zscore_riskparity", top_frac=args.top_frac)),
        ("5. + khử beta thị trường", PortfolioSpec(mode="zscore_riskparity", top_frac=args.top_frac,
                                                   beta_neutral=True)),
        ("6. + vùng đệm thứ hạng", PortfolioSpec(mode="rank_riskparity", top_frac=args.top_frac,
                                                  exit_frac=args.top_frac * 1.5)),
        ("7. + MVO hiệp phương sai co", PortfolioSpec(mode="mvo", top_frac=args.top_frac,
                                                       beta_neutral=True)),
    ]:
        results[label] = run(spec, realistic, label, b1)

    print("-" * 104)
    print("Đọc bảng: cột ΔSharpe so với dòng 1 (cùng giả định chi phí). Dòng 0 vs 1 cho thấy "
          "\nmức phí giả định cũ đã thổi phồng bao nhiêu.")
    return results


# ---------------------------------------------------------------------------
def stage_signals(args):
    """Đo phần MỞ RỘNG TÍN HIỆU: cùng cách dựng danh mục, chỉ đổi tín hiệu."""
    panel, funding, mask = load_data(args.universe, args.interval, "train", args.source)
    close_full = panel["close"]

    rebal = args.rebalance
    marks = close_full.index[::rebal]
    marks = marks[np.isin(marks, close_full.index[mask])]
    close = close_full.reindex(marks)
    fund = funding.reindex(marks)
    period_hours = interval_hours(args.interval) * rebal
    ppy = HOURS_PER_YEAR / period_hours

    per_symbol = estimate_cost_bps(panel, notional_usd=args.notional)
    cost = CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=args.maker_ratio,
                     half_spread_bps=0.0, per_symbol_bps=per_symbol * (1 - args.maker_ratio),
                     min_bps=1.0)
    spec = PortfolioSpec(mode="zscore_riskparity", top_frac=args.top_frac, beta_neutral=True)

    print(f"\n{'='*104}")
    print(f"GIAI ĐOẠN 2 — THƯ VIỆN TÍN HIỆU  ({len(SIGNAL_REGISTRY)} tín hiệu / {len(FAMILIES)} họ)")
    print("=" * 104)

    # --- IC từng tín hiệu, ở chân trời bằng đúng chu kỳ giữ vị thế ---
    print(f"\n{'tín hiệu':<20}{'họ':<16}{'IC':>8}{'IC IR':>8}{'t-stat':>8}{'tự tương quan':>15}")
    print("-" * 78)
    ic_rows = []
    for name, sd in SIGNAL_REGISTRY.items():
        sig_full = build_signal(name, panel, funding)
        sig = sig_full.reindex(marks)
        if sig.notna().sum().sum() < 500:
            continue
        res = information_coefficient(sig, close, horizon=1)
        ac = float(signal_autocorrelation(sig, [1]).iloc[0])
        s = res.summary()
        s.update({"name": name, "family": sd.family, "autocorr": ac})
        ic_rows.append(s)
        print(f"{name:<20}{sd.family:<16}{s['ic_mean']:>8.4f}{s['ic_ir']:>8.3f}"
              f"{s['t_stat']:>8.2f}{ac:>15.3f}")

    ic_df = pd.DataFrame(ic_rows)

    # --- Tín hiệu cấp họ ---
    fam_sigs_full = build_all_families(panel, funding)
    fam_sigs = {k: v.reindex(marks) for k, v in fam_sigs_full.items()}

    print(f"\n{'HỌ (gộp các biến thể)':<34}{'IC':>8}{'IC IR':>8}{'t-stat':>8}")
    print("-" * 58)
    for f, sig in fam_sigs.items():
        s = information_coefficient(sig, close, horizon=1).summary()
        print(f"{f:<34}{s['ic_mean']:>8.4f}{s['ic_ir']:>8.3f}{s['t_stat']:>8.2f}")

    print("\nTƯƠNG QUAN GIỮA CÁC CHUỖI IC (bổ trợ thật, không phải tương quan lợi suất):")
    print(ic_correlation_matrix(fam_sigs, close, horizon=1).round(2).to_string())

    # --- Backtest ---
    print(f"\n{HEADER}")
    print("-" * 104)

    from aegis.research.cross_sectional import (
        signal_funding_carry, signal_momentum, signal_ofi, signal_funding_momentum,
    )
    b4h = int(round(4 / interval_hours(args.interval)))
    old = combine_signals_zscore({
        "funding_carry": signal_funding_carry(funding, 6 * b4h),
        "momentum_90": signal_momentum(close_full, 90 * b4h),
        "ofi_flow": signal_ofi(panel["ofi"], 6 * b4h),
        "funding_mom": signal_funding_momentum(funding, 42 * b4h),
    }, min_coverage=3).reindex(marks)

    def run(sig, label, base=None):
        W = build_weights(sig, close, spec)
        res = simulate(W, close, fund, cost, bar_hours=period_hours, rebalance_every=1)
        s = stats_of(res, ppy)
        print(fmt_row(label, s, base))
        return s

    s_old = run(old, "A. 4 tín hiệu cũ")
    b = s_old["sharpe"]

    per_family = {}
    for f, sig in fam_sigs.items():
        per_family[f] = run(sig, f"B. chỉ họ {f}", b)

    all_fam = combine_signals_zscore(fam_sigs, min_coverage=2).reindex(marks)
    s_all = run(all_fam, "C. GỘP 5 HỌ (đều trọng số)", b)

    # Gộp có trọng số theo IC: họ dự báo tốt hơn được tiếng nói lớn hơn.
    ic_w = {f: max(information_coefficient(sig, close, horizon=1).mean, 0.0)
            for f, sig in fam_sigs.items()}
    tot = sum(ic_w.values()) or 1.0
    weighted = sum(sig * (ic_w[f] / tot) for f, sig in fam_sigs.items())
    s_icw = run(weighted, "D. GỘP 5 HỌ (trọng số theo IC)", b)

    return {"ic_table": ic_df, "old": s_old, "families": per_family,
            "all_equal": s_all, "all_ic_weighted": s_icw, "ic_weights": ic_w}


# ---------------------------------------------------------------------------
def stage_frequency(args):
    """
    Chọn KHUNG THỜI GIAN và CHU KỲ TÁI CÂN BẰNG bằng đường cong suy giảm IC.

    Đây là đòn bẩy độ rộng: cùng một IC, quyết định 4 lần/ngày cho độ rộng gấp 4
    lần quyết định 1 lần/ngày — theo định luật cơ bản là gấp 2 lần IR. Điều kiện là
    IC không suy giảm nhanh hơn và chi phí không nuốt hết phần tăng thêm.
    """
    print(f"\n{'='*104}")
    print("GIAI ĐOẠN 3 — TẦN SUẤT & ĐỘ RỘNG")
    print("=" * 104)

    per_symbol_cache = {}
    rows = []
    for interval in args.intervals:
        try:
            panel, funding, mask = load_data(args.universe, interval, "train", args.source)
        except Exception as exc:
            print(f"  {interval}: bỏ qua ({type(exc).__name__}: {exc})")
            continue

        close_full = panel["close"]
        fam = build_all_families(panel, funding)
        comb_full = combine_signals_zscore(fam, min_coverage=2)
        per_symbol = per_symbol_cache.setdefault(
            interval, estimate_cost_bps(panel, notional_usd=args.notional))
        h = interval_hours(interval)

        print(f"\n--- {interval} ({close_full.shape[1]} cặp, {int(mask.sum())} nến train) ---")
        decay = ic_decay_curve(comb_full.loc[mask], close_full.loc[mask],
                               horizons=[1, 2, 3, 6, 12, 24, 48])
        print(decay[["ic_mean", "ic_ir", "t_stat", "ic_per_sqrt_period"]].round(4).to_string())

        for rebal in args.rebalances:
            marks = close_full.index[::rebal]
            marks = marks[np.isin(marks, close_full.index[mask])]
            if len(marks) < 200:
                continue
            close = close_full.reindex(marks)
            fund = funding.reindex(marks)
            comb = comb_full.reindex(marks)
            period_hours = h * rebal
            ppy = HOURS_PER_YEAR / period_hours

            cost = CostModel(taker_fee_bps=5.0, maker_fee_bps=2.0, maker_ratio=args.maker_ratio,
                             half_spread_bps=0.0,
                             per_symbol_bps=per_symbol * (1 - args.maker_ratio), min_bps=1.0)
            spec = PortfolioSpec(mode="zscore_riskparity", top_frac=args.top_frac,
                                 beta_neutral=True)
            W = build_weights(comb, close, spec)
            res = simulate(W, close, fund, cost, bar_hours=period_hours, rebalance_every=1)
            s = stats_of(res, ppy)

            br = effective_breadth(comb, 1, ppy, n_positions=int(s["avg_positions"]))
            s.update({"interval": interval, "rebalance": rebal, "period_hours": period_hours,
                      "eff_breadth": br["effective_breadth"],
                      "implied_sharpe": implied_sharpe(
                          information_coefficient(comb, close, 1).mean,
                          br["effective_breadth"])})
            rows.append(s)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    print(f"\n{'khung':<8}{'rebal':>7}{'giờ/kỳ':>8}{'ann':>9}{'vol':>7}{'sharpe':>8}"
          f"{'t':>7}{'maxDD':>8}{'turn':>7}{'phí/năm':>9}{'độ rộng':>10}{'Sharpe lý thuyết':>18}")
    print("-" * 108)
    for _, r in df.sort_values("sharpe", ascending=False).iterrows():
        print(f"{r['interval']:<8}{int(r['rebalance']):>7}{r['period_hours']:>8.0f}"
              f"{r['ann_return']*100:>8.1f}%{r['ann_vol']*100:>6.1f}%{r['sharpe']:>8.2f}"
              f"{r['t_stat']:>7.2f}{r['max_dd']*100:>7.1f}%{r['avg_turnover']:>7.2f}"
              f"{r['cost_drag_ann']*100:>8.1f}%{r['eff_breadth']:>10.0f}{r['implied_sharpe']:>18.2f}")
    return {"frequency_table": df}


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["portfolio", "signals", "frequency", "all"])
    ap.add_argument("--universe", default="artifacts/universe_filtered.json")
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--source", default="1h", help="Khung nguồn để tổng hợp (1h hoặc 4h)")
    ap.add_argument("--rebalance", type=int, default=6)
    ap.add_argument("--top-frac", type=float, default=0.10, dest="top_frac")
    ap.add_argument("--maker-ratio", type=float, default=0.5, dest="maker_ratio")
    ap.add_argument("--notional", type=float, default=20.0,
                    help="Notional mỗi lệnh (USD) — dùng để ước lượng tác động giá")
    ap.add_argument("--intervals", nargs="*", default=["4h", "1h"])
    ap.add_argument("--rebalances", nargs="*", type=int, default=[1, 2, 3, 6, 12])
    ap.add_argument("--out", default="artifacts/lab_results.json")
    a = ap.parse_args(argv)

    out = {}
    if a.stage in ("portfolio", "all"):
        out["portfolio"] = stage_portfolio(a)
    if a.stage in ("signals", "all"):
        out["signals"] = stage_signals(a)
    if a.stage in ("frequency", "all"):
        out["frequency"] = stage_frequency(a)

    def encode(o):
        if isinstance(o, pd.DataFrame):
            return json.loads(o.to_json(orient="records"))
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return str(o)

    json.dump(out, open(a.out, "w"), indent=2, default=encode)
    print(f"\nkết quả -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
