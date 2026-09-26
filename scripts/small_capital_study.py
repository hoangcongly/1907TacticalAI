#!/usr/bin/env python3
"""
VỐN 1 TRIỆU VND (~$38): cấu hình nào LỜI NHIỀU NHẤT mà không cháy — đo, không đoán.

VÌ SAO SCRIPT NÀY TỒN TẠI: các nghiên cứu 18-24/09 chọn cấu hình theo Sharpe ở vốn
testnet $5.000 rồi mang sang vốn thật. Ở $38 có BA thứ mà backtest cũ không thấy:

  1. MIN NOTIONAL ép đòn bẩy: mỗi lệnh >= $5 x an toàn 1,2 = $6, nên n vị thế ĐÒI
     đòn bẩy tối thiểu 6n/38. Cấu hình 50 vị thế bị ÉP chạy 7,9x.
  2. [F54] Live BỎ mọi vị thế dưới $5 (`build_rebalance_plan`), không co giãn lại.
     Trọng số `zscore_riskparity` chênh nhau nhiều lần, nên ở $38 phần nhỏ bị bỏ, sổ
     co lại và lệch trung lập. Backtest cũ giao dịch cả những vị thế đó.
  3. NGẮT MẠCH tính trên sụt giảm EQUITY (10% cắt nửa, 30% kill) — ở đòn bẩy cao nó
     bắn liên tục, và kill là hấp thụ: vốn đứng ở ~70% đỉnh.

Script quét (cách đánh trọng số) x (số vị thế n) x (đòn bẩy L >= mức sàn đặt lệnh)
trên đường vốn 4h (`run_v3_fine`, có bỏ vị thế dưới min notional ĐÚNG như live), chi
phí đo thật, rồi bootstrap khối 72h ra phân phối 1 năm — KÈM ngắt mạch như live.

═══════════════════════════════════════════════════════════════════════════════
LUẬT CHỌN — CHỐT TRƯỚC KHI CHẠY
═══════════════════════════════════════════════════════════════════════════════
Chọn cấu hình có vốn cuối năm TRUNG VỊ (có ngắt mạch như live) cao nhất trên cơ sở
THẬN TRỌNG (toàn lịch sử), với ràng buộc:
  (a) P(cháy trong 1 năm, KHÔNG có ngắt mạch) <= 2% ở CẢ HAI cơ sở — ngắt mạch có thể
      không bắn (F41: daemon chết 2 ngày mà không ai biết), nên cấu hình phải sống
      được cả khi nó hỏng;
  (b) |net|/gross trung bình sau khi bỏ vị thế dưới min notional <= 2% — trần F25 của
      live. Vượt trần là mỗi lượt đều NEUTRALITY_BREACH, tức không phải chiến lược
      đã kiểm định.
Cơ sở lạc quan (kỷ nguyên >=100 cặp) chỉ được dùng để LOẠI, không để CHỌN — nó chưa
từng gặp cú đuôi dày nào (CLAUDE.md, mục đòn bẩy 22/09).

Các cặp min notional $20/$50 (ETH, LTC, LINK, ETC, BCH, BTC — F40) bị loại khỏi bể
chọn: ở $38 chúng không mua nổi.

Chạy xong trên dữ liệu thật, script ghi `artifacts/strategy_v3_small.json` cho cấu hình
được chọn. Đổi daemon sang file đó là việc của NGƯỜI VẬN HÀNH (đổi đường tiền).

    python scripts/small_capital_study.py                       # 1 triệu VND, 15,7bp
    python scripts/small_capital_study.py --capital-vnd 2000000
    python scripts/small_capital_study.py --synthetic           # CHỈ kiểm tra đường chạy
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

BASE_CONFIG = "artifacts/strategy_v3.json"     # bản 12 vị thế ĐÃ kiểm định holdout
OUT_CONFIG = "artifacts/strategy_v3_small.json"
OUT_CSV = "artifacts/small_capital_study.csv"
VND_PER_USD = 26_300.0
MIN_ORDER_USD = 5.0 * 1.2
MAKER = 0.39
BARS_YEAR = 2190                               # nến 4h / năm
BLOCK = 18                                     # khối bootstrap = 72h = 1 chu kỳ giữ
HIGH_MIN_NOTIONAL = ("ETHUSDT", "LTCUSDT", "LINKUSDT", "ETCUSDT", "BCHUSDT", "BTCUSDT")
N_GRID = (6, 8, 10, 12, 16, 20, 24, 30, 40, 50)
LEV_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
MAX_P_RUIN = 0.02
MAX_NET = 0.02                                 # = LiveConfig.max_net_exposure (F25)
BASES = ("toàn lịch sử", "kỷ nguyên >=100")
#: (nhãn, mode, trần theo n). `rank_binary` ở live chạy trần 1,0 (xem config_from_json).
FAMILIES = (
    ("V3 zscore trần 0,2", "zscore_riskparity", lambda n: 0.20),
    ("zscore trần 1/n", "zscore_riskparity", lambda n: 1.0 / n),
    ("đều tuyệt đối", "rank_binary", lambda n: 1.0),
)


def weekly(x: float) -> float:
    return (x ** (1 / 52) - 1) if x > 0 else -1.0


def write_config(row: pd.Series, cap_usd: float, capital_vnd: float, n_trials: int) -> None:
    """Cấu hình được chọn, dựng từ JSON gốc. Khối train/holdout thuộc n=12 nên bị xoá nếu khác."""
    base = json.load(open(BASE_CONFIG))
    out = copy.deepcopy(base)
    c = out["config"]
    c["n_positions"] = int(row["n"])
    c["portfolio"]["mode"] = row["mode"]
    c["portfolio"]["max_weight"] = float(row["max_weight"])
    same = (c["n_positions"] == base["config"]["n_positions"]
            and c["portfolio"] == base["config"]["portfolio"])
    if not same:
        for k in ("train", "holdout", "deflated_sharpe", "kelly_leverage_full",
                  "leverage_table", "required_sharpe_for_50pct_week"):
            out.pop(k, None)
        out["holdout_uses"] = "KHÔNG áp dụng — cấu hình này chưa từng được chấm trên holdout."
    out["num_trials_declared"] = int(base.get("num_trials_declared", 250)) + n_trials
    out["recommended_gross_leverage"] = float(row["L"])
    out["capital_vnd"] = capital_vnd
    out["derived_from"] = BASE_CONFIG
    out["small_capital_study"] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                                      else v) for k, v in row.items()}
    out["note"] = (f"CẤU HÌNH VỐN NHỎ — chọn bởi scripts/small_capital_study.py theo luật chốt "
                   f"trước, ở vốn {capital_vnd:,.0f} VND (~${cap_usd:.0f}). Đòn bẩy KHÔNG nằm "
                   f"trong file này: chạy daemon với --leverage {row['L']:.2f}. Bằng chứng là "
                   f"backtest + bootstrap, KHÔNG phải kiểm định ngoài mẫu.")
    with open(OUT_CONFIG, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital-vnd", type=float, default=1_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=15.7,
                    help="chi phí một chiều ĐO THẬT (replay 24/09: 15,7bp trên testnet)")
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

    sig = combined_signal(data, cfg0)              # tầng gộp: tính MỘT lần cho mọi cấu hình
    drop = [c for c in sig.columns if c in HIGH_MIN_NOTIONAL]
    if drop:
        sig = sig.copy()
        sig[drop] = np.nan
    close_marks = data.close.reindex(sig.index)
    hi = data.close.notna().sum(axis=1) >= 100
    hi_idx = set(hi.index[hi.values])

    rows = []
    for fam, mode, cap_fn in FAMILIES:
        for n in N_GRID:
            lmin = min_leverage(n, cap, MIN_ORDER_USD)
            if lmin > max(LEV_GRID):
                continue
            port = replace(cfg0.portfolio, mode=mode, n_positions=n, max_weight=cap_fn(n))
            cfg = replace(cfg0, n_positions=n, portfolio=port)
            w0 = build_weights(sig, close_marks, port)
            cache = {}                             # cùng tập bị bỏ -> cùng đường vốn
            for L in sorted({round(lmin, 2)} | {x for x in LEV_GRID if x >= lmin}):
                wl = drop_below_min_notional(w0, cap, L, cfg.min_notional_usd)
                key = (wl.fillna(0) == 0).to_numpy().tobytes()
                if key not in cache:
                    cache[key] = run_v3_fine(data, cfg, maker_ratio=MAKER, cost_bps=a.cost_bps,
                                             sig=sig, capital_usd=cap, leverage=L).returns.dropna()
                r = cache[key]
                ds = drop_stats(w0, wl)
                worst72 = float(np.expm1(pd.Series(np.log1p(r.to_numpy()))
                                         .rolling(BLOCK).sum().min()))
                for bname, rv in zip(BASES, (r.to_numpy(), r[r.index.isin(hi_idx)].to_numpy())):
                    if len(rv) < BLOCK * 20:
                        continue
                    seed = zlib.crc32(f"{fam}|{n}|{bname}".encode())   # tái lập được
                    paths = block_bootstrap_paths(rv, BARS_YEAR, a.paths, seed, BLOCK)
                    rows.append({"weighting": fam, "mode": mode, "max_weight": cap_fn(n),
                                 "n": n, "L": L, "L_min": lmin, "basis": bname,
                                 "sharpe_4h": float(rv.mean() / rv.std(ddof=1)
                                                    * np.sqrt(BARS_YEAR)),
                                 "worst72_at_L": worst72 * L, **ds,
                                 **evaluate_paths(paths, L, BLOCK)})
            print(f"  xong: {fam}, n={n} (L sàn {lmin:.2f}x, {len(cache)} lần mô phỏng)",
                  flush=True)

    df = pd.DataFrame(rows)
    vals = ["median_live", "median_raw", "p_ruin", "p_kill", "p_loss_live", "p10_live",
            "tier1_share"]
    wide = df.pivot_table(index=["weighting", "mode", "max_weight", "n", "L"],
                          columns="basis", values=vals)
    wide.columns = [f"{v}|{b}" for v, b in wide.columns]
    for v in vals:
        for b in BASES:                    # cơ sở thiếu dữ liệu (vd panel giả < 100 cặp) -> NaN
            if f"{v}|{b}" not in wide.columns:
                wide[f"{v}|{b}"] = np.nan
    wide = wide.reset_index()
    netg = df.groupby(["weighting", "n", "L"])[["net_over_gross", "gross_kept"]].first()
    wide = wide.join(netg, on=["weighting", "n", "L"])
    c, o = BASES
    ok = ((wide[f"p_ruin|{c}"] <= MAX_P_RUIN) & (wide[f"p_ruin|{o}"].fillna(0) <= MAX_P_RUIN)
          & (wide["net_over_gross"] <= MAX_NET))
    best = wide[ok].sort_values(f"median_live|{c}", ascending=False)

    def show(frame, title):
        print("\n" + "=" * 124)
        print(title)
        print("=" * 124)
        print(f"{'trọng số':<20}{'n':>4}{'L':>7}{'x/năm':>7}{'%/tuần':>8}{'VND/tuần':>10}"
              f"{'p10':>6}{'lỗ':>6}{'kill':>6}{'nửa':>6}{'x gốc':>7}{'cháy':>6}{'x lạc q':>8}"
              f"{'gross':>7}{'net':>6}")
        print("-" * 124)
        for _, b in frame.iterrows():
            x = b[f"median_live|{c}"]
            print(f"{b['weighting']:<20}{int(b['n']):>4}{b['L']:>6.2f}x{x:>7.2f}"
                  f"{weekly(x)*100:>7.2f}%{weekly(x)*a.capital_vnd:>10,.0f}"
                  f"{b[f'p10_live|{c}']:>6.2f}{b[f'p_loss_live|{c}']*100:>5.0f}%"
                  f"{b[f'p_kill|{c}']*100:>5.0f}%{b[f'tier1_share|{c}']*100:>5.0f}%"
                  f"{b[f'median_raw|{c}']:>7.2f}{b[f'p_ruin|{c}']*100:>5.1f}%"
                  f"{b[f'median_live|{o}']:>8.2f}{b['gross_kept']*100:>6.0f}%"
                  f"{b['net_over_gross']*100:>5.1f}%")

    show(best.head(10), f"VỐN {a.capital_vnd:,.0f} VND (~${cap:.0f}), chi phí {a.cost_bps}bp/chiều"
         f" — TOP 10 theo vốn cuối năm TRUNG VỊ có ngắt mạch như live (cơ sở THẬN TRỌNG)")
    print("  x/năm, %/tuần, VND/tuần, p10, lỗ, kill, nửa: CÓ ngắt mạch như live (kill = hấp thụ)")
    print("  x gốc, cháy: KHÔNG ngắt mạch | x lạc q: cơ sở kỷ nguyên >=100 cặp")
    print("  gross/net: phần sổ còn lại và độ lệch trung lập SAU khi bỏ vị thế dưới $5 (F54)")

    ref = wide[((wide["n"] == 50) & (wide["weighting"] == "zscore trần 1/n"))
               | ((wide["n"] == 30) & (wide["weighting"] == "zscore trần 1/n") & (wide["L"] == 5.0))
               | ((wide["n"] == 12) & (wide["weighting"] == "V3 zscore trần 0,2"))]
    show(ref.sort_values(["n", "L"]), "THAM CHIẾU — cấu hình wide (n=50, ép >= 7,89x), daemon "
         "--leverage 5 ở vốn này (F43 cắt còn ~30), và V3 n=12 đã kiểm định")

    if best.empty:
        print("\n=> KHÔNG cấu hình nào đạt luật chọn — vốn này không đủ cho chiến lược.")
    else:
        b = best.iloc[0]
        print(f"\n=> CHỌN: {b['weighting']} ({b['mode']}), n={int(b['n'])}, đòn bẩy {b['L']:.2f}x "
              f"(sàn đặt lệnh {b['n'] * MIN_ORDER_USD / cap:.2f}x)")
        if not a.synthetic:
            write_config(b, cap, a.capital_vnd, n_trials=len(wide))
            print(f"   đã ghi {OUT_CONFIG}. Đổi daemon sang file này + --leverage {b['L']:.2f} là "
                  f"việc của người vận hành.")
    print("\n⚠️ Kill ở live KHÔNG tự đóng vị thế — chỉ ngừng giao dịch. Mô phỏng coi vốn đứng yên")
    print("   từ lúc kill, tức giả định bạn /kill tay ngay khi có báo động.")

    if not a.synthetic:
        df.to_csv(OUT_CSV, index=False)
        print(f"đã ghi {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
