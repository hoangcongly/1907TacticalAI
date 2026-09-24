#!/usr/bin/env python3
"""
SỨC CHỨA VỐN — chiến lược này nuôi được bao nhiêu tiền trước khi tự giết mình.

VÌ SAO CẦN: mọi con số trong repo này đo ở vốn $38 (1 triệu VND). Ở quy mô đó,
`estimate_cost_bps` nói thẳng rằng tác động giá chỉ 0.01-0.1bp — tức KHÔNG TỒN TẠI.
Nghĩa là toàn bộ Sharpe đã công bố chưa từng bị chi phí quy mô chạm vào.

Điều đó không sai, nhưng nó khiến một câu hỏi bị bỏ trống: nếu bơm $1M vào thì
Sharpe còn lại bao nhiêu? Một chiến lược lãi ở $5k mà chết ở $500k không phải một
doanh nghiệp, và không có bảng nào trong repo trả lời được điều đó.

CÁCH ĐO: giữ nguyên tín hiệu và trọng số (chúng không phụ thuộc vốn), chỉ quét vốn
qua đúng mô hình chi phí của repo — `estimate_cost_bps` với luật căn bậc hai
`sigma_ngày * sqrt(N/ADV)`. Mỗi mức vốn cho một chuỗi lợi suất ròng khác nhau, và
từ đó một Sharpe khác nhau. Đường cong đó chính là sức chứa.

HAI CẢNH BÁO PHẢI ĐỌC KÈM:
  - `estimate_cost_bps` KẸP tác động giá ở 20bp. Vượt trần đó, mô hình không còn
    mô tả thực tế nữa mà chỉ nói "rất đắt". Bảng dưới đánh dấu tỷ lệ cặp chạm trần;
    khi tỷ lệ đó lớn, con số Sharpe là LẠC QUAN chứ không bi quan.
  - Mô hình chỉ tính tác động TỨC THỜI của một lệnh. Nó không tính việc nắm một vị
    thế lớn nhiều ngày trong sổ lệnh mỏng, cũng không tính chuyện người khác nhìn
    thấy mình. Sức chứa thật luôn THẤP HƠN con số ở đây.

    python scripts/capacity_study.py
    python scripts/capacity_study.py --leverage 5.0 --n-positions 50
"""
import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from aegis.research.backtest_v2 import estimate_cost_bps
from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)

OUT = "artifacts/capacity_study.csv"

AUM_LEVELS = [5_000, 25_000, 100_000, 250_000, 500_000,
              1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000]


def _sharpe(r, ppy: float) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leverage", type=float, default=5.0,
                    help="Đòn bẩy gộp daemon đang chạy.")
    ap.add_argument("--n-positions", type=int, default=50)
    ap.add_argument("--impact-coef", type=float, default=1.0)
    a = ap.parse_args(argv)

    print(f"nạp dữ liệu (vintage ghim {V3_VINTAGE_MS})...", flush=True)
    data = load_v3_data(V3, verbose=True, end_ms=V3_VINTAGE_MS)
    ppy = V3.periods_per_year

    # Tín hiệu KHÔNG phụ thuộc vốn — dựng một lần, dùng lại cho mọi mức.
    sig = combined_signal(data, V3)

    # ADV và biến động để báo cáo tỷ lệ tham gia thị trường.
    close = data.panel["close"]
    dollar_vol = (data.panel["volume"] * close).iloc[-720:]
    adv = (dollar_vol.median() * 6).replace(0.0, np.nan)

    idx = data.index
    rows = []
    print(f"\nquét {len(AUM_LEVELS)} mức vốn ở đòn bẩy {a.leverage}x, "
          f"{a.n_positions} vị thế...\n", flush=True)

    for aum in AUM_LEVELS:
        notional = aum * a.leverage / a.n_positions      # notional mỗi lệnh
        bps = estimate_cost_bps(data.panel, notional_usd=notional,
                                bars_per_day=6, impact_coef=a.impact_coef)
        data.per_symbol_bps = bps

        res_all = run_v3(data, V3, sig=sig, n_positions=a.n_positions)
        res_ho = run_v3(data, V3, mask=idx >= data.split_ts, sig=sig,
                        n_positions=a.n_positions)

        # Chẩn đoán quy mô: bao nhiêu cặp đã chạm trần kẹp 20bp, và tham gia bao nhiêu % ADV.
        at_cap = float((bps >= 19.99).mean())
        part = (notional / adv).replace([np.inf, -np.inf], np.nan)
        rows.append({
            "aum_usd": aum,
            "notional_per_order": notional,
            "slippage_bq_bps": float(bps.median()),
            "pct_cap_20bps": at_cap * 100,
            "participation_median_pct": float(part.median() * 100),
            "participation_p90_pct": float(part.quantile(0.90) * 100),
            "sharpe_toan_mau": _sharpe(res_all.returns, ppy),
            "sharpe_holdout": _sharpe(res_ho.returns, ppy),
            "ann_return_pct": float(np.nanmean(res_all.returns) * ppy * 100),
        })
        r = rows[-1]
        print(f"  ${aum:>12,.0f} | ${notional:>9,.0f}/lệnh | "
              f"trượt {r['slippage_bq_bps']:>6.2f}bp | chạm trần {at_cap*100:>5.1f}% | "
              f"Sharpe {r['sharpe_toan_mau']:>5.2f} (OOS {r['sharpe_holdout']:>5.2f})",
              flush=True)

    df = pd.DataFrame(rows)
    pathlib.Path(OUT).parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)

    base = df.iloc[0]["sharpe_toan_mau"]
    df["pct_sharpe_con_lai"] = df["sharpe_toan_mau"] / base * 100

    print("\n" + "=" * 86)
    print("ĐƯỜNG CONG SỨC CHỨA")
    print("=" * 86)
    print(f"{'vốn':>14}{'trượt giá':>12}{'chạm trần':>11}{'%ADV(p90)':>11}"
          f"{'Sharpe':>9}{'OOS':>8}{'% còn lại':>11}")
    print("-" * 86)
    for _, r in df.iterrows():
        print(f"${r['aum_usd']:>13,.0f}{r['slippage_bq_bps']:>11.2f}bp"
              f"{r['pct_cap_20bps']:>10.1f}%{r['participation_p90_pct']:>10.2f}%"
              f"{r['sharpe_toan_mau']:>9.2f}{r['sharpe_holdout']:>8.2f}"
              f"{r['pct_sharpe_con_lai']:>10.0f}%")

    half = df[df["pct_sharpe_con_lai"] < 50]
    cap50 = f"${half.iloc[0]['aum_usd']:,.0f}" if len(half) else "> mức cao nhất đã quét"
    dead = df[df["sharpe_toan_mau"] < 1.0]
    cap1 = f"${dead.iloc[0]['aum_usd']:,.0f}" if len(dead) else "> mức cao nhất đã quét"
    print("-" * 86)
    print(f"  Vốn làm MẤT NỬA Sharpe      : {cap50}")
    print(f"  Vốn đẩy Sharpe xuống dưới 1 : {cap1}")
    print(f"\n  -> {OUT}")
    print("\n  NHỚ: mô hình kẹp tác động giá ở 20bp và bỏ qua chi phí nắm giữ dài")
    print("  trong sổ mỏng. Sức chứa THẬT thấp hơn bảng này.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
