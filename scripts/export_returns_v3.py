#!/usr/bin/env python3
"""
Xuất chuỗi lợi suất v3 ra artifact — và CHỨNG MINH bản tái tạo là trung thực.

VÌ SAO CẦN: `scripts/validate_v3.py` tính ra chuỗi lợi suất train/holdout rồi VỨT
ĐI, chỉ ghi lại thống kê tổng hợp. Hệ quả là mọi phân tích downstream (đòn bẩy, xác
suất đạt mục tiêu, quy hoạch động, stress) phải dựng lại toàn bộ panel 170 cặp —
khoảng 1-2 phút mỗi lần — và mỗi lần dựng lại là một cơ hội để cấu hình bị chép sai.

Script này chạy `research/strategy_v3.run_v3` (đường chạy dùng chung) rồi:
  1. So từng thống kê với `artifacts/strategy_v3.json` — lệch > 1e-9 là DỪNG.
  2. Ghi `artifacts/returns_v3.csv` gồm cả lợi suất gộp, chi phí, funding, turnover.

Bước 1 không phải nghi thức: nó là bằng chứng rằng module mới và script gốc là CÙNG
MỘT chiến lược. Không có nó, việc tái cấu trúc chỉ là niềm tin.

    python scripts/export_returns_v3.py
    python scripts/export_returns_v3.py --no-verify   # khi strategy_v3.json chưa có
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, load_v3_data, run_v3_fine, split_train_holdout,
)

OUT_CSV = "artifacts/returns_v3.csv"
REF_JSON = "artifacts/strategy_v3.json"

#: Thống kê là hàm THUẦN của chuỗi trọng số — chi phí không vào được.
#: `avg_turnover` và `avg_positions` đọc thẳng từ trọng số; `funding_ann` là trọng số
#: nhân rate; `n` là số kỳ. Sai lệch ở ĐÂY nghĩa là trọng số, lưới tái cân bằng hoặc
#: thành phần rổ đã đổi — tức một thay đổi THẬT về chiến lược. Khoá chặt.
#:
#: Mọi thống kê khác (kể cả `ann_vol`, `hit_rate`, `skew`, `kurtosis`) là mô-men của
#: chuỗi lợi suất RÒNG, mà lợi suất ròng = gộp - chi phí + funding. Chúng nhạy với chi
#: phí dù rất yếu, nên thuộc nhóm tương đối. Xếp nhầm chúng vào đây là tự tạo báo động
#: giả — đã thử và đúng là báo động giả (lệch 1e-4 trên skew = 5e-5 tương đối).
STRUCTURAL = ("n", "avg_turnover", "avg_positions", "funding_ann")
TOL_STRUCTURAL = 1e-4

#: Thống kê CÓ phụ thuộc chi phí. `estimate_cost_bps` lấy 720 nến CUỐI panel để ước
#: lượng biến động và ADV, nên nó nhạy với việc panel dài thêm vài nến — và ta không
#: biết chính xác panel lúc `validate_v3.py` chạy dài tới nến nào (chỉ biết tới giờ).
#: Phần dư đo được là ~3e-4 trên cost_drag_ann, tức ~0,03%/năm. Ngưỡng 1% tương đối
#: rộng hơn phần dư đó hơn một bậc, nhưng vẫn hẹp hơn MỌI lỗi thật cả hai bậc.
TOL_RELATIVE = 0.01


def _frame(res, label: str) -> pd.DataFrame:
    """Gom mọi chuỗi theo kỳ của một lần chạy thành một bảng."""
    df = pd.DataFrame({
        "net_return": res.returns,
        "gross_return": res.gross_returns,
        "cost_drag": res.cost_drag,
        "funding_pnl": res.funding_pnl,
        "turnover": res.turnover,
        "n_positions": res.n_positions,
        "net_exposure": res.net_exposure,
        "gross_exposure": res.gross_exposure,
    })
    df.index.name = "timestamp_ms"
    df.insert(0, "segment", label)
    df.insert(1, "datetime", pd.to_datetime(df.index, unit="ms"))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--ref", default=REF_JSON)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--end", default="vintage",
                    help="'vintage' (mốc tái lập được strategy_v3.json), 'latest' "
                         "(toàn bộ dữ liệu, dùng cho quyết định live), hoặc ISO date")
    a = ap.parse_args(argv)

    if a.end == "vintage":
        end_ms = V3_VINTAGE_MS
    elif a.end == "latest":
        end_ms = None
    else:
        end_ms = int(pd.Timestamp(a.end).value // 10**6)

    data = load_v3_data(V3, end_ms=end_ms)
    print(f"mốc dữ liệu: {pd.to_datetime(data.data_end_ms, unit='ms')} "
          f"| {data.close.shape[1]} cặp")
    res_tr, res_ho = split_train_holdout(data, V3)
    ppy = V3.periods_per_year
    s_tr, s_ho = res_tr.stats(ppy), res_ho.stats(ppy)

    print("\n" + "=" * 78)
    print(f"{'':<26}{'kỳ':>6}{'ann':>9}{'vol':>8}{'sharpe':>8}{'maxDD':>9}{'skew':>7}")
    print("-" * 78)
    for lbl, s in (("TRAIN", s_tr), ("HOLDOUT", s_ho)):
        print(f"{lbl:<26}{s['n']:>6.0f}{s['ann_return']*100:>8.1f}%{s['ann_vol']*100:>7.1f}%"
              f"{s['sharpe']:>8.2f}{s['max_dd']*100:>8.1f}%{s['skew']:>7.2f}")
    print("=" * 78)

    # ---- 1. Kiểm chứng parity với con số đã ghi ----------------------------
    ref_path = pathlib.Path(a.ref)
    if not a.no_verify and ref_path.is_file():
        ref = json.loads(ref_path.read_text())
        bad, worst_s, worst_r, n_checked = [], 0.0, 0.0, 0
        for seg, got in (("train", s_tr), ("holdout", s_ho)):
            for k, want in ref.get(seg, {}).items():
                if k not in got:
                    continue
                n_checked += 1
                g, w = float(got[k]), float(want)
                if k in STRUCTURAL:
                    d = abs(g - w)
                    worst_s = max(worst_s, d)
                    if d > TOL_STRUCTURAL:
                        bad.append(f"  [cấu trúc] {seg}.{k}: {g!r} != {w!r} (lệch tuyệt đối {d:.3e})")
                else:
                    rel = abs(g - w) / max(abs(w), 1e-12)
                    worst_r = max(worst_r, rel)
                    if rel > TOL_RELATIVE:
                        bad.append(f"  [chi phí]  {seg}.{k}: {g!r} != {w!r} (lệch tương đối {rel:.3%})")
        if bad:
            print("\n❌ KHÔNG KHỚP với artifacts/strategy_v3.json — bản tái tạo KHÔNG trung thực:")
            print("\n".join(bad))
            print("\nĐừng dùng chuỗi lợi suất này cho bất cứ quyết định nào cho tới khi khớp.")
            return 1
        print(f"✅ PARITY {n_checked} thống kê khớp strategy_v3.json")
        print(f"   nhóm cấu trúc (không phụ thuộc chi phí): lệch tối đa {worst_s:.2e} "
              f"(ngưỡng {TOL_STRUCTURAL:g})")
        print(f"   nhóm phụ thuộc chi phí                 : lệch tối đa {worst_r:.3%} "
              f"(ngưỡng {TOL_RELATIVE:.0%})")
    elif not a.no_verify:
        print(f"⚠️  không thấy {a.ref} — bỏ qua kiểm chứng")

    # ---- 2. Ghi chuỗi ------------------------------------------------------
    df = pd.concat([_frame(res_tr, "train"), _frame(res_ho, "holdout")]).sort_index()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)

    meta = {
        "config": V3.to_dict(),
        "split_ts": data.split_ts,
        "data_end_ms": int(data.data_end_ms),
        "data_end": str(pd.to_datetime(data.data_end_ms, unit="ms")),
        "end_mode": a.end,
        "n_symbols": int(data.close.shape[1]),
        "periods_per_year": ppy,
        "train": {k: float(v) for k, v in s_tr.items()},
        "holdout": {k: float(v) for k, v in s_ho.items()},
        "verified_against": str(ref_path) if not a.no_verify and ref_path.is_file() else None,
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))

    print(f"\n-> {out}  ({len(df)} kỳ: {(df.segment=='train').sum()} train / "
          f"{(df.segment=='holdout').sum()} holdout)")
    print(f"-> {meta_path}")

    # ---- 3. Lưới MỊN (4h) — cùng chiến lược, đo ở độ phân giải sàn nhìn thấy ----
    # Lưới 72h không thấy biến động trong kỳ, nên maxDD của nó luôn lạc quan. Lưới mịn
    # cũng là lưới DUY NHẤT có đủ điểm quyết định để bàn về điều khiển đòn bẩy: một
    # tuần chỉ có 2,33 kỳ 72h nhưng có 42 nến 4h.
    idx = data.index
    f_tr = run_v3_fine(data, V3, mask=idx < data.split_ts)
    f_ho = run_v3_fine(data, V3, mask=idx >= data.split_ts)
    df_f = pd.concat([_frame(f_tr, "train"), _frame(f_ho, "holdout")]).sort_index()
    fine_path = out.with_name(out.stem + "_fine.csv")
    df_f.to_csv(fine_path)

    def _mdd(r):
        e = np.cumprod(1.0 + np.asarray(r.dropna()))
        return float((1.0 - e / np.maximum.accumulate(e)).max())

    ppy_f = 24 * 365.0 / V3.bar_hours
    print(f"\nLƯỚI MỊN 4h — cùng chiến lược, độ phân giải sàn thực sự nhìn thấy:")
    print(f"{'':<12}{'nến':>7}{'sharpe':>9}{'maxDD':>9}{'maxDD 72h':>12}{'bị che':>9}")
    print("-" * 58)
    for lbl, rf, rc in (("train", f_tr.returns, res_tr.returns),
                        ("holdout", f_ho.returns, res_ho.returns)):
        x = rf.dropna()
        sh = float(x.mean() / x.std(ddof=1) * np.sqrt(ppy_f))
        print(f"{lbl:<12}{len(x):>7}{sh:>9.2f}{_mdd(rf)*100:>8.1f}%{_mdd(rc)*100:>11.1f}%"
              f"{(_mdd(rf)-_mdd(rc))*100:>8.1f}pp")
    print(f"\n-> {fine_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
