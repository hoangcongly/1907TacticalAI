#!/usr/bin/env python3
"""
TẦNG RỦI RO: mục tiêu biến động + van drawdown có nâng được TỐC ĐỘ TĂNG TRƯỞNG không.

PHÁT HIỆN DẪN TỚI SCRIPT NÀY: `research/leverage.py` có đủ `volatility_target_series`,
`drawdown_throttle`, `apply_leverage` — và đường v3 KHÔNG GỌI CÁI NÀO. `validate_v3.py`
import `apply_leverage` rồi không dùng; live chạy gross cố định 2,0x với đúng một van
thô `0.5 nếu drawdown >= 10%`. Cả một tầng đã viết xong nằm chết ngoài đường tiền.

VÌ SAO ĐÁNG THỬ, VÀ VÌ SAO KHÔNG PHẢI LÀ ĐOÁN MÒ:
Biến động có tính CỤM và DỰ BÁO ĐƯỢC; lợi suất thì không. Mục tiêu biến động là cách
duy nhất khai thác thứ duy nhất trong chuỗi lợi suất thực sự dự báo được. Và với tốc
độ tăng trưởng, lợi ích còn lớn hơn với Sharpe: tăng trưởng log là
`L*mu - L^2*sigma^2/2`, nên số hạng phạt tỷ lệ với BÌNH PHƯƠNG biến động. Chạy gross
cố định qua một giai đoạn biến động gấp đôi là trả gấp bốn lần phần phạt đó.

Đo được ở đây: Sharpe theo giai đoạn của v3 dao động 0,04 - 2,96 qua 8 giai đoạn con.
Biến động cũng dao động tương ứng. Đó là điều kiện lý tưởng cho tầng này.

RÀNG BUỘC KHẢ THI — chỗ mà mọi nghiên cứu vol-target ở tài khoản nhỏ thường bỏ sót:
mục tiêu biến động 30%/năm trên chiến lược vol 45% đòi gross 0,67x. Nhưng gross 0,67x
với $38 nghĩa là $2,12/vị thế — DƯỚI min notional $5, sàn từ chối từng lệnh. Nên đòn
bẩy khả thi là `{0} hợp [1,58x, trần]`, y hệt ràng buộc trong `goal_dp.py`. Script này
áp ràng buộc đó; bỏ nó đi sẽ cho ra một kết quả đẹp mà không đặt được lệnh.

CHỌN THEO ĐỘ ỔN ĐỊNH, KHÔNG THEO ĐỈNH. Lưới ở đây có 4 tham số tự do — thừa đủ để
overfit. Vì vậy mọi cấu hình được chấm bằng trung vị qua 8 giai đoạn con và tỷ lệ giai
đoạn dương, và bảng in ra CẢ MẶT để thấy đó là cao nguyên hay một cái gai.

    python scripts/risk_overlay_study.py
    python scripts/risk_overlay_study.py --folds 8 --grid-vol 0.7,0.9,1.1,1.3
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, combined_signal, load_v3_data, run_v3,
)

OUT_CSV = "artifacts/risk_overlay_study.csv"


# ---------------------------------------------------------------------------
def executable_leverage(
    returns: pd.Series,
    target_ann_vol: float,
    vol_window: int,
    periods_per_year: float,
    lev_floor: float,
    lev_cap: float,
    dd_start: float = 0.0,
    dd_stop: float = 0.0,
    peak_window: int = 60,
) -> pd.DataFrame:
    """
    Chuỗi đòn bẩy KHẢ THI theo thời gian, cộng đường vốn khi áp nó.

    Khác `leverage.apply_leverage` ở đúng một điểm nhưng là điểm quyết định: đòn bẩy
    bị KẸP vào `{0} hợp [lev_floor, lev_cap]`. Dưới sàn thì không đặt được lệnh, nên
    lựa chọn duy nhất là ĐỨNG NGOÀI. `apply_leverage` cho phép mọi giá trị liên tục —
    đúng cho quỹ lớn, sai cho tài khoản $38.

    Van drawdown chạy TUẦN TỰ vì đường vốn phụ thuộc chính đòn bẩy đã dùng; không
    vector hoá được mà không nhìn trước.

    ⚠️ `peak_window` KHÔNG phải chi tiết vụn — nó vá một cái bẫy chết người.
    Van drawdown đo theo đỉnh MỌI THỜI ĐẠI, cộng với sàn đòn bẩy, tạo ra một trạng
    thái HẤP THỤ: van đẩy L xuống dưới sàn -> L = 0 -> không còn lợi suất -> vốn đứng
    yên dưới đỉnh cũ -> drawdown không bao giờ giảm -> L kẹt ở 0 VĨNH VIỄN. Đo được ở
    bản đầu: đứng ngoài thị trường 93% thời gian và không bao giờ quay lại.

    Bẫy này không lộ ra ở quỹ lớn vì ở đó không có sàn đòn bẩy — L giảm dần chứ không
    rơi về 0, nên vốn vẫn động đậy và vẫn hồi được. Nó CHỈ xuất hiện khi min notional
    biến "giảm rủi ro" thành "đóng sạch". Tài khoản nhỏ gặp đúng phiên bản nguy hiểm.

    Cách vá: đo drawdown theo đỉnh TRONG CỬA SỔ TRƯỢT `peak_window` kỳ, không theo
    đỉnh mọi thời đại. Đỉnh cũ tự hết hạn, nên hệ thống luôn có đường quay lại. Đây
    cũng là cách mọi hệ thống thật làm — một đỉnh từ ba năm trước không nói gì về rủi
    ro hôm nay.
    """
    r = returns.dropna()
    realized = r.rolling(vol_window, min_periods=max(5, vol_window // 3)).std().shift(1)
    ann = realized * np.sqrt(periods_per_year)
    raw = (target_ann_vol / ann.replace(0.0, np.nan)).reindex(r.index)

    rv, raw_v = r.to_numpy(), raw.to_numpy()
    out_r = np.zeros(len(r)); out_l = np.zeros(len(r))
    equity = 1.0
    hist = [1.0]                               # đường vốn để lấy đỉnh cửa sổ trượt
    span = max(dd_stop - dd_start, 1e-9)

    for i in range(len(r)):
        L = raw_v[i]
        if not np.isfinite(L):
            L = lev_floor                      # chưa đủ dữ liệu -> mức tối thiểu
        if dd_stop > dd_start:
            peak = max(hist[-peak_window:])    # đỉnh TRONG CỬA SỔ, không phải mọi thời đại
            dd = 1.0 - equity / peak if peak > 0 else 0.0
            if dd > dd_start:
                L *= float(np.clip(1.0 - (dd - dd_start) / span, 0.0, 1.0))
        # Kẹp vào tập khả thi: dưới sàn thì ĐỨNG NGOÀI, trên trần thì cắt.
        L = 0.0 if L < lev_floor else min(L, lev_cap)

        step = max(L * rv[i], -0.999)
        equity *= (1.0 + step)
        hist.append(equity)
        out_r[i], out_l[i] = step, L

    return pd.DataFrame({"returns": out_r, "leverage": out_l,
                         "equity": np.cumprod(1.0 + out_r)}, index=r.index)


def _fold_stats(r: np.ndarray, bounds, ppy):
    """
    Sharpe từng giai đoạn con. Giai đoạn ĐỨNG NGOÀI HOÀN TOÀN trả về 0.0, không phải NaN.

    Trả NaN sẽ khiến `nanmedian` bỏ qua giai đoạn đó — và một cấu hình đứng ngoài 7/8
    giai đoạn sẽ được chấm bằng đúng giai đoạn nó có mặt, tức là được thưởng vì đã
    trốn. Đứng ngoài là một QUYẾT ĐỊNH và nó kiếm được 0; phải tính là 0.
    """
    out = []
    for i in range(len(bounds) - 1):
        x = r[bounds[i]:bounds[i + 1]]
        x = x[np.isfinite(x)]
        if len(x) < 10:
            out.append(np.nan)
        elif x.std(ddof=1) < 1e-15:
            out.append(0.0)                    # không giao dịch => Sharpe 0
        else:
            out.append(float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)))
    return out


def _growth_and_dd(r: np.ndarray):
    r = r[np.isfinite(r)]
    eq = np.cumprod(1.0 + r)
    dd = float((1.0 - eq / np.maximum.accumulate(eq)).max())
    return float(np.log(eq[-1]) / len(r)), dd      # log-growth mỗi kỳ, maxDD


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--grid-vol", default="0.6,0.8,0.9,1.0,1.2,1.5")
    ap.add_argument("--grid-window", default="20,40,60,90")
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args(argv)

    vols = [float(x) for x in a.grid_vol.split(",")]
    wins = [int(x) for x in a.grid_window.split(",")]

    data = load_v3_data(V3, end_ms=V3_VINTAGE_MS)
    sig = combined_signal(data, V3)
    base = run_v3(data, V3, sig=sig).returns.dropna()       # gross 1.0, toàn dòng thời gian
    ppy = V3.periods_per_year
    lev_floor = V3.n_positions * V3.min_notional_usd / V3.capital_usd
    ppw = V3.periods_per_week

    n = len(base)
    bounds = np.linspace(0, n, a.folds + 1).astype(int)
    labels = [f"{pd.to_datetime(base.index[bounds[i]], unit='ms'):%y/%m}" for i in range(a.folds)]

    print(f"\nnền: {n} kỳ, Sharpe {base.mean()/base.std(ddof=1)*np.sqrt(ppy):.2f}, "
          f"vol {base.std(ddof=1)*np.sqrt(ppy)*100:.1f}%/năm ở gross 1.0x")
    print(f"sàn đòn bẩy khả thi {lev_floor:.2f}x ({V3.n_positions} vị thế x "
          f"${V3.min_notional_usd:.0f} / ${V3.capital_usd:.0f}), trần {a.cap:.1f}x\n")

    rows = []

    def evaluate(label, series, lev_series):
        r = np.asarray(series)
        folds = _fold_stats(r, bounds, ppy)
        g, dd = _growth_and_dd(r)
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ppy))
        med = float(np.nanmedian(folds))
        return {"config": label, "sharpe": sh, "fold_median": med,
                "frac_pos": float(np.mean([f > 0 for f in folds if np.isfinite(f)])),
                "worst_fold": float(np.nanmin(folds)),
                "growth_week": float(np.expm1(g * ppw)),
                "max_dd": dd, "ann_vol": float(np.std(r, ddof=1) * np.sqrt(ppy)),
                "avg_lev": float(np.mean(lev_series)),
                "frac_flat": float(np.mean(np.asarray(lev_series) < 1e-9)),
                **{f"fold_{i}": folds[i] for i in range(a.folds)}}

    # --- mốc so sánh: gross CỐ ĐỊNH (hành vi hiện tại) ---------------------
    for L in (1.58, 2.0, 2.5, 3.0, 3.5):
        r = np.maximum(base.to_numpy() * L, -0.999)
        rows.append(evaluate(f"cố định {L:.2f}x", r, np.full(n, L)))

    # --- mục tiêu biến động, có và không có van drawdown -------------------
    for tv in vols:
        for w in wins:
            for dd_pair in ((0.0, 0.0), (0.15, 0.35)):
                res = executable_leverage(base, tv, w, ppy, lev_floor, a.cap,
                                          dd_start=dd_pair[0], dd_stop=dd_pair[1])
                tag = f"vol {tv:.0%}/w{w}" + (" +van" if dd_pair[1] > 0 else "")
                rows.append(evaluate(tag, res["returns"].to_numpy(),
                                     res["leverage"].to_numpy()))

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    base_row = df[df["config"] == "cố định 2.00x"].iloc[0]

    def show(sub, title):
        print("=" * 118)
        print(title)
        print("=" * 118)
        print(f"{'cấu hình':<20}{'sharpe':>8}{'tr.vị fold':>11}{'%dương':>8}{'tệ nhất':>9}"
              f"{'vol':>8}{'maxDD':>8}{'L tb':>7}{'ngoài':>7}{'tăng/tuần':>11}{'VND/tuần':>11}")
        print("-" * 118)
        for _, r in sub.iterrows():
            star = " *" if r["growth_week"] > base_row["growth_week"] else "  "
            print(f"{r['config']:<20}{r['sharpe']:>8.2f}{r['fold_median']:>11.2f}"
                  f"{r['frac_pos']*100:>7.0f}%{r['worst_fold']:>9.2f}"
                  f"{r['ann_vol']*100:>7.0f}%{r['max_dd']*100:>7.0f}%{r['avg_lev']:>7.2f}"
                  f"{r['frac_flat']*100:>6.0f}%{r['growth_week']*100:>10.2f}%"
                  f"{r['growth_week']*1_000_000:>10,.0f}{star}")

    show(df[df["config"].str.startswith("cố định")], "MỐC SO SÁNH — gross cố định (hành vi hiện tại)")
    top = df[~df["config"].str.startswith("cố định")].nlargest(14, "fold_median")
    print()
    show(top, "MỤC TIÊU BIẾN ĐỘNG — 14 cấu hình có trung vị fold cao nhất  (* = tăng trưởng > 2.0x)")

    print(f"\n-> {a.out}")
    best = df.loc[df["fold_median"].idxmax()]
    print(f"\nTốt nhất theo ĐỘ ỔN ĐỊNH: {best['config']}")
    print(f"  Sharpe {best['sharpe']:.2f} (nền 2.0x: {base_row['sharpe']:.2f}) | "
          f"trung vị fold {best['fold_median']:.2f} (nền {base_row['fold_median']:.2f})")
    print(f"  tăng trưởng {best['growth_week']*100:.2f}%/tuần = "
          f"{best['growth_week']*1_000_000:,.0f} VND (nền {base_row['growth_week']*1_000_000:,.0f} VND)")
    print(f"  maxDD {best['max_dd']*100:.0f}% (nền {base_row['max_dd']*100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
