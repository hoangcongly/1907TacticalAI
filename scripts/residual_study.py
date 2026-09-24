#!/usr/bin/env python3
"""
ĐỘNG LƯỢNG PHẦN DƯ và KHỬ BETA ở n=50 — hai hướng tài liệu CHƯA được thử trên cấu hình wide.

NGUỒN:
  * Blitz, Huij & Martens (2011), "Residual momentum", J. Empirical Finance: xếp hạng
    trên lợi suất đã khử nhân tố cho lợi nhuận điều chỉnh rủi ro ~2 lần động lượng
    thô, ổn định hơn, và giữ được ngoài mẫu sau khi công bố.
  * Blitz, Huij, Lansdorp & Verbeek (2013), "Short-term residual reversal", J.
    Financial Markets: cùng kết quả ~2 lần ở chân trời ngắn.

VÌ SAO CÓ THỂ HỢP VỚI CRYPTO HƠN CỔ PHIẾU: altcoin phần lớn chạy theo cả thị trường.
Động lượng thô vì thế xếp hạng `beta x lợi suất thị trường quá khứ` — sổ trung lập
đô-la vẫn ngầm cược thị trường. `tests/research/test_residual_momentum.py` khoá cơ
chế đó trên panel tổng hợp: thô tương quan hạng 0,60-0,97 với beta, phần dư 0,00-0,07.

VÌ SAO KHỬ BETA ĐƯỢC ĐO LẠI dù §8 báo cáo v3 ghi "≈ hoà (−0,04)": con số đó đo ở
n=12, và chính báo cáo giải thích "với 12 vị thế, ràng buộc beta cứng đòi dịch chuyển
trọng số quá nhiều". Ở n=50 lý do đó yếu đi hẳn. Quyết định chốt dưới một cấu hình
KHÔNG tự chuyển sang cấu hình khác — bài học n_positions=12 ngày 18/09.

VÌ SAO KHÔNG THỬ hai cơ chế tài liệu khác (chia lô tái cân bằng — Hoffstein, Faber &
Braun; giao dịch từng phần — Gârleanu & Pedersen 2013): cả hai sinh ra nhiều lệnh
nhỏ, mà ở vốn $38 x 5x / 50 vị thế thì mỗi lệnh đã sát min notional $5. Chúng không
đặt được lệnh, nên đo cũng vô ích.

═══════════════════════════════════════════════════════════════════════════════
LUẬT QUYẾT ĐỊNH — CHỐT TRƯỚC KHI CHẠY (in ra cùng kết quả, không sửa sau khi nhìn)
═══════════════════════════════════════════════════════════════════════════════
Một biến thể chỉ được coi là THẮNG khi đủ CẢ BA:
  1. Sharpe tăng >= +0,10 ở CẢ HAI cơ sở (kỷ nguyên >=100 cặp VÀ toàn lịch sử).
  2. Bootstrap khối GHÉP CẶP: P(Sharpe biến thể > Sharpe gốc) >= 90% ở kỷ nguyên >=100.
  3. Fold tệ nhất (8 đoạn con) không tệ hơn bản gốc.
Thắng rồi VẪN chưa được đưa vào live: holdout đã dùng 2 lần, kỷ nguyên >=100 chồng lên
holdout, nên đây là bằng chứng ĐỘ ỔN ĐỊNH chứ không phải ngoài mẫu. Bước sau bắt buộc
là giao dịch giấy tiến về phía trước, như mọi thay đổi chiến lược khác.

5 biến thể = 5 lần thử thêm vào 250 đã khai báo (`num_trials_declared`). Ghi vào
`artifacts/residual_study.csv` để `dsr_report.py` tính được.

    python scripts/residual_study.py              # dữ liệu thật (máy có data/)
    python scripts/residual_study.py --synthetic  # CHỈ kiểm tra đường chạy, KHÔNG phải kết quả
"""
import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from aegis.research.signal_library import build_signal
from aegis.research.strategy_v3 import (
    V3_SIGNALS, V3_VINTAGE_MS, V3Data, combined_signal, config_from_json, load_v3_data, run_v3,
)
from aegis.risk.portfolio import PortfolioSpec

#: [F51] So với ĐÚNG cấu hình daemon chạy (`strategy_v3_wide.json`, kể cả tầng gộp
#: `max_step=0,025`), không phải tầng gộp của V3 (0,05) mà các nghiên cứu wide cũ dùng.
# Đo trên MỘT lô, tường minh: so sánh TÍN HIỆU không phụ thuộc cách chia lô, và `run_v3`
# từ chối cấu hình chia lô thay vì âm thầm bỏ qua nó.
BASE = replace(config_from_json(), n_tranches=1)
N = BASE.n_positions
MAKER = 0.39                     # mức THẬT đo ở lượt 19/09
RMOM = ("rmom_fast", "rmom_mid", "rmom_slow", "rmom_vlong")
SWAP = dict(zip(("mom_fast", "mom_mid", "mom_slow", "mom_vlong"), RMOM))
OUT_CSV = "artifacts/residual_study.csv"

MIN_DSHARPE = 0.10
MIN_P_BETTER = 0.90


def _spec(beta_neutral: bool) -> PortfolioSpec:
    return replace(BASE.portfolio, beta_neutral=beta_neutral)


def variants():
    """(nhãn, danh sách tín hiệu, khử beta?). Bản gốc đứng ĐẦU — mọi so sánh lấy nó làm mốc."""
    swapped = tuple(SWAP.get(n, n) for n in V3_SIGNALS)
    return [
        ("V3 wide (đang chạy)", V3_SIGNALS, False),
        ("THAY 4 mom -> rmom", swapped, False),
        ("THÊM 4 rmom (30 tín hiệu)", V3_SIGNALS + RMOM, False),
        ("khử beta", V3_SIGNALS, True),
        ("THAY + khử beta", swapped, True),
    ]


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------
def _sharpe(r: np.ndarray, ppy: float) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def _stats(r: pd.Series, mkt: pd.Series, ppy: float) -> dict:
    v = r.to_numpy()
    folds = [_sharpe(f, ppy) for f in np.array_split(v, 8) if len(f) > 5]
    m = mkt.reindex(r.index).to_numpy()
    ok = np.isfinite(m)
    beta = (np.cov(v[ok], m[ok])[0, 1] / np.var(m[ok], ddof=1)) if ok.sum() > 5 else np.nan
    return {
        "sharpe": _sharpe(v, ppy), "ann": v.mean() * ppy, "vol": v.std(ddof=1) * np.sqrt(ppy),
        "worst": v.min(), "fold_med": float(np.median(folds)),
        "fold_pos": float(np.mean(np.array(folds) > 0)), "fold_min": float(np.min(folds)),
        "beta_mkt": float(beta), "n": len(v),
    }


def paired_p_better(a: np.ndarray, b: np.ndarray, ppy: float, block: int = 6,
                    n_boot: int = 5000, seed: int = 7) -> float:
    """
    P(Sharpe(b) > Sharpe(a)) bằng bootstrap khối vòng GHÉP CẶP.

    Ghép cặp là bắt buộc: hai biến thể chạy trên CÙNG thị trường nên lợi suất tương
    quan rất cao; bootstrap riêng từng chuỗi sẽ thổi phồng sai số của hiệu số và làm
    mọi khác biệt trông như nhiễu. Lấy mẫu theo khối để giữ cụm biến động.
    """
    n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
    A, B = a[idx], b[idx]

    def sh(x):
        return x.mean(1) / x.std(1, ddof=1) * np.sqrt(ppy)
    return float(np.mean(sh(B) > sh(A)))


# ---------------------------------------------------------------------------
# Dữ liệu tổng hợp — CHỈ để kiểm tra đường chạy trong môi trường không có data/
# ---------------------------------------------------------------------------
def synthetic_data(n_bars: int = 6 * 365 * 3, n_assets: int = 130, seed: int = 0) -> V3Data:
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.5, 2.0, n_assets)
    m = 0.0002 + 0.012 * rng.standard_normal(n_bars)
    alpha = np.zeros((n_bars, n_assets))
    for t in range(1, n_bars):                       # drift riêng bền (AR(1))
        alpha[t] = 0.995 * alpha[t - 1] + 0.00005 * rng.standard_normal(n_assets)
    r = m[:, None] * betas + alpha + 0.01 * rng.standard_normal((n_bars, n_assets))
    idx = pd.Index(1_600_000_000_000 + np.arange(n_bars) * 4 * 3_600_000, name="ts")
    cols = [f"S{i:03d}USDT" for i in range(n_assets)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    vol = pd.DataFrame(np.exp(rng.normal(14, 1, (n_bars, n_assets))), index=idx, columns=cols)
    panel = {
        "close": close, "high": close * (1 + np.abs(r) + 0.002),
        "low": close * (1 - np.abs(r) - 0.002), "volume": vol,
        "ofi": pd.DataFrame(np.tanh(50 * r + rng.standard_normal(r.shape)), index=idx,
                            columns=cols),
    }
    funding = pd.DataFrame(0.0001 * rng.standard_normal((n_bars, n_assets)), index=idx,
                           columns=cols)
    names = tuple(dict.fromkeys(V3_SIGNALS + RMOM))
    signals = {n: build_signal(n, panel, funding) for n in names}
    return V3Data(panel=panel, funding=funding, signals=signals,
                  per_symbol_bps=pd.Series(3.0, index=cols),
                  split_ts=int(idx[len(idx) * 2 // 3]), data_end_ms=int(idx[-1]))


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="chi phí một chiều ĐO THẬT (vd 15.7); bỏ trống = mô hình")
    ap.add_argument("--synthetic", action="store_true",
                    help="panel giả lập — CHỈ kiểm tra đường chạy, KHÔNG phải kết quả")
    a = ap.parse_args(argv)

    if a.synthetic:
        print("!" * 96)
        print("!! CHẾ ĐỘ TỔNG HỢP — mọi con số dưới đây là của dữ liệu GIẢ, chỉ chứng minh script")
        print("!! chạy hết đường. KHÔNG dùng để quyết định gì. KHÔNG ghi artifact.")
        print("!" * 96)
        data = synthetic_data()
    else:
        cfg_all = replace(BASE, signals=tuple(dict.fromkeys(V3_SIGNALS + RMOM)))
        data = load_v3_data(cfg_all, end_ms=V3_VINTAGE_MS)

    ppy = BASE.periods_per_year
    marks = data.close.index[::BASE.rebalance_every]
    px = data.close.reindex(marks)
    mkt = (px.shift(-1) / px - 1.0).mean(axis=1)       # lợi suất TIẾN, khớp quy ước `simulate`
    hi = data.close.notna().sum(axis=1) >= 100
    bases = {"kỷ nguyên >=100 cặp": set(hi.index[hi.values]), "toàn lịch sử": None}

    rets = {}
    for label, names, bn in variants():
        cfg = replace(BASE, signals=tuple(names))
        sig = combined_signal(data, cfg)
        rets[label] = run_v3(data, cfg, maker_ratio=MAKER, portfolio=_spec(bn),
                             sig=sig, cost_bps=a.cost_bps).returns.dropna()
        print(f"  xong: {label}", flush=True)

    base_label = variants()[0][0]
    rows = []
    for bname, keep in bases.items():
        print("\n" + "=" * 110)
        print(f"{bname.upper()} — n={N}, trọng số đều (max_w=1/{N}), maker {MAKER}")
        print("=" * 110)
        print(f"{'biến thể':<28}{'Sharpe':>8}{'ΔSharpe':>9}{'ann%':>7}{'vol%':>7}{'kỳ tệ%':>8}"
              f"{'fold tv':>8}{'%fold+':>7}{'fold min':>9}{'beta TT':>9}{'corr gốc':>9}"
              f"{'P(hơn)':>8}")
        print("-" * 110)
        r0 = rets[base_label]
        r0 = r0 if keep is None else r0[r0.index.isin(keep)]
        s0 = _stats(r0, mkt, ppy)
        for label, _, _ in variants():
            r = rets[label]
            r = r if keep is None else r[r.index.isin(keep)]
            common = r.index.intersection(r0.index)
            st = _stats(r.loc[common], mkt, ppy)
            a0, a1 = r0.loc[common].to_numpy(), r.loc[common].to_numpy()
            p = np.nan if label == base_label else paired_p_better(a0, a1, ppy)
            corr = float(np.corrcoef(a0, a1)[0, 1])
            d = st["sharpe"] - s0["sharpe"]
            print(f"{label:<28}{st['sharpe']:>8.2f}{d:>+9.2f}{st['ann']*100:>7.1f}"
                  f"{st['vol']*100:>7.1f}{st['worst']*100:>8.2f}{st['fold_med']:>8.2f}"
                  f"{st['fold_pos']*100:>6.0f}%{st['fold_min']:>9.2f}{st['beta_mkt']:>+9.3f}"
                  f"{corr:>9.3f}{'' if np.isnan(p) else format(p, '.0%'):>8}")
            rows.append({"base": bname, "variant": label, "d_sharpe": d, "p_better": p,
                         "corr_base": corr, **st})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(f"PHÁN QUYẾT theo luật chốt trước: ΔSharpe >= +{MIN_DSHARPE:.2f} ở CẢ HAI cơ sở, "
          f"P(hơn) >= {MIN_P_BETTER:.0%} ở kỷ nguyên >=100, fold tệ nhất không tệ hơn")
    print("=" * 110)
    for label, _, _ in variants()[1:]:
        v = df[df.variant == label].set_index("base")
        g = df[df.variant == base_label].set_index("base")
        c1 = bool((v["d_sharpe"] >= MIN_DSHARPE).all())
        c2 = bool(v.loc["kỷ nguyên >=100 cặp", "p_better"] >= MIN_P_BETTER)
        c3 = bool((v["fold_min"] >= g["fold_min"] - 1e-12).all())
        verdict = "THẮNG -> giao dịch giấy tiến về phía trước" if (c1 and c2 and c3) else "ĐÓNG"
        print(f"  {label:<28} ΔSharpe {'✓' if c1 else '✗'}  P(hơn) {'✓' if c2 else '✗'}  "
              f"fold {'✓' if c3 else '✗'}   =>  {verdict}")
    print("\n'beta TT' = beta lợi suất chiến lược với rổ đều trọng số. Động lượng phần dư phải")
    print("kéo |beta| về gần 0 — nếu không, tín hiệu không làm đúng việc tài liệu mô tả.")

    if not a.synthetic:
        df.to_csv(OUT_CSV, index=False)
        print(f"\nđã ghi {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
