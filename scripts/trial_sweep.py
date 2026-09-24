#!/usr/bin/env python3
"""
ĐO `variance_of_srs` — tham số còn thiếu của Deflated Sharpe Ratio.

VÌ SAO CẦN: `StrategyV3Config.num_trials_declared = 250` khai báo rằng 250 cấu hình
đã được thử trước khi chốt V3. DSR trừ hao đúng phần lợi thế mà việc thử-rồi-chọn
tạo ra, nhưng nó cần BIẾT các lần thử đó phân tán ra sao — `variance_of_srs`.

Con số đó chưa từng được ghi. `logs/experiments/*.jsonl` có 470 bản ghi nhưng toàn
metric hạ tầng (`parity_verified`, `throughput_ticks_sec`); không một trường Sharpe.
Hệ quả đo được (21/09/2026): DSR của holdout trải từ 0.805 xuống 0.003 tuỳ vào giá
trị `variance_of_srs` giả định, và ngưỡng lật nằm ở 0.00168. Tức là toàn bộ kết luận
"chiến lược có vượt nhiễu chọn lọc hay không" phụ thuộc vào một con số chưa ai đo.

Script này chạy lại đúng đường chạy nghiên cứu (`strategy_v3.split_train_holdout`)
trên một lưới cấu hình, ghi Sharpe train VÀ holdout của từng lần thử, rồi tính
phương sai thực nghiệm.

MỘT ĐIỀU PHẢI GIỮ ĐÚNG: V3 được chốt bằng cách chọn trên TẬP TRAIN
(`StrategyV3Config` docstring: "ĐÃ CHỐT trên tập train qua scripts/stability.py").
Vậy `variance_of_srs` phải tính từ Sharpe TRAIN. Sharpe holdout được ghi kèm chỉ để
chẩn đoán — dùng nó để chọn hay để tính phương sai là biến holdout thành train, đúng
cái lỗi mà `frozen=True` trên StrategyV3Config được đặt ra để chặn.

    python scripts/trial_sweep.py --trials 250
    python scripts/trial_sweep.py --trials 40 --seed 7     # chạy nhanh để thử
"""
import argparse
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from aegis.core.experiment_tracker import ExperimentTracker
from aegis.core.trial_classes import TrialClass
from aegis.research.adaptive_combiner import CombinerSpec
from aegis.research.strategy_v3 import (
    V3, V3_VINTAGE_MS, StrategyV3Config, combined_signal, load_v3_data,
    split_train_holdout,
)
from aegis.risk.portfolio import PortfolioSpec

OUT = "artifacts/trial_sharpes.json"

#: Không gian tìm kiếm. Mỗi trục là một tham số TỰ DO thật sự của chiến lược —
#: thứ một người dò tham số sẽ động vào. Giá trị đang chạy của V3 luôn nằm trong
#: danh sách, nên lưới này bao được cấu hình đã chốt.
COMBINER_GRID = {
    "lookback":       [250, 375, 500, 750],
    "min_periods":    [80, 120, 180],
    "t_threshold":    [1.0, 1.5, 2.0, 2.5],
    "max_abs_weight": [0.15, 0.20, 0.30, 0.45],
    "max_step":       [0.03, 0.05, 0.10],
}
PORTFOLIO_GRID = {
    "mode":          ["zscore_riskparity", "rank_riskparity", "zscore", "rank_binary"],
    "n_positions":   [8, 12, 16, 20],
    "max_weight":    [0.15, 0.20, 0.30],
    "vol_window":    [40, 60, 90],
    "z_clip":        [2.0, 2.5, 3.0],
    "vol_floor_pct": [0.05, 0.10, 0.20],
}


def _sharpe(returns, ppy: float) -> tuple:
    """Sharpe năm hoá + số kỳ hữu hiệu. Bỏ NaN — kỳ cuối panel luôn dở dang."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan"), len(r)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy)), len(r)


def _sample(rng, grid: dict) -> dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in grid.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=V3.num_trials_declared)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--portfolios-per-combiner", type=int, default=10,
                    help="Gộp nhiều cấu hình danh mục lên cùng một tín hiệu đã gộp "
                         "— tái dùng `sig` nhanh gấp 3.8x mà kết quả không đổi.")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    tracker = ExperimentTracker()

    print(f"nạp dữ liệu (ghim vintage {V3_VINTAGE_MS})...", flush=True)
    t0 = time.time()
    data = load_v3_data(V3, verbose=True, end_ms=V3_VINTAGE_MS)
    print(f"  xong {time.time() - t0:.1f}s", flush=True)

    # --- PARITY: baseline phải tái lập đúng số đã công bố ------------------
    ref = json.load(open("artifacts/returns_v3.meta.json"))
    ppy = V3.periods_per_year
    tr0, ho0 = split_train_holdout(data, V3)
    s_tr, n_tr = _sharpe(tr0.returns, ppy)
    s_ho, n_ho = _sharpe(ho0.returns, ppy)
    ok = (abs(s_tr - ref["train"]["sharpe"]) < 1e-6 and n_tr == ref["train"]["n"]
          and abs(s_ho - ref["holdout"]["sharpe"]) < 1e-6 and n_ho == ref["holdout"]["n"])
    print(f"\nPARITY baseline: train SR={s_tr:.6f} (ref {ref['train']['sharpe']:.6f}) | "
          f"holdout SR={s_ho:.6f} (ref {ref['holdout']['sharpe']:.6f}) "
          f"{'✅' if ok else '❌'}", flush=True)
    if not ok:
        print("DỪNG: baseline không tái lập được số đã công bố — sweep sẽ vô nghĩa.")
        return 1

    n_comb = max(1, a.trials // a.portfolios_per_combiner)
    print(f"\nsweep {a.trials} trial = {n_comb} combiner x {a.portfolios_per_combiner} "
          f"danh mục, seed={a.seed}\n", flush=True)

    rows, t_start, done = [], time.time(), 0
    for ci in range(n_comb):
        cpar = _sample(rng, COMBINER_GRID)
        cfg_c = StrategyV3Config(combiner=CombinerSpec(**cpar))
        try:
            sig = combined_signal(data, cfg_c)
        except Exception as exc:                      # cấu hình gộp không hợp lệ
            print(f"  [combiner {ci}] LỖI {type(exc).__name__}: {exc}", flush=True)
            continue

        for _ in range(a.portfolios_per_combiner):
            ppar = _sample(rng, PORTFOLIO_GRID)
            done += 1
            try:
                spec = PortfolioSpec(top_frac=V3.top_frac, **ppar)
                cfg = StrategyV3Config(combiner=CombinerSpec(**cpar), portfolio=spec)
                tr, ho = split_train_holdout(data, cfg, sig=sig)
                sr_tr, ntr = _sharpe(tr.returns, ppy)
                sr_ho, nho = _sharpe(ho.returns, ppy)
                status = "ok"
            except Exception as exc:
                sr_tr = sr_ho = float("nan"); ntr = nho = 0
                status = f"{type(exc).__name__}: {exc}"

            params = {"combiner": cpar, "portfolio": ppar}
            metrics = {"sharpe_train": sr_tr, "sharpe_holdout": sr_ho,
                       "n_train": ntr, "n_holdout": nho, "status": status}
            tracker.log_trial(TrialClass.STRATEGY_SELECTION, params, metrics)
            rows.append({**{f"c_{k}": v for k, v in cpar.items()},
                         **{f"p_{k}": v for k, v in ppar.items()}, **metrics})

            if done % 25 == 0:
                el = time.time() - t_start
                print(f"  {done}/{a.trials} trial | {el/60:.1f}p | "
                      f"còn ~{el/done*(a.trials-done)/60:.1f}p", flush=True)

    df = pd.DataFrame(rows)
    good = df[df.status == "ok"].dropna(subset=["sharpe_train"])
    # DSR nhận Sharpe THEO KỲ (sample_length tính bằng kỳ), nên quy ngược về.
    sr_tr_per = good["sharpe_train"].values / np.sqrt(ppy)
    var_srs = float(np.var(sr_tr_per, ddof=1))

    out = {
        "seed": a.seed,
        "n_trials_run": int(len(df)),
        "n_trials_ok": int(len(good)),
        "vintage_ms": V3_VINTAGE_MS,
        "variance_of_srs_per_period": var_srs,
        "std_of_srs_per_period": float(np.sqrt(var_srs)),
        "sharpe_train_annual": {
            "mean": float(good["sharpe_train"].mean()),
            "std": float(good["sharpe_train"].std(ddof=1)),
            "min": float(good["sharpe_train"].min()),
            "max": float(good["sharpe_train"].max()),
            "q25": float(good["sharpe_train"].quantile(0.25)),
            "q50": float(good["sharpe_train"].quantile(0.50)),
            "q75": float(good["sharpe_train"].quantile(0.75)),
        },
        "sharpe_holdout_annual": {
            "mean": float(good["sharpe_holdout"].mean()),
            "std": float(good["sharpe_holdout"].std(ddof=1)),
        },
        "v3_baseline": {"sharpe_train": s_tr, "sharpe_holdout": s_ho},
        "grids": {"combiner": COMBINER_GRID, "portfolio": PORTFOLIO_GRID},
    }
    pathlib.Path(OUT).write_text(json.dumps(out, indent=2))
    df.to_csv("artifacts/trial_sharpes.csv", index=False)

    print("\n" + "=" * 70)
    print(f"XONG {len(good)}/{len(df)} trial hợp lệ — {(time.time()-t_start)/60:.1f} phút")
    print("=" * 70)
    print(f"  Sharpe train (năm hoá): TB {out['sharpe_train_annual']['mean']:.3f} | "
          f"đlc {out['sharpe_train_annual']['std']:.3f} | "
          f"min {out['sharpe_train_annual']['min']:.3f} | "
          f"max {out['sharpe_train_annual']['max']:.3f}")
    print(f"  V3 đã chốt            : train {s_tr:.3f} | holdout {s_ho:.3f}")
    print(f"\n  variance_of_srs (theo kỳ) = {var_srs:.6f}")
    print(f"  -> ghi {OUT} và artifacts/trial_sharpes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
