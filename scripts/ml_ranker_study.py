#!/usr/bin/env python3
"""Mô hình học máy có xếp hạng tốt hơn bộ gộp thích ứng không?

Chạy sau `scripts/download_vision_history.py`. Mọi ứng viên dùng CÙNG 26 tín hiệu, cùng universe
theo từng thời điểm (mọi cặp từng niêm yết), cùng chi phí, và được chấm trên CÙNG các kỳ.
Danh sách ứng viên và siêu tham số được cố định trước khi xem kết quả; không dò gì trên dữ liệu
đánh giá. Mô hình tại kỳ t chỉ học từ các kỳ trước t - purge.

    python scripts/ml_ranker_study.py
"""

import argparse
import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aegis.research import ml_ranker as ml  # noqa: E402
from aegis.research import survivorship as study  # noqa: E402
from aegis.research.rank_model import RankModelSpec  # noqa: E402
from aegis.research.strategy_v3 import (
    V3,
    V3_VINTAGE_MS,
    combined_signal,
    load_v3_data,
)  # noqa: E402
from aegis.validation.dsr import compute_deflated_sharpe_ratio  # noqa: E402

PORTFOLIOS = {"n12": {}, "n50_equal": {"n_positions": 50, "max_weight": 0.02}}
ERAS = {
    "evaluated": "2000-01-01",
    "since_2023_11": "2023-11-01",
    "since_2025_02": "2025-02-01",
}
BASELINE = "combiner"


def cross_sectional_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Hạng trong kỳ, thuộc [-0.5, 0.5]: mọi tín hiệu về cùng thang đo, không dùng thông tin kỳ khác."""
    return frame.rank(axis=1, pct=True) - 0.5


def candidates(spec: RankModelSpec) -> dict:
    return {
        "ridge": ml.ridge_model(),
        "lambdarank": ml.lambdarank_model(spec),
        "mlp": ml.mlp_model(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-universe", default="artifacts/universe_all_listed.json")
    parser.add_argument("--min-daily-usd", type=float, default=20_000_000.0)
    parser.add_argument("--end-ms", type=int, default=V3_VINTAGE_MS)
    parser.add_argument("--out", default="artifacts/ml_ranker_study.json")
    args = parser.parse_args(argv)
    ppy = V3.periods_per_year
    spec = RankModelSpec()

    cfg = replace(V3, universe_file=args.all_universe, min_coverage=0.0)
    data = load_v3_data(cfg, end_ms=args.end_ms)
    close = data.close
    eligible = study.point_in_time_eligibility(
        close, data.panel["volume"], min_daily_usd=args.min_daily_usd
    )
    masked = replace(data, signals=study.mask_signals(data.signals, eligible))
    marks = close.index[:: cfg.rebalance_every]
    exits = study.terminal_close(close, cfg.rebalance_every).reindex(marks)
    forward = exits.shift(-1) / close.reindex(marks) - 1.0
    features = {
        name: cross_sectional_rank(masked.signals[name].reindex(marks))
        for name in cfg.signal_names()
    }

    scores = {BASELINE: combined_signal(masked, cfg)}
    for name, model in candidates(spec).items():
        scores[name] = ml.walk_forward_scores(features, forward, model, spec)
        print(f"  {name} xong", flush=True)
    scores["blend"] = (
        cross_sectional_rank(scores[BASELINE])
        + cross_sectional_rank(scores["lambdarank"])
    ) / 2

    # Chấm trên các kỳ mà MỌI ứng viên đều có điểm số.
    ready = pd.concat(
        [frame.notna().sum(axis=1) >= spec.min_names for frame in scores.values()],
        axis=1,
    ).all(axis=1)
    first = ready[ready].index[0]
    report = {
        "end_ms": args.end_ms,
        "symbols": int(close.shape[1]),
        "features": len(features),
        "evaluated_from": str(pd.to_datetime(first, unit="ms").date()),
        "schedule": {
            "train_periods": spec.train_periods,
            "retrain_every": spec.retrain_every,
            "purge": spec.purge,
            "min_train_periods": spec.min_train_periods,
        },
        "candidates": list(scores),
        "rank_ic": {},
        "runs": {},
        "versus_combiner": {},
        "deflated_sharpe": {},
    }
    returns = {}
    for name, frame in scores.items():
        frame = frame.where(frame.index >= first)
        report["rank_ic"][name] = ml.rank_ic(frame, forward)
        for label, change in PORTFOLIOS.items():
            result = study.simulate_scores(
                data, cfg, frame, replace(cfg.portfolio, **change)
            )
            returns[(name, label)] = result.returns[result.returns.index >= first]
    for (name, label), series in returns.items():
        when = pd.to_datetime(series.index, unit="ms")
        report["runs"].setdefault(label, {})[name] = {
            era: study.performance(series[when >= pd.Timestamp(start)], ppy)
            for era, start in ERAS.items()
        }
    for label in PORTFOLIOS:
        base = returns[(BASELINE, label)]
        report["versus_combiner"][label] = {}
        for name in scores:
            if name == BASELINE:
                continue
            series = returns[(name, label)]
            recent = series.index >= int(
                pd.Timestamp(ERAS["since_2023_11"]).timestamp() * 1000
            )
            report["versus_combiner"][label][name] = {
                "evaluated": study.sharpe_difference(series, base, ppy),
                "since_2023_11": study.sharpe_difference(
                    series[recent], base.reindex(series.index)[recent], ppy
                ),
                "correlation": round(float(series.corr(base)), 3),
            }
        # Sharpe đã khấu trừ cho việc thử nhiều ứng viên, tính theo đơn vị mỗi kỳ.
        per_period = {
            name: float(
                returns[(name, label)].mean() / returns[(name, label)].std(ddof=1)
            )
            for name in scores
        }
        best = max(per_period, key=per_period.get)
        series = returns[(best, label)].dropna()
        outcome = compute_deflated_sharpe_ratio(
            sr_estimated=per_period[best],
            variance_of_srs=float(np.var(list(per_period.values()), ddof=1)),
            sample_length=int(len(series)),
            num_trials=len(scores),
            skewness=float(series.skew()),
            kurtosis=float(series.kurt() + 3.0),
        )
        report["deflated_sharpe"][label] = {
            "best": best,
            "trials": len(scores),
            "dsr": round(outcome["dsr"], 3),
            "approved": outcome["is_approved"],
        }

    json.dump(report, open(args.out, "w"), indent=1)
    print("=== RESULT ===", flush=True)
    print(json.dumps(report), flush=True)
    print("ML RANKER STUDY DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
