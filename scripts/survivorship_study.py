#!/usr/bin/env python3
"""Thiên vị sống sót trong backtest v3 đáng bao nhiêu Sharpe?

Chạy sau `scripts/download_vision_history.py`. So bốn cách dựng universe trên CÙNG dữ
liệu, cùng tín hiệu, cùng mốc dữ liệu:

    repo            `artifacts/universe_wide.json` — top khối lượng của NGÀY TẢI, cặp còn sống
    repo_terminal   như trên, nhưng tính lợi suất kỳ cuối khi giá biến mất (phép thử: phải y hệt)
    pit_survivors   lọc thanh khoản theo từng thời điểm, chỉ trong các cặp còn sống ở cuối panel
    pit_all         lọc thanh khoản theo từng thời điểm, trên MỌI cặp từng niêm yết

`pit_survivors - repo`      = phần do chọn universe bằng thanh khoản của tương lai.
`pit_all - pit_survivors`   = phần do bỏ sót cặp đã huỷ niêm yết.

    python scripts/survivorship_study.py
"""

import argparse
import json
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, "src")

from aegis.research import survivorship as study  # noqa: E402
from aegis.research.strategy_v3 import (
    V3,
    V3_VINTAGE_MS,
    combined_signal,
    load_v3_data,
    run_v3,
)  # noqa: E402

PORTFOLIOS = {"n12": {}, "n50_equal": {"n_positions": 50, "max_weight": 0.02}}
ERAS = {
    "all": "2000-01-01",
    "since_2023_11": "2023-11-01",
    "since_2025_02": "2025-02-01",
}


def era_stats(returns: pd.Series, ppy: float, split_ts: int) -> dict:
    when = pd.to_datetime(returns.index, unit="ms")
    out = {
        name: study.performance(returns[when >= pd.Timestamp(start)], ppy)
        for name, start in ERAS.items()
    }
    out["holdout"] = study.performance(returns[returns.index >= split_ts], ppy)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-universe", default="artifacts/universe_all_listed.json")
    parser.add_argument("--min-daily-usd", type=float, default=20_000_000.0)
    parser.add_argument("--end-ms", type=int, default=V3_VINTAGE_MS)
    parser.add_argument("--out", default="artifacts/survivorship_study.json")
    args = parser.parse_args(argv)
    ppy = V3.periods_per_year
    report = {
        "end_ms": args.end_ms,
        "min_daily_usd": args.min_daily_usd,
        "runs": {},
        "differences": {},
    }

    # 1) Universe của repo, đúng cấu hình đã kiểm định.
    base = load_v3_data(V3, end_ms=args.end_ms)
    base_signal = combined_signal(base, V3)
    everything = pd.DataFrame(True, index=base.close.index, columns=base.close.columns)
    returns = {}
    for label, change in PORTFOLIOS.items():
        spec = replace(V3.portfolio, **change)
        returns[("repo", label)] = run_v3(
            base, V3, portfolio=spec, sig=base_signal
        ).returns
        returns[("repo_terminal", label)] = study.run_point_in_time(
            base, V3, everything, portfolio=spec
        ).returns
    report["repo_symbols"] = int(base.close.shape[1])
    del base, base_signal

    # 2) Mọi cặp từng niêm yết; `min_coverage=0` vì lọc theo độ phủ TOÀN lịch sử là nhìn trước.
    wide_cfg = replace(V3, universe_file=args.all_universe, min_coverage=0.0)
    wide = load_v3_data(wide_cfg, end_ms=args.end_ms)
    close, volume = wide.close, wide.panel["volume"]
    eligible = study.point_in_time_eligibility(
        close, volume, min_daily_usd=args.min_daily_usd
    )
    alive = study.alive_at_end(close)
    survivors_only = eligible & alive.reindex(close.columns).to_numpy()[None, :]
    marks = close.index[:: V3.rebalance_every]
    years = pd.to_datetime(marks, unit="ms").year
    report["all_symbols"] = int(close.shape[1])
    report["delisted_symbols"] = int((~alive).sum())
    report["eligible_per_period_by_year"] = {
        name: {
            int(year): int(frame.reindex(marks).sum(axis=1)[years == year].median())
            for year in sorted(set(years))
        }
        for name, frame in (("pit_all", eligible), ("pit_survivors", survivors_only))
    }
    report["delisting"] = {}
    for label, change in PORTFOLIOS.items():
        spec = replace(V3.portfolio, **change)
        for name, mask in (("pit_survivors", survivors_only), ("pit_all", eligible)):
            result = study.run_point_in_time(wide, wide_cfg, mask, portfolio=spec)
            returns[(name, label)] = result.returns
            if name == "pit_all":
                exits = study.terminal_close(close, V3.rebalance_every)
                report["delisting"][label] = study.delisting_exposure(
                    result.weights, close, exits, ~alive
                )
        print(f"  {label} xong", flush=True)

    for (name, label), series in returns.items():
        report["runs"].setdefault(label, {})[name] = era_stats(
            series, ppy, wide.split_ts
        )
    for label in PORTFOLIOS:
        pairs = {
            "terminal_fix_on_repo": ("repo_terminal", "repo"),
            "future_liquidity_selection": ("pit_survivors", "repo"),
            "missing_delisted_symbols": ("pit_all", "pit_survivors"),
            "total": ("pit_all", "repo"),
        }
        report["differences"][label] = {}
        for key, (a, b) in pairs.items():
            first, second = returns[(a, label)], returns[(b, label)]
            recent = first.index >= int(
                pd.Timestamp(ERAS["since_2023_11"]).timestamp() * 1000
            )
            report["differences"][label][key] = {
                "all": study.sharpe_difference(first, second, ppy),
                "since_2023_11": study.sharpe_difference(
                    first[recent], second.reindex(first.index)[recent], ppy
                ),
            }

    json.dump(report, open(args.out, "w"), indent=1)
    print("=== RESULT ===", flush=True)
    print(json.dumps(report), flush=True)
    print("SURVIVORSHIP STUDY DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
