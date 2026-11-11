"""Đo thiên vị sống sót: universe theo từng thời điểm và lợi suất của kỳ huỷ niêm yết.

Hai lỗi đo lường mà một universe "các cặp đang giao dịch hôm nay" che mất:

1. Cặp đã huỷ niêm yết không có mặt trong backtest.
2. `simulate` coi lợi suất của một vị thế là 0 khi giá ở mốc kế tiếp bị thiếu. Với cặp
   còn sống điều đó hiếm khi xảy ra; với cặp bị huỷ giữa kỳ, đó chính là cú giảm cuối.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd


def terminal_close(close: pd.DataFrame, max_gap_bars: int) -> pd.DataFrame:
    """Kéo giá cuối cùng thêm tối đa một kỳ, để vị thế đang giữ thoát ở giá khớp cuối."""
    return close.ffill(limit=max_gap_bars)


def point_in_time_eligibility(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_daily_usd: float = 20_000_000.0,
    window_bars: int = 180,
    min_history_bars: int = 200,
    bars_per_day: int = 6,
) -> pd.DataFrame:
    """Cặp đủ điều kiện tại t, chỉ dùng dữ liệu <= t.

    Cùng ngưỡng thanh khoản với `download_wide_universe.py` ($20M/ngày), nhưng đo trên
    cửa sổ trượt TRƯỚC mỗi mốc thay vì trên khối lượng 24h của ngày tải dữ liệu.
    """
    dollar = close * volume
    daily = (
        dollar.rolling(window_bars, min_periods=window_bars // 2).mean() * bars_per_day
    )
    age = close.notna().cumsum()
    return (daily >= min_daily_usd) & (age >= min_history_bars) & close.notna()


def alive_at_end(close: pd.DataFrame, grace_bars: int = 42) -> pd.Series:
    """Cặp còn có giá trong `grace_bars` nến cuối của panel."""
    return close.iloc[-grace_bars:].notna().any()


def mask_signals(
    signals: Dict[str, pd.DataFrame], eligible: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    return {
        name: frame.where(
            eligible.reindex(index=frame.index, columns=frame.columns, fill_value=False)
        )
        for name, frame in signals.items()
    }


def run_point_in_time(
    data,
    cfg,
    eligible: pd.DataFrame,
    portfolio=None,
    maker_ratio: Optional[float] = None,
):
    """Như `run_v3`, nhưng tín hiệu bị che theo `eligible` và kỳ cuối của mỗi cặp được tính."""
    from aegis.research.backtest_v2 import CostModel, simulate
    from aegis.research.strategy_v3 import combined_signal
    from aegis.risk.portfolio import build_weights

    maker_ratio = cfg.maker_ratio if maker_ratio is None else maker_ratio
    masked = replace(data, signals=mask_signals(data.signals, eligible))
    marks = data.close.index[:: cfg.rebalance_every]
    signal = combined_signal(masked, cfg)
    weights = build_weights(
        signal, data.close.reindex(marks), portfolio or cfg.portfolio
    )
    cost = CostModel(
        maker_ratio=maker_ratio,
        half_spread_bps=0.0,
        per_symbol_bps=data.per_symbol_bps * (1 - maker_ratio),
        min_bps=0.5,
    )
    exit_prices = terminal_close(data.close, cfg.rebalance_every).reindex(marks)
    return simulate(
        weights,
        exit_prices,
        data.funding.reindex(marks),
        cost,
        bar_hours=cfg.bar_hours * cfg.rebalance_every,
        rebalance_every=1,
    )


def performance(returns: pd.Series, periods_per_year: float) -> dict:
    r = returns.dropna()
    if len(r) < 20 or r.std(ddof=1) < 1e-15:
        return {"periods": int(len(r)), "sharpe": None}
    equity = (1.0 + r).cumprod()
    return {
        "periods": int(len(r)),
        "sharpe": round(float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)), 3),
        "ann_return": round(float(r.mean() * periods_per_year), 4),
        "ann_vol": round(float(r.std(ddof=1) * np.sqrt(periods_per_year)), 4),
        "max_drawdown": round(float((equity / equity.cummax() - 1.0).min()), 4),
        "worst_period": round(float(r.min()), 4),
    }


def sharpe_difference(
    a: pd.Series,
    b: pd.Series,
    periods_per_year: float,
    block: int = 10,
    resamples: int = 2000,
    seed: int = 7,
) -> dict:
    """Sharpe(a) - Sharpe(b) trên cùng các kỳ, khoảng tin cậy bằng bootstrap khối ghép cặp."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    values = both.to_numpy()
    count = len(values)
    if count < 2 * block:
        return {"periods": count, "difference": None}

    def sharpe(column):
        deviation = column.std(ddof=1)
        return (
            column.mean() / deviation * np.sqrt(periods_per_year)
            if deviation > 0
            else np.nan
        )

    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(count / block))
    draws = []
    for _ in range(resamples):
        starts = rng.integers(0, count - block + 1, blocks)
        rows = np.concatenate([np.arange(s, s + block) for s in starts])[:count]
        draws.append(sharpe(values[rows, 0]) - sharpe(values[rows, 1]))
    return {
        "periods": count,
        "difference": round(float(sharpe(values[:, 0]) - sharpe(values[:, 1])), 3),
        "ci95": [
            round(float(np.nanpercentile(draws, 2.5)), 3),
            round(float(np.nanpercentile(draws, 97.5)), 3),
        ],
    }


def delisting_exposure(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    exit_prices: pd.DataFrame,
    dead: pd.Series,
) -> dict:
    """Vị thế đang giữ đúng lúc cặp biến mất, và lợi suất mà `simulate` cũ đã bỏ qua."""
    marks = weights.index
    plain = close.reindex(marks)
    filled = exit_prices.reindex(marks)
    forward = filled.shift(-1) / filled - 1.0
    lost = plain.shift(-1).isna() & filled.shift(-1).notna() & plain.notna()
    held = lost & (weights != 0.0)
    contribution = (weights * forward).where(held)
    dead_columns = [c for c in weights.columns if bool(dead.get(c, False))]
    on_dead = (weights[dead_columns] * forward[dead_columns]).where(
        weights[dead_columns] != 0.0
    )
    return {
        "positions_held_into_a_missing_price": int(held.sum().sum()),
        "longs": int((held & (weights > 0)).sum().sum()),
        "shorts": int((held & (weights < 0)).sum().sum()),
        "return_ignored_by_plain_simulate": round(
            float(np.nansum(contribution.to_numpy())), 4
        ),
        "total_return_from_delisted_symbols": round(
            float(np.nansum(on_dead.to_numpy())), 4
        ),
    }
