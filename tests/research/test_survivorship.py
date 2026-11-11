"""Thiên vị sống sót: kỳ huỷ niêm yết phải được tính, và universe không được nhìn trước."""

import numpy as np
import pandas as pd
import pytest

from aegis.research import survivorship as study
from aegis.research.backtest_v2 import CostModel, simulate

FREE = CostModel(
    maker_ratio=1.0,
    maker_fee_bps=0.0,
    taker_fee_bps=0.0,
    half_spread_bps=0.0,
    min_bps=0.0,
)


def _prices():
    index = pd.RangeIndex(0, 6)
    alive = pd.Series([100.0, 100, 100, 100, 100, 100], index=index)
    dead = pd.Series([100.0, 100, 50, np.nan, np.nan, np.nan], index=index)
    return pd.DataFrame({"ALIVE": alive, "DEAD": dead})


def test_terminal_close_extends_the_last_price_by_one_period_only():
    filled = study.terminal_close(_prices(), max_gap_bars=2)
    assert filled["DEAD"].tolist()[:5] == [100.0, 100.0, 50.0, 50.0, 50.0]
    assert np.isnan(filled["DEAD"].iloc[5])


def test_plain_simulate_hides_the_delisting_loss_and_the_fix_shows_it():
    close = _prices()
    marks = close.index[::2]  # tái cân bằng ở 0, 2, 4; DEAD mất giá từ 3
    weights = pd.DataFrame(
        {"ALIVE": [-0.5, -0.5, 0.0], "DEAD": [0.5, 0.5, 0.0]}, index=marks
    )
    weights.loc[2, "DEAD"] = 0.5
    crash = close.copy()
    crash.loc[3, "DEAD"] = 10.0  # giá khớp cuối, giữa hai mốc
    hidden = simulate(
        weights, crash.reindex(marks), cost=FREE, bar_hours=8, rebalance_every=1
    )
    shown = simulate(
        weights,
        study.terminal_close(crash, 2).reindex(marks),
        cost=FREE,
        bar_hours=8,
        rebalance_every=1,
    )
    assert hidden.gross_returns.loc[2] == pytest.approx(0.0)
    assert shown.gross_returns.loc[2] == pytest.approx(0.5 * (10.0 / 50.0 - 1.0))


def _panel(bars=400, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.RangeIndex(0, bars)
    close = pd.DataFrame(
        {
            name: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, bars)))
            for name in ("BIG", "SMALL", "DEAD")
        },
        index=index,
    )
    volume = pd.DataFrame({"BIG": 1e5, "SMALL": 1e2, "DEAD": 1e5}, index=index)
    close.loc[300:, "DEAD"] = np.nan
    volume.loc[300:, "DEAD"] = np.nan
    return close, volume


def test_eligibility_needs_liquidity_history_and_a_live_price():
    close, volume = _panel()
    eligible = study.point_in_time_eligibility(
        close, volume, min_daily_usd=1e6, window_bars=60, min_history_bars=100
    )
    assert not eligible["SMALL"].any()
    assert not eligible["BIG"].iloc[:99].any() and eligible["BIG"].iloc[150:].all()
    assert (
        eligible["DEAD"].iloc[150:300].all() and not eligible["DEAD"].iloc[300:].any()
    )


def test_eligibility_does_not_look_ahead():
    close, volume = _panel()
    early = study.point_in_time_eligibility(
        close, volume, min_daily_usd=1e6, window_bars=60, min_history_bars=100
    )
    changed = volume.copy()
    changed.iloc[250:] = 0.0
    late = study.point_in_time_eligibility(
        close, changed, min_daily_usd=1e6, window_bars=60, min_history_bars=100
    )
    assert early.iloc[:250].equals(late.iloc[:250])


def test_masked_signals_are_missing_where_ineligible():
    close, volume = _panel(bars=10)
    eligible = pd.DataFrame(False, index=close.index, columns=close.columns)
    eligible.loc[5:, "BIG"] = True
    masked = study.mask_signals({"s": close}, eligible)["s"]
    assert masked["BIG"].iloc[:5].isna().all() and masked["BIG"].iloc[5:].notna().all()
    assert masked["SMALL"].isna().all()


def test_alive_at_end_flags_delisted_symbols():
    close, _ = _panel()
    alive = study.alive_at_end(close)
    assert alive["BIG"] and not alive["DEAD"]


def test_performance_reports_drawdown_and_worst_period():
    returns = pd.Series([0.01, -0.02] * 20)
    stats = study.performance(returns, periods_per_year=100)
    assert (
        stats["periods"] == 40
        and stats["worst_period"] == -0.02
        and stats["max_drawdown"] < 0
    )


def test_sharpe_difference_is_zero_for_identical_series_and_signed_otherwise():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.001, 0.01, 300))
    same = study.sharpe_difference(base, base, 100)
    assert same["difference"] == 0 and same["ci95"] == [0, 0]
    better = study.sharpe_difference(base + 0.004, base, 100)
    assert better["difference"] > 0 and better["ci95"][0] > 0


def test_delisting_exposure_counts_only_held_positions():
    close = _prices()
    crash = close.copy()
    crash.loc[3, "DEAD"] = 10.0
    marks = close.index[::2]
    weights = pd.DataFrame(
        {"ALIVE": [-0.5, -0.5, 0.0], "DEAD": [0.0, 0.5, 0.0]}, index=marks
    )
    report = study.delisting_exposure(
        weights,
        crash.reindex(marks),
        study.terminal_close(crash, 2).reindex(marks),
        pd.Series({"ALIVE": False, "DEAD": True}),
    )
    assert (
        report["positions_held_into_a_missing_price"] == 1
        and report["longs"] == 1
        and report["shorts"] == 0
    )
    assert report["return_ignored_by_plain_simulate"] == pytest.approx(-0.4)
    assert report["total_return_from_delisted_symbols"] == pytest.approx(-0.4)
