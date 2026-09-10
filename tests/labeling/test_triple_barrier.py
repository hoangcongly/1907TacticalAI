"""Unit tests for Dynamic HMM Triple-Barrier Labeling (Module C.2)."""
import numpy as np
import pandas as pd
import pytest
from aegis.labeling.triple_barrier import (
    compute_dynamic_barriers,
    apply_triple_barrier_single_event,
    generate_meta_labels_triple_barrier,
)


def test_compute_dynamic_barriers_symmetry():
    # Long event
    tp_long, sl_long = compute_dynamic_barriers(
        entry_price=100.0,
        side=1,
        p_trend=0.8,
        p_chop=0.2,
        sigma=0.01,
        is_toxic=False,
        c_trade=0.0004,
    )
    assert tp_long > 100.0
    assert sl_long < 100.0

    # Short event
    tp_short, sl_short = compute_dynamic_barriers(
        entry_price=100.0,
        side=-1,
        p_trend=0.8,
        p_chop=0.2,
        sigma=0.01,
        is_toxic=False,
        c_trade=0.0004,
    )
    assert tp_short < 100.0
    assert sl_short > 100.0

    # Distance to TP should match
    dist_tp_long = tp_long - 100.0
    dist_tp_short = 100.0 - tp_short
    assert pytest.approx(dist_tp_long, 1e-6) == dist_tp_short


def test_triple_barrier_take_profit_hit():
    # Long trade hitting TP
    entry_price = 100.0
    side = 1
    tp = 103.0
    sl = 98.0
    future_highs = np.array([101.0, 102.0, 103.5, 101.0])
    future_lows = np.array([99.5, 100.5, 101.5, 99.0])
    future_closes = np.array([100.5, 101.5, 103.0, 100.0])

    res = apply_triple_barrier_single_event(
        entry_idx=10,
        entry_price=entry_price,
        side=side,
        tp_price=tp,
        sl_price=sl,
        future_highs=future_highs,
        future_lows=future_lows,
        future_closes=future_closes,
    )
    assert res["hit_barrier"] == "TP"
    assert res["label"] == 1
    assert res["exit_idx"] == 13  # 10 + 1 + 2 = 13 (offset 2 hit TP)
    assert res["realized_return"] > 0.0


def test_triple_barrier_stop_loss_hit():
    # Long trade hitting SL
    entry_price = 100.0
    side = 1
    tp = 105.0
    sl = 98.0
    future_highs = np.array([101.0, 100.0])
    future_lows = np.array([99.0, 97.5])
    future_closes = np.array([99.5, 98.0])

    res = apply_triple_barrier_single_event(
        entry_idx=5,
        entry_price=entry_price,
        side=side,
        tp_price=tp,
        sl_price=sl,
        future_highs=future_highs,
        future_lows=future_lows,
        future_closes=future_closes,
    )
    assert res["hit_barrier"] == "SL"
    assert res["label"] == 0
    assert res["exit_idx"] == 7  # 5 + 1 + 1 = 7
    assert res["realized_return"] < 0.0


def test_triple_barrier_vertical_time_stop():
    # Trade reaches end of horizon without touching TP or SL
    entry_price = 100.0
    side = 1
    tp = 110.0
    sl = 90.0
    future_highs = np.array([101.0, 102.0])
    future_lows = np.array([99.5, 100.0])
    future_closes = np.array([100.5, 101.5])

    res = apply_triple_barrier_single_event(
        entry_idx=20,
        entry_price=entry_price,
        side=side,
        tp_price=tp,
        sl_price=sl,
        future_highs=future_highs,
        future_lows=future_lows,
        future_closes=future_closes,
    )
    assert res["hit_barrier"] == "TIME"
    assert res["exit_idx"] == 22
    # Closed at 101.5 > 100.0 -> positive return -> label = 1
    assert res["label"] == 1
    assert pytest.approx(res["realized_return"], 1e-4) == 0.015


def test_generate_meta_labels_triple_barrier_full():
    n = 100
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    highs = prices + 0.5
    lows = prices - 0.5
    closes = prices
    
    events_idx = np.array([10, 30, 50, 70])
    sides = np.array([1, -1, 1, -1])
    p_trend = np.full(n, 0.7)
    p_chop = np.full(n, 0.3)
    sigmas = np.full(n, 0.01)
    is_toxic = np.zeros(n, dtype=bool)

    labels_df = generate_meta_labels_triple_barrier(
        events_idx=events_idx,
        sides=sides,
        closes=closes,
        highs=highs,
        lows=lows,
        p_trend=p_trend,
        p_chop=p_chop,
        sigmas=sigmas,
        is_toxic=is_toxic,
        t_window=10,
    )
    assert len(labels_df) == 4
    assert set(labels_df["label"].unique()).issubset({0, 1})
    assert np.all(labels_df["t1"] > labels_df["t0"])
