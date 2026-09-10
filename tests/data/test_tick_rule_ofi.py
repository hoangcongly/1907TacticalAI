"""Unit tests for Tick Rule Order Flow Imbalance (OFI)."""
import numpy as np
import pytest
from aegis.data.bars.tick_rule_ofi import (
    classify_tick_rule,
    compute_tick_rule_ofi,
    compute_rolling_ofi,
)


def test_classify_tick_rule_basic():
    prices = np.array([100.0, 101.0, 101.0, 100.5, 100.5, 102.0])
    signs = classify_tick_rule(prices)
    assert len(signs) == 6
    assert signs[0] == 1.0
    assert signs[1] == 1.0
    assert signs[2] == 1.0  # Zero tick maintains previous sign (+1)
    assert signs[3] == -1.0
    assert signs[4] == -1.0  # Zero tick maintains previous sign (-1)
    assert signs[5] == 1.0


def test_compute_tick_rule_ofi_bounds():
    prices_up = np.array([100.0, 101.0, 102.0, 103.0])
    vols = np.array([1.0, 2.0, 3.0, 4.0])
    ofi_up = compute_tick_rule_ofi(prices_up, vols)
    assert pytest.approx(ofi_up, 1e-4) == 1.0

    prices_down = np.array([103.0, 102.0, 101.0, 100.0])
    ofi_down = compute_tick_rule_ofi(prices_down, vols)
    assert pytest.approx(ofi_down, 1e-4) == -1.0

    prices_mixed = np.array([100.0, 101.0, 100.0])
    vols_balanced = np.array([10.0, 10.0, 20.0])
    ofi_balanced = compute_tick_rule_ofi(prices_mixed, vols_balanced)
    assert pytest.approx(ofi_balanced, 1e-4) == 0.0


def test_compute_tick_rule_ofi_empty_or_zero():
    assert compute_tick_rule_ofi(np.array([]), np.array([])) == 0.0
    assert compute_tick_rule_ofi(np.array([100.0]), np.array([0.0])) == 0.0


def test_compute_rolling_ofi():
    prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0, 101.0])
    vols = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rolling_ofi = compute_rolling_ofi(prices, vols, window=3)
    assert len(rolling_ofi) == 6
    assert np.isnan(rolling_ofi[0])
    assert np.isnan(rolling_ofi[1])
    assert not np.isnan(rolling_ofi[2])
    assert np.all(rolling_ofi[2:] >= -1.0) and np.all(rolling_ofi[2:] <= 1.0)
