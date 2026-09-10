import pytest
import numpy as np
import pandas as pd
from aegis.labeling.sample_weights import (
    compute_num_concurrent_events,
    compute_average_uniqueness,
    compute_sample_weights,
)


def test_average_uniqueness_hand_calculated():
    """
    B-4-5 core: Khớp tính tay trên tập nhãn chồng lấp.
    Lệnh 1: 0->2, Lệnh 2: 1->3, Lệnh 3: 2->3.
    """
    t1 = pd.Series({0: 2, 1: 3, 2: 3})
    c_t = compute_num_concurrent_events(t1)

    # c_t tính tay:
    # Bar 0: Lệnh 1 => c_0 = 1
    # Bar 1: Lệnh 1, 2 => c_1 = 2
    # Bar 2: Lệnh 1, 2, 3 => c_2 = 3
    # Bar 3: Lệnh 2, 3 => c_3 = 2
    assert c_t.loc[0] == 1
    assert c_t.loc[1] == 2
    assert c_t.loc[2] == 3
    assert c_t.loc[3] == 2

    u_bar = compute_average_uniqueness(t1, c_t)

    # u_bar tính tay:
    # Lệnh 1 (0->2): (1/1 + 1/2 + 1/3) / 3 = 0.611111
    # Lệnh 2 (1->3): (1/2 + 1/3 + 1/2) / 3 = 0.444444
    # Lệnh 3 (2->3): (1/3 + 1/2) / 2 = 0.416666
    np.testing.assert_almost_equal(u_bar.loc[0], 0.6111111)
    np.testing.assert_almost_equal(u_bar.loc[1], 0.4444444)
    np.testing.assert_almost_equal(u_bar.loc[2], 0.4166666)


def test_sample_weights_all_zero_returns():
    """
    FIX #1: Khi mọi return = 0, hàm KHÔNG được crash (ZeroDivisionError).
    Phải trả trọng số đều.
    """
    t1 = pd.Series({0: 2, 1: 3})
    c_t = compute_num_concurrent_events(t1)
    returns = pd.Series({0: 0.0, 1: 0.0})
    w = compute_sample_weights(t1, c_t, returns)
    assert not w.isna().any(), f"Trọng số chứa NaN: {w}"
    assert not np.isinf(w).any(), f"Trọng số chứa Inf: {w}"
    np.testing.assert_almost_equal(w.loc[0], 1.0)
    np.testing.assert_almost_equal(w.loc[1], 1.0)


def test_concurrent_events_performance():
    """
    FIX #2: 10,000 events phải chạy dưới 2 giây (sweep-line).
    """
    import time
    t1 = pd.Series(index=range(10000), data=range(5, 10005))
    start = time.time()
    c_t = compute_num_concurrent_events(t1)
    elapsed = time.time() - start
    assert elapsed < 2.0, f"compute_num_concurrent_events mất {elapsed:.2f}s, quá chậm!"
    # Verify correctness: mỗi event kéo 5 bars, nên c_t peak ~ 5
    assert c_t.max() <= 6  # Cho phép sai số nhỏ do biên


def test_bar_index_must_be_sorted():
    """FIX #2 guard: bar_index không sorted phải raise."""
    t1 = pd.Series({0: 2, 1: 3})
    unsorted_index = pd.Index([3, 1, 0, 2])
    with pytest.raises(ValueError, match="monotonic increasing"):
        compute_num_concurrent_events(t1, bar_index=unsorted_index)
