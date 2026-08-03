"""
Tests cho rolling utilities, đặc biệt là insufficient_history mask.
"""

import numpy as np
from aegis.features.rolling import get_insufficient_history_mask

def test_insufficient_history_mask_basic_warmup():
    """
    Kiểm tra flag đúng W bar ở giai đoạn khởi động (không có gap).
    """
    timestamps = np.array([100, 200, 300, 400, 500, 600, 700], dtype=np.float64)
    gap_threshold = 150.0
    window_size = 3
    
    mask = get_insufficient_history_mask(timestamps, gap_threshold, window_size)
    
    # W=3 nên 3 bar đầu là True, phần còn lại là False
    expected = np.array([True, True, True, False, False, False, False])
    np.testing.assert_array_equal(mask, expected)

def test_insufficient_history_mask_after_gap():
    """
    [TASK A-3-2] Test: flag đúng W bar sau gap giả lập.
    """
    # 100, 200, 300: bình thường
    # 1000: nhảy vọt (gap = 1000 - 300 = 700 > 150)
    # 1100, 1200, 1300, 1400: bình thường trở lại
    timestamps = np.array([100, 200, 300, 1000, 1100, 1200, 1300, 1400], dtype=np.float64)
    gap_threshold = 150.0
    window_size = 3
    
    mask = get_insufficient_history_mask(timestamps, gap_threshold, window_size)
    
    # 3 bar đầu warmup: True (idx 0, 1, 2)
    # idx 3 là bar ngay sau gap -> warmup lại 3 bar (idx 3, 4, 5)
    # idx 6, 7 an toàn: False
    expected = np.array([True, True, True, True, True, True, False, False])
    np.testing.assert_array_equal(mask, expected)

def test_insufficient_history_mask_overlapping_gaps():
    """
    Kiểm tra nếu gap xảy ra liên tục (chưa kịp hết warmup đã có gap mới).
    """
    # 100
    # 1000 (gap)
    # 2000 (gap)
    # 2100, 2200, 2300, 2400
    timestamps = np.array([100, 1000, 2000, 2100, 2200, 2300, 2400], dtype=np.float64)
    gap_threshold = 150.0
    window_size = 3
    
    mask = get_insufficient_history_mask(timestamps, gap_threshold, window_size)
    
    # idx 0: warmup đầu tiên (True)
    # idx 1: gap -> idx 1, 2, 3 đáng ra là True.
    # idx 2: lại gap -> idx 2, 3, 4 lại True.
    # Nên: 0(T), 1(T), 2(T), 3(T), 4(T), 5(F), 6(F)
    expected = np.array([True, True, True, True, True, False, False])
    np.testing.assert_array_equal(mask, expected)
