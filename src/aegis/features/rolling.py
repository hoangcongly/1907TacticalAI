"""
[TASK A-3-2] Rolling Indicator Utilities
Chứa các hàm hỗ trợ cho việc tính toán rolling indicator.
Đặc biệt là xử lý flag insufficient_history cho các nến sau gap.
"""

import numpy as np
from numba import njit


@njit(nopython=True)
def get_insufficient_history_mask(
    timestamps_ms: np.ndarray, 
    gap_threshold_ms: float, 
    window_size: int
) -> np.ndarray:
    """
    [TASK A-3-2] Tạo mask insufficient_history cho rolling indicator.
    
    Logic:
    - window_size nến đầu tiên luôn là True (Warm-up).
    - Nếu khoảng cách thời gian giữa 2 nến liên tiếp > gap_threshold_ms,
      đó là một gap. window_size nến tiếp theo (bắt đầu từ nến ngay sau gap)
      sẽ được đánh dấu là True (thiếu lịch sử cho rolling).
      
    Args:
        timestamps_ms: Mảng 1D chứa timestamp (milliseconds).
        gap_threshold_ms: Ngưỡng thời gian để xác định đứt gãy (ví dụ: mất kết nối, qua đêm).
        window_size: Kích thước cửa sổ (W) của rolling indicator.
        
    Returns:
        Mảng boolean 1D, cùng kích thước với timestamps_ms, True tại các vị trí insufficient_history.
    """
    n = len(timestamps_ms)
    mask = np.zeros(n, dtype=np.bool_)
    
    if n == 0:
        return mask
        
    # 1. Warm-up ở điểm bắt đầu
    for i in range(min(window_size, n)):
        mask[i] = True
        
    # 2. Xử lý sau khi có Gap
    for i in range(1, n):
        if timestamps_ms[i] - timestamps_ms[i-1] > gap_threshold_ms:
            # Tìm thấy gap. Đánh dấu window_size nến tiếp theo (từ vị trí i)
            limit = min(i + window_size, n)
            for j in range(i, limit):
                mask[j] = True
                
    return mask
