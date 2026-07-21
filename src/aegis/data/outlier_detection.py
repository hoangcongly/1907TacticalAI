"""
[TASK A-1-1] Nền Tảng Toán Học Lọc Nhiễu Tick (MAD 5σ & Cross-Venue Parity).
Tính toán Median Absolute Deviation trên cửa sổ trượt 100-tick và quy đổi thành Robust Sigma.
"""

import numpy as np
from numba import njit


@njit
def compute_rolling_mad(prices: np.ndarray, window: int = 100) -> np.ndarray:
    """
    [TASK A-1-1] Tính Robust Sigma dựa trên Median Absolute Deviation (MAD) cửa sổ trượt.
    
    Toán học (AFML & Aegis Master Blueprint):
        w = P[i-window : i]  (không bao gồm P[i] để tránh look-ahead bias)
        MAD_i = median(|w - median(w)|)
        sigma_MAD_i = 1.4826 * MAD_i
        
    Tham số:
        prices (np.ndarray): Mảng giá tick 1D.
        window (int): Kích thước cửa sổ trượt (Mặc định 100 tick).
        
    Trả về:
        np.ndarray: Mảng Robust Sigma có cùng độ dài với mảng giá. 
                    (Các phần tử khởi động < window được gán bằng NaN).
    """
    n = len(prices)
    robust_sigmas = np.full(n, np.nan, dtype=np.float64)

    # Nếu dữ liệu quá ngắn, trả về toàn NaN
    if n <= window:
        return robust_sigmas

    for i in range(window, n):
        # 1. Trích xuất cửa sổ nghiêm ngặt không nhìn trước tương lai (Causal Window)
        w = prices[i - window : i]
        
        # 2. Tính trung vị của cửa sổ
        m = np.median(w)
        
        # 3. Tính độ lệch tuyệt đối
        d = np.abs(w - m)
        
        # 4. Tính MAD
        mad_i = np.median(d)
        
        # 5. Quy đổi thành Robust Standard Deviation
        robust_sigmas[i] = 1.4826 * mad_i

    return robust_sigmas
