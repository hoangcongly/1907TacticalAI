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


@njit
def detect_bad_tick_core(prices: np.ndarray, volumes: np.ndarray, robust_sigmas: np.ndarray, window: int = 100):
    """
    [TASK A-1-2] Phân loại Bad Tick và Tail Event dựa trên 3 điều kiện lõi.
    
    Điều kiện:
      1. Lệch cực đoan: |P_i - P_{i-1}| > 5 * sigma_i
      2. Khối lượng tĩnh: V_i < 2 * median(V_{i-window : i-1})
      3. Đảo chiều chớp nhoáng: |P_{i+1} - P_{i-1}| < 0.3 * |P_i - P_{i-1}|
      
    Tham số:
        prices: Mảng 1D giá khớp lệnh.
        volumes: Mảng 1D khối lượng.
        robust_sigmas: Mảng sigma kháng nhiễu (đã tính từ compute_rolling_mad).
        window: Cửa sổ trượt (Mặc định 100).
        
    Trả về:
        (is_bad_tick, is_tail_event): Tuple chứa 2 mảng boolean phân loại tick.
    """
    n = len(prices)
    is_bad_tick = np.zeros(n, dtype=np.bool_)
    is_tail_event = np.zeros(n, dtype=np.bool_)
    
    # Không đủ dữ liệu để tính toán
    if n <= window:
        return is_bad_tick, is_tail_event
        
    # Phải dừng ở n-2 vì ĐK3 yêu cầu P_{i+1}
    for i in range(window, n - 1):
        sigma = robust_sigmas[i]
        if np.isnan(sigma) or sigma == 0:
            continue
            
        diff_price = np.abs(prices[i] - prices[i-1])
        
        # ĐK1: Extreme Deviation Check
        cond_1 = diff_price > 5.0 * sigma
        
        if cond_1:
            # ĐK2: Volume Consistency Check
            v_window = volumes[i - window : i]
            median_vol = np.median(v_window)
            cond_2 = volumes[i] < 2.0 * median_vol
            
            # ĐK3: Micro-Reversal Check
            reversal_diff = np.abs(prices[i+1] - prices[i-1])
            cond_3 = reversal_diff < 0.3 * diff_price
            
            if cond_2 and cond_3:
                # Khớp cả 3 ĐK -> Đích thị là Bad Tick (Nhiễu)
                is_bad_tick[i] = True
            elif not cond_2:
                # Bị vi phạm ĐK2 (Volume rất lớn) -> Dòng tiền thực, là sự kiện đuôi đen
                is_tail_event[i] = True

    return is_bad_tick, is_tail_event
