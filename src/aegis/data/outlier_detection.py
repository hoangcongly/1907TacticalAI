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

        # [BUG FIX #4 - NAN PRICE GUARD]
        # Neu gia quan sat tai tick i hoac tick truoc (i-1) la NaN, tinh diff_price = NaN.
        # Tat ca phep so sanh NaN > x deu tra ve False trong NumPy, khien tick NaN
        # tham lam luot qua dieu kien 1 ma khong bi danh dau is_bad_tick.
        # He qua: tick NaN lot vao pipeline tin hieu nhu mot Good Tick.
        # Fix: Phan loai truc tiep NaN tick la bad_tick va bo qua cac dieu kien tiep theo.
        if np.isnan(prices[i]) or np.isnan(prices[i-1]):
            is_bad_tick[i] = True
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

@njit
def detect_bad_tick_cross_venue(
    timestamps: np.ndarray,
    ref_timestamps: np.ndarray,
    ref_prices: np.ndarray,
    robust_sigmas_ref: np.ndarray,
    window_ms: int = 500,
    eta_confirm: float = 2.0
) -> np.ndarray:
    """
    [TASK A-1-3] Điều kiện 4: Kiểm tra Cross-Venue Parity.
    Trả về mảng boolean (True = xác nhận là Bad Tick, False = KHÔNG phải Bad Tick).
    Khi thiếu feed sàn phụ -> fallback False (KHÔNG lọc, ưu tiên giữ Tail Event).
    """
    n = len(timestamps)
    m = len(ref_timestamps)
    
    # Mặc định False: Khi mất feed hoặc thiếu dữ liệu sàn phụ, ta Fallback về việc
    # KHÔNG coi nó là Bad Tick (để bảo toàn Tail Event thật, thiên về không lọc mất data).
    condition4_satisfied = np.zeros(n, dtype=np.bool_)

    lo_ptr = 0
    hi_ptr = 0
    anchor_ptr = 0

    for i in range(n):
        t_i = timestamps[i]
        t_lo = t_i - window_ms
        t_hi = t_i + window_ms

        while lo_ptr < m and ref_timestamps[lo_ptr] < t_lo:
            lo_ptr += 1
        while hi_ptr < m and ref_timestamps[hi_ptr] <= t_hi:
            hi_ptr += 1
        while anchor_ptr < m and ref_timestamps[anchor_ptr] <= t_i:
            anchor_ptr += 1
        anchor_idx = anchor_ptr - 1

        if lo_ptr >= hi_ptr or anchor_idx < 0:
            continue  # không đủ dữ liệu sàn phụ -> giữ mặc định False

        sigma_ref = robust_sigmas_ref[anchor_idx]
        if sigma_ref <= 0.0 or np.isnan(sigma_ref):
            continue  # sigma suy biến -> không đủ tin cậy để bác bỏ -> giữ mặc định False

        ref_anchor_price = ref_prices[anchor_idx]  # baseline của CHÍNH sàn phụ
        max_move_ref = 0.0
        for k in range(lo_ptr, hi_ptr):
            move = np.abs(ref_prices[k] - ref_anchor_price)
            if move > max_move_ref:
                max_move_ref = move

        # True (Bad Tick) nếu độ biến động sàn phụ NHỎ HƠN ngưỡng (tức là sàn phụ đứng yên).
        # False (Not Bad Tick) nếu sàn phụ CŨNG biến động mạnh (xác nhận Tail Event).
        condition4_satisfied[i] = max_move_ref < eta_confirm * sigma_ref

    return condition4_satisfied

