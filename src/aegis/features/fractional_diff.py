"""
Module A.3 — Windowed FFD O(W*) Mặc Định & Sum-of-Exponentials O(M) nếu approved.
[TASK A-4-1] Tính toán trọng số đệ quy nhị thức (Binomial FFD Weights).
"""

import numpy as np
from numba import njit
# [BUG FIX #8] statsmodels là thư viện rất nặng (~30MB). Import module-level gây delay
# 300-500ms cho mọi module nào import fractional_diff (kể cả qua __init__.py).
# Chuyển sang lazy import bên trong find_optimal_d_star để chỉ tải khi cần.

from aegis.core.experiment_tracker import ExperimentTracker
from aegis.core.trial_classes import TrialClass

@njit(nopython=True)
def compute_ffd_weights(d: float, tau: float = 1e-5) -> np.ndarray:
    """
    [TASK A-4-1] Tính toán trọng số vi phân từng phần (Fractional Differentiation Weights)
    bằng công thức đệ quy nhị thức. Dừng khi |w_k| < tau.
    
    Công thức gốc (AFML):
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k
    
    Args:
        d (float): Bậc vi phân (Fractional differentiation order).
        tau (float): Ngưỡng cắt đuôi (Truncation threshold). Trọng số nhỏ hơn |tau| sẽ bị loại.
                     Theo chuẩn production, mặc định là 1e-5.
                     
    Returns:
        np.ndarray: Mảng 1D chứa các trọng số w_0, w_1, ..., w_W.
                    (Với W là cửa sổ đạt tới ngưỡng tau).
    """
    if d < 0 or d > 1:
        # Trong FFD tài chính thường d thuộc [0, 1]
        pass # Vẫn hỗ trợ tính toán bình thường theo công thức toán học
        
    w = [1.0] # w_0 = 1
    k = 1
    
    while True:
        # w_k = -w_{k-1} * (d - k + 1) / k
        w_k = -w[-1] * (d - k + 1) / k
        
        if abs(w_k) < tau:
            break
            
        w.append(w_k)
        k += 1
        
    # Return as numpy array, reverse it if needed by rolling window later,
    # but strictly AFML returns w_0, w_1, w_2... 
    # For dot product with past prices P_t, P_{t-1}... we keep this order.
    return np.array(w, dtype=np.float64)


def apply_ffd(series: np.ndarray, d: float, tau: float = 1e-5) -> np.ndarray:
    """
    [TASK A-4-2] Áp dụng Fractional Differentiation lên chuỗi thời gian sử dụng cửa sổ (Windowed FFD).
    Tính toán các trọng số và áp dụng qua phép chập (convolution) để đảm bảo tính causal.
    
    Args:
        series (np.ndarray): Mảng 1D chứa giá trị chuỗi thời gian đầu vào.
        d (float): Bậc vi phân.
        tau (float): Ngưỡng cắt đuôi.
        
    Returns:
        np.ndarray: Mảng 1D sau khi đã được vi phân phân số. Có độ dài bằng `len(series) - len(w) + 1` 
                    để đảm bảo strictly causal (không tính các giá trị bị khuyết lúc bắt đầu).
    """
    w = compute_ffd_weights(d, tau)
    # Với mode='valid', np.convolve(x, w) tính: sum_m(x[n-m] * w[m]) 
    # Do w[m] là trọng số của lag m (x_{t-m}), việc truyền trực tiếp w khớp chính xác 
    # với công thức: x_t*w_0 + x_{t-1}*w_1 + ... + x_{t-W}*w_W.
    # Chiều dài đầu ra = len(series) - len(w) + 1
    if len(series) < len(w):
        return np.array([], dtype=np.float64)
    return np.convolve(series, w, mode='valid')


def find_optimal_d_star(
    series: np.ndarray, 
    d_grid: np.ndarray, 
    tau: float = 1e-5, 
    p_value_threshold: float = 0.05
) -> float:
    """
    [TASK A-4-2] Tìm bậc vi phân tối ưu d* nhỏ nhất làm cho chuỗi thời gian trở nên dừng
    (stationary) bằng kiểm định Augmented Dickey-Fuller (ADF).
    
    Args:
        series (np.ndarray): Chuỗi thời gian đầu vào.
        d_grid (np.ndarray): Mảng các giá trị d để test (thường là [0.0, 0.1, ..., 1.0]).
        tau (float): Ngưỡng cắt đuôi trọng số FFD.
        p_value_threshold (float): Ngưỡng p-value cho kiểm định ADF.
        
    Returns:
        float: Giá trị d* tối ưu tìm được, hoặc giá trị lớn nhất trong d_grid nếu không tìm thấy d* nào thỏa mãn.
    """
    # [BUG FIX #7] Guard cho d_grid rỗng: tránh IndexError: index -1 is out of bounds
    if not hasattr(d_grid, '__len__') or len(d_grid) == 0:
        raise ValueError(
            "[find_optimal_d_star] d_grid không được rỗng. "
            "Truyền vào mảng các giá trị d cần kiểm tra (ví dụ: np.linspace(0.0, 1.0, 11))."
        )
    d_grid = np.asarray(d_grid, dtype=np.float64)

    # [BUG FIX #8] Lazy import: statsmodels chỉ được nạp khi hàm này được gọi thực sự
    from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415

    tracker = ExperimentTracker()
    
    optimal_d = d_grid[-1]
    optimal_pvalue = 1.0
    
    # Iterate theo thứ tự tăng dần của d (tìm d nhỏ nhất)
    sorted_grid = np.sort(d_grid)
    
    for d in sorted_grid:
        ffd_series = apply_ffd(series, float(d), tau)
        if len(ffd_series) < 10:  # Không đủ data cho ADF
            continue
            
        # ADF Test
        # Tham số autolag='AIC' thường được dùng trong chuẩn thống kê
        try:
            adf_result = adfuller(ffd_series, autolag='AIC')
            pvalue = adf_result[1]
            
            if pvalue < p_value_threshold:
                optimal_d = d
                optimal_pvalue = pvalue
                break
        except Exception:
            # Bỏ qua nếu adfuller lỗi (ví dụ variance = 0)
            continue
            
    # Ghi nhận trial bằng ExperimentTracker (A-0-1)
    tracker.log_trial(
        trial_class=TrialClass.MODEL_FITTING,
        params={
            "d_grid_min": float(np.min(d_grid)),
            "d_grid_max": float(np.max(d_grid)),
            "d_grid_size": len(d_grid),
            "tau": tau,
            "p_value_threshold": p_value_threshold,
            "module": "find_optimal_d_star"
        },
        metrics={
            "optimal_d_star": float(optimal_d),
            "p_value_achieved": float(optimal_pvalue)
        }
    )
    
    return float(optimal_d)
