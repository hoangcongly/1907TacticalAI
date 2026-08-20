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

@njit
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


def fit_sum_of_exponentials_v2(weights: np.ndarray, M: int = 6, epsilon_approx: float = 1e-4):
    """
    [TASK A-4-3] Fit Sum of Exponentials (Prony Approximation) cho trọng số FFD.
    SỬA LỖI v11.5: cho phép rho ÂM để tái tạo dấu đổi luân phiên của w_k ở vùng k nhỏ.
    
    Args:
        weights (np.ndarray): Mảng trọng số gốc w_k(d).
        M (int): Số lượng hàm mũ.
        epsilon_approx (float): Ngưỡng sai số để approve thuật toán.
        
    Returns:
        tuple: (c, rho, approx_error_abs, weighted_rel_error, approved)
    """
    from scipy.optimize import curve_fit  # Lazy import để tránh overhead

    k = np.arange(len(weights))

    def model(k, *params):
        c = np.array(params[:M])
        rho = np.array(params[M:])
        # Use np.abs(rho) if k is fractional, but k is integer so rho**k is fine.
        # But rho < 0 and k is float can cause NaN. 
        # Actually k is np.arange (integers) so rho**k is valid for negative rho.
        return np.sum(c[:, None] * (rho[:, None] ** k[None, :]), axis=0)

    rho_init = np.concatenate([
        np.linspace(0.5, 0.995, M - M // 2),
        np.linspace(-0.5, -0.9, M // 2)
    ])
    c_init = np.ones(M) * (weights[0] / M)
    p0 = np.concatenate([c_init, rho_init])
    bounds = ([-np.inf] * M + [-0.9999] * M, [np.inf] * M + [0.9999] * M)

    popt, _ = curve_fit(model, k, weights, p0=p0, bounds=bounds, maxfev=50000)
    c, rho = popt[:M], popt[M:]
    reconstructed = model(k, *popt)

    approx_error_abs = float(np.max(np.abs(weights - reconstructed)))
    # Tránh chia cho 0
    denom = np.sum(weights ** 4)
    weighted_rel_error = float(np.sum(((weights - reconstructed) * weights) ** 2) / denom) if denom > 0 else 0.0
    
    approved = approx_error_abs < epsilon_approx and weighted_rel_error < epsilon_approx
    return c, rho, approx_error_abs, weighted_rel_error, approved


def select_ffd_production_engine(ffd_weights_exact: np.ndarray, M_prony: int = 6) -> dict:
    """
    [TASK A-4-3] Windowed FFD (Phương án 1) là MẶC ĐỊNH production duy nhất. Sum-of-Exponentials
    (Phương án 2) CHỈ được phép thay thế nếu approved=True TRÊN CHÍNH bộ trọng số d*
    của cấu hình đã qua Module F.
    """
    c, rho, err_abs, err_rel, approved = fit_sum_of_exponentials_v2(ffd_weights_exact, M=M_prony)

    if approved:
        return {
            "engine_type": "sum_of_exp", 
            "c": c.tolist(), 
            "rho": rho.tolist(),
            "validation": {"approx_error_abs": err_abs, "weighted_rel_error": err_rel}
        }
    else:
        w_star = ffd_weights_exact[np.abs(ffd_weights_exact) >= 1e-5]
        return {
            "engine_type": "windowed", 
            "weights": w_star.tolist(),
            "validation": {"approx_error_abs": err_abs, "weighted_rel_error": err_rel, "reason": "sum_of_exp REJECTED"}
        }
