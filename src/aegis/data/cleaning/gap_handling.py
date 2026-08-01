"""
[TASK A-3-1] Gap Handling — Giao thức chiếu Kalman qua khoảng đứt gãy.
Mô phỏng sự biến đổi của trạng thái hệ thống khi hoàn toàn mất tín hiệu (Predict-Only)
thông qua vòng lặp Numba bảo toàn tính Xác định Dương (PD).
"""

import numpy as np
from typing import Tuple
from numba import njit
from aegis.data.cleaning.tick_kalman_replacer import ensure_pd_matrix_2x2_numba


@njit(nopython=True)
def _kalman_predict_n_steps_numba(
    x_0: np.ndarray,
    P_0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vòng lặp Numba cấp thấp thực thi chiếu trạng thái.
    Thời gian chạy O(N) nhưng trên nền C-level (~1ms/1,000,000 steps).
    """
    # Khởi tạo bản sao để tránh thay đổi tham chiếu gốc
    x = x_0.copy()
    P = P_0.copy()

    for _ in range(n_steps):
        x = F @ x
        P = F @ P @ F.T + Q
        # [ARMOR GUARD] Bọc thép ma trận PD ở mọi bước để ngăn ngừa tích lũy sai số float
        P = ensure_pd_matrix_2x2_numba(P)

    return x, P


def kalman_predict_only_n_steps(
    x_0: np.ndarray,
    P_0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    n_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [TASK A-3-1] Chiếu trạng thái hệ thống (State) và ma trận Hiệp phương sai (Covariance)
    đi trước n_steps mà không sử dụng bất kỳ quan sát mới nào.
    
    Phục vụ cho:
    - Xử lý mất kết nối feed dài hạn (Data Gaps).
    - Tạo cầu nối an toàn đi qua các vùng Purge trong kiểm định chéo CPCV.
    
    Args:
        x_0: Vector trạng thái 2x1 [level, trend]^T tại t=0.
        P_0: Ma trận hiệp phương sai 2x2 tại t=0.
        F: Ma trận chuyển trạng thái 2x2 (vd: Identity hoặc Local Linear Trend).
        Q: Ma trận hiệp phương sai nhiễu quá trình 2x2.
        n_steps: Số lượng bước (ticks/chu kỳ) bị đứt gãy.
        
    Returns:
        Tuple[x_n, P_n]: Trạng thái và ma trận hiệp phương sai tại t=n.
    """
    # 1. Input Sanitization & Type Enforcement (SOP Phase 2)
    if not isinstance(n_steps, (int, np.integer)):
        raise TypeError(f"n_steps phải là số nguyên, nhận {type(n_steps)}")
    if n_steps < 0:
        raise ValueError(f"n_steps không được âm, nhận {n_steps}")
    
    if n_steps == 0:
        return x_0.copy(), P_0.copy()

    x_0_arr = np.asarray(x_0, dtype=np.float64)
    P_0_arr = np.asarray(P_0, dtype=np.float64)
    F_arr = np.asarray(F, dtype=np.float64)
    Q_arr = np.asarray(Q, dtype=np.float64)

    # 2. Shape Verification (Cố định thiết kế 2x2 cho Track A)
    if x_0_arr.shape != (2, 1):
        raise ValueError(f"x_0 phải có kích thước (2, 1), nhận {x_0_arr.shape}")
    for mat_name, mat in [("P_0", P_0_arr), ("F", F_arr), ("Q", Q_arr)]:
        if mat.shape != (2, 2):
            raise ValueError(f"Ma trận {mat_name} phải có kích thước (2, 2), nhận {mat.shape}")

    # 3. Thực thi chiếu qua Numba Engine
    x_n, P_n = _kalman_predict_n_steps_numba(x_0_arr, P_0_arr, F_arr, Q_arr, int(n_steps))
    
    return x_n, P_n
