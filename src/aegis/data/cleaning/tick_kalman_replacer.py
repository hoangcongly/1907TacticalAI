"""
[TASK A-1-4] TickLevelKalmanReplacer — Predict-Only Protocol khi phát hiện Bad Tick hoặc mất quan sát (NaN).
Bộ lọc Kalman 2 trạng thái [P_t, \nu_t]^T giữ vững động lượng vi cấu trúc, loại bỏ hiện tượng đứt gãy hoặc bạt ngang,
sử dụng phân rã Cholesky bảo vệ tính xác định dương (PD) theo chuẩn SOP.
"""

import math
import numpy as np
from typing import Optional, Tuple
from numba import njit


@njit
def ensure_pd_matrix_2x2_numba(mat: np.ndarray, min_jitter: float = 1e-8) -> np.ndarray:
    """
    [ARMOR GUARD] Đảm bảo ma trận hiệp phương sai 2x2 là Positive Definite (PD) trong Numba
    bằng cách đối xứng hóa và kiểm tra phân rã Cholesky (np.linalg.cholesky) theo quy định SOP.
    """
    res = np.empty((2, 2), dtype=np.float64)
    # 1. Đối xứng hóa (Symmetrize)
    res[0, 0] = float(mat[0, 0])
    res[0, 1] = 0.5 * float(mat[0, 1] + mat[1, 0])
    res[1, 0] = res[0, 1]
    res[1, 1] = float(mat[1, 1])

    # 2. Đảm bảo đường chéo chính và định thức strictly positive trước khi gọi Cholesky
    if res[0, 0] <= min_jitter:
        res[0, 0] = min_jitter
    if res[1, 1] <= min_jitter:
        res[1, 1] = min_jitter

    det = res[0, 0] * res[1, 1] - res[0, 1] * res[0, 1]
    if det <= min_jitter * min_jitter:
        # Bổ sung jitter chéo (ridge regularization) để đảm bảo định thức dương
        needed_prod = res[0, 1] * res[0, 1] + 1e-6
        diag_val = np.sqrt(needed_prod) + 1e-4
        if res[0, 0] < diag_val:
            res[0, 0] = diag_val
        if res[1, 1] < diag_val:
            res[1, 1] = diag_val

    # 3. Sử dụng Cholesky decomposition (np.linalg.cholesky) theo đúng chỉ thị SOP
    L = np.linalg.cholesky(res)
    # Trả về ma trận được tái tạo từ Cholesky factor (chắc chắn PD tuyệt đối)
    return L @ L.T


@njit
def kalman_replacer_filter_series_numba(
    prices: np.ndarray,
    is_bad_ticks: np.ndarray,
    Q_tick: np.ndarray,
    R_tick: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [TASK A-1-4 - OPTIMIZED NUMBA ENGINE] Thực thi Kalman Replacer trên toàn bộ mảng tick bằng C-speed Numba.
    
    Tham số:
        prices: Mảng 1D giá khớp lệnh thô.
        is_bad_ticks: Mảng 1D cờ boolean đánh dấu Bad Tick (hoặc truyền toàn False nếu chỉ lọc NaN).
        Q_tick: Ma trận 2x2 hiệp phương sai nhiễu quá trình.
        R_tick: Hiệp phương sai (variance) nhiễu quan sát.
        
    Trả về:
        Tuple (replaced_prices, level_estimates, trend_estimates):
        - replaced_prices: Chuỗi giá đã được thay thế (Good Tick giữ giá quan sát, Bad Tick dùng y_hat predict).
        - level_estimates: Trạng thái Level (P_t) sau khi chạy bộ lọc.
        - trend_estimates: Trạng thái Momentum (\nu_t).
    """
    n = len(prices)
    replaced_prices = np.full(n, np.nan, dtype=np.float64)
    level_estimates = np.zeros(n, dtype=np.float64)
    trend_estimates = np.zeros(n, dtype=np.float64)

    if n == 0:
        return replaced_prices, level_estimates, trend_estimates

    # Khởi tạo ma trận F, H, Q, R
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = ensure_pd_matrix_2x2_numba(Q_tick)
    R = max(float(R_tick), 1e-12)  # Cap against invalid zero/negative R

    # Trạng thái ban đầu x = [P_0, \nu_0]^T và covariance P
    x = np.zeros((2, 1), dtype=np.float64)
    P = np.eye(2, dtype=np.float64) * 1.0
    is_initialized = False

    for i in range(n):
        p_obs = prices[i]
        bad_tick = is_bad_ticks[i]
        is_unobserved = np.isnan(p_obs) or p_obs <= 0.0 or math.isinf(p_obs)
        skip_update = bad_tick or is_unobserved

        if not is_initialized:
            if not skip_update:
                x[0, 0] = p_obs
                x[1, 0] = 0.0
                P = np.eye(2, dtype=np.float64) * 1.0
                is_initialized = True
                replaced_prices[i] = p_obs
                level_estimates[i] = p_obs
                trend_estimates[i] = 0.0
            else:
                # Chưa được khởi tạo từ giá hợp lệ -> Giữ nguyên NaN/0.0
                replaced_prices[i] = np.nan
                level_estimates[i] = 0.0
                trend_estimates[i] = 0.0
            continue

        # --- BƯỚC PREDICT (Luôn thực hiện) ---
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        P_pred = ensure_pd_matrix_2x2_numba(P_pred)  # Bảo vệ tính PD

        y_hat_one_step = float((H @ x_pred)[0, 0])

        if skip_update:
            # --- GIAO THỨC PREDICT-ONLY (Bỏ qua Cập nhật) ---
            # Xử lý Bad Tick hoặc nhánh predict-only không dùng giá quan sát (NaN gap)
            x = x_pred
            P = P_pred
            replaced_prices[i] = y_hat_one_step
        else:
            # --- BƯỚC UPDATE (Khi gặp Good Tick hợp lệ) ---
            S_scalar = float((H @ P_pred @ H.T)[0, 0]) + R
            # Cap scalars để ngăn chia cho 0 hoặc suy biến
            if S_scalar < 1e-12:
                S_scalar = 1e-12

            K = (P_pred @ H.T) / S_scalar
            innovation = p_obs - y_hat_one_step
            
            # Cập nhật trạng thái
            x = x_pred + K * innovation
            
            # Cập nhật hiệp phương sai P = (I - K H) P_pred
            I2 = np.eye(2, dtype=np.float64)
            P = (I2 - K @ H) @ P_pred
            P = ensure_pd_matrix_2x2_numba(P)  # Bảo vệ tính PD tuyệt đối bằng Cholesky

            replaced_prices[i] = p_obs

        # Lưu log trạng thái
        level_estimates[i] = float(x[0, 0])
        trend_estimates[i] = float(x[1, 0])

    return replaced_prices, level_estimates, trend_estimates


class TickLevelKalmanReplacer:
    """
    [TASK A-1-4] TickLevelKalmanReplacer — Master Blueprint Compliant Implementation.
    Bộ lọc Kalman vi cấu trúc 2 trạng thái [P_t, \\nu_t]^T chuyên xử lý Bad Tick và các
    nhánh predict-only không dùng giá quan sát (khi rớt feed, đứt gãy hoặc tick bị lỗi).
    """

    def __init__(self, Q_tick: Optional[np.ndarray] = None, R_tick: float = 1e-2):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)

        if Q_tick is None:
            # Default process noise covariance: level_noise = 1e-4, trend_noise = 1e-5
            Q_tick = np.array([[1e-4, 0.0], [0.0, 1e-5]], dtype=np.float64)
        else:
            Q_tick = np.asarray(Q_tick, dtype=np.float64)
            if Q_tick.shape != (2, 2):
                raise ValueError(f"Q_tick phải có kích thước (2, 2), nhận {Q_tick.shape}")

        # Bảo đảm Q_tick là Positive Definite ngay từ khi khởi tạo (SOP constraint)
        self.Q = ensure_pd_matrix_2x2_numba(Q_tick)

        if not isinstance(R_tick, (int, float)) or math.isnan(R_tick) or math.isinf(R_tick):
            raise ValueError(f"R_tick phải là con số hợp lệ, nhận {R_tick}")
        self.R = max(float(R_tick), 1e-12)

        self.x: Optional[np.ndarray] = None
        self.P = np.eye(2, dtype=np.float64) * 1.0

    def reset(self) -> None:
        """Reset trạng thái bộ lọc."""
        self.x = None
        self.P = np.eye(2, dtype=np.float64) * 1.0

    def step(self, price_observed: float, is_bad_tick: bool = False) -> float:
        """
        [TASK A-1-4] Thực thi 1 tick (Real-Time OOP Interface).
        Nếu is_bad_tick = True hoặc price_observed bị NaN/không hợp lệ, kích hoạt nhánh
        Predict-Only không dùng giá quan sát để duy trì động lượng.
        """
        is_unobserved = np.isnan(price_observed) or price_observed <= 0.0 or math.isinf(price_observed)
        skip_update = bool(is_bad_tick) or is_unobserved

        if self.x is None:
            if not skip_update:
                self.x = np.array([[float(price_observed)], [0.0]], dtype=np.float64)
                self.P = np.eye(2, dtype=np.float64) * 1.0
                return float(price_observed)
            else:
                # Chưa khởi tạo được bộ lọc do tick đầu tiên đã là lỗi hoặc NaN
                return np.nan if is_unobserved else float(price_observed)

        # --- PREDICT STEP ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        P_pred = ensure_pd_matrix_2x2_numba(P_pred)  # Bảo quản tính PD qua Cholesky

        y_hat_one_step = float((self.H @ x_pred)[0, 0])

        if skip_update:
            # --- PREDICT-ONLY PROTOCOL ---
            # Không cập nhật từ quan sát lỗi/mất; trượt tiếp theo động lượng hiện tại
            self.x = x_pred
            self.P = P_pred
            return y_hat_one_step
        else:
            # --- UPDATE STEP ---
            S_val = float((self.H @ P_pred @ self.H.T)[0, 0]) + self.R
            if S_val < 1e-12:
                S_val = 1e-12  # Cap extreme scalars

            K = (P_pred @ self.H.T) / S_val
            innovation = float(price_observed) - y_hat_one_step

            self.x = x_pred + (K.flatten().reshape(2, 1) * innovation)
            I2 = np.eye(2, dtype=np.float64)
            self.P = (I2 - K @ self.H) @ P_pred
            self.P = ensure_pd_matrix_2x2_numba(self.P)  # Đảm bảo Cholesky decomposition

            return float(price_observed)

    def filter_series(
        self,
        prices: np.ndarray,
        is_bad_ticks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [TASK A-1-4] Chạy xử lý hàng loạt trên mảng tick-level sử dụng Numba C-level Engine.
        Đạt tốc độ O(1) per-tick cho dữ liệu tần suất cao.
        
        Trả về Tuple: (replaced_prices, level_estimates, trend_estimates)
        """
        prices_arr = np.asarray(prices, dtype=np.float64)
        n = len(prices_arr)

        if is_bad_ticks is None:
            bad_ticks_arr = np.zeros(n, dtype=np.bool_)
        else:
            bad_ticks_arr = np.asarray(is_bad_ticks, dtype=np.bool_)

        if len(bad_ticks_arr) != n:
            raise ValueError("Độ dài mảng prices và is_bad_ticks phải hoàn toàn khớp nhau")

        rep, lvl, trnd = kalman_replacer_filter_series_numba(
            prices=prices_arr,
            is_bad_ticks=bad_ticks_arr,
            Q_tick=self.Q,
            R_tick=self.R
        )
        
        # Cập nhật trạng thái cuối vào object để có thể step tiếp seamless
        if n > 0 and not np.isnan(lvl[-1]):
            self.x = np.array([[lvl[-1]], [trnd[-1]]], dtype=np.float64)
            
        return rep, lvl, trnd
