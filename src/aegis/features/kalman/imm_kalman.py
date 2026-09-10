"""
IMM Kalman 2D — 2 bộ lọc Trending & Choppy chạy song song kết hợp sanitize_covariance_matrix.
Module B.2 (PATCH D v11.5) theo đặc tả kiến trúc Mục 2.3.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from aegis.features.kalman.covariance_utils import sanitize_covariance_matrix


class IMMKalman2D:
    """
    Interacting Multiple Model (IMM) Kalman Filter 2D:
    - Bộ lọc j=1 (Trending): Q_trend lớn bám sát vận tốc drift nu_t.
    - Bộ lọc j=2 (Choppy): Q_chop nhỏ triệt tiêu dao động quanh mean.
    - Làm sạch hiệp phương sai: sanitize_covariance_matrix chạy ngay sau mỗi bước Update.
    - Master Regime Probability Injection: Nhận xác suất từ Causal HMM.
    """

    def __init__(
        self,
        q_trend: Optional[np.ndarray] = None,
        q_chop: Optional[np.ndarray] = None,
        r_meas: float = 1e-3,
        transition_matrix: Optional[np.ndarray] = None,
        eigenvalue_floor: float = 1e-10,
    ):
        # Ma trận chuyển trạng thái F và ma trận đo lường H
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.I = np.eye(2, dtype=np.float64)

        # Ma trận nhiễu quá trình Q_trend và Q_chop
        self.Q_trend = (
            np.asarray(q_trend, dtype=np.float64)
            if q_trend is not None
            else np.diag([1e-4, 1e-2])
        )
        self.Q_chop = (
            np.asarray(q_chop, dtype=np.float64)
            if q_chop is not None
            else np.diag([1e-6, 1e-7])
        )
        self.R = float(r_meas)

        # Ma trận chuyển tiếp giữa 2 regime A
        self.A = (
            np.asarray(transition_matrix, dtype=np.float64)
            if transition_matrix is not None
            else np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)
        )
        self.eigenvalue_floor = float(eigenvalue_floor)

        # Trạng thái ban đầu [P, nu]^T
        self.x1 = np.zeros((2, 1), dtype=np.float64)
        self.x2 = np.zeros((2, 1), dtype=np.float64)
        self.P1 = sanitize_covariance_matrix(np.eye(2, dtype=np.float64), self.eigenvalue_floor)
        self.P2 = sanitize_covariance_matrix(np.eye(2, dtype=np.float64), self.eigenvalue_floor)

        self.p_prev = np.array([0.5, 0.5], dtype=np.float64)
        self._initialized = False

    def reset(self, initial_price: Optional[float] = None) -> None:
        """Khởi động lại bộ lọc."""
        p0 = float(initial_price) if initial_price is not None else 0.0
        self.x1 = np.array([[p0], [0.0]], dtype=np.float64)
        self.x2 = np.array([[p0], [0.0]], dtype=np.float64)
        self.P1 = sanitize_covariance_matrix(np.eye(2, dtype=np.float64), self.eigenvalue_floor)
        self.P2 = sanitize_covariance_matrix(np.eye(2, dtype=np.float64), self.eigenvalue_floor)
        self.p_prev = np.array([0.5, 0.5], dtype=np.float64)
        self._initialized = initial_price is not None

    def step(
        self,
        observation: float,
        p_trend: float = 0.5,
        p_chop: float = 0.5,
        atr: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Thực hiện 1 bước lặp đầy đủ 7 bước IMM Kalman 2D:
        observation: Giá quan sát hiện tại (Close / P_t).
        p_trend: Xác suất regime Trending từ HMM.
        p_chop: Xác suất regime Choppy từ HMM.
        atr: Độ biến động ATR_14 hiện tại.
        
        Trả về: (P_hat, nu_hat, trend_score)
        """
        y_t = float(observation)
        if not self._initialized:
            self.x1[0, 0] = y_t
            self.x2[0, 0] = y_t
            self._initialized = True

        # ====================================================================
        # BƯỚC 1: MIXING PROBABILITIES
        # c_j = sum_{i=1}^2 A_{ij} * p_{prev, i}
        # mu_{i|j} = (A_{ij} * p_{prev, i}) / c_j
        # ====================================================================
        # A có shape (2, 2) với A[i, j] là chuyển từ i sang j
        c = self.p_prev @ self.A
        c = np.maximum(c, 1e-12)

        mu = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                mu[i, j] = (self.A[i, j] * self.p_prev[i]) / c[j]

        # ====================================================================
        # BƯỚC 2: MIXED INITIAL CONDITIONS & SPREAD-OF-MEANS
        # ====================================================================
        x_list = [self.x1, self.x2]
        P_list = [self.P1, self.P2]

        x0 = [np.zeros((2, 1), dtype=np.float64), np.zeros((2, 1), dtype=np.float64)]
        P0 = [np.zeros((2, 2), dtype=np.float64), np.zeros((2, 2), dtype=np.float64)]

        for j in range(2):
            # Trộn trạng thái x0j
            x0[j] = mu[0, j] * x_list[0] + mu[1, j] * x_list[1]
            # Trộn ma trận hiệp phương sai P0j kèm spread-of-means
            for i in range(2):
                diff = x_list[i] - x0[j]
                spread = diff @ diff.T
                P0[j] += mu[i, j] * (P_list[i] + spread)
            P0[j] = sanitize_covariance_matrix(P0[j], self.eigenvalue_floor)

        # ====================================================================
        # BƯỚC 3: PREDICTION (Dự báo độc lập từng mô hình)
        # ====================================================================
        x_pred1 = self.F @ x0[0]
        P_pred1 = self.F @ P0[0] @ self.F.T + self.Q_trend

        x_pred2 = self.F @ x0[1]
        P_pred2 = self.F @ P0[1] @ self.F.T + self.Q_chop

        # ====================================================================
        # BƯỚC 4: MEASUREMENT UPDATE & LÀM SẠCH HIỆP PHƯƠNG SAI
        # ====================================================================
        # Update model 1 (Trending)
        S1 = float((self.H @ P_pred1 @ self.H.T)[0, 0] + self.R)
        K1 = (P_pred1 @ self.H.T) / max(S1, 1e-12)
        innov1 = y_t - float((self.H @ x_pred1)[0, 0])
        self.x1 = x_pred1 + K1 * innov1
        P1_raw = (self.I - K1 @ self.H) @ P_pred1
        self.P1 = sanitize_covariance_matrix(P1_raw, self.eigenvalue_floor)

        # Update model 2 (Choppy)
        S2 = float((self.H @ P_pred2 @ self.H.T)[0, 0] + self.R)
        K2 = (P_pred2 @ self.H.T) / max(S2, 1e-12)
        innov2 = y_t - float((self.H @ x_pred2)[0, 0])
        self.x2 = x_pred2 + K2 * innov2
        P2_raw = (self.I - K2 @ self.H) @ P_pred2
        self.P2 = sanitize_covariance_matrix(P2_raw, self.eigenvalue_floor)

        # ====================================================================
        # BƯỚC 5: MASTER REGIME PROBABILITY INJECTION (Từ Causal HMM)
        # ====================================================================
        w_trend = max(0.0, float(p_trend))
        w_chop = max(0.0, float(p_chop))
        sum_w = w_trend + w_chop
        if sum_w > 1e-12:
            w_trend /= sum_w
            w_chop /= sum_w
        else:
            w_trend, w_chop = 0.5, 0.5
        self.p_prev = np.array([w_trend, w_chop], dtype=np.float64)

        # ====================================================================
        # BƯỚC 6: COMBINED OUTPUT
        # ====================================================================
        x_comb = w_trend * self.x1 + w_chop * self.x2
        p_hat = float(x_comb[0, 0])
        nu_hat = float(x_comb[1, 0])

        # ====================================================================
        # BƯỚC 7: TREND SCORE OUTPUT (Chuẩn hóa theo ATR)
        # Trend_Score = nu_hat / (ATR_14 + 10^-8)
        # ====================================================================
        atr_safe = max(float(atr) if (atr is not None and not np.isnan(atr)) else 1.0, 1e-8)
        trend_score = nu_hat / atr_safe

        return p_hat, nu_hat, trend_score


def filter_imm_kalman_series(
    prices: np.ndarray,
    p_trend_series: np.ndarray,
    p_chop_series: np.ndarray,
    atr_series: np.ndarray,
    imm_filter: Optional[IMMKalman2D] = None,
) -> pd.DataFrame:
    """
    Chạy bộ lọc IMM Kalman 2D qua toàn bộ chuỗi thời gian nến.
    Trả về pd.DataFrame gồm các cột: ['p_hat', 'nu_hat', 'trend_score'].
    """
    n = len(prices)
    p_hat = np.empty(n, dtype=np.float64)
    nu_hat = np.empty(n, dtype=np.float64)
    trend_score = np.empty(n, dtype=np.float64)

    imm = imm_filter if imm_filter is not None else IMMKalman2D()
    if n > 0:
        imm.reset(float(prices[0]))

    for i in range(n):
        pt = float(p_trend_series[i]) if not np.isnan(p_trend_series[i]) else 0.5
        pc = float(p_chop_series[i]) if not np.isnan(p_chop_series[i]) else 0.5
        at = float(atr_series[i]) if not np.isnan(atr_series[i]) else 1.0
        p_h, nu_h, t_s = imm.step(float(prices[i]), p_trend=pt, p_chop=pc, atr=at)
        p_hat[i] = p_h
        nu_hat[i] = nu_h
        trend_score[i] = t_s

    return pd.DataFrame({
        "p_hat": p_hat,
        "nu_hat": nu_hat,
        "trend_score": trend_score,
    })
