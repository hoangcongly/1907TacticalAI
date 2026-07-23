"""
Local Linear Trend Kalman Filter 2D [P_t, nu_t], xuất Trend_Score theo ATR.
Triển khai thuần bằng Numpy để hỗ trợ Gap-handling predict-only (không update khi dữ liệu NaN).
"""
import numpy as np

class LocalLinearTrendKalman:
    def __init__(self, process_noise_level: float = 1e-4, process_noise_trend: float = 1e-5, observation_noise: float = 1e-2):
        """
        Khởi tạo bộ lọc Kalman Local Linear Trend (LLT).
        x = [mu, nu]^T (Level, Trend)
        """
        self.F = np.array([[1.0, 1.0], 
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        
        # Ma trận hiệp phương sai nhiễu quá trình (Q)
        self.Q = np.array([[process_noise_level, 0.0],
                           [0.0, process_noise_trend]])
                           
        # Ma trận hiệp phương sai nhiễu quan sát (R)
        self.R = np.array([[observation_noise]])
        
        # Trạng thái hiện tại
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        
        self.is_initialized = False

    def reset(self):
        """Reset trạng thái bộ lọc."""
        self.is_initialized = False
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)

    def step(self, z: float) -> tuple[float, float]:
        """
        Thực hiện một bước (predict & update) trên quan sát mới z.
        Nếu z là NaN, hệ thống thực hiện gap-handling (chỉ predict, không update).
        Trả về (level, trend).
        """
        # --- Khởi tạo ---
        if not self.is_initialized and not np.isnan(z):
            self.x = np.array([[z], [0.0]])  # type: ignore
            self.P = np.eye(2) * 1.0
            self.is_initialized = True
            return float(self.x[0, 0]), float(self.x[1, 0])
            
        if not self.is_initialized:
            # Nếu truyền NaN ngay từ đầu, giữ nguyên 0
            return 0.0, 0.0

        # --- PREDICT ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- GAP-HANDLING ---
        if np.isnan(z):
            # Không có quan sát -> trạng thái predict trở thành trạng thái hiện tại (chỉ trôi theo trend)
            self.x = x_pred  # type: ignore
            self.P = P_pred  # type: ignore
            return float(self.x[0, 0]), float(self.x[1, 0])

        # --- UPDATE ---
        y = z - (self.H @ x_pred)[0, 0] # Innovation
        S = self.H @ P_pred @ self.H.T + self.R # Innovation covariance
        K = (P_pred @ self.H.T) / S[0, 0] # Kalman Gain

        self.x = x_pred + K * y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def filter_series(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Chạy bộ lọc qua toàn bộ mảng giá.
        Trích xuất chuỗi (Level, Trend).
        """
        levels = np.zeros_like(prices, dtype=float)
        trends = np.zeros_like(prices, dtype=float)
        
        self.reset()
        for i, z in enumerate(prices):
            lvl, trnd = self.step(z)
            levels[i] = lvl
            trends[i] = trnd
            
        return levels, trends
