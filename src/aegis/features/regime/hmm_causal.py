"""
Causal-Only Hidden Markov Model (HMM) - Forward Alpha Pass.
Đảm bảo 100% không có look-ahead bias (không dùng backward smoothing).
Kiến trúc chốt cứng N=2 trạng thái (Trending, Choppy).
"""
import numpy as np

class CausalHMM2State:
    def __init__(self, transition_matrix: np.ndarray, means: np.ndarray, stds: np.ndarray):
        """
        Khởi tạo HMM 2 trạng thái.
        transition_matrix: 2x2 (A)
        means: [mu_0, mu_1]
        stds: [sigma_0, sigma_1]
        """
        # [BUG FIX #3] Dùng if/raise thay vì assert (assert bị vô hiệu hóa bởi python -O trong production Docker)
        if transition_matrix.shape != (2, 2):
            raise ValueError(
                f"[CausalHMM2State] transition_matrix phải có kích thước (2, 2), nhận {transition_matrix.shape}"
            )
        if len(means) != 2:
            raise ValueError(
                f"[CausalHMM2State] means phải có đúng 2 phần tử (2 trạng thái), nhận {len(means)}"
            )
        if len(stds) != 2:
            raise ValueError(
                f"[CausalHMM2State] stds phải có đúng 2 phần tử (2 trạng thái), nhận {len(stds)}"
            )
        if np.any(np.array(stds) <= 0):
            raise ValueError(
                f"[CausalHMM2State] Tất cả stds phải > 0 (phân phối Gaussian hợp lệ), nhận {stds}"
            )
        
        self.A = transition_matrix
        self.means = means
        self.stds = stds
        
        # Khởi tạo prior ban đầu (trạng thái cân bằng hoặc đồng đều)
        self.alpha = np.array([0.5, 0.5])
        
    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Hàm mật độ xác suất phân phối chuẩn."""
        if std <= 0:
            return 1e-9
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def step(self, observation: float) -> np.ndarray:
        """
        Thực hiện Forward step.
        observation: Lợi nhuận (log-return) hoặc volatility của một nến.
        Trả về: np.array([P(State=0 | O_{1:t}), P(State=1 | O_{1:t})])
        """
        if np.isnan(observation):
            # Nếu NaN, chỉ nhân với ma trận chuyển trạng thái (trôi theo prior)
            alpha_pred = self.alpha @ self.A
            self.alpha = alpha_pred
            return self.alpha
            
        # Tính xác suất phát xạ B(O_t)
        b0 = self._gaussian_pdf(observation, self.means[0], self.stds[0])
        b1 = self._gaussian_pdf(observation, self.means[1], self.stds[1])
        B = np.array([b0, b1])
        
        # Ngăn chặn underflow
        B = np.clip(B, 1e-15, None)

        # Cập nhật alpha: alpha_t = B * (alpha_{t-1} * A)
        alpha_pred = self.alpha @ self.A
        alpha_new = B * alpha_pred
        
        # Chuẩn hóa (Normalization)
        sum_alpha = np.sum(alpha_new)
        if sum_alpha > 0:
            self.alpha = alpha_new / sum_alpha
        else:
            # Fallback nếu số quá nhỏ, reset về dự đoán ban đầu
            self.alpha = alpha_pred
            
        return self.alpha

    def filter_series(self, observations: np.ndarray) -> np.ndarray:
        """
        Chạy qua chuỗi thời gian, trả về ma trận xác suất posterior [T, 2].
        """
        self.alpha = np.array([0.5, 0.5]) # Reset
        
        n = len(observations)
        posteriors = np.zeros((n, 2))
        
        for i in range(n):
            posteriors[i] = self.step(observations[i])
            
        return posteriors
