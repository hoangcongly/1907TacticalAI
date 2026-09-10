"""
[FIX F6] Phân loại chế độ thị trường nhân quả, KHÔNG có tham số phải fit.

## Vì sao thay HMM

`signal_pipeline` cũ dựng `CausalHMM2State` bằng hằng số bịa:
    means = [0.001, 0.0],  stds = [0.02, 0.005]
Các số này chưa bao giờ được fit (`fit_hmm_2state_loglik` không tồn tại) và cố định
bất kể tài sản hay khung thời gian. Với BTC nến 1 phút (σ ≈ 0.0005), CẢ HAI trạng
thái đều lệch xa thực tế, và vì stds chênh nhau 4 lần nên "trend" thực chất chỉ là
"biến động cao" — không phải xu hướng.

## Giải pháp: Efficiency Ratio (Kaufman)

    ER_t = |P_t - P_{t-n}| / Σ|P_i - P_{i-1}|     trên cửa sổ n nến gần nhất

Ý nghĩa: quãng đường ròng chia cho tổng quãng đường đã đi.
    ER → 1  : giá đi thẳng một mạch      -> XU HƯỚNG
    ER → 0  : giá đi tới đi lui, triệt tiêu -> ĐI NGANG (chop)

Vì sao phù hợp hơn HMM ở đây:
- **Không có tham số nào phải fit** -> không thể overfit (gốc rễ của F6).
- **Tự chuẩn hoá**: là tỷ số nên tự thích ứng với mọi tài sản và khung thời gian,
  không cần hiệu chỉnh lại — đúng thứ tham số hardcode của HMM không làm được.
- **Nhân quả tuyệt đối**: chỉ dùng cửa sổ các nến <= t.
- Nằm sẵn trong [0, 1] nên dùng trực tiếp làm p_trend, p_chop = 1 - p_trend.

Giữ nguyên hợp đồng dữ liệu: p_trend + p_chop = 1.0 (chuẩn n_states = 2).
"""

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

# Cửa sổ mặc định. Đủ dài để lọc nhiễu, đủ ngắn để bám chuyển chế độ.
DEFAULT_ER_WINDOW = 20

# Hệ số làm mượt EWMA cho posterior, chống nhảy giật giữa hai chế độ.
DEFAULT_SMOOTHING = 0.30


class CausalEfficiencyRegime:
    """
    Phân loại chế độ 2 trạng thái (Trend / Chop) bằng Efficiency Ratio làm mượt EWMA.

    Dùng streaming: gọi `step()` cho từng nến theo thứ tự thời gian.
    """

    def __init__(
        self,
        window: int = DEFAULT_ER_WINDOW,
        smoothing: float = DEFAULT_SMOOTHING,
    ):
        if not isinstance(window, int) or window < 2:
            raise ValueError(f"window phải là số nguyên >= 2, nhận {window}")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError(f"smoothing phải thuộc (0, 1], nhận {smoothing}")

        self.window = window
        self.smoothing = float(smoothing)
        self._prices: Deque[float] = deque(maxlen=window + 1)
        self._p_trend: Optional[float] = None

    def reset(self) -> None:
        self._prices.clear()
        self._p_trend = None

    def step(self, price: float) -> Tuple[float, float]:
        """
        Nạp một mức giá, trả về `(p_trend, p_chop)` với p_trend + p_chop = 1.0.

        Trong lúc chưa đủ cửa sổ, trả về (0.5, 0.5) — trung lập, không thiên vị.
        """
        if price is None or not np.isfinite(price):
            # Giá rác: giữ nguyên posterior gần nhất thay vì bịa ra tín hiệu.
            current = 0.5 if self._p_trend is None else self._p_trend
            return float(current), float(1.0 - current)

        self._prices.append(float(price))

        if len(self._prices) < 3:
            return 0.5, 0.5

        prices = np.asarray(self._prices, dtype=np.float64)
        net_move = abs(float(prices[-1] - prices[0]))
        total_move = float(np.sum(np.abs(np.diff(prices))))

        # Giá đứng yên hoàn toàn -> không có thông tin xu hướng.
        if total_move <= 1e-12:
            er = 0.0
        else:
            er = net_move / total_move

        er = min(max(er, 0.0), 1.0)

        # Làm mượt EWMA để posterior không nhảy giật từng nến.
        if self._p_trend is None:
            self._p_trend = er
        else:
            self._p_trend = self.smoothing * er + (1.0 - self.smoothing) * self._p_trend

        p_trend = min(max(float(self._p_trend), 0.0), 1.0)
        return p_trend, float(1.0 - p_trend)

    def filter_series(self, prices: np.ndarray) -> np.ndarray:
        """
        Chạy qua toàn chuỗi, trả về ma trận [T, 2] = [[p_trend, p_chop], ...].
        Giữ cùng chữ ký với `CausalHMM2State.filter_series` để thay thế trực tiếp.
        """
        self.reset()
        out = np.zeros((len(prices), 2), dtype=np.float64)
        for i, price in enumerate(prices):
            out[i] = self.step(float(price))
        return out
