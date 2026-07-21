"""Unit tests cho thuật toán lọc nhiễu Outlier Detection (Task A-1-1)."""

import numpy as np
import pytest
from aegis.data.outlier_detection import compute_rolling_mad


def test_compute_rolling_mad_synthetic():
    """
    [TASK A-1-1] Kiểm thử hiệu năng và độ chính xác của hàm compute_rolling_mad
    trên chuỗi synthetic có MAD (Sigma) biết trước.
    """
    # 1. Tạo chuỗi Synthetic chuẩn
    np.random.seed(42)
    n = 1000
    true_mu = 100.0
    true_sigma = 2.0
    
    # Phân phối chuẩn N(100, 2)
    clean_prices = np.random.normal(true_mu, true_sigma, n)
    
    # Bơm 50 nhiễu cực đoan (Bad Ticks) ngẫu nhiên
    dirty_prices = clean_prices.copy()
    outlier_indices = np.random.choice(range(100, n), size=50, replace=False)
    # Nhiễu tăng/giảm kịch trần (30 giá đơn vị ~ 15 lần sigma)
    dirty_prices[outlier_indices] += np.random.choice([-30.0, 30.0], size=50)

    # 2. Tính Robust Sigma
    window = 100
    robust_sigmas = compute_rolling_mad(dirty_prices, window=window)

    # 3. Phân tích kết quả
    # Điểm chưa đủ lịch sử phải bằng NaN
    assert np.isnan(robust_sigmas[:window]).all(), "Các điểm khởi động phải là NaN"
    
    # Phân tích sai số Robust Sigma tại vùng dữ liệu ổn định (không xét các nhiễu mới xuất hiện)
    # Tại mọi điểm i, robust_sigma được ước lượng dựa trên window quá khứ.
    # Trong window 100 điểm, có trung bình 5 điểm nhiễu. Median rất ít bị ảnh hưởng bởi 5/100 nhiễu.
    
    valid_sigmas = robust_sigmas[window:]
    
    # Sigma trung bình ước lượng phải xấp xỉ true_sigma (2.0)
    estimated_sigma_mean = np.mean(valid_sigmas)
    
    # Chênh lệch so với sigma thực tế (Tolerance < 10%)
    assert abs(estimated_sigma_mean - true_sigma) / true_sigma < 0.1, \
        f"Robust Sigma quá sai lệch: Tính được {estimated_sigma_mean}, Thực tế {true_sigma}"

    # Để so sánh, nếu dùng np.std trên chuỗi dirty, sigma sẽ bị phình to:
    naive_std = np.std(dirty_prices)
    assert naive_std > 5.0, f"Chuỗi dirty_prices không bị nhiễu đúng cách (Naive Std: {naive_std})"
    
    print(f"✅ [TASK A-1-1] Robust Sigma: {estimated_sigma_mean:.4f} (True: {true_sigma}) - Đánh bại Naive Std: {naive_std:.4f}")

def test_compute_rolling_mad_empty_or_short():
    """Kiểm tra phản ứng với mảng quá ngắn."""
    prices = np.array([100.0, 101.0, 102.0])
    res = compute_rolling_mad(prices, window=10)
    assert np.isnan(res).all(), "Mảng quá ngắn phải trả về chuỗi NaN tương ứng"
    assert len(res) == 3, "Độ dài kết quả phải khớp mảng gốc"
