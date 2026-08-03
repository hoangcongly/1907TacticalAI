import numpy as np
from aegis.features.regime.ghe import compute_ghe

def test_ghe_distinguishes_random_walk_and_trend():
    """
    Kiểm chứng GHE (Generalized Hurst Exponent) phân biệt được:
    - Random Walk: H ≈ 0.5
    - Trending (Persistent): H > 0.55
    """
    np.random.seed(42)
    n = 2000
    
    # 1. Random Walk
    rw = np.cumsum(np.random.randn(n))
    h_rw = compute_ghe(rw, max_lag=20)
    
    # H của Random Walk thường dao động từ 0.45 đến 0.55
    assert 0.40 < h_rw < 0.60, f"H của Random Walk phải gần 0.5, hiện tại: {h_rw}"
    
    # 2. Trending Series (rất mượt)
    # y = t + noise nhẹ
    trend = np.arange(n) + np.random.randn(n) * 2.0
    h_trend = compute_ghe(trend, max_lag=20)
    
    # H của Trending thường > 0.8
    assert h_trend > 0.60, f"H của Trending Series phải cao (>0.6), hiện tại: {h_trend}"

def test_ghe_nan_handling():
    """
    [BUG FIX #4] Sau khi vá lỗi, compute_ghe phải raise ValueError khi prices chứa NaN.
    Hành vi cũ (bỏ qua NaN silent) là sai vì tạo ra H bị lệch do discontinuity bị được bỏ qua.
    Caller phải lọc NaN trước (ví dụ dùng Kalman replacer hoặc forward-fill) rồi mới gọi GHE.
    """
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(500))
    rw_nan = rw.copy()
    rw_nan[100:110] = np.nan
    rw_nan[250:260] = np.nan

    # Sau fix: NaN trong prices phải raise ValueError rõ ràng thay vì trả về H sai
    import pytest
    with pytest.raises(ValueError, match="prices chứa NaN"):
        compute_ghe(rw_nan, max_lag=10)

    # Series sạch (sau khi đã lọc NaN bằng forward-fill) vẫn hoạt động bình thường
    rw_clean = rw.copy()  # không có NaN
    h_clean = compute_ghe(rw_clean, max_lag=10)
    assert 0.40 < h_clean < 0.60, f"GHE trên series sạch phải gần 0.5, nhận {h_clean}"


def test_ghe_inf_handling():
    """
    [BUG FIX #4] compute_ghe phải raise ValueError khi prices chứa Inf.
    Inf tạo ra H=NaN silent, không crash, không có cảnh báo.
    """
    import pytest
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(200))
    rw_inf = rw.copy()
    rw_inf[50] = np.inf
    with pytest.raises(ValueError, match="prices chứa Inf"):
        compute_ghe(rw_inf, max_lag=5)
