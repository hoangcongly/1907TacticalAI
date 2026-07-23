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
    Kiểm chứng GHE bỏ qua NaN khi tính moments.
    """
    np.random.seed(42)
    rw = np.cumsum(np.random.randn(500))
    # Gắn NaN vào một số điểm
    rw[100:110] = np.nan
    rw[250:260] = np.nan
    
    h_nan = compute_ghe(rw, max_lag=10)
    assert 0.40 < h_nan < 0.60, "GHE không xử lý được NaN!"
