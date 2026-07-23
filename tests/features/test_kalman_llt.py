import numpy as np
from aegis.features.kalman.local_linear_trend import LocalLinearTrendKalman

def test_kalman_llt_recovers_trend_direction():
    """
    Kiểm chứng Kalman LLT phục hồi đúng dấu của xu hướng 
    trên dữ liệu tổng hợp (synthetic) sạch.
    """
    kf = LocalLinearTrendKalman(process_noise_level=1e-4, process_noise_trend=1e-5, observation_noise=1e-2)
    
    # 1. Tạo chuỗi giá có xu hướng tăng rõ rệt
    n = 100
    # y = 100 + 0.5 * t
    prices_up = 100.0 + 0.5 * np.arange(n)
    _, trends_up = kf.filter_series(prices_up)
    
    # Cuối chuỗi, trend phải dương và xấp xỉ 0.5
    assert trends_up[-1] > 0.0
    assert np.isclose(trends_up[-1], 0.5, atol=0.1)

    # 2. Tạo chuỗi giá có xu hướng giảm rõ rệt
    # y = 100 - 0.3 * t
    prices_down = 100.0 - 0.3 * np.arange(n)
    _, trends_down = kf.filter_series(prices_down)
    
    # Cuối chuỗi, trend phải âm và xấp xỉ -0.3
    assert trends_down[-1] < 0.0
    assert np.isclose(trends_down[-1], -0.3, atol=0.1)

def test_kalman_llt_gap_handling():
    """
    Kiểm chứng Gap-handling predict-only:
    Khi gặp dữ liệu bị khuyết (NaN), hệ thống ngoại suy tiếp (predict)
    chứ không bị lỗi hay đứng hình (đóng băng).
    """
    kf = LocalLinearTrendKalman(process_noise_level=1e-4, process_noise_trend=1e-5, observation_noise=1e-2)
    
    prices = 100.0 + 1.0 * np.arange(20)
    # Cố tình đục lỗ ở giữa
    prices[10:15] = np.nan
    
    levels, trends = kf.filter_series(prices)
    
    # Ở index 14 (đang bị NaN), level phải được ngoại suy tiếp lên khoảng 114
    # Trend vẫn giữ nguyên chiều dương (khoảng 1.0)
    assert not np.isnan(levels[14])
    assert not np.isnan(trends[14])
    
    assert trends[14] > 0.5
    # Level được ngoại suy theo trend
    assert levels[14] > levels[9]
