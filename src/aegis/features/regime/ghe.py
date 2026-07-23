"""
Generalized Hurst Exponent (GHE) tính toán mức độ Persistent của chuỗi giá.
H = 0.5: Random Walk
H > 0.5: Trending (Persistent)
H < 0.5: Mean Reverting (Anti-persistent)
"""
import numpy as np

def compute_ghe(prices: np.ndarray, max_lag: int = 10, q: int = 1) -> float:
    """
    Tính Generalized Hurst Exponent bậc q.
    Dựa trên quan hệ: E[|P(t+tau) - P(t)|^q] ~ tau^(q*H)
    """
    n = len(prices)
    if n < max_lag + 2:
        return 0.5 # Mặc định Random walk nếu mẫu quá bé

    lags = np.arange(1, max_lag + 1)
    moments = np.zeros(max_lag)

    for i, tau in enumerate(lags):
        # Tính sai phân bậc tau
        diffs = np.abs(prices[tau:] - prices[:-tau])
        
        # Bỏ qua các giá trị NaN trong quá trình tính
        valid_diffs = diffs[~np.isnan(diffs)]
        if len(valid_diffs) == 0:
            moments[i] = np.nan
        else:
            moments[i] = np.mean(valid_diffs ** q)

    # Loại bỏ các lag tính ra NaN hoặc 0 (để log không bị lỗi)
    valid_mask = (~np.isnan(moments)) & (moments > 0)
    if np.sum(valid_mask) < 2:
        return 0.5
        
    x = np.log(lags[valid_mask])
    y = np.log(moments[valid_mask])
    
    # Hồi quy tuyến tính y = H*q * x + c
    # slope = H * q
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    
    H = slope / q
    return float(H)
