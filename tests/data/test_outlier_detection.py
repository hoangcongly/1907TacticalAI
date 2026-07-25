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
    
    print(f"[OK] [TASK A-1-1] Robust Sigma: {estimated_sigma_mean:.4f} (True: {true_sigma}) - Beat Naive Std: {naive_std:.4f}")

def test_compute_rolling_mad_empty_or_short():
    """Kiểm tra phản ứng với mảng quá ngắn."""
    prices = np.array([100.0, 101.0, 102.0])
    res = compute_rolling_mad(prices, window=10)
    assert np.isnan(res).all(), "Mảng quá ngắn phải trả về chuỗi NaN tương ứng"
    assert len(res) == 3, "Độ dài kết quả phải khớp mảng gốc"


from aegis.data.outlier_detection import detect_bad_tick_core

def test_detect_bad_tick_core_with_tail_events():
    """
    [TASK A-1-2] TDD Kiểm thử thuật toán phân loại Bad Tick và Tail Event.
    """
    n = 200
    window = 100
    prices = np.full(n, 100.0)
    volumes = np.full(n, 10.0)
    
    # Bước 1: Khởi tạo dữ liệu giả lập chuẩn
    for i in range(window):
        prices[i] = 100.0 + np.random.normal(0, 0.1)
        volumes[i] = 10.0 + np.random.normal(0, 1.0)
        
    # Chuẩn bị trước một mảng Sigma ổn định (giả định Sigma ~ 0.1)
    robust_sigmas = np.full(n, 0.15)
    
    # --- Scenario A: Bad Tick (Nhiễu chớp nhoáng) ---
    # Giá giật mạnh (ĐK1 thỏa mãn), Volume bình thường (ĐK2 thỏa mãn), Giá giật về cũ ở tick sau (ĐK3 thỏa mãn)
    idx_bad = 120
    prices[idx_bad-1] = 100.0
    prices[idx_bad] = 105.0     # Diff = 5.0 > 5 * 0.15
    volumes[idx_bad] = 12.0     # Vol = 12 < 2 * median(10)
    prices[idx_bad+1] = 100.1   # Diff tương lai (100.1 - 100) = 0.1 < 0.3 * 5.0
    
    # --- Scenario B: Tail Event (Có dòng tiền thật đẩy giá) ---
    # Giá giật mạnh (ĐK1 thỏa mãn), Volume khổng lồ (ĐK2 vi phạm -> Tail Event)
    idx_tail = 150
    prices[idx_tail-1] = 100.0
    prices[idx_tail] = 105.0    # Diff = 5.0 > 5 * 0.15
    volumes[idx_tail] = 50.0    # Vol = 50 > 2 * median(10)
    prices[idx_tail+1] = 104.9  # Không có đảo chiều (Nhưng không cần xét vì đã vi phạm ĐK2)
    
    # --- Scenario C: Bước Giá Bình Thường (Không Reversal) ---
    # Giá giật mạnh, Volume bình thường, nhưng giá TICK SAU không đảo chiều về (Không thỏa ĐK3)
    idx_norm = 180
    prices[idx_norm-1] = 100.0
    prices[idx_norm] = 105.0    # Diff = 5.0 > 5 * 0.15
    volumes[idx_norm] = 12.0    # Vol = 12 < 2 * median(10)
    prices[idx_norm+1] = 106.0  # Diff tương lai (106 - 100) = 6.0 KHÔNG nhỏ hơn 0.3 * 5.0
    
    # Thực thi Numba function
    is_bad, is_tail = detect_bad_tick_core(prices, volumes, robust_sigmas, window)
    
    # Kiểm chứng kết quả
    assert is_bad[idx_bad] == True, "Scenario A phải được gán cờ Bad Tick"
    assert is_tail[idx_bad] == False, "Scenario A không phải Tail Event"
    
    assert is_bad[idx_tail] == False, "Scenario B KHÔNG ĐƯỢC gán cờ Bad Tick vì có Volume chống lưng"
    assert is_tail[idx_tail] == True, "Scenario B phải được gán cờ Tail Event"
    
    assert is_bad[idx_norm] == False, "Scenario C không phải Bad Tick vì giá trụ lại được (Không Reversal)"
    assert is_tail[idx_norm] == False, "Scenario C không phải Tail Event"

from aegis.data.outlier_detection import detect_bad_tick_cross_venue

def test_detect_bad_tick_cross_venue():
    """
    [TASK A-1-3] Kiểm thử thuật toán Cross-Venue Parity.
    """
    n = 5
    timestamps = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    
    # Sàn phụ có dữ liệu xung quanh các timestamps
    m = 7
    ref_timestamps = np.array([500, 1500, 2500, 3200, 3800, 4500, 5500], dtype=np.int64)
    ref_prices = np.array([100.0, 101.0, 100.5, 120.0, 101.0, 102.0, 100.0])
    robust_sigmas_ref = np.full(m, 2.0)
    
    # t_i = 3000 (timestamps[2]). Cửa sổ +/- 500ms -> [2500, 3500].
    # Các tick sàn phụ trong khoảng này: index 2 (2500, p=100.5) và index 3 (3200, p=120.0).
    # Anchor point (ngay trước hoặc tại 3000): index 2 (2500). Giá anchor = 100.5, sigma = 2.0.
    # Trong cửa sổ [2500, 3500], max_move là abs(120.0 - 100.5) = 19.5.
    # Ngưỡng eta * sigma = 2.0 * 2.0 = 4.0.
    # 19.5 KHÔNG nhỏ hơn 4.0 -> condition4_satisfied = False (Không phải Bad Tick, xác nhận Tail Event).
    
    res = detect_bad_tick_cross_venue(
        timestamps, ref_timestamps, ref_prices, robust_sigmas_ref,
        window_ms=500, eta_confirm=2.0
    )
    
    assert res[2] == False, "Sàn phụ giật mạnh (19.5 > 4.0) -> phải trả về False (Xác nhận sự kiện thật)"
    
    # t_i = 4000 (timestamps[3]). Cửa sổ [3500, 4500].
    # Tick phụ trong khoảng: index 4 (3800, p=101.0), index 5 (4500, p=102.0).
    # Anchor: index 4 (3800). Giá anchor = 101.0.
    # max_move = abs(102.0 - 101.0) = 1.0.
    # Ngưỡng 4.0. 1.0 < 4.0 -> condition4_satisfied = True (Xác nhận Bad Tick).
    
    assert res[3] == True, "Sàn phụ đứng yên (1.0 < 4.0) -> phải trả về True (Xác nhận Bad Tick)"
    
def test_detect_bad_tick_cross_venue_missing_data():
    """Kiểm tra fallback khi thiếu dữ liệu sàn phụ."""
    timestamps = np.array([1000, 2000, 3000], dtype=np.int64)
    # Cố tình để mảng rỗng
    ref_timestamps = np.array([], dtype=np.int64)
    ref_prices = np.array([])
    robust_sigmas_ref = np.array([])
    
    res = detect_bad_tick_cross_venue(timestamps, ref_timestamps, ref_prices, robust_sigmas_ref)
    
    # Fallback mặc định là False (thiên về giữ Tail Event)
    assert not res.any(), "Khi thiếu dữ liệu sàn phụ, tất cả phải Fallback về False"

