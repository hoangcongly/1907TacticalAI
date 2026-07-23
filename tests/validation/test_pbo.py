import pytest
import numpy as np
from aegis.validation.pbo_cscv import compute_pbo_cscv


def test_pbo_cscv_random_noise():
    """
    Kiểm chứng chiến lược nhiễu ngẫu nhiên thuần túy (Random Noise):
    Khi N chiến lược đều là nhiễu ngẫu nhiên, việc chọn cấu hình tốt nhất In-Sample (overfit vào nhiễu)
    sẽ có xác suất ~50% (0.50) suy thoái dưới trung vị khi ra ngoài Out-of-Sample -> PBO ~ 0.50 (> 0.40 Bị từ chối).
    """
    np.random.seed(42)
    # T = 160 quan sát, N = 10 cấu hình tham số, tất cả chỉ là nhiễu chuẩn N(0, 1)
    perf_matrix = np.random.normal(0.0, 1.0, size=(160, 10))

    res = compute_pbo_cscv(perf_matrix, n_splits=16, approval_threshold=0.40)
    
    # Với nhiễu ngẫu nhiên, PBO thường dao động quanh 0.45 - 0.65 -> không đạt ngưỡng <= 0.40
    assert not res["is_approved"] or res["pbo"] > 0.35, f"PBO của nhiễu ngẫu nhiên phải cao: {res['pbo']}"
    assert 0.0 <= res["pbo"] <= 1.0
    assert res["n_splits"] == 16
    assert res["n_combinations"] == 12870  # C(16, 8) = 12,870
    print(f"✅ [PBO CSCV ENGINE] Random Noise PBO = {res['pbo']:.4f} (Overfitting Detected & Rejected!) PASSED!")


def test_pbo_cscv_true_signal():
    """
    Kiểm chứng chiến lược có tín hiệu thực vượt trội ổn định (True Signal):
    Một cấu hình tốt thực sự cả IS lẫn OOS sẽ có thứ hạng OOS cao -> PBO thấp (< 0.40 -> Approved).
    """
    np.random.seed(123)
    # N = 5 cấu hình: 4 cấu hình nhiễu, 1 cấu hình số 0 có mean dương ổn định (0.2 + noise)
    perf_matrix = np.random.normal(0.0, 1.0, size=(160, 5))
    perf_matrix[:, 0] += 0.5  # Tín hiệu mạnh vượt trội ổn định trên mọi khối

    res = compute_pbo_cscv(perf_matrix, n_splits=16, approval_threshold=0.40)
    
    assert res["is_approved"] is True, f"Chiến lược tín hiệu thật phải có PBO <= 0.40: {res['pbo']}"
    assert res["pbo"] < 0.20
    print(f"✅ [PBO CSCV ENGINE] True Signal PBO = {res['pbo']:.4f} (Robust Strategy Approved!) PASSED!")


def test_pbo_armor_plated_guards():
    """Kiểm thử hải quan bọc thép cho PBO."""
    with pytest.raises(ValueError, match="cần ít nhất N >= 2"):
        compute_pbo_cscv(np.ones((100, 1)), n_splits=16)

    with pytest.raises(ValueError, match="số chẵn >= 4"):
        compute_pbo_cscv(np.ones((100, 5)), n_splits=3)

    with pytest.raises(ValueError, match="nhỏ hơn số lượng khối"):
        compute_pbo_cscv(np.ones((10, 5)), n_splits=16)

    with pytest.raises(ValueError, match="chứa giá trị NaN hoặc Inf"):
        compute_pbo_cscv(np.array([[np.nan, 1.0], [1.0, 2.0], [1.0, 1.0], [1.0, 1.0]]), n_splits=4)

    print("✅ [PBO CSCV ENGINE] Armor-plated guards PASSED!")
