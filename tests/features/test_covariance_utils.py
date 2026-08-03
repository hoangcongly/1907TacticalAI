"""
Tests for Covariance Matrix Sanitization (Task A-4-5)
"""

import numpy as np
import pytest
from aegis.features.kalman.covariance_utils import sanitize_covariance_matrix

def test_sanitize_covariance_matrix_nearly_singular():
    """
    [TASK A-4-5] Test: ma trận gần suy biến -> trả về đối xứng, PD, min-eigenvalue >= floor
    """
    # Khởi tạo ma trận hiệp phương sai 2x2 suy biến (định thức = 0, eigenvalue = 2 và 0)
    cov_singular = np.array([
        [1.0, 1.0],
        [1.0, 1.0] 
    ])
    
    # Gây nhiễu để tạo ra eigenvalue âm cực nhỏ (mô phỏng sai số floating point)
    cov_singular[1, 1] -= 1e-12
    
    floor = 1e-10
    cov_sanitized = sanitize_covariance_matrix(cov_singular, eigenvalue_floor=floor)
    
    # 1. Kiểm tra tính đối xứng (Symmetric)
    np.testing.assert_array_almost_equal(cov_sanitized, cov_sanitized.T, err_msg="Ma trận trả về không đối xứng")
    
    # 2. Kiểm tra tính Positive Definite (PD) thông qua Cholesky
    # Nếu ma trận không PD, hàm cholesky sẽ raise LinAlgError
    try:
        np.linalg.cholesky(cov_sanitized)
    except np.linalg.LinAlgError:
        pytest.fail("Ma trận không Positive Definite (Cholesky phân rã thất bại)")
        
    # 3. Kiểm tra min-eigenvalue >= floor
    eigenvalues, _ = np.linalg.eigh(cov_sanitized)
    
    # Chấp nhận sai số siêu nhỏ do tính toán ma trận (floating point precision)
    assert np.all(eigenvalues >= floor - 1e-15), f"Có eigenvalue nhỏ hơn mức sàn {floor}: {eigenvalues}"
