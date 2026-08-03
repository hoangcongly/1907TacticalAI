import pytest
import numpy as np

from aegis.data.cleaning.gap_handling import kalman_predict_only_n_steps
from aegis.data.cleaning.tick_kalman_replacer import ensure_pd_matrix_2x2_numba


def test_kalman_predict_n_steps_identity():
    """
    [Test 1] Kiểm định với F = Identity (I).
    Logic: Trạng thái x_n = x_0, Hiệp phương sai P_n = P_0 + n * Q.
    """
    x_0 = np.array([[100.0], [0.0]])
    P_0 = np.eye(2)
    F = np.eye(2)
    Q = np.array([[1e-4, 0.0], [0.0, 1e-5]])
    n_steps = 10
    
    x_n, P_n = kalman_predict_only_n_steps(x_0, P_0, F, Q, n_steps)
    
    # 1. Trạng thái đứng im
    np.testing.assert_array_almost_equal(x_n, x_0)
    
    # 2. Hiệp phương sai mở rộng tuyến tính
    expected_P_n = P_0 + n_steps * Q
    np.testing.assert_array_almost_equal(P_n, expected_P_n)


def test_kalman_predict_n_steps_llt():
    """
    [Test 2] Kiểm định với F = Local Linear Trend (LLT).
    Logic: x_n = x_0 + n * trend. P_n khớp với tính toán thủ công suy diễn tay cho n=1, 2, 3.
    """
    level_0 = 50000.0
    trend_0 = 10.0
    x_0 = np.array([[level_0], [trend_0]])
    P_0 = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[1e-4, 0.0], [0.0, 1e-5]])
    
    # Kịch bản n=1
    x_1, P_1 = kalman_predict_only_n_steps(x_0, P_0, F, Q, 1)
    expected_x_1 = np.array([[level_0 + 1 * trend_0], [trend_0]])
    P_manual_1 = ensure_pd_matrix_2x2_numba(F @ P_0 @ F.T + Q)
    np.testing.assert_array_almost_equal(x_1, expected_x_1)
    np.testing.assert_array_almost_equal(P_1, P_manual_1)

    # Kịch bản n=2
    x_2, P_2 = kalman_predict_only_n_steps(x_0, P_0, F, Q, 2)
    expected_x_2 = np.array([[level_0 + 2 * trend_0], [trend_0]])
    P_manual_2 = ensure_pd_matrix_2x2_numba(F @ P_manual_1 @ F.T + Q)
    np.testing.assert_array_almost_equal(x_2, expected_x_2)
    np.testing.assert_array_almost_equal(P_2, P_manual_2)

    # Kịch bản n=3
    x_3, P_3 = kalman_predict_only_n_steps(x_0, P_0, F, Q, 3)
    expected_x_3 = np.array([[level_0 + 3 * trend_0], [trend_0]])
    P_manual_3 = ensure_pd_matrix_2x2_numba(F @ P_manual_2 @ F.T + Q)
    np.testing.assert_array_almost_equal(x_3, expected_x_3)
    np.testing.assert_array_almost_equal(P_3, P_manual_3)


def test_kalman_predict_n_steps_armor_guard_and_performance():
    """
    [Test 3] Armor Guard PD Protection & Hiệu năng với n_steps cực lớn.
    """
    x_0 = np.array([[50000.0], [5.0]])
    # Đưa vào một P_0 suy biến nguy hiểm (det = 0)
    P_0 = np.array([[1.0, 1.0], [1.0, 1.0]]) 
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[1e-4, 0.0], [0.0, 1e-5]])
    n_steps = 100_000 # 100,000 steps mô phỏng đứt gãy 1 ngày
    
    # Hàm không được crash, Numba sẽ xử lý trong nháy mắt
    x_n, P_n = kalman_predict_only_n_steps(x_0, P_0, F, Q, n_steps)
    
    # P_n phải vượt qua bài test Cholesky để chứng minh nó Positive Definite
    try:
        np.linalg.cholesky(P_n)
    except np.linalg.LinAlgError:
        pytest.fail("Armor Guard thất bại: Ma trận P_n không Positive Definite!")


def test_kalman_predict_n_steps_input_sanitization():
    """
    Kiểm định chặn input sai (SOP Phase 2).
    """
    x_0 = np.array([[100.0], [0.0]])
    P_0 = np.eye(2)
    F = np.eye(2)
    Q = np.eye(2)
    
    # Âm bước
    with pytest.raises(ValueError):
        kalman_predict_only_n_steps(x_0, P_0, F, Q, -5)
        
    # Sai type
    with pytest.raises(TypeError):
        kalman_predict_only_n_steps(x_0, P_0, F, Q, 1.5)
        
    # Sai shape (chặn NxN)
    with pytest.raises(ValueError):
        kalman_predict_only_n_steps(np.array([100.0]), P_0, F, Q, 5)
    with pytest.raises(ValueError):
        kalman_predict_only_n_steps(x_0, np.eye(3), F, Q, 5)
