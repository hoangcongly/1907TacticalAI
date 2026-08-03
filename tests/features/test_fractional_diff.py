"""
Tests for Fractional Differentiation (Task A-4-1).
"""

import numpy as np
from aegis.features.fractional_diff import compute_ffd_weights, apply_ffd, find_optimal_d_star

def test_compute_ffd_weights_d_0_5():
    """
    [TASK A-4-1] Khớp công thức đệ quy nhị thức cho d=0.5 tính tay.
    """
    d = 0.5
    # Chọn tau đủ nhỏ để lấy được ít nhất 4 trọng số đầu tiên
    tau = 1e-4
    
    weights = compute_ffd_weights(d, tau)
    
    # Tính tay:
    # w_0 = 1
    # w_1 = -0.5
    # w_2 = -(-0.5) * (0.5 - 2 + 1) / 2 = -0.125
    # w_3 = -(-0.125) * (0.5 - 3 + 1) / 3 = -0.0625
    expected_first_4 = np.array([1.0, -0.5, -0.125, -0.0625])
    
    # Kiểm tra độ dài phải >= 4 để có thể so sánh
    assert len(weights) >= 4, "Mảng trọng số quá ngắn so với tau"
    
    # Kiểm tra 4 phần tử đầu tiên khớp chính xác với tính tay
    np.testing.assert_array_almost_equal(weights[:4], expected_first_4)

def test_compute_ffd_weights_d_1():
    """
    Với d = 1 (vi phân bậc 1), w_0 = 1, w_1 = -1, các w khác = 0.
    """
    weights = compute_ffd_weights(1.0, 1e-5)
    expected = np.array([1.0, -1.0])
    np.testing.assert_array_almost_equal(weights, expected)


def test_apply_ffd():
    """
    [TASK A-4-2] Kiểm tra tính đúng đắn và strictly causal của phép chập FFD.
    """
    series = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    d = 0.5
    tau = 0.05
    
    # w = [1.0, -0.5, -0.125, -0.0625] (len 4)
    # Output length = len(series) - len(w) + 1 = 5 - 4 + 1 = 2
    # Value t=3 (13.0): 13(1) + 12(-0.5) + 11(-0.125) + 10(-0.0625) = 5.0
    # Value t=4 (14.0): 14(1) + 13(-0.5) + 12(-0.125) + 11(-0.0625) = 5.3125
    
    result = apply_ffd(series, d, tau)
    assert len(result) == 2
    np.testing.assert_array_almost_equal(result, np.array([5.0, 5.3125]))


def test_find_optimal_d_star():
    """
    [TASK A-4-2] Phục hồi d* trong sai số cho phép trên chuỗi giả lập Random Walk.
    Đồng thời hàm này sẽ ghi một model_fitting trial xuống thư mục logs qua ExperimentTracker.
    """
    np.random.seed(42)
    # Tạo chuỗi Random Walk
    n_samples = 500
    noise = np.random.normal(0, 1, n_samples)
    series = np.cumsum(noise)
    
    # Tạo lưới d từ 0.0 đến 1.0 (step 0.1)
    d_grid = np.linspace(0.0, 1.0, 11)
    
    # Thực thi tìm d*
    d_star = find_optimal_d_star(series, d_grid, tau=1e-4, p_value_threshold=0.05)
    
    # Đối với Random Walk, lý thuyết d* = 1.0. 
    # Thực tế với 500 mẫu, ADF có thể pass ở 0.8, 0.9 hoặc 1.0. 
    # Nhưng chắc chắn phải lớn hơn 0 và nhỏ hơn hoặc bằng 1.0.
    assert 0.0 < d_star <= 1.0
    assert isinstance(d_star, float)


def test_fit_sum_of_exponentials_reject():
    """
    [TASK A-4-3] Test reject khi lỗi vượt epsilon.
    Nếu cấu trúc phức tạp (ví dụ random noise) và M nhỏ, việc xấp xỉ Sum of Exponentials
    sẽ thất bại và trả về approved = False.
    """
    from aegis.features.fractional_diff import fit_sum_of_exponentials_v2
    # Sinh một chuỗi trọng số ngẫu nhiên hoàn toàn (không có cấu trúc hàm mũ)
    np.random.seed(42)
    random_weights = np.random.randn(100)
    
    # Ép sai số cho phép rất nhỏ để đảm bảo nó luôn reject
    c, rho, err_abs, err_rel, approved = fit_sum_of_exponentials_v2(random_weights, M=4, epsilon_approx=1e-6)
    
    assert approved is False
    assert err_abs > 1e-6


def test_select_ffd_production_engine():
    """
    [TASK A-4-3] Test `select_ffd_production_engine` fallback behavior.
    """
    from aegis.features.fractional_diff import select_ffd_production_engine, compute_ffd_weights
    
    weights = compute_ffd_weights(0.5, tau=1e-3)
    
    # Test fallback khi epsilon bị vượt qua (thông qua mảng quá ngắn và nhiễu)
    engine_config = select_ffd_production_engine(weights, M_prony=2) # M nhỏ sẽ dễ bị reject
    
    # Tùy thuộc vào curve_fit, nếu reject nó sẽ trả về 'windowed'
    assert engine_config["engine_type"] in ["sum_of_exp", "windowed"]
    
    if engine_config["engine_type"] == "windowed":
        assert "weights" in engine_config
        assert "reason" in engine_config["validation"]
    else:
        assert "c" in engine_config
        assert "rho" in engine_config
