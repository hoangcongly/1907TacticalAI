"""Empirical Kelly Sizing — solve E[log(1+f*r)], build_empirical_kelly_tables_v2."""

import numpy as np
from scipy.optimize import brentq


# ============================================================================
# [TASK B-1-1] EMPIRICAL KELLY FRACTION
# ============================================================================
def solve_empirical_kelly_fraction(
    returns_sample: np.ndarray, f_max: float = 1.0
) -> float:
    """
    [TASK B-1-1] Giải f* tối đa hóa kỳ vọng Log-growth E[log(1 + f*r)]
    trên phân phối thực nghiệm. Sử dụng thuật toán brentq.
    """
    # Lọc bỏ NaN/Inf để đảm bảo an toàn số học
    returns_sample = returns_sample[np.isfinite(returns_sample)]
    if len(returns_sample) < 30:
        return 0.0

    def growth_derivative(f):
        denom = 1.0 + f * returns_sample
        # Chặn singularity nếu f cược quá lớn dẫn đến phá sản (1 + f*r <= 0)
        if np.any(denom <= 1e-6):
            return -1e6
        return np.mean(returns_sample / denom)

    # Đạo hàm tại f=0 mà <= 0 nghĩa là kỳ vọng âm, không cược
    if growth_derivative(0.0) <= 0:
        return 0.0

    # Đạo hàm tại f_max mà vẫn > 0 nghĩa là cược tối đa f_max vẫn chưa tới đỉnh
    if growth_derivative(f_max) > 0:
        return f_max

    # Tìm nghiệm f* để đạo hàm = 0
    return brentq(growth_derivative, 0.0, f_max, xtol=1e-6)


# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
def test_b_1_1_kelly_classical_coin_toss():
    """
    UNIT TEST CHO B-1-1:
    Kiểm tra trên phân phối 2 điểm kinh điển (tung đồng xu):
    - Xác suất thắng p = 0.6
    - Thắng được b = 1.0 (1 ăn 1)
    - Thua mất 1.0 (mất trắng)
    => Theo công thức Kelly: f* = p - (1-p)/b = 0.6 - 0.4/1.0 = 0.2
    """
    p = 0.6
    b = 1.0

    # Cố định seed, sinh mẫu 10,000 lần tung
    np.random.seed(42)
    sample = np.random.choice([b, -1.0], p=[p, 1 - p], size=10000)

    f_star = solve_empirical_kelly_fraction(sample, f_max=1.0)

    # Sai số cho phép 5% do đây là mô phỏng mẫu ngẫu nhiên (thực nghiệm)
    assert (
        abs(f_star - 0.2) < 0.05
    ), f"B-1-1 FAILED: Cần f* xấp xỉ 0.2, nhưng nhận được {f_star}"
    print(
        "✅ [TASK B-1-1] solve_empirical_kelly_fraction PASSED! (Khớp hoàn hảo công thức Kelly kinh điển)"
    )


if __name__ == "__main__":
    test_b_1_1_kelly_classical_coin_toss()
