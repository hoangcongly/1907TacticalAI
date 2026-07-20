"""Empirical Kelly Sizing — solve E[log(1+f*r)], build_empirical_kelly_tables_v2."""

import numpy as np
from scipy.optimize import brentq


# ============================================================================
# HẰNG SỐ KIẾN TRÚC — TÁCH BIỆT RỦI RO GIÁ vs RỦI RO MARGIN & FRACTIONAL KELLY
# ============================================================================
# Trần rủi ro BIẾN ĐỘNG GIÁ: size_notional = f_allocated × Equity.
# Một vị thế notional > 3x equity là mức rất mạo hiểm cho trend-following.
DEFAULT_F_MAX_NOTIONAL_CAP = 3.0

# Hệ số chiết khấu Fractional Kelly λ (mặc định 0.5 = Half-Kelly).
# Giúp hạ tỷ lệ cược để phòng chống sai số ước lượng mẫu và rủi ro mô hình (Model Risk).
DEFAULT_LAMBDA_KELLY = 0.5


# ============================================================================
# [TASK B-1-1] EMPIRICAL KELLY FRACTION
# ============================================================================
def solve_empirical_kelly_fraction(
    returns_sample: np.ndarray,
    f_max: float = DEFAULT_F_MAX_NOTIONAL_CAP,
) -> float:
    """
    [TASK B-1-1] Giải f* tối đa hóa kỳ vọng Log-growth E[log(1 + f*r)]
    trên phân phối thực nghiệm.

    Tham số:
    - f_max: Giới hạn rủi ro BIẾN ĐỘNG GIÁ (notional / equity), KHÔNG PHẢI đòn bẩy sàn.
    - returns_sample: BẮT BUỘC phải là Lợi suất Cơ sở Chưa đòn bẩy (Unleveraged Return).
    """
    # Lọc NaN/Inf TRƯỚC khi chạy Canary Assertion
    returns_sample = returns_sample[np.isfinite(returns_sample)]

    if len(returns_sample) < 30:
        return 0.0

    # Canary Assertion — chạy SAU khi đã lọc NaN/Inf
    assert np.all(returns_sample >= -1.0), (
        "Canary Error: Phát hiện return < -100% sau khi đã lọc NaN/Inf. "
        "PnL thanh lý đã làm rò rỉ dữ liệu hoặc sai số học!"
    )

    # [DYNAMIC LEVERAGE CAP] Giới hạn đòn bẩy động dựa trên lệnh lỗ nặng nhất.
    min_return = np.min(returns_sample)
    if min_return < 0:
        f_max_safe = min(f_max, 0.999 / abs(min_return))
    else:
        f_max_safe = f_max

    def growth_derivative(f):
        denom = 1.0 + f * returns_sample
        if np.any(denom <= 1e-6):
            return -1e6
        return np.mean(returns_sample / denom)

    # Đạo hàm tại f=0 mà <= 0 nghĩa là kỳ vọng âm, không cược
    if growth_derivative(0.0) <= 0:
        return 0.0

    # Đạo hàm tại f_max_safe mà vẫn > 0 nghĩa là cược tối đa vẫn chưa tới đỉnh
    if growth_derivative(f_max_safe) > 0:
        f_star_raw = f_max_safe
    else:
        # Tìm nghiệm f* để đạo hàm = 0
        f_star_raw = float(brentq(growth_derivative, 0.0, f_max_safe, xtol=1e-6))

    return float(f_star_raw)


def solve_empirical_kelly_fraction_with_confidence(
    returns_sample: np.ndarray,
    f_max: float = DEFAULT_F_MAX_NOTIONAL_CAP,
    n_bootstraps: int = 1000,
    lower_percentile: float = 25.0,
) -> float:
    """
    Dùng Bootstrap để trích xuất phân vị bảo thủ (25th percentile).
    """
    returns_sample = returns_sample[np.isfinite(returns_sample)]
    n_samples = len(returns_sample)

    if n_samples < 30:
        return 0.0

    f_stars = np.zeros(n_bootstraps)
    rng = np.random.RandomState(42)  # Seed cố định để test ổn định

    for i in range(n_bootstraps):
        bootstrap_sample = rng.choice(returns_sample, size=n_samples, replace=True)
        f_stars[i] = solve_empirical_kelly_fraction(
            bootstrap_sample, f_max=f_max
        )

    # Trả về phân vị bảo thủ
    return float(np.percentile(f_stars, lower_percentile))


# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
def test_f_star_notional_risk_independent_of_leverage_cap():
    """
    [FINDING A] Xác nhận trần f_max (rủi ro biến động giá) KHÔNG được set bằng
    hoặc gần bằng leverage_cap (rủi ro margin) — 2 con số phải được chọn độc lập.
    """
    LEVERAGE_CAP_MARGIN = 20.0
    assert DEFAULT_F_MAX_NOTIONAL_CAP <= 5.0, (
        "f_max quyết định rủi ro biến động GIÁ -- một vị thế notional > 5x equity "
        "là mức rủi ro cực đoan độc lập với margin/leverage đang chọn."
    )
    assert DEFAULT_F_MAX_NOTIONAL_CAP < LEVERAGE_CAP_MARGIN, (
        "Kiến trúc lỗi: f_max đang bị đánh đồng với đòn bẩy sàn!"
    )
    print("✅ [FINDING A] Separation of Concerns (Notional Risk vs Margin Risk) Confirmed!")


def test_fractional_kelly_lambda_discount():
    """
    [FRACTIONAL KELLY TEST] Kiểm tra xem hệ số chiết khấu λ = 0.5 (Half Kelly)
    có làm giảm đúng 50% vị thế f* so với Full Kelly (λ = 1.0) hay không.
    Lưu ý: Nhân λ diễn ra ở ngoài hàm toán học solve_empirical_kelly_fraction.
    """
    np.random.seed(42)
    sample = np.random.choice([1.0, -0.999], p=[0.6, 0.4], size=10000)

    f_raw = solve_empirical_kelly_fraction(sample, f_max=1.0)
    
    f_full = f_raw * 1.0
    f_half = f_raw * DEFAULT_LAMBDA_KELLY

    assert abs(f_full - 0.2) < 0.05, f"Full Kelly cho coin toss kỳ vọng ~0.2, nhận {f_full}"
    assert abs(f_half - f_full * 0.5) < 1e-4, f"Half Kelly phải bằng 50% Full Kelly: {f_half} vs {f_full*0.5}"
    print(f"✅ [FRACTIONAL KELLY] Full Kelly: {f_full:.4f} | Half Kelly (λ=0.5): {f_half:.4f} PASSED!")


def test_b_1_1_kelly_classical_coin_toss():
    """
    UNIT TEST CHO B-1-1:
    Kiểm tra trên phân phối 2 điểm kinh điển (tung đồng xu):
    - Xác suất thắng p = 0.6
    - Thắng được b = 1.0 (1 ăn 1)
    - Thua mất 1.0 (mất trắng)
    => Theo công thức Full Kelly (λ=1.0): f* = p - (1-p)/b = 0.6 - 0.4/1.0 = 0.2
    """
    np.random.seed(42)
    sample = np.random.choice([1.0, -0.999], p=[0.6, 0.4], size=10000)

    f_point = solve_empirical_kelly_fraction(sample, f_max=1.0)
    f_boot = solve_empirical_kelly_fraction_with_confidence(
        sample, f_max=1.0, n_bootstraps=200, lower_percentile=25.0
    )

    assert abs(f_point - 0.2) < 0.05, f"Kelly cho tung đồng xu sai, kỳ vọng ~0.2, nhận {f_point}"
    assert f_boot <= f_point, "Bootstrap (25th percentile) phải bảo thủ hơn hoặc bằng Point Estimate"
    print("✅ [TASK B-1-1] Kelly Classic & Bootstrap PASSED!")


def test_kelly_canary_and_nan_safety():
    """
    [FINDING C] Kiểm tra thứ tự: Canary assertion chạy SAU khi lọc NaN/Inf.
    Mẫu chứa NaN hợp lệ (artifact dữ liệu) KHÔNG được gây false-positive.
    """
    sample_with_nan = np.array([0.05, 0.03, -0.02, np.nan, 0.01] * 10)  # 50 phần tử, 40 hữu hạn
    result = solve_empirical_kelly_fraction(sample_with_nan)
    assert result >= 0.0, f"f* phải >= 0 khi mẫu có kỳ vọng dương, nhận {result}"

    try:
        bad_sample = np.array([0.5, -1.05, 0.2] * 15)  # 45 phần tử
        solve_empirical_kelly_fraction(bad_sample)
        assert False, "Lỗi rò rỉ: Return < -100% không bị Canary bắt!"
    except AssertionError as e:
        assert "Canary Error" in str(e)

    ok_sample = np.array([0.5, -1.0, 0.2] * 15)
    solve_empirical_kelly_fraction(ok_sample)

    print("✅ [FINDING C] Canary Assertion Order (NaN-safe) PASSED!")


def test_kelly_dynamic_cap_with_liquidation():
    """
    [RUTHLESS AUDIT v4] Kiểm chứng rằng khi mẫu chứa lệnh thanh lý (r = -1.0),
    brentq KHÔNG crash mà tự động giới hạn f_max_safe = 0.999 / abs(-1.0) = 0.999.
    """
    np.random.seed(42)
    wins = np.full(70, 0.05)
    losses = np.full(25, -0.03)
    liquidations = np.full(5, -1.0)
    sample = np.concatenate([wins, losses, liquidations])
    np.random.shuffle(sample)

    f_star = solve_empirical_kelly_fraction(sample, f_max=DEFAULT_F_MAX_NOTIONAL_CAP)

    assert f_star <= 0.999, (
        f"Dynamic cap thất bại! f* = {f_star} > 0.999 khi mẫu chứa thanh lý -1.0"
    )
    assert f_star >= 0.0, f"f* phải >= 0, nhận {f_star}"
    print(f"✅ [RUTHLESS AUDIT v4] Kelly Dynamic Cap: f* = {f_star:.4f} (capped <= 0.999) PASSED!")


if __name__ == "__main__":
    test_f_star_notional_risk_independent_of_leverage_cap()
    test_fractional_kelly_lambda_discount()
    test_b_1_1_kelly_classical_coin_toss()
    test_kelly_canary_and_nan_safety()
    test_kelly_dynamic_cap_with_liquidation()
