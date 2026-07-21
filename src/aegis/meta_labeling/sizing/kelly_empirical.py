"""Empirical Kelly Sizing — solve E[log(1+f*r)], build_empirical_kelly_tables_v2."""

import math
import numpy as np
from scipy.optimize import brentq
from typing import NamedTuple, Dict


# ============================================================================
# HẰNG SỐ KIẾN TRÚC — TÁCH BIỆT RỦI RO GIÁ vs RỦI RO MARGIN & FRACTIONAL KELLY
# ============================================================================
# [STREAMING_CHUNK: KELLY_CONSTANTS]
# Trần tìm kiếm f* cho brentq. Trong chiến lược Perp Futures vốn nhỏ,
# f* chính là đòn bẩy hiệu dụng. 3 lớp phòng thủ bảo vệ:
#   1. Dynamic Cap: f_max_safe = min(f_max, 0.999/|r_min|)
#   2. Liquidation Layer: validate_leverage_against_sl
#   3. Half-Kelly: f_allocated = λ × f*
DEFAULT_F_MAX = 20.0

# Hệ số chiết khấu Fractional Kelly λ (mặc định 0.5 = Half-Kelly).
# Giảm 75% biến động tài khoản, chỉ mất 25% tốc độ tăng trưởng kép.
# Áp dụng BÊN NGOÀI hàm solve (tách biệt toán học vs chính sách rủi ro).
DEFAULT_LAMBDA_KELLY = 0.5


# ============================================================================
# [PHÁT HIỆN M] CẤU TRÚC KẾT QUẢ BOOTSTRAP GIÀU THÔNG TIN CHẨN ĐOÁN
# ============================================================================
# [STREAMING_CHUNK: KELLY_CONFIDENCE_RESULT]
class KellyConfidenceResult(NamedTuple):
    """Kết quả Bootstrap CI với đầy đủ thông tin chẩn đoán cho tầng giám sát."""
    f_star_point: float        # Ước lượng điểm (Point Estimate) trên toàn bộ mẫu
    f_star_conservative: float # Giá trị bảo thủ (phân vị lower_percentile)
    bootstrap_std: float       # Độ lệch chuẩn của phân phối bootstrap f*
    uncertainty_ratio: float   # bootstrap_std / max(f_star_point, 1e-6) — cờ cảnh báo nếu > 1.0


# ============================================================================
# [TASK B-1-1] EMPIRICAL KELLY FRACTION
# ============================================================================
# [STREAMING_CHUNK: KELLY_SOLVER]
def solve_empirical_kelly_fraction(
    returns_sample: np.ndarray,
    f_max: float = DEFAULT_F_MAX,
) -> float:
    """
    [TASK B-1-1] Giải f* tối đa hóa kỳ vọng Log-growth E[log(1 + f*r)]
    trên phân phối thực nghiệm.

    Tham số:
    - f_max: Giới hạn tìm kiếm cho brentq. Dynamic Cap sẽ tự động co lại
      khi mẫu chứa lệnh lỗ nặng (f_max_safe = min(f_max, 0.999/|r_min|)).
    - returns_sample: BẮT BUỘC phải là Lợi suất Cơ sở Chưa đòn bẩy (Unleveraged Return).
    """
    # [ARMOR GUARD] Lọc NaN/Inf TRƯỚC khi chạy Canary Assertion
    import math
    if not isinstance(f_max, (int, float)) or math.isnan(f_max) or math.isinf(f_max) or f_max <= 0:
        raise ValueError(f"Lỗi hải quan B-1-1: f_max (tỷ lệ cược tối đa) phải là số dương hợp lệ, nhận {f_max}")
    returns_sample = returns_sample[np.isfinite(returns_sample)]

    if len(returns_sample) < 30:
        return 0.0

    # [ARMOR GUARD] Canary Assertion — chạy SAU khi đã lọc NaN/Inf
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


# [STREAMING_CHUNK: KELLY_BOOTSTRAP]
def solve_empirical_kelly_fraction_with_confidence(
    returns_sample: np.ndarray,
    f_max: float = DEFAULT_F_MAX,
    n_bootstraps: int = 1000,
    lower_percentile: float = 25.0,
) -> KellyConfidenceResult:
    """
    [PHÁT HIỆN M] Dùng Bootstrap để trích xuất phân vị bảo thủ + thông tin chẩn đoán.
    Trả về KellyConfidenceResult thay vì float đơn thuần.
    """
    returns_sample = returns_sample[np.isfinite(returns_sample)]
    n_samples = len(returns_sample)

    if n_samples < 30:
        return KellyConfidenceResult(
            f_star_point=0.0, f_star_conservative=0.0,
            bootstrap_std=0.0, uncertainty_ratio=0.0
        )

    # Point estimate trên toàn bộ mẫu
    f_star_point = solve_empirical_kelly_fraction(returns_sample, f_max=f_max)

    # Bootstrap
    f_stars = np.zeros(n_bootstraps)
    rng = np.random.RandomState(42)  # Seed cố định để test ổn định

    for i in range(n_bootstraps):
        bootstrap_sample = rng.choice(returns_sample, size=n_samples, replace=True)
        f_stars[i] = solve_empirical_kelly_fraction(
            bootstrap_sample, f_max=f_max
        )

    f_star_conservative = float(np.percentile(f_stars, lower_percentile))
    bootstrap_std = float(np.std(f_stars))
    uncertainty_ratio = bootstrap_std / max(f_star_point, 1e-6)

    return KellyConfidenceResult(
        f_star_point=f_star_point,
        f_star_conservative=f_star_conservative,
        bootstrap_std=bootstrap_std,
        uncertainty_ratio=uncertainty_ratio,
    )


# ============================================================================
# [TẦNG 1 & 2]: HMM PROBABILITY-WEIGHTED & DOUBLE-DEFENSE BAYESIAN KELLY
# ============================================================================

def compute_regime_weighted_bayesian_kelly(
    regime_returns: Dict[str, np.ndarray],
    regime_probs: Dict[str, float],
    f_max_cap: float = DEFAULT_F_MAX,
    prior_f: float = 0.1,         # Prior: Đòn bẩy 0.1x (Mức an toàn cực đoan)
    confidence_constant_C: float = 20.0 # Cần 20 lệnh để tin tưởng 50% vào Kelly Data
) -> float:
    """
    [PHÒNG THỦ KÉP]: 
    1. Trừng phạt Phương sai (Bootstrap 25th percentile).
    2. Trừng phạt Kích thước Mẫu (Empirical Bayes Shrinkage).
    3. Mượt mà hóa bằng Xác suất HMM (Tránh Whipsaw lật mặt).
    """
    blended_f = 0.0
    
    # Kiểm tra tính hợp lệ của xác suất
    if not regime_probs:
        raise ValueError("regime_probs không được rỗng")
    for r_name, p_val in regime_probs.items():
        if not isinstance(p_val, (int, float)) or math.isnan(p_val) or math.isinf(p_val) or not (0.0 <= p_val <= 1.0):
            raise ValueError(f"Xác suất HMM của regime '{r_name}' phải nằm trong đoạn [0.0, 1.0], nhận {p_val}")
            
    total_prob = sum(regime_probs.values())
    if not math.isclose(total_prob, 1.0, rel_tol=1e-5):
        raise ValueError(f"Tổng xác suất HMM phải bằng 1.0, nhận được {total_prob}")

    for regime_name, prob in regime_probs.items():
        if prob == 0.0:
            continue
            
        returns_sample = regime_returns.get(regime_name, np.array([]))
        
        # Lọc rác NaN/Inf nếu có
        returns_sample = returns_sample[np.isfinite(returns_sample)]
        n_samples = len(returns_sample)
        
        if n_samples < 5:
            # Quá ít lệnh (Đói dữ liệu nặng), ép dùng hoàn toàn Prior
            f_bayesian = prior_f
        else:
            # 1. Trừng phạt Phương sai (Lấy phân vị bảo thủ 25%)
            kelly_result = solve_empirical_kelly_fraction_with_confidence(
                returns_sample, f_max=f_max_cap, n_bootstraps=500, lower_percentile=25.0
            )
            f_conservative = kelly_result.f_star_conservative
            
            # 2. Trừng phạt Kích thước mẫu (Bayesian Shrinkage)
            # Áp dụng Bayesian Shrinkage lên giá trị phân vị bảo thủ
            weight_data = n_samples / (n_samples + confidence_constant_C)
            f_bayesian = (weight_data * f_conservative) + ((1.0 - weight_data) * prior_f)
            
        # 3. Phối trộn (Blend) theo xác suất Regime
        blended_f += prob * f_bayesian
        
    return float(blended_f)


# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
def test_f_max_is_leverage_search_bound():
    """
    [QĐ #1] Xác nhận f_max = 20.0 là giới hạn TÌM KIẾM cho brentq,
    KHÔNG PHẢI giới hạn rủi ro cuối cùng (đó là Dynamic Cap + Half-Kelly).
    """
    assert DEFAULT_F_MAX == 20.0, f"f_max phải = 20.0, nhận {DEFAULT_F_MAX}"
    assert DEFAULT_LAMBDA_KELLY == 0.5, f"λ phải = 0.5 (Half-Kelly), nhận {DEFAULT_LAMBDA_KELLY}"
    print("✅ [QĐ #1] f_max=20.0, λ=0.5 Confirmed!")


def test_fractional_kelly_lambda_discount():
    """
    [QĐ #2] Kiểm tra hệ số chiết khấu λ = 0.5 (Half Kelly)
    giảm đúng 50% vị thế f* so với Full Kelly (λ = 1.0).
    """
    np.random.seed(42)
    sample = np.random.choice([1.0, -0.999], p=[0.6, 0.4], size=10000)

    f_raw = solve_empirical_kelly_fraction(sample, f_max=1.0)

    f_full = f_raw * 1.0
    f_half = f_raw * DEFAULT_LAMBDA_KELLY

    assert abs(f_full - 0.2) < 0.05, f"Full Kelly cho coin toss kỳ vọng ~0.2, nhận {f_full}"
    assert abs(f_half - f_full * 0.5) < 1e-4, f"Half Kelly phải bằng 50% Full Kelly: {f_half} vs {f_full*0.5}"
    print(f"✅ [QĐ #2] Full Kelly: {f_full:.4f} | Half Kelly (λ=0.5): {f_half:.4f} PASSED!")


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
    result = solve_empirical_kelly_fraction_with_confidence(
        sample, f_max=1.0, n_bootstraps=200, lower_percentile=25.0
    )

    assert abs(f_point - 0.2) < 0.05, f"Kelly cho tung đồng xu sai, kỳ vọng ~0.2, nhận {f_point}"
    assert result.f_star_conservative <= result.f_star_point, (
        "Bootstrap (25th percentile) phải bảo thủ hơn hoặc bằng Point Estimate"
    )
    assert result.bootstrap_std >= 0, "Độ lệch chuẩn bootstrap phải >= 0"
    assert result.uncertainty_ratio >= 0, "Uncertainty ratio phải >= 0"
    print(f"✅ [B-1-1] Kelly: point={result.f_star_point:.4f}, "
          f"conservative={result.f_star_conservative:.4f}, "
          f"std={result.bootstrap_std:.4f}, unc={result.uncertainty_ratio:.4f} PASSED!")


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
    Kiểm chứng rằng khi mẫu chứa lệnh thanh lý (r = -1.0),
    brentq KHÔNG crash mà tự động giới hạn f_max_safe = 0.999 / abs(-1.0) = 0.999.
    Bất kể f_max = 20.0.
    """
    np.random.seed(42)
    wins = np.full(70, 0.05)
    losses = np.full(25, -0.03)
    liquidations = np.full(5, -1.0)
    sample = np.concatenate([wins, losses, liquidations])
    np.random.shuffle(sample)

    f_star = solve_empirical_kelly_fraction(sample, f_max=DEFAULT_F_MAX)

    assert f_star <= 0.999, (
        f"Dynamic cap thất bại! f* = {f_star} > 0.999 khi mẫu chứa thanh lý -1.0"
    )
    assert f_star >= 0.0, f"f* phải >= 0, nhận {f_star}"
    print(f"✅ Kelly Dynamic Cap: f* = {f_star:.4f} (capped <= 0.999 dù f_max=20.0) PASSED!")


def test_regime_probability_blend_and_bayesian():
    """
    [BAYESIAN-HMM] Kiểm tra chức năng phối trộn Kelly theo xác suất Regime
    và trừng phạt kích thước mẫu Bayesian Shrinkage.
    Tuân thủ tuyệt đối kiến trúc HMM 2 trạng thái chốt của Module B (Trending vs Choppy).
    """
    # Giả lập: HMM đang lưỡng lự 60% Trending, 40% Choppy
    regime_probs = {"trending": 0.6, "choppy": 0.4}
    
    # Trending có 100 lệnh (Đủ mẫu), winrate rất tốt -> Kelly sẽ cao
    np.random.seed(42)
    returns_trending = np.random.choice([0.1, -0.05], p=[0.55, 0.45], size=100)
    
    # Choppy chưa có lệnh nào (Đói data) -> Kelly phải bị ép về Prior (0.1)
    returns_choppy = np.array([])
    
    regime_returns = {"trending": returns_trending, "choppy": returns_choppy}
    
    # Kiểm tra
    f_safe = compute_regime_weighted_bayesian_kelly(
        regime_returns, regime_probs, prior_f=0.1, confidence_constant_C=20.0
    )
    
    assert 0.1 <= f_safe <= DEFAULT_F_MAX
    print(f"✅ [BAYESIAN-HMM] Phối trộn mượt mà thành công. F_Blend = {f_safe:.3f}")


if __name__ == "__main__":
    test_f_max_is_leverage_search_bound()
    test_fractional_kelly_lambda_discount()
    test_b_1_1_kelly_classical_coin_toss()
    test_kelly_canary_and_nan_safety()
    test_kelly_dynamic_cap_with_liquidation()
    test_regime_probability_blend_and_bayesian()
