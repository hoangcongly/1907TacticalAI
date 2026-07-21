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
    - returns_sample: Lợi suất Cơ sở Hiệu dụng (Effective Unleveraged Payoff).
      LƯU Ý: Với lệnh bị thanh lý (Liquidation), giá trị này BẮT BUỘC phải là
      -1/leverage. Đây KHÔNG phải là lỗi rò rỉ dữ liệu, mà là hệ quả toán học
      phản ánh chính xác 100% cú sốc tài sản lên Equity khi mất trắng Margin
      (Loss = -f/leverage).
    """
    # [ARMOR GUARD] Lọc NaN/Inf TRƯỚC khi chạy Canary Assertion
    returns_sample = returns_sample[np.isfinite(returns_sample)]

    if len(returns_sample) < 30:
        return 0.0

    # [ARMOR GUARD] Canary Assertion — chạy SAU khi đã lọc NaN/Inf
    # LƯU Ý: Lệnh thanh lý sẽ có payoff = -1.0/leverage, do đó >= -1.0 là an toàn tuyệt đối.
    assert np.all(returns_sample >= -1.0), (
        "Canary Error: Phát hiện return < -100% sau khi đã lọc NaN/Inf. "
        "Với thị trường Spot/Perp, lợi suất cơ sở hiệu dụng (Effective Unleveraged Payoff) "
        "kể cả khi thanh lý cũng chỉ giới hạn ở mức -1/leverage (>= -100%). "
        "Lỗi này chỉ xảy ra do sai số học hoặc truyền nhầm dữ liệu ĐÃ nhân đòn bẩy!"
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
    total_prob = sum(regime_probs.values())
    if not math.isclose(total_prob, 1.0, rel_tol=1e-5):
        raise ValueError(f"Tổng xác suất HMM phải bằng 1.0, nhận được {total_prob}")

    for regime_name, prob in regime_probs.items():
        if prob <= 0.0:
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



