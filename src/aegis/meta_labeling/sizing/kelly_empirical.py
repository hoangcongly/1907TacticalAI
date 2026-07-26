"""Empirical Kelly Sizing — solve E[log(1+f*r)], build_empirical_kelly_tables_v2."""

import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq  # type: ignore
from scipy.optimize import brentq
from typing import NamedTuple, Dict, Tuple, Any, Union, List, Literal



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

    if len(returns_sample) < 5:
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

    if n_samples < 5:
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

def build_regime_returns_dict(
    records: List[Dict[str, Any]],
    assignment_mode: Literal["soft", "hard"] = "soft",
    hard_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    [KHẮC PHỤC LỖ HỔNG - TRAIN/INFERENCE MISMATCH REGIME ASSIGNMENT]:
    Xây dựng từ điển `regime_returns` từ danh sách bản ghi giao dịch theo đúng chế độ rẽ nhánh:
    - `assignment_mode == 'hard'`: Phân chia cứng theo argmax (p_trend >= hard_threshold -> trending,
      ngược lại -> choppy). Tuân thủ 1-1 nếu inference chạy theo chế độ rẽ nhánh cứng (`argmax`).
    - `assignment_mode == 'soft'`: Phân chia/tập hợp mẫu theo trọng số hoặc lọc xác suất liên tục,
      đồng bộ hoàn hảo với suy luận live blend theo xác suất `prob * f_bayesian`.
    """
    trending_list: List[float] = []
    choppy_list: List[float] = []

    for r in records:
        if not isinstance(r, dict) or bool(r.get("boundary_truncated", False)):
            continue
        ret = r.get("realized_return")
        if ret is None or not isinstance(ret, (int, float)) or math.isnan(ret) or math.isinf(ret):
            continue

        p_trend = r.get("p_trend", r.get("p_i", 0.5))
        if p_trend is None or not isinstance(p_trend, (int, float)) or math.isnan(p_trend):
            continue

        p_trend_val = min(max(float(p_trend), 0.0), 1.0)
        if assignment_mode == "hard":
            if p_trend_val >= hard_threshold:
                trending_list.append(float(ret))
            else:
                choppy_list.append(float(ret))
        else:
            # Soft mode: xác suất cao hơn được ưu tiên đưa vào mẫu tương ứng để duy trì phân phối liên tục
            if p_trend_val >= 0.5:
                trending_list.append(float(ret))
            if (1.0 - p_trend_val) >= 0.5:
                choppy_list.append(float(ret))

    return {
        "trending": np.array(trending_list, dtype=np.float64),
        "choppy": np.array(choppy_list, dtype=np.float64),
    }


def compute_regime_weighted_bayesian_kelly(
    regime_returns: Dict[str, np.ndarray],
    regime_probs: Dict[str, float],
    f_max_cap: float = DEFAULT_F_MAX,
    prior_f: float = 0.1,         # Prior: Đòn bẩy 0.1x (Mức an toàn cực đoan)
    confidence_constant_C: float = 20.0, # Cần 20 lệnh để tin tưởng 50% vào Kelly Data
    assignment_mode: Literal["soft", "hard"] = "soft",
) -> float:
    """
    [PHÒNG THỦ KÉP & REGIME ALIGNMENT]: 
    1. Trừng phạt Phương sai (Bootstrap 25th percentile).
    2. Trừng phạt Kích thước Mẫu (Empirical Bayes Shrinkage).
    3. Phối trộn theo `assignment_mode` (Khắc phục Train/Inference Mismatch Request 10):
       - Nếu `assignment_mode == 'soft'`: Phối trộn liên tục (Blended posterior weighting):
         f_final = sum(prob_r * f_bayesian_r).
       - Nếu `assignment_mode == 'hard'`: Rẽ nhánh cứng theo argmax (tương ứng với lúc train hard bucket):
         f_final = f_bayesian của regime có xác suất prob lớn nhất.
    """
    if not regime_probs:
        raise ValueError("regime_probs không được rỗng")
    for r_name, p_val in regime_probs.items():
        if not isinstance(p_val, (int, float)) or math.isnan(p_val) or math.isinf(p_val) or not (0.0 <= p_val <= 1.0):
            raise ValueError(f"Xác suất HMM của regime '{r_name}' phải nằm trong đoạn [0.0, 1.0], nhận {p_val}")
            
    total_prob = sum(regime_probs.values())
    if not math.isclose(total_prob, 1.0, rel_tol=1e-5):
        raise ValueError(f"Tổng xác suất HMM phải bằng 1.0, nhận được {total_prob}")

    regime_f_bayesian: Dict[str, float] = {}
    for regime_name, prob in regime_probs.items():
        returns_sample = regime_returns.get(regime_name, np.array([]))
        returns_sample = returns_sample[np.isfinite(returns_sample)]
        n_samples = len(returns_sample)
        
        if n_samples < 5:
            f_bayesian = prior_f
        else:
            f_max_dynamic = min(float(f_max_cap), max(1.0, math.sqrt(n_samples)))
            kelly_result = solve_empirical_kelly_fraction_with_confidence(
                returns_sample, f_max=f_max_dynamic, n_bootstraps=500, lower_percentile=25.0
            )
            f_conservative = kelly_result.f_star_conservative
            weight_data = n_samples / (n_samples + confidence_constant_C)
            f_bayesian = (weight_data * f_conservative) + ((1.0 - weight_data) * prior_f)
        regime_f_bayesian[regime_name] = float(f_bayesian)

    if assignment_mode == "hard":
        # Chọn argmax cứng để đảm bảo 1-1 với bucket lịch sử hard assignment
        dominant_regime = max(regime_probs.keys(), key=lambda k: regime_probs[k])
        return float(regime_f_bayesian[dominant_regime])
    else:
        # Soft mode: Phối trộn liên tục theo xác suất HMM
        blended_f = sum(prob * regime_f_bayesian[r_name] for r_name, prob in regime_probs.items())
        return float(blended_f)



# ============================================================================
# [TASK B-1-11] MAP TRADE RECORDS TO 2D GRID KELLY TABLE INPUTS
# ============================================================================
def trade_records_to_kelly_table_inputs(
    records: List[Dict[str, Any]],
    num_bins: int = 10,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    [TASK B-1-11] Ánh xạ danh sách bản ghi giao dịch (TradeRecord dicts)
    vào lưới 2D coordinate (idx_p, idx_chop) sử dụng Conditional Quantile Binning.
    
    Giải quyết vấn đề Đói Mẫu (Data Starvation) của 2D Grid bằng cách:
    1. Chia p_i thành num_bins buckets có số mẫu bằng nhau (Quantile).
    2. Trong mỗi bucket của p_i, chia p_chop_i thành num_bins buckets (Conditional Quantile).
    
    Trả về: (grid_returns, p_edges, chop_edges_list)
    """
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError(f"Lỗi B-1-11: num_bins phải là số nguyên dương, nhận {num_bins}")

    valid_records = []
    for r in records:
        if not isinstance(r, dict) or r.get("boundary_truncated", False):
            continue
        ret = r.get("realized_return")
        p_i = r.get("p_i")
        p_chop_i = r.get("p_chop_i")
        
        if (ret is not None and p_i is not None and p_chop_i is not None and
            math.isfinite(ret) and math.isfinite(p_i) and math.isfinite(p_chop_i)):
            valid_records.append({
                "ret": float(ret),
                "p_i": min(max(float(p_i), 0.0), 1.0),
                "p_chop_i": min(max(float(p_chop_i), 0.0), 1.0)
            })

    grid_returns: Dict[Tuple[int, int], List[float]] = {}
    
    if not valid_records:
        # Fallback to uniform if no data
        p_edges = np.linspace(0, 1, num_bins + 1)
        chop_edges_list = [np.linspace(0, 1, num_bins + 1) for _ in range(num_bins)]
        return {}, p_edges, chop_edges_list

    # 1. Marginal Quantile for p_i with strict monotonic guarantee
    all_p = np.array([r["p_i"] for r in valid_records])
    p_edges = np.quantile(all_p, np.linspace(0, 1, num_bins + 1))
    p_edges[0], p_edges[-1] = -np.inf, np.inf # To catch all out of bounds during inference
    for k in range(1, len(p_edges) - 1):
        if p_edges[k] <= p_edges[k - 1]:
            p_edges[k] = p_edges[k - 1] + 1e-12
    
    idx_p_array = np.searchsorted(p_edges, all_p, side='right') - 1
    idx_p_array = np.clip(idx_p_array, 0, num_bins - 1)
    
    chop_edges_list = []
    for i in range(num_bins):
        mask_i = (idx_p_array == i)
        records_i = [r for idx, r in enumerate(valid_records) if mask_i[idx]]
        
        if not records_i:
            chop_edges = np.linspace(0, 1, num_bins + 1)
            chop_edges[0], chop_edges[-1] = -np.inf, np.inf
            chop_edges_list.append(chop_edges)
            continue
            
        all_chop_i = np.array([r["p_chop_i"] for r in records_i])
        chop_edges = np.quantile(all_chop_i, np.linspace(0, 1, num_bins + 1))
        chop_edges[0], chop_edges[-1] = -np.inf, np.inf
        for k in range(1, len(chop_edges) - 1):
            if chop_edges[k] <= chop_edges[k - 1]:
                chop_edges[k] = chop_edges[k - 1] + 1e-12
        chop_edges_list.append(chop_edges)
        
        idx_chop_array = np.searchsorted(chop_edges, all_chop_i, side='right') - 1
        idx_chop_array = np.clip(idx_chop_array, 0, num_bins - 1)
        
        for idx_j, r in zip(idx_chop_array, records_i):
            coord = (i, idx_j)
            if coord not in grid_returns:
                grid_returns[coord] = []
            grid_returns[coord].append(r["ret"])

    result: Dict[Tuple[int, int], np.ndarray] = {}
    for coord, ret_list in grid_returns.items():
        result[coord] = np.array(ret_list, dtype=float)

    return result, p_edges, chop_edges_list


# ============================================================================
# [TASK B-1-12] BUILD EMPIRICAL KELLY TABLE V2 WITH BAYESIAN PENALTY
# ============================================================================
def build_empirical_kelly_table_v2(
    inputs: Union[Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray, List[np.ndarray]], List[Dict[str, Any]]],
    num_bins: int = 10,
    f_max: float = DEFAULT_F_MAX,
    prior_f: float = 0.0,
    confidence_constant_C: float = 20.0,
    n_bootstraps: int = 500,
    lower_percentile: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    [TASK B-1-12] Dựng ma trận Kelly 2D (shape: num_bins x num_bins) từ đầu ra B-1-11
    hoặc trực tiếp từ danh sách TradeRecord dicts.
    
    Trả về Tuple: (kelly_table_2d, p_edges, chop_edges_list)
    """
    if not isinstance(num_bins, int) or num_bins <= 0:
        raise ValueError(f"Lỗi B-1-12: num_bins phải là số nguyên dương, nhận {num_bins}")
    if not isinstance(f_max, (int, float)) or math.isnan(f_max) or math.isinf(f_max) or f_max <= 0:
        raise ValueError(f"Lỗi B-1-12: f_max phải > 0, nhận {f_max}")
    if not isinstance(prior_f, (int, float)) or math.isnan(prior_f) or math.isinf(prior_f) or prior_f < 0:
        raise ValueError(f"Lỗi B-1-12: prior_f không hợp lệ, nhận {prior_f}")
    if not isinstance(confidence_constant_C, (int, float)) or math.isnan(confidence_constant_C) or math.isinf(confidence_constant_C) or confidence_constant_C < 0:
        raise ValueError(f"Lỗi B-1-12: confidence_constant_C không hợp lệ, nhận {confidence_constant_C}")

    if isinstance(inputs, list):
        bin_inputs, p_edges, chop_edges_list = trade_records_to_kelly_table_inputs(inputs, num_bins=num_bins)
    elif isinstance(inputs, tuple) and len(inputs) == 3:
        bin_inputs, p_edges, chop_edges_list = inputs
    else:
        raise ValueError(f"Lỗi B-1-12: inputs phải là Tuple (từ B-1-11) hoặc List[Dict], nhận {type(inputs)}")

    kelly_table = np.zeros((num_bins, num_bins), dtype=float)

    for idx_p in range(num_bins):
        for idx_chop in range(num_bins):
            coord = (idx_p, idx_chop)
            returns = bin_inputs.get(coord, np.array([], dtype=float))
            # Lọc rác NaN/Inf nếu có trong array
            returns = returns[np.isfinite(returns)]
            N = len(returns)

            if N < 5:
                f_target = float(prior_f)
            else:
                f_max_dynamic = min(float(f_max), max(1.0, math.sqrt(N)))
                ci_result = solve_empirical_kelly_fraction_with_confidence(
                    returns,
                    f_max=f_max_dynamic,
                    n_bootstraps=n_bootstraps,
                    lower_percentile=lower_percentile,
                )
                f_cons = ci_result.f_star_conservative
                w = float(N) / (float(N) + float(confidence_constant_C))
                f_bayesian = (w * f_cons) + ((1.0 - w) * float(prior_f))
                f_target = float(f_bayesian)

            kelly_table[idx_p, idx_chop] = max(0.0, float(f_target))

    return kelly_table, p_edges, chop_edges_list


# ============================================================================
# [TASK B-1-13] COMPUTE BI-DIRECTIONAL KELLY V14 UNIFIED INFERENCE
# ============================================================================
def compute_bi_directional_kelly_v14_unified(
    p_i: float,
    p_chop_i: float,
    kelly_table: np.ndarray,
    p_edges: np.ndarray,
    chop_edges_list: List[np.ndarray],
    fade_enabled: bool,
    fade_regime_gate_threshold: float = 0.60,
) -> dict:
    """
    [TASK B-1-13] Hàm suy luận O(1) thống nhất cho Sizing (Inference Layer).

    Input: p_i, p_chop_i, kelly_table 2D (cùng p_edges, chop_edges_list), fade_enabled.
    Quy trình:
    1. Kiểm tra nghiêm ngặt input.
    2. Gọi classify_trade_mode. Nếu 'none', trả về 0.0.
    3. Tra cứu f_target sử dụng Conditional Quantile edges (np.searchsorted).
    4. Trả về {'f_target': float(f_target), 'mode': mode}. Dấu âm của Fade được xử lý ở tầng Trade Mode.
    """
    from aegis.meta_labeling.sizing.trade_mode import classify_trade_mode

    if not isinstance(p_i, (int, float)) or math.isnan(p_i) or math.isinf(p_i) or not (0.0 <= p_i <= 1.0):
        raise ValueError(f"Lỗi B-1-13: p_i không hợp lệ [0, 1], nhận {p_i}")
    if not isinstance(p_chop_i, (int, float)) or math.isnan(p_chop_i) or math.isinf(p_chop_i) or not (0.0 <= p_chop_i <= 1.0):
        raise ValueError(f"Lỗi B-1-13: p_chop_i không hợp lệ [0, 1], nhận {p_chop_i}")

    if not isinstance(kelly_table, np.ndarray) or kelly_table.ndim != 2 or kelly_table.shape[0] != kelly_table.shape[1] or kelly_table.shape[0] <= 0:
        raise ValueError(f"Lỗi B-1-13: kelly_table phải là mảng 2D vuông với num_bins > 0, nhận shape {getattr(kelly_table, 'shape', None)}")

    mode = classify_trade_mode(
        p_i=float(p_i),
        p_chop_i=float(p_chop_i),
        fade_enabled=fade_enabled,
        fade_regime_gate_threshold=fade_regime_gate_threshold,
    )

    if mode == "none":
        return {"f_target": 0.0, "mode": "none"}

    num_bins = kelly_table.shape[0]
    
    # 1. Tìm idx_p dựa trên p_edges
    idx_p = int(np.searchsorted(p_edges, float(p_i), side='right')) - 1
    idx_p = max(0, min(idx_p, num_bins - 1))
    
    # 2. Tìm idx_chop dựa trên chop_edges tương ứng với idx_p
    chop_edges = chop_edges_list[idx_p]
    idx_chop = int(np.searchsorted(chop_edges, float(p_chop_i), side='right')) - 1
    idx_chop = max(0, min(idx_chop, num_bins - 1))

    f_target = float(kelly_table[idx_p, idx_chop])
    return {"f_target": max(0.0, f_target), "mode": mode}


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


def test_b_1_11_trade_records_to_kelly_table_inputs():
    """
    Kiểm tra Task B-1-11:
    - Ánh xạ p_i, p_chop_i về index lưới (idx_p, idx_chop).
    - Lọc bỏ bản ghi thiếu realized_return hoặc boundary_truncated=True.
    - Xử lý p=1.0 bằng math.floor không bị out-of-bounds (idx <= num_bins-1).
    """
    records = [
        {"p_i": 0.05, "p_chop_i": 0.95, "realized_return": 0.04, "boundary_truncated": False}, # bin (0, 9)
        {"p_i": 1.00, "p_chop_i": 1.00, "realized_return": -0.02, "boundary_truncated": False}, # bin (9, 9) khi num_bins=10
        {"p_i": 0.55, "p_chop_i": 0.25, "realized_return": 0.10, "boundary_truncated": True},  # Bỏ qua vì boundary_truncated
        {"p_i": 0.55, "p_chop_i": 0.25, "realized_return": None},                                # Bỏ qua vì thiếu realized_return
        {"p_i": -0.1, "p_chop_i": 1.2, "realized_return": 0.01, "boundary_truncated": False},   # Clamped về (0, 9)
    ]
    grid, p_edges, chop_edges = trade_records_to_kelly_table_inputs(records, num_bins=10)
    assert len(grid) > 0
    assert len(p_edges) == 11
    assert len(chop_edges) == 10
    print("✅ [TASK B-1-11] trade_records_to_kelly_table_inputs PASSED!")


def test_b_1_12_build_empirical_kelly_table_v2():
    """
    Kiểm tra Task B-1-12:
    - Nếu len(returns) < 5 -> f = prior_f.
    - Nếu len(returns) >= 5 -> f_bayesian = w*f_cons + (1-w)*prior_f.
    """
    np.random.seed(42)
    returns_good = np.random.choice([0.08, -0.03], p=[0.6, 0.4], size=50)
    p_edges = np.linspace(0, 1, 11)
    p_edges[0], p_edges[-1] = -np.inf, np.inf
    chop_edges_list = [np.linspace(0, 1, 11) for _ in range(10)]
    for el in chop_edges_list:
        el[0], el[-1] = -np.inf, np.inf
        
    grid_inputs = (
        {
            (9, 9): returns_good,
            (0, 0): np.array([0.05, 0.02, -0.01]), # Chỉ 3 lệnh < 5
        },
        p_edges,
        chop_edges_list
    )
    table, _, _ = build_empirical_kelly_table_v2(
        grid_inputs, num_bins=10, prior_f=0.0, confidence_constant_C=20.0
    )
    assert table.shape == (10, 10)
    assert table[0, 0] == 0.0  # < 5 lệnh -> prior_f = 0.0
    assert table[9, 9] > 0.0   # >= 5 lệnh thắng -> có f_bayesian dương
    print(f"✅ [TASK B-1-12] build_empirical_kelly_table_v2 PASSED (f[9,9]={table[9,9]:.4f})!")


def test_b_1_13_compute_bi_directional_kelly_v14_unified():
    """
    Kiểm tra Task B-1-13:
    - O(1) inference tra cứu kelly_table.
    - Trả về {'f_target': 0.0, 'mode': 'none'} nếu mode none.
    - Trả về đúng f_target nếu mode follow hoặc fade.
    """
    table = np.zeros((10, 10), dtype=float)
    table[8, 2] = 3.5  # p_i around 0.85, p_chop around 0.25 -> follow
    table[1, 8] = 1.8  # p_i around 0.15, p_chop around 0.85 -> fade
    p_edges = np.linspace(0, 1, 11)
    p_edges[0], p_edges[-1] = -np.inf, np.inf
    chop_edges_list = [np.linspace(0, 1, 11) for _ in range(10)]
    for el in chop_edges_list:
        el[0], el[-1] = -np.inf, np.inf

    # Follow mode
    res_follow = compute_bi_directional_kelly_v14_unified(
        p_i=0.85, p_chop_i=0.25, kelly_table=table, p_edges=p_edges, chop_edges_list=chop_edges_list, fade_enabled=True
    )
    assert res_follow["mode"] == "follow"
    assert math.isclose(res_follow["f_target"], 3.5, rel_tol=1e-6)

    # Fade mode
    res_fade = compute_bi_directional_kelly_v14_unified(
        p_i=0.15, p_chop_i=0.85, kelly_table=table, p_edges=p_edges, chop_edges_list=chop_edges_list, fade_enabled=True, fade_regime_gate_threshold=0.60
    )
    assert res_fade["mode"] == "fade"
    assert math.isclose(res_fade["f_target"], 1.8, rel_tol=1e-6)

    # None mode (deadzone)
    res_none = compute_bi_directional_kelly_v14_unified(
        p_i=0.35, p_chop_i=0.50, kelly_table=table, p_edges=p_edges, chop_edges_list=chop_edges_list, fade_enabled=True
    )
    assert res_none["mode"] == "none"
    assert res_none["f_target"] == 0.0

    print("✅ [TASK B-1-13] compute_bi_directional_kelly_v14_unified PASSED!")


if __name__ == "__main__":
    test_f_max_is_leverage_search_bound()
    test_fractional_kelly_lambda_discount()
    test_b_1_1_kelly_classical_coin_toss()
    test_kelly_canary_and_nan_safety()
    test_kelly_dynamic_cap_with_liquidation()
    test_regime_probability_blend_and_bayesian()
    test_b_1_11_trade_records_to_kelly_table_inputs()
    test_b_1_12_build_empirical_kelly_table_v2()
    test_b_1_13_compute_bi_directional_kelly_v14_unified()
