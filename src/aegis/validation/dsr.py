"""
[v11.9] Deflated Sharpe Ratio (DSR >= 0.95) & Sensitivity Analysis Engine.
Đánh giá tỷ lệ Sharpe chiết khấu số lần thử nghiệm (Multiple Testing Discount) theo Bailey & Lopez de Prado (2014).
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats


def euler_mascheroni_approx_max_sr(
    sr_benchmark: float,
    num_trials: int,
    variance_of_srs: float
) -> float:
    """
    Xấp xỉ kỳ vọng giá trị lớn nhất của Sharpe Ratio dưới giả thuyết Null (SR_0^*)
    từ N lần thử nghiệm độc lập (num_trials) với phương sai variance_of_srs.
    """
    if num_trials <= 1 or variance_of_srs <= 0.0:
        return float(sr_benchmark)

    gamma = 0.5772156649015328606  # Hằng số Euler-Mascheroni
    std_sr = math.sqrt(variance_of_srs)

    # Dùng chuẩn norm.ppf của scipy để xấp xỉ chính xác theo phân phối chuẩn
    try:
        z1 = stats.norm.ppf(1.0 - 1.0 / num_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (num_trials * math.e))
        expected_max = sr_benchmark + std_sr * ((1.0 - gamma) * z1 + gamma * z2)
    except Exception:
        # Fallback closed-form nếu num_trials cực lớn hoặc ppf tràn
        log_n = math.log(num_trials)
        expected_max = sr_benchmark + std_sr * (math.sqrt(2.0 * log_n) - (math.log(math.pi * log_n) + 2.0 * math.log(2.0)) / (2.0 * math.sqrt(2.0 * log_n)))

    return float(expected_max)


def compute_probabilistic_sharpe_ratio(
    sr_estimated: float,
    sr_benchmark: float,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0
) -> float:
    """
    Tính Probabilistic Sharpe Ratio (PSR) — Xác suất SR ước lượng vượt qua SR_benchmark
    dựa trên phân phối tiệm cận của Sharpe Ratio (có xét độ lệch Skewness & độ nhọn Kurtosis).
    """
    if sample_length <= 1:
        raise ValueError(f"Lỗi hải quan DSR: sample_length phải > 1, nhận {sample_length}")

    # Mẫu số: chuẩn hóa độ lệch chuẩn của Sharpe Ratio ước lượng
    denom_sq = 1.0 - skewness * sr_estimated + ((kurtosis - 1.0) / 4.0) * (sr_estimated ** 2)
    if denom_sq <= 1e-12:
        denom_sq = 1e-12

    std_sr_asymp = math.sqrt(denom_sq / (sample_length - 1.0))
    z_score = (sr_estimated - sr_benchmark) / std_sr_asymp

    psr = float(stats.norm.cdf(z_score))
    return psr


def compute_deflated_sharpe_ratio(
    sr_estimated: float,
    variance_of_srs: float,
    sample_length: int,
    num_trials: int = 1,
    sr_benchmark: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    approval_threshold: float = 0.95,
    use_experiment_tracker: bool = False,
) -> Dict[str, Any]:
    """
    [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
    - Chặn sample_length <= 1, num_trials < 1, variance_of_srs < 0.
    - Chặn NaN / Inf cho mọi tham số đầu vào.
    
    Kết quả trả về:
    - dsr: Giá trị Deflated Sharpe Ratio (0.0 đến 1.0).
    - sr_expected_max: Ngưỡng SR kỳ vọng tối đa do Multiple Testing (SR_0^*).
    - is_approved: True nếu dsr >= approval_threshold (mặc định 0.95).
    """
    # Hải quan kiểm tra
    if not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError(f"Lỗi hải quan DSR: num_trials phải là số nguyên >= 1, nhận {num_trials}")

    if not isinstance(sample_length, int) or sample_length <= 1:
        raise ValueError(f"Lỗi hải quan DSR: sample_length phải là số nguyên > 1, nhận {sample_length}")

    for name, val in [
        ("sr_estimated", sr_estimated),
        ("variance_of_srs", variance_of_srs),
        ("sr_benchmark", sr_benchmark),
        ("skewness", skewness),
        ("kurtosis", kurtosis),
        ("approval_threshold", approval_threshold)
    ]:
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Lỗi hải quan DSR: {name} không hợp lệ (NaN/Inf), nhận {val}")

    if variance_of_srs < 0.0:
        raise ValueError(f"Lỗi hải quan DSR: variance_of_srs không được âm, nhận {variance_of_srs}")

    # ====================================================================
    # [FIX F17] SỐ TRIALS PHẢI TƯỜNG MINH — KHÔNG ĐỌC TRẠNG THÁI TOÀN CỤC
    # ====================================================================
    # Bản cũ gọi ExperimentTracker().get_total_trials() (đếm số DÒNG trong
    # logs/experiments/*.jsonl) rồi lấy max() với num_trials của caller.
    # Ba vấn đề nghiêm trọng:
    #   1. DSR KHÔNG TÁI LẬP ĐƯỢC: cùng input cho kết quả khác nhau tuỳ máy đã
    #      chạy bao nhiêu thí nghiệm. Clone mới và máy cũ ra hai con số khác hẳn.
    #   2. Âm thầm ghi đè tham số caller truyền vào.
    #   3. Số dòng log KHÔNG phải số trial — mỗi lần chạy pipeline ghi nhiều dòng.
    # Hình phạt multiple-testing vẫn đúng về mặt lý thuyết, nhưng phải do caller
    # cung cấp TƯỜNG MINH qua `num_trials` (hoặc bật `use_experiment_tracker`).
    effective_num_trials = max(int(num_trials), 1)
    if use_experiment_tracker:
        from aegis.core.experiment_tracker import ExperimentTracker

        effective_num_trials = max(
            effective_num_trials, int(ExperimentTracker().get_total_trials()), 1
        )

    # Bước 1: Tính kỳ vọng SR tối đa (SR_0^*) từ N lần thử nghiệm
    sr_expected_max = euler_mascheroni_approx_max_sr(sr_benchmark, effective_num_trials, variance_of_srs)

    # Bước 2: Tính PSR với SR_benchmark = sr_expected_max
    dsr = compute_probabilistic_sharpe_ratio(
        sr_estimated=sr_estimated,
        sr_benchmark=sr_expected_max,
        sample_length=sample_length,
        skewness=skewness,
        kurtosis=kurtosis
    )

    return {
        "dsr": float(dsr),
        "sr_expected_max": float(sr_expected_max),
        "sr_estimated": float(sr_estimated),
        "num_trials": int(effective_num_trials),
        "is_approved": bool(dsr >= approval_threshold)
    }


def compute_dsr_sensitivity(
    sr_estimated: float,
    variance_of_srs: float,
    sample_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_trials_list: Optional[List[int]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    [MODULE E SENSITIVITY DIRECTIVE]:
    Thực hiện phân tích độ nhạy của Deflated Sharpe Ratio (DSR) với các mức số lượng thử nghiệm N.
    Mặc định theo chỉ thị là N = [30, 100, 200].
    
    Khi N tăng lên (nhiều cấu hình được backtest hơn), SR_expected_max tăng lên,
    khiến DSR bị chiết khấu mạnh hơn (độ tin cậy chống quá khớp khắt khe hơn).
    """
    if n_trials_list is None:
        n_trials_list = [30, 100, 200]

    sensitivity_results = {}
    for n_trials in n_trials_list:
        res = compute_deflated_sharpe_ratio(
            sr_estimated=sr_estimated,
            variance_of_srs=variance_of_srs,
            sample_length=sample_length,
            num_trials=n_trials,
            skewness=skewness,
            kurtosis=kurtosis
        )
        sensitivity_results[n_trials] = res

    return sensitivity_results
