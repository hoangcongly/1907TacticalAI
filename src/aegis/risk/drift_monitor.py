"""
drift_monitor.py — CUSUM Brier drift monitoring & refresh_cusum_thresholds (Task B-2-1 / Module J).

Quy chuẩn và Giải quyết Xung đột Đặt tên Artifact (Namespace Protocol):
- `refresh_cusum_thresholds`: Cập nhật định kỳ các ngưỡng lọc CUSUM biến động cho GIÁ (Price CUSUM)
  sau mỗi đợt `production_fit`. Tính toán thống kê và lưu trữ kết quả cấu hình chuẩn hóa vào file JSON
  với không gian tên riêng (`price_cusum_thresholds.json`) trong thư mục /artifacts theo đúng quy chuẩn
  giao thức bàn giao nhị phân sang Rust RTK (Phần VI).
- `monitor_brier_score_cusum_drift`: Theo dõi độ trôi sai số dự báo (Brier score drift) theo
  thuật toán Page (1954), tự động reset về 0 khi vọt ngưỡng báo động. Cấu hình ngưỡng giám sát trôi
  được xuất độc lập ra `brier_drift_cusum_thresholds.json`, tuyệt đối không ghi đè cấu hình giá.
"""

import json
import math
from pathlib import Path
from typing import Optional, Union
import numpy as np
from aegis.labeling.cusum_events import compute_dynamic_cusum_thresholds


# ============================================================================
# [TASK B-2-1] REFRESH CUSUM THRESHOLDS FOR PRODUCTION FIT / RTK EXPORT
# ============================================================================
def refresh_cusum_thresholds(
    prices: Union[list[float], np.ndarray],
    atr_series: Union[list[float], np.ndarray],
    base_multiplier: float = 2.5,
    anchor_span: int = 50,
    min_rel_threshold: float = 1e-4,
    max_rel_threshold: float = 0.05,
    artifact_output_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Tính toán và tái tạo cấu hình ngưỡng CUSUM sau production_fit (Task B-2-1).

    Trả về dictionary thông tin chuẩn hóa và tùy chọn lưu ra file JSON
    (chuẩn hóa không gian tên: `artifacts/price_cusum_thresholds.json`).
    """
    thresholds = compute_dynamic_cusum_thresholds(
        prices=prices,
        atr_series=atr_series,
        base_multiplier=base_multiplier,
        anchor_span=anchor_span,
        min_rel_threshold=min_rel_threshold,
        max_rel_threshold=max_rel_threshold,
    )

    thresholds_np = np.asarray(thresholds, dtype=np.float64)
    tail_sample = [float(x) for x in thresholds_np[-10:]]

    config_output = {
        "base_multiplier": float(base_multiplier),
        "anchor_span": int(anchor_span),
        "min_rel_threshold": float(min_rel_threshold),
        "max_rel_threshold": float(max_rel_threshold),
        "current_last_threshold": float(thresholds_np[-1]),
        "summary_stats": {
            "mean": float(np.mean(thresholds_np)),
            "std": float(np.std(thresholds_np)),
            "min": float(np.min(thresholds_np)),
            "max": float(np.max(thresholds_np)),
            "median": float(np.median(thresholds_np)),
        },
        "thresholds_sample_tail": tail_sample,
    }

    if artifact_output_path is not None:
        out_path = Path(artifact_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config_output, f, indent=4, ensure_ascii=False)

    return config_output


# ============================================================================
# [TASK B-2-1 / MODULE J] BRIER SCORE DRIFT MONITORING WITH PAGE RESET
# ============================================================================
def monitor_brier_score_cusum_drift(
    brier_scores: Union[list[float], np.ndarray],
    baseline_brier: float,
    drift_threshold: float = 0.05,
    reset_on_alarm: bool = True,
    artifact_output_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Theo dõi độ trôi sai số Brier Score bằng bộ lọc CUSUM (Page 1954).
    Khi S_t^+ > drift_threshold: kích hoạt cảnh báo trôi mô hình (Model Drift Alarm)
    và tự động reset accumulators về 0.

    Có tùy chọn xuất trạng thái và ngưỡng theo dõi ra file JSON chuyên biệt
    (chuẩn hóa không gian tên: `artifacts/brier_drift_cusum_thresholds.json`),
    đảm bảo không bị xung đột với `price_cusum_thresholds.json` tại tầng Rust RTK.
    """
    if not isinstance(brier_scores, (list, tuple, np.ndarray)):
        raise ValueError("brier_scores phải là mảng hoặc list.")
    scores_np = np.asarray(brier_scores, dtype=np.float64)
    if scores_np.ndim != 1 or len(scores_np) == 0:
        raise ValueError("brier_scores phải là mảng 1D không rỗng.")
    if not isinstance(baseline_brier, (int, float)) or baseline_brier < 0 or math.isnan(baseline_brier):
        raise ValueError(f"baseline_brier không hợp lệ: {baseline_brier}")
    if not isinstance(drift_threshold, (int, float)) or drift_threshold <= 0 or math.isnan(drift_threshold):
        raise ValueError(f"drift_threshold phải là số thực dương: {drift_threshold}")

    if np.any(np.isnan(scores_np)) or np.any(np.isinf(scores_np)) or np.any(scores_np < 0) or np.any(scores_np > 1.0):
        raise ValueError("brier_scores chứa giá trị rác NaN/Inf hoặc ngoài đoạn [0, 1].")

    s_plus = 0.0
    alarms = []
    accumulated_series = []

    for idx, score in enumerate(scores_np):
        excess_error = score - baseline_brier
        s_plus = max(0.0, s_plus + excess_error)
        accumulated_series.append(float(s_plus))

        if s_plus > drift_threshold:
            alarms.append({
                "bar_idx": int(idx),
                "brier_score": float(score),
                "s_plus_at_trigger": float(s_plus),
            })
            if reset_on_alarm:
                s_plus = 0.0

    result = {
        "alarms_triggered": len(alarms) > 0,
        "alarms_count": len(alarms),
        "alarms_details": alarms,
        "final_s_plus": float(s_plus),
        "accumulated_series": accumulated_series,
        "drift_threshold": float(drift_threshold),
        "baseline_brier": float(baseline_brier),
    }

    if artifact_output_path is not None:
        out_path = Path(artifact_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

    return result
