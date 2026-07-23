"""
[v11.9] Probability of Backtest Overfitting (PBO <= 0.40) & Combinatorial Symmetric Cross-Validation (CSCV).
Thực hiện chia S sub-blocks đối xứng, đánh giá xác suất chiến lược tối ưu in-sample bị suy thoái out-of-sample dưới mức trung vị.
"""

import math
import itertools
from typing import Any, Dict, List, Optional
import numpy as np


def compute_pbo_cscv(
    performance_matrix: np.ndarray,
    n_splits: int = 16,
    approval_threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Tính Probability of Backtest Overfitting (PBO) theo phương pháp Combinatorial Symmetric Cross-Validation (CSCV).

    Tham số:
    - performance_matrix: mảng numpy 2D shape (T, N), với T là độ dài chuỗi thời gian (số kỳ/lợi suất/thao tác),
      N là số lượng chiến lược/cấu hình tham số được backtest.
    - n_splits: Số lượng khối chia (sub-blocks S, phải là số chẵn >= 4, chuẩn S=16).
    - approval_threshold: Ngưỡng xác suất tối đa cho phép (mặc định 0.40 = 40%).

    [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
    - Chặn mảng không phải 2D hoặc rác NaN/Inf.
    - Chặn N < 2 (bắt buộc phải có từ 2 cấu hình tham số trở lên để đánh giá overfitting so sánh).
    - Chặn T < n_splits (số quan sát phải đủ lớn để chia thành n_splits khối).
    - Chặn n_splits lẻ hoặc < 4.
    """
    if not isinstance(performance_matrix, np.ndarray):
        performance_matrix = np.asarray(performance_matrix, dtype=float)

    if performance_matrix.ndim != 2:
        raise ValueError(
            f"Lỗi hải quan PBO: performance_matrix phải là mảng 2D (T, N), nhận shape {performance_matrix.shape}"
        )

    T, N = performance_matrix.shape
    if N < 2:
        raise ValueError(
            f"Lỗi hải quan PBO: cần ít nhất N >= 2 cấu hình chiến lược để đánh giá PBO, nhận N={N}"
        )

    if not isinstance(n_splits, int) or n_splits < 4 or n_splits % 2 != 0:
        raise ValueError(
            f"Lỗi hải quan PBO: n_splits phải là số chẵn >= 4, nhận {n_splits}"
        )

    if T < n_splits:
        raise ValueError(
            f"Lỗi hải quan PBO: độ dài chuỗi T ({T}) nhỏ hơn số lượng khối n_splits ({n_splits})"
        )

    if np.any(np.isnan(performance_matrix)) or np.any(np.isinf(performance_matrix)):
        raise ValueError("Lỗi hải quan PBO: performance_matrix chứa giá trị NaN hoặc Inf!")

    # Bước 1: Chia T quan sát thành S = n_splits khối liên tục (contiguous sub-blocks)
    indices = np.array_split(np.arange(T), n_splits)
    block_returns_list = []
    for idx_block in indices:
        # Tính tổng lợi suất (hoặc hiệu suất tổng hợp) của từng cấu hình trên khối
        block_returns_list.append(np.sum(performance_matrix[idx_block, :], axis=0))
    block_returns = np.array(block_returns_list)  # shape (S, N)

    # Bước 2: Tạo tất cả các tổ hợp chọn S/2 khối làm tập Train (In-Sample)
    half_s = n_splits // 2
    all_blocks = set(range(n_splits))
    combinations = list(itertools.combinations(range(n_splits), half_s))

    logits = []
    relative_ranks = []
    n_overfit = 0

    for train_blocks in combinations:
        test_blocks = list(all_blocks - set(train_blocks))

        # Hiệu suất In-Sample (IS) trên tập Train
        is_perf = np.sum(block_returns[list(train_blocks), :], axis=0)  # shape (N,)
        # Hiệu suất Out-of-Sample (OOS) trên tập Test
        oos_perf = np.sum(block_returns[test_blocks, :], axis=0)        # shape (N,)

        # Chọn cấu hình tốt nhất In-Sample
        best_is_idx = int(np.argmax(is_perf))

        # Xếp hạng của tất cả N cấu hình trên tập OOS (từ thấp đến cao)
        # scipy/numpy argsort giúp lấy thứ hạng tương đối
        # Thứ hạng từ 1 đến N (1 là thấp nhất, N là cao nhất)
        order = np.argsort(oos_perf)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, N + 1, dtype=float)

        rank_best_oos = ranks[best_is_idx]
        # Xếp hạng tương đối w_c thuộc [0, 1]
        w_c = (rank_best_oos - 1.0) / (N - 1.0) if N > 1 else 0.5
        relative_ranks.append(float(w_c))

        # Logit lambda_c = ln( w_c / (1 - w_c) )
        # Bảo vệ cận 0 và 1 để không lỗi chia hoặc log(0)
        w_c_clamped = max(1e-6, min(1.0 - 1e-6, w_c))
        logit_c = math.log(w_c_clamped / (1.0 - w_c_clamped))
        logits.append(float(logit_c))

        # Nếu xếp hạng OOS dưới trung vị (w_c < 0.5 hoặc logit < 0), chiến lược bị suy thoái
        if w_c < 0.5:
            n_overfit += 1

    pbo = n_overfit / len(combinations)
    is_approved = pbo <= approval_threshold

    return {
        "pbo": float(pbo),
        "is_approved": bool(is_approved),
        "logits": logits,
        "relative_ranks": relative_ranks,
        "n_splits": int(n_splits),
        "n_combinations": len(combinations),
        "approval_threshold": float(approval_threshold)
    }
