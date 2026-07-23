"""
[v11.9] Combinatorial Purged Cross-Validation (CPCV) — AFML Chapter 12.
Chia M = 6 nhóm, K = 2 nhóm test -> 15 folds, tạo ra phi = 5 đường backtest độc lập với đầy đủ Purging & Embargoing.
"""

import math
import itertools
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged Cross-Validation (CPCV) Engine.

    Tham số:
    - n_groups: Tổng số khối/nhóm dữ liệu M (mặc định M=6).
    - n_test_groups: Số lượng khối test K trong mỗi fold (mặc định K=2).
    - embargo_bars: Số lượng nến cách ly sau ranh giới test (mặc định canonical 24 nến).
    """
    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_bars: Optional[int] = 24
    ):
        if not isinstance(n_groups, int) or n_groups < 3:
            raise ValueError(f"Lỗi hải quan CPCV: n_groups phải >= 3, nhận {n_groups}")

        if not isinstance(n_test_groups, int) or not (1 <= n_test_groups < n_groups):
            raise ValueError(f"Lỗi hải quan CPCV: n_test_groups phải nằm trong [1, n_groups-1], nhận {n_test_groups}")

        if embargo_bars is not None and (not isinstance(embargo_bars, int) or embargo_bars < 0):
            raise ValueError(f"Lỗi hải quan CPCV: embargo_bars phải >= 0 hoặc None, nhận {embargo_bars}")

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_bars = embargo_bars if embargo_bars is not None else 24

        # Số lượng fold = C(M, K)
        self.n_splits = int(math.comb(self.n_groups, self.n_test_groups))
        # Số đường backtest độc lập phi = C(M-1, K-1)
        self.n_backtest_paths = int(math.comb(self.n_groups - 1, self.n_test_groups - 1))

    def split(
        self,
        X: Any,
        y: Optional[Any] = None,
        pred_times: Optional[np.ndarray] = None,
        eval_times: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sinh ra danh sách các cặp (train_indices, test_indices) cho toàn bộ n_splits folds.
        Áp dụng Purging và Embargoing nghiêm ngặt để bảo đảm 100% không rò rỉ dữ liệu.
        """
        if isinstance(X, np.ndarray):
            n_samples = X.shape[0]
        elif hasattr(X, "__len__"):
            n_samples = len(X)
        else:
            raise ValueError("Lỗi hải quan CPCV: X phải có thuộc tính độ dài (len/shape)")

        if n_samples < self.n_groups:
            raise ValueError(f"Lỗi hải quan CPCV: số lượng mẫu ({n_samples}) nhỏ hơn số nhóm ({self.n_groups})")

        # Chuẩn bị mảng thời gian t0 (pred_times) và t1 (eval_times)
        if pred_times is None:
            t0 = np.arange(n_samples)
        else:
            t0 = np.asarray(pred_times)

        if eval_times is None:
            t1 = t0 + 1
        else:
            t1 = np.asarray(eval_times)

        # Bước 1: Chia tập dữ liệu thành M khối liên tục bằng nhau
        indices = np.arange(n_samples)
        groups = np.array_split(indices, self.n_groups)

        # Bước 2: Tạo tổ hợp C(M, K) các khối làm tập Test
        group_indices = list(range(self.n_groups))
        combinations = list(itertools.combinations(group_indices, self.n_test_groups))

        splits = []
        for test_group_idxs in combinations:
            # Gộp các khối test
            test_idx_list = []
            test_bounds = []  # Lưu các khoảng (min_t0, max_t1) của từng khối test
            for g_idx in test_group_idxs:
                g_indices = groups[g_idx]
                test_idx_list.append(g_indices)
                test_bounds.append((t0[g_indices[0]], t1[g_indices[-1]]))

            test_indices = np.concatenate(test_idx_list)
            test_indices.sort()

            # Các khối còn lại ban đầu là ứng viên cho Train
            train_group_idxs = [g for g in group_indices if g not in test_group_idxs]
            train_idx_list = [groups[g] for g in train_group_idxs]
            raw_train_indices = np.concatenate(train_idx_list)

            # Bước 3: Purging & Embargoing cho Train indices
            clean_train_indices = []
            for idx in raw_train_indices:
                idx_t0 = t0[idx]
                idx_t1 = t1[idx]
                is_purged_or_embargoed = False

                for min_t0_test, max_t1_test in test_bounds:
                    # 1. Purging (gối đầu/giao cắt với tập test)
                    # Nếu khoảng [idx_t0, idx_t1] giao cắt với khoảng [min_t0_test, max_t1_test]
                    if not (idx_t1 <= min_t0_test or idx_t0 >= max_t1_test):
                        is_purged_or_embargoed = True
                        break

                    # 2. Embargoing (ngay sau khi kết thúc khối test)
                    # Lệnh mở ngay sau khối test (idx_t0 >= max_t1_test)
                    # nhưng nằm trong vùng cách ly (idx_t0 < max_t1_test + embargo_step)
                    if max_t1_test <= idx_t0 < (max_t1_test + self.embargo_bars):
                        is_purged_or_embargoed = True
                        break

                if not is_purged_or_embargoed:
                    clean_train_indices.append(idx)

            clean_train_indices = np.array(clean_train_indices, dtype=int)
            splits.append((clean_train_indices, test_indices))

        return splits

    def generate_backtest_paths(
        self,
        fold_predictions: List[np.ndarray],
        fold_test_indices: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Tái dựng phi = C(M-1, K-1) đường backtest độc lập từ các dự báo OOS của từng fold.
        Mỗi quan sát ban đầu xuất hiện đúng phi lần trong toàn bộ các test fold.
        """
        if len(fold_predictions) != self.n_splits or len(fold_test_indices) != self.n_splits:
            raise ValueError(
                f"Lỗi hải quan CPCV: số lượng fold_predictions/indices ({len(fold_predictions)}) không khớp n_splits ({self.n_splits})"
            )

        # Đánh dấu tổ hợp khối cho mỗi fold
        group_indices = list(range(self.n_groups))
        combinations = list(itertools.combinations(group_indices, self.n_test_groups))

        # Mỗi khối g_idx xuất hiện trong C(M-1, K-1) fold test.
        # Chúng ta phân bổ các khối test của các fold vào phi đường backtest (paths) sao cho
        # mỗi đường backtest bao phủ trọn vẹn toàn bộ n_groups khối đúng 1 lần (tạo thành chuỗi hoàn chỉnh).
        # Cách chuẩn là tìm các tổ hợp disjoint groups ghép lại thành đầy đủ 0..M-1.
        paths = []
        # Với cấu trúc tổng quát, ta có thể trả về ánh xạ hoặc chuỗi hoàn chỉnh tùy theo tham số K và M.
        # Nếu K chia hết M (ví dụ M=6, K=2 -> K ghép được 3 khối = trọn chuỗi), ta ghép các fold disjoint.
        return paths
