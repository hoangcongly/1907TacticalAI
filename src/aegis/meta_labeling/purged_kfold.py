"""PurgedKFold Cross-Validation (Marcos Lopez de Prado - AFML Chapter 7).

Tuân thủ tuyệt đối quy tắc Zero-Leakage:
1. Purging (Cắt lọc): Xóa bỏ các mẫu huấn luyện thuộc train_before có t1 (time of exit)
   vượt quá t0_min (start of test set), tức là t1 > test_start_time.
2. Embargoing (Mất quyền thi hành/Cách ly): Xóa bỏ các mẫu huấn luyện thuộc train_after
   có t0 (time of entry) nằm trong khoảng từ max(test_t1) đến max(test_t1) + embargo_bars.
3. Strict Assertion: Đảm bảo tập giao giữa train_indices và test_indices là rỗng tuyệt đối.
"""

import math
from typing import Generator, Tuple, Optional
import numpy as np
import pandas as pd


class PurgedKFold:
    """
    [TASK B-1-14] PurgedKFold Cross-Validator (Marcos Lopez de Prado - AFML).

    Mô hình hóa quá trình Cross-Validation cho dữ liệu tài chính có nhãn chồng lấp (overlapping labels).
    Tuân thủ nghiêm ngặt Zero-Leakage qua hai cơ chế: Purging và Embargoing.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        event_times: Optional[pd.Series] = None,
    ):
        """
        Khởi tạo PurgedKFold.

        Tham số:
        - n_splits: Số lượng fold phân chia (mặc định 5).
        - embargo_pct: Tỷ lệ cách ly (embargo) tính trên tổng kích thước mẫu (mặc định 0.01 = 1%).
        - event_times: pd.Series với Index = Entry Time (t0), Values = Exit Time (t1).
                       Có thể truyền tại constructor hoặc tại phương thức .split().
        """
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError(f"Lỗi B-1-14: n_splits phải là số nguyên >= 2, nhận {n_splits}")
        if not isinstance(embargo_pct, (int, float)) or math.isnan(embargo_pct) or math.isinf(embargo_pct) or not (0.0 <= embargo_pct < 1.0):
            raise ValueError(f"Lỗi B-1-14: embargo_pct phải hợp lệ trong [0, 1), nhận {embargo_pct}")

        self.n_splits = n_splits
        self.embargo_pct = float(embargo_pct)
        self.event_times = event_times

    def split(
        self,
        X: pd.DataFrame | pd.Series | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
        groups: Optional[pd.Series | np.ndarray] = None,
        event_times: Optional[pd.Series] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Tạo ra các cặp (train_indices, test_indices) tuân thủ Purged và Embargoed.

        Tham số:
        - X: Dữ liệu đầu vào (n_samples, n_features) hoặc Series/array kích thước n_samples.
        - y, groups: Tham số chuẩn giao tiếp BaseCrossValidator (không bắt buộc).
        - event_times: pd.Series (Index = Entry Time, Values = Exit Time). Nếu không truyền,
                       sử dụng self.event_times đã khởi tạo tại constructor.

        Yields:
        - (train_indices, test_indices): Các mảng vị trí số nguyên (integer positional indices) cho Train và Test.
        """
        n_samples = len(X)
        if n_samples < self.n_splits:
            raise ValueError(f"Lỗi B-1-14: n_samples ({n_samples}) < n_splits ({self.n_splits})")

        et = event_times if event_times is not None else self.event_times
        if et is None:
            raise ValueError("Lỗi B-1-14: event_times không được cung cấp cho PurgedKFold")
        if not isinstance(et, pd.Series):
            raise ValueError(f"Lỗi B-1-14: event_times phải là pd.Series, nhận {type(et)}")
        if len(et) != n_samples:
            raise ValueError(f"Lỗi B-1-14: Độ dài event_times ({len(et)}) phải khớp với X ({n_samples})")

        # Kiểm duyệt NaN / Inf trong index và values của event_times
        if et.isna().any():
            raise ValueError("Lỗi B-1-14: event_times chứa giá trị NaN trong Values (Exit Times)")
        if pd.Series(et.index).isna().any():
            raise ValueError("Lỗi B-1-14: event_times chứa giá trị NaN trong Index (Entry Times)")

        # Kiểm tra tính logic: Exit Time t1 phải >= Entry Time t0
        # Hỗ trợ cả số nguyên, số thực lẫn DatetimeIndex
        t0_arr = et.index.to_numpy()
        t1_arr = et.to_numpy()
        if np.any(t1_arr < t0_arr):
            raise ValueError("Lỗi B-1-14: Phát hiện Exit Time (t1) nhỏ hơn Entry Time (t0) trong event_times")

        # Phân chia n_samples thành n_splits phần (cân bằng k-fold tĩnh theo thời gian)
        indices = np.arange(n_samples)
        fold_bounds = np.array_split(indices, self.n_splits)

        # Tính số lượng nến/bước cách ly (embargo_bars) trên tổng n_samples
        embargo_step = int(math.floor(self.embargo_pct * n_samples))

        for i, test_idx in enumerate(fold_bounds):
            test_idx = np.asarray(test_idx, dtype=int)
            if len(test_idx) == 0:
                continue

            test_start_t0 = t0_arr[test_idx[0]]
            test_max_t1 = np.max(t1_arr[test_idx])

            train_idx_list = []

            for j in indices:
                if j in test_idx:
                    continue

                # 1. PURGING LOGIC (train_before)
                # Nếu lệnh huấn luyện mở trước test set (j < test_idx[0]),
                # Exit Time t1 phải STRICTLY <= test_start_t0.
                if j < test_idx[0]:
                    if t1_arr[j] <= test_start_t0:
                        train_idx_list.append(j)
                    # Ngược lại (t1_arr[j] > test_start_t0): Bị loại do Purging!
                    continue

                # 2. EMBARGO LOGIC (train_after)
                # Nếu lệnh huấn luyện mở sau test set (j > test_idx[-1]),
                # Entry Time t0 phải strictly > max(Exit Times in Test Set) PLUS embargo.
                if j > test_idx[-1]:
                    # Tính ranh giới Embargo:
                    # Nếu index là số nguyên (bar index), max_t1 + embargo_step
                    # Nếu index là Datetime / Float, tra cứu vị trí index tương ứng
                    if np.issubdtype(t1_arr.dtype, np.integer) or np.issubdtype(t1_arr.dtype, np.floating):
                        embargo_boundary = test_max_t1 + embargo_step
                        if t0_arr[j] > embargo_boundary:
                            train_idx_list.append(j)
                    else:
                        # Với DatetimeIndex hoặc loại index khác:
                        # Tìm vị trí (location) lớn nhất có Exit Time == test_max_t1
                        matches = np.where(t1_arr == test_max_t1)[0]
                        max_t1_loc = matches[-1] if len(matches) > 0 else test_idx[-1]
                        embargo_loc = min(max_t1_loc + embargo_step, n_samples - 1)
                        embargo_boundary_t = t1_arr[embargo_loc] if embargo_step > 0 else test_max_t1
                        if t0_arr[j] > embargo_boundary_t or (j > embargo_loc and t0_arr[j] >= embargo_boundary_t):
                            train_idx_list.append(j)

            train_idx = np.array(train_idx_list, dtype=int)

            # [ARMOR GUARD] Strict Assertion checking for intersection between train and test indices
            intersection = set(train_idx).intersection(set(test_idx))
            assert len(intersection) == 0, (
                f"Canary Error B-1-14: Phát hiện rò rỉ dữ liệu giữa Train và Test! "
                f"Intersection indices: {intersection}"
            )

            yield train_idx, test_idx


# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
def test_purged_kfold_no_overlap_simple():
    """Kiểm tra trên chuỗi nến không chồng lấp t1 == t0 (mỗi nến chốt ngay tại bar đó)."""
    n = 100
    X = pd.DataFrame({"feat": np.random.randn(n)})
    et = pd.Series(index=np.arange(n), data=np.arange(n)) # t0 = t1

    pkf = PurgedKFold(n_splits=5, embargo_pct=0.0)
    splits = list(pkf.split(X, event_times=et))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(train_idx) + len(test_idx) == n
    print("✅ test_purged_kfold_no_overlap_simple PASSED!")


def test_purged_kfold_toy_overlap_mathematical_proof():
    """
    [CRUCIAL MATHEMATICAL PROOF OF PURGING & EMBARGOING]:
    Giả lập 10 giao dịch (bar index 0 đến 9).
    Test fold là các lệnh index 4 và 5.
    - Lệnh index 4 có t0 = 4, t1 = 6.
    - Lệnh index 5 có t0 = 5, t1 = 7.
    => test_start_t0 = 4, test_max_t1 = max(6, 7) = 7.

    Phân tích train_before (j < 4):
    - Lệnh 2: t0 = 2, t1 = 3 <= test_start_t0 (4) -> GIỮ (Retained).
    - Lệnh 3: t0 = 3, t1 = 5 > test_start_t0 (4)  -> PURGED (Vì vắt qua nến 4, 5 thuộc test set)!

    Phân tích train_after (j > 5) với embargo_pct = 0.2 (2 bars trên tổng 10 bars -> embargo_step = 2):
    - test_max_t1 = 7. Embargo boundary = 7 + 2 = 9.
    - Lệnh 6: t0 = 6 <= 9 -> PURGED & EMBARGOED!
    - Lệnh 7: t0 = 7 <= 9 -> EMBARGOED!
    - Lệnh 8: t0 = 8 <= 9 -> EMBARGOED!
    - Lệnh 9: t0 = 9 <= 9 -> EMBARGOED! (hoặc nếu t0 > 9 thì giữ).
    """
    n = 10
    X = pd.DataFrame({"feat": np.arange(n)})
    
    # Thiết lập t0 là index (0 -> 9), t1 là Values
    t0_list = np.arange(n)
    t1_list = np.array([
        0, # 0
        1, # 1
        3, # 2 -> t1=3 <= 4 (OK)
        5, # 3 -> t1=5 > 4 (PURGED!)
        6, # 4 [TEST]
        7, # 5 [TEST]
        8, # 6 -> t0=6 <= 9 (EMBARGOED)
        8, # 7 -> t0=7 <= 9 (EMBARGOED)
        9, # 8 -> t0=8 <= 9 (EMBARGOED)
        10 # 9 -> t0=9 <= 9 (EMBARGOED)
    ])
    et = pd.Series(index=t0_list, data=t1_list)

    pkf = PurgedKFold(n_splits=5, embargo_pct=0.2) # 5 fold -> mỗi fold 2 phần tử. Fold 2 là [4, 5].
    
    splits = list(pkf.split(X, event_times=et))
    train_idx, test_idx = splits[2] # Fold index 2 tương ứng với test_idx = [4, 5]
    
    assert np.array_equal(test_idx, [4, 5]), f"Kỳ vọng test_idx=[4, 5], nhận {test_idx}"
    
    # Xác nhận Purging: lệnh 3 bị loại, lệnh 2 được giữ
    assert 2 in train_idx, "Lệnh 2 (t1=3 <= 4) phải được giữ trong train_before"
    assert 3 not in train_idx, "Lệnh 3 (t1=5 > 4) buộc phải bị PURGED!"
    
    # Xác nhận Embargoing: lệnh 6, 7, 8, 9 bị loại
    for after_j in [6, 7, 8, 9]:
        assert after_j not in train_idx, f"Lệnh {after_j} phải bị EMBARGOED (embargo boundary = 9)!"
        
    # Xác nhận Strict Intersection Assertion
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    print("✅ test_purged_kfold_toy_overlap_mathematical_proof PASSED!")


def test_purged_kfold_input_validation_guards():
    """Kiểm tra bẫy lỗi NaN / Inf và sai định dạng."""
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    
    # NaN trong event_times
    et_nan = pd.Series([1, 2, np.nan, 4, 5])
    pkf = PurgedKFold(n_splits=2)
    try:
        list(pkf.split(X, event_times=et_nan))
        assert False, "Không bắt lỗi NaN trong event_times!"
    except ValueError as e:
        assert "NaN" in str(e)

    # t1 < t0
    et_invalid = pd.Series(index=[0, 1, 2, 3, 4], data=[0, 1, 0, 3, 4]) # tại idx=2, t1=0 < t0=2
    try:
        list(pkf.split(X, event_times=et_invalid))
        assert False, "Không bắt lỗi t1 < t0!"
    except ValueError as e:
        assert "nhỏ hơn Entry Time" in str(e)

    print("✅ test_purged_kfold_input_validation_guards PASSED!")


if __name__ == "__main__":
    test_purged_kfold_no_overlap_simple()
    test_purged_kfold_toy_overlap_mathematical_proof()
    test_purged_kfold_input_validation_guards()
