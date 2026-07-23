import pytest
import numpy as np
from aegis.validation.cpcv import CombinatorialPurgedKFold


def test_cpcv_end_to_end_splits_and_leakage_guards():
    """
    [INTEGRATION TEST — CPCV END TO END]:
    Kiểm chứng Combinatorial Purged Cross-Validation (M=6, K=2 -> 15 splits).
    Khẳng định không có bất kỳ quan sát Train nào xâm lấn vào Test hoặc vùng cách ly Embargo.
    """
    n_samples = 300
    cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2, embargo_bars=10)
    
    assert cpcv.n_splits == 15          # C(6, 2) = 15
    assert cpcv.n_backtest_paths == 5   # C(5, 1) = 5

    # Thời gian đóng lệnh mỗi nến vắt qua 5 nến tiếp theo (t1 = t0 + 5)
    t0 = np.arange(n_samples)
    t1 = t0 + 5

    splits = cpcv.split(X=np.zeros((n_samples, 1)), pred_times=t0, eval_times=t1)
    assert len(splits) == 15

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        assert len(test_idx) > 0
        assert len(train_idx) > 0

        # Khẳng định không trùng lặp chỉ số (Index Non-overlap)
        assert len(set(train_idx).intersection(set(test_idx))) == 0

        # Khẳng định bọc thép bất biến thời gian (Temporal Invariant Assertion) cho từng khối test liên tục
        # Trong CPCV (K=2), test_idx có thể gồm 2 khối rời nhau (ví dụ group 0 và group 2).
        # Ta tách test_idx thành các khối liên tục và kiểm tra ranh giới với từng khối.
        test_blocks = np.split(test_idx, np.where(np.diff(test_idx) > 1)[0] + 1)
        test_bounds = [(t0[blk[0]], t1[blk[-1]]) for blk in test_blocks]

        for idx in train_idx:
            idx_t0 = t0[idx]
            idx_t1 = t1[idx]
            for min_test_t0, max_test_t1 in test_bounds:
                # 1. Không giao cắt với từng khối test
                assert idx_t1 <= min_test_t0 or idx_t0 >= max_test_t1, (
                    f"Fold {fold_idx}: Lệnh Train {idx} [{idx_t0}, {idx_t1}] giao cắt với Test block [{min_test_t0}, {max_test_t1}]"
                )
                # 2. Lệnh sau khối Test phải nằm ngoài vùng cách ly Embargo (10 nến)
                if idx_t0 >= max_test_t1:
                    assert idx_t0 >= max_test_t1 + 10, (
                        f"Fold {fold_idx}: Lệnh Train {idx} [{idx_t0}, {idx_t1}] vi phạm Embargo sau Test block [{min_test_t0}, {max_test_t1}]!"
                    )

    print("✅ [CPCV END-TO-END] 15 Folds Combinatorial Purged Cross-Validation với Zero-Leakage & Embargo PASSED!")


def test_cpcv_armor_plated_guards():
    """Kiểm thử hải quan bọc thép cho CPCV."""
    with pytest.raises(ValueError, match="n_groups phải >= 3"):
        CombinatorialPurgedKFold(n_groups=2)

    with pytest.raises(ValueError, match="n_test_groups phải nằm trong"):
        CombinatorialPurgedKFold(n_groups=6, n_test_groups=6)

    with pytest.raises(ValueError, match="số lượng mẫu .* nhỏ hơn số nhóm"):
        cpcv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2)
        cpcv.split(np.zeros((4, 1)))

    print("✅ [CPCV ENGINE] Armor-plated guards PASSED!")
