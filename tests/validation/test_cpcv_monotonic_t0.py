import pytest
import pandas as pd
import numpy as np
from aegis.validation.cpcv import PurgedKFold

def test_cpcv_monotonic_t0():
    """
    A3 / G2: Đảm bảo thời gian bắt đầu của label (t0) hoặc array index phải tăng nghiêm ngặt (monotonic).
    Nếu không tăng nghiêm ngặt, cơ chế purge/embargo sẽ bị sai (vì nó dựa vào việc searchsorted).
    """
    n_samples = 100
    times = pd.Series(pd.date_range("2023-01-01", periods=n_samples, freq="1H"))
    
    # Tạo label array sao cho t0 bị XÁO TRỘN.
    # Trong AEGIS, pred_times đóng vai trò là mảng nhãn, cần phải được sort.
    # PurgedKFold sẽ raise ValueError nếu pred_times không đơn điệu.
    t0_shuffled = pd.Series(np.random.permutation(times.values))
    t1 = t0_shuffled + pd.Timedelta(hours=1)
    
    cv = PurgedKFold(n_splits=3, n_test_splits=1)
    
    with pytest.raises(ValueError, match="monotonically increasing"):
        list(cv.split(X=np.zeros(n_samples), pred_times=t0_shuffled, eval_times=t1))

def test_cpcv_embargo_bars():
    """
    A4 / G2: Kiểm tra Embargo dùng Bars.
    """
    n_samples = 100
    times = pd.Series(pd.date_range("2023-01-01", periods=n_samples, freq="1H"))
    t0 = times
    t1 = t0 + pd.Timedelta(hours=2) # Mỗi lệnh kéo dài 2 bars
    
    cv = PurgedKFold(n_splits=2, n_test_splits=1, embargo_bars=5)
    splits = list(cv.split(X=np.zeros(n_samples), pred_times=t0, eval_times=t1))
    
    for train_indices, test_indices in splits:
        # Tập train đứng ngay sau tập test phải bị dời đi 5 bars so với điểm kết thúc (t1) của test
        # Lấy index lớn nhất trong test
        max_test = test_indices.max()
        # Tìm các index trong train mà lớn hơn max_test
        train_after_test = train_indices[train_indices > max_test]
        if len(train_after_test) > 0:
            min_train_after = train_after_test.min()
            # test index t kết thúc ở t+2. Cộng thêm 5 bars embargo -> t+7.
            # Do đó min_train_after phải >= max_test + 2 + 5 = max_test + 7
            assert min_train_after >= max_test + 7, "Embargo bars không được thực thi đúng kích thước!"
