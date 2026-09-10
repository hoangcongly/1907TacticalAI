"""
Isotonic Calibration using Purged K-Fold.
Module B-6-2.

Fixes applied:
- #7: PurgedKFoldAdapter to make PurgedKFold compatible with sklearn CalibratedClassifierCV
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional
import pandas as pd


class PurgedKFoldAdapter:
    """
    Adapter giúp PurgedKFold tương thích với sklearn CalibratedClassifierCV.
    
    sklearn gọi cv.split(X, y, groups) nội bộ — không truyền event_times.
    Adapter này pre-bind event_times để PurgedKFold nhận đúng tham số.
    """

    def __init__(self, purged_kfold, event_times: pd.Series):
        """
        Args:
            purged_kfold: Instance PurgedKFold đã khởi tạo.
            event_times: pd.Series (Index=t0, Values=t1).
        """
        self.purged_kfold = purged_kfold
        self.event_times = event_times

    def split(self, X, y=None, groups=None):
        """Gọi PurgedKFold.split() với event_times đã bind sẵn."""
        return self.purged_kfold.split(X=X, event_times=self.event_times)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Trả về số fold."""
        return self.purged_kfold.n_splits


def build_calibrated_classifier(base_estimator, cv_gen, event_times: Optional[pd.Series] = None):
    """
    Tạo mô hình nắn chuẩn xác suất (Calibration) theo chuẩn AFML.
    
    Random Forest thường đẩy xác suất dự đoán về mức 0.5 (under-confidence ở biên).
    Dùng Isotonic Regression để nắn lại.
    BẮT BUỘC phải dùng PurgedKFold trong cross-validation để tránh leak.

    Args:
        base_estimator: Mô hình cơ sở (WeightedBootstrapForestClassifier, ...).
            PHẢI có predict_proba().
        cv_gen: Cross-validator (PurgedKFold hoặc KFold).
        event_times: pd.Series cho PurgedKFold. Nếu None và cv_gen là PurgedKFold,
            sẽ raise lỗi rõ ràng.

    Returns:
        CalibratedClassifierCV sẵn sàng fit(X, y).
    """
    if not hasattr(base_estimator, 'predict_proba'):
        raise ValueError(
            "base_estimator phải có method predict_proba() "
            "để Isotonic Calibration hoạt động."
        )

    # Nếu cv_gen là PurgedKFold (có thuộc tính embargo_bars), bọc adapter
    is_purged = hasattr(cv_gen, 'embargo_bars') or hasattr(cv_gen, 'embargo_pct')
    if is_purged:
        if event_times is None:
            raise ValueError(
                "event_times là BẮT BUỘC khi dùng PurgedKFold. "
                "Truyền pd.Series(index=t0, data=t1)."
            )
        cv_wrapped = PurgedKFoldAdapter(cv_gen, event_times)
    else:
        cv_wrapped = cv_gen

    calibrated_clf = CalibratedClassifierCV(
        estimator=base_estimator,
        method='isotonic',
        cv=cv_wrapped
    )

    return calibrated_clf
