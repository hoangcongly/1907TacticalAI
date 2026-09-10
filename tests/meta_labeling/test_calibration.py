import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from aegis.meta_labeling.calibration import build_calibrated_classifier, PurgedKFoldAdapter
from aegis.meta_labeling.weighted_bootstrap_forest import WeightedBootstrapForestClassifier


def test_calibration_with_kfold():
    """B-6-2: CalibratedClassifierCV wrapper hoạt động với KFold thường."""
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 2))
    y = (X[:, 0] > 0).astype(int)

    base_clf = WeightedBootstrapForestClassifier(n_estimators=5, random_state=42)
    cv = KFold(n_splits=2)

    calibrated = build_calibrated_classifier(base_clf, cv)
    calibrated.fit(X, y)
    proba = calibrated.predict_proba(X)

    assert proba.shape == (100, 2)
    # Isotonic nắn xác suất — sum mỗi hàng phải ~1.0
    row_sums = proba.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)


def test_calibration_with_purged_kfold():
    """
    FIX #7: CalibratedClassifierCV phải hoạt động với PurgedKFold thông qua adapter.
    """
    from aegis.meta_labeling.purged_kfold import PurgedKFold

    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, (n, 2))
    y = (X[:, 0] > 0).astype(int)
    et = pd.Series(index=range(n), data=range(n))  # t0 = t1 (không chồng lấp)

    base_clf = WeightedBootstrapForestClassifier(n_estimators=5, random_state=42)
    cv = PurgedKFold(n_splits=2, embargo_bars=0)

    calibrated = build_calibrated_classifier(base_clf, cv, event_times=et)
    calibrated.fit(X, y)
    proba = calibrated.predict_proba(X)
    assert proba.shape == (n, 2)


def test_calibration_purged_without_event_times_raises():
    """FIX #7 guard: PurgedKFold mà không truyền event_times phải raise."""
    from aegis.meta_labeling.purged_kfold import PurgedKFold

    base_clf = WeightedBootstrapForestClassifier(n_estimators=5, random_state=42)
    cv = PurgedKFold(n_splits=2, embargo_bars=0)

    with pytest.raises(ValueError, match="event_times là BẮT BUỘC"):
        build_calibrated_classifier(base_clf, cv)  # Không truyền event_times


def test_calibration_no_predict_proba_raises():
    """Guard: mô hình không có predict_proba phải raise."""
    from sklearn.svm import LinearSVC

    base_clf = LinearSVC()
    cv = KFold(n_splits=2)

    with pytest.raises(ValueError, match="predict_proba"):
        build_calibrated_classifier(base_clf, cv)
