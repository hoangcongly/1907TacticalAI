import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from aegis.meta_labeling.weighted_bootstrap_forest import WeightedBootstrapForestClassifier


def test_weighted_bootstrap_forest_basic():
    """B-6-1: Forest cơ bản phải predict được."""
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 2))
    y = (X[:, 0] > 0).astype(int)

    clf = WeightedBootstrapForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf.fit(X, y)

    pred = clf.predict(X)
    proba = clf.predict_proba(X)

    assert len(pred) == 100
    assert proba.shape == (100, 2)
    assert accuracy_score(y, pred) > 0.6


def test_weighted_bootstrap_with_sample_weights():
    """B-6-1: Trọng số ưu tiên mẫu 0-49."""
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 2))
    y = (X[:, 0] > 0).astype(int)

    w = np.zeros(100)
    w[:50] = 1000.0
    w[50:] = 1.0

    clf = WeightedBootstrapForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf.fit(X, y, sample_weight=w)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)


def test_forest_missing_class_in_bootstrap():
    """
    FIX #5: Khi 1 class cực hiếm, cây con có thể không thấy nó trong bootstrap.
    predict_proba KHÔNG ĐƯỢC crash.
    """
    np.random.seed(42)
    # 98 mẫu class 0, 2 mẫu class 1
    X = np.vstack([
        np.random.normal(0, 1, (98, 2)),
        np.random.normal(5, 0.01, (2, 2)),
    ])
    y = np.array([0] * 98 + [1] * 2)

    # Trọng số ép gần như toàn bộ sang class 0
    w = np.array([100.0] * 98 + [0.001] * 2)

    clf = WeightedBootstrapForestClassifier(n_estimators=20, random_state=42, max_depth=2)
    clf.fit(X, y, sample_weight=w)

    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2), f"Shape sai: {proba.shape}, kỳ vọng (100, 2)"
    # Tổng xác suất mỗi hàng phải ~1.0
    row_sums = proba.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)
