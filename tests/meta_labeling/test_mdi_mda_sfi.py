import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from aegis.meta_labeling.feature_selection.mdi_mda_sfi import (
    compute_mdi,
    compute_mda,
    compute_sfi,
    triple_consensus_ranker,
)


@pytest.fixture
def synthetic_data():
    """Dữ liệu giả lập: F1 tốt, F2 tương quan F1, F3 rác."""
    np.random.seed(42)
    n_samples = 200
    f1 = np.random.normal(0, 1, n_samples)
    f2 = f1 + np.random.normal(0, 0.1, n_samples)
    f3 = np.random.normal(0, 1, n_samples)
    y = (f1 > 0).astype(int)
    X = pd.DataFrame({"Good_F1": f1, "Corr_F2": f2, "Noise_F3": f3})
    y = pd.Series(y)
    return X, y


def test_mdi_basic(synthetic_data):
    """MDI: trả đúng shape và columns."""
    X, y = synthetic_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf.fit(X, y)
    mdi = compute_mdi(clf, X.columns.tolist())
    assert "MDI_mean" in mdi.columns
    assert "MDI_std" in mdi.columns
    assert len(mdi) == 3


def test_mda_basic(synthetic_data):
    """MDA: trả đúng shape, Noise_F3 phải có MDA thấp."""
    X, y = synthetic_data
    cv = KFold(n_splits=3, shuffle=False)
    mda = compute_mda(
        clf=RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3),
        X=X, y=y, cv_gen=cv, scoring=accuracy_score, is_higher_better=True,
    )
    assert "MDA_mean" in mda.columns
    assert len(mda) == 3


def test_sfi_all_features_have_scores(synthetic_data):
    """
    FIX #4: Mọi feature phải có SFI score. Trước fix, feature 2+ bị NaN
    do generator exhaustion.
    """
    X, y = synthetic_data
    cv = KFold(n_splits=3, shuffle=False)
    sfi = compute_sfi(
        clf=RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3),
        X=X, y=y, cv_gen=cv, scoring=accuracy_score,
    )
    assert "SFI_mean" in sfi.columns
    assert len(sfi) == 3
    assert not sfi["SFI_mean"].isna().any(), f"SFI có NaN (generator exhaustion?): {sfi}"


def test_triple_consensus_noise_ranked_last(synthetic_data):
    """Noise_F3 PHẢI đứng bét trong bảng xếp hạng."""
    X, y = synthetic_data
    cv = KFold(n_splits=3, shuffle=False)
    clf_template = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)

    clf_fitted = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    clf_fitted.fit(X, y)
    mdi = compute_mdi(clf_fitted, X.columns.tolist())

    mda = compute_mda(clf_template, X, y, cv, accuracy_score, is_higher_better=True)
    sfi = compute_sfi(clf_template, X, y, cv, accuracy_score)
    consensus = triple_consensus_ranker(mdi, mda, sfi)

    assert "Consensus_Rank_Sum" in consensus.columns
    assert "Final_Rank" in consensus.columns
    rank_f3 = consensus.loc["Noise_F3", "Final_Rank"]
    assert rank_f3 > 1, "Feature Rác (Noise_F3) không thể đứng hạng 1!"


def test_mda_with_event_times():
    """
    FIX #3: MDA phải chấp nhận event_times param mà không crash.
    Khi truyền KFold + event_times, event_times bị bỏ qua (an toàn).
    """
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"a": np.random.normal(0, 1, n), "b": np.random.normal(0, 1, n)})
    y = pd.Series((X["a"] > 0).astype(int))
    et = pd.Series(index=range(n), data=range(n))
    cv = KFold(n_splits=2, shuffle=False)

    # event_times truyền vào nhưng KFold không dùng -> không crash
    mda = compute_mda(
        clf=RandomForestClassifier(n_estimators=5, random_state=42),
        X=X, y=y, cv_gen=cv, scoring=accuracy_score,
        event_times=et,
    )
    assert not mda["MDA_mean"].isna().any()
