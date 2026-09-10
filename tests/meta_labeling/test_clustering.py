import pytest
import numpy as np
import pandas as pd
from aegis.meta_labeling.feature_selection.clustering import (
    compute_distance_matrix,
    cluster_features_hierarchical,
    apply_consensus_filter,
)


def test_clustering_basic():
    """B-5-4: F1 và F2 tương quan cao phải chung cụm, F3 F4 riêng."""
    np.random.seed(42)
    n = 100
    f1 = np.random.normal(0, 1, n)
    f2 = f1 + np.random.normal(0, 0.05, n)
    f3 = np.random.normal(0, 1, n)
    f4 = np.random.normal(0, 1, n)
    X = pd.DataFrame({"F1": f1, "F2": f2, "F3": f3, "F4": f4})

    dist = compute_distance_matrix(X)
    assert dist.shape == (4, 4)
    assert dist.loc["F1", "F2"] < 0.1
    assert dist.loc["F1", "F3"] > 0.8

    clusters = cluster_features_hierarchical(dist, corr_threshold=0.70)
    assert len(clusters) == 3  # {F1,F2}, {F3}, {F4}


def test_consensus_filter_keeps_best():
    """B-5-4: Trong cụm {F1,F2}, giữ F1 (hạng 1), loại F2 (hạng 4)."""
    np.random.seed(42)
    n = 100
    f1 = np.random.normal(0, 1, n)
    f2 = f1 + np.random.normal(0, 0.05, n)
    f3 = np.random.normal(0, 1, n)
    f4 = np.random.normal(0, 1, n)
    X = pd.DataFrame({"F1": f1, "F2": f2, "F3": f3, "F4": f4})

    consensus_ranks = pd.DataFrame(
        {"Final_Rank": [1, 4, 2, 3]}, index=["F1", "F2", "F3", "F4"]
    )

    selected = apply_consensus_filter(X, consensus_ranks, corr_threshold=0.70)
    assert len(selected) == 3
    assert "F1" in selected
    assert "F2" not in selected
    assert "F3" in selected
    assert "F4" in selected


def test_distance_matrix_not_mutated():
    """
    FIX #8: compute_distance_matrix + cluster không được thay đổi input.
    """
    np.random.seed(42)
    X = pd.DataFrame({"a": np.random.normal(0, 1, 50), "b": np.random.normal(0, 1, 50)})
    dist = compute_distance_matrix(X)
    dist_copy = dist.copy()

    # Gọi clustering — trước fix, dist bị fill_diagonal(0) mutate
    _ = cluster_features_hierarchical(dist, corr_threshold=0.50)

    # dist gốc PHẢI không đổi
    pd.testing.assert_frame_equal(dist, dist_copy)
