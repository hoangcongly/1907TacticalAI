"""
Weighted Bootstrap Forest Classifier.
Module B-6-1 (AFML Chapter 4 & 8).

Fixes applied:
- #5: Align predict_proba shape when a tree misses a class during bootstrap
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from typing import Optional


class WeightedBootstrapForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest tùy chỉnh chuẩn AFML:
    Bốc mẫu (bootstrap) với xác suất tỷ lệ thuận với Average Uniqueness (\bar{u}_i)
    thay vì đồng đều 1/N.
    """

    # sklearn 1.6+ cần tag protocol tường minh để is_classifier() trả True
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        """Override tag protocol cho sklearn 1.6+ — BaseEstimator.__sklearn_tags__
        không gọi super() nên ClassifierMixin không bao giờ được chạy."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Huấn luyện forest bằng weighted bootstrap sampling."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []

        n_samples = X.shape[0]

        # Tính xác suất bốc mẫu
        if sample_weight is None:
            p = np.ones(n_samples) / n_samples
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.min() < 0:
                raise ValueError("sample_weight không được âm.")
            total = sample_weight.sum()
            if total == 0 or np.isnan(total):
                p = np.ones(n_samples) / n_samples
            else:
                p = sample_weight / total

        for _ in range(self.n_estimators):
            # AFML: Bốc mẫu có hoàn lại với xác suất P(u_bar_i)
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=p)

            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.randint(0, 2 ** 31)
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X):
        """
        Dự đoán xác suất, xử lý đúng trường hợp cây con thiếu class.
        
        FIX #5: Khi bootstrap chỉ bốc được mẫu 1 class, tree.predict_proba
        trả về shape (n, 1) thay vì (n, n_classes). Phải align thủ công 
        theo self.classes_.
        """
        check_is_fitted(self, "estimators_")
        X = check_array(X)

        all_proba = np.zeros((X.shape[0], self.n_classes_))

        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)
            # Align: tree.classes_ có thể là subset của self.classes_
            if np.array_equal(tree.classes_, self.classes_):
                # Fast path: shape khớp hoàn toàn
                all_proba += tree_proba
            else:
                # Slow path: map từng class của cây vào vị trí đúng trong self.classes_
                for k, cls in enumerate(tree.classes_):
                    col = np.searchsorted(self.classes_, cls)
                    all_proba[:, col] += tree_proba[:, k]

        all_proba /= self.n_estimators
        return all_proba

    def predict(self, X):
        """Dự đoán class dựa trên argmax xác suất."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
