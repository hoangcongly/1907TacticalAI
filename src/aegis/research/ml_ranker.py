"""Mô hình học máy xếp hạng mặt cắt ngang, huấn luyện tiến về phía trước.

Cùng một lịch huấn luyện cho mọi mô hình (mặc định lấy từ `RankModelSpec`): mô hình dùng tại
kỳ t chỉ học từ các kỳ < t - purge, nhãn là phân vị lợi suất TRONG kỳ, và không có siêu tham
số nào được dò trên dữ liệu đánh giá.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from aegis.research.rank_model import RankModelSpec

#: (X_train, y_train, sizes_train, X_test) -> điểm số cho X_test; `sizes` là số tài sản mỗi kỳ.
FitPredict = Callable[[np.ndarray, np.ndarray, List[int], np.ndarray], np.ndarray]


def build_rows(
    signals: Dict[str, pd.DataFrame],
    forward: pd.DataFrame,
    min_names: int = 15,
    min_features: float = 0.6,
):
    """Bảng phẳng (kỳ, tài sản). Tín hiệu thiếu được điền 0 nếu tài sản có đủ `min_features` tín hiệu.

    Trả về X, nhãn phân vị trong kỳ thuộc [0, 1] (NaN nếu chưa có lợi suất), chỉ số kỳ và cột.
    Hàng chưa có nhãn vẫn được giữ: chúng cần cho dự báo, và không bao giờ vào tập huấn luyện.
    """
    names = list(signals)
    index, columns = forward.index, forward.columns
    cube = np.stack(
        [
            signals[n].reindex(index=index, columns=columns).to_numpy(dtype=np.float64)
            for n in names
        ],
        axis=2,
    )
    target = forward.to_numpy(dtype=np.float64)
    X, y, period, column = [], [], [], []
    for i in range(len(index)):
        present = np.isfinite(cube[i]).mean(axis=1) >= min_features
        if present.sum() < min_names:
            continue
        where = np.flatnonzero(present)
        returns = target[i, where]
        labels = np.full(len(where), np.nan)
        known = np.isfinite(returns)
        if known.sum() >= min_names:
            labels[known] = pd.Series(returns[known]).rank(pct=True).to_numpy()
        X.append(np.nan_to_num(cube[i, where], nan=0.0))
        y.append(labels)
        period.append(np.full(len(where), i))
        column.append(where)
    if not X:
        return (
            np.empty((0, len(names))),
            np.empty(0),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
        )
    return (
        np.vstack(X),
        np.concatenate(y),
        np.concatenate(period),
        np.concatenate(column),
    )


def walk_forward_scores(
    signals: Dict[str, pd.DataFrame],
    forward: pd.DataFrame,
    fit_predict: FitPredict,
    spec: RankModelSpec | None = None,
) -> pd.DataFrame:
    """Điểm số ngoài mẫu trên lưới của `forward`; kỳ chưa đủ dữ liệu huấn luyện để NaN."""
    spec = spec or RankModelSpec()
    X, y, period, column = build_rows(signals, forward, spec.min_names)
    out = np.full(forward.shape, np.nan)
    periods = np.unique(period)
    for start in range(spec.min_train_periods, len(periods), spec.retrain_every):
        block = periods[start : start + spec.retrain_every]
        train_periods = periods[max(0, start - spec.train_periods) : start - spec.purge]
        train = np.isin(period, train_periods) & np.isfinite(y)
        test = np.isin(period, block)
        if (
            len(np.unique(period[train])) < spec.min_train_periods - spec.purge
            or not test.any()
        ):
            continue
        sizes = np.unique(period[train], return_counts=True)[1].tolist()
        scores = np.asarray(
            fit_predict(X[train], y[train], sizes, X[test]), dtype=np.float64
        )
        out[period[test], column[test]] = scores
    return pd.DataFrame(out, index=forward.index, columns=forward.columns)


def ridge_model(alpha: float = 10.0) -> FitPredict:
    def fit_predict(X, y, sizes, X_test):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=alpha).fit(X, y - 0.5).predict(X_test)

    return fit_predict


def lambdarank_model(spec: RankModelSpec | None = None) -> FitPredict:
    spec = spec or RankModelSpec()

    def fit_predict(X, y, sizes, X_test):
        import lightgbm as lgb

        labels = np.clip((y * spec.n_labels).astype(int), 0, spec.n_labels - 1)
        model = lgb.LGBMRanker(
            objective="lambdarank",
            num_leaves=spec.num_leaves,
            n_estimators=spec.n_estimators,
            learning_rate=spec.learning_rate,
            min_child_samples=spec.min_child_samples,
            subsample=spec.subsample,
            subsample_freq=1,
            colsample_bytree=spec.colsample_bytree,
            reg_lambda=spec.reg_lambda,
            random_state=spec.seed,
            n_jobs=-1,
            verbose=-1,
            label_gain=list(range(spec.n_labels)),
        )
        return model.fit(X, labels, group=sizes).predict(X_test)

    return fit_predict


def mlp_model(
    hidden=(64, 32), alpha: float = 1e-2, epochs: int = 60, seed: int = 7
) -> FitPredict:
    def fit_predict(X, y, sizes, X_test):
        from sklearn.neural_network import MLPRegressor

        clipped, clipped_test = np.clip(X, -5, 5), np.clip(X_test, -5, 5)
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            max_iter=epochs,
            batch_size=512,
            learning_rate_init=1e-3,
            random_state=seed,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(clipped, y - 0.5).predict(clipped_test)

    return fit_predict


def rank_ic(scores: pd.DataFrame, forward: pd.DataFrame) -> dict:
    """Tương quan hạng giữa điểm số và lợi suất kỳ sau, tính từng kỳ rồi lấy trung bình."""
    values = []
    for ts in scores.index:
        pair = pd.concat([scores.loc[ts], forward.loc[ts]], axis=1).dropna()
        if len(pair) >= 10 and pair.iloc[:, 0].nunique() > 1:
            values.append(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))
    values = np.asarray(values)
    if len(values) < 20:
        return {"periods": int(len(values)), "mean": None}
    return {
        "periods": int(len(values)),
        "mean": round(float(values.mean()), 4),
        "t_stat": round(
            float(values.mean() / values.std(ddof=1) * np.sqrt(len(values))), 2
        ),
        "share_positive": round(float((values > 0).mean()), 3),
    }
