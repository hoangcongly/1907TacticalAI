"""Xếp hạng bằng học máy: không rò rỉ tương lai, và học được quan hệ có thật."""

import numpy as np
import pandas as pd
import pytest

from aegis.research import ml_ranker as ml
from aegis.research.rank_model import RankModelSpec

SPEC = RankModelSpec(
    train_periods=60, retrain_every=10, purge=2, min_train_periods=30, min_names=10
)


def _market(periods=120, names=30, seed=0, strength=1.0):
    rng = np.random.default_rng(seed)
    index, columns = pd.RangeIndex(periods), [f"S{i}" for i in range(names)]
    good = pd.DataFrame(rng.normal(size=(periods, names)), index=index, columns=columns)
    noise = pd.DataFrame(
        rng.normal(size=(periods, names)), index=index, columns=columns
    )
    forward = strength * 0.01 * good + pd.DataFrame(
        rng.normal(0, 0.01, (periods, names)), index=index, columns=columns
    )
    forward.iloc[-1] = np.nan  # kỳ cuối chưa có lợi suất
    return {"good": good, "noise": noise}, forward


def test_rows_fill_sparse_signals_and_drop_assets_with_too_few():
    signals, forward = _market(periods=5, names=12)
    signals["good"].iloc[0, 0] = np.nan  # còn 1/2 tín hiệu < 0.6 -> loại
    X, y, period, column = ml.build_rows(signals, forward, min_names=10)
    first = column[period == 0]
    assert 0 not in first and len(first) == 11
    signals["extra"] = signals["noise"] * 0.0
    X, y, period, column = ml.build_rows(signals, forward, min_names=10)
    assert 0 in column[period == 0]  # 2/3 tín hiệu -> giữ, điền 0
    assert X[(period == 0) & (column == 0)][0, 0] == 0.0


def test_rows_without_a_label_are_kept_for_prediction_only():
    signals, forward = _market(periods=5, names=12)
    X, y, period, column = ml.build_rows(signals, forward, min_names=10)
    assert np.isnan(y[period == 4]).all() and np.isfinite(y[period < 4]).all()


def test_labels_are_ranks_within_each_period():
    signals, forward = _market(periods=5, names=12)
    forward.iloc[1] += 5.0  # cả thị trường tăng: phân vị trong kỳ không đổi
    _, y, period, _ = ml.build_rows(signals, forward, min_names=10)
    assert y[period == 1].min() == pytest.approx(1 / 12) and y[period == 1].max() == 1.0


def test_training_never_sees_the_purged_or_future_periods():
    signals, forward = _market()
    seen = []

    def spy(X, y, sizes, X_test):
        seen.append((len(sizes), len(X_test)))
        assert np.isfinite(y).all() and sum(sizes) == len(X)
        return np.zeros(len(X_test))

    ml.walk_forward_scores(signals, forward, spy, SPEC)
    assert seen[0][0] == SPEC.min_train_periods - SPEC.purge  # kỳ 0..27 cho khối 30..39
    assert max(count for count, _ in seen) == SPEC.train_periods - SPEC.purge


def test_changing_the_future_does_not_change_earlier_scores():
    signals, forward = _market()
    before = ml.walk_forward_scores(signals, forward, ml.ridge_model(), SPEC)
    changed = forward.copy()
    changed.iloc[80:] = -changed.iloc[80:]
    after = ml.walk_forward_scores(signals, changed, ml.ridge_model(), SPEC)
    pd.testing.assert_frame_equal(before.iloc[:80], after.iloc[:80])
    assert not before.iloc[90:110].equals(after.iloc[90:110])


def test_scores_start_only_after_the_minimum_history():
    signals, forward = _market()
    scores = ml.walk_forward_scores(signals, forward, ml.ridge_model(), SPEC)
    assert scores.iloc[: SPEC.min_train_periods].isna().all().all()
    assert scores.iloc[SPEC.min_train_periods :].notna().all().all()


@pytest.mark.parametrize("factory", [ml.ridge_model, ml.lambdarank_model, ml.mlp_model])
def test_models_learn_a_real_relationship(factory):
    if factory is ml.lambdarank_model:
        pytest.importorskip("lightgbm")
    signals, forward = _market(strength=2.0)
    scores = ml.walk_forward_scores(signals, forward, factory(), SPEC)
    ic = ml.rank_ic(scores, forward)
    assert ic["mean"] > 0.3 and ic["t_stat"] > 5


def test_models_find_nothing_in_noise():
    signals, forward = _market(strength=0.0)
    scores = ml.walk_forward_scores(signals, forward, ml.ridge_model(), SPEC)
    assert abs(ml.rank_ic(scores, forward)["t_stat"]) < 3
