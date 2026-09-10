"""
Triple Consensus Feature Selection (MDI, MDA, SFI).
Module B-5-x (AFML Chapter 8).

Fixes applied:
- #3: Added event_times param for PurgedKFold compatibility
- #4: Materialize splits once to prevent generator exhaustion
- #6: Removed silent try/except that swallowed scoring errors
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Optional
from sklearn.metrics import log_loss, accuracy_score, f1_score
import copy


def compute_mdi(clf, feature_names: List[str]) -> pd.DataFrame:
    """
    Tính Mean Decrease Impurity (MDI).
    Chỉ áp dụng cho các mô hình dạng Tree-based Ensemble (như RandomForest).
    
    Args:
        clf: Mô hình đã fit, phải có thuộc tính estimators_ (RandomForest, BaggingClassifier, ...).
        feature_names: Danh sách tên feature đúng thứ tự.
        
    Returns:
        pd.DataFrame với cột MDI_mean và MDI_std, index = feature_names.
    """
    if not hasattr(clf, "estimators_"):
        raise ValueError("MDI yêu cầu mô hình dạng Tree-ensemble (có thuộc tính estimators_).")

    # Extract feature importances from each tree
    importances = {i: tree.feature_importances_ for i, tree in enumerate(clf.estimators_)}
    imp_df = pd.DataFrame.from_dict(importances, orient='index')
    imp_df.columns = feature_names

    # Handle NaNs
    imp_df = imp_df.replace([np.inf, -np.inf], np.nan)
    imp_df = imp_df.dropna(how='all', axis=1)

    out = pd.DataFrame({
        "MDI_mean": imp_df.mean(),
        "MDI_std": imp_df.std() * imp_df.shape[0] ** -0.5
    })
    return out


def _build_split_kwargs(X, event_times, cv_gen):
    """Helper: tạo kwargs cho cv_gen.split(), tương thích cả KFold lẫn PurgedKFold."""
    kwargs = {"X": X}
    # Chỉ truyền event_times nếu cv_gen thực sự nhận nó (PurgedKFold).
    # KFold/StratifiedKFold của sklearn sẽ crash nếu nhận keyword lạ.
    if event_times is not None and hasattr(cv_gen, 'embargo_bars'):
        kwargs["event_times"] = event_times
    return kwargs


def _resolve_prediction(fit_clf, X_data, scoring):
    """
    Helper: Quyết định gọi predict_proba hay predict dựa trên scoring function.
    Trả về prediction phù hợp cho hàm scoring.
    """
    needs_proba = hasattr(fit_clf, "predict_proba") and "loss" in getattr(scoring, "__name__", "").lower()
    if needs_proba:
        return fit_clf.predict_proba(X_data)
    return fit_clf.predict(X_data)


def compute_mda(clf, X: pd.DataFrame, y: pd.Series, cv_gen,
                scoring: Callable, is_higher_better: bool = True,
                random_state: int = 42,
                event_times: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Tính Mean Decrease Accuracy (MDA) bằng phương pháp OOS xáo trộn.
    
    Args:
        clf: Mô hình chưa fit (sẽ được deepcopy và fit trong mỗi fold).
        X: DataFrame features.
        y: Series labels.
        cv_gen: Cross-validator (KFold hoặc PurgedKFold).
        scoring: Hàm đánh giá (accuracy_score, log_loss, ...).
        is_higher_better: True nếu score càng cao càng tốt (Accuracy, F1).
                          False nếu score càng thấp càng tốt (LogLoss, MSE).
        random_state: Seed cho shuffle.
        event_times: pd.Series (Index=t0, Values=t1). BẮT BUỘC khi cv_gen là PurgedKFold.
        
    Returns:
        pd.DataFrame với cột MDA_mean và MDA_std, index = feature_names.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X phải là pandas DataFrame để giữ nguyên tên cột (feature_names).")

    rng = np.random.RandomState(random_state)
    feature_names = X.columns.tolist()

    # FIX #4: Materialize splits MỘT LẦN để tránh generator exhaustion
    split_kwargs = _build_split_kwargs(X, event_times, cv_gen)
    splits = list(cv_gen.split(**split_kwargs))

    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=feature_names)

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        fit_clf = copy.deepcopy(clf)
        fit_clf.fit(X_train, y_train)

        # FIX #6: Không nuốt exception. Nếu scoring crash, đó là bug ở caller.
        pred = _resolve_prediction(fit_clf, X_test, scoring)
        base_score = scoring(y_test, pred)
        scr0.loc[i] = base_score

        # OOS score after shuffling each feature
        for j in feature_names:
            X_test_shuffled = X_test.copy(deep=True)
            shuffled_vals = X_test_shuffled[j].values.copy()
            rng.shuffle(shuffled_vals)
            X_test_shuffled[j] = shuffled_vals

            pred_shuffled = _resolve_prediction(fit_clf, X_test_shuffled, scoring)
            shuffled_score = scoring(y_test, pred_shuffled)

            if is_higher_better:
                drop = base_score - shuffled_score
            else:
                drop = shuffled_score - base_score

            scr1.loc[i, j] = drop

    out = pd.DataFrame({
        "MDA_mean": scr1.mean(),
        "MDA_std": scr1.std() * scr1.shape[0] ** -0.5
    })
    return out


def compute_sfi(clf, X: pd.DataFrame, y: pd.Series, cv_gen,
                scoring: Callable, is_higher_better: bool = True,
                event_times: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Tính Single Feature Importance (SFI).
    Train/Test OOS với duy nhất 1 feature tại một thời điểm.
    
    Args:
        clf: Mô hình chưa fit.
        X: DataFrame features.
        y: Series labels.
        cv_gen: Cross-validator.
        scoring: Hàm đánh giá.
        is_higher_better: True nếu score càng cao càng tốt.
        event_times: pd.Series cho PurgedKFold.
        
    Returns:
        pd.DataFrame với cột SFI_mean và SFI_std, index = feature_names.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X phải là pandas DataFrame.")

    feature_names = X.columns.tolist()

    # FIX #3 + #4: Materialize splits MỘT LẦN, truyền event_times nếu có
    split_kwargs = _build_split_kwargs(X, event_times, cv_gen)
    splits = list(cv_gen.split(**split_kwargs))

    imp = pd.DataFrame(columns=feature_names)

    for j in feature_names:
        X_single = X[[j]]  # DataFrame with 1 column

        fold_scores = []
        for train_idx, test_idx in splits:  # Dùng lại list, KHÔNG gọi .split() lần nữa
            X_train, y_train = X_single.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X_single.iloc[test_idx], y.iloc[test_idx]

            fit_clf = copy.deepcopy(clf)
            fit_clf.fit(X_train, y_train)

            # FIX #6: Không nuốt exception
            pred = _resolve_prediction(fit_clf, X_test, scoring)
            score = scoring(y_test, pred)
            fold_scores.append(score)

        imp[j] = fold_scores

    out = pd.DataFrame({
        "SFI_mean": imp.mean(),
        "SFI_std": imp.std() * imp.shape[0] ** -0.5
    })
    return out


def triple_consensus_ranker(mdi_df: pd.DataFrame, mda_df: pd.DataFrame,
                            sfi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp điểm MDI, MDA, SFI thành một bảng xếp hạng (Ranking).
    Cột có Rank 1 là tốt nhất.
    """
    # Gộp 3 bảng
    consensus = pd.concat([mdi_df['MDI_mean'], mda_df['MDA_mean'], sfi_df['SFI_mean']], axis=1)

    # Xếp hạng (Descending: Giá trị càng cao, hạng càng nhỏ/số 1)
    consensus['Rank_MDI'] = consensus['MDI_mean'].rank(ascending=False)
    consensus['Rank_MDA'] = consensus['MDA_mean'].rank(ascending=False)
    consensus['Rank_SFI'] = consensus['SFI_mean'].rank(ascending=False)

    # Tính Consensus Score (Tổng hạng: số càng nhỏ càng tốt)
    consensus['Consensus_Rank_Sum'] = consensus['Rank_MDI'] + consensus['Rank_MDA'] + consensus['Rank_SFI']

    # Xếp hạng cuối cùng
    consensus['Final_Rank'] = consensus['Consensus_Rank_Sum'].rank(ascending=True)
    consensus = consensus.sort_values('Final_Rank')

    return consensus
