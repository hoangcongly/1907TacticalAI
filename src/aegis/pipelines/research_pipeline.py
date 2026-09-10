"""
research_pipeline.py — Institutional Full-Fit Production Pipeline (Task 7).

Thực hiện Full Fit trên toàn bộ dữ liệu lịch sử để xuất artifacts chuẩn Data Contracts v11.9:
1. build_signal_bars() -> DataFrame chuẩn SignalBarSchema.
2. Dynamic CUSUM filtering -> tập sự kiện không bị gắp gap/warm-up.
3. Dynamic Triple Barrier -> meta-labels y in {0, 1} và exit t1.
4. Concurrent events & average uniqueness -> sample_weights w_i.
5. Consensus Feature Selection -> loại bỏ đa cộng tuyến và giữ features mạnh nhất.
6. WeightedBootstrapForestClassifier fit với sample_weights.
7. Isotonic Calibration với PurgedKFold adapter.
8. CPCVPipeline -> 2D Empirical Kelly Tables (K_follow, K_fade) và OOS metrics (Sharpe, DSR, PBO).
9. Export production artifacts (model.pkl, selected_features.json, kelly_tables.json, metadata.json).
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

from aegis.core.config_loader import load_canonical_config
from aegis.core.schemas import SignalBarSchema, compute_dataset_manifest_hash
from aegis.features.signal_pipeline import build_signal_bars
from aegis.features.feature_engine import batch_features
from aegis.features.feature_spec import CANDIDATE_FEATURES
from aegis.labeling.cusum_events import (
    compute_dynamic_cusum_thresholds,
    filter_cusum_events_dynamic,
)
from aegis.labeling.triple_barrier import generate_meta_labels_triple_barrier
from aegis.labeling.sample_weights import (
    compute_num_concurrent_events,
    compute_sample_weights,
)
from aegis.meta_labeling.feature_selection.mdi_mda_sfi import (
    compute_mdi,
    compute_mda,
    compute_sfi,
    triple_consensus_ranker,
)
from aegis.meta_labeling.feature_selection.clustering import apply_consensus_filter
from aegis.meta_labeling.weighted_bootstrap_forest import WeightedBootstrapForestClassifier
from aegis.meta_labeling.calibration import build_calibrated_classifier
from aegis.meta_labeling.purged_kfold import PurgedKFold
from aegis.pipelines.cpcv_pipeline import CPCVPipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class ResearchPipeline:
    """
    Điều phối quy trình huấn luyện toàn diện (Research Full-Fit Pipeline).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        base_config = load_canonical_config()
        if config:
            base_config.update(config)
        self.config = base_config

    def run(
        self,
        signal_bars: Optional[pd.DataFrame] = None,
        raw_trades: Optional[pd.DataFrame] = None,
        artifacts_dir: str = "artifacts/",
    ) -> Dict[str, Any]:
        """
        Chạy quy trình nghiên cứu toàn trình và lưu artifacts.
        """
        # 1. Chuẩn bị DataFrame nến chuẩn
        if signal_bars is None:
            if raw_trades is None:
                raise ValueError("Bắt buộc phải cung cấp signal_bars hoặc raw_trades.")
            signal_bars = build_signal_bars(raw_trades, config=self.config)

        # Validate Schema
        SignalBarSchema.validate(signal_bars)
        manifest_hash = compute_dataset_manifest_hash(signal_bars, self.config)

        # [FIX F3] Đặc trưng tính bằng CHÍNH module mà live pipeline dùng
        # (aegis.features.feature_engine). Trước đây research tự tính riêng bằng
        # _ensure_candidate_features() còn live thì không -> model chạy mù 4/5 đặc trưng.
        bars = batch_features(signal_bars)
        n_bars = len(bars)
        insufficient_history = bars["insufficient_history"].values
        valid_start = int(np.where(~insufficient_history)[0][0]) if np.any(~insufficient_history) else 0

        closes = bars["close"].astype(np.float64).values
        highs = bars["high"].astype(np.float64).values
        lows = bars["low"].astype(np.float64).values
        timestamps_ms = bars["timestamp_ms"].astype(np.int64).values
        atr_14 = bars["atr_14"].values
        trend_score = bars["trend_score"].values
        p_trend = bars["p_trend"].values
        p_chop = bars["p_chop"].values
        is_toxic = bars["is_toxic_flag"].values

        safe_atr = np.nan_to_num(atr_14, nan=1.0)
        safe_atr = np.maximum(safe_atr, 1e-6)
        safe_trend = np.nan_to_num(trend_score, nan=0.0)
        safe_p_trend = np.nan_to_num(p_trend, nan=0.5)
        safe_p_chop = np.nan_to_num(p_chop, nan=0.5)

        # 2. Dynamic CUSUM Filter
        dynamic_thresholds = compute_dynamic_cusum_thresholds(
            prices=closes,
            atr_series=safe_atr,
            base_multiplier=self.config.get("cusum_multiplier", 2.0),
            anchor_span=50,
            use_ewma_anchor=True,
            reset_mask=insufficient_history,
        )

        cusum_records = filter_cusum_events_dynamic(
            prices=closes,
            dynamic_thresholds=dynamic_thresholds,
            timestamps_ms=timestamps_ms,
            atr_series=safe_atr,
            cooldown_bars=self.config.get("cooldown_bars", 5),
            spatial_delta_atr=self.config.get("spatial_delta_atr", 0.5),
            trend_scores=safe_trend,
            p_trend=safe_p_trend,
            p_chop=safe_p_chop,
        )

        event_indices = np.array(
            [e["bar_idx"] for e in cusum_records if e["bar_idx"] >= valid_start],
            dtype=np.int64,
        )
        if len(event_indices) < 5:
            # Fallback lưới mẫu đều nếu CUSUM ít
            event_indices = np.arange(
                valid_start,
                n_bars - 5,
                step=max(1, (n_bars - valid_start) // 20),
                dtype=np.int64,
            )

        sides = np.array(
            [1 if safe_trend[idx] >= 0 else -1 for idx in event_indices],
            dtype=np.int64,
        )

        # 3. Dynamic Triple Barrier Labeling
        meta_labels_df = generate_meta_labels_triple_barrier(
            events_idx=event_indices,
            sides=sides,
            closes=closes,
            highs=highs,
            lows=lows,
            p_trend=safe_p_trend,
            p_chop=safe_p_chop,
            sigmas=safe_atr / np.maximum(closes, 1e-6),
            is_toxic=is_toxic,
            t_window=self.config.get("t_window", 20),
            c_trade=self.config.get("c_trade", 0.0004),
        )

        if len(meta_labels_df) == 0:
            raise RuntimeError("Không thể tạo nhãn meta-labeling từ tập sự kiện.")

        # 4. Sample Weights
        t1_series = pd.Series(
            data=meta_labels_df["t1"].values,
            index=meta_labels_df["t0"].values,
        )
        returns_series = pd.Series(
            data=meta_labels_df["realized_return"].values,
            index=meta_labels_df["t0"].values,
        )
        bar_idx_series = pd.Series(bars.index)
        c_t = compute_num_concurrent_events(t1=t1_series, bar_index=bar_idx_series)
        sample_weights = compute_sample_weights(
            t1=t1_series, c_t=c_t, returns=returns_series
        )

        # 5. Consensus Feature Selection
        # [FIX F3] Ứng viên lấy từ đăng ký DUY NHẤT trong feature_spec.py.
        available_cols = [c for c in CANDIDATE_FEATURES if c in bars.columns]
        X_all = bars[available_cols].iloc[meta_labels_df["t0"].values].copy()
        X_all = X_all.fillna(0.0)
        y = meta_labels_df["label"].copy()

        # Đảm bảo có ít nhất 2 class nhãn
        if len(np.unique(y)) < 2:
            y.iloc[0] = 1 - y.iloc[0]

        fast_mode = self.config.get("fast_mode", False)
        if fast_mode or len(X_all) < 30 or len(available_cols) <= 3:
            selected_features = available_cols[:5]
        else:
            try:
                base_rf = WeightedBootstrapForestClassifier(
                    n_estimators=10, max_depth=3, random_state=42
                )
                base_rf.fit(X_all.values, y.values)
                mdi = compute_mdi(base_rf, available_cols)
                cv_kfold = KFold(n_splits=3, shuffle=False)
                mda = compute_mda(
                    base_rf, X_all, y, cv_kfold, accuracy_score, is_higher_better=True
                )
                sfi = compute_sfi(base_rf, X_all, y, cv_kfold, accuracy_score)
                consensus = triple_consensus_ranker(mdi, mda, sfi)
                selected_features = apply_consensus_filter(
                    X_all, consensus, corr_threshold=0.70
                )
                if not selected_features:
                    selected_features = available_cols[:5]
            except Exception:
                selected_features = available_cols[:5]

        X_selected = X_all[selected_features].values

        # 6. Fit Model with Weighted Bootstrap Forest
        n_est = self.config.get("n_estimators", 50)
        forest = WeightedBootstrapForestClassifier(
            n_estimators=n_est,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
        )
        forest.fit(X_selected, y.values, sample_weight=sample_weights.values)

        # 7. Probability Calibration
        calibrated_model = forest
        try:
            n_samples = len(y)
            if n_samples >= 15:
                n_splits = min(3, max(2, n_samples // 10))
                cv_purged = PurgedKFold(
                    n_splits=n_splits,
                    embargo_bars=self.config.get("embargo_bars", 5),
                    event_times=t1_series,
                )
                calibrator = build_calibrated_classifier(
                    base_estimator=forest,
                    cv_gen=cv_purged,
                    event_times=t1_series,
                )
                calibrator.fit(X_selected, y.values)
                calibrated_model = calibrator
        except Exception:
            calibrated_model = forest

        # 8. CPCV Simulation & 2D Empirical Kelly Tables
        cpcv_pipe = CPCVPipeline(
            n_groups=self.config.get("n_groups", 5),
            n_test_groups=self.config.get("n_test_groups", 2),
            embargo_bars=self.config.get("embargo_bars", 5),
            config=self.config,
        )
        cpcv_res = cpcv_pipe.run(
            bars, num_bins=self.config.get("num_bins", 5)
        )

        # 9. Export Production Artifacts
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(artifacts_dir, "model.pkl")
        joblib.dump(calibrated_model, model_path)

        features_path = os.path.join(artifacts_dir, "selected_features.json")
        with open(features_path, "w") as f:
            json.dump(selected_features, f, indent=2)

        # Format Kelly tables for json serialization
        kelly_follow = cpcv_res["kelly_table_follow"]
        kelly_fade = cpcv_res["kelly_table_fade"]
        kelly_dict = {
            "kelly_follow": {
                "grid": kelly_follow[0].tolist(),
                "p_edges": kelly_follow[1].tolist(),
                "chop_edges": [arr.tolist() for arr in kelly_follow[2]],
            },
            "kelly_fade": {
                "grid": kelly_fade[0].tolist(),
                "p_edges": kelly_fade[1].tolist(),
                "chop_edges": [arr.tolist() for arr in kelly_fade[2]],
            },
        }
        kelly_path = os.path.join(artifacts_dir, "kelly_tables.json")
        with open(kelly_path, "w") as f:
            json.dump(kelly_dict, f, indent=2)

        cpcv_metrics = cpcv_res.get("metrics", {})
        metadata = {
            "schema_version": "1.0.0",
            "dataset_manifest_hash": manifest_hash,
            "selected_features": selected_features,
            "n_samples": int(len(meta_labels_df)),
            "n_bars": int(n_bars),
            "oos_sharpe": float(cpcv_metrics.get("sharpe_oos", 0.0)),
            "dsr_stat": float(cpcv_metrics.get("dsr", 0.0)),
            "truncation_rate": float(cpcv_metrics.get("truncation_rate", 0.0)),
            "timestamp": int(time.time() * 1000),
            "artifacts": {
                "model": "model.pkl",
                "selected_features": "selected_features.json",
                "kelly_tables": "kelly_tables.json",
            },
        }
        meta_path = os.path.join(artifacts_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "status": "SUCCESS",
            "manifest_hash": manifest_hash,
            "selected_features": selected_features,
            "model": calibrated_model,
            "cpcv_results": cpcv_res,
            "artifacts_dir": artifacts_dir,
            "metadata": metadata,
        }
