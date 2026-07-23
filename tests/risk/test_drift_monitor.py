"""
Bộ kiểm thử cho drift_monitor.py (Task B-2-1 / Module J).
"""

import os
from pathlib import Path
import pytest
import numpy as np
from aegis.risk.drift_monitor import (
    refresh_cusum_thresholds,
    monitor_brier_score_cusum_drift,
)


def test_b_2_1_refresh_cusum_thresholds_json_export(tmp_path: Path):
    """
    [TDD VERIFICATION - REFRESH CUSUM THRESHOLDS & RTK EXPORT NAMESPACED]:
    Kiểm tra tính toán thống kê ngưỡng CUSUM giá và xuất cấu hình sang `price_cusum_thresholds.json`.
    """
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    atr = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    out_file = tmp_path / "artifacts" / "price_cusum_thresholds.json"

    res = refresh_cusum_thresholds(
        prices=prices,
        atr_series=atr,
        base_multiplier=2.5,
        anchor_span=10,
        min_rel_threshold=0.001,
        max_rel_threshold=0.05,
        artifact_output_path=out_file,
    )

    assert os.path.exists(out_file)
    assert res["base_multiplier"] == 2.5
    assert "summary_stats" in res
    assert res["summary_stats"]["mean"] > 0.0
    assert len(res["thresholds_sample_tail"]) <= 10


def test_monitor_brier_score_cusum_drift_and_reset(tmp_path: Path):
    """
    [TDD VERIFICATION - BRIER SCORE DRIFT MONITOR WITH PAGE RESET & NAMESPACED EXPORT]:
    Kiểm chứng phát hiện trôi mô hình khi sai số Brier vượt ngưỡng và xuất độc lập ra `brier_drift_cusum_thresholds.json`.
    """
    scores = np.array([0.25, 0.26, 0.16, 0.14], dtype=np.float64)
    out_file_brier = tmp_path / "artifacts" / "brier_drift_cusum_thresholds.json"

    res = monitor_brier_score_cusum_drift(
        brier_scores=scores,
        baseline_brier=0.15,
        drift_threshold=0.20,
        reset_on_alarm=True,
        artifact_output_path=out_file_brier,
    )

    assert os.path.exists(out_file_brier)
    assert res["alarms_triggered"] is True
    assert res["alarms_count"] == 1
    assert res["alarms_details"][0]["bar_idx"] == 1
    assert res["alarms_details"][0]["s_plus_at_trigger"] == pytest.approx(0.21)
    assert res["final_s_plus"] == pytest.approx(0.0)

