import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from aegis.pipelines.research_pipeline import ResearchPipeline
from aegis.core.schemas import SignalBarSchema, compute_dataset_manifest_hash


@pytest.fixture
def synthetic_signal_bars():
    np.random.seed(42)
    n_bars = 120
    timestamps = 1700000000000 + np.arange(n_bars) * 60000
    prices = 100.0 + np.cumsum(np.random.normal(0.05, 0.5, n_bars))
    highs = prices + np.abs(np.random.normal(0.5, 0.2, n_bars))
    lows = prices - np.abs(np.random.normal(0.5, 0.2, n_bars))
    volumes = np.random.uniform(10.0, 100.0, n_bars)

    df = pd.DataFrame({
        "bar_idx": np.arange(n_bars, dtype=np.int64),
        "symbol": "BTCUSDT",
        "timestamp_ms": timestamps,
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "ofi": np.clip(np.random.normal(0.0, 0.3, n_bars), -1.0, 1.0),
        "tick_count": np.random.randint(10, 50, n_bars, dtype=np.int64),
        "is_toxic_flag": np.zeros(n_bars, dtype=bool),
        "is_tail_event": np.zeros(n_bars, dtype=bool),
        "insufficient_history": np.array([True] * 20 + [False] * (n_bars - 20), dtype=bool),
        "trend_score": [np.nan] * 20 + list(np.random.normal(0.2, 0.5, n_bars - 20)),
        "p_trend": [np.nan] * 20 + list(np.random.uniform(0.4, 0.8, n_bars - 20)),
        "p_chop": [np.nan] * 20 + list(np.random.uniform(0.2, 0.6, n_bars - 20)),
        "atr_14": [np.nan] * 20 + list(np.random.uniform(0.5, 1.5, n_bars - 20)),
        "hurst_value": [np.nan] * 20 + list(np.random.uniform(0.4, 0.6, n_bars - 20)),
        "d_star_used": [0.35] * n_bars,
    })
    return df


def test_research_pipeline_full_fit(synthetic_signal_bars):
    temp_dir = tempfile.mkdtemp()
    try:
        config = {
            "n_groups": 4,
            "n_test_groups": 1,
            "embargo_bars": 3,
            "n_estimators": 10,
            "fast_mode": True,
        }
        pipeline = ResearchPipeline(config=config)
        res = pipeline.run(signal_bars=synthetic_signal_bars, artifacts_dir=temp_dir)

        assert res["status"] == "SUCCESS"
        assert "manifest_hash" in res
        assert "selected_features" in res
        assert len(res["selected_features"]) > 0
        assert "cpcv_results" in res

        # Check artifacts written
        assert os.path.exists(os.path.join(temp_dir, "model.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "selected_features.json"))
        assert os.path.exists(os.path.join(temp_dir, "kelly_tables.json"))
        assert os.path.exists(os.path.join(temp_dir, "metadata.json"))

        # Verify model predict_proba
        model = res["model"]
        sample_x = np.zeros((1, len(res["selected_features"])))
        prob = model.predict_proba(sample_x)
        assert prob.shape[1] == 2
        assert 0.0 <= prob[0, 1] <= 1.0

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
