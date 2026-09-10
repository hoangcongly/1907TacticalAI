"""Unit tests for CPCVPipeline (Module F — v11.8 5-Step Process)."""
import numpy as np
import pandas as pd
import pytest
from aegis.pipelines.cpcv_pipeline import CPCVPipeline, filter_boundary_truncated_for_kelly_table
from aegis.features.signal_pipeline import build_signal_bars
from aegis.core.schemas import TradeRecordSchema


@pytest.fixture
def synthetic_signal_bars():
    n = 200
    np.random.seed(42)
    t0 = 1700000000000
    times = [t0 + i * 60000 for i in range(n)]
    
    closes = 100.0 + np.cumsum(np.random.normal(0, 0.5, n))
    highs = closes + np.abs(np.random.normal(0.5, 0.2, n))
    lows = closes - np.abs(np.random.normal(0.5, 0.2, n))
    opens = (highs + lows) / 2.0
    vols = np.random.exponential(100.0, n)
    
    ohlcv = pd.DataFrame({
        "timestamp_ms": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "tick_count": np.random.randint(50, 200, n),
    })
    return build_signal_bars(ohlcv, symbol="BTCUSDT", warmup_window=10)


def test_filter_boundary_truncated_for_kelly_table():
    records = [
        {"entry_idx": 1, "realized_return": 0.05, "boundary_truncated": False},
        {"entry_idx": 2, "realized_return": 0.02, "boundary_truncated": True},
        {"entry_idx": 3, "realized_return": -0.01, "boundary_truncated": False},
    ]
    clean, diag = filter_boundary_truncated_for_kelly_table(records)
    assert len(clean) == 2
    assert len(diag) == 1
    assert clean[0]["entry_idx"] == 1
    assert clean[1]["entry_idx"] == 3
    assert diag[0]["entry_idx"] == 2


def test_cpcv_pipeline_run(synthetic_signal_bars):
    # Dùng n_groups=4, n_test_groups=1 (4 folds) để test nhanh
    pipeline = CPCVPipeline(n_groups=4, n_test_groups=1, embargo_bars=5)
    res = pipeline.run(synthetic_signal_bars, num_bins=5)
    
    assert "trade_records" in res
    assert "clean_records" in res
    assert "diagnostic_records" in res
    assert "kelly_table_follow" in res
    assert "kelly_table_fade" in res
    assert "metrics" in res
    
    trade_df = pd.DataFrame(res["trade_records"])
    if not trade_df.empty:
        # Schema validation
        TradeRecordSchema.validate(trade_df)
        
        # Absolute indexing invariant: exit_idx_absolute == entry_idx + 1 + exit_idx_relative
        assert np.all(trade_df["exit_idx_absolute"] == trade_df["entry_idx"] + 1 + trade_df["exit_idx_relative"])
        assert np.all(trade_df["exit_timestamp_ms"] >= trade_df["entry_timestamp_ms"])
