"""Unit tests for AegisSignalEngine & signal_pipeline.py."""
import numpy as np
import pandas as pd
import pytest
from aegis.features.signal_pipeline import AegisSignalEngine, build_signal_bars
from aegis.core.schemas import SignalBarSchema


@pytest.fixture
def synthetic_ohlcv():
    n = 100
    np.random.seed(42)
    t0 = 1700000000000
    times = [t0 + i * 60000 for i in range(n)]
    
    closes = 100.0 + np.cumsum(np.random.normal(0, 0.5, n))
    highs = closes + np.abs(np.random.normal(0.5, 0.2, n))
    lows = closes - np.abs(np.random.normal(0.5, 0.2, n))
    opens = (highs + lows) / 2.0
    vols = np.random.exponential(100.0, n)
    
    return pd.DataFrame({
        "timestamp_ms": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "tick_count": np.random.randint(50, 200, n),
    })


def test_process_ohlcv_to_signal_bars_schema_compliance(synthetic_ohlcv):
    engine = AegisSignalEngine(symbol="BTCUSDT", warmup_window=10)
    bars_df = engine.process_ohlcv_to_signal_bars(synthetic_ohlcv)
    
    assert len(bars_df) == len(synthetic_ohlcv)
    # Validate against Pandera schema with strict checks
    validated = SignalBarSchema.validate(bars_df)
    assert not validated.empty


def test_insufficient_history_behavioral_contract(synthetic_ohlcv):
    warmup = 15
    engine = AegisSignalEngine(symbol="BTCUSDT", warmup_window=warmup)
    bars_df = engine.process_ohlcv_to_signal_bars(synthetic_ohlcv)
    
    # First 15 bars must have insufficient_history = True
    assert bars_df["insufficient_history"].iloc[:warmup].all()
    # Signal columns must be null for warmup bars
    for col in ["trend_score", "p_trend", "p_chop", "atr_14", "hurst_value", "d_star_used"]:
        assert bars_df[col].iloc[:warmup].isna().all()
        
    # After warmup, insufficient_history = False and signals must be populated
    assert not bars_df["insufficient_history"].iloc[warmup:].any()
    assert bars_df["p_trend"].iloc[warmup:].notna().all()
    assert bars_df["trend_score"].iloc[warmup:].notna().all()


def test_build_signal_bars_convenience_function(synthetic_ohlcv):
    df = build_signal_bars(synthetic_ohlcv, symbol="ETHUSDT", warmup_window=10)
    assert df["symbol"].iloc[0] == "ETHUSDT"
    SignalBarSchema.validate(df)
