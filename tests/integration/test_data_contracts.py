import pytest
import pandas as pd
import numpy as np
import pandera as pa

from aegis.core.schemas import (
    SignalBarSchema, 
    TradeRecordSchema, 
    assert_trade_records_match_bar_version
)

def test_signal_bar_schema_valid():
    df = pd.DataFrame({
        "bar_idx": [0, 1],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "timestamp_ms": [1600000000000, 1600000060000],
        "open": [100.0, 101.0],
        "high": [105.0, 102.0],
        "low": [99.0, 100.0],
        "close": [101.0, 101.5],
        "volume": [10.5, 5.0],
        "ofi": [0.5, -0.2],
        "tick_count": [150, 50],
        "is_toxic_flag": [False, True],
        "is_tail_event": [False, False],
        "insufficient_history": [False, True],
        "trend_score": [1.5, np.nan],
        "p_trend": [0.8, np.nan],
        "p_chop": [0.2, np.nan],
        "atr_14": [2.5, np.nan],
        "hurst_value": [0.6, np.nan],
        "d_star_used": [0.45, np.nan]
    })
    
    # Ép kiểu để khớp schema khắt khe
    df["bar_idx"] = df["bar_idx"].astype("int64")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["tick_count"] = df["tick_count"].astype("int64")
    
    # Validate bằng Pandera
    validated_df = SignalBarSchema.validate(df)
    assert not validated_df.empty

def test_behavioral_contract_insufficient_history_violation():
    # Cố tình gán insufficient_history = True nhưng vẫn điền trend_score -> Phải văng lỗi
    df = pd.DataFrame({
        "bar_idx": [0], "symbol": ["BTCUSDT"], "timestamp_ms": [1600000000000],
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0],
        "volume": [10.0], "ofi": [0.0], "tick_count": [10],
        "is_toxic_flag": [False], "is_tail_event": [False],
        "insufficient_history": [True], 
        "trend_score": [1.5],  # VIOLATION HERE
        "p_trend": [np.nan], "p_chop": [np.nan], "atr_14": [np.nan],
        "hurst_value": [np.nan], "d_star_used": [np.nan]
    })
    
    df["bar_idx"] = df["bar_idx"].astype("int64")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["tick_count"] = df["tick_count"].astype("int64")

    with pytest.raises(pa.errors.SchemaError):
        SignalBarSchema.validate(df)

def test_trade_record_schema_absolute_index_violation():
    df = pd.DataFrame({
        "schema_version": ["1.0.0"], "dataset_manifest_hash": ["dummy_hash"],
        "fold_id": [None], "symbol": ["BTCUSDT"],
        "entry_idx": [100], "entry_price": [50000.0],
        "p_i": [0.8], "p_chop_i": [0.2], "mode": ["follow"], "side": [1],
        "sl_initial": [49000.0], "size_notional": [1.0],
        "exit_idx_relative": [10],
        "exit_idx_absolute": [999], # VIOLATION HERE: 100 + 1 + 10 != 999
        "exit_reason": ["TRAIL"], "fill_price_exit": [51000.0],
        "boundary_truncated": [False], "fee_entry": [10.0], "fee_exit": [10.0],
        "funding_accrued": [0.0], "gross_pnl": [1000.0], "realized_return": [0.02]
    })
    
    with pytest.raises(pa.errors.SchemaError):
        TradeRecordSchema.validate(df)

def test_lineage_and_versioning_assertion():
    expected_hash = "abcdef123456"
    trade_df = pd.DataFrame({"dataset_manifest_hash": ["wrong_hash_789"]})
    
    with pytest.raises(AssertionError, match="FATAL: DATA LINEAGE MISMATCH"):
        assert_trade_records_match_bar_version(trade_df, expected_hash)