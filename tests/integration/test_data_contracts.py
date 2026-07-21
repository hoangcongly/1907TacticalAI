import pytest
import pandas as pd
import numpy as np
import pandera as pa

from aegis.core.schemas import (
    SignalBarSchema,
    TradeRecordSchema,
    TradeRecord,
    compute_dataset_manifest_hash,
    assert_trade_records_match_bar_version,
    check_insufficient_history_nulls,
)
import inspect
from aegis.execution.pnl import compute_realized_pnl


def test_signal_bar_schema_valid():
    df = pd.DataFrame(
        {
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
            "d_star_used": [0.45, np.nan],
        }
    )

    # Ép kiểu để khớp schema khắt khe
    df["bar_idx"] = df["bar_idx"].astype("int64")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["tick_count"] = df["tick_count"].astype("int64")

    # Validate bằng Pandera
    validated_df = SignalBarSchema.validate(df)
    assert not validated_df.empty


def test_behavioral_contract_insufficient_history_violation():
    # Cố tình gán insufficient_history = True nhưng vẫn điền trend_score -> Phải văng lỗi
    df = pd.DataFrame(
        {
            "bar_idx": [0],
            "symbol": ["BTCUSDT"],
            "timestamp_ms": [1600000000000],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [10.0],
            "ofi": [0.0],
            "tick_count": [10],
            "is_toxic_flag": [False],
            "is_tail_event": [False],
            "insufficient_history": [True],
            "trend_score": [1.5],  # VIOLATION HERE
            "p_trend": [np.nan],
            "p_chop": [np.nan],
            "atr_14": [np.nan],
            "hurst_value": [np.nan],
            "d_star_used": [np.nan],
        }
    )

    df["bar_idx"] = df["bar_idx"].astype("int64")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["tick_count"] = df["tick_count"].astype("int64")

    with pytest.raises(pa.errors.SchemaError):
        SignalBarSchema.validate(df)


def test_behavioral_contract_insufficient_history_mixed_violation():
    # Kiểm thử dữ liệu trộn lẫn (mixed index): có cả insufficient_history = False (hợp lệ) và True (vi phạm ở dòng thứ 3)
    df = pd.DataFrame(
        {
            "bar_idx": [0, 1, 2],
            "symbol": ["BTCUSDT"] * 3,
            "timestamp_ms": [1600000000000, 1600000060000, 1600000120000],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [10.0, 5.0, 8.0],
            "ofi": [0.1, -0.1, 0.2],
            "tick_count": [10, 15, 20],
            "is_toxic_flag": [False] * 3,
            "is_tail_event": [False] * 3,
            "insufficient_history": [False, True, True],
            "trend_score": [
                1.5,
                np.nan,
                2.0,
            ],  # Dòng 0 hợp lệ (False -> có số), Dòng 1 hợp lệ (True -> Null), Dòng 2 VI PHẠM (True -> có số 2.0)
            "p_trend": [0.8, np.nan, np.nan],
            "p_chop": [0.2, np.nan, np.nan],
            "atr_14": [2.5, np.nan, np.nan],
            "hurst_value": [0.6, np.nan, np.nan],
            "d_star_used": [0.45, np.nan, np.nan],
        }
    )

    df["bar_idx"] = df["bar_idx"].astype("int64")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
    df["tick_count"] = df["tick_count"].astype("int64")

    with pytest.raises(pa.errors.SchemaError):
        SignalBarSchema.validate(df)


def test_trade_record_schema_absolute_index_violation():
    df = pd.DataFrame(
        {
            "schema_version": ["1.0.0"],
            "dataset_manifest_hash": ["dummy_hash"],
            "fold_id": [None],
            "symbol": ["BTCUSDT"],
            "entry_idx": [100],
            "entry_price": [50000.0],
            "p_i": [0.8],
            "p_chop_i": [0.2],
            "mode": ["follow"],
            "side": [1],
            "sl_initial": [49000.0],
            "size_notional": [1.0],
            "exit_idx_relative": [10],
            "exit_idx_absolute": [999],  # VIOLATION HERE: 100 + 1 + 10 != 999
            "exit_reason": ["TRAIL"],
            "fill_price_exit": [51000.0],
            "boundary_truncated": [False],
            "fee_entry": [10.0],
            "fee_exit": [10.0],
            "funding_accrued": [0.0],
            "gross_pnl": [1000.0],
            "realized_return": [0.02],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        TradeRecordSchema.validate(df)


def test_lineage_and_versioning_assertion():
    expected_hash = "abcdef123456"
    trade_df = pd.DataFrame({"dataset_manifest_hash": ["wrong_hash_789"]})

    with pytest.raises(AssertionError, match="FATAL: DATA LINEAGE MISMATCH"):
        assert_trade_records_match_bar_version(trade_df, expected_hash)


def test_check_insufficient_history_nulls_mixed_true_false():
    """
    Case trước đây KHÔNG được test: dữ liệu trộn lẫn cả True lẫn False trong
    insufficient_history — đây chính là case mà bug index-alignment lộ ra.
    """
    df = pd.DataFrame({
        "insufficient_history": [True, False, True, False, False],
        "trend_score": [None, 0.5, None, 0.3, 0.1],
        "p_trend": [None, 0.6, None, 0.4, 0.2],
        "p_chop": [None, 0.4, None, 0.6, 0.8],
        "atr_14": [None, 1.2, None, 1.5, 1.1],
        "hurst_value": [None, 0.55, None, 0.45, 0.5],
    })
    result = check_insufficient_history_nulls(df)

    assert len(result) == len(df), f"Độ dài lệch: result={len(result)}, df={len(df)}"
    assert list(result.index) == list(df.index), "Index không khớp df gốc"
    assert result.tolist() == [True, True, True, True, True], (
        f"Kỳ vọng toàn True (không dòng nào vi phạm), nhận {result.tolist()}"
    )

    # Case vi phạm: dòng insufficient_history=True nhưng có giá trị không null
    df_bad = df.copy()
    df_bad.loc[0, "trend_score"] = 0.9  # vi phạm: insufficient_history=True nhưng có số
    result_bad = check_insufficient_history_nulls(df_bad)
    assert result_bad.tolist() == [False, True, True, True, True], (
        f"Kỳ vọng dòng 0 = False (vi phạm), nhận {result_bad.tolist()}"
    )

    print("✅ test_check_insufficient_history_nulls_mixed_true_false PASSED")

def test_schema_column_count_matches_typeddict():
    """
    [TASK B-1] (Phát hiện 7) Đảm bảo số lượng cột trong TradeRecordSchema
    phải khớp chính xác với số lượng trường định nghĩa trong TradeRecord (TypedDict),
    tránh trường hợp schema bị cập nhật sót so với logic code (Magic Number column count).
    """
    schema_cols = set(TradeRecordSchema.columns.keys())
    # Lấy các trường (keys) từ TradeRecord TypedDict
    # typing.get_type_hints hoặc __annotations__ đều được
    import typing
    typed_dict_keys = set(typing.get_type_hints(TradeRecord).keys())
    
    missing_in_schema = typed_dict_keys - schema_cols
    missing_in_dict = schema_cols - typed_dict_keys
    
    assert schema_cols == typed_dict_keys, (
        f"Lệch cột giữa Schema và TypedDict.\n"
        f"Thiếu trong Schema: {missing_in_schema}\n"
        f"Thiếu trong TypedDict: {missing_in_dict}"
    )

def test_leverage_does_not_affect_pnl_for_non_liquidated_exits():
    """
    [TASK B-1] Khóa cứng hàm tính PnL thường (compute_realized_pnl).
    Với các lệnh không phải thanh lý, PnL CHỈ phụ thuộc vào side, size_notional, fill_price, fee.
    Thay đổi đòn bẩy (leverage) sẽ KHÔNG làm thay đổi kết quả PnL.
    """
    res_lev_2 = compute_realized_pnl(
        entry_price=100.0, exit_price=110.0, side=1, size_notional=1000.0,
        leverage=2.0, exit_reason="TRAIL"
    )
    res_lev_10 = compute_realized_pnl(
        entry_price=100.0, exit_price=110.0, side=1, size_notional=1000.0,
        leverage=10.0, exit_reason="TRAIL"
    )
    
    assert res_lev_2["net_pnl"] == res_lev_10["net_pnl"], (
        "Lỗi kiến trúc: Đòn bẩy đã làm rò rỉ và thay đổi PnL của một lệnh thoát bình thường!"
    )


def test_schemas_empty_dataframes_safe():
    """Kiểm chứng hệ thống xử lý an toàn khi mảng nến hoặc bảng giao dịch rỗng (không crash ValueError / AssertionError)."""
    df_bar = pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close"])
    h = compute_dataset_manifest_hash(df_bar, {})
    assert isinstance(h, str) and len(h) == 64
    
    df_trade = pd.DataFrame(columns=["dataset_manifest_hash"])
    # Không được ném ngoại lệ khi bảng trade rỗng (0 giao dịch trong fold)
    assert_trade_records_match_bar_version(df_trade, h)
    print("✅ [SCHEMAS] Empty DataFrames handling PASSED!")


