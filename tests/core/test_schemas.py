"""
Unit tests cho module schemas (Đặc biệt kiểm định tính bất biến của ImmutableTradeRecord - Lỗ hổng 9).
"""
import pytest
from dataclasses import FrozenInstanceError
from aegis.core.schemas import ImmutableTradeRecord, TradeRecord


def test_immutable_trade_record_mutation_raises_error():
    """Kiểm tra thuộc tính frozen=True ngăn chặn sửa đổi trực tiếp (Lỗ hổng 9)."""
    record = ImmutableTradeRecord(
        schema_version="1.0",
        dataset_manifest_hash="abc_hash",
        symbol="BTC/USDT",
        entry_idx=10,
        entry_timestamp_ms=1700000000000,
        entry_price=50000.0,
        p_i=0.8,
        p_chop_i=0.2,
        mode="follow",
        side=1,
        sl_initial=49000.0,
        size_notional=10000.0,
        exit_idx_relative=5,
        exit_idx_absolute=15,
        exit_timestamp_ms=1700000300000,
        exit_reason="SL",
        fill_price_exit=49000.0,
        boundary_truncated=False,
        fee_entry=5.0,
        fee_exit=4.9,
        funding_accrued=0.1,
        gross_pnl=-1000.0,
        realized_return=-0.02,
        fold_id="fold_0"
    )

    with pytest.raises(FrozenInstanceError):
        record.entry_price = 51000.0  # type: ignore

    with pytest.raises(FrozenInstanceError):
        record.gross_pnl = -500.0  # type: ignore


def test_immutable_trade_record_from_dict_and_update():
    """Kiểm tra khởi tạo từ dict và hàm update trả về instance mới không ảnh hưởng bản cũ."""
    raw_dict: TradeRecord = {
        "schema_version": "1.0",
        "dataset_manifest_hash": "abc_hash",
        "fold_id": "fold_1",
        "symbol": "ETH/USDT",
        "entry_idx": 100,
        "entry_timestamp_ms": 1700000000000,
        "entry_price": 3000.0,
        "p_i": 0.7,
        "p_chop_i": 0.3,
        "mode": "follow",
        "side": 1,
        "sl_initial": 2900.0,
        "size_notional": 6000.0,
        "exit_idx_relative": 10,
        "exit_idx_absolute": 110,
        "exit_timestamp_ms": 1700000600000,
        "exit_reason": "TRAIL",
        "fill_price_exit": 3200.0,
        "boundary_truncated": False,
        "fee_entry": 3.0,
        "fee_exit": 3.2,
        "funding_accrued": 0.0,
        "gross_pnl": 400.0,
        "realized_return": 0.0667,
    }

    record = ImmutableTradeRecord.from_dict(raw_dict)
    assert record.symbol == "ETH/USDT"
    assert record.entry_price == 3000.0

    # Test update() returns a NEW object with modified fields
    updated_record = record.update(fill_price_exit=3250.0, gross_pnl=500.0)
    assert updated_record is not record
    assert updated_record.fill_price_exit == 3250.0
    assert updated_record.gross_pnl == 500.0
    # Original record remains completely unmodified
    assert record.fill_price_exit == 3200.0
    assert record.gross_pnl == 400.0
    assert record.to_dict()["gross_pnl"] == 400.0
