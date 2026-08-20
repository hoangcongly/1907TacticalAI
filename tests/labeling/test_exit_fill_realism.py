import pytest
import numpy as np
from aegis.labeling.trailing_exit import finalize_trade_record

def test_exit_fill_slippage_is_strictly_worse():
    """
    F7 / G1: Bất biến hướng: fill_price_exit luôn bất lợi hơn hoặc bằng giá lý thuyết (close).
    Slippage không bao giờ được có lợi.
    """
    full_closes = np.array([100.0, 105.0, 110.0, 100.0])

    partial_record_long = {
        "entry_idx": 0,
        "entry_price": 100.0,
        "exit_idx_absolute": 2,  # Close is 110.0
        "exit_idx_relative": 2,
        "side": 1,
        "leverage_used": 1.0,
        "exit_reason": "TRAIL",
        "p_i": 0.8,
        "p_chop_i": 0.2,
        "mode": "follow",
        "sl_initial": 90.0,
        "liquidation_price": 85.0,
        "boundary_truncated": False,
    }

    res_long = finalize_trade_record(
        partial_record=partial_record_long,
        full_closes=full_closes,
        size_notional=1000.0,
        fee_entry_rate=0.0004,
        fee_exit_rate=0.0004
    )
    # Long thoat o 110.0, nhung bi slippage -> gia phai < 110.0
    # gross_pnl for long = (exit - entry)/entry * size = (exit - 100)/100 * 1000
    # If exit was exactly 110.0, gross_pnl = 100.0; with slippage it must be < 100.0
    assert res_long["gross_pnl"] < 100.0, "Slippage chưa được áp dụng hoặc áp dụng sai hướng cho lệnh Long!"

    partial_record_short = {
        "entry_idx": 0,
        "entry_price": 100.0,
        "exit_idx_absolute": 2,  # Close is 110.0
        "exit_idx_relative": 2,
        "side": -1,
        "leverage_used": 1.0,
        "exit_reason": "TRAIL",
        "p_i": 0.2,
        "p_chop_i": 0.8,
        "mode": "fade",
        "sl_initial": 110.0,
        "liquidation_price": 115.0,
        "boundary_truncated": False,
    }

    res_short = finalize_trade_record(
        partial_record=partial_record_short,
        full_closes=full_closes,
        size_notional=1000.0,
        fee_entry_rate=0.0004,
        fee_exit_rate=0.0004
    )
    # Short thoat o 110.0. If exit exactly 110.0, gross_pnl = (100 - 110)/100 * 1000 = -100.0
    # Slippage for short: exit_price is actually > 110.0, so loss is bigger.
    # Therefore gross_pnl should be < -100.0 (more negative)
    assert res_short["gross_pnl"] < -100.0, "Slippage chưa được áp dụng hoặc áp dụng sai hướng cho lệnh Short!"
