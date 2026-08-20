import math
import numpy as np
import pytest
from aegis.labeling.trailing_exit import (
    compute_sl_initial,
    compute_regime_aware_trailing_exit_v3_liquidation_aware,
    simulate_trailing_exit_within_fold_bounds,
)

def test_b_1_3_compute_sl_initial():
    """
    Kiểm tra tính đối xứng gương tuyệt đối và kiểm thử bẻ gãy (Fault-Injection).
    """
    entry_price = 100.0
    m_sl = 2.0
    sigma = 0.05  # Biến động 5%
    c_trade_adj = 0.01  # Phí + Slippage = 1%

    # 1. Test cho phe Long (side = 1)
    sl_long = compute_sl_initial(
        entry_price, side=1, m_sl=m_sl, sigma=sigma, c_trade_adj=c_trade_adj
    )
    expected_sl_long = 100.0 * math.exp(-0.11)  # ~ 89.5834
    assert (
        abs(sl_long - expected_sl_long) < 1e-9
    ), f"Lỗi SL Long: Cần {expected_sl_long}, Nhận {sl_long}"

    # 2. Test cho phe Short/Fade (side = -1)
    sl_short = compute_sl_initial(
        entry_price, side=-1, m_sl=m_sl, sigma=sigma, c_trade_adj=c_trade_adj
    )
    expected_sl_short = 100.0 * math.exp(0.11)  # ~ 111.6278
    assert (
        abs(sl_short - expected_sl_short) < 1e-9
    ), f"Lỗi SL Short: Cần {expected_sl_short}, Nhận {sl_short}"

    # 3. Test tính đối xứng tuyệt đối (Hình học Logarithm)
    dist_long_log = math.log(entry_price / sl_long)
    dist_short_log = math.log(sl_short / entry_price)
    assert (
        abs(dist_long_log - dist_short_log) < 1e-9
    ), f"Lỗi Đối xứng Geometric: Long ({dist_long_log}) != Short ({dist_short_log})"

    # 4. [ARMOR-PLATED TESTS] Bắt lỗi nghiêm ngặt khi truyền side = 0
    try:
        compute_sl_initial(
            entry_price, side=0, m_sl=m_sl, sigma=sigma, c_trade_adj=c_trade_adj
        )
        assert False, "Lỗi rò rỉ: side=0 lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "side bắt buộc phải là +1" in str(e)

    # 5. [ARMOR-PLATED TESTS] Bắt lỗi số âm / NaN / Inf
    invalid_inputs = [
        (-100.0, 1, 2.0, 0.05, 0.01),
        (100.0, 1, -2.0, 0.05, 0.01),
        (100.0, 1, 2.0, -0.05, 0.01),
        (float("nan"), 1, 2.0, 0.05, 0.01),
        (100.0, 1, float("inf"), 0.05, 0.01),
    ]
    for ep, sd, m, sig, cadj in invalid_inputs:
        try:
            compute_sl_initial(ep, sd, m, sig, cadj)
            assert (
                False
            ), f"Lỗi rò rỉ: Input dị thường ({ep}, {sd}, {m}, {sig}, {cadj}) không bị chặn!"
        except ValueError:
            pass

    # 6. [ARMOR-PLATED TESTS] Bắt lỗi rủi ro vượt quá 100% với lệnh Long
    try:
        compute_sl_initial(
            entry_price=100.0, side=1, m_sl=3.0, sigma=0.40, c_trade_adj=0.05, max_reasonable_cushion=2.0 # override để lọt qua max_reasonable_cushion
        )  # Total = 1.25 (125%)
        assert False, "Lỗi rò rỉ: Stop-loss âm lọt qua mà không bị chặn!"
    except ValueError as e:
        assert "Tổng rủi ro trừ hao" in str(e)

    # 7. [ARMOR-PLATED TESTS] Bắt lỗi cushion lớn bất thường (đối xứng cho cả 2 chiều)
    try:
        compute_sl_initial(entry_price=100.0, side=-1, m_sl=2.0, sigma=0.30, c_trade_adj=0.01) # Total = 0.61 > 0.5
        assert False, "Lỗi rò rỉ: Cushion lớn phi lý lọt qua mà không bị chặn!"
    except ValueError as e:
        assert "vượt ngưỡng hợp lý" in str(e)

    print(
        "✅ [TASK B-1-3] compute_sl_initial PASSED! (Đối xứng gương hoàn hảo & Khóa 100% lỗ hổng side=0 / NaN / SL âm)"
    )


def test_b_1_4_regime_aware_trailing_exit_symmetry():
    """
    [TDD] Kiểm tra tính đối xứng, logic Regime-Flip và khả năng bẻ gãy lỗ hổng (Fault-Injection).
    """
    atr = np.full(10, 1.0)
    p_trend_flat = np.full(10, 0.5)

    # Test 1: side=+1 (Long/Follow), giá giảm chạm SL tại k=3
    highs_l = np.array([101, 102, 103, 90, 90, 90, 90, 90, 90, 90], dtype=float)
    lows_l = np.array([100, 101, 102, 85, 85, 85, 85, 85, 85, 85], dtype=float)
    res_l = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        100.0, 1, "follow", highs_l, lows_l, atr, p_trend_flat, 95.0, t_max_live=10
    )
    assert (
        res_l["reason"] == "SL" and res_l["exit_idx"] == 3
    ), f"Long SL test FAILED: {res_l}"

    # Test 2: side=-1 (Short/Fade), giá tăng vượt SL tại k=3
    highs_s = np.array(
        [101, 102, 102.5, 110, 110, 110, 110, 110, 110, 110], dtype=float
    )
    lows_s = np.array([100, 101, 101.5, 105, 105, 105, 105, 105, 105, 105], dtype=float)
    res_s = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        100.0, -1, "fade", highs_s, lows_s, atr, p_trend_flat, 105.0, t_max_live=10
    )
    assert (
        res_s["reason"] == "SL" and res_s["exit_idx"] == 3
    ), f"Short SL test FAILED: {res_s}"

    # Test 3: Regime-Flip cho Fade — p_trend TĂNG vượt ngưỡng 0.65 phải kích hoạt thoát tại k=3 (cần 2 nến liên tiếp)
    highs_f = np.full(10, 100.5)
    lows_f = np.full(10, 99.5)
    p_trend_rising = np.array([0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    res_f_flip = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        100.0,
        -1,
        "fade",
        highs_f,
        lows_f,
        atr,
        p_trend_rising,
        110.0,
        t_max_live=10,
        p_trend_exit_threshold_fade=0.65,
        consecutive_bars_required=2,
    )
    assert (
        res_f_flip["reason"] == "REGIME_FLIP" and res_f_flip["exit_idx"] == 3
    ), f"Fade regime-flip test FAILED: {res_f_flip}"

    # Test 4: Follow cùng chuỗi p_trend_rising — KHÔNG kích hoạt Regime-Flip vì Follow chỉ thoát khi trend yếu
    res_fl_no_flip = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        100.0,
        1,
        "follow",
        highs_f,
        lows_f,
        atr,
        p_trend_rising,
        90.0,
        t_max_live=10,
        p_trend_exit_threshold_follow=0.35,
        consecutive_bars_required=2,
    )
    assert (
        res_fl_no_flip["reason"] == "TIME_STOP"
    ), f"Follow false-positive test FAILED: {res_fl_no_flip}"

    # 5. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử mảng rỗng cận biên fold (Empty array -> v11.9 boundary truncated)
    res_empty = compute_regime_aware_trailing_exit_v3_liquidation_aware(100.0, 1, "follow", [], [], [], [], 95.0)
    assert res_empty is not None and res_empty.get("boundary_truncated") is True and res_empty.get("reason") == "TIME_STOP", f"Boundary truncated test FAILED: {res_empty}"

    # 6. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy độ dài mảng lệch nhau (Mismatched arrays check)
    try:
        compute_regime_aware_trailing_exit_v3_liquidation_aware(
            100.0, 1, "follow", highs_l[:5], lows_l[:4], atr[:5], p_trend_flat[:5], 95.0
        )
        assert False, "Lỗi rò rỉ: Mảng lệch độ dài lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "Độ dài các mảng tương lai lệch nhau" in str(e)

    # 7. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy ATR âm (Negative ATR check)
    atr_bad = atr.copy()
    atr_bad[2] = -1.0
    try:
        compute_regime_aware_trailing_exit_v3_liquidation_aware(
            100.0, 1, "follow", highs_l, lows_l, atr_bad, p_trend_flat, 95.0
        )
        assert False, "Lỗi rò rỉ: ATR âm lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "chứa số âm" in str(e)

    # 8. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy side = 0 hoặc trade_mode không hợp lệ
    try:
        compute_regime_aware_trailing_exit_v3_liquidation_aware(
            100.0, 0, "follow", highs_l, lows_l, atr, p_trend_flat, 95.0
        )
        assert False, "Lỗi rò rỉ: side=0 lọt qua mà không ném lỗi ValueError!"
    except ValueError:
        pass

    try:
        compute_regime_aware_trailing_exit_v3_liquidation_aware(
            100.0, 1, "invalid_mode", highs_l, lows_l, atr, p_trend_flat, 95.0
        )
        assert False, "Lỗi rò rỉ: trade_mode rác lọt qua mà không ném lỗi ValueError!"
    except ValueError:
        pass

    print(
        "✅ [TASK B-1-4] compute_regime_aware_trailing_exit_v3_liquidation_aware PASSED! (Đối xứng hoàn hảo, Regime-Flip chuẩn & Chống 100% mảng rỗng/lệch/ATR âm/mode rác)"
    )


def test_b_1_5_no_leakage_past_fold_boundary():
    """
    Xác nhận bản sửa không bao giờ nhìn thấy dữ liệu ngoài fold (Pre-Slice Zero-Leakage).
    """
    n = 60
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    atr = np.full(n, 1.0)
    p_trend = np.full(n, 0.5)

    full_highs = np.concatenate([[100.0], highs])
    full_lows = np.concatenate([[100.0], lows])
    full_atr = np.concatenate([[1.0], atr])
    full_p_trend = np.concatenate([[0.5], p_trend])

    entry_idx = 0
    fold_end_idx = 30

    result = simulate_trailing_exit_within_fold_bounds(
        entry_idx=entry_idx,
        entry_price=100.0,
        side=1,
        trade_mode="follow",
        test_window_end_idx=fold_end_idx,
        full_highs=full_highs,
        full_lows=full_lows,
        full_atr=full_atr,
        full_p_trend=full_p_trend,
        sl_initial=90.0,
        liquidation_price=80.0,
        t_max_live=120,
    )

    assert (
        result["exit_idx_absolute"] <= fold_end_idx
    ), f"LEAK: exit_idx_absolute={result['exit_idx_absolute']} vượt fold_end_idx={fold_end_idx}"
    assert (
        result["exit_reason"] == "TIME_STOP"
    ), f"Kỳ vọng TIME_STOP, nhận {result['exit_reason']}"
    assert result["boundary_truncated"] is True

    # Kiểm thử thêm case: Nếu lệnh rơi đúng vào sát vách fold_end_idx (future mảng rỗng)
    result_empty = simulate_trailing_exit_within_fold_bounds(
        entry_idx=fold_end_idx - 1, # Lệnh mở đúng nến cuối của test_window, mảng future rỗng
        entry_price=100.0,
        side=1,
        trade_mode="follow",
        test_window_end_idx=fold_end_idx,
        full_highs=full_highs,
        full_lows=full_lows,
        full_atr=full_atr,
        full_p_trend=full_p_trend,
        sl_initial=90.0,
        liquidation_price=80.0,
        t_max_live=120,
    )
    assert result_empty is not None and result_empty["boundary_truncated"] is True, (
        f"Kỳ vọng bản ghi boundary_truncated=True cho lệnh mảng rỗng, nhận {result_empty}"
    )

    # [STREAMING_CHUNK: TEST_TRAILING_ONE_BAR_BOUNDARY]
    # Kiểm thử case: Nếu lệnh vào cận biên fold (chỉ còn đúng 1 bar tương lai, len=1)
    result_one_bar = simulate_trailing_exit_within_fold_bounds(
        entry_idx=fold_end_idx - 2, # Còn đúng 1 bar tương lai trong test_window
        entry_price=100.0,
        side=1,
        trade_mode="follow",
        test_window_end_idx=fold_end_idx,
        full_highs=full_highs,
        full_lows=full_lows,
        full_atr=full_atr,
        full_p_trend=full_p_trend,
        sl_initial=90.0,
        liquidation_price=80.0,
        t_max_live=120,
    )
    assert result_one_bar is not None and result_one_bar["boundary_truncated"] is True, (
        f"Kỳ vọng bản ghi boundary_truncated=True khi chỉ còn 1 bar tương lai, nhận {result_one_bar}"
    )

    from aegis.meta_labeling.sizing.liquidation_layer import compute_liquidation_loss

    # [QĐ #7] compute_liquidation_loss trả về -(margin) thuần.
    # Phí vào lệnh được xử lý thống nhất tại pnl.py.
    pnl_liq = compute_liquidation_loss(
        size_notional=100000.0,
        leverage=10.0,
    )
    # Kỳ vọng: Mất trắng margin = -(100000 / 10) = -10000.0
    assert abs(pnl_liq - (-10000.0)) < 1e-4, f"Sai tính toán PnL Liquidation: {pnl_liq}"

    print(
        "✅ [TASK B-1-5] test_b_1_5_no_leakage_past_fold_boundary & Liquidation PnL PASSED!"
    )


# ============================================================================
# [TASK B-1-7 -> B-1-9] UNIT & INTEGRATION TESTS
# ============================================================================
from aegis.labeling.trailing_exit import (
    resolve_absolute_exit_idx,
    run_trailing_exit_for_oos_event,
    finalize_trade_record,
)


def test_resolve_absolute_exit_idx():
    assert resolve_absolute_exit_idx(entry_idx=100, exit_idx_relative=5) == 106
    assert resolve_absolute_exit_idx(entry_idx=0, exit_idx_relative=0) == 1
    assert resolve_absolute_exit_idx(entry_idx=999, exit_idx_relative=0) == 1000
    assert resolve_absolute_exit_idx(entry_idx=50, exit_idx_relative=119) == 170


def test_run_trailing_exit_for_oos_event_full_pipeline():
    n = 20
    highs = np.full(n, 100.5)
    lows = np.full(n, 99.5)
    atr = np.full(n, 1.0)
    p_trend = np.full(n, 0.5)
    # Case 1: SL hit bình thường (Follow)
    highs2 = highs.copy(); lows2 = lows.copy()
    lows2[5] = 80.0  # giá giảm mạnh tại bar 5
    result = run_trailing_exit_for_oos_event(
        entry_idx=0, entry_price=100.0, test_window_end_idx=n,
        p_i=0.8, p_chop_i=0.3, side_primary=1,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001,
        fade_enabled=True, fade_regime_gate_threshold=0.6,
        full_highs=highs2, full_lows=lows2, full_atr=atr, full_p_trend=p_trend
    )
    assert result is not None
    assert result["exit_reason"] in ("SL", "TRAIL", "TIME_STOP", "LIQUIDATION", "REGIME_FLIP")
    assert result["exit_idx_absolute"] == result["entry_idx"] + 1 + result["exit_idx_relative"]
    assert set(["entry_idx","entry_price","p_i","p_chop_i","mode","side","sl_initial",
                "leverage_used","liquidation_price","exit_idx_relative",
                "exit_idx_absolute","exit_reason","boundary_truncated"]) <= set(result.keys())

    # Case 2: mode == "none" (deadzone) -> None
    result_none = run_trailing_exit_for_oos_event(
        entry_idx=0, entry_price=100.0, test_window_end_idx=n,
        p_i=0.3, p_chop_i=0.5, side_primary=1,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001,
        fade_enabled=True, fade_regime_gate_threshold=0.6,
        full_highs=highs, full_lows=lows, full_atr=atr, full_p_trend=p_trend
    )
    assert result_none is None

    # Case 3: entry tại nến cuối cùng của fold -> future array rỗng -> trả về bản ghi boundary_truncated=True (thay vì None)
    result_boundary = run_trailing_exit_for_oos_event(
        entry_idx=n - 1, entry_price=100.0, test_window_end_idx=n,
        p_i=0.8, p_chop_i=0.3, side_primary=1,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001,
        fade_enabled=True, fade_regime_gate_threshold=0.6,
        full_highs=highs, full_lows=lows, full_atr=atr, full_p_trend=p_trend
    )
    assert result_boundary is not None and result_boundary["boundary_truncated"] is True, (
        "Zero-length slice PHẢI trả về bản ghi boundary_truncated=True theo đúng Data Contracts v11.9"
    )


def test_finalize_trade_record_reads_price_at_absolute_index():
    closes = np.arange(100.0, 130.0)  # closes[i] = 100+i, dễ kiểm tra bằng mắt
    partial = {
        "entry_idx": 5, "entry_price": 105.0, "p_i": 0.8, "p_chop_i": 0.3,
        "mode": "follow", "side": 1, "sl_initial": 95.0,
        "leverage_used": 5.0, "liquidation_price": 90.0,
        "exit_idx_relative": 3, "exit_idx_absolute": 9,  # entry_idx+1+relative = 5+1+3=9
        "exit_reason": "TRAIL", "boundary_truncated": False,
    }
    result = finalize_trade_record(partial, closes, size_notional=1000.0)
    expected_exit_price = closes[9]  # PHẢI đọc tại index 9 (absolute), KHÔNG phải index 3
    # Xác nhận gián tiếp qua việc realized_return khớp công thức dùng đúng closes[9]
    # finalize_trade_record áp dụng slippage trước khi tính PnL
    from aegis.execution.pnl import compute_realized_pnl
    spread_pct = 0.0002
    slippage_penalty_factor = 0.1
    bar_volume = 1e6
    size_notional_val = 1000.0
    slippage = expected_exit_price * (spread_pct / 2 + slippage_penalty_factor * (size_notional_val / bar_volume))
    exit_price_with_slippage = expected_exit_price - 1 * slippage  # side=1 (Long)
    expected_pnl = compute_realized_pnl(
        entry_price=105.0,
        exit_price=exit_price_with_slippage,
        side=1, size_notional=1000.0, leverage=5.0, exit_reason="TRAIL",
        entry_fill_type="taker", exit_fill_type="taker"
    )["net_pnl"]
    assert abs(result["realized_return"] - expected_pnl / 1000.0) < 1e-9


def test_finalize_trade_record_liquidation_branch_uses_margin_formula():
    closes = np.arange(100.0, 130.0)
    partial = {
        "entry_idx": 5, "entry_price": 105.0, "p_i": 0.1, "p_chop_i": 0.8,
        "mode": "fade", "side": -1, "sl_initial": 115.0,
        "leverage_used": 5.0, "liquidation_price": 110.0,
        "exit_idx_relative": 1, "exit_idx_absolute": 7,
        "exit_reason": "LIQUIDATION", "boundary_truncated": False,
    }
    result = finalize_trade_record(partial, closes, size_notional=1000.0)
    # [QĐ #7] realized_return cho nhánh LIQUIDATION = price_delta_pct (unleveraged price return)
    # Short side=-1: price_delta_pct = (entry_price - liq_price) / entry_price
    # = (105.0 - 110.0) / 105.0 = -0.047619...
    expected_return = (105.0 - 110.0) / 105.0
    assert abs(result["realized_return"] - expected_return) < 1e-9


def test_finalize_trade_record_output_passes_schema():
    import pandas as pd
    from aegis.core.schemas import TradeRecordSchema
    closes = np.arange(100.0, 130.0)
    partial = {
        "entry_idx": 5, "entry_price": 105.0, "p_i": 0.8, "p_chop_i": 0.3,
        "mode": "follow", "side": 1, "sl_initial": 95.0,
        "leverage_used": 5.0, "liquidation_price": 90.0,
        "exit_idx_relative": 3, "exit_idx_absolute": 9,
        "exit_reason": "TRAIL", "boundary_truncated": False,
    }
    result = finalize_trade_record(partial, closes, size_notional=1000.0)
    df = pd.DataFrame([result])
    TradeRecordSchema.validate(df)


def test_zero_atr_trailing_collapse_fixed():
    """
    [Vá BỌ SỐ 2: Zero-ATR Trailing Collapse — Instrument-Adaptive & Percentage Anchored Floor]
    Kiểm chứng khi ATR = 0.0 (thanh khoản cạn kiệt), hệ thống tự động kẹp safe_atr theo
    max(min_tick_size * min_ticks_cushion, entry_price * min_atr_pct), giữ cho trail_cushion > 0
    và ngăn trail_stop ôm sát khít đỉnh/đáy cho CẢ tài sản giá cao (BTC $60,000) lẫn altcoin ($0.001).
    """
    # 1. Kiểm chứng với BTC ($60,000, tick_size = 0.1)
    # Giá nhích xuống 1 tick ($0.1) hoặc $1, ATR = 0
    btc_highs = np.array([60000.0, 60000.0, 60000.0])
    btc_lows = np.array([60000.0, 59999.0, 60000.0]) # Giảm $1 (10 ticks)
    btc_atr = np.array([0.0, 0.0, 0.0])
    btc_p_trend = np.array([0.8, 0.8, 0.8])

    res_btc = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        entry_price=60000.0,
        side=1,
        trade_mode="follow",
        future_highs=btc_highs,
        future_lows=btc_lows,
        future_atr=btc_atr,
        future_p_trend=btc_p_trend,
        sl_initial=55000.0,
        min_tick_size=0.1,
        min_ticks_cushion=10,
        min_atr_pct=0.001, # safe_atr_floor = max(1.0, 60.0) = 60.0
    )
    assert res_btc is None or res_btc["exit_idx"] != 1, "Lỗi: BTC bị stop-out oan uổng do ATR = 0!"

    # 2. Kiểm chứng với Altcoin ($0.001, tick_size = 0.0001)
    alt_highs = np.array([0.001, 0.001, 0.001])
    alt_lows = np.array([0.001, 0.000999, 0.001]) # Giảm 1 micro-tick
    alt_atr = np.array([0.0, 0.0, 0.0])
    alt_p_trend = np.array([0.8, 0.8, 0.8])

    res_alt = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        entry_price=0.001,
        side=1,
        trade_mode="follow",
        future_highs=alt_highs,
        future_lows=alt_lows,
        future_atr=alt_atr,
        future_p_trend=alt_p_trend,
        sl_initial=0.0008,
        min_tick_size=1e-4,
        min_ticks_cushion=10,
        min_atr_pct=0.001, # safe_atr_floor = max(0.001, 0.000001) = 0.001
    )
    assert res_alt is None or res_alt["exit_idx"] != 1, "Lỗi: Altcoin bị stop-out oan uổng do ATR = 0!"

    print("✅ [Vá BỌ SỐ 2] Instrument-Adaptive & Percentage Anchored Zero-ATR Trailing Collapse PASSED!")


def test_finalize_trade_record_timestamp_plumbing():
    """
    [Vá BỌ SỐ 3: Temporal Blindness] Kiểm chứng entry_timestamp_ms và exit_timestamp_ms
    được trích xuất chuẩn xác từ full_timestamps theo đúng entry_idx và exit_idx_absolute.
    """
    import pandas as pd
    from aegis.core.schemas import TradeRecordSchema
    closes = np.arange(100.0, 130.0)
    timestamps = np.arange(1600000000000, 1600000000000 + 30 * 60000, 60000, dtype=int)
    partial = {
        "entry_idx": 5, "entry_price": 105.0, "p_i": 0.8, "p_chop_i": 0.3,
        "mode": "follow", "side": 1, "sl_initial": 95.0,
        "leverage_used": 5.0, "liquidation_price": 90.0,
        "exit_idx_relative": 3, "exit_idx_absolute": 9,
        "exit_reason": "TRAIL", "boundary_truncated": False,
    }
    result = finalize_trade_record(partial, closes, size_notional=1000.0, full_timestamps=timestamps)
    assert result["entry_timestamp_ms"] == timestamps[5], f"Entry ts sai: {result['entry_timestamp_ms']}"
    assert result["exit_timestamp_ms"] == timestamps[9], f"Exit ts sai: {result['exit_timestamp_ms']}"
    
    df = pd.DataFrame([result])
    TradeRecordSchema.validate(df)
    print("✅ [Vá BỌ SỐ 3] Temporal Blindness Timestamp Plumbing PASSED!")


def test_round_sl_safe_asymmetric_behavior():
    """
    [TDD VERIFICATION - BẪY 2: ASYMMETRIC STOP-LOSS SAFE ROUNDING]:
    Kiểm tra làm tròn SL bất đối xứng theo bước nhảy tick_size của sàn:
    - Long (side = 1): SL phải làm tròn XUỐNG (floor) để xa entry hơn, bảo vệ không cắn SL sớm.
    - Short/Fade (side = -1): SL phải làm tròn LÊN (ceil) để xa entry hơn, bảo vệ không cắn SL sớm.
    """
    from aegis.labeling.trailing_exit import round_sl_safe

    # Long (side = 1): SL thô là 49999.8, bước nhảy 1.0 -> phải floor về 49999.0
    sl_long = round_sl_safe(sl_raw=49999.8, tick_size=1.0, side=1)
    assert sl_long == pytest.approx(49999.0)

    # Short/Fade (side = -1): SL thô là 50000.2, bước nhảy 1.0 -> phải ceil lên 50001.0
    sl_short = round_sl_safe(sl_raw=50000.2, tick_size=1.0, side=-1)
    assert sl_short == pytest.approx(50001.0)


