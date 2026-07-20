import math
import numpy as np
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

    # 5. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy mảng rỗng (Empty array crash check)
    try:
        compute_regime_aware_trailing_exit_v3_liquidation_aware(100.0, 1, "follow", [], [], [], [], 95.0)
        assert False, "Lỗi rò rỉ: Mảng rỗng lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "Mảng future_highs rỗng" in str(e)

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
    assert result_empty is None, f"Kỳ vọng None cho lệnh mảng rỗng, nhận {result_empty}"

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
