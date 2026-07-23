"""
Bộ kiểm thử unit test cho Task B-2-1: CUSUM Event Filter & Dynamic Threshold Scaling (Log-Return & Sqrt(T) Scaling).
"""

import pytest
import numpy as np
import math
from aegis.labeling.cusum_events import (
    compute_dynamic_cusum_thresholds,
    filter_cusum_events_dynamic,
)


def test_b_2_1_compute_dynamic_cusum_thresholds_log_return_and_sqrt_t():
    """
    [TDD VERIFICATION - KHẮC PHỤC LỖ HỔNG 1: PRICE SCALING DIVISION HAZARD & THE EWMA LAG TRAP]:
    Kiểm chứng compute_dynamic_cusum_thresholds tính toán trong không gian Log-Return
    và tuân thủ chuẩn hóa theo định luật Căn bậc hai của Thời gian (sqrt(T) scaling).
    """
    prices = np.array([100.0, 100.0, 100.0, 50.0, 50.0], dtype=np.float64)
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    # 1. Khi bars_per_atr_period=1.0 (ATR cùng khung thời gian intraday):
    thresholds = compute_dynamic_cusum_thresholds(
        prices, atr, base_multiplier=2.5, min_rel_threshold=1e-4, max_rel_threshold=0.05, bars_per_atr_period=1.0
    )
    # sigma_rel = 2.0 / 100.0 = 0.02 => clamped = min(2.5 * 0.02, 0.05) = 0.05
    # Tại idx=3 (Flash Crash về 50): sigma_rel = 2.0 / 50.0 = 0.04 => clamped = min(2.5 * 0.04, 0.05) = 0.05
    assert len(thresholds) == 5
    assert thresholds[0] == 0.05
    assert thresholds[3] == 0.05  # Log-return threshold ổn định, không bị biến dạng bởi độ trễ mẫu số EWMA

    # 2. Kiểm chứng chuẩn hóa Sqrt(T) khi ATR đo trên nến Ngày (1440 nến phút/ngày):
    # bars_per_atr_period = 1440.0 => sigma_r = (ATR/P) / sqrt(1440)
    thresholds_sqrt = compute_dynamic_cusum_thresholds(
        prices, atr, base_multiplier=2.5, min_rel_threshold=1e-4, max_rel_threshold=0.05, bars_per_atr_period=1440.0
    )
    expected_sigma_rel = (2.0 / 100.0) * (1.0 / math.sqrt(1440.0))
    expected_threshold = np.clip(2.5 * expected_sigma_rel, 1e-4, 0.05)
    np.testing.assert_allclose(thresholds_sqrt[0], expected_threshold, rtol=1e-5)


def test_b_2_1_compute_dynamic_cusum_thresholds_clipping_and_guards():
    """
    [TDD VERIFICATION - ARMOR GUARDS & CLIPPING]:
    Kiểm tra kẹp biên min/max và ném ValueError khi nhận dữ liệu rác NaN/Inf/âm hoặc bars_per_atr_period <= 0.
    """
    prices = np.array([100.0, 100.0, 100.0], dtype=np.float64)
    # ATR cực lớn để kiểm tra kẹp biên max_rel_threshold
    atr_large = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    thresholds = compute_dynamic_cusum_thresholds(
        prices, atr_large, base_multiplier=2.5, min_rel_threshold=0.001, max_rel_threshold=0.05
    )
    np.testing.assert_allclose(thresholds, np.array([0.05, 0.05, 0.05]), rtol=1e-5)

    # Kiểm tra rác NaN / Inf / Negative & bars_per_atr_period <= 0
    with pytest.raises(ValueError, match="prices chứa NaN"):
        compute_dynamic_cusum_thresholds(np.array([100.0, np.nan]), np.array([2.0, 2.0]))

    with pytest.raises(ValueError, match="giá <= 0 phi lý"):
        compute_dynamic_cusum_thresholds(np.array([100.0, -10.0]), np.array([2.0, 2.0]))

    with pytest.raises(ValueError, match="ATR < 0 phi lý"):
        compute_dynamic_cusum_thresholds(np.array([100.0, 100.0]), np.array([2.0, -1.0]))

    with pytest.raises(ValueError, match="bars_per_atr_period phải là số thực dương"):
        compute_dynamic_cusum_thresholds(prices, atr_large, bars_per_atr_period=0.0)


def test_b_2_1_filter_cusum_events_spatial_temporal_gating():
    """
    [TDD VERIFICATION - SPATIAL-TEMPORAL COOLDOWN GATING IN LOG-RETURN SPACE & v11.6 C.4 METADATA]:
    Kiểm chứng bộ lọc CUSUM trong không gian Log-Return, tuân thủ cooldown và spatial delta,
    đồng thời gán metadata trade_mode / side.
    """
    prices = np.array([100.0, 103.0, 104.0, 105.0, 110.0, 109.0, 115.0], dtype=np.float64)
    # Ngưỡng log-return tĩnh = 0.02
    thresholds = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float64)
    timestamps = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000], dtype=np.int64)
    atr = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # p_trend, p_chop và trend_scores để kiểm tra C.4 metadata
    p_trend = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float64)
    p_chop = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    trend_scores = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], dtype=np.float64)

    # 1. Với cooldown_bars = 2, spatial_delta_atr = 1.0
    # bar 1: r_1 = ln(103/100) ~ 0.02955 > 0.02 => TRIGGERED (idx=1, price=103.0). Reset s_plus=0.
    # bar 2: r_2 = ln(104/103) ~ 0.00966 <= 0.02
    # bar 3: r_3 = ln(105/104) ~ 0.00956 => s_plus = 0.01922 <= 0.02
    # bar 4: r_4 = ln(110/105) ~ 0.04652 => s_plus = 0.0657 > 0.02 => TRIGGERED (idx=4, price=110.0).
    #        Kiểm tra gating tại idx=4: temporal = 4 - 1 = 3 >= 2 (cooldown_bars), spatial = |110 - 103| = 7 > 1.0 * 1.0 => PASS!
    # bar 5: r_5 = ln(109/110) ~ -0.00913 => s_minus = -0.00913
    # bar 6: r_6 = ln(115/109) ~ 0.05354 => s_plus = 0.05354 > 0.02 => TRIGGERED (idx=6, price=115.0).
    #        Kiểm tra gating tại idx=6: temporal = 6 - 4 = 2 >= 2 (cooldown_bars), spatial = |115 - 110| = 5 > 1.0 * 1.0 => PASS!
    events = filter_cusum_events_dynamic(
        prices,
        thresholds,
        timestamps,
        atr,
        cooldown_bars=2,
        spatial_delta_atr=1.0,
        trend_scores=trend_scores,
        p_trend=p_trend,
        p_chop=p_chop,
    )
    assert len(events) == 3
    assert events[0]["bar_idx"] == 1
    assert events[0]["trade_mode"] == "follow"
    assert events[0]["side"] == 1
    assert events[1]["bar_idx"] == 4
    assert events[2]["bar_idx"] == 6

    # 2. Nếu tăng cooldown_bars = 4, sự kiện idx=4 (cách idx=1 là 3 bar < 4) sẽ bị block!
    events_cooldown = filter_cusum_events_dynamic(
        prices, thresholds, timestamps, atr, cooldown_bars=4, spatial_delta_atr=1.0
    )
    assert len(events_cooldown) == 2
    assert events_cooldown[0]["bar_idx"] == 1
    assert events_cooldown[1]["bar_idx"] == 6  # idx=4 bị block bởi temporal cooldown


def test_compute_dynamic_cusum_thresholds_ewma_anchor_and_gap_reset():
    """
    [TDD VERIFICATION - EWMA ANCHOR PRICE & GAP-HANDLING RESET PROTOCOL]:
    Kiểm chứng khi `use_ewma_anchor=True`, giá neo được làm mượt bởi EWMA span=50.
    Đặc biệt, khi xuất hiện khoảng trống giá (gap) hoặc ranh giới CPCV fold (`reset_mask[t] == True`),
    bộ nhớ EWMA lập tức reset về giá hiện tại P_t (predict-only reset), ngăn chặn ô nhiễm ranh giới.
    """
    # 5 bar đầu giá 100, bar 5 nhảy gap lên 200 (ví dụ sang fold mới hoặc qua đêm)
    prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0], dtype=np.float64)
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    # Nếu KHÔNG reset tại bar 5 (idx=5), EWMA sẽ bị kéo lag phía dưới (chưa tới 200 ngay)
    thresholds_no_reset = compute_dynamic_cusum_thresholds(
        prices, atr, base_multiplier=2.5, anchor_span=9, min_rel_threshold=1e-4, max_rel_threshold=0.05,
        use_ewma_anchor=True, reset_mask=None
    )

    # Nếu CÓ reset tại bar 5 (reset_mask[5] = True) -> EWMA tại idx=5 lập tức re-seed bằng 200.0
    reset_mask = np.array([False, False, False, False, False, True, False], dtype=bool)
    thresholds_with_reset = compute_dynamic_cusum_thresholds(
        prices, atr, base_multiplier=2.5, anchor_span=9, min_rel_threshold=1e-4, max_rel_threshold=0.05,
        use_ewma_anchor=True, reset_mask=reset_mask
    )

    # Kiểm chứng: Tại idx=5, khi reset về P=200, sigma_rel = (2 / 200) * 2.5 = 0.025
    # Khi KHÔNG reset, EWMA < 200 nên mẫu số nhỏ hơn -> sigma_rel lớn hơn 0.025
    assert thresholds_with_reset[5] == pytest.approx(0.025, rel=1e-5)
    assert thresholds_no_reset[5] > thresholds_with_reset[5]


def test_cusum_flash_crash_no_lag_trap():
    """
    [TDD VERIFICATION - FLASH CRASH ZERO-LAG TRAP (LỖ HỔNG 2)]:
    Kiểm chứng khi xảy ra cú sập giá chớp nhoáng (Flash Crash: giá giảm từ 100 xuống 50),
    ngưỡng CUSUM phản ánh biến động tương đối tức thời (ATR/P_t = 2/50 = 0.04) mà không bị bóp nghẹt
    bởi mẫu số EWMA(P_t) trễ (chưa kịp giảm từ 100 xuống 50).
    """
    prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 50.0], dtype=np.float64)
    atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)

    thresholds = compute_dynamic_cusum_thresholds(
        prices, atr, base_multiplier=1.0, anchor_span=9, min_rel_threshold=1e-4, max_rel_threshold=0.10,
        use_ewma_anchor=True, reset_mask=None
    )

    # Nếu dùng EWMA(P_t) mẫu số (cũ), EWMA tại bar 5 ~ 90 -> threshold ~ 2/90 = 0.022
    # Với chuẩn hóa mới inst_rel_vol (mới), inst_rel_vol tại bar 5 là 2/50 = 0.04.
    # EWMA của inst_rel_vol tại bar 5 sẽ tăng lên từ 0.02 hướng tới 0.04 (tức > 0.023).
    assert thresholds[5] > 0.023
    assert thresholds[5] > thresholds[4]  # Biến động tương đối phải tăng lên khi giá giảm sâu!


