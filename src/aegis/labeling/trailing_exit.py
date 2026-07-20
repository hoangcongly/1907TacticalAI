import math
import numpy as np
from typing import Optional


# ============================================================================
# [TASK B-1-3] SYMMETRIC INITIAL STOP-LOSS WITH ARMOR-PLATED GUARDS
# ============================================================================
def compute_sl_initial(
    entry_price: float,
    side: int,
    m_sl: float,
    sigma: float,
    c_trade_adj: float,
    max_reasonable_cushion: float = 0.5,
) -> float:
    """
    [TASK B-1-3] Tính toán mức Cắt lỗ gốc tĩnh (Initial Stop-Loss) hoàn toàn đối xứng.

    Tham số:
    - entry_price: Giá khớp lệnh đầu vào.
    - side: Chiều giao dịch (+1 cho Long, -1 cho Short/Fade). BẮT BUỘC ĐÃ ĐƯỢC ĐẢO DẤU CHUẨN.
    - m_sl: Hệ số nhân rào cản cắt lỗ (Stop-loss multiplier).
    - sigma: Biến động nội tại của nến (VD: ATR_14 tính bằng %).
    - c_trade_adj: Chi phí giao dịch + Trượt giá dự kiến.
    - max_reasonable_cushion: Giới hạn đệm an toàn hợp lý.
    """
    if side not in (1, -1):
        raise ValueError(
            f"Lỗi hải quan B-1-3: side bắt buộc phải là +1 (Long) hoặc -1 (Short/Fade), nhận {side}"
        )

    if (
        not isinstance(entry_price, (int, float))
        or math.isnan(entry_price)
        or math.isinf(entry_price)
        or entry_price <= 0
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: entry_price phải là số dương hợp lệ, nhận {entry_price}"
        )

    if (
        not isinstance(sigma, (int, float))
        or math.isnan(sigma)
        or math.isinf(sigma)
        or sigma < 0
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: sigma (biến động) phải >= 0 và không được là NaN/Inf, nhận {sigma}"
        )

    if (
        not isinstance(m_sl, (int, float))
        or math.isnan(m_sl)
        or math.isinf(m_sl)
        or m_sl < 0
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: m_sl (hệ số cắt lỗ) phải >= 0 và hợp lệ, nhận {m_sl}"
        )

    if (
        not isinstance(c_trade_adj, (int, float))
        or math.isnan(c_trade_adj)
        or math.isinf(c_trade_adj)
        or c_trade_adj < 0
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: c_trade_adj (phí & slippage) phải >= 0 và hợp lệ, nhận {c_trade_adj}"
        )

    total_cushion = m_sl * sigma + c_trade_adj

    if total_cushion > max_reasonable_cushion:
        raise ValueError(
            f"Cushion {total_cushion:.2%} vượt ngưỡng hợp lý {max_reasonable_cushion:.2%} "
            f"-- khả năng sigma bị lỗi đơn vị hoặc NaN thoát dạng số lớn bất thường."
        )

    if side > 0:
        if total_cushion >= 1.0:
            raise ValueError(
                f"Lỗi rủi ro cực đại B-1-3: Tổng rủi ro trừ hao ({total_cushion:.2%}) >= 100% giá trị tài sản với lệnh Long, dẫn đến Stop-Loss <= 0!"
            )
        sl = entry_price * math.exp(-total_cushion)
        return float(max(sl, 1e-4))
    else:
        sl = entry_price * math.exp(total_cushion)
        return float(sl)


# ============================================================================
# UNIT TESTS (TDD & FAULT-INJECTION STRESS TESTS)
# ============================================================================
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


# ============================================================================
# [TASK B-1-4] REGIME-AWARE TRAILING EXIT V2 (ĐỐI XỨNG & REGIME-FLIP WITH ARMOR GUARDS)
# ============================================================================
def _update_regime_flip(
    p_trend_k: float, trade_mode: str, threshold: float, prev_count: int
) -> int:
    """
    SỬA LỖI v11.6: chiều kích hoạt Regime-Flip phụ thuộc trade_mode, KHÔNG phụ thuộc side.
    Follow: thoát khi trend suy yếu (p_trend < threshold).
    Fade: thoát khi choppy bị phá vỡ (p_trend > threshold) — chiều NGƯỢC LẠI hoàn toàn.

    [ARMOR-PLATED GUARDS — BẢO VỆ CHỐNG LỖ HỔNG]:
    - Chặn đứng p_trend_k bị NaN, Inf hoặc ngoài đoạn [0, 1] ngăn reset/kích hoạt sai bộ đếm.
    - Khóa chặt trade_mode chỉ chấp nhận 'follow' hoặc 'fade'.
    """
    if (
        not isinstance(p_trend_k, (int, float))
        or math.isnan(p_trend_k)
        or math.isinf(p_trend_k)
        or not (0.0 <= p_trend_k <= 1.0)
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-4 (_update_regime_flip): p_trend_k dị thường tại nến Trailing: {p_trend_k}"
        )

    if trade_mode == "follow":
        triggered = p_trend_k < threshold
    elif trade_mode == "fade":
        triggered = p_trend_k > threshold
    else:
        raise ValueError(
            f"Lỗi hải quan B-1-4: trade_mode không hợp lệ cho Regime Flip: {trade_mode}"
        )

    return prev_count + 1 if triggered else 0


# UNIT TESTS (TDD & ARMOR-PLATED FAULT-INJECTION STRESS TESTS)
# ============================================================================
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


# ============================================================================
# [v11.9 / v3] REGIME-AWARE TRAILING EXIT V3 (LIQUIDATION AWARE)
# ============================================================================
def compute_regime_aware_trailing_exit_v3_liquidation_aware(
    entry_price: float,
    side: int,
    trade_mode: str,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    future_atr: np.ndarray,
    future_p_trend: np.ndarray,
    sl_initial: float,
    liquidation_price: Optional[float] = None,
    m_trail_base: float = 2.0,
    gamma: float = 0.75,
    p_trend_exit_threshold_follow: float = 0.35,
    p_trend_exit_threshold_fade: float = 0.65,
    consecutive_bars_required: int = 2,
    t_max_live: int = 120,
    max_lookforward_override: Optional[int] = None,
) -> dict:
    """
    [v3 / v11.9] Nâng cấp từ v2: Tích hợp kiểm tra giá thanh lý (Liquidation Price).
    Nếu nến xuyên phá qua cả Liquidation Price trước khi hoặc cùng lúc với SL/Trail, ưu tiên chốt LIQUIDATION hoặc SL.
    """
    if side not in (1, -1):
        raise ValueError(
            f"Lỗi hải quan B-1-5 (v3): side bắt buộc phải là +1 hoặc -1, nhận {side}"
        )

    if trade_mode not in ("follow", "fade"):
        raise ValueError(
            f"Lỗi hải quan B-1-5 (v3): trade_mode bắt buộc phải là 'follow' hoặc 'fade', nhận {trade_mode}"
        )

    highs = np.asarray(future_highs, dtype=float)
    lows = np.asarray(future_lows, dtype=float)
    atr = np.asarray(future_atr, dtype=float)
    p_trend = np.asarray(future_p_trend, dtype=float)

    n_bars = len(highs)
    if n_bars == 0:
        raise ValueError("Lỗi B-1-5 (v3): Mảng future_highs rỗng (0 nến tương lai)!")

    if len(lows) != n_bars or len(atr) != n_bars or len(p_trend) != n_bars:
        raise ValueError(
            f"Lỗi B-1-5 (v3): Độ dài các mảng tương lai lệch nhau: highs={n_bars}, lows={len(lows)}, atr={len(atr)}, p_trend={len(p_trend)}"
        )

    if (
        np.any(np.isnan(highs))
        or np.any(np.isinf(highs))
        or np.any(np.isnan(lows))
        or np.any(np.isinf(lows))
    ):
        raise ValueError(
            "Lỗi B-1-4: Mảng giá tương lai (highs/lows) chứa giá trị rác NaN hoặc Inf!"
        )

    if np.any(atr < 0) or np.any(np.isnan(atr)) or np.any(np.isinf(atr)):
        raise ValueError(
            "Lỗi B-1-4: Mảng future_atr chứa số âm (<0) hoặc NaN/Inf gây sai lệch Trailing Stop!"
        )

    if np.any(highs < lows):
        raise ValueError(
            "Lỗi B-1-4: Phát hiện nến dị thường có High < Low trong mảng tương lai!"
        )

    effective_t_max = (
        min(t_max_live, max_lookforward_override)
        if max_lookforward_override is not None
        else t_max_live
    )
    if effective_t_max <= 0:
        raise ValueError(
            f"Lỗi B-1-5 (v3): effective_t_max ({effective_t_max}) phải > 0!"
        )

    # [FINDING F] Zero-length slice: nếu chỉ có 0 hoặc 1 nến tương lai,
    # không đủ dữ liệu để mô phỏng exit có ý nghĩa → trả None thay vì
    # tạo bản ghi TIME_STOP giả với thời gian nắm giữ = 0.
    if min(effective_t_max, n_bars) <= 1:
        return None

    threshold = (
        p_trend_exit_threshold_follow
        if trade_mode == "follow"
        else p_trend_exit_threshold_fade
    )
    consecutive_flip_count = 0

    if side > 0:
        extreme_price = entry_price
        for k in range(min(effective_t_max, n_bars)):
            # 0. Kiểm tra Liquidation trước (nếu có và hợp lệ)
            if (
                liquidation_price is not None
                and liquidation_price > 0
                and lows[k] <= liquidation_price
            ):
                return {
                    "exit_idx": k,
                    "reason": "LIQUIDATION",
                    "boundary_truncated": False,
                }
            # 1. Kiểm tra Stop-loss cứng
            if lows[k] <= sl_initial:
                return {"exit_idx": k, "reason": "SL", "boundary_truncated": False}
            # 2. Tính Trailing Stop TỪ extreme_price của nến trước (Geometric Symmetry)
            trail_cushion = m_trail_base * (1 + gamma * p_trend[k]) * atr[k] / extreme_price
            trail_stop = extreme_price * math.exp(-trail_cushion)
            # Kẹp (clamp): trail_stop KHÔNG được phép lỏng hơn sl_initial (Long: không thấp hơn SL)
            trail_stop = max(trail_stop, sl_initial)
            if lows[k] <= trail_stop:
                return {"exit_idx": k, "reason": "TRAIL", "boundary_truncated": False}
            # 2.1 Cập nhật cực đại cho nến sau
            extreme_price = max(extreme_price, highs[k])

            # 3. Kiểm tra Regime-Flip
            consecutive_flip_count = _update_regime_flip(
                p_trend[k], trade_mode, threshold, consecutive_flip_count
            )
            if consecutive_flip_count >= consecutive_bars_required:
                return {
                    "exit_idx": k,
                    "reason": "REGIME_FLIP",
                    "boundary_truncated": False,
                }
    else:
        extreme_price = entry_price
        for k in range(min(effective_t_max, n_bars)):
            # 0. Kiểm tra Liquidation cho Short (khi giá tăng lên vượt liq_price)
            if (
                liquidation_price is not None
                and liquidation_price > 0
                and highs[k] >= liquidation_price
            ):
                return {
                    "exit_idx": k,
                    "reason": "LIQUIDATION",
                    "boundary_truncated": False,
                }
            # 1. Kiểm tra Stop-loss cứng
            if highs[k] >= sl_initial:
                return {"exit_idx": k, "reason": "SL", "boundary_truncated": False}
            # 2. Tính Trailing Stop TỪ extreme_price của nến trước (Geometric Symmetry)
            trail_cushion = m_trail_base * (1 + gamma * p_trend[k]) * atr[k] / extreme_price
            trail_stop = extreme_price * math.exp(trail_cushion)
            # Kẹp (clamp): trail_stop KHÔNG được phép lỏng hơn sl_initial (Short: không cao hơn SL)
            trail_stop = min(trail_stop, sl_initial)
            if highs[k] >= trail_stop:
                return {"exit_idx": k, "reason": "TRAIL", "boundary_truncated": False}
            # 2.1 Cập nhật cực tiểu cho nến sau
            extreme_price = min(extreme_price, lows[k])

            # 3. Kiểm tra Regime-Flip
            consecutive_flip_count = _update_regime_flip(
                p_trend[k], trade_mode, threshold, consecutive_flip_count
            )
            if consecutive_flip_count >= consecutive_bars_required:
                return {
                    "exit_idx": k,
                    "reason": "REGIME_FLIP",
                    "boundary_truncated": False,
                }

    last_idx = min(effective_t_max, n_bars) - 1
    is_truncated = (max_lookforward_override is not None) and (
        effective_t_max < t_max_live
    )
    return {
        "exit_idx": max(last_idx, 0),
        "reason": "TIME_STOP",
        "boundary_truncated": is_truncated,
    }


# Alias chuẩn hóa tên gọi
compute_regime_aware_trailing_exit_v3 = (
    compute_regime_aware_trailing_exit_v3_liquidation_aware
)


# ============================================================================
# [TASK B-1-5] SỬA LẠI — bound-trước-khi-tính, KHÔNG patch-sau-khi-tính
# ============================================================================
def simulate_trailing_exit_within_fold_bounds(
    entry_idx: int,
    entry_price: float,
    side: int,
    trade_mode: str,
    test_window_end_idx: int,
    full_highs: np.ndarray,
    full_lows: np.ndarray,
    full_atr: np.ndarray,
    full_p_trend: np.ndarray,
    sl_initial: float,
    liquidation_price: Optional[float] = None,
    t_max_live: int = 120,
    **trailing_exit_kwargs,
) -> dict:
    """
    [TASK B-1-5] SỬA LỖI: quay lại đúng nguyên tắc gốc — CẮT mảng future_* theo
    biên fold TRƯỚC KHI gọi hàm trailing-exit, để hàm tính toán không bao giờ có
    khả năng đọc dữ liệu ngoài fold, thay vì để nó tính tự do rồi patch kết quả sau.

    [ARMOR-PLATED GUARDS — BỌC THÉP CHỐNG RÒ RỈ & MẢNG RỖNG]:
    - Cắt vật lý ngay tại min(entry_idx + 1 + t_max_live, test_window_end_idx, len(full_highs)).
    - Xử lý mượt mà kịch bản mảng rỗng sát biên fold.
    """
    if not (0 <= entry_idx < len(full_highs)):
        raise ValueError(
            f"Lỗi hải quan B-1-5: entry_idx ({entry_idx}) nằm ngoài độ dài mảng full_highs ({len(full_highs)})"
        )

    effective_end = min(
        entry_idx + 1 + t_max_live, test_window_end_idx, len(full_highs)
    )
    max_lookforward = max(test_window_end_idx - (entry_idx + 1), 0)

    # Cắt mảng vật lý trước khi đưa vào hàm (Pre-Slice)
    future_highs = full_highs[entry_idx + 1 : effective_end]
    future_lows = full_lows[entry_idx + 1 : effective_end]
    future_atr = full_atr[entry_idx + 1 : effective_end]
    future_p_trend = full_p_trend[entry_idx + 1 : effective_end]

    # Guard an toàn nếu lệnh vào đúng nến cuối của Fold hoặc vượt biên
    if len(future_highs) == 0:
        return None

    exit_result = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        entry_price=entry_price,
        side=side,
        trade_mode=trade_mode,
        future_highs=future_highs,
        future_lows=future_lows,
        future_atr=future_atr,
        future_p_trend=future_p_trend,
        sl_initial=sl_initial,
        liquidation_price=liquidation_price,
        t_max_live=t_max_live,
        max_lookforward_override=max_lookforward,
        **trailing_exit_kwargs,
    )

    exit_idx_relative = int(exit_result["exit_idx"])
    exit_idx_absolute = entry_idx + 1 + exit_idx_relative

    return {
        "entry_idx": entry_idx,
        "exit_idx_relative": exit_idx_relative,
        "exit_idx_absolute": exit_idx_absolute,
        "exit_reason": exit_result["reason"],
        "boundary_truncated": bool(exit_result["boundary_truncated"]),
    }




# ============================================================================
# UNIT TESTS (TASK B-1-5 & LIQUIDATION PnL ARMOR-PLATED TESTS)
# ============================================================================
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

    # Kiểm thử PnL Liquidation: Isolated Margin, mất mát tối đa = margin = size_notional / leverage.
    # Phí thanh lý đã được tính vào việc đẩy giá thanh lý gần entry hơn (trong compute_liquidation_price),
    # KHÔNG tính thêm ở đây (tránh double-count).
    pnl_liq = compute_liquidation_loss(
        size_notional=100000.0,
        leverage=10.0,
    )
    # Kỳ vọng: Mất trắng margin = -(100000 / 10) = -10000.0
    assert abs(pnl_liq - (-10000.0)) < 1e-4, f"Sai tính toán PnL Liquidation: {pnl_liq}"

    print(
        "✅ [TASK B-1-5] test_b_1_5_no_leakage_past_fold_boundary & Liquidation PnL PASSED!"
    )


if __name__ == "__main__":
    test_b_1_3_compute_sl_initial()
    test_b_1_4_regime_aware_trailing_exit_symmetry()
    test_b_1_5_no_leakage_past_fold_boundary()
