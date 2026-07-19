import math
import numpy as np

# ============================================================================
# [TASK B-1-3] SYMMETRIC INITIAL STOP-LOSS WITH ARMOR-PLATED GUARDS
# ============================================================================
def compute_sl_initial(entry_price: float, side: int, m_sl: float, sigma: float, c_trade_adj: float) -> float:
    """
    [TASK B-1-3] Tính giá Stop-Loss ban đầu, đối xứng tuyệt đối cho Long/Short.
    
    [ARMOR-PLATED GUARDS — BẢO VỆ CHỐNG LỖ HỔNG]:
    - Chặn đứng side == 0 (Neutral/Stand Aside) ngăn nhiễm độc PnL.
    - Chặn đứng số rác NaN / Inf hoặc số âm cho entry_price, sigma, m_sl, c_trade_adj.
    - Ngăn chặn thảm họa Stop-Loss âm (sl <= 0) khi thị trường biến động quá lớn (>100% rủi ro).
    
    Tham số:
    - entry_price: Giá khớp lệnh đầu vào.
    - side: Chiều giao dịch (+1 cho Long, -1 cho Short/Fade). BẮT BUỘC ĐÃ ĐƯỢC ĐẢO DẤU CHUẨN.
    - m_sl: Hệ số nhân rào cản cắt lỗ (Stop-loss multiplier).
    - sigma: Biến động nội tại của nến (VD: ATR_14 tính bằng %).
    - c_trade_adj: Chi phí giao dịch + Trượt giá dự kiến.
    """
    if side not in (1, -1):
        raise ValueError(f"Lỗi hải quan B-1-3: side bắt buộc phải là +1 (Long) hoặc -1 (Short/Fade), nhận {side}")
        
    if not isinstance(entry_price, (int, float)) or math.isnan(entry_price) or math.isinf(entry_price) or entry_price <= 0:
        raise ValueError(f"Lỗi hải quan B-1-3: entry_price phải là số dương hợp lệ, nhận {entry_price}")
        
    if not isinstance(sigma, (int, float)) or math.isnan(sigma) or math.isinf(sigma) or sigma < 0:
        raise ValueError(f"Lỗi hải quan B-1-3: sigma (biến động) phải >= 0 và không được là NaN/Inf, nhận {sigma}")
        
    if not isinstance(m_sl, (int, float)) or math.isnan(m_sl) or math.isinf(m_sl) or m_sl < 0:
        raise ValueError(f"Lỗi hải quan B-1-3: m_sl (hệ số cắt lỗ) phải >= 0 và hợp lệ, nhận {m_sl}")
        
    if not isinstance(c_trade_adj, (int, float)) or math.isnan(c_trade_adj) or math.isinf(c_trade_adj) or c_trade_adj < 0:
        raise ValueError(f"Lỗi hải quan B-1-3: c_trade_adj (phí & slippage) phải >= 0 và hợp lệ, nhận {c_trade_adj}")

    total_cushion = m_sl * sigma + c_trade_adj

    if side > 0:
        if total_cushion >= 1.0:
            raise ValueError(f"Lỗi rủi ro cực đại B-1-3: Tổng rủi ro trừ hao ({total_cushion:.2%}) >= 100% giá trị tài sản với lệnh Long, dẫn đến Stop-Loss <= 0!")
        sl = entry_price * (1.0 - total_cushion)
        return float(max(sl, 1e-4))
    else:
        sl = entry_price * (1.0 + total_cushion)
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
    sl_long = compute_sl_initial(entry_price, side=1, m_sl=m_sl, sigma=sigma, c_trade_adj=c_trade_adj)
    expected_sl_long = 100.0 * (1.0 - 0.11) # = 89.0
    assert abs(sl_long - expected_sl_long) < 1e-9, f"Lỗi SL Long: Cần {expected_sl_long}, Nhận {sl_long}"
    
    # 2. Test cho phe Short/Fade (side = -1)
    sl_short = compute_sl_initial(entry_price, side=-1, m_sl=m_sl, sigma=sigma, c_trade_adj=c_trade_adj)
    expected_sl_short = 100.0 * (1.0 + 0.11) # = 111.0
    assert abs(sl_short - expected_sl_short) < 1e-9, f"Lỗi SL Short: Cần {expected_sl_short}, Nhận {sl_short}"
    
    # 3. Test tính đối xứng tuyệt đối
    dist_long = entry_price - sl_long
    dist_short = sl_short - entry_price
    assert abs(dist_long - dist_short) < 1e-9, f"Lỗi Đối xứng: Long ({dist_long}) != Short ({dist_short})"
    
    # 4. [ARMOR-PLATED TESTS] Bắt lỗi nghiêm ngặt khi truyền side = 0
    try:
        compute_sl_initial(entry_price, side=0, m_sl=m_sl, sigma=sigma, c_trade_adj=c_trade_adj)
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
            assert False, f"Lỗi rò rỉ: Input dị thường ({ep}, {sd}, {m}, {sig}, {cadj}) không bị chặn!"
        except ValueError:
            pass
            
    # 6. [ARMOR-PLATED TESTS] Bắt lỗi rủi ro vượt quá 100% với lệnh Long
    try:
        compute_sl_initial(entry_price=100.0, side=1, m_sl=3.0, sigma=0.40, c_trade_adj=0.05) # Total = 1.25 (125%)
        assert False, "Lỗi rò rỉ: Stop-loss âm lọt qua mà không bị chặn!"
    except ValueError as e:
        assert "Tổng rủi ro trừ hao" in str(e)
    
    print("✅ [TASK B-1-3] compute_sl_initial PASSED! (Đối xứng gương hoàn hảo & Khóa 100% lỗ hổng side=0 / NaN / SL âm)")

# ============================================================================
# [TASK B-1-4] REGIME-AWARE TRAILING EXIT V2 (ĐỐI XỨNG & REGIME-FLIP WITH ARMOR GUARDS)
# ============================================================================
def _update_regime_flip(p_trend_k: float, trade_mode: str, threshold: float, prev_count: int) -> int:
    """
    SỬA LỖI v11.6: chiều kích hoạt Regime-Flip phụ thuộc trade_mode, KHÔNG phụ thuộc side.
    Follow: thoát khi trend suy yếu (p_trend < threshold).
    Fade: thoát khi choppy bị phá vỡ (p_trend > threshold) — chiều NGƯỢC LẠI hoàn toàn.
    
    [ARMOR-PLATED GUARDS — BẢO VỆ CHỐNG LỖ HỔNG]:
    - Chặn đứng p_trend_k bị NaN, Inf hoặc ngoài đoạn [0, 1] ngăn reset/kích hoạt sai bộ đếm.
    - Khóa chặt trade_mode chỉ chấp nhận 'follow' hoặc 'fade'.
    """
    if not isinstance(p_trend_k, (int, float)) or math.isnan(p_trend_k) or math.isinf(p_trend_k) or not (0.0 <= p_trend_k <= 1.0):
        raise ValueError(f"Lỗi hải quan B-1-4 (_update_regime_flip): p_trend_k dị thường tại nến Trailing: {p_trend_k}")

    if trade_mode == "follow":
        triggered = p_trend_k < threshold
    elif trade_mode == "fade":
        triggered = p_trend_k > threshold
    else:
        raise ValueError(f"Lỗi hải quan B-1-4: trade_mode không hợp lệ cho Regime Flip: {trade_mode}")

    return prev_count + 1 if triggered else 0

def compute_regime_aware_trailing_exit_v2(
    entry_price: float, side: int, trade_mode: str,
    future_highs: np.ndarray, future_lows: np.ndarray, future_atr: np.ndarray,
    future_p_trend: np.ndarray, sl_initial: float,
    m_trail_base: float = 2.0, gamma: float = 0.75,
    p_trend_exit_threshold_follow: float = 0.35,
    p_trend_exit_threshold_fade: float = 0.65,
    consecutive_bars_required: int = 2,
    t_max_live: int = 120,
    max_lookforward_override: int = None
) -> dict:
    """
    SỬA LỖI v11.6 Patch A: Đối xứng hóa hoàn toàn SL/Trailing cho side<0 và đảo chiều Regime-Flip.
    v11.7 Patch C: t_max_live được truyền vào đã được resolve chính xác theo mode.
    LƯU Ý v11.8: exit_idx trả về từ hàm này là OFFSET TƯƠNG ĐỐI k tính từ tương lai của entry_idx+1.
    
    [ARMOR-PLATED GUARDS — BẢO VỆ CHỐNG LỖ HỔNG]:
    1. Chặn side == 0 (Stand Aside) hoặc không hợp lệ.
    2. Chặn trade_mode ngoài ('follow', 'fade').
    3. Chặn mảng tương lai rỗng (0 nến) hoặc t_max_live/override <= 0.
    4. Chặn độ dài các mảng future_highs, future_lows, future_atr, future_p_trend lệch nhau.
    5. Chặn mảng chứa ATR âm (< 0) hoặc NaN/Inf làm nghịch đảo Trailing Stop.
    """
    if side not in (1, -1):
        raise ValueError(f"Lỗi hải quan B-1-4: side bắt buộc phải là +1 (Long) hoặc -1 (Short/Fade), nhận {side}")
        
    if trade_mode not in ("follow", "fade"):
        raise ValueError(f"Lỗi hải quan B-1-4: trade_mode bắt buộc phải là 'follow' hoặc 'fade', nhận {trade_mode}")

    # Chuẩn hóa sang numpy ndarray để đảm bảo an toàn thao tác mảng
    highs = np.asarray(future_highs, dtype=float)
    lows = np.asarray(future_lows, dtype=float)
    atr = np.asarray(future_atr, dtype=float)
    p_trend = np.asarray(future_p_trend, dtype=float)

    n_bars = len(highs)
    if n_bars == 0:
        raise ValueError("Lỗi B-1-4: Mảng future_highs rỗng (0 nến tương lai)!")
        
    if len(lows) != n_bars or len(atr) != n_bars or len(p_trend) != n_bars:
        raise ValueError(f"Lỗi B-1-4: Độ dài các mảng tương lai lệch nhau: highs={n_bars}, lows={len(lows)}, atr={len(atr)}, p_trend={len(p_trend)}")

    if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or np.any(np.isnan(lows)) or np.any(np.isinf(lows)):
        raise ValueError("Lỗi B-1-4: Mảng giá tương lai (highs/lows) chứa giá trị rác NaN hoặc Inf!")

    if np.any(atr < 0) or np.any(np.isnan(atr)) or np.any(np.isinf(atr)):
        raise ValueError("Lỗi B-1-4: Mảng future_atr chứa số âm (<0) hoặc NaN/Inf gây sai lệch Trailing Stop!")

    if np.any(highs < lows):
        raise ValueError("Lỗi B-1-4: Phát hiện nến dị thường có High < Low trong mảng tương lai!")

    effective_t_max = min(t_max_live, max_lookforward_override) if max_lookforward_override is not None else t_max_live
    if effective_t_max <= 0:
        raise ValueError(f"Lỗi B-1-4: effective_t_max ({effective_t_max}) phải > 0!")

    threshold = p_trend_exit_threshold_follow if trade_mode == "follow" else p_trend_exit_threshold_fade
    consecutive_flip_count = 0

    # ---------------------------------------------------------
    # 1. NHÁNH LONG (side > 0)
    # ---------------------------------------------------------
    if side > 0:
        extreme_price = entry_price  # highest high kể từ lúc vào lệnh
        for k in range(min(effective_t_max, n_bars)):
            # Kiểm tra Stop-loss cứng trước
            if lows[k] <= sl_initial:
                return {"exit_idx": k, "reason": "SL", "boundary_truncated": False}
            
            # Cập nhật mức cao nhất và tính khoảng cách Trailing
            extreme_price = max(extreme_price, highs[k])
            trail_stop = extreme_price - m_trail_base * (1 + gamma * p_trend[k]) * atr[k]
            
            # Kiểm tra Trailing Stop
            if lows[k] <= trail_stop:
                return {"exit_idx": k, "reason": "TRAIL", "boundary_truncated": False}
            
            # Kiểm tra Regime-Flip (Thoát sớm dựa trên HMM)
            consecutive_flip_count = _update_regime_flip(p_trend[k], trade_mode, threshold, consecutive_flip_count)
            if consecutive_flip_count >= consecutive_bars_required:
                return {"exit_idx": k, "reason": "REGIME_FLIP", "boundary_truncated": False}

    # ---------------------------------------------------------
    # 2. NHÁNH SHORT/FADE (side < 0)
    # ---------------------------------------------------------
    else:
        extreme_price = entry_price  # lowest low kể từ lúc vào lệnh
        for k in range(min(effective_t_max, n_bars)):
            # Kiểm tra Stop-loss cứng (Với lệnh Short, SL nằm bên TRÊN)
            if highs[k] >= sl_initial:
                return {"exit_idx": k, "reason": "SL", "boundary_truncated": False}
            
            # Cập nhật mức thấp nhất và tính Trailing nằm bên TRÊN giá
            extreme_price = min(extreme_price, lows[k])
            trail_stop = extreme_price + m_trail_base * (1 + gamma * p_trend[k]) * atr[k]
            
            # Kiểm tra Trailing Stop
            if highs[k] >= trail_stop:
                return {"exit_idx": k, "reason": "TRAIL", "boundary_truncated": False}
            
            # Kiểm tra Regime-Flip (Thoát sớm dựa trên HMM)
            consecutive_flip_count = _update_regime_flip(p_trend[k], trade_mode, threshold, consecutive_flip_count)
            if consecutive_flip_count >= consecutive_bars_required:
                return {"exit_idx": k, "reason": "REGIME_FLIP", "boundary_truncated": False}

    # ---------------------------------------------------------
    # 3. CHƯA CHẠM NGƯỠNG NÀO -> HẾT THỜI GIAN (TIME_STOP)
    # ---------------------------------------------------------
    last_idx = min(effective_t_max, n_bars) - 1
    is_truncated = (max_lookforward_override is not None) and (effective_t_max < t_max_live)
    return {"exit_idx": max(last_idx, 0), "reason": "TIME_STOP", "boundary_truncated": is_truncated}

# ============================================================================
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
    lows_l  = np.array([100, 101, 102, 85, 85, 85, 85, 85, 85, 85], dtype=float)
    res_l = compute_regime_aware_trailing_exit_v2(100.0, 1, "follow", highs_l, lows_l, atr, p_trend_flat, 95.0, t_max_live=10)
    assert res_l["reason"] == "SL" and res_l["exit_idx"] == 3, f"Long SL test FAILED: {res_l}"

    # Test 2: side=-1 (Short/Fade), giá tăng vượt SL tại k=3
    highs_s = np.array([101, 102, 102.5, 110, 110, 110, 110, 110, 110, 110], dtype=float)
    lows_s  = np.array([100, 101, 101.5, 105, 105, 105, 105, 105, 105, 105], dtype=float)
    res_s = compute_regime_aware_trailing_exit_v2(100.0, -1, "fade", highs_s, lows_s, atr, p_trend_flat, 105.0, t_max_live=10)
    assert res_s["reason"] == "SL" and res_s["exit_idx"] == 3, f"Short SL test FAILED: {res_s}"

    # Test 3: Regime-Flip cho Fade — p_trend TĂNG vượt ngưỡng 0.65 phải kích hoạt thoát tại k=3 (cần 2 nến liên tiếp)
    highs_f = np.full(10, 100.5)
    lows_f  = np.full(10, 99.5)
    p_trend_rising = np.array([0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    res_f_flip = compute_regime_aware_trailing_exit_v2(
        100.0, -1, "fade", highs_f, lows_f, atr, p_trend_rising, 110.0,
        t_max_live=10, p_trend_exit_threshold_fade=0.65, consecutive_bars_required=2
    )
    assert res_f_flip["reason"] == "REGIME_FLIP" and res_f_flip["exit_idx"] == 3, f"Fade regime-flip test FAILED: {res_f_flip}"

    # Test 4: Follow cùng chuỗi p_trend_rising — KHÔNG kích hoạt Regime-Flip vì Follow chỉ thoát khi trend yếu
    res_fl_no_flip = compute_regime_aware_trailing_exit_v2(
        100.0, 1, "follow", highs_f, lows_f, atr, p_trend_rising, 90.0,
        t_max_live=10, p_trend_exit_threshold_follow=0.35, consecutive_bars_required=2
    )
    assert res_fl_no_flip["reason"] == "TIME_STOP", f"Follow false-positive test FAILED: {res_fl_no_flip}"

    # 5. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy mảng rỗng (Empty array crash check)
    try:
        compute_regime_aware_trailing_exit_v2(100.0, 1, "follow", [], [], [], [], 95.0)
        assert False, "Lỗi rò rỉ: Mảng rỗng lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "Mảng future_highs rỗng" in str(e)

    # 6. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy độ dài mảng lệch nhau (Mismatched arrays check)
    try:
        compute_regime_aware_trailing_exit_v2(100.0, 1, "follow", highs_l[:5], lows_l[:4], atr[:5], p_trend_flat[:5], 95.0)
        assert False, "Lỗi rò rỉ: Mảng lệch độ dài lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "Độ dài các mảng tương lai lệch nhau" in str(e)

    # 7. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy ATR âm (Negative ATR check)
    atr_bad = atr.copy()
    atr_bad[2] = -1.0
    try:
        compute_regime_aware_trailing_exit_v2(100.0, 1, "follow", highs_l, lows_l, atr_bad, p_trend_flat, 95.0)
        assert False, "Lỗi rò rỉ: ATR âm lọt qua mà không ném lỗi ValueError!"
    except ValueError as e:
        assert "chứa số âm" in str(e)

    # 8. [ARMOR-PLATED FAULT-INJECTION] Kiểm thử bẻ gãy side = 0 hoặc trade_mode không hợp lệ
    try:
        compute_regime_aware_trailing_exit_v2(100.0, 0, "follow", highs_l, lows_l, atr, p_trend_flat, 95.0)
        assert False, "Lỗi rò rỉ: side=0 lọt qua mà không ném lỗi ValueError!"
    except ValueError:
        pass

    try:
        compute_regime_aware_trailing_exit_v2(100.0, 1, "invalid_mode", highs_l, lows_l, atr, p_trend_flat, 95.0)
        assert False, "Lỗi rò rỉ: trade_mode rác lọt qua mà không ném lỗi ValueError!"
    except ValueError:
        pass

    print("✅ [TASK B-1-4] compute_regime_aware_trailing_exit_v2 PASSED! (Đối xứng hoàn hảo, Regime-Flip chuẩn & Chống 100% mảng rỗng/lệch/ATR âm/mode rác)")

if __name__ == "__main__":
    test_b_1_3_compute_sl_initial()
    test_b_1_4_regime_aware_trailing_exit_symmetry()