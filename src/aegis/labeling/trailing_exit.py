import math
import numpy as np
from typing import Optional


# ============================================================================
# [KHẮC PHỤC BẪY 2 - ASYMMETRIC STOP-LOSS SAFE ROUNDING]
# ============================================================================
def round_sl_safe(sl_raw: float, tick_size: float, side: int) -> float:
    """
    Làm tròn giá Stop-Loss theo bước nhảy `tick_size` đảm bảo an toàn bất đối xứng:
    - Long (side > 0): Stop-Loss nằm DƯỚI entry -> làm tròn XUỐNG (floor) để đẩy SL ra xa, tránh cắn sớm.
    - Short/Fade (side < 0): Stop-Loss nằm TRÊN entry -> làm tròn LÊN (ceil) để đẩy SL ra xa, tránh cắn sớm.
    """
    if not isinstance(tick_size, (int, float)) or tick_size <= 0 or math.isnan(tick_size) or math.isinf(tick_size):
        raise ValueError(f"tick_size phải > 0 hợp lệ, nhận {tick_size}")
    if not isinstance(sl_raw, (int, float)) or sl_raw <= 0 or math.isnan(sl_raw) or math.isinf(sl_raw):
        raise ValueError(f"sl_raw phải > 0 hợp lệ, nhận {sl_raw}")
    if side not in (1, -1):
        raise ValueError(f"side phải là 1 hoặc -1, nhận {side}")

    if side > 0:
        steps = math.floor((sl_raw + 1e-12) / tick_size)
    else:
        steps = math.ceil((sl_raw - 1e-12) / tick_size)
    return float(max(1e-4, steps * tick_size))


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
    tick_size: Optional[float] = None,
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
    - tick_size: Bước nhảy giá tối thiểu của sàn. Nếu cung cấp, SL sẽ được làm tròn an toàn (round_sl_safe).
    """
    if side not in (1, -1):
        raise ValueError(
            f"Lỗi hải quan B-1-3: side bắt buộc phải là +1 (Long) hoặc -1 (Short/Fade), nhận {side}"
        )
    #kiểm tra price nhập có hợp lệ không
    if (
        not isinstance(entry_price, (int, float))   #kiểm tra kiểu dữ liệu
        or math.isnan(entry_price) #kiểm tra có phải NaN không
        or math.isinf(entry_price)  #kiểm tra có phải Inf không
        or entry_price <= 0         #kiểm tra có phải số âm không
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: entry_price phải là số dương hợp lệ, nhận {entry_price}"
        )
    #kiểm tra sigma có hợp lệ không     
    if (
        not isinstance(sigma, (int, float))  #kiểm tra kiểu dữ liệu 
        or math.isnan(sigma) #kiểm tra có phải NaN không
        or math.isinf(sigma)  #kiểm tra có phải Inf không
        or sigma < 0 #kiểm tra có phải số âm không
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: sigma (biến động) phải >= 0 và không được là NaN/Inf, nhận {sigma}"
        )
    #kiểm tra m_sl có hợp lệ không
    if (
        not isinstance(m_sl, (int, float))  #kiểm tra kiểu dữ liệu
        or math.isnan(m_sl)                 #kiểm tra có phải NaN không
        or math.isinf(m_sl)                 #kiểm tra có phải Inf không
        or m_sl < 0                         #kiểm tra có phải số âm không
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: m_sl (hệ số cắt lỗ) phải >= 0 và hợp lệ, nhận {m_sl}"
        )
    #kiểm tra c_trade_adj có hợp lệ không
    if (
        not isinstance(c_trade_adj, (int, float))  #kiểm tra kiểu dữ liệu
        or math.isnan(c_trade_adj)                 #kiểm tra có phải NaN không
        or math.isinf(c_trade_adj)                 #kiểm tra có phải Inf không
        or c_trade_adj < 0                         #kiểm tra có phải số âm không
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: c_trade_adj (phí & slippage) phải >= 0 và hợp lệ, nhận {c_trade_adj}"
        )
    #kiểm tra max_reasonable_cushion có hợp lệ không
    if (
        not isinstance(max_reasonable_cushion, (int, float))  #kiểm tra kiểu dữ liệu
        or math.isnan(max_reasonable_cushion)                 #kiểm tra có phải NaN không
        or math.isinf(max_reasonable_cushion)                 #kiểm tra có phải Inf không
        or max_reasonable_cushion <= 0                         #kiểm tra có phải số âm không
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-3: max_reasonable_cushion phải > 0 và hợp lệ, nhận {max_reasonable_cushion}"
        )
    #tính tổng rủi ro trừ hao
    total_cushion = m_sl * sigma + c_trade_adj
    #kiểm tra tổng rủi ro trừ hao có vượt quá ngưỡng hợp lý không
    if total_cushion > max_reasonable_cushion:
        raise ValueError(
            f"Cushion {total_cushion:.2%} vượt ngưỡng hợp lý {max_reasonable_cushion:.2%} "
            f"-- khả năng sigma bị lỗi đơn vị hoặc NaN thoát dạng số lớn bất thường."
        )
    #kiểm tra rủi ro cực đại với lệnh Long
    if side > 0:
        if total_cushion >= 1.0:
            raise ValueError(
                f"Lỗi rủi ro cực đại B-1-3: Tổng rủi ro trừ hao ({total_cushion:.2%}) >= 100% giá trị tài sản với lệnh Long, dẫn đến Stop-Loss <= 0!"
            )
        sl_raw = entry_price * math.exp(-total_cushion)
        sl = max(sl_raw, 1e-4)
    else:  #kiểm tra rủi ro cực đại với lệnh Short/Fade
        sl = entry_price * math.exp(total_cushion)

    if tick_size is not None and tick_size > 0:
        sl = round_sl_safe(sl, tick_size, side)

    return float(sl)


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
    min_tick_size: float = 1e-4,
    min_ticks_cushion: int = 10,
    min_atr_pct: float = 0.001,
) -> Optional[dict]:
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

    # [Vá BỌ SỐ 2: Zero-ATR Trailing Collapse — Instrument-Adaptive & Percentage Anchored Floor]
    # Khi thanh khoản cạn kiệt, ATR tiệm cận 0 -> cushion = 0 -> trailing ôm sát khít 100% gây stop-out oan uổng.
    # Để không phụ thuộc vào một hằng số sàn cố định cho mọi tài sản (như BTC $60,000 vs altcoin $0.001),
    # safe_atr được neo theo 2 ngưỡng bảo vệ:
    # (1) Ngưỡng tuyệt đối theo bước giá sàn (min_tick_size * min_ticks_cushion, ví dụ 10 ticks).
    # (2) Ngưỡng tương đối theo mức giá hiện tại của tài sản (entry_price * min_atr_pct, ví dụ 0.1% giá).
    safe_atr_floor = max(min_tick_size * min_ticks_cushion, entry_price * min_atr_pct)
    safe_atr = np.maximum(atr, safe_atr_floor)

    n_bars = len(highs)
    effective_t_max = (
        min(t_max_live, max_lookforward_override)
        if max_lookforward_override is not None
        else t_max_live
    )

    # [FINDING F & DATA CONTRACT v11.9] Zero-length slice or immediate boundary:
    # Nếu n_bars == 0 hoặc effective_t_max <= 0 (như lệnh mở ngay sát biên Fold OOS),
    # trả về ngay bản ghi TIME_STOP bị cắt cụt bởi biên fold (boundary_truncated = True)
    # mà không cho phép phát sinh crash ValueError.
    if n_bars == 0 or effective_t_max <= 0:
        return {"exit_idx": 0, "reason": "TIME_STOP", "boundary_truncated": True}

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
            trail_cushion = m_trail_base * (1 + gamma * p_trend[k]) * safe_atr[k] / extreme_price
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
            trail_cushion = m_trail_base * (1 + gamma * p_trend[k]) * safe_atr[k] / extreme_price
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
    is_truncated = (
        (max_lookforward_override is not None and effective_t_max < t_max_live)
        or (n_bars < t_max_live)
        or (n_bars <= 1)
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
) -> Optional[dict]:
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

    # Guard an toàn nếu lệnh vào đúng nến cuối của Fold hoặc vượt biên:
    # Trả về bản ghi TIME_STOP với cờ boundary_truncated=True thay vì None
    # để giữ trọn vẹn trong thống kê tổng thể OOS, đồng thời được bộ lọc Kelly tự động loại bỏ.
    if len(future_highs) == 0:
        exit_idx_absolute = min(entry_idx + 1, len(full_highs) - 1)
        return {
            "entry_idx": entry_idx,
            "exit_idx": 0,
            "exit_idx_relative": 0,
            "exit_idx_absolute": exit_idx_absolute,
            "reason": "TIME_STOP",
            "exit_reason": "TIME_STOP",
            "boundary_truncated": True,
        }

    # [STREAMING_CHUNK: SIMULATE_TRAILING_EXIT_CALL]
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

    # [ARMOR GUARD — BẢO VỆ CHỐNG LỖ HỔNG NẾN CẬN BIÊN]:
    # Nếu exit_result là None (kịch bản cực đoan), trả về bản ghi boundary_truncated=True
    if exit_result is None:
        exit_idx_absolute = min(entry_idx + 1, len(full_highs) - 1)
        return {
            "entry_idx": entry_idx,
            "exit_idx": 0,
            "exit_idx_relative": 0,
            "exit_idx_absolute": exit_idx_absolute,
            "reason": "TIME_STOP",
            "exit_reason": "TIME_STOP",
            "boundary_truncated": True,
        }

    exit_idx_relative = int(exit_result["exit_idx"])
    exit_idx_absolute = min(entry_idx + 1 + exit_idx_relative, len(full_highs) - 1)

    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx_relative,
        "exit_idx_relative": exit_idx_relative,
        "exit_idx_absolute": exit_idx_absolute,
        "reason": exit_result["reason"],
        "exit_reason": exit_result["reason"],
        "boundary_truncated": bool(exit_result["boundary_truncated"]),
    }


# ============================================================================
# [TASK B-1-7] PURE FUNCTION — RESOLVE ABSOLUTE EXIT INDEX
# ============================================================================
def resolve_absolute_exit_idx(entry_idx: int, exit_idx_relative: int) -> int:
    """
    Chuyển offset TƯƠNG ĐỐI (k, trả về từ compute_regime_aware_trailing_exit_v3,
    tính từ nến kế tiếp sau entry_idx) sang chỉ số TUYỆT ĐỐI trên toàn bộ mảng dữ
    liệu gốc. Công thức: exit_idx_absolute = entry_idx + 1 + exit_idx_relative.
    """
    return entry_idx + 1 + exit_idx_relative


# ============================================================================
# [TASK B-1-8] GLUE B-1-5 -> B-1-6 -> B-1-7 — RUN TRAILING EXIT FOR OOS EVENT
# ============================================================================
def run_trailing_exit_for_oos_event(
    entry_idx: int,
    entry_price: float,
    test_window_end_idx: int,
    p_i: float,
    p_chop_i: float,
    side_primary: int,
    m_sl: float,
    sigma: float,
    c_trade_adj: float,
    fade_enabled: bool,
    fade_regime_gate_threshold: float,
    full_highs: np.ndarray,
    full_lows: np.ndarray,
    full_atr: np.ndarray,
    full_p_trend: np.ndarray,
    t_max_live_follow: int = 120,
    t_max_live_fade: int = 40,
    leverage_requested: float = 10.0,
    maintenance_margin_rate: float = 0.005,
    fee_rate: float = 0.0004,
    liquidation_fee_rate: float = 0.001,
    full_timestamps: Optional[np.ndarray] = None,
    **trailing_exit_kwargs,
) -> dict | None:
    """
    [TASK B-1-8] Hàm glue nối 3 khâu: mô phỏng Trailing-Exit trong biên fold (B-1-5),
    resolve tham số thực thi (B-1-6), và chuyển đổi chỉ số tuyệt đối (B-1-7).
    """
    from aegis.meta_labeling.sizing.trade_mode import resolve_trade_execution_params

    # 1. resolved = resolve_trade_execution_params(...)
    resolved = resolve_trade_execution_params(
        p_i=p_i,
        p_chop_i=p_chop_i,
        entry_price=entry_price,
        side_primary=side_primary,
        m_sl=m_sl,
        sigma=sigma,
        c_trade_adj=c_trade_adj,
        fade_enabled=fade_enabled,
        fade_regime_gate_threshold=fade_regime_gate_threshold,
        t_max_live_follow=t_max_live_follow,
        t_max_live_fade=t_max_live_fade,
        leverage_requested=leverage_requested,
        maintenance_margin_rate=maintenance_margin_rate,
        fee_rate=fee_rate,
        liquidation_fee_rate=liquidation_fee_rate,
    )

    # 2. if resolved is None: return None
    if resolved is None:
        return None

    # 3. exit_result = simulate_trailing_exit_within_fold_bounds(...)
    exit_result = simulate_trailing_exit_within_fold_bounds(
        entry_idx=entry_idx,
        entry_price=entry_price,
        side=resolved["side"],
        trade_mode=resolved["mode"],
        test_window_end_idx=test_window_end_idx,
        full_highs=full_highs,
        full_lows=full_lows,
        full_atr=full_atr,
        full_p_trend=full_p_trend,
        sl_initial=resolved["sl_initial"],
        liquidation_price=resolved["liquidation_price"],
        t_max_live=resolved["t_max_live"],
        **trailing_exit_kwargs,
    )

    # 4. if exit_result is None: return None
    if exit_result is None:
        return None

    # 5. exit_idx_absolute = resolve_absolute_exit_idx(entry_idx, exit_result["exit_idx"])
    rel_idx = int(exit_result.get("exit_idx", exit_result["exit_idx_relative"]))
    exit_idx_absolute = resolve_absolute_exit_idx(entry_idx, rel_idx)
    reason = exit_result.get("reason", exit_result["exit_reason"])

    entry_ts = int(full_timestamps[entry_idx]) if full_timestamps is not None and 0 <= entry_idx < len(full_timestamps) else 0
    exit_ts = int(full_timestamps[exit_idx_absolute]) if full_timestamps is not None and 0 <= exit_idx_absolute < len(full_timestamps) else 0

    # 6. Trả về dict tuân thủ TRADE_RECORD_SCHEMA (bản ghi TẠM, chưa có realized_return)
    return {
        "entry_idx": entry_idx,
        "entry_timestamp_ms": entry_ts,
        "entry_price": entry_price,
        "p_i": p_i,
        "p_chop_i": p_chop_i,
        "mode": resolved["mode"],
        "side": resolved["side"],
        "sl_initial": resolved["sl_initial"],
        "leverage_used": resolved["leverage_used"],
        "liquidation_price": resolved["liquidation_price"],
        "exit_idx_relative": rel_idx,
        "exit_idx_absolute": exit_idx_absolute,
        "exit_timestamp_ms": exit_ts,
        "exit_reason": reason,
        "boundary_truncated": bool(exit_result["boundary_truncated"]),
    }


# ============================================================================
# [TASK B-1-9] FINALIZE TRADE RECORD (STUB REALIZED PNL)
# ============================================================================
def finalize_trade_record(
    partial_record: dict,
    full_closes: np.ndarray,  # STUB: giá đóng cửa thô. Sẽ thay bằng fill_price thật ở B-8-4.
    size_notional: float,
    fee_entry_rate: float = 0.0004,
    fee_exit_rate: float = 0.0004,
    funding_accrued: float = 0.0,
    full_timestamps: Optional[np.ndarray] = None,
) -> dict:
    """
    [TASK B-1-9] Bước hoàn thiện bản ghi cuối cùng: gắn realized_return vào bản ghi tạm từ B-1-8.

    # TODO(B-8-4): Thay full_closes[exit_idx_absolute] bằng fill_price thật lấy từ
    # Module G Execution Simulator (simulate_market_fill / simulate_limit_fill_with_queue,
    # có tính latency + square-root market impact). Interface exit_idx_absolute PHẢI
    # giữ nguyên -- Module G cũng tra cứu theo đúng chỉ số này, không đổi.
    """
    from aegis.execution.pnl import compute_realized_pnl

    # 1. entry_price = partial_record["entry_price"]
    entry_price = float(partial_record["entry_price"])

    # 2. Xử lý exit_price_stub và gọi compute_realized_pnl
    # PHẢI dùng đúng "exit_idx_absolute", TUYỆT ĐỐI KHÔNG dùng "exit_idx_relative"
    exit_idx_abs = int(partial_record["exit_idx_absolute"])
    
    side = int(partial_record["side"])
    leverage = float(partial_record["leverage_used"])
    from typing import cast, Literal
    exit_reason = cast(Literal['SL', 'TRAIL', 'REGIME_FLIP', 'TIME_STOP', 'LIQUIDATION', 'BOUNDARY_TRUNCATED'], str(partial_record["exit_reason"]))

    if exit_reason == "LIQUIDATION":
        # [KHẮC PHỤC LỖ HỔNG #1]: Đối với thanh lý, giá thoát lệnh chính là giá thanh lý
        exit_price_stub = float(partial_record["liquidation_price"])
    else:
        # [KHẮC PHỤC LỖ HỔNG F7]: Slippage mô phỏng (TODO(B-8-4) removed)
        close_price = float(full_closes[exit_idx_abs])
        spread_pct = 0.0002
        slippage_penalty_factor = 0.1
        # Giả định volume hiện tại (có thể mock nếu chưa truyền vào)
        bar_volume = 1e6
        slippage = close_price * (spread_pct / 2 + slippage_penalty_factor * (float(size_notional) / bar_volume))
        exit_price_stub = close_price - side * slippage

    # Default to taker if not explicitly provided
    entry_fill_type = cast(Literal["maker", "taker"], str(partial_record.get("entry_fill_type", "taker")))
    exit_fill_type = cast(Literal["maker", "taker"], str(partial_record.get("exit_fill_type", "taker")))

    # 3. Tính PnL CHUNG MỘT LUỒNG bằng Module G
    pnl_res = compute_realized_pnl(
        entry_price=entry_price,
        exit_price=exit_price_stub,
        side=side,
        size_notional=size_notional,
        leverage=leverage,
        exit_reason=exit_reason,
        entry_fill_type=entry_fill_type,
        exit_fill_type=exit_fill_type,
        maker_fee_rate=0.0001,
        taker_fee_rate=0.0004,
        funding_accrued_usd=funding_accrued,
        is_notional_in_usd=True,
    )
    
    pnl = float(pnl_res["net_pnl"])
    fee_entry = float(size_notional) * float(fee_entry_rate)
    fee_exit = float(pnl_res["fee_paid"]) - fee_entry
    if fee_exit < 0:
        fee_exit = 0.0
    gross_pnl = float(pnl_res["gross_pnl"])
    
    # [KHẮC PHỤC LỖ HỔNG #1]: Lấy realized_return trực tiếp từ compute_realized_pnl (đã tính r_u chuẩn)
    realized_return = float(pnl_res["realized_return"])

    entry_ts = int(partial_record.get("entry_timestamp_ms", full_timestamps[int(partial_record["entry_idx"])] if full_timestamps is not None and 0 <= int(partial_record["entry_idx"]) < len(full_timestamps) else 0))
    exit_ts = int(partial_record.get("exit_timestamp_ms", full_timestamps[exit_idx_abs] if full_timestamps is not None and 0 <= exit_idx_abs < len(full_timestamps) else 0))

    # 5. Trả về dict đầy đủ tuân thủ TradeRecordSchema (Pandera)
    return {
        "schema_version": partial_record.get("schema_version", "1.0.0"),
        "dataset_manifest_hash": partial_record.get("dataset_manifest_hash", "0" * 64),
        "fold_id": partial_record.get("fold_id", None),
        "symbol": partial_record.get("symbol", "UNKNOWN"),
        "entry_idx": int(partial_record["entry_idx"]),
        "entry_timestamp_ms": entry_ts,
        "entry_price": entry_price,
        "p_i": float(partial_record["p_i"]),
        "p_chop_i": float(partial_record["p_chop_i"]),
        "mode": partial_record["mode"],
        "side": side,
        "sl_initial": float(partial_record["sl_initial"]),
        "size_notional": float(size_notional),
        "exit_idx_relative": int(partial_record.get("exit_idx_relative", partial_record.get("exit_idx", 0))),
        "exit_idx_absolute": exit_idx_abs,
        "exit_timestamp_ms": exit_ts,
        "exit_reason": exit_reason,
        "fill_price_exit": exit_price_stub,
        "boundary_truncated": bool(partial_record["boundary_truncated"]),
        "fee_entry": float(fee_entry),
        "fee_exit": float(fee_exit),
        "funding_accrued": float(funding_accrued),
        "gross_pnl": float(gross_pnl),
        "realized_return": float(realized_return),
    }





