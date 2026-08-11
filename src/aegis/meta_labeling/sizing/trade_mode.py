"""classify_trade_mode() — HÀM DUY NHẤT phân loại follow/fade/none (v11.6 Patch C.1)."""

import math
from typing import Literal


# ============================================================================
# [TASK B-1-2] TRADE MODE CLASSIFICATION WITH ARMOR-PLATED GUARD
# ============================================================================
def classify_trade_mode(
    p_i: float,
    p_chop_i: float,
    fade_enabled: bool,
    fade_regime_gate_threshold: float = 0.60,
    n_states: int = 2,
) -> Literal["follow", "fade", "none"]:
    """
    Hàm DUY NHẤT phân loại Follow / Fade / None.
    Dùng chung cho cả bước dựng bảng Kelly và lúc chạy Inference.

    [ARMOR-PLATED GUARDS — BẢO VỆ CHỐNG LỖ HỔNG]:
    - Kiểm tra nghiêm ngặt p_i và p_chop_i phải thuộc đoạn [0.0, 1.0].
    - Chặn đứng số rác NaN / Inf trước khi đi vào logic rẽ nhánh.
    - Hằng số kiến trúc n_states (Architectural Constant) truyền trực tiếp vào hàm để rẽ nhánh,
      tuyệt đối không dùng float addition guessing.

    Quy tắc:
    - p_i >= 0.5: Tin tưởng xu hướng (Follow).
    - Khóa cổng Fade theo n_states:
      + Nếu n_states == 2 (HMM 2 trạng thái Trend vs Chop, p_trend + p_chop = 1.0):
        Chỉ kích hoạt Fade nếu p_chop_i > max(0.80, fade_regime_gate_threshold).
      + Nếu n_states >= 3 (HMM >= 3 trạng thái Bull, Bear, Chop, độc lập):
        Chỉ kích hoạt Fade nếu p_i < 0.2 VÀ p_chop_i > fade_regime_gate_threshold.
    """
    if not isinstance(n_states, int) or n_states < 2:
        raise ValueError(
            f"Lỗi hải quan B-1-2: n_states phải là số nguyên >= 2, nhận {n_states}"
        )

    if (
        not isinstance(p_i, (int, float))
        or math.isnan(p_i)
        or math.isinf(p_i)
        or not (0.0 <= p_i <= 1.0)
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-2: p_i phải là số thực hợp lệ trong đoạn [0, 1], nhận {p_i}"
        )

    if (
        not isinstance(p_chop_i, (int, float))
        or math.isnan(p_chop_i)
        or math.isinf(p_chop_i)
        or not (0.0 <= p_chop_i <= 1.0)
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-2: p_chop_i phải là số thực hợp lệ trong đoạn [0, 1], nhận {p_chop_i}"
        )

    if fade_enabled:
        if not isinstance(fade_regime_gate_threshold, (int, float)) or math.isnan(fade_regime_gate_threshold) or math.isinf(fade_regime_gate_threshold) or not (0.0 <= fade_regime_gate_threshold <= 1.0):
            raise ValueError(
                f"Lỗi hải quan B-1-2: fade_regime_gate_threshold phải là số hợp lệ [0, 1], nhận {fade_regime_gate_threshold}"
            )

    if p_i >= 0.5:
        return "follow"

    # Khóa cổng: Fade chỉ kích hoạt nếu xác suất choppy > ngưỡng theo kiến trúc n_states
    if fade_enabled:
        if n_states == 2:
            if p_chop_i > max(0.80, fade_regime_gate_threshold):
                return "fade"
        else:
            if p_i < 0.2 and p_chop_i > fade_regime_gate_threshold:
                return "fade"

    return "none"


# ============================================================================
# [TASK B-1-6] GLUE LAYER — RESOLVE TRADE EXECUTION PARAMS
# ============================================================================
def resolve_trade_execution_params(
    p_i: float,
    p_chop_i: float,
    entry_price: float,
    side_primary: int,
    m_sl: float,
    sigma: float,
    c_trade_adj: float,
    fade_enabled: bool,
    fade_regime_gate_threshold: float = 0.60,
    n_states: int = 2,
    t_max_live_follow: int = 120,
    t_max_live_fade: int = 40,
    leverage_requested: float = 10.0,
    maintenance_margin_rate: float = 0.005,
    fee_rate: float = 0.0004,
    liquidation_fee_rate: float = 0.005,
    safety_buffer_pct: float = 0.15,
    leverage_cap: float = 20.0,
    max_expected_funding_loss: float = 0.0,
) -> dict | None:
    """
    [TASK B-1-6] Hàm glue NỐI 3 module đã có (B-1-2 classify_trade_mode,
    B-1-3 compute_sl_initial, v11.9 liquidation_layer.py).
    """
    from aegis.labeling.trailing_exit import compute_sl_initial
    from aegis.meta_labeling.sizing.liquidation_layer import (
        resolve_max_safe_leverage,
        compute_liquidation_price,
    )

    # 1. Gọi classify_trade_mode
    mode = classify_trade_mode(
        p_i=p_i,
        p_chop_i=p_chop_i,
        fade_enabled=fade_enabled,
        fade_regime_gate_threshold=fade_regime_gate_threshold,
        n_states=n_states,
    )

    # 2. Nếu mode == "none": return None ngay lập tức
    if mode == "none":
        return None

    # 3. Đảo dấu bắt buộc cho Fade
    side_actual = side_primary if mode == "follow" else -side_primary

    # 4. Tính sl_initial theo side_actual (TUYỆT ĐỐI KHÔNG dùng side_primary)
    sl_initial = compute_sl_initial(
        entry_price=entry_price,
        side=side_actual,
        m_sl=m_sl,
        sigma=sigma,
        c_trade_adj=c_trade_adj,
    )

    # 5. Xác định t_max_live theo chế độ
    t_max_live = t_max_live_follow if mode == "follow" else t_max_live_fade

    # 6. Tính toán và kiểm tra đòn bẩy an toàn max_safe
    max_safe = resolve_max_safe_leverage(
        entry_price=entry_price,
        side=side_actual,
        sl_initial=sl_initial,
        maintenance_margin_rate=maintenance_margin_rate,
        safety_buffer_pct=safety_buffer_pct,
        leverage_cap=leverage_cap,
        fee_rate=fee_rate,
        liquidation_fee_rate=liquidation_fee_rate,
        max_expected_funding_loss=max_expected_funding_loss,
    )

    if (
        not isinstance(max_safe, (int, float))
        or math.isnan(max_safe)
        or math.isinf(max_safe)
        or max_safe < 1.0
    ):
        raise ValueError(
            f"Lỗi hải quan B-1-6: Không có đòn bẩy an toàn nào phù hợp (max_safe={max_safe} < 1.0)"
        )

    leverage_used = min(float(leverage_requested), float(max_safe))
    if leverage_used < 1.0:
        raise ValueError(
            f"Lỗi hải quan B-1-6: leverage_used ({leverage_used}) < 1.0"
        )

    # 7. Tính giá thanh lý liquidation_price
    liquidation_price = compute_liquidation_price(
        entry_price=entry_price,
        side=side_actual,
        leverage=leverage_used,
        maintenance_margin_rate=maintenance_margin_rate,
        fee_rate=fee_rate,
        liquidation_fee_rate=liquidation_fee_rate,
    )

    # 8. Trả về từ điển thông số thực thi
    return {
        "mode": mode,
        "side": side_actual,
        "sl_initial": sl_initial,
        "t_max_live": t_max_live,
        "leverage_used": leverage_used,
        "liquidation_price": liquidation_price,
    }




