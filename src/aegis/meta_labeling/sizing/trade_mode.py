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
) -> Literal["follow", "fade", "none"]:
    """
    Hàm DUY NHẤT phân loại Follow / Fade / None.
    Dùng chung cho cả bước dựng bảng Kelly và lúc chạy Inference.

    [ARMOR-PLATED GUARDS — BẢO VỆ CHỐNG LỖ HỔNG]:
    - Kiểm tra nghiêm ngặt p_i và p_chop_i phải thuộc đoạn [0.0, 1.0].
    - Chặn đứng số rác NaN / Inf trước khi đi vào logic rẽ nhánh.

    Quy tắc:
    - p_i >= 0.5: Tin tưởng xu hướng (Follow).
    - p_i < 0.2: Xu hướng rất yếu. Đủ điều kiện xét Fade.
    - Khóa cổng Fade: Chỉ kích hoạt nếu xác suất choppy (p_chop_i) > ngưỡng an toàn.
    """
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

    # Khóa cổng: Fade chỉ kích hoạt nếu xác suất choppy > ngưỡng
    if fade_enabled and p_i < 0.2 and p_chop_i > fade_regime_gate_threshold:
        return "fade"

    return "none"



