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

    if p_i >= 0.5:
        return "follow"

    # Khóa cổng: Fade chỉ kích hoạt nếu xác suất choppy > ngưỡng
    if fade_enabled and p_i < 0.2 and p_chop_i > fade_regime_gate_threshold:
        return "fade"

    return "none"


# ============================================================================
# UNIT TESTS (TDD & STRESS TESTS)
# ============================================================================
def test_b_1_2_trade_mode():
    """
    Kiểm tra chặt chẽ các trường hợp biên và kiểm thử bẻ gãy (Fault-Injection).
    """
    # 1. Nhánh Follow
    assert (
        classify_trade_mode(0.5, 0.4, True) == "follow"
    ), "Lỗi: p=0.5 ngay tại biên phải là follow"
    assert (
        classify_trade_mode(0.6, 0.4, True) == "follow"
    ), "Lỗi: p >= 0.5 phải là follow"

    # 2. Vùng Deadzone (Đứng ngoài)
    assert (
        classify_trade_mode(0.3, 0.8, True) == "none"
    ), "Lỗi: p nằm trong [0.2, 0.5) phải là none (Deadzone)"
    assert (
        classify_trade_mode(0.1, 0.5, True) == "none"
    ), "Lỗi: p < 0.2 nhưng p_chop <= 0.6 phải bị khóa (none)"

    # 3. Nhánh Fade
    assert (
        classify_trade_mode(0.1, 0.7, True) == "fade"
    ), "Lỗi: Đủ điều kiện Fade nhưng không kích hoạt"

    # 4. Fade bị Disable
    assert (
        classify_trade_mode(0.1, 0.7, False) == "none"
    ), "Lỗi: Fade đang bị disable thì không được kích hoạt"

    # 5. [ARMOR-PLATED TESTS] Kiểm thử bẻ gãy input rác (NaN / Inf / Out-of-bounds)
    invalid_inputs = [
        (float("nan"), 0.5, True),
        (0.5, float("nan"), True),
        (float("inf"), 0.5, True),
        (-0.1, 0.5, True),
        (1.1, 0.5, True),
        (0.3, -0.05, True),
        (0.3, 1.05, True),
    ]
    for p, p_chop, fe in invalid_inputs:
        try:
            classify_trade_mode(p, p_chop, fe)
            assert (
                False
            ), f"Lỗi rò rỉ: Input dị thường ({p}, {p_chop}) lọt qua hải quan mà không báo lỗi!"
        except ValueError:
            pass

    print(
        "✅ [TASK B-1-2] classify_trade_mode PASSED! (Xử lý mượt mà điểm mù & chống 100% rác NaN/Out-of-bounds)"
    )


if __name__ == "__main__":
    test_b_1_2_trade_mode()
