from aegis.meta_labeling.sizing.trade_mode import classify_trade_mode

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
