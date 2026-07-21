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


# ============================================================================
# [TASK B-1-6] UNIT TESTS
# ============================================================================
from aegis.meta_labeling.sizing.trade_mode import resolve_trade_execution_params
from aegis.labeling.trailing_exit import compute_sl_initial
from aegis.meta_labeling.sizing.liquidation_layer import validate_leverage_against_sl


def test_resolve_trade_execution_params_symmetry():
    # Case Fade: side phải đảo dấu, sl_initial PHẢI khác với sl_initial nếu tính
    # (sai) theo side_primary chưa đảo.
    entry_price = 100.0
    side_primary = 1
    resolved_fade = resolve_trade_execution_params(
        p_i=0.1, p_chop_i=0.75, entry_price=entry_price, side_primary=side_primary,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001, fade_enabled=True
    )
    assert resolved_fade is not None
    assert resolved_fade["mode"] == "fade"
    assert resolved_fade["side"] == -side_primary
    sl_if_wrongly_reused = compute_sl_initial(entry_price, side_primary, 2.0, 0.01, 0.001)
    assert resolved_fade["sl_initial"] != sl_if_wrongly_reused
    assert resolved_fade["t_max_live"] == 40  # t_max_live_fade mặc định

    resolved_follow = resolve_trade_execution_params(
        p_i=0.8, p_chop_i=0.3, entry_price=entry_price, side_primary=side_primary,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001, fade_enabled=True
    )
    assert resolved_follow is not None
    assert resolved_follow["mode"] == "follow"
    assert resolved_follow["side"] == side_primary
    assert resolved_follow["t_max_live"] == 120


def test_resolve_trade_execution_params_deadzone_returns_none():
    result = resolve_trade_execution_params(
        p_i=0.3, p_chop_i=0.5, entry_price=100.0, side_primary=1,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001, fade_enabled=True
    )
    assert result is None


def test_resolve_trade_execution_params_liquidation_safety():
    # Xác nhận leverage_used luôn giữ sl_initial an toàn trước liquidation_price
    resolved = resolve_trade_execution_params(
        p_i=0.8, p_chop_i=0.3, entry_price=100.0, side_primary=1,
        m_sl=2.0, sigma=0.01, c_trade_adj=0.001, fade_enabled=True
    )
    assert resolved is not None
    safety = validate_leverage_against_sl(
        100.0, resolved["side"], resolved["sl_initial"], resolved["leverage_used"],
        maintenance_margin_rate=0.005
    )
    assert safety["is_safe"] is True

