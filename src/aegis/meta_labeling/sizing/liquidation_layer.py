"""
[v11.9] LIQUIDATION LAYER — XẤP XỈ GIÁ THANH LÝ & BẢO VỆ ĐÒN BẦY AN TOÀN PERPETUAL FUTURES.
Được bọc thép 100% (Armor-Plated Guards) ngăn chặn chia cho 0, đòn bẩy âm, SL ngược chiều và side = 0.
"""

import math
from typing import Dict, Any


# ============================================================================
# [TASK v11.9] TÍNH TOÁN GIÁ THANH LÝ VỚI BỌC THÉP BẢO MẬT
# ============================================================================
def compute_liquidation_price(
    entry_price: float, side: int, leverage: float, maintenance_margin_rate: float
) -> float:
    """
    [v11.9] Xấp xỉ giá thanh lý cho Isolated Margin Perpetual Futures.

    [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
    - Chặn side == 0 hoặc ngoài (+1, -1).
    - Chặn leverage < 1.0 hoặc rác NaN/Inf gây lỗi chia cho số 0 (ZeroDivisionError).
    - Chặn entry_price <= 0 và maintenance_margin_rate ngoài đoạn [0, 1).
    """
    if side not in (1, -1):
        raise ValueError(
            f"Lỗi hải quan v11.9 (compute_liquidation_price): side bắt buộc phải là +1 (Long) hoặc -1 (Short), nhận {side}"
        )

    if (
        not isinstance(entry_price, (int, float))
        or math.isnan(entry_price)
        or math.isinf(entry_price)
        or entry_price <= 0
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: entry_price phải là số dương hợp lệ (> 0), nhận {entry_price}"
        )

    if (
        not isinstance(leverage, (int, float))
        or math.isnan(leverage)
        or math.isinf(leverage)
        or leverage < 1.0
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: đòn bẩy leverage phải >= 1.0 và hợp lệ, nhận {leverage}"
        )

    if (
        not isinstance(maintenance_margin_rate, (int, float))
        or math.isnan(maintenance_margin_rate)
        or math.isinf(maintenance_margin_rate)
        or not (0.0 <= maintenance_margin_rate < 1.0)
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: maintenance_margin_rate phải nằm trong đoạn [0.0, 1.0), nhận {maintenance_margin_rate}"
        )

    if side > 0:
        p_liq = entry_price * (1.0 - 1.0 / leverage + maintenance_margin_rate)
        return float(max(p_liq, 1e-4))
    else:
        p_liq = entry_price * (1.0 + 1.0 / leverage - maintenance_margin_rate)
        return float(p_liq)


def validate_leverage_against_sl(
    entry_price: float,
    side: int,
    sl_initial: float,
    leverage: float,
    maintenance_margin_rate: float,
    safety_buffer_pct: float = 0.15,
) -> Dict[str, Any]:
    """
    [v11.9] Pre-Flight Check: Đảm bảo khoảng cách Cắt Lỗ (SL) đủ an toàn trước khi chạm giá thanh lý.

    [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
    - Chặn safety_buffer_pct ngoài đoạn [0.0, 0.9] ngăn phá vỡ logic đệm an toàn.
    - Chặn điểm Cắt Lỗ sl_initial đặt sai chiều hoặc bằng giá mua (Inverted Stop-Loss Guard).
    """
    if (
        not isinstance(safety_buffer_pct, (int, float))
        or math.isnan(safety_buffer_pct)
        or math.isinf(safety_buffer_pct)
        or not (0.0 <= safety_buffer_pct <= 0.9)
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: safety_buffer_pct chỉ được phép từ 0.0 đến 0.9 (0% - 90%), nhận {safety_buffer_pct}"
        )

    if (
        not isinstance(sl_initial, (int, float))
        or math.isnan(sl_initial)
        or math.isinf(sl_initial)
        or sl_initial <= 0
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: sl_initial phải là số dương > 0, nhận {sl_initial}"
        )

    liq_price = compute_liquidation_price(
        entry_price, side, leverage, maintenance_margin_rate
    )

    if side > 0:
        distance_to_liq = entry_price - liq_price
        distance_to_sl = entry_price - sl_initial
    else:
        distance_to_liq = liq_price - entry_price
        distance_to_sl = sl_initial - entry_price

    if distance_to_sl <= 0:
        raise ValueError(
            f"Lỗi hải quan v11.9: Điểm Cắt Lỗ sl_initial ({sl_initial}) đặt sai chiều hoặc bằng Entry ({entry_price}) cho lệnh side={side}!"
        )

    if distance_to_liq <= 0:
        return {
            "is_safe": False,
            "liq_price": liq_price,
            "distance_to_liq": distance_to_liq,
            "distance_to_sl": distance_to_sl,
            "reason": "LIQ_PRICE_CROSSED_ENTRY",
        }

    is_safe = distance_to_sl <= distance_to_liq * (1.0 - safety_buffer_pct)

    return {
        "is_safe": bool(is_safe),
        "liq_price": float(liq_price),
        "distance_to_liq": float(distance_to_liq),
        "distance_to_sl": float(distance_to_sl),
        "reason": "SAFE" if is_safe else "SL_TOO_CLOSE_TO_LIQ",
    }


def resolve_max_safe_leverage(
    entry_price: float,
    side: int,
    sl_initial: float,
    maintenance_margin_rate: float,
    safety_buffer_pct: float = 0.15,
    leverage_cap: float = 20.0,
) -> float:
    """
    [v11.9] Giải closed-form đòn bẩy tối đa cho phép để đảm bảo SL luôn nằm trong vùng an toàn.
    Công thức giải tích chính xác: L_max = 1 / [ (SL_frac / (1 - buffer)) + MaintRate ]

    [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
    - Chặn sl_initial ngược chiều gây mẫu số âm dẫn đến đòn bẩy ảo khổng lồ.
    - Chặn leverage_cap < 1.0.
    """
    if side not in (1, -1):
        raise ValueError(
            f"Lỗi hải quan v11.9 (resolve_max_safe_leverage): side bắt buộc phải là +1 hoặc -1, nhận {side}"
        )

    if (
        not isinstance(entry_price, (int, float))
        or math.isnan(entry_price)
        or math.isinf(entry_price)
        or entry_price <= 0
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: entry_price phải > 0, nhận {entry_price}"
        )

    if (
        not isinstance(sl_initial, (int, float))
        or math.isnan(sl_initial)
        or math.isinf(sl_initial)
        or sl_initial <= 0
    ):
        raise ValueError(f"Lỗi hải quan v11.9: sl_initial phải > 0, nhận {sl_initial}")

    if (
        not isinstance(safety_buffer_pct, (int, float))
        or math.isnan(safety_buffer_pct)
        or not (0.0 <= safety_buffer_pct <= 0.9)
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: safety_buffer_pct phải từ [0.0, 0.9], nhận {safety_buffer_pct}"
        )

    if (
        not isinstance(leverage_cap, (int, float))
        or math.isnan(leverage_cap)
        or leverage_cap < 1.0
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: leverage_cap phải >= 1.0, nhận {leverage_cap}"
        )

    if side > 0:
        sl_distance_frac = (entry_price - sl_initial) / entry_price
    else:
        sl_distance_frac = (sl_initial - entry_price) / entry_price

    if sl_distance_frac <= 0:
        raise ValueError(
            f"Lỗi hải quan v11.9: Điểm Cắt Lỗ sl_initial ({sl_initial}) đặt sai chiều hoặc bằng Entry ({entry_price}) cho lệnh side={side}!"
        )

    denom = (sl_distance_frac / (1.0 - safety_buffer_pct)) + maintenance_margin_rate
    max_leverage = 1.0 / max(denom, 1e-6)
    return float(min(max_leverage, leverage_cap))


# ============================================================================
# UNIT TESTS (TDD & FAULT-INJECTION STRESS TESTS)
# ============================================================================
def test_liquidation_layer_armor_plated():
    """
    Kiểm thử TDD tính chính xác xấp xỉ giá thanh lý và kiểm thử áp lực bẻ gãy bọc thép.
    """
    entry = 100.0
    sl_long = 90.0  # Cắt lỗ 10% cho Long
    sl_short = 110.0  # Cắt lỗ 10% cho Short
    lev = 5.0
    maint = 0.005  # 0.5%

    # 1. Test case Long chuẩn
    # Liq = 100 * (1 - 0.2 + 0.005) = 80.5
    check_long = validate_leverage_against_sl(
        entry, 1, sl_long, lev, maint, safety_buffer_pct=0.15
    )
    assert check_long["is_safe"] is True, f"Long safe check failed: {check_long}"
    assert (
        abs(check_long["liq_price"] - 80.5) < 1e-6
    ), f"Sai giá thanh lý Long: {check_long['liq_price']}"

    # 2. Test case Short chuẩn
    # Liq = 100 * (1 + 0.2 - 0.005) = 119.5
    check_short = validate_leverage_against_sl(
        entry, -1, sl_short, lev, maint, safety_buffer_pct=0.15
    )
    assert check_short["is_safe"] is True, f"Short safe check failed: {check_short}"
    assert (
        abs(check_short["liq_price"] - 119.5) < 1e-6
    ), f"Sai giá thanh lý Short: {check_short['liq_price']}"

    # 3. Test giải closed-form đòn bẩy tối đa cho Long
    # denom = (0.10 / 0.85) + 0.005 = 0.117647 + 0.005 = 0.122647
    # L_max = 1 / 0.122647 = 8.153478
    max_lev_long = resolve_max_safe_leverage(
        entry, 1, sl_long, maint, safety_buffer_pct=0.15, leverage_cap=20.0
    )
    assert abs(max_lev_long - 8.153478) < 1e-4, f"Sai max safe leverage: {max_lev_long}"

    # 4. [ARMOR-PLATED GUARDS] Khóa lỗi chia cho số 0 (leverage < 1.0 hoặc 0)
    try:
        compute_liquidation_price(entry, 1, 0.0, maint)
        assert False, "Lỗi rò rỉ: leverage = 0.0 không bị chặn!"
    except ValueError as e:
        assert "đòn bẩy leverage phải >= 1.0" in str(e)

    # 5. [ARMOR-PLATED GUARDS] Khóa lỗi side = 0
    try:
        compute_liquidation_price(entry, 0, lev, maint)
        assert False, "Lỗi rò rỉ: side = 0 không bị chặn!"
    except ValueError as e:
        assert "side bắt buộc phải là +1" in str(e)

    # 6. [ARMOR-PLATED GUARDS] Khóa lỗi Cắt lỗ đặt sai chiều (Inverted Stop-Loss)
    try:
        resolve_max_safe_leverage(
            entry, 1, 105.0, maint
        )  # Long nhưng SL ở 105 (> entry)
        assert False, "Lỗi rò rỉ: SL ngược chiều cho Long không bị chặn!"
    except ValueError as e:
        assert "đặt sai chiều hoặc bằng" in str(e)

    # 7. [ARMOR-PLATED GUARDS] Khóa lỗi Lớp đệm an toàn phi lý (safety_buffer_pct > 0.9 hoặc < 0)
    try:
        validate_leverage_against_sl(
            entry, 1, sl_long, lev, maint, safety_buffer_pct=1.2
        )
        assert False, "Lỗi rò rỉ: safety_buffer_pct = 1.2 không bị chặn!"
    except ValueError as e:
        assert "chỉ được phép từ 0.0 đến 0.9" in str(e)

    print(
        "✅ [v11.9] Liquidation Layer PASSED! (Tính chính xác tuyệt đối & Chống 100% chia cho 0 / SL ngược / đòn bẩy rác)"
    )


if __name__ == "__main__":
    test_liquidation_layer_armor_plated()
