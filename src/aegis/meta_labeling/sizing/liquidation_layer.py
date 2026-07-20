"""
[v11.9] LIQUIDATION LAYER — XẤP XỈ GIÁ THANH LÝ & BẢO VỆ ĐÒN BẦY AN TOÀN PERPETUAL FUTURES.
Được bọc thép 100% (Armor-Plated Guards) ngăn chặn chia cho 0, đòn bẩy âm, SL ngược chiều và side = 0.
"""

import math
from typing import Dict, Any, Optional, List


# ============================================================================
# HẰNG SỐ KIẾN TRÚC — TRẦN MARGIN EFFICIENCY (độc lập với Kelly notional cap)
# ============================================================================
# Trần MARGIN EFFICIENCY: chỉ ảnh hưởng khoảng cách tới liquidation, không ảnh hưởng PnL.
# Có thể để cao CHỈ KHI safety_buffer_pct (15%) đã được xác nhận đủ an toàn qua Micro-Live.
MARGIN_LEVERAGE_CAP = 20.0


# ============================================================================
# [TASK v11.9] TÍNH TOÁN GIÁ THANH LÝ VỚI BỌC THÉP BẢO MẬT
# ============================================================================
def compute_liquidation_price(
    entry_price: float, side: int, leverage: float, maintenance_margin_rate: float,
    fee_rate: float = 0.0004, liquidation_fee_rate: float = 0.005
) -> float:
    """
    [v11.9] Xấp xỉ giá thanh lý cho Isolated Margin Perpetual Futures.

    Tham số:
    - fee_rate: Phí giao dịch thường khi mở lệnh (taker/maker fee, ví dụ 0.04%).
    - liquidation_fee_rate: Phí phạt thanh lý cưỡng chế (Liquidation Clearance Fee, ví dụ 0.5%).
      Sàn thu phí này thay vì fee_rate khi cưỡng chế đóng lệnh. Thường cao gấp 10-20 lần fee_rate.

    [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
    - Chặn side == 0 hoặc ngoài (+1, -1).
    - Chặn leverage < 1.0 hoặc rác NaN/Inf gây lỗi chia cho số 0 (ZeroDivisionError).
    - Chặn entry_price <= 0 và maintenance_margin_rate ngoài đoạn [0, 1).
    - Chặn fee_rate ngoài đoạn [0, 0.05).
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

    if (
        not isinstance(fee_rate, (int, float))
        or math.isnan(fee_rate)
        or math.isinf(fee_rate)
        or not (0.0 <= fee_rate < 0.05)
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: fee_rate phải nằm trong đoạn [0.0, 0.05), nhận {fee_rate}"
        )

    # Khoảng cho phép lỗ trước khi sàn thanh lý (trừ hao mọi loại phí)
    margin_loss_allowance = (1.0 / leverage) - maintenance_margin_rate - fee_rate - liquidation_fee_rate

    # Nếu margin_loss_allowance <= 0, đòn bẩy quá cao so với tổng phí → giá thanh lý = entry
    if margin_loss_allowance <= 0:
        return float(entry_price)

    if side > 0:
        p_liq = entry_price * (1.0 - margin_loss_allowance)
        return float(max(p_liq, 1e-4))
    else:
        p_liq = entry_price * (1.0 + margin_loss_allowance)
        return float(p_liq)


def validate_leverage_against_sl(
    entry_price: float,
    side: int,
    sl_initial: float,
    leverage: float,
    maintenance_margin_rate: float,
    safety_buffer_pct: float = 0.15,
    fee_rate: float = 0.0004,
    liquidation_fee_rate: float = 0.005,
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
        entry_price, side, leverage, maintenance_margin_rate, fee_rate, liquidation_fee_rate
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


def get_maintenance_margin_rate(size_notional: float, margin_tier_table: Optional[List[dict]] = None) -> float:
    """
    Tra cứu tỷ lệ Ký quỹ duy trì (MMR) theo quy mô vị thế.
    [VÁ LỖ HỔNG 3]: Bọc thép chống số âm và rác NaN/Inf.
    """
    if math.isnan(size_notional) or math.isinf(size_notional) or size_notional < 0:
        raise ValueError(f"size_notional không hợp lệ: {size_notional}")

    if margin_tier_table is None:
        return 0.005
    for tier in margin_tier_table:
        if size_notional <= tier["max_notional"]:
            return tier["mmr"]
    return margin_tier_table[-1]["mmr"]


def resolve_max_safe_leverage(
    entry_price: float,
    side: int,
    sl_initial: float,
    maintenance_margin_rate: float,
    safety_buffer_pct: float = 0.15,
    leverage_cap: float = MARGIN_LEVERAGE_CAP,
    fee_rate: float = 0.0004,
    liquidation_fee_rate: float = 0.005,
) -> float:
    """
    [v11.9] Giải closed-form đòn bẩy tối đa cho phép.
    Công thức: L_max = 1 / [ (SL_frac / (1 - buffer)) + MaintRate + fee_rate + liquidation_fee_rate ]

    Tham số:
    - fee_rate: Phí giao dịch khi mở lệnh (taker fee, ví dụ 0.04%).
    - liquidation_fee_rate: Phí phạt thanh lý cưỡng chế (Liquidation Clearance Fee, ví dụ 0.5%).
    - leverage_cap: Trần MARGIN EFFICIENCY (mặc định MARGIN_LEVERAGE_CAP = 20.0).
      Đây là trần margin, KHÔNG PHẢI trần rủi ro giá (đó là f_max trong Kelly).
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
            f"Lỗi hải quan v11.9: safety_buffer_pct chỉ được phép từ 0.0 đến 0.9 (0% - 90%), nhận {safety_buffer_pct}"
        )

    if (
        not isinstance(fee_rate, (int, float))
        or math.isnan(fee_rate)
        or math.isinf(fee_rate)
        or not (0.0 <= fee_rate < 0.05)
    ):
        raise ValueError(
            f"Lỗi hải quan v11.9: fee_rate phải nằm trong đoạn [0.0, 0.05), nhận {fee_rate}"
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

    denom = (sl_distance_frac / (1.0 - safety_buffer_pct)) + maintenance_margin_rate + fee_rate + liquidation_fee_rate
    max_leverage = 1.0 / max(denom, 1e-6)
    return float(min(max_leverage, leverage_cap))


def compute_liquidation_loss(
    size_notional: float,
    leverage: float,
    fee_entry_rate: float = 0.0005
) -> float:
    """
    [VÁ LỖ HỔNG 1]: Tính tổng tổn thất khi bị thanh lý.
    Bao gồm vốn ký quỹ đã cọc (Isolated Margin) VÀ phí chìm lúc vào lệnh (Sunk-cost).
    """
    if not isinstance(size_notional, (int, float)) or size_notional <= 0:
        raise ValueError(f"size_notional phải > 0, nhận {size_notional}")
    if not isinstance(leverage, (int, float)) or leverage < 1.0:
        raise ValueError(f"leverage phải >= 1.0, nhận {leverage}")
    if math.isnan(size_notional) or math.isnan(leverage):
        raise ValueError("Input chứa rác NaN")

    theoretical_margin_loss = size_notional / leverage
    sunk_fee_cost = size_notional * fee_entry_rate
    
    return float(- (theoretical_margin_loss + sunk_fee_cost))


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
    # margin_loss_allowance = 1/5 - 0.005 - 0.0004 - 0.005 = 0.1896
    # Liq = 100 * (1 - 0.1896) = 81.04
    check_long = validate_leverage_against_sl(
        entry, 1, sl_long, lev, maint, safety_buffer_pct=0.15, fee_rate=0.0004
    )
    assert check_long["is_safe"] is True, f"Long safe check failed: {check_long}"
    expected_liq_long = 100.0 * (1.0 - (1.0/5.0 - 0.005 - 0.0004 - 0.005))  # = 81.04
    assert (
        abs(check_long["liq_price"] - expected_liq_long) < 1e-4
    ), f"Sai giá thanh lý Long: {check_long['liq_price']}, kỳ vọng {expected_liq_long}"

    # 2. Test case Short chuẩn
    check_short = validate_leverage_against_sl(
        entry, -1, sl_short, lev, maint, safety_buffer_pct=0.15, fee_rate=0.0004
    )
    assert check_short["is_safe"] is True, f"Short safe check failed: {check_short}"
    expected_liq_short = 100.0 * (1.0 + (1.0/5.0 - 0.005 - 0.0004 - 0.005))  # = 118.96
    assert (
        abs(check_short["liq_price"] - expected_liq_short) < 1e-4
    ), f"Sai giá thanh lý Short: {check_short['liq_price']}, kỳ vọng {expected_liq_short}"

    # 3. Test giải closed-form đòn bẩy tối đa cho Long
    max_lev_long = resolve_max_safe_leverage(
        entry, 1, sl_long, maint, safety_buffer_pct=0.15, leverage_cap=20.0, fee_rate=0.0004
    )
    # denom = (0.10 / 0.85) + 0.005 + 0.0004 + 0.005 = 0.117647 + 0.0104 = 0.128047...
    expected_l_max = 1.0 / ((0.10 / 0.85) + 0.005 + 0.0004 + 0.005)
    assert abs(max_lev_long - expected_l_max) < 1e-3, f"Sai max safe leverage: {max_lev_long}, kỳ vọng {expected_l_max}"

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
            entry, 1, 105.0, maint  # Long nhưng SL ở 105 (> entry)
        )
        assert False, "Lỗi rò rỉ: SL ngược chiều cho Long không bị chặn!"
    except ValueError as e:
        assert "đặt sai chiều hoặc bằng" in str(e)

    # 7. [ARMOR-PLATED GUARDS] Khóa lỗi Lớp đệm an toàn phi lý (safety_buffer_pct > 0.9 hoặc < 0)
    try:
        validate_leverage_against_sl(
            entry, 1, sl_long, lev, maint, safety_buffer_pct=1.0, fee_rate=0.0004
        )
        assert False, "Lỗi rò rỉ: safety_buffer_pct = 1.0 không bị chặn!"
    except ValueError as e:
        assert "chỉ được phép từ 0.0 đến 0.9" in str(e)

    print(
        "✅ [v11.9] Liquidation Layer PASSED! (Tính chính xác tuyệt đối & Chống 100% chia cho 0 / SL ngược / đòn bẩy rác)"
    )


def test_liquidation_fee_impact():
    """
    [CODE REVIEW V2] So sánh L_max với và không có phí thanh lý,
    chứng minh rằng phí thanh lý siết chặt đòn bẩy.
    """
    l_max_no_liq_fee = resolve_max_safe_leverage(
        100, 1, 95, 0.005, fee_rate=0.0005, liquidation_fee_rate=0.0, safety_buffer_pct=0.15
    )
    l_max_with_liq_fee = resolve_max_safe_leverage(
        100, 1, 95, 0.005, fee_rate=0.0005, liquidation_fee_rate=0.01, safety_buffer_pct=0.15
    )

    assert l_max_with_liq_fee < l_max_no_liq_fee, (
        "L_max phải bị siết chặt hơn khi có phí thanh lý!"
    )
    print(
        f"✅ [LIQ FEE IMPACT] L_max No Clearance Fee: {l_max_no_liq_fee:.2f}x | "
        f"L_max WITH Clearance Fee: {l_max_with_liq_fee:.2f}x"
    )


def test_get_maintenance_margin_rate_guards():
    try:
        get_maintenance_margin_rate(-1000)
        assert False, "Lỗi: Không chặn size_notional âm"
    except ValueError:
        pass
        
    try:
        get_maintenance_margin_rate(float('nan'))
        assert False, "Lỗi: Không chặn size_notional rác NaN"
    except ValueError:
        pass
    print("✅ [LỖ HỔNG 3 VÁ THÀNH CÔNG] Guard chặn rác cho tra cứu MMR hoạt động hoàn hảo!")

def test_compute_liquidation_loss_with_sunk_cost():
    notional = 1000.0
    lev = 10.0
    # Tiền cọc = 100 USD. Phí vào lệnh (0.05%) = 0.5 USD.
    # Tổng lỗ phải là -100.5 USD
    loss = compute_liquidation_loss(notional, lev, fee_entry_rate=0.0005)
    assert abs(loss - (-100.5)) < 1e-6, f"Lỗi tính toán: nhận {loss}, kỳ vọng -100.5"
    print("✅ [LỖ HỔNG 1 VÁ THÀNH CÔNG] Phí chìm Sunk-Cost đã được trừ tuyệt đối vào PnL thanh lý!")

if __name__ == "__main__":
    test_liquidation_layer_armor_plated()
    test_liquidation_fee_impact()
    test_get_maintenance_margin_rate_guards()
    test_compute_liquidation_loss_with_sunk_cost()
