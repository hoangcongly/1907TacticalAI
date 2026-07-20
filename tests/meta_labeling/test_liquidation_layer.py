import math
from typing import Any, Dict, List, Optional

from aegis.meta_labeling.sizing.liquidation_layer import (
    compute_liquidation_price,
    validate_leverage_against_sl,
    get_maintenance_margin_rate,
    resolve_max_safe_leverage,
    compute_liquidation_loss,
)

def test_liquidation_layer_armor_plated():
    """
    [v11.9] TDD - RIGOROUS VULNERABILITY AUDIT
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

def test_compute_liquidation_loss_margin_only():
    """
    [QĐ #7] compute_liquidation_loss trả về -(margin) thuần.
    Phí vào lệnh được xử lý thống nhất tại pnl.py.
    """
    notional = 1000.0
    lev = 10.0
    # Tiền cọc = 100 USD. Trả về -100.0 (KHÔNG trừ fee ở đây nữa)
    loss = compute_liquidation_loss(notional, lev)
    assert abs(loss - (-100.0)) < 1e-6, f"Lỗi tính toán: nhận {loss}, kỳ vọng -100.0"
    print("✅ [QĐ #7] compute_liquidation_loss trả về -(margin) thuần, phí xử lý tại pnl.py!")
