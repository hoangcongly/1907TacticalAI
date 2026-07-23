import pytest
from aegis.execution.position_sizer import compute_position_size, AccountStateTracker

def test_position_size_uses_current_equity():
    """
    [PHÁT HIỆN O] Xác nhận size_notional PHẢI thay đổi khi Equity thay đổi,
    chứng minh hệ thống dùng vốn hiện tại (compounding), không phải vốn gốc cố định.
    """
    f_star = 0.3
    size_1000 = compute_position_size(f_star=f_star, current_equity=1000.0)
    size_2000 = compute_position_size(f_star=f_star, current_equity=2000.0)

    assert abs(size_2000 - size_1000 * 2.0) < 1e-6, (
        f"size_notional phải tỷ lệ thuận với Equity: {size_2000} vs {size_1000 * 2.0}"
    )
    # Kiểm tra giá trị cụ thể: 0.3 * 0.5 * 1000 = 150
    assert abs(size_1000 - 150.0) < 1e-6, f"Kỳ vọng 150.0, nhận {size_1000}"
    print("✅ [PHÁT HIỆN O] size_notional tỷ lệ thuận với Equity PASSED!")


def test_position_size_zero_f_star():
    """f_star = 0 → Không cược tiền (kỳ vọng âm hoặc thiếu dữ liệu)."""
    size = compute_position_size(f_star=0.0, current_equity=10000.0)
    assert size == 0.0, f"f_star=0 phải trả về 0, nhận {size}"
    print("✅ f_star=0 → size=0 PASSED!")


def test_position_size_max_cap():
    """Trần tuyệt đối giới hạn size_notional."""
    size = compute_position_size(
        f_star=10.0, current_equity=100000.0,
        lambda_kelly=0.5, max_notional_cap=50000.0
    )
    assert abs(size - 50000.0) < 1e-6, f"Phải bị giới hạn ở 50000, nhận {size}"
    print("✅ max_notional_cap PASSED!")


def test_position_size_armor_guards():
    """[ARMOR GUARD] Chặn input rác."""
    bad_inputs = [
        {"f_star": -1.0, "current_equity": 1000.0},
        {"f_star": float("nan"), "current_equity": 1000.0},
        {"f_star": 0.3, "current_equity": -1000.0},
        {"f_star": 0.3, "current_equity": 0.0},
        {"f_star": 0.3, "current_equity": 1000.0, "lambda_kelly": 0.0},
        {"f_star": 0.3, "current_equity": 1000.0, "lambda_kelly": 1.5},
    ]
    for kwargs in bad_inputs:
        try:
            compute_position_size(**kwargs)
            assert False, f"Không chặn được input rác: {kwargs}"
        except ValueError:
            pass
    print("✅ [ARMOR GUARD] Position Sizer chặn mọi input rác PASSED!")


def test_position_size_vol_ratio_black_swan():
    """
    [VOL-TARGETING] Kiểm tra chức năng Volatility Scaling Ratio (Bóp nghẹt Thiên Nga Đen).
    """
    f_star = 2.0
    equity = 100_000.0
    
    # TH1: Thị trường bình thường (ATR_t = ATR_hist = 2%)
    size_normal = compute_position_size(
        f_star=f_star, current_equity=equity, lambda_kelly=0.5,
        atr_hist_mean_pct=0.02, atr_current_pct=0.02
    )
    # Size = 100k * 2.0 * 0.5 * min(1.0, 1.0) = 100k
    assert abs(size_normal - 100_000.0) < 1.0
    
    # TH2: Flash Crash (ATR_t vọt lên 10%, gấp 5 lần quá khứ)
    size_crash = compute_position_size(
        f_star=f_star, current_equity=equity, lambda_kelly=0.5,
        atr_hist_mean_pct=0.02, atr_current_pct=0.10
    )
    # Size = 100k * 2.0 * 0.5 * min(1.0, 0.2) = 20k (Bị chém mất 80% sức mua)
    assert abs(size_crash - 20_000.0) < 1.0
    print(f"✅ [VOL-RATIO] Bóp nghẹt chuẩn xác thứ nguyên. Size Bình thường: ${size_normal:,.0f} -> Size Flash Crash: ${size_crash:,.0f}")


def test_position_size_inf_guards():
    """Kiểm chứng hệ thống chặn đứng input ATR hoặc max_notional_cap bị Inf."""
    import pytest
    with pytest.raises(ValueError, match="ATR hiện tại rác"):
        compute_position_size(f_star=1.0, current_equity=1000.0, atr_hist_mean_pct=0.02, atr_current_pct=float("inf"))
    with pytest.raises(ValueError, match="ATR lịch sử rác"):
        compute_position_size(f_star=1.0, current_equity=1000.0, atr_hist_mean_pct=float("inf"), atr_current_pct=0.02)
    with pytest.raises(ValueError, match="max_notional_cap phải > 0 và hợp lệ"):
        compute_position_size(f_star=1.0, current_equity=1000.0, max_notional_cap=float("inf"))
    print("✅ [ARMOR GUARD] Position Sizer chặn đứng Inf cho ATR/Cap PASSED!")


def test_compute_position_size_round_notional_down():
    """
    [TDD VERIFICATION - KHẮC PHỤC LỖ HỔNG 5: LOT SIZE PRECISION]:
    Kiểm chứng quy mô danh nghĩa (size_notional) được làm tròn xuống (floor / round down)
    theo bước nhảy lot_step_size của sàn, bảo vệ không bao giờ làm tròn lên vượt đòn bẩy/vốn.
    """
    # f_star * lambda_kelly * equity = 0.5 * 0.5 * 1000 = 250.0.
    # Giả sử do vol_multiplier hay trần dẫn tới thô là 257.8 USD, bước nhảy lot_step_size = 10.0
    size = compute_position_size(
        f_star=0.5156, current_equity=1000.0, lambda_kelly=0.5, lot_step_size=10.0
    )
    # 0.5156 * 0.5 * 1000 = 257.8 -> floor(257.8 / 10.0) * 10.0 = 250.0
    assert size == pytest.approx(250.0)


def test_vol_targeting_clamped_by_lmax_safety_gate():
    """
    [TDD VERIFICATION - VÁ LỖ HỔNG 7 & BẪY 1: VOL-TARGETING WITH L_MAX GATE]:
    Kiểm chứng hệ thống cho phép upscaling biến động (vol_ratio up to 2.5x khi ATR hiện tại rất nhỏ),
    nhưng bắt buộc bị kẹp bởi rào chắn L_max safety gate kiểm duyệt lần cuối.
    """
    # Khi ATR_current = 1%, ATR_hist = 5% -> vol_ratio = 5.0 -> bị kẹp ở max_vol_multiplier = 2.5
    # f_star = 1.0, lambda = 0.5, equity = 1000.0 -> size ban đầu = 1.0 * 0.5 * 1000 * 2.5 = 1250.0
    size_no_gate = compute_position_size(
        f_star=1.0, current_equity=1000.0, lambda_kelly=0.5,
        atr_hist_mean_pct=0.05, atr_current_pct=0.01,
        max_vol_multiplier=2.5, max_safe_leverage=None
    )
    assert abs(size_no_gate - 1250.0) < 1e-6, f"Kỳ vọng 1250.0, nhận {size_no_gate}"

    # Khi có rào chắn L_max = 1.0x (chẳng hạn SL đặt rất xa, hoặc margin cap giới hạn ở 1.0x = 1000 USD)
    size_with_gate = compute_position_size(
        f_star=1.0, current_equity=1000.0, lambda_kelly=0.5,
        atr_hist_mean_pct=0.05, atr_current_pct=0.01,
        max_vol_multiplier=2.5, max_safe_leverage=1.0
    )
    assert abs(size_with_gate - 1000.0) < 1e-6, (
        f"Size phải bị kẹp lại ở mức L_max * equity = 1000.0, nhận {size_with_gate}"
    )
    print("✅ [L_MAX SAFETY GATE] Vol-targeting upscaling bị kẹp bởi L_max safety gate PASSED!")


def test_account_state_tracker_isolated_margin_no_upnl_leak():
    """
    [TDD VERIFICATION - VÁ LỖ HỔNG 10 & BẪY 3: ACCOUNT STATE TRACKER]:
    Kiểm chứng AccountStateTracker phân định wallet_balance và unrealized_pnl,
    đảm bảo available_margin tuyệt đối không bao gồm uPnL (lãi lơ lửng chưa chốt)
    trong chế độ Isolated Margin.
    """
    # Ví thật = 10,000 USD, đang mở lệnh khác tốn 2,000 USD cọc, đang lãi lơ lửng 5,000 USD
    tracker = AccountStateTracker(
        wallet_balance=10_000.0,
        unrealized_pnl=5_000.0,
        used_initial_margin=2_000.0
    )
    assert abs(tracker.margin_balance - 15_000.0) < 1e-6, "Mark-to-Market equity phải là 15k"
    assert abs(tracker.available_margin - 8_000.0) < 1e-6, "Available margin chỉ được là 10k - 2k = 8k, không cộng uPnL!"

    # Khi đưa vào compute_position_size ở chế độ chuẩn Isolated Margin
    size = compute_position_size(f_star=1.0, current_equity=tracker, lambda_kelly=0.5)
    # size = 1.0 * 0.5 * 8000.0 = 4000.0
    assert abs(size - 4000.0) < 1e-6, f"Kỳ vọng 4000.0 (dựa trên 8k available_margin), nhận {size}"

    # Kiểm tra guard rác
    with pytest.raises(ValueError):
        AccountStateTracker(wallet_balance=-100.0)
    with pytest.raises(ValueError):
        AccountStateTracker(wallet_balance=100.0, used_initial_margin=-50.0)

    print("✅ [ACCOUNT STATE TRACKER] Phân định wallet_balance & uPnL, chống lọt cọc ảo PASSED!")



