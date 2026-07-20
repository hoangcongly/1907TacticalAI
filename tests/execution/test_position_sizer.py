from aegis.execution.position_sizer import compute_position_size

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
