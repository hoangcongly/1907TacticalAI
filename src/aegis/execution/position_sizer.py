"""
[PHÁT HIỆN O] Module G — Cầu Nối Kelly → Lệnh Thật.
Hàm compute_position_size biến f* (tỷ lệ trừu tượng) thành size_notional thật
bằng công thức: size_notional = f* × λ × current_equity.

PHẢI sử dụng current_equity (mark-to-market, cập nhật mỗi lệnh),
KHÔNG PHẢI vốn gốc cố định — để phát huy lợi thế compounding.
"""

import math

from aegis.meta_labeling.sizing.kelly_empirical import DEFAULT_LAMBDA_KELLY


# ============================================================================
# [STREAMING_CHUNK: POSITION_SIZER]
# ============================================================================
def compute_position_size(
    f_star: float,
    current_equity: float,
    lambda_kelly: float = DEFAULT_LAMBDA_KELLY,
    max_notional_cap: float = None,
) -> float:
    """
    [PHÁT HIỆN O] Biến f* thành size_notional cho lệnh thật.

    Công thức: size_notional = f* × λ × current_equity

    Tham số:
    - f_star: Tỷ lệ cược tối ưu từ solve_empirical_kelly_fraction.
    - current_equity: Vốn tài khoản HIỆN TẠI (mark-to-market).
    - lambda_kelly: Hệ số chiết khấu Fractional Kelly (mặc định 0.5 = Half-Kelly).
    - max_notional_cap: Trần tuyệt đối cho size_notional (USD). None = không giới hạn.
    """
    # [ARMOR GUARD] Chặn input rác
    if math.isnan(f_star) or math.isinf(f_star) or f_star < 0:
        raise ValueError(f"f_star phải >= 0 và hợp lệ, nhận {f_star}")
    if math.isnan(current_equity) or math.isinf(current_equity) or current_equity <= 0:
        raise ValueError(f"current_equity phải > 0 và hợp lệ, nhận {current_equity}")
    if math.isnan(lambda_kelly) or math.isinf(lambda_kelly) or not (0.0 < lambda_kelly <= 1.0):
        raise ValueError(f"lambda_kelly phải nằm trong (0, 1], nhận {lambda_kelly}")

    # f_star = 0 → Không cược (kỳ vọng âm hoặc thiếu dữ liệu)
    if f_star == 0.0:
        return 0.0

    # Công thức lõi
    size_notional = f_star * lambda_kelly * current_equity

    # Trần tuyệt đối (nếu có)
    if max_notional_cap is not None:
        if math.isnan(max_notional_cap) or max_notional_cap <= 0:
            raise ValueError(f"max_notional_cap phải > 0, nhận {max_notional_cap}")
        size_notional = min(size_notional, max_notional_cap)

    return float(size_notional)


# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
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
    import pytest

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


if __name__ == "__main__":
    test_position_size_uses_current_equity()
    test_position_size_zero_f_star()
    test_position_size_max_cap()
    test_position_size_armor_guards()
