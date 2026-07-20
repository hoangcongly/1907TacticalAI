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
    atr_hist_mean_pct: float = None,
    atr_current_pct: float = None,
) -> float:
    """
    [PHÁT HIỆN O] Biến f* thành size_notional cho lệnh thật.

    Công thức: size_notional = f* × λ × current_equity × min(1.0, ATR_hist / ATR_current)

    Tham số:
    - f_star: Tỷ lệ cược tối ưu từ solve_empirical_kelly_fraction.
    - current_equity: Vốn tài khoản HIỆN TẠI (mark-to-market).
    - lambda_kelly: Hệ số chiết khấu Fractional Kelly (mặc định 0.5 = Half-Kelly).
    - max_notional_cap: Trần tuyệt đối cho size_notional (USD). None = không giới hạn.
    - atr_hist_mean_pct: ATR trung bình (tính bằng %) của mẫu quá khứ Kelly.
    - atr_current_pct: ATR hiện tại (tính bằng %). Nếu cao hơn quá khứ, size sẽ bị cắt giảm.
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
        
    # [TẦNG 3]: VOLATILITY TARGETING (BÓP NGHẸT THIÊN NGA ĐEN)
    vol_multiplier = 1.0
    if atr_hist_mean_pct is not None and atr_current_pct is not None:
        if atr_current_pct <= 0 or math.isnan(atr_current_pct):
            raise ValueError("ATR hiện tại rác (<=0 hoặc NaN), dừng cấp vốn!")
        if atr_hist_mean_pct <= 0 or math.isnan(atr_hist_mean_pct):
            raise ValueError("ATR lịch sử rác, không có cơ sở tham chiếu!")
            
        # Tính Tỷ lệ Bóp nghẹt (Volatility Scaling Ratio)
        vol_ratio = atr_hist_mean_pct / atr_current_pct
        
        # Kẹp max = 1.0 (Chỉ được giảm size khi bão tới, cấm tăng size khi thị trường quá phẳng lặng)
        vol_multiplier = min(1.0, vol_ratio)

    # Công thức lõi
    size_notional = f_star * lambda_kelly * current_equity * vol_multiplier

    # Trần tuyệt đối (nếu có)
    if max_notional_cap is not None:
        if math.isnan(max_notional_cap) or max_notional_cap <= 0:
            raise ValueError(f"max_notional_cap phải > 0, nhận {max_notional_cap}")
        size_notional = min(size_notional, max_notional_cap)

    return float(size_notional)



