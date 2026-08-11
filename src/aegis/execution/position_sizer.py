"""
[PHÁT HIỆN O] Module G — Cầu Nối Kelly → Lệnh Thật.
Hàm compute_position_size biến f* (tỷ lệ trừu tượng) thành size_notional thật
bằng công thức: size_notional = f* × λ × current_equity.

PHẢI sử dụng current_equity (mark-to-market, cập nhật mỗi lệnh),
KHÔNG PHẢI vốn gốc cố định — để phát huy lợi thế compounding.
"""

import math
from typing import Optional

from aegis.meta_labeling.sizing.kelly_empirical import DEFAULT_LAMBDA_KELLY


# ============================================================================
# [KHẮC PHỤC LỖ HỔNG 5 - LOT SIZE PRECISION]
# ============================================================================
def round_notional_down(size_notional: float, lot_step_size: float) -> float:
    """
    Làm tròn xuống (round down / floor) quy mô danh nghĩa theo bước nhảy `lot_step_size` của sàn.
    Đảm bảo không bao giờ vượt quá vốn/đòn bẩy an toàn do làm tròn lên.
    """
    if not isinstance(lot_step_size, (int, float)) or lot_step_size <= 0 or math.isnan(lot_step_size) or math.isinf(lot_step_size):
        raise ValueError(f"lot_step_size phải > 0 hợp lệ, nhận {lot_step_size}")
    if not isinstance(size_notional, (int, float)) or size_notional <= 0 or math.isnan(size_notional) or math.isinf(size_notional):
        if size_notional == 0.0:
            return 0.0
        raise ValueError(f"size_notional không hợp lệ: {size_notional}")
    steps = math.floor((size_notional + 1e-12) / lot_step_size)
    return float(max(0.0, steps * lot_step_size))


# ============================================================================
# [KHẮC PHỤC LỖ HỔNG 10 & BẪY 3 - ACCOUNT STATE TRACKER]
# ============================================================================
class AccountStateTracker:
    """
    Phân định minh bạch 3 tầng số dư ví tài khoản Isolated Margin:
    - wallet_balance: Số dư ví thực tế đã chốt (Realized Cash Balance).
    - unrealized_pnl: Lãi/lỗ chưa chốt (uPnL) của các lệnh đang mở.
    - used_initial_margin: Tổng ký quỹ ban đầu đã cọc cho các lệnh đang mở.
    
    Thuộc tính tính toán:
    - margin_balance: wallet_balance + unrealized_pnl (Mark-to-Market Equity).
    - available_margin: max(0.0, wallet_balance - used_initial_margin) (Số dư tiền mặt khả dụng để cọc lệnh mới).
    """
    def __init__(self, wallet_balance: float, unrealized_pnl: float = 0.0, used_initial_margin: float = 0.0):
        if not isinstance(wallet_balance, (int, float)) or math.isnan(wallet_balance) or math.isinf(wallet_balance) or wallet_balance < 0:
            raise ValueError(f"wallet_balance phải >= 0 hợp lệ, nhận {wallet_balance}")
        if not isinstance(unrealized_pnl, (int, float)) or math.isnan(unrealized_pnl) or math.isinf(unrealized_pnl):
            raise ValueError(f"unrealized_pnl không hợp lệ, nhận {unrealized_pnl}")
        if not isinstance(used_initial_margin, (int, float)) or math.isnan(used_initial_margin) or math.isinf(used_initial_margin) or used_initial_margin < 0:
            raise ValueError(f"used_initial_margin phải >= 0 hợp lệ, nhận {used_initial_margin}")
            
        self.wallet_balance = float(wallet_balance)
        self.unrealized_pnl = float(unrealized_pnl)
        self.used_initial_margin = float(used_initial_margin)

    @property
    def margin_balance(self) -> float:
        """Tổng giá trị tài sản ròng Mark-to-Market (wallet_balance + uPnL)."""
        return float(self.wallet_balance + self.unrealized_pnl)

    @property
    def available_margin(self) -> float:
        """
        Số dư tiền mặt khả dụng để đặt cọc mở lệnh mới trong chế độ Isolated Margin.
        TUYỆT ĐỐI KHÔNG cộng uPnL chưa chốt vào available_margin.
        """
        return float(max(0.0, self.wallet_balance - self.used_initial_margin))

    def get_effective_equity_for_sizing(self, allow_upnl_compounding: bool = False) -> float:
        """
        Trả về nguồn vốn hiệu dụng dùng cho tính toán size_notional:
        - Nếu allow_upnl_compounding = False (chuẩn Isolated Margin an toàn): chỉ dùng available_margin.
        - Tuyệt đối bảo vệ không cọc quá available_margin.
        """
        if allow_upnl_compounding:
            return float(max(0.0, self.margin_balance - self.used_initial_margin))
        return float(self.available_margin)


# ============================================================================
# [STREAMING_CHUNK: POSITION_SIZER]
# ============================================================================
def compute_position_size(
    f_star: float,
    current_equity: float | AccountStateTracker,
    lambda_kelly: float = DEFAULT_LAMBDA_KELLY,
    max_notional_cap: Optional[float] = None,
    atr_hist_mean_pct: Optional[float] = None,
    atr_current_pct: Optional[float] = None,
    lot_step_size: Optional[float] = None,
    min_vol_multiplier: float = 0.2,
    max_vol_multiplier: float = 2.5,
    max_safe_leverage: float = 20.0,  # [KHẮC PHỤC LỖ HỔNG #2] Đổi thành tham số float bắt buộc
) -> float:
    """
    [PHÁT HIỆN O + KHẮC PHỤC LỖ HỔNG 7 & BẪY 1] Biến f* thành size_notional cho lệnh thật.

    Công thức: size_notional = f* × λ × effective_equity × clamp(ATR_hist / ATR_current, min_vol, max_vol)

    Tham số:
    - f_star: Tỷ lệ cược tối ưu từ solve_empirical_kelly_fraction.
    - current_equity: Vốn tài khoản HIỆN TẠI (hoặc instance AccountStateTracker).
    - lambda_kelly: Hệ số chiết khấu Fractional Kelly (mặc định 0.5 = Half-Kelly).
    - max_notional_cap: Trần tuyệt đối cho size_notional (USD). None = không giới hạn.
    - atr_hist_mean_pct: ATR trung bình (tính bằng %) của mẫu quá khứ Kelly.
    - atr_current_pct: ATR hiện tại (tính bằng %).
    - min_vol_multiplier: Kẹp dưới cho hệ số biến động (mặc định 0.2).
    - max_vol_multiplier: Kẹp trên cho hệ số biến động (mặc định 2.5 - chống Cash Drag).
    - max_safe_leverage: Rào chắn L_max kiểm duyệt cuối cùng (Mandatory Safety Gate).
    """
    # [ARMOR GUARD] Chặn input rác
    if math.isnan(f_star) or math.isinf(f_star) or f_star < 0:
        raise ValueError(f"f_star phải >= 0 và hợp lệ, nhận {f_star}")

    if isinstance(current_equity, AccountStateTracker):
        effective_equity = current_equity.get_effective_equity_for_sizing()
    else:
        if not isinstance(current_equity, (int, float)) or math.isnan(current_equity) or math.isinf(current_equity) or current_equity <= 0:
            raise ValueError(f"current_equity phải > 0 và hợp lệ, nhận {current_equity}")
        effective_equity = float(current_equity)

    if math.isnan(lambda_kelly) or math.isinf(lambda_kelly) or not (0.0 < lambda_kelly <= 1.0):
        raise ValueError(f"lambda_kelly phải nằm trong (0, 1], nhận {lambda_kelly}")

    if not isinstance(min_vol_multiplier, (int, float)) or math.isnan(min_vol_multiplier) or min_vol_multiplier < 0:
        raise ValueError(f"min_vol_multiplier không hợp lệ: {min_vol_multiplier}")
    if not isinstance(max_vol_multiplier, (int, float)) or math.isnan(max_vol_multiplier) or max_vol_multiplier < min_vol_multiplier:
        raise ValueError(f"max_vol_multiplier không hợp lệ: {max_vol_multiplier}")

    # f_star = 0 → Không cược (kỳ vọng âm hoặc thiếu dữ liệu)
    if f_star == 0.0:
        return 0.0
        
    # [TẦNG 3]: VOLATILITY TARGETING WITH CLAMP AND SAFETY GATE
    vol_multiplier = 1.0
    if atr_hist_mean_pct is not None and atr_current_pct is not None:
        if atr_current_pct <= 0 or math.isnan(atr_current_pct) or math.isinf(atr_current_pct):
            raise ValueError("ATR hiện tại rác (<=0, NaN hoặc Inf), dừng cấp vốn!")
        if atr_hist_mean_pct <= 0 or math.isnan(atr_hist_mean_pct) or math.isinf(atr_hist_mean_pct):
            raise ValueError("ATR lịch sử rác (<=0, NaN hoặc Inf), không có cơ sở tham chiếu!")
            
        # Tính Tỷ lệ Bóp nghẹt/Mở rộng (Volatility Scaling Ratio)
        vol_ratio = atr_hist_mean_pct / atr_current_pct
        
        # Mở kẹp clamp(vol_ratio, min_vol_multiplier, max_vol_multiplier)
        vol_multiplier = max(min_vol_multiplier, min(max_vol_multiplier, vol_ratio))

    # Công thức lõi
    size_notional = f_star * lambda_kelly * effective_equity * vol_multiplier

    # Kiểm duyệt cuối cùng qua rào chắn L_max (Mandatory Safety Gate)
    if max_safe_leverage is not None:
        if not isinstance(max_safe_leverage, (int, float)) or math.isnan(max_safe_leverage) or math.isinf(max_safe_leverage) or max_safe_leverage <= 0:
            raise ValueError(f"max_safe_leverage phải > 0 và hợp lệ, nhận {max_safe_leverage}")
        size_notional = min(size_notional, max_safe_leverage * effective_equity)

    # Trần tuyệt đối (nếu có)
    if max_notional_cap is not None:
        if math.isnan(max_notional_cap) or math.isinf(max_notional_cap) or max_notional_cap <= 0:
            raise ValueError(f"max_notional_cap phải > 0 và hợp lệ, nhận {max_notional_cap}")
        size_notional = min(size_notional, max_notional_cap)

    if lot_step_size is not None and lot_step_size > 0:
        size_notional = round_notional_down(size_notional, lot_step_size)

    return float(size_notional)



