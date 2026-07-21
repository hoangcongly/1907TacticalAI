import math
from typing import Literal

# Import hàm xấp xỉ lỗ thanh lý từ tầng Liquidation Layer
from aegis.meta_labeling.sizing.liquidation_layer import compute_liquidation_loss

# ============================================================================
# [MODULE G] ĐỘNG CƠ TÍNH TOÁN LỢI NHUẬN THỰC TẾ (REALIZED PNL ENGINE)
# ============================================================================

# [STREAMING_CHUNK: PNL_ENGINE]
def compute_realized_pnl(
    entry_price: float, 
    exit_price: float, 
    side: int, 
    size_notional: float,
    leverage: float,
    exit_reason: Literal["SL", "TRAIL", "REGIME_FLIP", "TIME_STOP", "LIQUIDATION", "BOUNDARY_TRUNCATED"],
    fee_entry_rate: float = 0.0005,
    fee_exit_rate: float = 0.0005,
    funding_accrued_usd: float = 0.0,
    is_notional_in_usd: bool = True
) -> dict:
    """
    [MODULE G] Động cơ duy nhất tính toán PnL trên toàn hệ thống.
    Được dời về Module G (Execution) để tuân thủ kiến trúc Separation of Concerns.
    
    [QĐ #7] Xử lý phí THỐNG NHẤT cả 2 nhánh:
      Normal:  gross = Δprice × notional    → net = gross - fee_entry - fee_exit - funding
      Liq:     gross = -(margin)            → net = gross - fee_entry - funding
    """
    # [ARMOR GUARD] Kiểm tra input
    if side not in (1, -1):
        raise ValueError("side phải là 1 (Long) hoặc -1 (Short)")
    if size_notional <= 0 or entry_price <= 0 or exit_price <= 0:
        raise ValueError("Kích thước vị thế và giá trị phải lớn hơn 0")
    if math.isnan(entry_price) or math.isnan(exit_price):
        raise ValueError("Phát hiện giá rác NaN, dừng tính toán để bảo vệ PnL!")

    if not isinstance(fee_entry_rate, (int, float)) or math.isnan(fee_entry_rate) or math.isinf(fee_entry_rate) or fee_entry_rate < 0:
        raise ValueError(f"Lỗi hải quan: fee_entry_rate phải >= 0, nhận {fee_entry_rate}")
    if not isinstance(fee_exit_rate, (int, float)) or math.isnan(fee_exit_rate) or math.isinf(fee_exit_rate) or fee_exit_rate < 0:
        raise ValueError(f"Lỗi hải quan: fee_exit_rate phải >= 0, nhận {fee_exit_rate}")
    if not isinstance(funding_accrued_usd, (int, float)) or math.isnan(funding_accrued_usd) or math.isinf(funding_accrued_usd):
        raise ValueError(f"Lỗi hải quan: funding_accrued_usd không hợp lệ, nhận {funding_accrued_usd}")

    # Phí vào lệnh luôn tính (chung cho cả 2 nhánh)
    fee_entry_cost = size_notional * fee_entry_rate

    # ====================================================================
    # 1. Nhánh Tàn Khốc: THANH LÝ CƯỠNG CHẾ
    # ====================================================================
    if exit_reason == "LIQUIDATION":
        # [STREAMING_CHUNK: PNL_LIQUIDATION_BRANCH]
        # compute_liquidation_loss trả về -(margin) thuần (QĐ #7)
        gross_pnl = compute_liquidation_loss(size_notional, leverage)
        
        # Phí xử lý THỐNG NHẤT giống nhánh Normal (QĐ #7)
        net_pnl = gross_pnl - fee_entry_cost - funding_accrued_usd
        
        # Lợi suất (chưa đòn bẩy) dùng cho Kelly
        realized_return = net_pnl / size_notional
        
        return {
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "realized_return": float(realized_return),
            "fee_paid": float(fee_entry_cost)
        }

    # ====================================================================
    # 2. Nhánh Tiêu Chuẩn: CHỐT LỜI / CẮT LỖ THÔNG THƯỜNG
    # ====================================================================
    # [STREAMING_CHUNK: PNL_NORMAL_BRANCH]
    price_delta_pct = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
    
    # Lợi nhuận gộp (Gross PnL)
    if is_notional_in_usd:
        gross_pnl = price_delta_pct * size_notional
    else:
        # Nếu notional tính bằng Coin (Coin-M)
        gross_pnl = price_delta_pct * size_notional * exit_price 
        
    # Tính chi phí (Fees)
    fee_exit_cost = size_notional * fee_exit_rate if is_notional_in_usd else (size_notional * exit_price) * fee_exit_rate
    total_fee_cost = fee_entry_cost + fee_exit_cost
    
    # Lợi nhuận ròng (Net PnL)
    net_pnl = gross_pnl - total_fee_cost - funding_accrued_usd
    
    # Lợi suất cơ sở dùng cho Kelly (Return trên toàn bộ quy mô danh nghĩa, không tính đòn bẩy)
    realized_return = net_pnl / size_notional
    
    return {
        "gross_pnl": float(gross_pnl),
        "net_pnl": float(net_pnl),
        "realized_return": float(realized_return),
        "fee_paid": float(total_fee_cost)
    }


