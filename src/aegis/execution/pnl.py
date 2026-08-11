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
    entry_fill_type: Literal["maker", "taker"],
    exit_fill_type: Literal["maker", "taker"],
    maker_fee_rate: float = 0.0001,
    taker_fee_rate: float = 0.0004,
    funding_accrued_usd: float = 0.0,
    is_notional_in_usd: bool = True
) -> dict:
    """
    [MODULE G] Động cơ duy nhất tính toán PnL trên toàn hệ thống.
    Được dời về Module G (Execution) để tuân thủ kiến trúc Separation of Concerns.
    
    [QĐ #7] Xử lý phí THỐNG NHẤT cả 2 nhánh:
      Normal:  gross = Δprice × notional    → net = gross - fee_entry - fee_exit - funding
      Liq:     gross = -(margin)            → net = gross - fee_entry (chống đếm kép funding theo QĐ #7 & Issue #4)
    """
    # [ARMOR GUARD] Kiểm tra input
    if side not in (1, -1):
        raise ValueError("side phải là 1 (Long) hoặc -1 (Short)")
    if size_notional <= 0 or entry_price <= 0 or exit_price <= 0:
        raise ValueError("Kích thước vị thế và giá trị phải lớn hơn 0")
    if math.isnan(entry_price) or math.isnan(exit_price):
        raise ValueError("Phát hiện giá rác NaN, dừng tính toán để bảo vệ PnL!")

    if entry_fill_type not in ("maker", "taker"):
        raise ValueError(f"entry_fill_type phải là 'maker' hoặc 'taker', nhận {entry_fill_type}")
    if exit_fill_type not in ("maker", "taker"):
        raise ValueError(f"exit_fill_type phải là 'maker' hoặc 'taker', nhận {exit_fill_type}")

    if not isinstance(maker_fee_rate, (int, float)) or math.isnan(maker_fee_rate) or math.isinf(maker_fee_rate) or maker_fee_rate < 0:
        raise ValueError(f"Lỗi hải quan: maker_fee_rate phải >= 0, nhận {maker_fee_rate}")
    if not isinstance(taker_fee_rate, (int, float)) or math.isnan(taker_fee_rate) or math.isinf(taker_fee_rate) or taker_fee_rate < 0:
        raise ValueError(f"Lỗi hải quan: taker_fee_rate phải >= 0, nhận {taker_fee_rate}")
    if not isinstance(funding_accrued_usd, (int, float)) or math.isnan(funding_accrued_usd) or math.isinf(funding_accrued_usd):
        raise ValueError(f"Lỗi hải quan: funding_accrued_usd không hợp lệ, nhận {funding_accrued_usd}")

    # Resolve fee rates based on fill types
    fee_entry_rate = maker_fee_rate if entry_fill_type == "maker" else taker_fee_rate
    fee_exit_rate = maker_fee_rate if exit_fill_type == "maker" else taker_fee_rate

    # Phí vào lệnh luôn tính (chung cho cả 2 nhánh)
    fee_entry_cost = size_notional * fee_entry_rate

    # ====================================================================
    # 1. Nhánh Tàn Khốc: THANH LÝ CƯỠNG CHẾ
    # ====================================================================
    if exit_reason == "LIQUIDATION":
        # [STREAMING_CHUNK: PNL_LIQUIDATION_BRANCH]
        # compute_liquidation_loss trả về -(margin) thuần (QĐ #7).
        gross_pnl = compute_liquidation_loss(size_notional, leverage)
        
        # [KHẮC PHỤC LỖ HỔNG #3]: TUYỆT ĐỐI KHÔNG KHẤU TRỪ THÊM FUNDING FEE!
        # Do funding fee đã bòn rút Initial Margin từ trước, nó là nguyên nhân 
        # đẩy giá thanh lý (P_liq) lại gần Entry hơn. Khoản lỗ tối đa chính bằng lượng 
        # margin thực tế mất đi. Trừ thêm lần nữa là lỗi ĐẾM KÉP (Double-Count).
        net_pnl = gross_pnl - fee_entry_cost
        
        # [KHẮC PHỤC LỖ HỔNG #1]: LỢI SUẤT CHƯA ĐÒN BẨY DÙNG CHO KELLY
        # KHÔNG ĐƯỢC chia net_pnl / size_notional (tạo ra ảo giác đòn bẩy cao rủi ro thấp).
        # Phải dùng biến động giá cơ sở tới ngưỡng thanh lý: r_u = (P_liq - P_entry) / P_entry * side
        # Ngầm định exit_price được truyền vào đây chính là liquidation_price.
        price_delta_pct = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
        realized_return = price_delta_pct
        
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
        
    # Tính chi phí thoát lệnh trên Exit Notional thực tế (Vá BỌ SỐ 1: Exit Fee Accounting Flaw)
    if is_notional_in_usd:
        exit_notional = max(0.0, size_notional + gross_pnl)
        fee_exit_cost = exit_notional * fee_exit_rate
    else:
        # Nếu notional tính bằng Coin (Coin-M), giá trị USD tại thời điểm thoát là size_notional * exit_price
        fee_exit_cost = (size_notional * exit_price) * fee_exit_rate
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


