import math
from typing import Literal

# Import hàm xấp xỉ lỗ thanh lý từ tầng Liquidation Layer
from aegis.meta_labeling.sizing.liquidation_layer import compute_liquidation_loss

# ============================================================================
# [MODULE G] ĐỘNG CƠ TÍNH TOÁN LỢI NHUẬN THỰC TẾ (REALIZED PNL ENGINE)
# ============================================================================

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
    [VÁ LỖ HỔNG 2]: Động cơ duy nhất tính toán PnL trên toàn hệ thống.
    Được dời về Module G (Execution) để tuân thủ kiến trúc Separation of Concerns.
    """
    if side not in (1, -1):
        raise ValueError("side phải là 1 (Long) hoặc -1 (Short)")
    if size_notional <= 0 or entry_price <= 0 or exit_price <= 0:
        raise ValueError("Kích thước vị thế và giá trị phải lớn hơn 0")
    if math.isnan(entry_price) or math.isnan(exit_price):
        raise ValueError("Phát hiện giá rác NaN, dừng tính toán để bảo vệ PnL!")

    # 1. Nhánh Tàn Khốc: THANH LÝ CƯỠNG CHẾ
    if exit_reason == "LIQUIDATION":
        # Ủy quyền cho Liquidation Layer tính toán thiệt hại (Cọc + Phí chìm)
        loss_liq_gross = compute_liquidation_loss(size_notional, leverage, fee_entry_rate)
        
        # Thêm chi phí Funding
        loss_liq_net = loss_liq_gross - funding_accrued_usd
        
        # Lợi suất (chưa đòn bẩy) dùng cho Kelly
        # Lỗ 100% Margin -> Lợi suất unleveraged = -1.0 / leverage
        realized_return = -1.0 / leverage
        
        return {
            "gross_pnl": float(loss_liq_gross),
            "net_pnl": float(loss_liq_net),
            "realized_return": float(realized_return),
            "fee_paid": size_notional * fee_entry_rate
        }

    # 2. Nhánh Tiêu Chuẩn: CHỐT LỜI / CẮT LỖ THÔNG THƯỜNG
    price_delta_pct = (exit_price - entry_price) / entry_price if side > 0 else (entry_price - exit_price) / entry_price
    
    # Lợi nhuận gộp (Gross PnL)
    if is_notional_in_usd:
        gross_pnl = price_delta_pct * size_notional
    else:
        # Nếu notional tính bằng Coin (Coin-M)
        gross_pnl = price_delta_pct * size_notional * exit_price 
        
    # Tính chi phí (Fees)
    fee_entry_cost = size_notional * fee_entry_rate
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
        "fee_paid": total_fee_cost
    }

# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
def test_pnl_normal_win():
    res = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 10.0, "TRAIL")
    # Gross = 10% của 1000 = +100 USD. Fees = 2 * (1000 * 0.0005) = 1 USD. Net = +99 USD.
    assert abs(res["net_pnl"] - 99.0) < 1e-6
    # Realized return (Unleveraged) = 99 / 1000 = 0.099 (9.9%)
    assert abs(res["realized_return"] - 0.099) < 1e-6
    print("✅ [PNL ENGINE] Lệnh thắng thông thường PASSED!")

def test_pnl_liquidation_branch():
    res = compute_realized_pnl(100.0, 90.0, 1, 1000.0, 10.0, "LIQUIDATION")
    # Margin = 1000/10 = 100. Sunk fee = 1000*0.0005 = 0.5. Net = -100.5 USD.
    assert abs(res["net_pnl"] - (-100.5)) < 1e-6
    # Realized return = -100% Margin -> Unleveraged = -10% = -0.1
    assert abs(res["realized_return"] - (-0.1)) < 1e-6
    print("✅ [LỖ HỔNG 2 VÁ THÀNH CÔNG] Động cơ PnL đã được di dời sang Module G và định tuyến thanh lý chuẩn xác!")

if __name__ == "__main__":
    test_pnl_normal_win()
    test_pnl_liquidation_branch()
