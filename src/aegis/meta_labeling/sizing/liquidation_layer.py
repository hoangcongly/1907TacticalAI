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
) -> float:
    """
    [QĐ #7] Tính tổn thất vốn ký quỹ (Isolated Margin) khi bị thanh lý.
    Trả về: -(size_notional / leverage)
    
    Phí vào lệnh (fee_entry) được xử lý THỐNG NHẤT tại tầng pnl.py
    (giống nhánh Normal), KHÔNG trừ ở đây để tránh double-count.
    """
    if not isinstance(size_notional, (int, float)) or size_notional <= 0:
        raise ValueError(f"size_notional phải > 0, nhận {size_notional}")
    if not isinstance(leverage, (int, float)) or leverage < 1.0:
        raise ValueError(f"leverage phải >= 1.0, nhận {leverage}")
    if math.isnan(size_notional) or math.isnan(leverage):
        raise ValueError("Input chứa rác NaN")
    return float(-(size_notional / leverage))
