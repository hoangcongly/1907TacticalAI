"""
limit_queue_sim.py — L2 Orderbook Queue Estimation & 3-Tier Progressive Fallback Protocol (Task B-2-2).

Quy chuẩn B-2-2:
- Khắc phục Lỗ Hổng 2 (Phantom Liquidity Trap / Spoofing & Iceberg): Không tin tưởng tuyệt đối
  vào tổng khối lượng hiển thị trên L2. Áp dụng hệ số chiết khấu thanh khoản ảo
  (`spoofing_discount = 0.70`), giả định 30% tường lệnh L2 là ảo sẽ bị hủy trước khi giá quét tới.
- Khắc phục Lỗ Hổng 3 (Network Partition Blindspot & Over-sensitive Cutoff): Loại bỏ cơ chế
  nhị phân (chỉ cần trễ >500ms là đổ hết sang Market Order gây trượt giá và cắn phí Taker).
  Thay thế bằng Hệ thống Suy thoái Đa tầng (3-Tier Progressive Fallback Protocol):
  + Tier 0 (lag <= 500ms): LIMIT order tiêu chuẩn với Q_effective.
  + Tier 1 (500ms < lag <= 2000ms): POST_ONLY_LIMIT với buffer an toàn x1.5, tránh cắn phí Taker.
  + Tier 2 (lag > 2000ms hoặc feed hỏng): CANCEL_ALL_RESTING. Nếu lệnh thoát hiểm (EXIT/SL) -> FORCE_MARKET.
    Nếu lệnh mở mới (ENTRY) -> ABORT_ENTRY (tránh mở mới giữa tâm bão mạng nghẽn).
"""

import math
from typing import Optional, Union
import numpy as np


# ============================================================================
# [TASK B-2-2] ESTIMATE QUEUE AHEAD WITH PHANTOM LIQUIDITY DISCOUNT
# ============================================================================
def estimate_queue_ahead(
    order_book_snapshot: dict,
    limit_price: float,
    side: int,
    current_timestamp_ms: int,
    spoofing_discount: float = 0.70,
    bar_avg_volume: Optional[float] = None,
) -> dict:
    """
    Ước tính khối lượng xếp hàng phía trước (Queue-Ahead Position) trên sổ lệnh L2.

    [KHẮC PHỤC LỖ HỔNG 2 - PHANTOM LIQUIDITY TRAP]:
    - Tính toán khối lượng hiển thị resting visible tại các mức giá ưu tiên hơn hoặc bằng limit_price:
      + Với lệnh Mua (side = +1): các bids có price >= limit_price.
      + Với lệnh Bán (side = -1): các asks có price <= limit_price.
    - Chiết khấu thanh khoản ảo (Spoofing/Iceberg discount):
      Q_effective = Q_visible * spoofing_discount (mặc định 0.70 tức chiết khấu 30% lệnh ảo).
    """
    if not isinstance(order_book_snapshot, dict):
        raise ValueError("order_book_snapshot phải là dictionary.")
    if not isinstance(limit_price, (int, float)) or limit_price <= 0 or math.isnan(limit_price) or math.isinf(limit_price):
        raise ValueError(f"limit_price phải là số thực dương, nhận {limit_price}")
    if side not in (-1, 1):
        raise ValueError(f"side phải là +1 (Buy) hoặc -1 (Sell), nhận {side}")
    if not isinstance(current_timestamp_ms, int) or current_timestamp_ms < 0:
        raise ValueError(f"current_timestamp_ms phải là số nguyên >= 0, nhận {current_timestamp_ms}")
    if not isinstance(spoofing_discount, (int, float)) or not (0.0 < spoofing_discount <= 1.0) or math.isnan(spoofing_discount):
        raise ValueError(f"spoofing_discount phải thuộc (0, 1.0], nhận {spoofing_discount}")

    bids_data = order_book_snapshot.get("bids", {})
    asks_data = order_book_snapshot.get("asks", {})

    # Kiểm tra cấu trúc snapshot hợp lệ hay hỏng
    is_snapshot_valid = True
    if not bids_data and not asks_data:
        is_snapshot_valid = False

    # Helper chuyển đổi dict/list sang generator hoặc list tuples [(price, vol)]
    def parse_levels(levels_obj) -> list[tuple[float, float]]:
        result = []
        if isinstance(levels_obj, dict):
            for p, v in levels_obj.items():
                try:
                    result.append((float(p), float(v)))
                except (ValueError, TypeError):
                    pass
        elif isinstance(levels_obj, (list, tuple)):
            for item in levels_obj:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        result.append((float(item[0]), float(item[1])))
                    except (ValueError, TypeError):
                        pass
        return result

    bids_parsed = parse_levels(bids_data)
    asks_parsed = parse_levels(asks_data)

    if not bids_parsed and not asks_parsed:
        is_snapshot_valid = False

    queue_visible = 0.0
    if is_snapshot_valid:
        if side == 1:
            # Lệnh Mua Limit: xếp hàng sau các bids có giá cao hơn hoặc bằng limit_price
            for p, v in bids_parsed:
                if p >= limit_price and v > 0:
                    queue_visible += v
        else:
            # Lệnh Bán Limit: xếp hàng sau các asks có giá thấp hơn hoặc bằng limit_price
            for p, v in asks_parsed:
                if p <= limit_price and v > 0:
                    queue_visible += v

    # Chiết khấu thanh khoản ảo
    queue_effective = queue_visible * float(spoofing_discount)

    # Ước tính số bar tiêu hao hết hàng đợi
    if bar_avg_volume is not None and isinstance(bar_avg_volume, (int, float)) and bar_avg_volume > 0:
        estimated_depletion_bars = queue_effective / float(bar_avg_volume)
    else:
        estimated_depletion_bars = 0.0

    # Tính tuổi snapshot
    snapshot_time_ms = order_book_snapshot.get("timestamp_ms", current_timestamp_ms)
    try:
        snapshot_time_int = int(snapshot_time_ms)
    except (ValueError, TypeError):
        snapshot_time_int = 0
        is_snapshot_valid = False

    snapshot_age_ms = max(0, int(current_timestamp_ms) - snapshot_time_int)

    return {
        "queue_visible": float(queue_visible),
        "queue_effective": float(queue_effective),
        "estimated_depletion_bars": float(estimated_depletion_bars),
        "snapshot_age_ms": int(snapshot_age_ms),
        "is_snapshot_valid": bool(is_snapshot_valid),
    }


# ============================================================================
# [TASK B-2-2] 3-TIER PROGRESSIVE FALLBACK PROTOCOL
# ============================================================================
def resolve_execution_mode_with_l2_fallback(
    intended_mode: str,
    l2_snapshot: dict,
    limit_price: float,
    side: int,
    current_timestamp_ms: int,
    trade_intent: str = "ENTRY",
    tier1_threshold_ms: int = 500,
    tier2_threshold_ms: int = 2000,
    spoofing_discount: float = 0.70,
    bar_avg_volume: Optional[float] = None,
) -> tuple[str, dict]:
    """
    Hệ thống Suy thoái Đa tầng (3-Tier Progressive Fallback Protocol).

    [KHẮC PHỤC LỖ HỔNG 3 - MULTI-TIER PROGRESSIVE DEGRADATION vs BINARY PANIC]:
    Khi intended_mode == "LIMIT":
    - Tier 0 (lag <= 500ms): Feed khỏe mạnh. Trả về ("LIMIT", queue_metrics).
    - Tier 1 (500ms < lag <= 2000ms): Mild Jitter / Rate Limit lag nhẹ.
      KHÔNG CHUYỂN SANG MARKET ORDER! Chuyển sang ("POST_ONLY_LIMIT", buffer x1.5)
      với Time-To-Live ngắn, vừa bảo vệ khỏi Adverse Selection vừa giữ rebate Maker.
    - Tier 2 (lag > 2000ms hoặc L2 hỏng): Critical Staleness / Disconnect.
      Phát tín hiệu hủy lệnh treo (CANCEL_ALL_RESTING_LIMITS).
      Phân loại theo Urgency Gate phân rã hạt mịn (Fine-Grained Urgency Gate):
      + EMERGENCY EXITS ("SL", "LIQUIDATION_PREVENTION", "EMERGENCY_EXIT", "STOP_LOSS"): -> ("FORCE_MARKET", ...)
        Thoát lệnh cắt lỗ/chống thanh lý khẩn cấp nhằm cứu tài khoản bằng mọi giá dù chịu phí Taker.
      + NON-EMERGENCY EXITS ("EXIT", "TRAIL_PROFIT", "REGIME_FLIP", "TAKE_PROFIT", "NORMAL_EXIT"): -> ("POSTPONE_NON_EMERGENCY_EXIT", ...)
        Thoát lệnh chốt lời hoặc lật pha thông thường không khẩn cấp. Trì hoãn lệnh/chuyển thụ động
        để tuyệt đối tránh cắn phí Taker đắt đỏ giữa lúc mạng nghẽn hoặc sổ lệnh hỏng.
      + ENTRY INTENTS ("ENTRY", "FOLLOW", "FADE", "OPEN"): -> ("ABORT_ENTRY", ...)
        Từ bỏ mở lệnh mới giữa tâm bão mạng đứt gãy để giữ an toàn vốn.
    """
    if not isinstance(intended_mode, str):
        raise ValueError("intended_mode phải là string.")
    if not isinstance(trade_intent, str):
        raise ValueError("trade_intent phải là string.")
    if not isinstance(tier1_threshold_ms, int) or not isinstance(tier2_threshold_ms, int) or tier1_threshold_ms < 0 or tier2_threshold_ms <= tier1_threshold_ms:
        raise ValueError(f"tier1 và tier2 thresholds không hợp lệ: ({tier1_threshold_ms}, {tier2_threshold_ms})")

    queue_info = estimate_queue_ahead(
        order_book_snapshot=l2_snapshot,
        limit_price=limit_price,
        side=side,
        current_timestamp_ms=current_timestamp_ms,
        spoofing_discount=spoofing_discount,
        bar_avg_volume=bar_avg_volume,
    )

    if intended_mode.upper() != "LIMIT":
        queue_info["fallback_triggered"] = False
        queue_info["fallback_tier"] = 0
        queue_info["action"] = "PASS_THROUGH_NON_LIMIT"
        return intended_mode.upper(), queue_info

    snapshot_age_ms = queue_info["snapshot_age_ms"]
    is_valid = queue_info["is_snapshot_valid"]
    intent_upper = trade_intent.upper()

    # Tier 2 Check: Critical staleness hoặc snapshot hỏng
    if not is_valid or snapshot_age_ms > tier2_threshold_ms:
        queue_info["fallback_triggered"] = True
        queue_info["fallback_tier"] = 2
        queue_info["action"] = "CANCEL_ALL_RESTING_LIMITS"
        queue_info["ttl_seconds"] = 0
        queue_info["backoff_multiplier"] = 1.0
        queue_info["max_retry_attempts"] = 0
        queue_info["next_retry_ms"] = 0

        if intent_upper in ("SL", "LIQUIDATION_PREVENTION", "EMERGENCY_EXIT", "STOP_LOSS", "LIQUIDATION"):
            return "FORCE_MARKET", queue_info
        elif intent_upper in ("EXIT", "TRAIL_PROFIT", "REGIME_FLIP", "TAKE_PROFIT", "NORMAL_EXIT"):
            return "POSTPONE_NON_EMERGENCY_EXIT", queue_info
        else:
            return "ABORT_ENTRY", queue_info

    # Tier 1 Check: Mild Jitter
    if snapshot_age_ms > tier1_threshold_ms:
        queue_info["fallback_triggered"] = True
        queue_info["fallback_tier"] = 1
        queue_info["action"] = "POST_ONLY_LIMIT_WITH_BUFFER"
        # Tăng buffer an toàn cho effective queue thêm 50%
        queue_info["queue_effective"] *= 1.5
        if bar_avg_volume is not None and bar_avg_volume > 0:
            queue_info["estimated_depletion_bars"] = queue_info["queue_effective"] / float(bar_avg_volume)
        # [KHẮC PHỤC LỖ HỔNG 4 - API RATE LIMIT & OTR BAN]:
        # Thiết lập TTL = 15s (10s-30s) dài và cơ chế Exponential Backoff tránh spam lệnh gây OTR Ban
        queue_info["ttl_seconds"] = 15
        queue_info["backoff_multiplier"] = 2.0
        queue_info["max_retry_attempts"] = 3
        queue_info["next_retry_ms"] = 2000
        return "POST_ONLY_LIMIT", queue_info

    # Tier 0 Check: Healthy Feed
    queue_info["fallback_triggered"] = False
    queue_info["fallback_tier"] = 0
    queue_info["action"] = "STANDARD_LIMIT"
    queue_info["ttl_seconds"] = 60
    queue_info["backoff_multiplier"] = 1.0
    queue_info["max_retry_attempts"] = 0
    queue_info["next_retry_ms"] = 0
    return "LIMIT", queue_info


# ============================================================================
# [TASK B-2-2] SIMULATE LIMIT FILL WITH QUEUE AHEAD
# ============================================================================
def simulate_limit_fill_with_queue(
    queue_effective: float,
    order_size: float,
    subsequent_volumes: Union[list[float], np.ndarray],
    timeout_bars: int = 5,
) -> dict:
    """
    Mô phỏng khả năng khớp lệnh giới hạn qua nhiều bar kề tiếp theo.
    Lệnh khớp nếu tổng volume giao dịch thị trường kể từ lúc đặt vượt qua
    lượng xếp hàng phía trước (queue_effective) + kích thước lệnh (order_size).
    """
    if not isinstance(queue_effective, (int, float)) or queue_effective < 0 or math.isnan(queue_effective):
        raise ValueError(f"queue_effective không hợp lệ: {queue_effective}")
    if not isinstance(order_size, (int, float)) or order_size <= 0 or math.isnan(order_size):
        raise ValueError(f"order_size không hợp lệ: {order_size}")
    if not isinstance(timeout_bars, int) or timeout_bars <= 0:
        raise ValueError(f"timeout_bars phải là số nguyên dương: {timeout_bars}")

    vols_np = np.asarray(subsequent_volumes, dtype=np.float64)
    if vols_np.ndim != 1 or np.any(vols_np < 0) or np.any(np.isnan(vols_np)):
        raise ValueError("subsequent_volumes phải là mảng 1D số thực >= 0 không chứa NaN.")

    required_volume = float(queue_effective) + float(order_size)
    cumulative_volume = 0.0
    filled = False
    fill_bar_idx = -1

    for idx in range(min(timeout_bars, len(vols_np))):
        cumulative_volume += float(vols_np[idx])
        if cumulative_volume >= required_volume:
            filled = True
            fill_bar_idx = idx + 1  # 1-indexed count of bars elapsed
            break

    return {
        "filled": bool(filled),
        "fill_bar_idx": int(fill_bar_idx),
        "cumulative_volume_processed": float(cumulative_volume),
        "required_volume": float(required_volume),
        "timeout_reached": bool(not filled and len(vols_np) >= timeout_bars),
    }
