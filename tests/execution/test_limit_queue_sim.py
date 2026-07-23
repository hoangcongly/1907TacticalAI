"""
Bộ kiểm thử unit test cho Task B-2-2: L2 Queue Estimation & 3-Tier Progressive Fallback Protocol.
"""

import pytest
import numpy as np
from aegis.execution.limit_queue_sim import (
    estimate_queue_ahead,
    resolve_execution_mode_with_l2_fallback,
    simulate_limit_fill_with_queue,
)


def test_b_2_2_estimate_queue_ahead_with_spoofing_discount():
    """
    [TDD VERIFICATION - KHẮC PHỤC LỖ HỔNG 2: PHANTOM LIQUIDITY TRAP]:
    Kiểm chứng hệ thống tính toán chính xác tổng volume hiển thị tại price >= limit_price (cho Buy Limit)
    và áp dụng hệ số chiết khấu thanh khoản ảo (mặc định 0.70 tức chiết khấu 30%).
    """
    # Snapshot tại thời điểm t = 10_000 ms
    snapshot = {
        "timestamp_ms": 10_000,
        "bids": [
            [100.5, 10.0],
            [100.4, 25.0],
            [100.3, 50.0],
            [100.2, 100.0],
        ],
        "asks": [
            [100.6, 15.0],
            [100.7, 30.0],
        ],
    }

    # Đặt Buy Limit (side = +1) tại mức giá limit_price = 100.4
    # Các bids nằm trước hoặc tại mức giá này là: 100.5 (vol 10.0) và 100.4 (vol 25.0)
    # Tổng queue_visible = 10.0 + 25.0 = 35.0
    # Với spoofing_discount = 0.70 => queue_effective = 35.0 * 0.70 = 24.5
    res = estimate_queue_ahead(
        order_book_snapshot=snapshot,
        limit_price=100.4,
        side=1,
        current_timestamp_ms=10_100,  # trễ 100ms
        spoofing_discount=0.70,
        bar_avg_volume=10.0,
    )

    assert res["is_snapshot_valid"] is True
    assert res["snapshot_age_ms"] == 100
    assert res["queue_visible"] == pytest.approx(35.0)
    assert res["queue_effective"] == pytest.approx(24.5)
    assert res["estimated_depletion_bars"] == pytest.approx(2.45)


def test_b_2_2_progressive_fallback_tier1_mild_jitter():
    """
    [TDD VERIFICATION - KHẮC PHỤC LỖ HỔNG 3: PROGRESSIVE FALLBACK TIER 1]:
    Khi mạng chỉ trễ nhẹ (ví dụ 1200ms nằm trong khoảng (500ms, 2000ms]),
    hệ thống KHÔNG BỊ HOẢNG LOẠN đập thẳng sang Market Order gây trượt giá,
    mà chuyển sang POST_ONLY_LIMIT và tăng biên độ an toàn Q_effective x1.5.
    """
    snapshot = {
        "timestamp_ms": 10_000,
        "bids": {"100.0": 20.0},
        "asks": {"100.5": 10.0},
    }

    mode, info = resolve_execution_mode_with_l2_fallback(
        intended_mode="LIMIT",
        l2_snapshot=snapshot,
        limit_price=100.0,
        side=1,
        current_timestamp_ms=11_200,  # trễ 1200ms -> thuộc Tier 1
        trade_intent="ENTRY",
        tier1_threshold_ms=500,
        tier2_threshold_ms=2000,
        spoofing_discount=0.70,
    )

    assert mode == "POST_ONLY_LIMIT"
    assert info["fallback_triggered"] is True
    assert info["fallback_tier"] == 1
    assert info["action"] == "POST_ONLY_LIMIT_WITH_BUFFER"
    # queue_visible = 20.0, spoofing=0.7 => 14.0, buffer Tier 1 x1.5 => 21.0
    assert info["queue_effective"] == pytest.approx(21.0)


def test_b_2_2_progressive_fallback_tier2_entry_vs_exit():
    """
    [TDD VERIFICATION - KHẮC PHỤC LỖ HỔNG 3: PROGRESSIVE FALLBACK TIER 2]:
    Khi độ trễ > 2000ms (ví dụ 3500ms) hoặc L2 hỏng, hệ thống kích hoạt Tier 2 (CANCEL_ALL_RESTING):
    - Nếu là lệnh mở mới (ENTRY) -> ABORT_ENTRY (không tốn phí Taker trong lúc bão mạng).
    - Nếu là lệnh cứu tài khoản (EXIT / SL) -> FORCE_MARKET (khớp bằng mọi giá để bảo toàn vốn).
    """
    snapshot = {
        "timestamp_ms": 10_000,
        "bids": {"100.0": 20.0},
        "asks": {"100.5": 10.0},
    }

    # Trường hợp 1: Lệnh mở mới ENTRY khi trễ 3500ms
    mode_entry, info_entry = resolve_execution_mode_with_l2_fallback(
        intended_mode="LIMIT",
        l2_snapshot=snapshot,
        limit_price=100.0,
        side=1,
        current_timestamp_ms=13_500,  # trễ 3500ms -> Tier 2
        trade_intent="ENTRY",
    )
    assert mode_entry == "ABORT_ENTRY"
    assert info_entry["fallback_tier"] == 2
    assert info_entry["action"] == "CANCEL_ALL_RESTING_LIMITS"

    # Trường hợp 2: Lệnh cắt lỗ khẩn cấp SL khi trễ 3500ms -> FORCE_MARKET
    mode_sl, info_sl = resolve_execution_mode_with_l2_fallback(
        intended_mode="LIMIT",
        l2_snapshot=snapshot,
        limit_price=100.0,
        side=-1,
        current_timestamp_ms=13_500,  # trễ 3500ms -> Tier 2
        trade_intent="SL",
    )
    assert mode_sl == "FORCE_MARKET"
    assert info_sl["fallback_tier"] == 2

    # Trường hợp 3: Lệnh chốt lời hoặc thoát thông thường (NON-EMERGENCY EXIT) khi trễ 3500ms -> POSTPONE_NON_EMERGENCY_EXIT
    mode_trail, info_trail = resolve_execution_mode_with_l2_fallback(
        intended_mode="LIMIT",
        l2_snapshot=snapshot,
        limit_price=100.0,
        side=-1,
        current_timestamp_ms=13_500,  # trễ 3500ms -> Tier 2
        trade_intent="TRAIL_PROFIT",
    )
    assert mode_trail == "POSTPONE_NON_EMERGENCY_EXIT"
    assert info_trail["fallback_tier"] == 2



def test_b_2_2_simulate_limit_fill_with_queue():
    """
    [TDD VERIFICATION - SIMULATE LIMIT FILL]:
    Kiểm tra mô phỏng khớp lệnh giới hạn khi tổng volume giao dịch thị trường vượt qua
    lượng xếp hàng phía trước (queue_effective) + kích thước lệnh (order_size).
    """
    # queue_effective = 30.0, order_size = 10.0 => required_volume = 40.0
    vols = np.array([15.0, 20.0, 10.0, 50.0], dtype=np.float64)
    # bar 1: cum = 15.0 < 40.0
    # bar 2: cum = 35.0 < 40.0
    # bar 3: cum = 45.0 >= 40.0 => FILLED tại bar 3!
    res = simulate_limit_fill_with_queue(
        queue_effective=30.0, order_size=10.0, subsequent_volumes=vols, timeout_bars=5
    )
    assert res["filled"] is True
    assert res["fill_bar_idx"] == 3
    assert res["cumulative_volume_processed"] == pytest.approx(45.0)

    # Khi timeout_bars = 2, lệnh chưa kịp khớp
    res_timeout = simulate_limit_fill_with_queue(
        queue_effective=30.0, order_size=10.0, subsequent_volumes=vols, timeout_bars=2
    )
    assert res_timeout["filled"] is False
    assert res_timeout["timeout_reached"] is True


def test_tier1_mild_jitter_exponential_backoff_and_ttl():
    """
    [TDD VERIFICATION - KHẮC PHỤC LỖ HỔNG 4: API RATE LIMIT & OTR BAN]:
    Kiểm chứng khi fallback Tier 1 (Mild Jitter) kích hoạt, thông tin queue_info
    thiết lập TTL = 15s (dài hơn để giữ lệnh không bị hủy liên tục) cùng các thông số
    Exponential Backoff (backoff_multiplier=2.0, max_retry_attempts=3, next_retry_ms=2000)
    ngăn chặn triệt để OTR Ban / HTTP 429 từ sàn giao dịch.
    """
    snapshot = {
        "timestamp_ms": 10_000,
        "bids": {"100.0": 20.0},
        "asks": {"100.5": 10.0},
    }

    mode, info = resolve_execution_mode_with_l2_fallback(
        intended_mode="LIMIT",
        l2_snapshot=snapshot,
        limit_price=100.0,
        side=1,
        current_timestamp_ms=11_200,  # trễ 1200ms -> thuộc Tier 1
        trade_intent="ENTRY",
        tier1_threshold_ms=500,
        tier2_threshold_ms=2000,
        spoofing_discount=0.70,
    )

    assert mode == "POST_ONLY_LIMIT"
    assert info["fallback_tier"] == 1
    assert info["ttl_seconds"] == 15
    assert info["backoff_multiplier"] == pytest.approx(2.0)
    assert info["max_retry_attempts"] == 3
    assert info["next_retry_ms"] == 2000

