"""
Module K.1 — Trừ chính xác chi phí funding (lãi qua đêm) của Perpetual Futures.

Funding là đặc trưng RIÊNG của hợp đồng vĩnh cửu (perpetual futures): sàn không có
ngày đáo hạn nên dùng funding rate để neo giá perp vào giá spot. Mỗi chu kỳ (Binance:
00:00 / 08:00 / 16:00 UTC), bên này trả cho bên kia:

    funding_payment = position_notional × funding_rate × side

Quy ước dấu THỐNG NHẤT toàn hệ thống:
    - Trả về > 0  => CHÚNG TA TRẢ tiền (chi phí, làm giảm PnL ròng).
    - Trả về < 0  => CHÚNG TA NHẬN tiền (rebate, làm tăng PnL ròng).

Với funding_rate > 0 (perp đắt hơn spot): LONG trả cho SHORT.
Với funding_rate < 0 (perp rẻ hơn spot): SHORT trả cho LONG.
Do đó: payment = notional × rate × side  (side = +1 Long, -1 Short).

Giá trị trả về khớp trực tiếp với tham số `funding_accrued_usd` của
`aegis.execution.pnl.compute_realized_pnl`.
"""

import math
from typing import Dict, List, Optional, Sequence, Union

# Chu kỳ funding chuẩn của Binance USDⓈ-M: 8 giờ, neo vào 00:00 UTC.
DEFAULT_FUNDING_INTERVAL_HOURS = 8
_MS_PER_HOUR = 3_600_000

# Funding rate danh nghĩa mặc định khi chưa có dữ liệu thật từ sàn (0.01% / 8h).
# Đây là mức "cân bằng" mà Binance dùng làm baseline cho phần lớn cặp giao dịch.
DEFAULT_FUNDING_RATE = 0.0001

# Trần vệ sinh dữ liệu: Binance kẹp funding rate trong ±0.75% mỗi chu kỳ.
MAX_ABS_FUNDING_RATE = 0.0075


def funding_timestamps_between(
    start_ts_ms: int,
    end_ts_ms: int,
    interval_hours: int = DEFAULT_FUNDING_INTERVAL_HOURS,
) -> List[int]:
    """
    Liệt kê các mốc quyết toán funding rơi vào khoảng (start_ts_ms, end_ts_ms].

    Mốc quyết toán là bội số của `interval_hours` tính từ epoch UTC — trùng khớp
    với 00:00 / 08:00 / 16:00 UTC của Binance khi interval_hours = 8.

    Biên nửa mở: vị thế mở ĐÚNG tại mốc quyết toán không phải trả cho mốc đó,
    nhưng vị thế đóng đúng tại mốc quyết toán thì phải trả (vẫn đang nắm giữ).
    """
    if not isinstance(interval_hours, int) or interval_hours <= 0:
        raise ValueError(f"interval_hours phải là số nguyên dương, nhận {interval_hours}")

    start = int(start_ts_ms)
    end = int(end_ts_ms)
    if end <= start:
        return []

    interval_ms = interval_hours * _MS_PER_HOUR

    # Mốc quyết toán đầu tiên NẰM SAU start (biên trái mở).
    first = ((start // interval_ms) + 1) * interval_ms

    return list(range(first, end + 1, interval_ms))


def _resolve_rate_for_timestamp(
    ts_ms: int,
    funding_rate: Union[float, Dict[int, float], Sequence],
) -> float:
    """
    Lấy funding rate áp dụng cho một mốc quyết toán.

    Chấp nhận:
    - float: rate hằng số cho mọi chu kỳ.
    - dict {timestamp_ms: rate}: lịch rate thật lấy từ sàn.
    """
    if isinstance(funding_rate, dict):
        rate = funding_rate.get(int(ts_ms))
        if rate is None:
            # Không có dữ liệu cho mốc này -> không phỏng đoán, coi như 0.
            return 0.0
        return float(rate)
    return float(funding_rate)


def compute_funding_accrued_usd(
    size_notional: float,
    side: int,
    entry_ts_ms: int,
    exit_ts_ms: int,
    funding_rate: Union[float, Dict[int, float]] = DEFAULT_FUNDING_RATE,
    interval_hours: int = DEFAULT_FUNDING_INTERVAL_HOURS,
) -> float:
    """
    Tổng chi phí funding (USD) mà vị thế phải gánh trong suốt thời gian nắm giữ.

    Tham số:
    - size_notional: Quy mô danh nghĩa vị thế (USD).
    - side: +1 (Long) hoặc -1 (Short).
    - entry_ts_ms / exit_ts_ms: Mốc thời gian vào/ra lệnh (epoch milliseconds).
    - funding_rate: Hằng số, hoặc dict {timestamp_ms: rate} lấy từ sàn.
    - interval_hours: Chu kỳ funding (Binance USDⓈ-M = 8 giờ).

    Trả về: USD. Dương = chúng ta TRẢ, âm = chúng ta NHẬN.

    [ARMOR GUARD] Chặn input rác theo đúng chuẩn bọc thép của hệ thống.
    """
    if side not in (1, -1):
        raise ValueError(f"side bắt buộc là +1 (Long) hoặc -1 (Short), nhận {side}")

    if (
        not isinstance(size_notional, (int, float))
        or math.isnan(size_notional)
        or math.isinf(size_notional)
        or size_notional < 0
    ):
        raise ValueError(f"size_notional phải >= 0 và hợp lệ, nhận {size_notional}")

    if size_notional == 0.0:
        return 0.0

    settlements = funding_timestamps_between(entry_ts_ms, exit_ts_ms, interval_hours)
    if not settlements:
        return 0.0

    total = 0.0
    for ts in settlements:
        rate = _resolve_rate_for_timestamp(ts, funding_rate)
        if math.isnan(rate) or math.isinf(rate):
            raise ValueError(f"funding_rate rác tại mốc {ts}: {rate}")
        # Vệ sinh dữ liệu: kẹp theo trần Binance để một tick lỗi của feed
        # không thể tạo ra khoản phí phi lý ăn sạch tài khoản.
        rate = max(-MAX_ABS_FUNDING_RATE, min(MAX_ABS_FUNDING_RATE, rate))
        total += size_notional * rate * float(side)

    return float(total)


def compute_funding_accrued_pct(
    size_notional: float,
    side: int,
    entry_ts_ms: int,
    exit_ts_ms: int,
    funding_rate: Union[float, Dict[int, float]] = DEFAULT_FUNDING_RATE,
    interval_hours: int = DEFAULT_FUNDING_INTERVAL_HOURS,
) -> float:
    """
    Funding cộng dồn dưới dạng TỶ LỆ trên notional.

    Đây chính là dạng mà `liquidation_layer.compute_liquidation_price` cần
    (tham số `funding_accrued_pct`): funding bòn rút ký quỹ nên đẩy giá thanh lý
    lại gần giá vào lệnh hơn.
    """
    if size_notional <= 0:
        return 0.0
    usd = compute_funding_accrued_usd(
        size_notional=size_notional,
        side=side,
        entry_ts_ms=entry_ts_ms,
        exit_ts_ms=exit_ts_ms,
        funding_rate=funding_rate,
        interval_hours=interval_hours,
    )
    return float(usd / size_notional)
