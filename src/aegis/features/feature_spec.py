"""
[FIX F3] Đăng ký đặc trưng — NGUỒN SỰ THẬT DUY NHẤT cho mọi feature phái sinh.

## Vì sao module này tồn tại

Trước đây `research_pipeline._ensure_candidate_features()` tự tính hl_spread/ret_1/
ret_5/vol_ratio khi huấn luyện, còn `live_pipeline.on_bar()` thì KHÔNG — nó dùng
`bar.get(col, 0.0)` nên 4/5 đặc trưng của model bị im lặng điền 0.0 lúc chạy thật.
Model mù 80% mà không có lỗi nào được ném ra.

## Cách module này khiến lỗi đó KHÔNG THỂ tái phát

Mỗi feature được định nghĩa ĐÚNG MỘT LẦN, dưới dạng hàm thuần trên một CỬA SỔ các
nến gần nhất kết thúc tại nến hiện tại:

    f(window) -> float

- Chế độ batch (research): trượt cửa sổ đó dọc lịch sử.
- Chế độ online (live): giữ deque các nến gần nhất, gọi CHÍNH hàm đó.

Vì cả hai đường đi gọi cùng một hàm, parity được bảo đảm **tự thân** — không phụ
thuộc vào việc ai đó nhớ đồng bộ hai bản cài đặt. Đây là điểm mấu chốt của thiết kế:
đánh đổi một chút tốc độ để lấy sự đúng đắn không thể phá vỡ.

## Tính nhân quả

Cửa sổ chỉ chứa các nến <= t. Không có cách nào nhìn thấy tương lai, kể cả do sơ ý.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

# Cửa sổ truyền vào hàm feature: tên cột -> mảng giá trị, phần tử CUỐI là nến hiện tại.
Window = Dict[str, np.ndarray]


@dataclass(frozen=True)
class FeatureSpec:
    """Định nghĩa một đặc trưng phái sinh."""
    name: str
    fn: Callable[[Window], float]
    lookback: int          # số nến cần giữ trong cửa sổ (kể cả nến hiện tại)
    warmup: int            # số nến đầu chuỗi mà giá trị chưa đáng tin
    inputs: Sequence[str]  # các cột thô bắt buộc phải có


def _safe_div(numerator: float, denominator: float, floor: float = 1e-6) -> float:
    return float(numerator / max(denominator, floor))


def _pct_change(window: Window, lag: int) -> float:
    """
    Biến động giá qua `lag` nến. Trả 0.0 khi chưa đủ lịch sử — khớp chính xác
    hành vi `pandas.Series.pct_change(lag).fillna(0.0)` của bản cài đặt cũ.
    """
    closes = window["close"]
    if len(closes) < lag + 1:
        return 0.0
    prev = float(closes[-(lag + 1)])
    if prev == 0.0:
        return 0.0
    return float((float(closes[-1]) - prev) / prev)


def _f_hl_spread(window: Window) -> float:
    """Biên độ nến chuẩn hoá theo giá đóng cửa — proxy biến động tức thời."""
    return _safe_div(float(window["high"][-1]) - float(window["low"][-1]),
                     float(window["close"][-1]))


def _f_ret_1(window: Window) -> float:
    """Lợi suất 1 nến."""
    return _pct_change(window, 1)


def _f_ret_5(window: Window) -> float:
    """Lợi suất 5 nến."""
    return _pct_change(window, 5)


def _f_vol_ratio(window: Window) -> float:
    """
    Khối lượng hiện tại so với trung bình 20 nến gần nhất.
    Tương đương `volume / volume.rolling(20, min_periods=1).mean()`.
    """
    volumes = window["volume"][-20:]
    return _safe_div(float(volumes[-1]), float(np.mean(volumes)))


# ============================================================================
# ĐẶC TRƯNG VI CẤU TRÚC
# ============================================================================
# Ở quy mô vốn nhỏ, edge nằm ở vi cấu trúc chứ không ở chế độ thị trường.
# Các đại lượng dưới đây suy ra trực tiếp từ nến Binance (không tốn thêm request).


def _f_range_position(window: Window) -> float:
    """
    Vị trí giá đóng cửa trong biên độ nến: (close - low) / (high - low).

    ~1.0 = đóng cửa sát đỉnh (áp lực mua thắng), ~0.0 = sát đáy (áp lực bán thắng).
    Proxy rẻ cho áp lực dòng lệnh nội nến.
    """
    high, low = float(window["high"][-1]), float(window["low"][-1])
    span = high - low
    if span <= 1e-12:
        return 0.5
    return float((float(window["close"][-1]) - low) / span)


def _f_ofi_ma5(window: Window) -> float:
    """OFI trung bình 5 nến — dòng lệnh DAI DẲNG có tín hiệu hơn OFI một nến."""
    ofi = window.get("ofi")
    if ofi is None or len(ofi) == 0:
        return 0.0
    return float(np.mean(ofi[-5:]))


def _f_volume_z(window: Window) -> float:
    """Z-score khối lượng trên cửa sổ 20 nến — nhận diện đột biến thanh khoản."""
    volumes = window["volume"][-20:]
    if len(volumes) < 5:
        return 0.0
    std = float(np.std(volumes))
    if std <= 1e-12:
        return 0.0
    return float((float(volumes[-1]) - float(np.mean(volumes))) / std)


def _f_ret_zscore(window: Window) -> float:
    """
    Lợi suất 1 nến chuẩn hoá theo độ lệch chuẩn 20 nến.

    Đây là tín hiệu HỒI QUY VỀ TRUNG BÌNH: |z| lớn = giá vừa đi quá xa so với
    nhiễu thường ngày, khả năng bật lại cao.
    """
    closes = window["close"][-21:]
    if len(closes) < 6:
        return 0.0
    rets = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
    std = float(np.std(rets))
    if std <= 1e-12:
        return 0.0
    return float(rets[-1] / std)


# ============================================================================
# ĐĂNG KÝ — thêm feature mới CHỈ cần thêm một dòng vào đây
# ============================================================================
DERIVED_FEATURES: List[FeatureSpec] = [
    FeatureSpec("hl_spread", _f_hl_spread, lookback=1, warmup=0,
                inputs=("high", "low", "close")),
    FeatureSpec("ret_1", _f_ret_1, lookback=2, warmup=1, inputs=("close",)),
    FeatureSpec("ret_5", _f_ret_5, lookback=6, warmup=5, inputs=("close",)),
    FeatureSpec("vol_ratio", _f_vol_ratio, lookback=20, warmup=0, inputs=("volume",)),
    # --- Vi cấu trúc ---
    FeatureSpec("range_position", _f_range_position, lookback=1, warmup=0,
                inputs=("high", "low", "close")),
    FeatureSpec("ofi_ma5", _f_ofi_ma5, lookback=5, warmup=4, inputs=("ofi",)),
    FeatureSpec("volume_z", _f_volume_z, lookback=20, warmup=5, inputs=("volume",)),
    FeatureSpec("ret_zscore", _f_ret_zscore, lookback=21, warmup=20, inputs=("close",)),
]

# Đặc trưng đi thẳng từ SignalBarSchema (đã được tính ở tầng signal_pipeline).
PASSTHROUGH_FEATURES: List[str] = [
    "ofi", "trend_score", "p_trend", "p_chop", "atr_14", "hurst_value",
]

# `ofi` là cột thô của nến Binance (suy từ taker buy/sell volume) nên vừa được dùng
# trực tiếp làm đặc trưng, vừa làm đầu vào cho ofi_ma5.

DERIVED_BY_NAME: Dict[str, FeatureSpec] = {spec.name: spec for spec in DERIVED_FEATURES}

# Toàn bộ cột ứng viên đưa vào bước chọn đặc trưng.
CANDIDATE_FEATURES: List[str] = [s.name for s in DERIVED_FEATURES] + PASSTHROUGH_FEATURES

# Số nến tối đa cần giữ để tính được mọi feature phái sinh.
MAX_LOOKBACK: int = max(spec.lookback for spec in DERIVED_FEATURES)

# Các cột thô mà cửa sổ phải mang theo.
REQUIRED_INPUTS: List[str] = sorted(
    {col for spec in DERIVED_FEATURES for col in spec.inputs}
)


def is_derived(name: str) -> bool:
    return name in DERIVED_BY_NAME
