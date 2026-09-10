"""
[FIX F3] Hai đường đi tính đặc trưng — batch (research) và online (live).

Cả hai đều gọi CHÍNH các hàm khai báo trong `feature_spec.py`, nên chúng không thể
lệch nhau. Test parity `tests/features/test_feature_parity.py` chốt điều này lại.

Dùng ở đâu:
- `batch_features(df)`      -> ResearchPipeline / CPCVPipeline
- `OnlineFeatureEngine`     -> AegisLivePipeline.on_bar()
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from aegis.features.feature_spec import (
    CANDIDATE_FEATURES,
    DERIVED_BY_NAME,
    DERIVED_FEATURES,
    MAX_LOOKBACK,
    PASSTHROUGH_FEATURES,
    REQUIRED_INPUTS,
    FeatureSpec,
    Window,
)


class MissingFeatureError(KeyError):
    """
    Ném ra khi model yêu cầu một đặc trưng mà nến hiện tại không cung cấp được.

    Đây chính là lỗ hổng F3: bản cũ im lặng điền 0.0 nên model chạy mù mà không ai
    biết. Thà dừng hệ thống còn hơn giao dịch bằng đặc trưng giả.
    """


def _window_from_frame(df: pd.DataFrame, end_idx: int, lookback: int) -> Window:
    """Cắt cửa sổ [end_idx-lookback+1 .. end_idx] (đã kẹp biên trái)."""
    start = max(0, end_idx - lookback + 1)
    return {
        col: df[col].to_numpy()[start:end_idx + 1]
        for col in REQUIRED_INPUTS
        if col in df.columns
    }


def batch_features(
    df: pd.DataFrame,
    names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Tính đặc trưng phái sinh theo lô cho toàn bộ DataFrame nến.

    Trả về BẢN SAO của `df` đã thêm các cột phái sinh. Cột đã tồn tại sẽ được
    ghi đè để đảm bảo luôn khớp với đặc tả (không tin cột có sẵn từ nguồn khác).
    """
    missing_inputs = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing_inputs:
        raise MissingFeatureError(
            f"Thiếu cột thô để tính đặc trưng: {missing_inputs}. Có: {list(df.columns)}"
        )

    wanted: List[FeatureSpec] = (
        list(DERIVED_FEATURES)
        if names is None
        else [DERIVED_BY_NAME[n] for n in names if n in DERIVED_BY_NAME]
    )

    out = df.copy()
    n = len(out)
    for spec in wanted:
        values = np.empty(n, dtype=np.float64)
        for i in range(n):
            values[i] = spec.fn(_window_from_frame(out, i, spec.lookback))
        out[spec.name] = values
    return out


class OnlineFeatureEngine:
    """
    Bộ tính đặc trưng streaming cho live trading.

    Giữ đúng `MAX_LOOKBACK` nến gần nhất rồi gọi CHÍNH hàm mà bản batch dùng.
    Không có công thức nào được viết lại ở đây — đó là toàn bộ mục đích của lớp này.
    """

    def __init__(self, lookback: Optional[int] = None):
        self.lookback = int(lookback or MAX_LOOKBACK)
        self._history: Dict[str, Deque[float]] = {
            col: deque(maxlen=self.lookback) for col in REQUIRED_INPUTS
        }
        self.n_seen: int = 0
        self._last_derived: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        for buffer in self._history.values():
            buffer.clear()
        self.n_seen = 0
        self._last_derived = None

    def _push(self, bar: Dict[str, Any]) -> None:
        """
        Đẩy nến vào cửa sổ — NGUYÊN TỬ: kiểm tra toàn bộ cột trước, rồi mới ghi.

        Nếu vừa ghi vừa kiểm tra, một nến thiếu cột ở giữa sẽ để lại cửa sổ lệch pha
        (cột này đã đẩy, cột kia chưa) và phá vỡ parity với batch một cách âm thầm.
        """
        missing = [c for c in REQUIRED_INPUTS if c not in bar or bar[c] is None]
        if missing:
            raise MissingFeatureError(
                f"Nến thiếu cột thô {missing} — không thể tính đặc trưng phái sinh. "
                f"Cần đủ: {list(REQUIRED_INPUTS)}."
            )
        values = {}
        for col in REQUIRED_INPUTS:
            try:
                values[col] = float(bar[col])
            except (TypeError, ValueError) as exc:
                raise MissingFeatureError(
                    f"Cột thô `{col}` không phải số: {bar[col]!r}"
                ) from exc

        for col, value in values.items():
            self._history[col].append(value)
        self.n_seen += 1

    def _window(self, lookback: int) -> Window:
        return {
            col: np.asarray(buffer, dtype=np.float64)[-lookback:]
            for col, buffer in self._history.items()
        }

    def update(self, bar: Dict[str, Any]) -> Dict[str, float]:
        """
        Nạp một nến mới, trả về TOÀN BỘ đặc trưng phái sinh tại nến đó.

        HỢP ĐỒNG: gọi ĐÚNG MỘT LẦN cho mỗi nến, theo thứ tự thời gian.
        Đây là hàm DUY NHẤT làm thay đổi trạng thái; `build_vector` chỉ đọc.
        """
        self._push(bar)
        self._last_derived = {
            spec.name: float(spec.fn(self._window(spec.lookback)))
            for spec in DERIVED_FEATURES
        }
        return dict(self._last_derived)

    def build_vector(
        self,
        bar: Dict[str, Any],
        selected_features: Sequence[str],
    ) -> np.ndarray:
        """
        Dựng vector đầu vào cho model theo ĐÚNG thứ tự `selected_features`.

        CHỈ ĐỌC — phải gọi `update(bar)` cho nến này trước. Tách bạch như vậy để
        live pipeline có thể nạp lịch sử ở MỌI nến nhưng chỉ dựng vector khi thật
        sự cần suy luận, mà không đẩy trùng nến vào cửa sổ.

        Ném `MissingFeatureError` nếu thiếu bất kỳ đặc trưng nào — TUYỆT ĐỐI không
        im lặng điền 0.0. Đó chính là hành vi đã tạo ra lỗ hổng F3.
        """
        if self._last_derived is None:
            raise MissingFeatureError(
                "Phải gọi update(bar) cho nến hiện tại trước khi build_vector()."
            )
        derived = self._last_derived

        values: List[float] = []
        missing: List[str] = []
        for name in selected_features:
            if name in derived:
                values.append(derived[name])
                continue
            if name in bar and bar[name] is not None and not _is_nan(bar[name]):
                values.append(float(bar[name]))
                continue
            missing.append(name)

        if missing:
            raise MissingFeatureError(
                f"Model yêu cầu đặc trưng {missing} nhưng nến hiện tại không có "
                f"(và chúng không phải đặc trưng phái sinh). "
                f"Đặc trưng phái sinh khả dụng: {sorted(derived)}; "
                f"cột có trong nến: {sorted(k for k in bar)}."
            )

        return np.array([values], dtype=np.float64)


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
