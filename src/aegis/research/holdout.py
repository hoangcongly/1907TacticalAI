"""
Kỷ luật train / holdout — hàng rào chống quá khớp khi tinh chỉnh chiến lược.

VẤN ĐỀ: mỗi lần thử một cải tiến rồi nhìn kết quả trên TOÀN BỘ dữ liệu, ta rò rỉ
thông tin của dữ liệu đó vào quyết định. Thử đủ nhiều, kiểu gì cũng tìm ra thứ
"hoạt động" — trên chính dữ liệu đã dùng để tìm nó.

QUY TẮC BẤT DI BẤT DỊCH:
  1. Mọi phát triển, dò tham số, chọn tín hiệu -> CHỈ trên `train`.
  2. `holdout` được mở ĐÚNG MỘT LẦN, khi đã chốt cấu hình cuối.
  3. Nếu holdout tệ -> KHÔNG được quay lại chỉnh rồi thử holdout lần nữa.
     Làm vậy là biến holdout thành train và mất sạch giá trị của nó.

Tách theo THỜI GIAN (không xáo trộn): holdout luôn là giai đoạn GẦN NHẤT, vì đó
mới là chế độ thị trường mà ta sắp giao dịch thật.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class TimeSplit:
    """Ranh giới tách train/holdout theo thời gian."""
    split_ts: int
    train_frac: float

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df.index < self.split_ts]

    def holdout(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df.index >= self.split_ts]

    def describe(self, index: pd.Index) -> str:
        tr = (index < self.split_ts).sum()
        ho = (index >= self.split_ts).sum()
        d0 = pd.to_datetime(self.split_ts, unit="ms").date()
        return (f"train {tr} nến (< {d0}) | holdout {ho} nến (>= {d0}) "
                f"[{tr/(tr+ho)*100:.0f}% / {ho/(tr+ho)*100:.0f}%]")


def make_split(index: pd.Index, train_frac: float = 0.70) -> TimeSplit:
    """Tách theo thời gian tại phân vị `train_frac` của chỉ mục."""
    if not (0.3 <= train_frac <= 0.9):
        raise ValueError(f"train_frac nên thuộc [0.3, 0.9], nhận {train_frac}")
    values = np.sort(np.asarray(index))
    return TimeSplit(split_ts=int(values[int(len(values) * train_frac)]),
                     train_frac=float(train_frac))


def split_panel(
    panel: Dict[str, pd.DataFrame],
    split: TimeSplit,
    part: str = "train",
) -> Dict[str, pd.DataFrame]:
    """Cắt toàn bộ panel về một phía của ranh giới."""
    if part not in ("train", "holdout"):
        raise ValueError(f"part phải là 'train' hoặc 'holdout', nhận {part}")
    fn = split.train if part == "train" else split.holdout
    return {k: fn(v) for k, v in panel.items()}
