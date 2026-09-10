"""
Dựng panel đa tài sản (time x symbol) cho nghiên cứu cross-sectional.

VÌ SAO CẦN: dự đoán hướng đi của MỘT tài sản có tỷ lệ tín hiệu/nhiễu rất tệ — beta
thị trường lấn át mọi tín hiệu riêng. Cross-sectional (so sánh TƯƠNG ĐỐI giữa các
tài sản) khử beta chung, nên cùng một lượng tín hiệu lại cho Sharpe cao hơn nhiều.
Đây là cách phần lớn quỹ định lượng thực sự kiếm tiền.
"""

import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_ROOT = "data/binance"


def load_panel(
    symbols: List[str],
    interval: str = "4h",
    fields: Optional[List[str]] = None,
    root: str = DATA_ROOT,
    min_coverage: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """
    Nạp nhiều cặp và căn theo lưới thời gian chung.

    Trả về dict {tên_trường: DataFrame(index=timestamp_ms, columns=symbol)}.
    Cặp có độ phủ dữ liệu dưới `min_coverage` bị loại (niêm yết muộn / gián đoạn).
    """
    fields = fields or ["close", "high", "low", "volume", "ofi"]
    frames: Dict[str, Dict[str, pd.Series]] = {f: {} for f in fields}

    for symbol in symbols:
        path = pathlib.Path(root) / f"{symbol}_{interval}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path).drop_duplicates("timestamp_ms").set_index("timestamp_ms")
        for field in fields:
            if field in df.columns:
                frames[field][symbol] = df[field].astype(np.float64)

    panel = {f: pd.DataFrame(cols).sort_index() for f, cols in frames.items() if cols}
    if not panel or "close" not in panel:
        raise RuntimeError("Không nạp được panel — chưa tải dữ liệu? Chạy scripts/download_universe.py")

    # Loại cặp thiếu dữ liệu quá nhiều (chống survivorship + chống NaN lan)
    coverage = panel["close"].notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    return {f: df[[c for c in keep if c in df.columns]] for f, df in panel.items()}


def load_funding_panel(
    symbols: List[str],
    close_index: pd.Index,
    root: str = DATA_ROOT,
) -> pd.DataFrame:
    """
    Panel funding rate căn theo lưới thời gian của giá.

    Funding quyết toán mỗi 8h, còn nến có thể 4h — nên rate được giữ nguyên (ffill)
    tới mốc quyết toán tiếp theo. KHÔNG nội suy: funding là sự kiện rời rạc.
    """
    series: Dict[str, pd.Series] = {}
    for symbol in symbols:
        path = pathlib.Path(root) / f"{symbol}_funding.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path).drop_duplicates("funding_time").sort_values("funding_time")
        s = pd.Series(df["rate"].to_numpy(), index=df["funding_time"].to_numpy())
        series[symbol] = s.reindex(s.index.union(close_index)).ffill().reindex(close_index)

    return pd.DataFrame(series).reindex(columns=[c for c in series])


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Lợi suất TƯƠNG LAI qua `horizon` nến — biến mục tiêu.

    Hàng t = lợi suất từ t đến t+horizon. Vào lệnh tại t nghĩa là ăn đúng lợi suất
    này, nên KHÔNG có nhìn trước: tín hiệu tại t chỉ dùng dữ liệu <= t.
    """
    return close.shift(-horizon) / close - 1.0
