"""Module tạo nến Dollar Volume Bars (Task A-2-3 & A-2-4)."""

import numpy as np
import polars as pl
from numba import njit


@njit(nopython=True)
def _pass1_extract_tick_counts(ticks: np.ndarray, daily_thresholds: np.ndarray) -> tuple:
    """
    [Lượt Quét 1 - Numba O(N)]
    Duyệt qua mảng ticks và đánh dấu lại số lượng tick cần thiết để hoàn thành từng nến,
    cùng với vị trí index cuối cùng kết thúc cây nến đó.
    Dùng kỹ thuật Cấp phát Trường hợp xấu nhất (Worst-Case Allocation) để không phải dùng np.append.
    """
    n_ticks = len(ticks)
    # Cấp phát kích thước tối đa: Giả định mỗi tick tạo ra 1 nến
    bar_tick_counts = np.empty(n_ticks, dtype=np.float64)
    bar_end_tick_idx = np.empty(n_ticks, dtype=np.int64)
    
    bar_count = 0
    cum_dollar = 0.0
    tick_count = 0

    for i in range(n_ticks):
        cum_dollar += ticks[i, 1] * ticks[i, 2]  # Giá * Khối lượng
        tick_count += 1
        
        # Nếu cộng dồn dollar volume vượt qua ngưỡng tại thời điểm tick đó
        if cum_dollar >= daily_thresholds[i]:
            bar_tick_counts[bar_count] = float(tick_count)
            bar_end_tick_idx[bar_count] = i
            bar_count += 1
            cum_dollar = 0.0
            tick_count = 0

    # Trả về các mảng đã được cắt gọn (truncate) theo số lượng nến thực tế
    return bar_tick_counts[:bar_count], bar_end_tick_idx[:bar_count]


def compute_median_ticks_to_fill_per_tick(
    ticks: np.ndarray, daily_thresholds: np.ndarray, window: int = 100
) -> np.ndarray:
    """
    [TASK A-2-3] Toán tử tính trung vị số lượng tick trên mỗi nến (PIT-Safe).
    
    Bảo đảm không Look-Ahead Bias nhờ kết hợp '.shift(1)' của Polars.
    Độ phức tạp O(N), hoàn thành bọc lót cấp phát cho toàn bộ mảng đầu ra.
    
    Args:
        ticks: np.ndarray chiều (N, 3+) với cột 1 là giá, cột 2 là volume.
        daily_thresholds: np.ndarray chiều (N,) chứa ngưỡng PIT cho từng tick.
        window: int, cửa sổ trượt để tính trung vị (mặc định 100 nến).
        
    Returns:
        np.ndarray chiều (N,): Chứa giá trị trung vị được ánh xạ ngược về từng tick.
    """
    n_ticks = len(ticks)
    if n_ticks == 0:
        return np.array([], dtype=np.float64)

    # Bước 1: Quét Numba để lấy mảng thống kê độ dài nến
    bar_tick_counts, bar_end_tick_idx = _pass1_extract_tick_counts(ticks, daily_thresholds)
    
    if len(bar_tick_counts) == 0:
        return np.full(n_ticks, np.inf, dtype=np.float64)

    # Bước 2: Dùng Polars tính Rolling Median với .shift(1) khóa chống dữ liệu tương lai
    bar_df = pl.DataFrame({"tick_count": bar_tick_counts, "end_tick_idx": bar_end_tick_idx})
    bar_df = bar_df.with_columns(
        pl.col("tick_count")
        .shift(1)
        .rolling_median(window_size=window)
        .alias("median_ticks_pit")
    )
    
    # Bước 3: Ánh xạ lại (Broadcast) mảng trung vị về kích thước nguyên bản ban đầu N ticks
    median_per_tick = np.full(n_ticks, np.inf, dtype=np.float64)
    end_indices = bar_df["end_tick_idx"].to_numpy()
    median_values = bar_df["median_ticks_pit"].to_numpy()
    
    prev_end = 0
    for k in range(len(end_indices)):
        current_end = int(end_indices[k])
        median_value = median_values[k]
        if not np.isnan(median_value):
            median_per_tick[prev_end : current_end + 1] = median_value
        prev_end = current_end + 1
        
    return median_per_tick
