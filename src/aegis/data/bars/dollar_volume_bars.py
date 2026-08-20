"""Module tạo nến Dollar Volume Bars (Task A-2-3 & A-2-4)."""

import numpy as np
import polars as pl
from numba import njit


@njit
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


@njit
def generate_dollar_volume_bars_v11(
    ticks: np.ndarray, 
    daily_thresholds: np.ndarray, 
    median_ticks_to_fill: np.ndarray,
    is_tail_event_ticks: np.ndarray
) -> np.ndarray:
    """
    [TASK A-2-4 & A-2-5] Thuật toán cốt lõi sinh Nến Dollar Volume theo chuẩn AFML v11.8.
    
    Tính toán OHLCV đồng thời gán đặc trưng OFI (Order Flow Imbalance), cờ Toxicity, 
    và cờ Tail Event (Thiên nga đen).
    Áp dụng thuật toán Tick Rule để phân loại lệnh mua chủ động/bán chủ động.
    
    Args:
        ticks: np.ndarray chiều (N, 3+). Cột 0: t_i, Cột 1: p_i, Cột 2: v_i
        daily_thresholds: np.ndarray chiều (N,) chứa ngưỡng $ cho từng tick.
        median_ticks_to_fill: np.ndarray chiều (N,) chứa trung vị lịch sử để tính toxicity.
        is_tail_event_ticks: np.ndarray chiều (N,) chứa cờ tail event (0.0 hoặc 1.0) từ Kalman.
        
    Returns:
        np.ndarray chiều (bar_count, 10)
        Format: [t_i, open, high, low, close, volume, ofi, tick_count, is_toxic, is_tail_event]
    """
    n_ticks = len(ticks)
    bars = np.empty((n_ticks, 10), dtype=np.float64)
    bar_count = 0
    cum_dollar = 0.0
    cum_volume = 0.0
    cum_buy_dollar = 0.0
    cum_sell_dollar = 0.0
    tick_count = 0
    bar_is_tail = 0.0
    
    if n_ticks == 0:
        return bars[:0]
        
    bar_open = ticks[0, 1]
    bar_high = ticks[0, 1]
    bar_low = ticks[0, 1]
    last_tick_rule = 1.0

    for i in range(n_ticks):
        t_i = ticks[i, 0]
        p_i = ticks[i, 1]
        v_i = ticks[i, 2]
        
        # Cập nhật cờ Tail Event cho Bar (toán tử OR)
        if is_tail_event_ticks[i] > 0.5:
            bar_is_tail = 1.0
            
        # 1. Tick Rule Classification
        if i > 0:
            if p_i > ticks[i - 1, 1]: 
                last_tick_rule = 1.0
            elif p_i < ticks[i - 1, 1]: 
                last_tick_rule = -1.0
                
        dollar = p_i * v_i
        cum_dollar += dollar
        cum_volume += v_i
        tick_count += 1
        
        # Phân rã Dollar Volume thành Buy/Sell
        if last_tick_rule > 0.0: 
            cum_buy_dollar += dollar
        else: 
            cum_sell_dollar += dollar
            
        # Cập nhật OHLC
        if p_i > bar_high: bar_high = p_i
        if p_i < bar_low: bar_low = p_i

        # 2. Ngưỡng Đóng Nến
        if cum_dollar >= daily_thresholds[i]:
            # Tính OFI (tránh chia 0)
            ofi = (cum_buy_dollar - cum_sell_dollar) / (cum_buy_dollar + cum_sell_dollar + 1e-8)
            
            # Cờ báo hiệu Toxicity (Độc hại / Sweep)
            is_toxic = 1.0 if (tick_count < 0.5 * median_ticks_to_fill[i] and median_ticks_to_fill[i] > 10.0) else 0.0
            
            # Chốt sổ nến
            bars[bar_count, 0] = t_i
            bars[bar_count, 1] = bar_open
            bars[bar_count, 2] = bar_high
            bars[bar_count, 3] = bar_low
            bars[bar_count, 4] = p_i
            bars[bar_count, 5] = cum_volume
            bars[bar_count, 6] = ofi
            bars[bar_count, 7] = float(tick_count)
            bars[bar_count, 8] = is_toxic
            bars[bar_count, 9] = bar_is_tail
            
            bar_count += 1
            
            # Reset biến tích lũy
            cum_dollar = 0.0
            cum_volume = 0.0
            cum_buy_dollar = 0.0
            cum_sell_dollar = 0.0
            tick_count = 0
            bar_is_tail = 0.0
            
            # Khởi tạo nến mới (nếu chưa phải tick cuối)
            if i + 1 < n_ticks:
                bar_open = ticks[i + 1, 1]
                bar_high = ticks[i + 1, 1]
                bar_low = ticks[i + 1, 1]

    return bars[:bar_count]
