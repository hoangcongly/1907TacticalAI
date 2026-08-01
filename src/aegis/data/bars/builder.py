"""
Module orchestrator (Entrypoint) cho Track A.
Tích hợp toàn bộ Data Cleaning (A-1) và Dollar Volume Bars Generation (A-2).
Đầu ra tuân thủ tuyệt đối Hợp đồng dữ liệu SIGNAL_BAR_SCHEMA.
"""

import numpy as np
import polars as pl
from typing import Optional

from aegis.data.cleaning.outlier_filter import clean_tick_stream
from aegis.data.bars.pit_threshold import (
    compute_pit_safe_daily_threshold,
    map_daily_threshold_to_ticks,
)
from aegis.data.bars.dollar_volume_bars import (
    compute_median_ticks_to_fill_per_tick,
    generate_dollar_volume_bars_v11,
)


def build_clean_dollar_bars(
    symbol: str,
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    target_daily_volume: float = 1_000_000.0,
    window_mad: int = 100,
    eta_confirm: float = 2.0,
    window_median_ticks: int = 100,
) -> pl.DataFrame:
    """
    [TASK A-2-5] End-to-End Orchestrator.
    
    Thực thi chuỗi thao tác:
    1. Lọc nhiễu Bad Ticks / Thay thế giá bằng bộ lọc Kalman (clean_tick_stream).
    2. Tính ngưỡng thanh khoản PIT-Safe (compute_pit_safe_daily_threshold).
    3. Trượt trung vị Toxicity (compute_median_ticks_to_fill_per_tick).
    4. Sinh nến O(N) (generate_dollar_volume_bars_v11) kèm OFI.
    
    Args:
        symbol: Mã giao dịch (VD: 'BTCUSDT').
        timestamps: np.ndarray thời gian tick (ms).
        prices: np.ndarray giá nguyên bản.
        volumes: np.ndarray khối lượng giao dịch.
        target_daily_volume: Ngưỡng khối lượng mục tiêu mỗi ngày (default: 1M$).
        window_mad: Cửa sổ tính toán nhiễu (MAD).
        eta_confirm: Hệ số Sigma xác nhận nhiễu (Kalman filter).
        window_median_ticks: Số lượng nến để tính trung vị Toxicity.
        
    Returns:
        pl.DataFrame: Bảng nến kết quả tuân thủ 100% SIGNAL_BAR_SCHEMA.
    """
    # 1. Clean the tick stream (Kalman Filter + Outlier Detection)
    clean_res = clean_tick_stream(
        timestamps=timestamps,
        prices=prices,
        volumes=volumes,
        window=window_mad,
        eta_confirm=eta_confirm
    )
    clean_prices = clean_res.clean_prices
    
    # 2. Compute Daily Dollar Volume for PIT Safe Threshold
    ticks_df = pl.DataFrame({
        "timestamp_ms": timestamps,
        "price": clean_prices,
        "volume": volumes
    }).with_columns((pl.col("price") * pl.col("volume")).alias("dollar_volume"))
    
    # Gom nhóm theo ngày (epoch day)
    df_daily = (
        ticks_df.with_columns(
            (pl.col("timestamp_ms") // 86_400_000).cast(pl.Int64).alias("date_epoch_day")
        )
        .group_by("date_epoch_day")
        .agg([
            pl.col("dollar_volume").sum().alias("daily_dollar_volume"),
            pl.col("timestamp_ms").first().alias("timestamp_ms") # giữ lại timestamp để ánh xạ
        ])
        .sort("date_epoch_day")
    )
    
    # Tính Threshold PIT-Safe
    daily_threshold_base = compute_pit_safe_daily_threshold(
        df_daily, window=window_mad, target_freq=target_daily_volume # Lợi dụng tham số target_daily_volume
    )
    
    # 3. Map threshold back to each tick
    daily_thresholds = map_daily_threshold_to_ticks(ticks_df, daily_threshold_base)
    
    # 4. Compute Median Ticks for Toxicity guard
    ticks_array = np.column_stack((timestamps, clean_prices, volumes))
    median_ticks_to_fill = compute_median_ticks_to_fill_per_tick(
        ticks=ticks_array,
        daily_thresholds=daily_thresholds,
        window=window_median_ticks
    )
    
    # 5. Generate Dollar Volume Bars (Numba C-Loop)
    # Chuyển đổi cờ is_tail_event (bool array) sang float64 cho Numba
    is_tail_event_ticks = clean_res.is_tail_event.astype(np.float64)
    
    bars = generate_dollar_volume_bars_v11(
        ticks=ticks_array,
        daily_thresholds=daily_thresholds,
        median_ticks_to_fill=median_ticks_to_fill,
        is_tail_event_ticks=is_tail_event_ticks
    )
    
    # 6. Schema Mapping (Track A -> Track B SIGNAL_BAR_SCHEMA)
    if len(bars) == 0:
        return pl.DataFrame()
        
    df = pl.DataFrame({
        "timestamp_ms": bars[:, 0].astype(np.int64),
        "open": bars[:, 1],
        "high": bars[:, 2],
        "low": bars[:, 3],
        "close": bars[:, 4],
        "volume": bars[:, 5],
        "ofi": bars[:, 6],
        "tick_count": bars[:, 7].astype(np.int64),
        "is_toxic_flag": bars[:, 8] > 0.5,
        "is_tail_event": bars[:, 9] > 0.5,
    })
    
    # Bổ sung các cột bắt buộc theo Hợp Đồng
    df = df.with_columns([
        pl.Series("bar_idx", np.arange(len(bars)), dtype=pl.Int64),
        pl.lit(symbol).alias("symbol"),
        pl.lit(None, dtype=pl.Float64).alias("trend_score"),
        pl.lit(None, dtype=pl.Float64).alias("p_trend"),
        pl.lit(None, dtype=pl.Float64).alias("p_chop"),
        pl.lit(None, dtype=pl.Float64).alias("atr_14"),
        pl.lit(None, dtype=pl.Float64).alias("hurst_value"),
        pl.lit(None, dtype=pl.Float64).alias("d_star_used"),
    ])
    
    # Bật cờ insufficient_history = True cho các nến trong vùng Warm-up
    df = df.with_columns(
        pl.when(pl.col("bar_idx") < window_median_ticks)
        .then(True)
        .otherwise(False)
        .alias("insufficient_history")
    )
    
    # Sắp xếp đúng thứ tự Cột theo Hợp Đồng
    ordered_cols = [
        "bar_idx", "symbol", "timestamp_ms", "open", "high", "low", "close", "volume",
        "ofi", "tick_count", "is_toxic_flag", "is_tail_event", "insufficient_history",
        "trend_score", "p_trend", "p_chop", "atr_14", "hurst_value", "d_star_used"
    ]
    df = df.select(ordered_cols)
    
    return df
