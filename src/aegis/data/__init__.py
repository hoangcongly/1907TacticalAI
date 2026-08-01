"""Module A: Tien xu ly du lieu vi cau truc, Loc nhieu, The cho Bad Tick, va Nguong Dollar Volume Bars (Tasks A-1 den A-2)."""

from aegis.data.outlier_detection import (
    compute_rolling_mad,
    detect_bad_tick_core,
    detect_bad_tick_cross_venue,
)
from aegis.data.cleaning.tick_kalman_replacer import (
    TickLevelKalmanReplacer,
    kalman_replacer_filter_series_numba,
    ensure_pd_matrix_2x2_numba,
)
from aegis.data.cleaning.outlier_filter import (
    CleanedTickStreamResult,
    filter_outliers_4_conditions,
    clean_tick_stream,
)
from aegis.data.bars.pit_threshold import (
    compute_pit_safe_daily_threshold,
    map_daily_threshold_to_ticks,
)
# NOTE: aegis.data.ingestion.binance_loader chưa được triển khai.
# Import sẽ được bổ sung khi module hoàn thiện (xem Task A-3-x).
from aegis.data.bars.builder import build_clean_dollar_bars
from aegis.data.bars.dollar_volume_bars import (
    compute_median_ticks_to_fill_per_tick,
    generate_dollar_volume_bars_v11,
)

__all__ = [
    # outlier_detection
    "compute_rolling_mad",
    "detect_bad_tick_core",
    "detect_bad_tick_cross_venue",
    # tick_kalman_replacer
    "TickLevelKalmanReplacer",
    "kalman_replacer_filter_series_numba",
    "ensure_pd_matrix_2x2_numba",
    # outlier_filter
    "CleanedTickStreamResult",
    "filter_outliers_4_conditions",
    "clean_tick_stream",
    # pit_threshold
    "compute_pit_safe_daily_threshold",
    "map_daily_threshold_to_ticks",
    # dollar_volume_bars
    "compute_median_ticks_to_fill_per_tick",
    "generate_dollar_volume_bars_v11",
    # builder (E2E Entrypoint)
    "build_clean_dollar_bars",
]
