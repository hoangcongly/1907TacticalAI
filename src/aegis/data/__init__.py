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

__all__ = [
    "compute_rolling_mad",
    "detect_bad_tick_core",
    "detect_bad_tick_cross_venue",
    "TickLevelKalmanReplacer",
    "kalman_replacer_filter_series_numba",
    "ensure_pd_matrix_2x2_numba",
    "CleanedTickStreamResult",
    "filter_outliers_4_conditions",
    "clean_tick_stream",
    "compute_pit_safe_daily_threshold",
    "map_daily_threshold_to_ticks",
]
