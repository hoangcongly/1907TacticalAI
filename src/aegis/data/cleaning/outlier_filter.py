"""
[TASK A-1-5] Module Outlier Filter & Clean Tick Stream Pipeline.
Tich hop Bo loc Outlier 4 Dieu Kien (MAD 5-sigma + Volume + Micro-reversal + Cross-venue Parity)
va Bo loc Kalman Replacer (Predict-Only Protocol) thanh pipeline clean_tick_stream hoan chinh.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

from aegis.data.outlier_detection import (
    compute_rolling_mad,
    detect_bad_tick_core,
    detect_bad_tick_cross_venue,
)
from aegis.data.cleaning.tick_kalman_replacer import (
    kalman_replacer_filter_series_numba,
    ensure_pd_matrix_2x2_numba,
)


@dataclass(frozen=True)
class CleanedTickStreamResult:
    """
    [TASK A-1-5] Output cau truc cua pipeline clean_tick_stream.
    Chua chuoi gia da duoc lam sach qua Kalman Predict-Only, cac estimate trang thai,
    va cac mang phan loai bad tick / tail event theo chuan Data Contracts.
    """

    timestamps: np.ndarray
    raw_prices: np.ndarray
    clean_prices: np.ndarray
    volumes: np.ndarray
    level_estimates: np.ndarray
    trend_estimates: np.ndarray
    is_bad_tick: np.ndarray
    is_tail_event: np.ndarray
    robust_sigmas: np.ndarray


def filter_outliers_4_conditions(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    ref_timestamps: Optional[np.ndarray] = None,
    ref_prices: Optional[np.ndarray] = None,
    window: int = 100,
    window_ms: int = 500,
    eta_confirm: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    [TASK A-1-5] Kiem Dinh 4 Dieu Kien Dong Thoi (Outlier Filter):
      1. Extreme Deviation: |P_i - P_{i-1}| > 5 * sigma_MAD_i
      2. Volume Consistency: V_i < 2 * median(V_{i-window : i-1})
      3. Micro-Reversal: |P_{i+1} - P_{i-1}| < 0.3 * |P_i - P_{i-1}|
      4. Cross-Venue Parity: max |P_ref(t) - P_ref(anchor)| < eta_confirm * sigma_ref

    Tra ve:
        Tuple (final_is_bad_tick, is_tail_event, robust_sigmas):
        - final_is_bad_tick: Mang bool danh dau Bad Tick can thay the (bao gom ca NaN gaps).
        - is_tail_event: Mang bool danh dau su kien duoi den (Dong tien that, KHONG loc, giu nguyen gia quan sat).
        - robust_sigmas: Mang Robust Standard Deviation (1.4826 * MAD).
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    n = len(prices)
    if len(timestamps) != n or len(volumes) != n:
        raise ValueError(
            f"Do dai mang timestamps ({len(timestamps)}), prices ({n}), "
            f"va volumes ({len(volumes)}) phai khop nhau tuyet doi."
        )

    if n == 0:
        return (
            np.zeros(0, dtype=np.bool_),
            np.zeros(0, dtype=np.bool_),
            np.full(0, np.nan, dtype=np.float64),
        )

    # 1. Tinh Robust Sigma tu MAD 100-tick cuoi qua khuc (Causal Window, Task A-1-1)
    robust_sigmas = compute_rolling_mad(prices, window=window)

    # 2. Phan loai 3 dieu kien core (Task A-1-2)
    # is_bad_tick_core thoa man dieu kien 1, 2, 3; is_tail_event thoa man DK 1 nhung vi pham DK 2 (Vol >= 2*median)
    is_bad_tick_core, is_tail_event = detect_bad_tick_core(
        prices, volumes, robust_sigmas, window=window
    )

    # 3. Kiem tra Dieu kien 4: Cross-Venue Parity Check (Task A-1-3)
    if ref_timestamps is not None and ref_prices is not None:
        ref_timestamps_arr = np.asarray(ref_timestamps, dtype=np.int64)
        ref_prices_arr = np.asarray(ref_prices, dtype=np.float64)
        if len(ref_timestamps_arr) != len(ref_prices_arr):
            raise ValueError("Do dai ref_timestamps va ref_prices phai khop nhau.")

        if len(ref_prices_arr) > 0:
            robust_sigmas_ref = compute_rolling_mad(ref_prices_arr, window=window)
            cond4_satisfied = detect_bad_tick_cross_venue(
                timestamps=timestamps,
                ref_timestamps=ref_timestamps_arr,
                ref_prices=ref_prices_arr,
                robust_sigmas_ref=robust_sigmas_ref,
                window_ms=window_ms,
                eta_confirm=eta_confirm,
            )
            # Bat buoc thoa man DONG THOI 4 dieu kien doi voi cac tick huu han
            final_is_bad_tick = is_bad_tick_core & cond4_satisfied
        else:
            # Fallback False khi san phu rong (khong loc bad tick khoi chuoi, uong tin bao toan data)
            final_is_bad_tick = np.zeros(n, dtype=np.bool_)
    else:
        # Chuan single-venue (khi khong truyen tham so reference venue) -> dung 3 dieu kien core
        final_is_bad_tick = is_bad_tick_core.copy()

    # [BUG FIX #4 COMPLIANCE]
    # Cac tick bi NaN hoac Inf ben trong chuoi gia luon luon duoc coi la bad/unobserved tick
    # de bo loc Kalman thuc hien Predict-Only, bat ke Dieu Kien 4 la gi.
    nan_mask = np.isnan(prices) | np.isinf(prices)
    if np.any(nan_mask):
        final_is_bad_tick |= nan_mask
        # Ticks bi NaN/Inf khong bao gio duoc coi la tail event hop le
        is_tail_event &= ~nan_mask

    return final_is_bad_tick, is_tail_event, robust_sigmas


def clean_tick_stream(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    ref_timestamps: Optional[np.ndarray] = None,
    ref_prices: Optional[np.ndarray] = None,
    window: int = 100,
    window_ms: int = 500,
    eta_confirm: float = 2.0,
    Q_tick: Optional[np.ndarray] = None,
    R_tick: float = 1e-2,
) -> CleanedTickStreamResult:
    """
    [TASK A-1-5] Ghep Ngan Kien Truc clean_tick_stream Pipeline (Master Blueprint Module A.0).
    Tich hop module phat hien Bad Tick & Tail Event (A-1-1 den A-1-3) voi TickLevelKalmanReplacer (A-1-4)
    tao thanh luong xu ly tick hoan chinh truoc khi ban giao cho module xay dung nen tin hieu (Track B).

    Quy tac hoat dong:
      - Khi gap Bad Tick hoac NaN gap (`is_bad_tick = True`): Bo loc Kalman kich hoat giao thuc Predict-Only
        (bo qua buoc Update), the cho gia tri rac bang gia y_hat du bao tu trang thai [P_t, nu_t],
        duy tri truyen tai dong luong vi cau truc mượt mà mà không gây giật bẻ ngang.
      - Khi gap Tail Event (`is_tail_event = True`): Tuoi dong tien that thao chay voi vol lon (>= 2x median),
        he thong KHONG XOA va KHONG LOC GIA, giu nguyen gia quan sat va cho phep Bo loc Kalman thuc hien
        buoc Update de bat trung cu nhay ngat quyen.

    Tham so:
        timestamps: Mang 1D int64 thoi gian ms (knowledge_time).
        prices: Mang 1D float64 gia khop lenh thien tiep.
        volumes: Mang 1D float64 khoi luong lenh.
        ref_timestamps, ref_prices: Mang 1D san doi chung (tu chon cho Dieu Kien 4 Cross-Venue Parity).
        window: Kich thuoc cua so trut MAD va median vol (Mac dinh 100).
        window_ms: Kich thuoc cua so thoi gian chéo sàn (Mac dinh 500ms).
        eta_confirm: Nguong he so chiu bien dong chéo sàn (Mac dinh 2.0 sigma).
        Q_tick: Ma tran hiep phuong sai nhieu qua trinh 2x2 cho Kalman Replacer.
        R_tick: Phuong sai nhieu quan sat cho Kalman Replacer.

    Tra ve:
        CleanedTickStreamResult object voi cac mang:
        - clean_prices, level_estimates, trend_estimates, is_bad_tick, is_tail_event, v.v.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)

    n = len(prices)
    if n == 0:
        return CleanedTickStreamResult(
            timestamps=timestamps,
            raw_prices=prices,
            clean_prices=np.full(0, np.nan, dtype=np.float64),
            volumes=volumes,
            level_estimates=np.zeros(0, dtype=np.float64),
            trend_estimates=np.zeros(0, dtype=np.float64),
            is_bad_tick=np.zeros(0, dtype=np.bool_),
            is_tail_event=np.zeros(0, dtype=np.bool_),
            robust_sigmas=np.full(0, np.nan, dtype=np.float64),
        )

    # 1. Phat hien va phan loai Outlier (4 Dieu Kien / Core + Cross-Venue Parity)
    is_bad_tick, is_tail_event, robust_sigmas = filter_outliers_4_conditions(
        timestamps=timestamps,
        prices=prices,
        volumes=volumes,
        ref_timestamps=ref_timestamps,
        ref_prices=ref_prices,
        window=window,
        window_ms=window_ms,
        eta_confirm=eta_confirm,
    )

    # 2. Khoi tao ma tran Q_tick mac dinh neu khong truyen vao
    if Q_tick is None:
        # Mac dinh level_noise = 1e-4, trend_noise = 1e-5 theo chuan A-1-4
        Q_tick_arr = np.array([[1e-4, 0.0], [0.0, 1e-5]], dtype=np.float64)
    else:
        Q_tick_arr = np.asarray(Q_tick, dtype=np.float64)
        if Q_tick_arr.shape != (2, 2):
            raise ValueError(f"Q_tick phai co kich thuoc (2, 2), nhan {Q_tick_arr.shape}")

    # Chinh dung Armor Guard Cholesky PD cho Q_tick truoc khi chay
    Q_pd = ensure_pd_matrix_2x2_numba(Q_tick_arr)

    # 3. Thuc thi Kalman Predict-Only Replacer boi Numba C-Engine (O(N) high performance)
    clean_prices, level_estimates, trend_estimates = kalman_replacer_filter_series_numba(
        prices=prices,
        is_bad_ticks=is_bad_tick,
        Q_tick=Q_pd,
        R_tick=float(R_tick),
    )

    return CleanedTickStreamResult(
        timestamps=timestamps,
        raw_prices=prices,
        clean_prices=clean_prices,
        volumes=volumes,
        level_estimates=level_estimates,
        trend_estimates=trend_estimates,
        is_bad_tick=is_bad_tick,
        is_tail_event=is_tail_event,
        robust_sigmas=robust_sigmas,
    )
