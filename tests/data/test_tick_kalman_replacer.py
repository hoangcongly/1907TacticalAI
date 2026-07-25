"""
Unit tests & Verification cho Task A-1-4: TickLevelKalmanReplacer (Predict-Only / Update).
Tuân thủ tuyệt đối quy trình 3-Phase SOP:
- Kiểm thử đối xứng Long/Short Trends.
- Kiểm định tính chính xác toán học trên Synthetic Data.
- Kiểm tra Edge Cases (rỗng, NaN, inf, suy biến ma trận covariance).
- Kiểm chứng Numba Parity và Benchmark hiệu năng C-speed.
- Ghi log thử nghiệm qua ExperimentTracker.
"""

import time
import numpy as np
import pytest
from aegis.data.cleaning.tick_kalman_replacer import (
    TickLevelKalmanReplacer,
    kalman_replacer_filter_series_numba,
    ensure_pd_matrix_2x2_numba,
)
from aegis.core.experiment_tracker import ExperimentTracker
from aegis.core.trial_classes import TrialClass


def test_kalman_replacer_long_short_symmetric_trends():
    """
    [PHASE 3: SYMMETRIC LONG/SHORT TESTS]
    Kiểm thử khả năng duy trì động lượng vi cấu trúc một cách đối xứng trên chuỗi xu hướng
    Tăng (Long Trending) và Giảm (Short Sell-off). Khi phát hiện Bad Tick hoặc mất quan sát (NaN),
    bộ lọc phải tiếp tục phóng chiếu (predict-only) theo đúng xu hướng thay vì đi ngang (flatline).
    """
    np.random.seed(42)
    n = 20
    t = np.arange(n, dtype=np.float64)

    # 1. Long Trend (Xu hướng giá lên: P_0 = 100.0, bước giá +1.0)
    long_prices = 100.0 + 1.0 * t
    long_bad_ticks = np.zeros(n, dtype=np.bool_)
    # Gợi ý một Bad Tick cực đoan tại tick 10 (giá giật xuống 80)
    long_prices[10] = 80.0
    long_bad_ticks[10] = True
    # Gợi ý đứt gãy quan sát (NaN gap) tại tick 15
    long_prices[15] = np.nan

    replacer_long = TickLevelKalmanReplacer(Q_tick=np.array([[1e-2, 0.0], [0.0, 1e-3]]), R_tick=1e-3)
    long_replaced, long_lvl, long_trnd = replacer_long.filter_series(long_prices, long_bad_ticks)

    # Kiểm chứng tại Bad Tick (i=10), giá thay thế phải tuân theo mốc xu hướng (~110.0)
    assert 109.0 <= long_replaced[10] <= 111.0, f"Long Bad Tick replacement sai lệch: {long_replaced[10]}"
    # Kiểm chứng tại NaN Gap (i=15), giá thay thế tiếp tục bám trend (~115.0)
    assert 114.0 <= long_replaced[15] <= 116.0, f"Long NaN Gap replacement sai lệch: {long_replaced[15]}"
    # Trend estimate phải mang giá trị dương (> 0.5)
    assert long_trnd[-1] > 0.5, f"Động lượng Long không dương: {long_trnd[-1]}"

    # 2. Short Trend (Xu hướng giá xuống đối xứng: P_0 = 100.0, bước giá -1.0)
    short_prices = 100.0 - 1.0 * t
    short_bad_ticks = np.zeros(n, dtype=np.bool_)
    # Bad Tick ngược hướng lên tại tick 10
    short_prices[10] = 120.0
    short_bad_ticks[10] = True
    # NaN gap tại tick 15
    short_prices[15] = np.nan

    replacer_short = TickLevelKalmanReplacer(Q_tick=np.array([[1e-2, 0.0], [0.0, 1e-3]]), R_tick=1e-3)
    short_replaced, short_lvl, short_trnd = replacer_short.filter_series(short_prices, short_bad_ticks)

    # Kiểm chứng tại Bad Tick (i=10), giá thay thế phải bám sát xu hướng xuống (~90.0)
    assert 89.0 <= short_replaced[10] <= 91.0, f"Short Bad Tick replacement sai lệch: {short_replaced[10]}"
    # Kiểm chứng tại NaN Gap (i=15), giá thay thế tiếp tục lao xuống (~85.0)
    assert 84.0 <= short_replaced[15] <= 86.0, f"Short NaN Gap replacement sai lệch: {short_replaced[15]}"
    # Trend estimate phải mang giá trị âm (< -0.5)
    assert short_trnd[-1] < -0.5, f"Động lượng Short không âm: {short_trnd[-1]}"

    print("[OK] [TASK A-1-4] Symmetric Long/Short momentum preservation verified successfully.")


def test_kalman_replacer_oop_vs_numba_parity():
    """
    [PHASE 3: REGRESSION/PARITY TEST]
    Chứng minh sự đồng ý tuyệt đối (100% numerical parity) giữa luồng chạy Python OOP step-by-step
    và luồng Numba C-speed vectorized batch execution.
    """
    np.random.seed(101)
    n = 500
    prices = 50.0 + np.cumsum(np.random.normal(0, 0.2, n))
    is_bad_ticks = np.random.rand(n) < 0.1  # 10% bad ticks

    # Gài 5% giá trị NaN ngẫu nhiên
    nan_indices = np.random.choice(n, size=int(n * 0.05), replace=False)
    prices[nan_indices] = np.nan

    Q = np.array([[5e-4, 0.0], [0.0, 5e-5]], dtype=np.float64)
    R = 0.01

    # 1. Chạy từng bước qua Python OOP
    replacer_oop = TickLevelKalmanReplacer(Q_tick=Q, R_tick=R)
    oop_replaced = np.zeros(n, dtype=np.float64)
    for i in range(n):
        oop_replaced[i] = replacer_oop.step(prices[i], bool(is_bad_ticks[i]))

    # 2. Chạy batch bằng Numba Engine
    replacer_numba = TickLevelKalmanReplacer(Q_tick=Q, R_tick=R)
    numba_replaced, numba_lvl, numba_trnd = replacer_numba.filter_series(prices, is_bad_ticks)

    # Kiểm định khớp hoàn toàn sai số < 1e-12
    np.testing.assert_allclose(
        oop_replaced, numba_replaced, rtol=1e-12, atol=1e-12, equal_nan=True,
        err_msg="Sự bất đồng phương sai giữa luồng OOP và luồng Numba!"
    )
    print("[OK] [TASK A-1-4] 100% Numerical Parity verified between Python OOP step and Numba engine.")


def test_kalman_replacer_edge_cases_and_armor_guards():
    """
    [PHASE 3: EDGE CASES & SANITIZATION]
    Kiểm thử độ an toàn, bảo vệ PD ma trận (Cholesky guards), xử lý chuỗi rỗng,
    toàn bộ NaN hoặc toàn bộ Bad Ticks không gây gián đoạn hệ thống.
    """
    # 1. Mảng rỗng
    rep, lvl, trnd = kalman_replacer_filter_series_numba(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.bool_),
        np.eye(2), 1e-2
    )
    assert len(rep) == 0 and len(lvl) == 0 and len(trnd) == 0

    # 2. Chuỗi khởi đầu là NaN hoặc Bad Tick
    prices = np.array([np.nan, 100.0, 101.0, np.nan], dtype=np.float64)
    bad_ticks = np.array([False, True, False, False], dtype=np.bool_)
    
    replacer = TickLevelKalmanReplacer()
    res, _, _ = replacer.filter_series(prices, bad_ticks)
    assert np.isnan(res[0]), "Tick đầu tiên là NaN phải giữ nguyên NaN do chưa đủ dữ liệu khởi tạo"
    assert np.isnan(res[1]), "Tick thứ 2 bị đánh dấu bad_tick khi chưa khởi tạo, không dùng làm anchor"
    assert res[2] == 101.0, "Tick thứ 3 là good tick đầu tiên sẽ khởi tạo bộ lọc an toàn"

    # 3. Ma trận hiệp phương sai bị suy biến / âm (Kiểm định Cholesky Armor Guard)
    degenerate_Q = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)  # Không phải PD (det = -1)
    pd_Q = ensure_pd_matrix_2x2_numba(degenerate_Q)
    # Kiểm định Cholesky thành công trên kết quả đầu ra
    L = np.linalg.cholesky(pd_Q)
    assert L.shape == (2, 2), "Cholesky armor guard không tạo ra được ma trận PD hợp lệ!"
    
    # Khởi tạo bộ lọc với R_tick = 0.0 hoặc âm (phải được tự động cap > 1e-12)
    rep_safe = TickLevelKalmanReplacer(Q_tick=degenerate_Q, R_tick=-0.5)
    assert rep_safe.R >= 1e-12
    print("[OK] [TASK A-1-4] Cholesky PD Armor Guards and Edge Cases verified safely.")


def test_kalman_replacer_performance_and_experiment_tracking():
    """
    [PHASE 3: BENCHMARK & AUDIT LOGGING]
    Đo lường tốc độ thực thi C-level trên quy mô lớn (200,000 ticks), chứng minh hiệu suất
    và ghi lại thông tin xác minh vào ExperimentTracker.
    """
    np.random.seed(999)
    n_big = 200000
    big_prices = 1000.0 + np.cumsum(np.random.normal(0, 0.5, n_big))
    big_bad_ticks = np.random.rand(n_big) < 0.05  # 5% bad ticks

    replacer = TickLevelKalmanReplacer()

    # Warmup Numba JIT compiling
    replacer.filter_series(big_prices[:50], big_bad_ticks[:50])

    # Benchmark Numba Batch Execution
    t_start = time.perf_counter()
    rep_numba, _, _ = replacer.filter_series(big_prices, big_bad_ticks)
    t_numba = time.perf_counter() - t_start

    # Benchmark OOP step execution trên 5,000 tick rồi suy diễn (để tránh chờ lâu)
    n_small = 5000
    replacer_oop = TickLevelKalmanReplacer()
    t_oop_start = time.perf_counter()
    for i in range(n_small):
        replacer_oop.step(big_prices[i], bool(big_bad_ticks[i]))
    t_oop_5k = time.perf_counter() - t_oop_start
    est_t_oop_big = (t_oop_5k / n_small) * n_big

    speedup = est_t_oop_big / max(1e-6, t_numba)
    print(f"\n[BENCHMARK] Numba Engine 200,000 ticks: {t_numba*1000:.2f} ms ({n_big/t_numba:,.0f} ticks/s)")
    print(f"[BENCHMARK] Estimated OOP loop runtime: {est_t_oop_big*1000:.2f} ms (Speedup: {speedup:.1f}x)")
    
    assert speedup > 5.0, f"Numba acceleration chưa đạt kỳ vọng speedup > 5x (đạt {speedup:.1f}x)"
    assert len(rep_numba) == n_big

    # Ghi nhận vào ExperimentTracker (SOP Mandate)
    tracker = ExperimentTracker()
    params = {
        "module": "Module A - Microstructure Preprocessing",
        "component": "TickLevelKalmanReplacer",
        "task_id": "A-1-4",
        "Q_tick": [[1e-4, 0.0], [0.0, 1e-5]],
        "R_tick": 1e-2,
        "n_ticks_test": n_big,
        "protocol": "Predict-Only on Bad Ticks/NaN"
    }
    metrics = {
        "numba_runtime_ms": round(t_numba * 1000, 2),
        "throughput_ticks_sec": round(n_big / t_numba, 0),
        "speedup_vs_oop": round(speedup, 1),
        "parity_verified": True
    }
    param_hash = tracker.log_trial(TrialClass.MODEL_FITTING, params, metrics)
    print(f"[AUDIT LOGGED] Trial Recorded! Param Hash: {param_hash}")
    assert len(param_hash) == 64
