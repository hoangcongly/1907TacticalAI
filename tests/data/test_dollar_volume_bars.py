import time
import numpy as np
import pytest
import polars as pl

from aegis.data.bars.dollar_volume_bars import (
    compute_median_ticks_to_fill_per_tick,
    _pass1_extract_tick_counts,
)
from aegis.core.experiment_tracker import ExperimentTracker, TrialClass


def test_pass1_extract_tick_counts_logic():
    """
    Kiểm tra logic tách nến (Worst-case allocation Numba O(N)) cơ bản.
    """
    # 6 ticks. Cấu trúc: [timestamp_ms, price, volume, ...]
    # Mỗi tick có volume = 1, price = 100 -> dollar volume = 100/tick
    ticks = np.array([
        [1000, 100.0, 1.0],
        [1001, 100.0, 1.0],
        [1002, 100.0, 1.0],  # Tick thứ 3: Tích lũy 300
        [1003, 100.0, 1.0],
        [1004, 100.0, 1.0],
        [1005, 100.0, 1.0],  # Tick thứ 6: Tích lũy 300
    ], dtype=np.float64)

    # Đặt threshold = 250 cho tất cả
    daily_thresholds = np.full(6, 250.0, dtype=np.float64)

    bar_counts, bar_end_idx = _pass1_extract_tick_counts(ticks, daily_thresholds)
    
    # Nến 1: tick 0, 1, 2 -> 3 ticks (dollar = 300 > 250). Kết thúc ở index 2.
    # Nến 2: tick 3, 4, 5 -> 3 ticks (dollar = 300 > 250). Kết thúc ở index 5.
    assert len(bar_counts) == 2
    assert bar_counts[0] == 3.0
    assert bar_counts[1] == 3.0
    assert bar_end_idx[0] == 2
    assert bar_end_idx[1] == 5


def test_compute_median_ticks_to_fill_pit_safe():
    """
    Kiểm tra tính năng Causal Point-in-Time (chống rò rỉ tương lai).
    """
    # 9 ticks, mỗi tick = $100
    ticks = np.array([[1000 + i, 100.0, 1.0] for i in range(9)], dtype=np.float64)
    daily_thresholds = np.full(9, 250.0, dtype=np.float64)
    # Sẽ sinh ra 3 nến:
    # Nến 1 (idx 0..2): 3 ticks
    # Nến 2 (idx 3..5): 3 ticks
    # Nến 3 (idx 6..8): 3 ticks
    
    median_per_tick = compute_median_ticks_to_fill_per_tick(ticks, daily_thresholds, window=2)
    
    # Nến 1 (idx 0..2): Chưa có nến trước đó (shift 1 null) -> NaN/Inf
    assert np.isinf(median_per_tick[0:3]).all()
    
    # Nến 2 (idx 3..5): Nến trước đó là nến 1. Nhưng window=2 đòi hỏi phải đủ 2 mẫu để tính median.
    # Do đó theo chuẩn mặc định của Polars, nó vẫn đang trong giai đoạn warm-up -> Inf
    assert np.isinf(median_per_tick[3:6]).all()
    
    # Nến 3 (idx 6..8): Nến trước là nến 1 và 2 ([3, 3]). Đã đủ 2 mẫu. Rolling median = 3.0
    assert np.allclose(median_per_tick[6:9], 3.0)


def test_compute_median_ticks_1_million_performance():
    """
    [TASK A-2-3 MANDATE] Thử nghiệm stress test Runtime Tuyến tính trên 1 Triệu Ticks!
    Đo lường sức mạnh cấp phát worst-case của thuật toán Two-Pass.
    """
    N_TICKS = 1_000_000
    
    # Sinh dữ liệu ngẫu nhiên 1M ticks
    np.random.seed(42)
    prices = np.random.uniform(100.0, 200.0, size=N_TICKS)
    volumes = np.random.uniform(0.1, 5.0, size=N_TICKS)
    timestamps = np.arange(N_TICKS) * 10
    
    ticks = np.column_stack((timestamps, prices, volumes)).astype(np.float64)
    
    # Ngưỡng giả lập: Trung bình 1 nến có 100 ticks (dollar volume = 150 * 2.5 * 100 = ~37500)
    daily_thresholds = np.full(N_TICKS, 37500.0, dtype=np.float64)
    
    # Bấm giờ (Gồm cả warm-up của Numba JIT)
    t0 = time.perf_counter()
    res1 = compute_median_ticks_to_fill_per_tick(ticks, daily_thresholds, window=100)
    t1 = time.perf_counter()
    
    # Bấm giờ (Lần 2, Numba đã JIT compiled -> Tốc độ thuần túy)
    t2 = time.perf_counter()
    res2 = compute_median_ticks_to_fill_per_tick(ticks, daily_thresholds, window=100)
    t3 = time.perf_counter()
    
    pure_runtime = t3 - t2
    
    print(f"\n[STRESS TEST A-2-3] Numba Cấp Phát Worst-Case trên {N_TICKS:,} Ticks:")
    print(f"  -> Lần 1 (Bao gồm thời gian dịch JIT): {t1 - t0:.4f} giây")
    print(f"  -> Lần 2 (Thời gian chạy thuần tính O(N)): {pure_runtime:.4f} giây")
    
    assert len(res2) == N_TICKS
    assert pure_runtime < 1.0, f"HIỆU NĂNG SUY GIẢM NGHIÊM TRỌNG! Chạy mất {pure_runtime} giây."
    
    # Ghi lại dấu ấn kiểm toán
    tracker = ExperimentTracker()
    tracker.log_trial(
        TrialClass.MODEL_FITTING,
        params={
            "module": "Module A.2 - Two-Pass Median Ticks Computation",
            "task_id": "A-2-3",
            "n_ticks_tested": N_TICKS,
            "worst_case_allocation": True,
            "complexity_guarantee": "O(N)",
        },
        metrics={
            "jit_compile_time_sec": t1 - t0,
            "pure_execution_time_sec": pure_runtime,
            "status_verified": True,
        },
    )

