"""
Unit & Integration tests cho Task A-1-5: clean_tick_stream Pipeline (Master Blueprint Module A.0).
Kiem dinh bo loc 4 Dieu Kien dong thoi va Giao thuc Predict-Only Tren Quy Mo 1 Ngay Tick Gia Lap
(86,400 ticks, chua hon hop Bad Ticks + Tail Events).
"""

import time
import pytest
import numpy as np

from aegis.data.cleaning.outlier_filter import (
    clean_tick_stream,
    filter_outliers_4_conditions,
    CleanedTickStreamResult,
)
from aegis.core.experiment_tracker import ExperimentTracker, TrialClass


def test_integration_1_day_simulated_ticks_bad_vs_tail():
    """
    [TASK A-1-5 - MASTER INTEGRATION TEST]
    Kiem dinh tich hop 1 ngay tick gia lap (86,400 ticks = 1 tick/s * 86400s) voi bad tick va tail event,
    xac nhan output pipeline xu ly dung chieu va chinh xac cho ca hai loai.
    """
    np.random.seed(42)  # Determinism per SOP Phase 1

    # ========================== WARM-UP NUMBA JIT ==========================
    # Thoi chan JIT compile o mảng nhỡ trước khi bấm giờ kiểm thử hiệu năng
    warmup_ts = np.arange(200, dtype=np.int64) * 1000
    warmup_p = np.random.normal(100.0, 1.0, size=200)
    warmup_v = np.random.uniform(5.0, 15.0, size=200)
    _ = clean_tick_stream(
        timestamps=warmup_ts,
        prices=warmup_p,
        volumes=warmup_v,
        ref_timestamps=warmup_ts.copy(),
        ref_prices=warmup_p.copy(),
        window=50,
        window_ms=500,
    )
    # =======================================================================

    n_ticks = 86400
    start_ts_ms = 1_700_000_000_000

    # 1. Khong gian thoi gian: 1 giay moi tick (86,400 giây = 24h)
    timestamps = start_ts_ms + np.arange(n_ticks, dtype=np.int64) * 1000

    # 2. Tao chuoi gia nền bang Geometric Brownian Motion (GBM)
    returns = np.random.normal(loc=0.0, scale=0.0002, size=n_ticks)
    prices = 100.0 * np.exp(np.cumsum(returns))

    # 3. Tao khoi luong (Volume) xoay quanh trung vi 10.0
    volumes = np.random.uniform(5.0, 15.0, size=n_ticks)

    # 4. Tao chuoi gia san doi chung (Reference Venue) dong bo thien tiep
    ref_timestamps = timestamps.copy()
    ref_prices = prices.copy()

    # --- TIEM NHIEU VI CAU TRUC (BAD TICKS & FLASH SPIKES) ---
    # Cac tick bi spike gia ngat quyen nhung vol nang tai (< 2x median), gia giat ve ngay thi n+1
    bad_tick_indices = [1500, 22000, 45000, 71000]
    for idx in bad_tick_indices:
        # Nhay gia boc vọt +50% hoac -50%
        prices[idx] = prices[idx - 1] * (1.5 if idx % 2 == 0 else 0.5)
        volumes[idx] = 8.0  # Nho hon 2 * median(~10) = 20
        # Gia tick ke tiep tra ve gia thuong (micro-reversal hop le)
        prices[idx + 1] = prices[idx - 1] * 1.0001
        # San phu dung yen, khong ghi nhan chuyen dong spike nay
        ref_prices[idx] = ref_prices[idx - 1]

    # --- TIEM NaN / DATA GAPS ---
    nan_indices = [8000, 64000]
    for idx in nan_indices:
        prices[idx] = np.nan
        volumes[idx] = 0.0
        # Tick k ke tiep tra ve gia tri thien tiep de khong gay vi pham doai tiep
        prices[idx + 1] = prices[idx - 1]

    # --- TIEM DONG TIEN THAT (TAIL EVENTS / INSTITUTIONAL BLACK SWANS) ---
    # Cac moc buoc ngoat suy lech lon, di kem khối lượng giao dịch siêu bùng nổ (>> 20.0)
    tail_event_indices = [5000, 31000, 58000, 82000]
    for idx in tail_event_indices:
        # Dong tien sweeping thuc su: gia nhay +25% hoac -20%
        jump_ratio = 1.25 if idx % 2 == 0 else 0.80
        new_level = prices[idx - 1] * jump_ratio
        prices[idx] = new_level
        prices[idx + 1] = new_level * 1.001  # Gia o vung den mai, KHONG giat ve
        volumes[idx] = 150.0  # Vol khong lo (>> 20) -> vi pham Dieu Kien 2, xac nhan Real Tail Event!

        # San phu CUNG dong bo thien tiep, xac nhan Tail Event la chan thuc
        ref_prices[idx] = new_level
        ref_prices[idx + 1] = new_level * 1.001

    # ========================== THUC THI CHAY PIPELINE ==========================
    start_time = time.perf_counter()
    result = clean_tick_stream(
        timestamps=timestamps,
        prices=prices,
        volumes=volumes,
        ref_timestamps=ref_timestamps,
        ref_prices=ref_prices,
        window=100,
        window_ms=1500,  # Cua so dong bo 1.5s (dap ung chu ky tick 1s)
        eta_confirm=2.0,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 1. Kiem dinh thong so cau truc output
    assert isinstance(result, CleanedTickStreamResult)
    assert len(result.clean_prices) == n_ticks
    assert len(result.is_bad_tick) == n_ticks
    assert len(result.is_tail_event) == n_ticks
    assert len(result.robust_sigmas) == n_ticks

    # 2. Kiem dinh xu ly chuan xac BAD TICKS (Predict-Only replacement)
    for idx in bad_tick_indices:
        assert result.is_bad_tick[idx] == True, f"Bad tick tai idx={idx} KHONG bi bat!"
        assert result.is_tail_event[idx] == False, f"Bad tick tai idx={idx} bi nham thanh Tail Event!"
        # Gia rac phai duoc the cho bang du bao Kalman y_hat (khac xa so ban dau)
        assert np.isfinite(result.clean_prices[idx]), f"Gia clean tai bad tick idx={idx} bi NaN/Inf!"
        assert abs(result.clean_prices[idx] - prices[idx]) / prices[idx] > 0.10, (
            f"Gia clean {result.clean_prices[idx]:.2f} qua giu nguyen gai rac {prices[idx]:.2f}"
        )

    # 3. Kiem dinh xu ly chuan xac NaN Gaps
    for idx in nan_indices:
        assert result.is_bad_tick[idx] == True, f"NaN gap tai idx={idx} khong duoc coi la bad tick!"
        assert np.isfinite(result.clean_prices[idx]), f"Gia clean tai NaN gap idx={idx} khong duoc Kalman ngoai suy!"

    # 4. Kiem dinh xu ly chuan xac TAIL EVENTS (Bao toan gia thuc ban dau)
    for idx in tail_event_indices:
        assert result.is_tail_event[idx] == True, f"Tail event tai idx={idx} KHONG duoc cam co Tail!"
        assert result.is_bad_tick[idx] == False, f"Tail event tai idx={idx} bi nhan nham thanh Bad Tick!"
        # Gia Tail Event PHAI DUOC GIU NGUYEN (price observed) de cap nhat Kalman dong luong
        assert np.isclose(result.clean_prices[idx], prices[idx], rtol=1e-9), (
            f"Gia Tail Event bi thay doi! Expected {prices[idx]}, got {result.clean_prices[idx]}"
        )

    # 5. Kiem dinh hieu nang Numba C-engine (86,400 ticks trong ngan sach thoi gian an toan < 30000 ms)
    assert elapsed_ms < 30000.0, f"Hieu nang xu ly bo loc qua 30 giay! Elapsed: {elapsed_ms:.2f}ms"
    print(
        f"\n[OK] [TASK A-1-5] 1-Day Simulated Tick Stream ({n_ticks} ticks) processed successfully in {elapsed_ms:.2f}ms. "
        f"Bad Ticks ({len(bad_tick_indices)} + {len(nan_indices)} NaNs) & Tail Events ({len(tail_event_indices)}) verified 100% accurate!"
    )

    # 6. Experiment Tracking Logging theo SOP Phase 3
    tracker = ExperimentTracker()
    params = {
        "module": "Module A.0 - Clean Tick Stream Pipeline",
        "task_id": "A-1-5",
        "window": 100,
        "window_ms": 1500,
        "eta_confirm": 2.0,
        "protocol": "Kalman Predict-Only v11.8",
    }
    metrics = {
        "n_ticks_processed": n_ticks,
        "runtime_ms": round(elapsed_ms, 2),
        "bad_ticks_detected": int(np.sum(result.is_bad_tick)),
        "tail_events_detected": int(np.sum(result.is_tail_event)),
        "status_verified": True,
    }
    param_hash = tracker.log_trial(TrialClass.MODEL_FITTING, params, metrics)
    print(f"[AUDIT LOGGED] Task A-1-5 Recorded! Param Hash: {param_hash}")
    assert len(param_hash) == 64


def test_clean_tick_stream_single_venue_and_fallback():
    """
    Kiem dinh che do Single-Venue (khong co san doi chung) va tinh nang Fallback False.
    """
    np.random.seed(101)
    n = 300
    timestamps = np.arange(n, dtype=np.int64) * 1000 + 1_700_000_000_000
    # Phu du lieu voi nhieu tieu chuan nhe de dam bao sigma > 0 (trong 0-sigma degenerate guard)
    prices = 50.0 + np.random.normal(0, 0.02, size=n)
    volumes = np.full(n, 10.0)

    # Diem thu 150 la Bad Tick core (spike 100.0 >> 5-sigma, low vol, immediate micro reversal)
    prices[149] = 50.0
    prices[150] = 100.0
    prices[151] = 50.02
    volumes[150] = 5.0  # Nho hon 2x median (~20.0)

    # Chay single-venue (ref_timestamps = None)
    res_single = clean_tick_stream(
        timestamps=timestamps, prices=prices, volumes=volumes, window=50
    )
    assert res_single.is_bad_tick[150] == True, "Single venue mode phai phat hien va the cho bad tick"
    assert np.isclose(res_single.clean_prices[150], 50.0, atol=0.5), "Gia clean the cho tai bad tick phai xap xi ~50.0"

    print("[OK] [TASK A-1-5] Single-venue filtering & Kalman substitution passed.")


def test_clean_tick_stream_edge_cases_and_guards():
    """
    Kiem dinh cac truong hop bien: Mang rong, warm-up ben trong cua so W, va bat loi khong khop kich thuoc.
    """
    # 1. Empty array check
    res_empty = clean_tick_stream(
        timestamps=np.array([], dtype=np.int64),
        prices=np.array([], dtype=np.float64),
        volumes=np.array([], dtype=np.float64),
    )
    assert len(res_empty.clean_prices) == 0

    # 2. Length mismatch guard
    with pytest.raises(ValueError, match="khop nhau tuyet doi"):
        clean_tick_stream(
            timestamps=np.arange(10, dtype=np.int64),
            prices=np.ones(9, dtype=np.float64),
            volumes=np.ones(10, dtype=np.float64),
        )

    print("[OK] [TASK A-1-5] Armor guards and edge cases verified.")
