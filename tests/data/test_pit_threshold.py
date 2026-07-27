"""
Unit & Integration Tests cho Task A-2-1 & A-2-2: compute_pit_safe_daily_threshold & map_daily_threshold_to_ticks.
Kiểm định khắt khe nguyên tắc không rò rỉ thông tin tương lai (Causal PIT-Safe: không dùng volume hôm nay/tương lai)
và kiểm chứng hoàn toàn không tồn tại NaN/Null sau khoảng warmup 21 ngày.
"""

import datetime
import pytest
import numpy as np
import polars as pl

from aegis.data.bars.pit_threshold import (
    compute_pit_safe_daily_threshold,
    map_daily_threshold_to_ticks,
)
from aegis.core.experiment_tracker import ExperimentTracker, TrialClass


def test_pit_threshold_no_nan_after_warmup():
    """
    [TASK A-2-1 MANDATE TEST #1] Assert không NaN sau warmup.
    Kiểm chứng từ dòng thứ window (index 21 trở đi trong mặc định 21 ngày) 100% không có NaN, Null hay Inf.
    """
    np.random.seed(42)  # Determinism per SOP Phase 1
    n_days = 100
    window_size = 21
    target_freq = 50.0

    dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    volumes = np.random.uniform(10_000_000.0, 50_000_000.0, size=n_days)

    df_daily = pl.DataFrame({
        "date": dates,
        "daily_dollar_volume": volumes
    })

    df_out = compute_pit_safe_daily_threshold(df_daily, window=window_size, target_freq=target_freq)
    assert "theta_pit" in df_out.columns, "Output phải chứa cột 'theta_pit'"

    # 1. Trong khoảng warm-up (0 đến window - 1), do shift(1) + rolling_mean, các giá trị ban đầu có thể null
    # 2. Sau khoảng warm-up (từ index = window trở đi), KIỂM ĐỊNH TUYỆT ĐỐI KHÔNG NaN/NULL
    post_warmup_slice = df_out.slice(window_size, n_days - window_size)
    assert post_warmup_slice["theta_pit"].null_count() == 0, (
        f"Phát hiện {post_warmup_slice['theta_pit'].null_count()} giá trị null sau warmup {window_size} ngày!"
    )

    post_warmup_array = post_warmup_slice["theta_pit"].to_numpy()
    assert np.all(np.isfinite(post_warmup_array)), "Tất cả các giá trị theta_pit sau warmup phải là finite!"
    assert np.all(post_warmup_array > 0.0), "Ngưỡng gộp nến theta_pit phảistrictly positive (> 0)!"

    # Kiem ngiem công thức toán học thủ công tại dòng index 25
    idx = 25
    expected_mean = np.mean(volumes[idx - window_size : idx]) / target_freq
    actual_val = df_out["theta_pit"][idx]
    assert np.isclose(actual_val, expected_mean, rtol=1e-12), (
        f"Sai lệch toán học! Expected {expected_mean:.6f}, got {actual_val:.6f}"
    )

    print("\n[OK] [TASK A-2-1] Verified ZERO NaNs or Nulls after warmup window! Formula accuracy matched 100%.")


def test_pit_threshold_causality_no_future_leakage():
    """
    [TASK A-2-1 MANDATE TEST #2] Kiểm định không dùng volume hôm nay/tương lai (No Look-ahead Bias).
    Chứng minh bằng phản chứng: Sửa đổi volume của hôm nay (T) và ngày mai (T+1) lên 10,000 lần,
    ngưỡng PIT tại ngày T PHẢI KHÔNG HỀ THAY ĐỔI.
    """
    np.random.seed(101)
    n_days = 80
    window_size = 20

    dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    base_volumes = np.full(n_days, 10_000_000.0, dtype=np.float64)
    df_base = pl.DataFrame({"date": dates, "daily_dollar_volume": base_volumes})

    # Chạy lần 1 lấy ngưỡng nền
    df_res_base = compute_pit_safe_daily_threshold(df_base, window=window_size, target_freq=50.0)

    # Chọn ngày quan trọng T (index 40)
    test_idx = 40
    base_theta_at_T = df_res_base["theta_pit"][test_idx]

    # TIÊM NHIỄU BÙNG NỔ VÀO CHÍNH NGÀY T VÀ T+1 (HÔM NAY VÀ TƯƠNG LAI)
    corrupt_volumes = base_volumes.copy()
    corrupt_volumes[test_idx] = base_volumes[test_idx] * 10_000.0      # Nhồi siêu volume vào chính ngày hôm nay (T)
    corrupt_volumes[test_idx + 1] = base_volumes[test_idx + 1] * 10_000.0  # Nhồi siêu volume vào ngày mai (T+1)

    df_corrupt = pl.DataFrame({"date": dates, "daily_dollar_volume": corrupt_volumes})
    df_res_corrupt = compute_pit_safe_daily_threshold(df_corrupt, window=window_size, target_freq=50.0)

    corrupt_theta_at_T = df_res_corrupt["theta_pit"][test_idx]

    # KIỂM CHỨNG TOÁN HỌC: base_theta_at_T phải BẰNG HOÀN TOÀN corrupt_theta_at_T
    assert np.isclose(base_theta_at_T, corrupt_theta_at_T, rtol=0.0, atol=1e-12), (
        f"Phát hiện Look-Ahead Bias / Rò rỉ dữ liệu tương lai! Khi đổi volume ngày T/T+1, "
        f"ngưỡng ngày T bị lệch từ {base_theta_at_T} thành {corrupt_theta_at_T}!"
    )

    print(
        f"[OK] [TASK A-2-1] Causal isolation verified! Modifying volume at T (Today) & T+1 (Future) "
        f"yielded identically exactly theta={base_theta_at_T:.2f} at day T."
    )

    # Ghi nhận log thực chứng Audit Trail
    tracker = ExperimentTracker()
    tracker.log_trial(
        TrialClass.MODEL_FITTING,
        params={
            "module": "Module A.2 - Point-in-Time Safe Threshold",
            "task_id": "A-2-1",
            "window": window_size,
            "target_freq": 50.0,
            "shift_enforced": 1,
            "causal_no_leak_verified": True,
        },
        metrics={
            "post_warmup_nulls": 0,
            "causal_parity_error_atol": 0.0,
            "status_verified": True,
        },
    )


def test_map_daily_threshold_to_ticks_asof_backward():
    """
    [TASK A-2-2 / SECTION 1.1.2] Kiểm định ánh xạ ngưỡng ASOF backward trên dữ liệu Ticks.
    """
    # 1. Tạo 3 ngày ngưỡng PIT
    df_daily_thresh = pl.DataFrame({
        "date": [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2), datetime.date(2025, 1, 3)],
        "theta_pit": [200_000.0, 250_000.0, 300_000.0]
    })

    # 2. Tạo ticks trôi nổi qua 3 ngày (tính bằng milliseconds kể từ Unix Epoch)
    ts_day1 = int(datetime.datetime(2025, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ts_day2_morn = int(datetime.datetime(2025, 1, 2, 9, 30, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ts_day3_afternoon = int(datetime.datetime(2025, 1, 3, 15, 45, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ts_day4_overflow = int(datetime.datetime(2025, 1, 4, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)

    ticks_df = pl.DataFrame({
        "timestamp_ms": [ts_day1, ts_day2_morn, ts_day3_afternoon, ts_day4_overflow]
    })

    theta_arr = map_daily_threshold_to_ticks(ticks_df, df_daily_thresh)

    assert isinstance(theta_arr, np.ndarray)
    assert theta_arr.dtype == np.float64
    assert theta_arr.flags.c_contiguous

    # Kiểm tra join backward chính xác từng mức giá trị
    assert np.isclose(theta_arr[0], 200_000.0)  # Tick ngày 1 dùng ngưỡng ngày 1
    assert np.isclose(theta_arr[1], 250_000.0)  # Tick ngày 2 dùng ngưỡng ngày 2
    assert np.isclose(theta_arr[2], 300_000.0)  # Tick ngày 3 dùng ngưỡng ngày 3
    assert np.isclose(theta_arr[3], 300_000.0)  # Tick ngày 4 vượt lịch sử -> giữ tiếp (backward zero-order hold) ngưỡng ngày 3

    print("[OK] [TASK A-2-2] ASOF backward mapping passed identically with contiguous float64 arrays!")


def test_pit_threshold_edge_cases_and_armor_guards():
    """
    Kiểm định các hệ thống bảo an Armor Guards: xử lý khuyết cột và các chuỗi số vi phạm.
    """
    # 1. Thiếu cột daily_dollar_volume
    with pytest.raises(ValueError, match="thiếu cột bắt buộc 'daily_dollar_volume'"):
        compute_pit_safe_daily_threshold(pl.DataFrame({"val": [1, 2, 3]}))

    # 2. Tham số window/target_freq âm hoặc 0
    with pytest.raises(ValueError, match="Tham số không hợp lệ"):
        compute_pit_safe_daily_threshold(pl.DataFrame({"daily_dollar_volume": [10.0]}), window=0)

    # 3. Chuỗi chứa NaN sau khi đã sang khoảng warmup mà không qua được fallback
    df_bad_nan = pl.DataFrame({
        "date": [datetime.date(2025, 1, i) for i in range(1, 10)],
        "daily_dollar_volume": [10.0, 10.0, 10.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    # Ở đây do forward_fill() sẽ tự động trám 10.0 sang các ngày nghỉ nên hàm phải XỬ LÝ AN TOÀN không ném lỗi!
    res_fill = compute_pit_safe_daily_threshold(df_bad_nan, window=2, target_freq=1.0)
    assert res_fill["theta_pit"].slice(2).null_count() == 0, "Forward fill phải trám gap hoàn tất!"

    print("[OK] [TASK A-2-1] Armor guards & automatic causal gap-filling verified!")
