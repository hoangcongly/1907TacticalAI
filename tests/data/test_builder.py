"""
Kiểm định End-to-End (E2E) cho toàn bộ dây chuyền Track A (Task A-2-5).
"""

import pytest
import numpy as np
import polars as pl

from aegis.data.bars.builder import build_clean_dollar_bars
from aegis.core.experiment_tracker import ExperimentTracker, TrialClass


def test_build_clean_dollar_bars_e2e_schema_compliance():
    """
    [TASK A-2-5] Chạy giả lập E2E từ Tick bẩn -> Nến sạch, kiểm tra Schema.
    """
    np.random.seed(42)
    n_ticks = 2000
    
    # 1. Tạo dữ liệu giả lập (tăng dần)
    timestamps = np.arange(1000, 1000 + n_ticks * 100, 100, dtype=np.int64)
    # Giá chạy ngẫu nhiên quanh 50,000
    prices = 50000.0 + np.cumsum(np.random.randn(n_ticks) * 2.0)
    volumes = np.random.uniform(0.1, 2.0, n_ticks)
    
    # Bơm nhiễu (Bad Ticks)
    prices[50] = 100000.0  # Spike
    prices[150] = 0.0      # Zero
    prices[300] = np.nan   # NaN
    
    # Bơm Tail Event (Sập hầm)
    # Rơi 500 giá liên tục trong 10 ticks
    for i in range(500, 510):
        prices[i] = prices[499] - (i - 499) * 50.0
        
    # 2. Chạy qua Orchestrator
    # Thu nhỏ cửa sổ lại để dễ test: 200 nến/ngày → ngưỡng nhỏ → tạo ra nhiều nến
    df_bars = build_clean_dollar_bars(
        symbol="BTCUSDT",
        timestamps=timestamps,
        prices=prices,
        volumes=volumes,
        target_bars_per_day=200,  # 200 nến/ngày → theta_pit nhỏ
        window_mad=20,
        eta_confirm=2.0,
        window_median_ticks=5
    )
    
    # 3. Kiểm định Schema (Data Contracts)
    assert isinstance(df_bars, pl.DataFrame), "Phải trả về Polars DataFrame"
    assert len(df_bars) > 0, "Không tạo được nến nào"
    
    expected_cols = [
        "bar_idx", "symbol", "timestamp_ms", "open", "high", "low", "close", "volume",
        "ofi", "tick_count", "is_toxic_flag", "is_tail_event", "insufficient_history",
        "trend_score", "p_trend", "p_chop", "atr_14", "hurst_value", "d_star_used"
    ]
    assert df_bars.columns == expected_cols, "Lệch SIGNAL_BAR_SCHEMA!"
    
    # Kiểm tra kiểu dữ liệu của một số cột quan trọng
    assert df_bars.schema["bar_idx"] == pl.Int64
    assert df_bars.schema["symbol"] == pl.String
    assert df_bars.schema["is_toxic_flag"] == pl.Boolean
    assert df_bars.schema["is_tail_event"] == pl.Boolean
    assert df_bars.schema["insufficient_history"] == pl.Boolean
    assert df_bars.schema["ofi"] == pl.Float64
    
    # 4. Kiểm tra logic insufficient_history (5 nến đầu phải là True)
    df_head = df_bars.head(5)
    assert df_head["insufficient_history"].all(), "5 nến đầu (Warm-up) phải bị gắn cờ insufficient_history=True"
    
    # Nến thứ 6 trở đi phải là False
    if len(df_bars) > 5:
        assert not df_bars["insufficient_history"][5], "Nến thứ 6 phải an toàn (False)"
        
    # Ghi nhận Audit
    tracker = ExperimentTracker()
    tracker.log_trial(
        TrialClass.MODEL_FITTING,
        params={
            "module": "Module A.2 - Builder E2E",
            "task_id": "A-2-5",
            "n_ticks_input": n_ticks,
            "target_daily_volume": 100000.0,
        },
        metrics={
            "n_bars_output": len(df_bars),
            "schema_compliant": True,
        },
    )
