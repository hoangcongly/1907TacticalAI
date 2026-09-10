import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from aegis.pipelines.research_pipeline import ResearchPipeline
from aegis.pipelines.live_pipeline import AegisLivePipeline, ActivePosition
from aegis.execution.position_sizer import AccountStateTracker
from aegis.risk.circuit_breaker import CircuitBreakerTier


@pytest.fixture
def fitted_artifacts_dir():
    temp_dir = tempfile.mkdtemp()
    np.random.seed(42)
    n_bars = 100
    timestamps = 1700000000000 + np.arange(n_bars) * 60000
    prices = 100.0 + np.cumsum(np.random.normal(0.05, 0.5, n_bars))
    highs = prices + 0.5
    lows = prices - 0.5
    volumes = np.random.uniform(10.0, 100.0, n_bars)

    bars = pd.DataFrame({
        "bar_idx": np.arange(n_bars, dtype=np.int64),
        "symbol": "BTCUSDT",
        "timestamp_ms": timestamps,
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "ofi": np.clip(np.random.normal(0.0, 0.3, n_bars), -1.0, 1.0),
        "tick_count": np.random.randint(10, 50, n_bars, dtype=np.int64),
        "is_toxic_flag": np.zeros(n_bars, dtype=bool),
        "is_tail_event": np.zeros(n_bars, dtype=bool),
        "insufficient_history": np.array([True] * 20 + [False] * (n_bars - 20), dtype=bool),
        "trend_score": [np.nan] * 20 + list(np.random.normal(0.2, 0.5, n_bars - 20)),
        "p_trend": [np.nan] * 20 + list(np.random.uniform(0.4, 0.8, n_bars - 20)),
        "p_chop": [np.nan] * 20 + list(np.random.uniform(0.2, 0.6, n_bars - 20)),
        "atr_14": [np.nan] * 20 + list(np.random.uniform(0.5, 1.5, n_bars - 20)),
        "hurst_value": [np.nan] * 20 + list(np.random.uniform(0.4, 0.6, n_bars - 20)),
        "d_star_used": [0.35] * n_bars,
    })

    research = ResearchPipeline(config={"fast_mode": True, "n_groups": 3, "n_test_groups": 1, "embargo_bars": 2})
    research.run(signal_bars=bars, artifacts_dir=temp_dir)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_live_pipeline_streaming_flow(fitted_artifacts_dir):
    live_pipeline = AegisLivePipeline.from_artifacts(
        artifacts_dir=fitted_artifacts_dir,
        initial_capital=10000.0,
        config={"max_safe_leverage": 2.0},
    )

    assert live_pipeline.account_tracker.wallet_balance == 10000.0
    assert live_pipeline.account_tracker.available_margin == 10000.0
    assert live_pipeline.current_position is None

    # Tạo chuỗi bar streaming
    base_time = 1700010000000
    for i in range(15):
        price = 100.0 + (i * 0.5 if i < 8 else (8 * 0.5 - (i - 8) * 1.5))
        bar = {
            "bar_idx": 100 + i,
            "symbol": "BTCUSDT",
            "timestamp_ms": base_time + i * 60000,
            "open": price,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
            "volume": 50.0,
            "ofi": 0.2,
            "tick_count": 25,
            "is_toxic_flag": False,
            "is_tail_event": False,
            "insufficient_history": False,
            "trend_score": 0.4,
            "p_trend": 0.7,
            "p_chop": 0.3,
            "atr_14": 1.0,
            "hurst_value": 0.55,
            "d_star_used": 0.35,
            "hl_spread": 0.006,
            "ret_1": 0.005,
            "ret_5": 0.01,
            "vol_ratio": 1.0,
        }
        res = live_pipeline.on_bar(bar)
        assert res is not None

    # Sau 15 bars, hoặc vị thế đang mở hoặc đã chốt lời/cắt lỗ
    assert live_pipeline.account_tracker.wallet_balance > 0.0
    assert live_pipeline.circuit_breaker.update_equity(
        live_pipeline.account_tracker.wallet_balance,
        current_time_ms=base_time + 15 * 60000,
    ).tier != CircuitBreakerTier.TIER_3_KILL
