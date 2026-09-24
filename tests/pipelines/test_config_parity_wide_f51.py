"""
[FIX F51] Parity CẤU HÌNH giữa research và file JSON daemon thật sự chạy.

Tách khỏi `test_research_live_parity_v3.py` có chủ ý: file đó bị skip CẢ MODULE khi
thiếu `data/` (`pytestmark`), kéo theo cả các test cấu hình vốn không cần dữ liệu giá.
Một test parity chỉ chạy trên máy có dữ liệu là test không canh được gì ở nơi code
được sửa.
"""
import pytest

from aegis.research.strategy_v3 import V3_SIGNALS


def test_cau_hinh_wide_ma_daemon_chay_trung_research_f51():
    """
    Test cấu hình cũ (`test_research_live_parity_v3.py`) chỉ kiểm `strategy_v3.json` (n=12),
    file daemon KHÔNG còn chạy.
    Daemon chạy `strategy_v3_wide.json` (n=50), và file đó ghi `max_step=0,025` trong
    khi mọi nghiên cứu cấu hình wide dùng tầng gộp của `V3` (`max_step=0,05`). Không
    test nào đứng giữa hai bên nên khe hở lọt qua.

    Nay nghiên cứu có `config_from_json` — dựng cấu hình từ ĐÚNG file live nạp. Test này
    khoá rằng hai phía đọc ra cùng một chiến lược, tham số nào cũng vậy.
    """
    import pathlib

    from aegis.pipelines.xs_live_pipeline import LiveConfig
    from aegis.research.strategy_v3 import WIDE_CONFIG_FILE, config_from_json

    if not pathlib.Path(WIDE_CONFIG_FILE).is_file():
        pytest.skip(f"chưa có {WIDE_CONFIG_FILE}")
    live = LiveConfig.from_artifacts(WIDE_CONFIG_FILE)
    res = config_from_json(WIDE_CONFIG_FILE)

    assert live.engine == "v3"
    assert live.n_positions == res.n_positions == res.portfolio.n_positions == 50
    assert live.rebalance_bars == res.rebalance_every
    # Chia lô: mỗi lô giữ `period_hours`, cả hệ tái cân bằng mỗi period/K giờ.
    assert live.n_tranches == res.n_tranches
    assert live.rebalance_hours == pytest.approx(res.period_hours / res.n_tranches)
    assert live.interval == res.interval
    assert live.weight_mode == res.portfolio.mode
    assert live.max_weight == pytest.approx(res.portfolio.max_weight)
    assert live.combiner_lookback == res.combiner.lookback
    assert live.combiner_min_periods == res.combiner.min_periods
    assert live.combiner_t_threshold == pytest.approx(res.combiner.t_threshold)
    assert live.combiner_max_abs_weight == pytest.approx(res.combiner.max_abs_weight)
    assert live.combiner_max_step == pytest.approx(res.combiner.max_step)
    assert live.top_frac == pytest.approx(res.top_frac)
    # Live dựng PortfolioSpec với beta_neutral=False cố định (xs_live_pipeline).
    assert res.portfolio.beta_neutral is False
    live_names = tuple(live.v3_signals) if live.v3_signals else V3_SIGNALS
    assert live_names == res.signal_names()
