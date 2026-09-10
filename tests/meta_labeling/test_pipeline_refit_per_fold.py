"""
[FIX F5 / F4] Test refit-per-fold CHẠY PIPELINE THẬT.

Bản trước của file này dùng MockEstimator và tự thú ngay trong docstring rằng nó
"không chạy pipeline thật". Chính vì vậy lỗ hổng F5 sống sót: CPCVPipeline bỏ trống
`train_idx`, không huấn luyện gì cả, mà vẫn có một test màu xanh mang tên
"refit_per_fold" đứng gác.

Test này gọi thẳng CPCVPipeline và kiểm tra hành vi thật.
"""
import warnings

import numpy as np
import pytest

import main as aegis_main
from aegis.core.config_loader import load_canonical_config
from aegis.pipelines.cpcv_pipeline import CPCVPipeline


@pytest.fixture(scope="module")
def cpcv_result():
    """Chạy CPCV trên chuỗi đủ dài để mọi fold có đủ mẫu train sau purge."""
    bars = aegis_main.generate_synthetic_signal_bars(4000)
    config = load_canonical_config()
    config["fast_mode"] = True
    pipe = CPCVPipeline(n_groups=5, n_test_groups=2, embargo_bars=24, config=config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pipe.run(bars, num_bins=5)
    return pipe, result, bars


def test_cpcv_actually_fits_a_model_per_fold(cpcv_result):
    """
    LÕI CỦA F5: `train_idx` phải thực sự được dùng để huấn luyện.
    Trước khi vá, số model fit được là 0 — "OOS" hoàn toàn hư cấu.
    """
    pipe, result, _ = cpcv_result
    n_fits = result["metrics"]["n_fits"]

    assert n_fits > 0, (
        "CPCV không fit model nào — mọi chỉ số OOS đều vô nghĩa (đây chính là F5)"
    )
    assert n_fits == pipe.n_fits
    assert result["metrics"]["n_folds_starved"] == 0, (
        "Với 4000 nến mọi fold phải đủ mẫu train; nếu đói dữ liệu thì test này vô nghĩa"
    )


def test_kelly_table_indexed_by_model_probability_not_p_trend(cpcv_result):
    """
    LÕI CỦA F4: p_i dùng để dựng bảng Kelly phải là xác suất của MODEL,
    đúng đại lượng mà live_pipeline dùng để tra bảng — không phải p_trend (HMM).
    """
    _, result, bars = cpcv_result
    records = result["clean_records"]
    assert records, "Cần có lệnh sạch để kiểm tra"

    p_i = np.round(np.array([r["p_i"] for r in records], dtype=np.float64), 6)
    p_trend = bars["p_trend"].to_numpy()
    p_trend = np.round(p_trend[~np.isnan(p_trend)], 6)

    assert not set(p_i).issubset(set(p_trend)), (
        "p_i trong bảng Kelly vẫn là p_trend (posterior HMM) — lỗi phạm trù F4 còn nguyên"
    )
    assert np.all((p_i >= 0.0) & (p_i <= 1.0)), "p_i phải là xác suất hợp lệ"


def test_purge_horizon_covers_full_holding_period(cpcv_result):
    """LÕI CỦA F16: purge phải phủ trọn thời gian nắm giữ tối đa, không phải 10 nến."""
    _, result, _ = cpcv_result
    metrics = result["metrics"]

    assert metrics["purge_horizon"] >= 120, (
        f"purge_horizon={metrics['purge_horizon']} nhỏ hơn t_max_live_follow=120 -> rò rỉ"
    )
    assert metrics["truncation_rate"] < 0.15, (
        f"Tỷ lệ cắt biên {metrics['truncation_rate']:.1%} vượt ngưỡng an toàn 15%"
    )


def test_sharpe_annualized_by_real_trade_frequency(cpcv_result):
    """LÕI CỦA F15: Sharpe quy năm bằng tần suất thật, không phải hằng số 252."""
    _, result, _ = cpcv_result
    metrics = result["metrics"]

    assert metrics["trades_per_year"] > 0.0, "Phải đo được tần suất giao dịch thật"

    returns = np.array(
        [r["realized_return"] for r in result["trade_records"]], dtype=np.float64
    )
    expected = (returns.mean() / (returns.std(ddof=1) + 1e-12)) * np.sqrt(
        metrics["trades_per_year"]
    )
    assert metrics["sharpe_oos"] == pytest.approx(expected, rel=1e-9)


def test_starved_folds_raise_a_warning():
    """
    Dữ liệu quá ngắn PHẢI cảnh báo to, không được lặng lẽ xuất bảng Kelly rác.
    Purge đúng (120 nến) ăn nhiều mẫu — bản cũ purge 10 nến nên che giấu điều này.
    """
    bars = aegis_main.generate_synthetic_signal_bars(600)
    config = load_canonical_config()
    config["fast_mode"] = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CPCVPipeline(n_groups=5, n_test_groups=2, embargo_bars=24, config=config).run(
            bars, num_bins=5
        )

    messages = " ".join(str(w.message) for w in caught)
    assert "ĐÓI DỮ LIỆU" in messages, (
        "Fold bị đói dữ liệu phải phát cảnh báo rõ ràng cho vận hành viên"
    )
