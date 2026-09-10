"""
[FIX F6] Test phân loại chế độ bằng Efficiency Ratio.

Thay thế HMM có tham số hardcode chưa bao giờ được fit. Điểm mấu chốt cần chốt lại:
phân loại phải TỰ CHUẨN HOÁ (không cần hiệu chỉnh theo tài sản/khung thời gian) —
đó chính là thứ tham số bịa `means=[0.001,0]`, `stds=[0.02,0.005]` không làm được.
"""
import numpy as np
import pytest

from aegis.features.regime.efficiency_regime import CausalEfficiencyRegime


def test_pure_trend_gives_high_p_trend():
    """Giá đi thẳng một mạch -> p_trend tiến về 1.0."""
    prices = np.arange(100.0, 200.0, 1.0)
    posteriors = CausalEfficiencyRegime().filter_series(prices)
    assert posteriors[-1, 0] > 0.95


def test_pure_chop_gives_low_p_trend():
    """Giá đi tới đi lui triệt tiêu -> p_trend tiến về 0.0."""
    prices = 100.0 + np.tile([0.0, 2.0, 0.0, -2.0], 30)
    posteriors = CausalEfficiencyRegime().filter_series(prices)
    assert posteriors[-1, 0] < 0.15


def test_posteriors_always_sum_to_one():
    """Hợp đồng dữ liệu n_states=2: p_trend + p_chop == 1.0 tuyệt đối."""
    rng = np.random.default_rng(3)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, 500))
    posteriors = CausalEfficiencyRegime().filter_series(prices)
    np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all((posteriors >= 0.0) & (posteriors <= 1.0))


def test_scale_invariance():
    """
    ĐIỂM CỐT LÕI CỦA F6: kết quả không được đổi khi nhân thang giá.

    HMM cũ hardcode stds=[0.02, 0.005] nên gắn chặt vào một mức biến động tuyệt đối;
    đổi tài sản hoặc khung thời gian là sai hoàn toàn. Efficiency Ratio là tỷ số nên
    bất biến theo thang.
    """
    rng = np.random.default_rng(11)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, 300))

    base = CausalEfficiencyRegime().filter_series(prices)
    scaled = CausalEfficiencyRegime().filter_series(prices * 1000.0)

    np.testing.assert_allclose(base, scaled, rtol=1e-9, atol=1e-12)


def test_causality_no_lookahead():
    """Posterior tại nến t không đổi khi thêm dữ liệu tương lai."""
    rng = np.random.default_rng(5)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, 200))

    short = CausalEfficiencyRegime().filter_series(prices[:100])
    full = CausalEfficiencyRegime().filter_series(prices)

    np.testing.assert_allclose(short, full[:100], rtol=1e-12, atol=1e-12)


def test_no_fitted_parameters_needed():
    """
    Không cần bước fit nào: khởi tạo xong là dùng được ngay.
    Đây là lý do nó không thể lặp lại lỗi "tham số chưa bao giờ được fit" của HMM.
    """
    regime = CausalEfficiencyRegime()
    p_trend, p_chop = regime.step(100.0)
    assert p_trend == pytest.approx(0.5)
    assert p_chop == pytest.approx(0.5)


def test_garbage_price_does_not_fabricate_signal():
    """Giá NaN/Inf phải giữ nguyên posterior gần nhất, không bịa ra tín hiệu mới."""
    regime = CausalEfficiencyRegime()
    for price in np.arange(100.0, 130.0, 1.0):
        regime.step(float(price))
    before = regime.step(130.0)

    assert regime.step(float("nan")) == pytest.approx(before)
    assert regime.step(float("inf")) == pytest.approx(before)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        CausalEfficiencyRegime(window=1)
    with pytest.raises(ValueError):
        CausalEfficiencyRegime(smoothing=0.0)
    with pytest.raises(ValueError):
        CausalEfficiencyRegime(smoothing=1.5)
