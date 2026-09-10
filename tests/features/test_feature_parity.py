"""
[FIX F3] Test parity research <-> live.

Đây là test QUAN TRỌNG NHẤT của toàn hệ thống đặc trưng: nó chốt bất biến I2
(cùng một code tính feature cho train lẫn inference). Lỗ hổng F3 tồn tại được
chính vì trước đây không có test nào như thế này.

Nếu test này đỏ, TUYỆT ĐỐI không đem hệ thống đi giao dịch.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.features.feature_engine import (
    MissingFeatureError,
    OnlineFeatureEngine,
    batch_features,
)
from aegis.features.feature_spec import (
    CANDIDATE_FEATURES,
    DERIVED_BY_NAME,
    PASSTHROUGH_FEATURES,
)


def _make_bars(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    return pd.DataFrame({
        "high": close + np.abs(rng.normal(0.4, 0.2, n)),
        "low": close - np.abs(rng.normal(0.4, 0.2, n)),
        "close": close,
        "volume": rng.uniform(1.0, 500.0, n),
        "ofi": rng.uniform(-1.0, 1.0, n),
    })


def test_online_matches_batch_elementwise():
    """
    Bất biến I2: streaming phải cho kết quả TRÙNG KHỚP với batch, từng phần tử.
    Đây là thứ ngăn lỗ hổng F3 tái phát.
    """
    bars = _make_bars()
    batch = batch_features(bars)

    engine = OnlineFeatureEngine()
    for i in range(len(bars)):
        online = engine.update(bars.iloc[i].to_dict())
        for name in DERIVED_BY_NAME:
            assert online[name] == pytest.approx(
                float(batch[name].iloc[i]), rel=1e-12, abs=1e-12
            ), f"Lệch parity tại nến {i}, đặc trưng `{name}`"


def test_online_is_causal():
    """
    Đặc trưng tại nến t không được đổi khi các nến TƯƠNG LAI xuất hiện.
    Nếu đổi, tức là có rò rỉ nhìn trước (look-ahead).
    """
    bars = _make_bars(n=120)

    short = batch_features(bars.iloc[:60])
    full = batch_features(bars)

    for name in DERIVED_BY_NAME:
        np.testing.assert_allclose(
            short[name].to_numpy(),
            full[name].to_numpy()[:60],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"`{name}` thay đổi khi có thêm dữ liệu tương lai -> rò rỉ nhân quả",
        )


def test_build_vector_respects_selected_feature_order():
    """Vector đầu vào phải theo ĐÚNG thứ tự selected_features model đã học."""
    bars = _make_bars(n=40)
    engine = OnlineFeatureEngine()
    for i in range(len(bars) - 1):
        engine.update(bars.iloc[i].to_dict())

    last = bars.iloc[-1].to_dict()
    last.update({"atr_14": 1.5})
    engine.update(last)

    order = ["ofi", "hl_spread", "atr_14", "ret_1"]
    vec = engine.build_vector(last, order)

    assert vec.shape == (1, 4)
    assert vec[0, 0] == pytest.approx(float(last["ofi"]))
    assert vec[0, 2] == pytest.approx(1.5)

    reordered = OnlineFeatureEngine()
    for i in range(len(bars) - 1):
        reordered.update(bars.iloc[i].to_dict())
    reordered.update(last)
    vec2 = reordered.build_vector(last, ["hl_spread", "ofi", "ret_1", "atr_14"])
    assert vec2[0, 0] == pytest.approx(vec[0, 1])
    assert vec2[0, 1] == pytest.approx(vec[0, 0])


def test_missing_feature_raises_instead_of_silent_zero():
    """
    LÕI CỦA F3: thiếu đặc trưng phải NÉM LỖI, tuyệt đối không im lặng điền 0.0.

    Bản cũ dùng `bar.get(col, 0.0)` khiến 4/5 đặc trưng của model bằng 0.0 suốt
    phiên live mà không có cảnh báo nào.
    """
    bars = _make_bars(n=30)
    engine = OnlineFeatureEngine()
    for i in range(len(bars) - 1):
        engine.update(bars.iloc[i].to_dict())

    last = bars.iloc[-1].to_dict()  # không có `atr_14`
    engine.update(last)
    with pytest.raises(MissingFeatureError, match="atr_14"):
        engine.build_vector(last, ["hl_spread", "atr_14"])


def test_nan_passthrough_feature_is_rejected():
    """Đặc trưng passthrough bằng NaN cũng phải bị từ chối, không được thành 0.0."""
    bars = _make_bars(n=30)
    engine = OnlineFeatureEngine()
    for i in range(len(bars) - 1):
        engine.update(bars.iloc[i].to_dict())

    last = bars.iloc[-1].to_dict()
    last["atr_14"] = float("nan")
    engine.update(last)
    with pytest.raises(MissingFeatureError):
        engine.build_vector(last, ["atr_14"])


def test_missing_raw_column_raises():
    """Thiếu cột thô (high/low/close/volume) phải lỗi ngay, không tính bừa."""
    with pytest.raises(MissingFeatureError, match="volume"):
        batch_features(_make_bars(n=10).drop(columns=["volume"]))


def test_candidate_features_cover_spec():
    """Danh sách ứng viên phải khớp đăng ký — chống lệch khi thêm feature mới."""
    assert set(CANDIDATE_FEATURES) == set(DERIVED_BY_NAME) | set(PASSTHROUGH_FEATURES)
