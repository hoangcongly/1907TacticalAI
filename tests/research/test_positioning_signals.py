"""
Họ tín hiệu VỊ THẾ — và bất biến quan trọng nhất: thêm nó KHÔNG được đụng tới V3.

Hai nhóm test:
  * Họ mới tự TẮT sạch khi chưa có dữ liệu — hệ thống phải chạy y hệt như trước khi
    có nó, không crash, không điền bừa.
  * Danh sách tín hiệu của V3 bị ĐÓNG BĂNG. `adaptive_weights` chuẩn hoá theo SỐ HỌ
    (`equal = 1/n`), nên thêm một họ — kể cả họ toàn NaN không bao giờ nhận trọng số —
    vẫn làm dịch kết quả của giai đoạn warm-up. Bất biến này là thứ giữ cho
    `artifacts/strategy_v3.json` còn tái lập được.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.data.panel_v2 import METRICS_FIELDS, load_metrics_panel_v2
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.research.strategy_v3 import V3, V3_SIGNALS, StrategyV3Config

POS = [n for n, s in SIGNAL_REGISTRY.items() if s.family == "positioning"]


@pytest.fixture
def panel():
    n, k = 600, 8
    rng = np.random.default_rng(0)
    idx = np.arange(n, dtype=np.int64) * 14_400_000
    cols = [f"S{i}" for i in range(k)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, k)), 0), idx, cols)
    return {
        "close": close,
        "high": close * 1.01, "low": close * 0.99, "open": close,
        "volume": pd.DataFrame(rng.lognormal(10, 1, (n, k)), idx, cols),
        "ofi": pd.DataFrame(rng.normal(0, 0.3, (n, k)), idx, cols),
        "tick_count": pd.DataFrame(rng.integers(100, 900, (n, k)), idx, cols),
    }


@pytest.fixture
def metrics(panel):
    n, k = panel["close"].shape
    rng = np.random.default_rng(1)
    idx, cols = panel["close"].index, panel["close"].columns
    oi = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (n, k)), 0) * 1e8, idx, cols)
    return {
        "sum_open_interest": oi / 100.0,
        "sum_open_interest_value": oi,
        "count_toptrader_long_short_ratio": pd.DataFrame(rng.lognormal(0, 0.2, (n, k)), idx, cols),
        "sum_toptrader_long_short_ratio": pd.DataFrame(rng.lognormal(0, 0.3, (n, k)), idx, cols),
        "count_long_short_ratio": pd.DataFrame(rng.lognormal(0, 0.25, (n, k)), idx, cols),
        "sum_taker_long_short_vol_ratio": pd.DataFrame(rng.lognormal(0, 0.15, (n, k)), idx, cols),
    }


# ---------------------------------------------------------------------------
# Tự tắt khi thiếu dữ liệu
# ---------------------------------------------------------------------------
def test_thieu_du_lieu_thi_tra_nan_chu_khong_crash(panel):
    """
    Chưa tải metrics là trạng thái BÌNH THƯỜNG, không phải lỗi. Họ này phải tự tắt.

    Điền 0 thay vì NaN sẽ tệ hơn crash: một cặp không có dữ liệu vị thế sẽ được xếp
    hạng "trung tính" và lọt vào danh mục dựa trên thông tin không tồn tại.
    """
    for n in POS:
        out = build_signal(n, panel, pd.DataFrame(index=panel["close"].index))
        assert out.shape == panel["close"].shape
        assert out.isna().all().all(), f"{n} phải toàn NaN khi thiếu metrics"


def test_co_du_lieu_thi_sinh_ra_gia_tri(panel, metrics):
    p = dict(panel); p.update(metrics)
    fund = pd.DataFrame(0.0001, index=panel["close"].index, columns=panel["close"].columns)
    for n in POS:
        out = build_signal(n, p, fund)
        frac = float(out.notna().mean().mean())
        assert frac > 0.3, f"{n} chỉ có {frac:.1%} ô có giá trị"


def test_nhan_qua_khong_nhin_truoc(panel, metrics):
    """
    Đổi dữ liệu SAU mốc t không được làm đổi tín hiệu TẠI t. Đây là bất biến duy nhất
    ngăn một tín hiệu trông tuyệt vời trên giấy và vô dụng khi chạy thật.
    """
    p = dict(panel); p.update(metrics)
    fund = pd.DataFrame(0.0001, index=panel["close"].index, columns=panel["close"].columns)
    cut = 400
    p2 = {k: v.copy() for k, v in p.items()}
    rng = np.random.default_rng(9)
    for k in metrics:
        p2[k].iloc[cut:] = rng.lognormal(0, 1, p2[k].iloc[cut:].shape) * 1e6

    for n in POS:
        a = build_signal(n, p, fund).iloc[:cut]
        b = build_signal(n, p2, fund).iloc[:cut]
        # xs_zscore chuẩn hoá theo hàng nên chỉ so các hàng đều có dữ liệu
        m = a.notna() & b.notna()
        if m.to_numpy().sum() == 0:
            continue
        assert np.allclose(a.to_numpy()[m.to_numpy()], b.to_numpy()[m.to_numpy()],
                           atol=1e-9), f"{n} nhìn trước tương lai"


def test_loader_thieu_thu_muc_thi_tra_khung_rong(tmp_path):
    idx = pd.Index(np.arange(50, dtype=np.int64) * 14_400_000)
    out = load_metrics_panel_v2(["AAAUSDT"], idx, root=str(tmp_path / "khong_co"))
    assert set(out) == set(METRICS_FIELDS)
    for f in METRICS_FIELDS:
        assert out[f].empty or out[f].isna().all().all()


def test_loader_khong_ffill(tmp_path):
    """
    Thiếu là THIẾU. ffill sẽ khiến một cặp ngừng báo cáo vẫn có 'vị thế' y như lần
    cuối, vĩnh viễn — và tín hiệu sẽ giao dịch trên một con số đã chết.
    """
    idx = np.arange(20, dtype=np.int64) * 14_400_000
    df = pd.DataFrame({"timestamp_ms": idx[:5],
                       **{f: np.arange(5, dtype=float) + 1 for f in METRICS_FIELDS}})
    df.to_parquet(tmp_path / "AAAUSDT_4h.parquet", index=False)
    out = load_metrics_panel_v2(["AAAUSDT"], pd.Index(idx), root=str(tmp_path))
    col = out["sum_open_interest"]["AAAUSDT"]
    assert col.iloc[:5].notna().all()
    assert col.iloc[5:].isna().all(), "không được điền tới"


# ---------------------------------------------------------------------------
# V3 bị đóng băng
# ---------------------------------------------------------------------------
def test_v3_van_dung_dung_26_tin_hieu():
    assert len(V3.signal_names()) == 26
    assert V3.signal_names() == V3_SIGNALS


def test_them_tin_hieu_vao_registry_khong_doi_v3():
    """
    Registry có 36 tín hiệu nhưng V3 phải vẫn thấy đúng 26. Nếu bất biến này gãy thì
    `artifacts/strategy_v3.json` không còn tái lập được, và ta mất khả năng phân biệt
    'code hỏng' với 'ai đó thêm một tín hiệu'.
    """
    assert len(SIGNAL_REGISTRY) > 26, "test này vô nghĩa nếu registry không rộng hơn V3"
    assert len(V3.signal_names()) == 26
    assert all(n in SIGNAL_REGISTRY for n in V3.signal_names())


def test_cau_hinh_mo_rong_thi_dung_bo_moi():
    from dataclasses import replace
    ext = replace(V3, signals=tuple(V3_SIGNALS) + tuple(POS))
    assert len(ext.signal_names()) == 26 + len(POS)
    assert len(V3.signal_names()) == 26, "cấu hình mới không được lây sang V3"
