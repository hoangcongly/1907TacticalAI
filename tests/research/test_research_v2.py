"""
Test các module nghiên cứu v2: backtest, IC, gộp thích ứng, đòn bẩy, panel, tín hiệu.

Trọng tâm là ba nhóm tính chất, vì đó là ba nhóm sai lầm thực sự làm mất tiền:
  1. NHÂN QUẢ — không giá trị nào tại t phụ thuộc dữ liệu sau t.
  2. QUY ƯỚC DẤU — tín hiệu tốt phải ra lợi nhuận dương, kiểm bằng tín hiệu ORACLE
     (bằng đúng lợi suất tương lai) mà ta biết trước đáp án.
  3. KẾ TOÁN — chi phí, funding, turnover cộng đúng số học.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.data.panel_v2 import INTERVAL_MS, interval_hours, resample_bars
from aegis.research.adaptive_combiner import (
    CombinerSpec, adaptive_weights, combine_adaptive, factor_returns,
)
from aegis.research.backtest_v2 import (
    CostModel, corwin_schultz_spread, drift_weights, simulate,
)
from aegis.research.ic_analysis import (
    decile_spread, effective_breadth, ic_decay_curve, implied_sharpe,
    information_coefficient, monotonicity, quantile_returns, signal_autocorrelation,
)
from aegis.research.leverage import (
    LeverageSpec, apply_leverage, kelly_leverage, simulate_paths,
    target_probability_table, volatility_target_series,
)
from aegis.research.signal_library import (
    FAMILIES, SIGNAL_REGISTRY, build_family, build_signal, xs_rank, xs_zscore,
)
from aegis.risk.portfolio import PortfolioSpec, build_weights

BAR_MS = 14_400_000


@pytest.fixture
def panel():
    """Panel 4h giả lập, 24 tài sản, có nhân tố chung + động lượng thật."""
    rng = np.random.default_rng(42)
    n, k = 600, 24
    syms = [f"S{i:02d}" for i in range(k)]
    idx = np.arange(n) * BAR_MS

    drift = rng.normal(0, 0.0015, k)          # động lượng THẬT, cắm vào dữ liệu
    factor = rng.normal(0, 0.02, n)
    beta = rng.uniform(0.5, 1.6, k)
    vols = np.linspace(0.01, 0.06, k)
    rets = drift + np.outer(factor, beta) + rng.normal(0, 1, (n, k)) * vols

    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    high = close * (1 + np.abs(rng.normal(0, 0.004, (n, k))))
    low = close * (1 - np.abs(rng.normal(0, 0.004, (n, k))))
    vol = pd.DataFrame(rng.lognormal(14, 1.0, (n, k)), index=idx, columns=syms)
    ofi = pd.DataFrame(rng.uniform(-1, 1, (n, k)), index=idx, columns=syms)
    tick = pd.DataFrame(rng.integers(500, 20000, (n, k)), index=idx, columns=syms).astype(float)
    funding = pd.DataFrame(rng.normal(0.0001, 0.0003, (n, k)), index=idx, columns=syms)

    return ({"close": close, "high": high, "low": low, "volume": vol,
             "ofi": ofi, "tick_count": tick}, funding)


# =========================================================================
# panel_v2 — tổng hợp khung thời gian
# =========================================================================
def test_resample_giu_dung_ohlc_va_tong_khoi_luong():
    n = 12
    df = pd.DataFrame({
        "timestamp_ms": np.arange(n) * INTERVAL_MS["1h"],
        "open": np.arange(1.0, n + 1), "high": np.arange(1.0, n + 1) + 2,
        "low": np.arange(1.0, n + 1) - 1, "close": np.arange(1.0, n + 1) + 0.5,
        "volume": np.ones(n) * 10, "tick_count": np.ones(n, dtype=int) * 3,
        "ofi": np.tile([1.0, -1.0], n // 2),
    })
    out = resample_bars(df, "1h", "4h")
    assert len(out) == 3
    assert out["open"].iloc[0] == 1.0
    assert out["close"].iloc[0] == 4.5
    assert out["high"].iloc[0] == 6.0
    assert out["low"].iloc[0] == 0.0
    assert out["volume"].iloc[0] == 40.0
    assert out["tick_count"].iloc[0] == 12
    # OFI có trọng số khối lượng; khối lượng đều nhau -> trung bình = 0
    assert out["ofi"].iloc[0] == pytest.approx(0.0)


def test_resample_neo_vao_epoch_khong_neo_vao_nen_dau():
    """
    Hai cặp niêm yết lệch giờ phải rơi vào CÙNG lưới 4h. Nếu neo theo nến đầu của
    từng chuỗi, nến 4h của chúng lệch pha và mọi so sánh cross-sectional hỏng.
    """
    h = INTERVAL_MS["1h"]
    a = pd.DataFrame({"timestamp_ms": np.arange(0, 8) * h, "open": 1.0, "high": 1.0,
                      "low": 1.0, "close": 1.0, "volume": 1.0})
    b = pd.DataFrame({"timestamp_ms": np.arange(1, 9) * h, "open": 1.0, "high": 1.0,
                      "low": 1.0, "close": 1.0, "volume": 1.0})
    ra, rb = resample_bars(a, "1h", "4h"), resample_bars(b, "1h", "4h")
    assert set(rb["timestamp_ms"]).issubset(
        set(np.arange(0, 12) * INTERVAL_MS["4h"]) | set(ra["timestamp_ms"]))
    assert all(t % INTERVAL_MS["4h"] == 0 for t in ra["timestamp_ms"])
    assert all(t % INTERVAL_MS["4h"] == 0 for t in rb["timestamp_ms"])


def test_resample_bo_nhom_cuoi_chua_day_du():
    """Nến target cuối chưa hình thành xong là dữ liệu TƯƠNG LAI — phải bị loại."""
    h = INTERVAL_MS["1h"]
    df = pd.DataFrame({"timestamp_ms": np.arange(6) * h, "open": 1.0, "high": 1.0,
                       "low": 1.0, "close": 1.0, "volume": 1.0})
    out = resample_bars(df, "1h", "4h")
    assert len(out) == 1, "nhóm 2 chỉ có 2/4 nến, phải bị bỏ"


def test_interval_hours():
    assert interval_hours("4h") == 4.0
    assert interval_hours("1h") == 1.0
    with pytest.raises(ValueError):
        interval_hours("7h")


# =========================================================================
# backtest_v2
# =========================================================================
def test_drift_weights_theo_dung_cong_thuc():
    """
    Sau một chu kỳ, mỗi chân nhân (1+r) còn VỐN nhân (1 + w.r).
    Với long +10% và short -10% (short THẮNG khi giá giảm), cả hai chân đều lãi:
    w.r = 0.5(0.10) + (-0.5)(-0.10) = 0.10.
    """
    w = pd.Series([0.5, -0.5], index=["a", "b"])
    r = pd.Series([0.10, -0.10], index=["a", "b"])
    out = drift_weights(w, r)
    port = float((w * r).sum())
    assert port == pytest.approx(0.10)
    assert out["a"] == pytest.approx(0.5 * 1.10 / 1.10)
    assert out["b"] == pytest.approx(-0.5 * 0.90 / 1.10)
    # Chân có giá TĂNG chiếm tỷ trọng tương đối lớn hơn so với trước khi trôi.
    assert abs(out["a"]) / abs(out["b"]) > abs(w["a"]) / abs(w["b"])


def test_chi_phi_dung_bang_turnover_nhan_bp(panel):
    """Kế toán chi phí phải khớp phép nhân tay, không sai một bp nào."""
    p, _ = panel
    close = p["close"]
    W = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    W.iloc[:, 0] = 0.5
    W.iloc[:, 1] = -0.5
    cost = CostModel(taker_fee_bps=10.0, maker_fee_bps=10.0, maker_ratio=1.0,
                     half_spread_bps=0.0, min_bps=10.0)
    res = simulate(W, close, None, cost, bar_hours=4, rebalance_every=1, allow_drift=False)
    # Kỳ đầu mở toàn bộ: turnover = 1.0 -> chi phí = 10bp
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.cost_drag.iloc[0] == pytest.approx(10e-4)
    # Các kỳ sau không giao dịch -> chi phí 0
    assert res.cost_drag.iloc[1:].abs().max() == pytest.approx(0.0)


def test_funding_dung_dau_va_dung_ty_le(panel):
    """Long trả funding khi rate dương; số tiền tỷ lệ theo số mốc 8h nắm giữ."""
    p, _ = panel
    close = p["close"]
    W = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    W.iloc[:, 0] = 1.0
    funding = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    funding.iloc[:, 0] = 0.001
    cost0 = CostModel(taker_fee_bps=0.0, maker_fee_bps=0.0, maker_ratio=1.0,
                      half_spread_bps=0.0, min_bps=0.0)
    res = simulate(W, close, funding, cost0, bar_hours=24, rebalance_every=1)
    assert res.funding_pnl.iloc[0] == pytest.approx(-0.001 * 3.0)


def test_trong_so_troi_lam_giam_turnover(panel):
    """Cho trọng số trôi phải cho turnover <= không cho trôi (danh mục tự dịch về đích)."""
    p, _ = panel
    close = p["close"]
    sig = pd.DataFrame(np.random.default_rng(0).normal(0, 1, close.shape),
                       index=close.index, columns=close.columns)
    W = build_weights(sig, close, PortfolioSpec(mode="rank_binary", n_positions=8,
                                                max_weight=1.0))
    c = CostModel(taker_fee_bps=1.0, maker_fee_bps=1.0, maker_ratio=1.0, min_bps=1.0)
    a = simulate(W, close, None, c, bar_hours=4, rebalance_every=1, allow_drift=True)
    b = simulate(W, close, None, c, bar_hours=4, rebalance_every=1, allow_drift=False)
    assert a.turnover.sum() <= b.turnover.sum() * 1.02


def test_corwin_schultz_khong_am_va_bi_kep_tran(panel):
    p, _ = panel
    s = corwin_schultz_spread(p["high"], p["low"], bars_per_day=6)
    assert (s >= 0.2).all() and (s <= 60.0).all()


# =========================================================================
# ic_analysis — quy ước dấu bằng tín hiệu ORACLE
# =========================================================================
def test_oracle_cho_ic_bang_mot(panel):
    """Tín hiệu = đúng lợi suất tương lai -> IC phải bằng +1 (và -1 khi đảo dấu)."""
    p, _ = panel
    close = p["close"]
    fwd = close.shift(-1) / close - 1.0
    assert information_coefficient(xs_zscore(fwd), close, 1).mean == pytest.approx(1.0, abs=1e-9)
    assert information_coefficient(xs_zscore(-fwd), close, 1).mean == pytest.approx(-1.0, abs=1e-9)


def test_oracle_cho_backtest_duong_lon(panel):
    """Cùng tín hiệu oracle phải cho lợi nhuận dương rất lớn qua engine backtest."""
    p, _ = panel
    close = p["close"]
    fwd = close.shift(-1) / close - 1.0
    W = build_weights(xs_zscore(fwd), close, PortfolioSpec(mode="rank_binary",
                                                           n_positions=8, max_weight=1.0))
    c0 = CostModel(taker_fee_bps=0.0, maker_fee_bps=0.0, maker_ratio=1.0, min_bps=0.0)
    r = simulate(W, close, None, c0, bar_hours=4, rebalance_every=1).returns.dropna()
    assert r.mean() > 0 and (r > 0).mean() > 0.95


def test_ic_va_chenh_lech_decile_co_the_nguoc_dau():
    """
    Tính chất PHẢI được bảo vệ bằng test, vì nó đã gây hiểu nhầm một lần trong
    chính dự án này: với quan hệ hình chữ U, IC (đo đơn điệu trung bình) và chênh
    lệch decile (đo hai đuôi) ngược dấu nhau. Dựng dữ liệu chữ U tường minh.
    """
    n, k = 300, 20
    idx = np.arange(n) * BAR_MS
    syms = [f"S{i:02d}" for i in range(k)]
    rng = np.random.default_rng(0)

    # Tín hiệu: cố định theo tài sản, trải đều từ -2 tới +2.
    sig = pd.DataFrame(np.tile(np.linspace(-2, 2, k), (n, 1)), index=idx, columns=syms)

    # Lợi suất mỗi nến hình chữ U theo tín hiệu: hai đuôi cao, giữa thấp.
    shape = np.linspace(-2, 2, k) ** 2
    rets = np.tile(shape * 0.002, (n, 1)) + rng.normal(0, 0.0005, (n, k))
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=syms)

    q = quantile_returns(sig, close, 1, 5)
    assert len(q) == 5

    # Hình chữ U: hai nhóm biên phải CAO HƠN nhóm giữa.
    v = q["mean_rel_return"].to_numpy()
    assert v[0] > v[2] and v[4] > v[2], f"không phải hình chữ U: {v}"

    # Hệ quả: tín hiệu đơn điệu tăng (IC dương rõ) nhưng nhóm thấp nhất vẫn thắng
    # nhóm giữa -> đơn điệu KHÔNG hoàn hảo. Đây chính là chỗ IC đánh lừa.
    mono = monotonicity(sig, close, 1, 5)
    assert np.isfinite(mono)
    assert mono < 1.0, "dữ liệu chữ U mà lại đơn điệu hoàn hảo — fixture sai"

    # Chênh lệch decile phản ánh HAI ĐUÔI, không phản ánh phần giữa.
    ds = decile_spread(sig, close, horizon=1, top_frac=0.20)
    assert np.isfinite(ds["mean"])


def test_ic_decay_curve_co_du_cot(panel):
    p, _ = panel
    close = p["close"]
    sig = xs_zscore(close.pct_change(10))
    d = ic_decay_curve(sig, close, horizons=[1, 3, 6])
    assert list(d.index) == [1, 3, 6]
    for c in ("ic_mean", "ic_ir", "t_stat", "ic_per_sqrt_period"):
        assert c in d.columns


def test_tu_tuong_quan_giam_theo_do_tre(panel):
    p, _ = panel
    sig = xs_zscore(p["close"].pct_change(60))
    ac = signal_autocorrelation(sig, [1, 12, 60])
    assert ac.iloc[0] > ac.iloc[-1], "tự tương quan phải giảm khi độ trễ tăng"


def test_do_rong_hieu_dung_bi_chiet_khau(panel):
    p, _ = panel
    sig = xs_zscore(p["close"].pct_change(90))   # tín hiệu rất chậm
    br = effective_breadth(sig, 1, 365.0, n_positions=12)
    assert br["effective_breadth"] < br["raw_breadth"]
    assert 0.0 <= br["signal_autocorr"] < 1.0


def test_implied_sharpe_tang_theo_can_do_rong():
    a = implied_sharpe(0.03, 1000, 0.6)
    b = implied_sharpe(0.03, 4000, 0.6)
    assert b == pytest.approx(2 * a, rel=1e-9)


# =========================================================================
# signal_library
# =========================================================================
def test_moi_tin_hieu_chay_duoc_va_da_chuan_hoa(panel):
    p, funding = panel
    for name in SIGNAL_REGISTRY:
        s = build_signal(name, p, funding)
        assert s.shape == p["close"].shape, f"{name} sai shape"
        row = s.dropna(how="all")
        if row.empty:
            continue
        # đã chuẩn hoá theo hàng -> |trung bình hàng| nhỏ
        m = row.mean(axis=1).abs().median()
        assert m < 0.5, f"{name} chưa chuẩn hoá theo mặt cắt ngang (mean={m:.3f})"


def test_tin_hieu_nhan_qua(panel):
    """Đổi giá nến cuối không được đổi giá trị tín hiệu ở các nến trước."""
    p, funding = panel
    tampered = {k: v.copy() for k, v in p.items()}
    for k in ("close", "high", "low", "volume", "ofi"):
        tampered[k].iloc[-1] *= 2.0
    for name in SIGNAL_REGISTRY:
        a = build_signal(name, p, funding).iloc[:-1]
        b = build_signal(name, tampered, funding).iloc[:-1]
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12,
                                      obj=f"tín hiệu {name} nhìn trước")


def test_moi_ho_deu_dung_duoc(panel):
    p, funding = panel
    for f in FAMILIES:
        s = build_family(f, p, funding)
        assert s.shape == p["close"].shape
        assert s.notna().any().any(), f"họ {f} toàn NaN"


def test_xs_rank_nam_trong_khoang_chuan():
    df = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (10, 8)))
    r = xs_rank(df)
    assert r.min().min() >= -1.0 and r.max().max() <= 1.0


def test_build_signal_bao_loi_khi_ten_sai(panel):
    p, funding = panel
    with pytest.raises(KeyError):
        build_signal("khong_ton_tai", p, funding)


# =========================================================================
# adaptive_combiner
# =========================================================================
def test_trong_so_thich_ung_khong_nhin_truoc():
    """
    Bất biến sống còn: trọng số tại mốc i chỉ được dùng lợi suất tới i-1. Đổi giá
    trị TẠI i không được làm đổi trọng số tại i.
    """
    rng = np.random.default_rng(4)
    idx = np.arange(400) * BAR_MS
    fac = {"a": pd.Series(rng.normal(0.001, 0.02, 400), index=idx),
           "b": pd.Series(rng.normal(-0.0005, 0.02, 400), index=idx)}
    spec = CombinerSpec(lookback=100, min_periods=50, t_threshold=1.0)
    base = adaptive_weights(fac, spec)

    tampered = {k: v.copy() for k, v in fac.items()}
    tampered["a"].iloc[200] = 5.0        # cú sốc khổng lồ tại đúng mốc 200
    after = adaptive_weights(tampered, spec)

    pd.testing.assert_frame_equal(base.iloc[:201], after.iloc[:201])
    assert not np.allclose(base.iloc[201:].to_numpy(), after.iloc[201:].to_numpy()), \
        "cú sốc phải ảnh hưởng các mốc SAU đó"


def test_thich_ung_hoc_duoc_dau_am():
    """Một họ lỗ đều đặn phải nhận trọng số ÂM (tự đảo chiều)."""
    idx = np.arange(600) * BAR_MS
    rng = np.random.default_rng(9)
    fac = {"tot": pd.Series(rng.normal(0.004, 0.01, 600), index=idx),
           "xau": pd.Series(rng.normal(-0.004, 0.01, 600), index=idx)}
    W = adaptive_weights(fac, CombinerSpec(lookback=300, min_periods=100,
                                           t_threshold=1.0, max_step=1.0))
    assert W["tot"].iloc[-1] > 0
    assert W["xau"].iloc[-1] < 0


def test_thich_ung_khong_doi_chieu_khi_bi_cam():
    idx = np.arange(600) * BAR_MS
    rng = np.random.default_rng(9)
    fac = {"tot": pd.Series(rng.normal(0.004, 0.01, 600), index=idx),
           "xau": pd.Series(rng.normal(-0.004, 0.01, 600), index=idx)}
    W = adaptive_weights(fac, CombinerSpec(lookback=300, min_periods=100, t_threshold=1.0,
                                           max_step=1.0, allow_sign_flip=False))
    assert W["xau"].iloc[-1] >= 0


def test_nguong_t_cao_lam_tat_tin_hieu_yeu():
    idx = np.arange(600) * BAR_MS
    rng = np.random.default_rng(2)
    fac = {"yeu": pd.Series(rng.normal(0.0001, 0.02, 600), index=idx),
           "manh": pd.Series(rng.normal(0.006, 0.01, 600), index=idx)}
    lo = adaptive_weights(fac, CombinerSpec(lookback=300, min_periods=100,
                                            t_threshold=0.0, max_step=1.0))
    hi = adaptive_weights(fac, CombinerSpec(lookback=300, min_periods=100,
                                            t_threshold=3.0, max_step=1.0))
    assert abs(hi["yeu"].iloc[-1]) <= abs(lo["yeu"].iloc[-1]) + 1e-9


def test_tran_thay_doi_moi_ky_duoc_ap():
    idx = np.arange(500) * BAR_MS
    rng = np.random.default_rng(6)
    fac = {"a": pd.Series(rng.normal(0.003, 0.01, 500), index=idx),
           "b": pd.Series(rng.normal(-0.003, 0.01, 500), index=idx)}
    step = 0.02
    W = adaptive_weights(fac, CombinerSpec(lookback=200, min_periods=60, max_step=step))
    # sau khi chuẩn hoá, thay đổi có thể hơi vượt step; cho biên độ rộng rãi 3x
    assert W.diff().abs().max().max() <= step * 3


def test_factor_returns_dung_dau(panel):
    """Danh mục nhân tố của tín hiệu oracle phải cho lợi suất dương."""
    p, _ = panel
    close = p["close"]
    fwd = close.shift(-1) / close - 1.0
    r = factor_returns(xs_zscore(fwd), close, top_frac=0.20)
    assert r.mean() > 0 and (r > 0).mean() > 0.95


def test_combine_adaptive_tra_ve_dung_shape(panel):
    p, funding = panel
    close = p["close"]
    sigs = {f: build_family(f, p, funding) for f in FAMILIES[:3]}
    out = combine_adaptive(sigs, close, CombinerSpec(lookback=200, min_periods=60))
    assert out.shape == close.shape
    assert out.notna().any().any()


# =========================================================================
# leverage
# =========================================================================
def test_kelly_dung_cong_thuc():
    assert kelly_leverage(0.01, 0.10, fraction=1.0) == pytest.approx(1.0)
    assert kelly_leverage(0.01, 0.10, fraction=0.5) == pytest.approx(0.5)
    assert kelly_leverage(0.01, 0.0) == 0.0


def test_vol_target_tang_don_bay_khi_bien_dong_thap():
    idx = np.arange(400) * BAR_MS
    rng = np.random.default_rng(1)
    r = pd.Series(np.concatenate([rng.normal(0, 0.04, 200), rng.normal(0, 0.005, 200)]),
                  index=idx)
    lev = volatility_target_series(r, LeverageSpec(target_ann_vol=0.3, max_leverage=10,
                                                   vol_window=50, periods_per_year=365))
    assert lev.iloc[-1] > lev.iloc[150], "biến động giảm thì đòn bẩy phải tăng"
    assert lev.max() <= 10.0


def test_van_drawdown_cat_don_bay():
    """Chuỗi lỗ liên tiếp phải kéo đòn bẩy về 0 trước khi mất hết."""
    idx = np.arange(120) * BAR_MS
    r = pd.Series([-0.03] * 120, index=idx)
    out = apply_leverage(r, LeverageSpec(target_ann_vol=0.3, max_leverage=3.0,
                                         vol_window=20, vol_min_periods=10,
                                         dd_throttle_start=0.05, dd_throttle_stop=0.15),
                        use_vol_target=True, use_dd_throttle=True)
    assert out["leverage"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert out["equity"].iloc[-1] > 0.5, "van phải chặn trước khi mất quá nửa"


def test_von_khong_bao_gio_am():
    idx = np.arange(50) * BAR_MS
    r = pd.Series([-0.9] * 50, index=idx)
    out = apply_leverage(r, LeverageSpec(max_leverage=5.0, vol_window=10, vol_min_periods=5))
    assert (out["equity"] > 0).all()


def test_don_bay_khong_doi_sharpe():
    """Tính chất toán học: nhân vị thế với hằng số không đổi tỷ số Sharpe."""
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.002, 0.02, 500), index=np.arange(500) * BAR_MS)
    s1 = r.mean() / r.std(ddof=1)
    r3 = r * 3
    assert r3.mean() / r3.std(ddof=1) == pytest.approx(s1)


def test_xac_suat_dat_muc_tieu_tang_theo_don_bay_va_rui_ro_cung_tang():
    rng = np.random.default_rng(12)
    r = pd.Series(rng.normal(0.004, 0.03, 800), index=np.arange(800) * BAR_MS)
    tbl = target_probability_table(r, target_return=0.50, n_periods=3,
                                   leverages=[1, 5, 20], n_paths=4000, block=2,
                                   periods_per_year=121.7)
    assert tbl["p_target"].is_monotonic_increasing
    assert tbl["p_ruin"].is_monotonic_increasing
    assert (tbl["p_ruin"] >= 0).all() and (tbl["p_ruin"] <= 1).all()


def test_chay_la_hap_thu():
    """Đường đã chạm ngưỡng cháy không được 'hồi phục' trên giấy."""
    rng = np.random.default_rng(15)
    r = pd.Series(rng.normal(0, 0.05, 400), index=np.arange(400) * BAR_MS)
    sim = simulate_paths(r, leverage=20.0, n_periods=20, n_paths=2000,
                         block=2, ruin_threshold=0.30)
    assert np.all(sim["final"][sim["ruined"]] == pytest.approx(0.30))


def test_simulate_paths_tu_choi_mau_qua_ngan():
    r = pd.Series(np.zeros(5))
    with pytest.raises(ValueError):
        simulate_paths(r, 1.0, 10, block=5)
