"""
Test tầng dựng danh mục: bất biến trung lập, chia đều rủi ro, và TÍNH NHÂN QUẢ.

Nguyên tắc viết test ở đây: mỗi test kiểm chứng một tính chất mà nếu sai thì tiền
thật sẽ mất, chứ không kiểm chứng "hàm chạy không lỗi".
"""
import numpy as np
import pandas as pd
import pytest

from aegis.risk.covariance_shrinkage import (
    ewma_covariance, ledoit_wolf_constant_correlation, nearest_positive_definite,
    shrink_covariance, single_index_covariance,
)
from aegis.risk.portfolio import (
    PortfolioSpec, apply_buffer, beta_neutralize, build_weights,
    inverse_vol_weights, project_neutral, realized_vol, rolling_beta,
)


@pytest.fixture
def market():
    """Panel giả lập: 20 tài sản, một nhân tố chung, biến động RẤT khác nhau."""
    rng = np.random.default_rng(11)
    n_bars, n_sym = 400, 20
    syms = [f"S{i:02d}" for i in range(n_sym)]
    # biến động từ 1% tới 10% mỗi nến -> chênh 10 lần, đủ để lộ lỗi chia đều VỐN
    vols = np.linspace(0.01, 0.10, n_sym)
    beta = np.linspace(0.4, 1.8, n_sym)
    factor = rng.normal(0, 0.02, n_bars)
    rets = np.outer(factor, beta) + rng.normal(0, 1, (n_bars, n_sym)) * vols
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0),
                         index=np.arange(n_bars) * 14_400_000, columns=syms)
    signal = pd.DataFrame(rng.normal(0, 1, (n_bars, n_sym)), index=close.index, columns=syms)
    return close, signal


# --------------------------------------------------------------------------
# Hiệp phương sai
# --------------------------------------------------------------------------
def test_ledoit_wolf_co_manh_hon_khi_mau_ngan():
    """
    Mẫu càng ngắn so với số tài sản, cường độ co delta phải càng lớn.

    Dữ liệu phải CÓ CẤU TRÚC NHÂN TỐ mới kiểm được tính chất này. Với nhiễu iid
    thuần tuý, mục tiêu co (tương quan hằng số, rbar ~ 0) CHÍNH LÀ sự thật, nên
    delta = 1.0 ở mọi độ dài mẫu — đúng về mặt lý thuyết nhưng không phân biệt được
    gì. Crypto có nhân tố chung rất mạnh, nên trường hợp có nhân tố mới là trường
    hợp cần bảo vệ.
    """
    rng = np.random.default_rng(3)
    beta = rng.uniform(0.5, 1.5, 40)
    factor = rng.normal(0, 0.03, 500)
    X = np.outer(factor, beta) + rng.normal(0, 0.02, (500, 40))

    _, d_long = ledoit_wolf_constant_correlation(X)
    _, d_short = ledoit_wolf_constant_correlation(X[:60])
    assert 0.0 <= d_long <= 1.0 and 0.0 <= d_short <= 1.0
    assert d_short > d_long, f"delta ngắn {d_short:.3f} phải > delta dài {d_long:.3f}"


def test_ledoit_wolf_co_toan_phan_khi_muc_tieu_dung():
    """Nhiễu iid: mục tiêu tương quan hằng số là đúng -> delta phải bằng 1.0."""
    rng = np.random.default_rng(3)
    _, delta = ledoit_wolf_constant_correlation(rng.normal(0, 0.02, (500, 40)))
    assert delta == pytest.approx(1.0, abs=1e-6)


def test_hiep_phuong_sai_luon_xac_dinh_duong_ke_ca_khi_mau_suy_bien():
    """T < N khiến hiệp phương sai mẫu suy biến; bản co PHẢI vẫn dương ngặt."""
    rng = np.random.default_rng(5)
    X = rng.normal(0, 0.02, (30, 50))
    assert np.linalg.eigvalsh(np.cov(X, rowvar=False)).min() < 1e-10
    for method in ("constant_correlation", "single_index", "ewma"):
        sigma = shrink_covariance(X, method=method)
        assert np.linalg.eigvalsh(sigma).min() > 0, f"{method} cho trị riêng <= 0"
        assert np.allclose(sigma, sigma.T), f"{method} không đối xứng"


def test_nearest_pd_giu_nguyen_ma_tran_da_dat_yeu_cau():
    sigma = np.eye(5) * 0.04
    assert np.allclose(nearest_positive_definite(sigma), sigma)


def test_ewma_bam_theo_che_do_moi_hon_cua_so_deu():
    """Biến động tăng gấp đôi ở nửa sau: EWMA phải phản ánh nhanh hơn mẫu đều."""
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(0, 0.01, (300, 4)), rng.normal(0, 0.04, (100, 4))])
    var_ewma = np.diag(ewma_covariance(X, halflife=30)).mean()
    var_sample = np.diag(np.cov(X, rowvar=False)).mean()
    assert var_ewma > var_sample


# --------------------------------------------------------------------------
# Ước lượng nhân quả
# --------------------------------------------------------------------------
def test_realized_vol_khong_nhin_truoc(market):
    """
    Đổi giá ở nến CUỐI không được làm đổi bất kỳ giá trị biến động nào trước đó.
    Đây là phép thử nhân quả trực tiếp nhất và bắt được mọi lỗi shift.
    """
    close, _ = market
    base = realized_vol(close, 60)
    tampered = close.copy()
    tampered.iloc[-1] *= 3.0
    after = realized_vol(tampered, 60)
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_rolling_beta_khong_nhin_truoc(market):
    close, _ = market
    base = rolling_beta(close, 120)
    tampered = close.copy()
    tampered.iloc[-1] *= 3.0
    after = rolling_beta(tampered, 120)
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_build_weights_khong_nhin_truoc(market):
    """Trọng số tại mọi mốc trước phải không đổi khi dữ liệu tương lai đổi."""
    close, signal = market
    spec = PortfolioSpec(mode="zscore_riskparity", n_positions=8)
    base = build_weights(signal, close, spec)
    tampered = close.copy()
    tampered.iloc[-1] *= 5.0
    after = build_weights(signal, tampered, spec)
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


# --------------------------------------------------------------------------
# Bất biến danh mục
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["rank_binary", "rank_riskparity", "zscore", "zscore_riskparity"])
def test_bat_bien_trung_lap_va_gross(market, mode):
    close, signal = market
    spec = PortfolioSpec(mode=mode, n_positions=10,
                         max_weight=1.0 if mode == "rank_binary" else 0.25)
    W = build_weights(signal, close, spec)
    active = W[(W != 0).any(axis=1)]
    assert len(active) > 50, "gần như không mở vị thế nào — cấu hình sai"

    net = active.sum(axis=1).abs()
    gross = active.abs().sum(axis=1)
    assert net.max() < 1e-8, f"net exposure tối đa {net.max():.2e} != 0"
    assert np.allclose(gross, spec.gross, atol=1e-8)
    assert ((active > 0).any(axis=1) & (active < 0).any(axis=1)).all()


def test_so_vi_the_tuyet_doi_duoc_ton_trong(market):
    """`n_positions` phải thắng `top_frac`, bất kể universe to nhỏ ra sao."""
    close, signal = market
    for n in (6, 8, 12):
        W = build_weights(signal, close, PortfolioSpec(mode="rank_binary",
                                                       n_positions=n, top_frac=0.40,
                                                       max_weight=1.0))
        counts = (W != 0).sum(axis=1)
        counts = counts[counts > 0]
        assert counts.max() <= n, f"n_positions={n} nhưng có lúc mở {counts.max()} vị thế"


def test_chia_deu_rui_ro_ha_ty_trong_cua_tai_san_bien_dong_manh(market):
    """
    Bất biến cốt lõi: cùng tập tài sản, chế độ inv-vol phải cho tài sản biến động
    CAO tỷ trọng THẤP hơn so với chế độ nhị phân.
    """
    close, signal = market
    vol = realized_vol(close, 60)
    Wb = build_weights(signal, close, PortfolioSpec(mode="rank_binary", n_positions=10,
                                                    max_weight=1.0))
    Wr = build_weights(signal, close, PortfolioSpec(mode="rank_riskparity", n_positions=10,
                                                    max_weight=1.0))
    ts = Wb.index[300]
    sel = Wb.loc[ts][Wb.loc[ts] != 0].index
    assert len(sel) >= 4
    v = vol.loc[ts, sel]
    hi, lo = v.idxmax(), v.idxmin()
    # so tỷ lệ |trọng số| giữa cặp biến động cao nhất và thấp nhất
    ratio_bin = abs(Wb.loc[ts, hi]) / abs(Wb.loc[ts, lo])
    ratio_rp = abs(Wr.loc[ts, hi]) / abs(Wr.loc[ts, lo])
    assert ratio_rp < ratio_bin, "inv-vol không hạ tỷ trọng tài sản biến động cao"


def test_tran_trong_so_duoc_ap_trong_dung_sai_da_cong_bo(market):
    """
    Trần trọng số được tôn trọng TRONG MỘT DUNG SAI NHỎ, và dung sai đó là CÓ CHỦ Ý.

    `build_weights` kết thúc bằng phép chiếu trung lập chứ không bằng kẹp trần, vì
    thứ tự ngược lại để sổ ra khỏi hàm với net exposure khác 0 (đo được tới 20%
    gross — đúng cơ chế từng khiến danh mục live lệch 33% khỏi trung lập). Đánh đổi
    là một cặp có thể vượt trần vài phần trăm tương đối. Test này chốt cả hai vế:
    trung lập TUYỆT ĐỐI, trần XẤP XỈ.
    """
    close, signal = market
    cap = 0.15
    W = build_weights(signal, close, PortfolioSpec(mode="zscore_riskparity",
                                                   n_positions=12, max_weight=cap))
    active = W[(W != 0).any(axis=1)]
    worst = float(active.abs().max().max())
    assert worst <= cap * 1.10, f"vượt trần {worst/cap-1:.1%} — quá nhiều"
    assert active.sum(axis=1).abs().max() < 1e-8, "trung lập phải TUYỆT ĐỐI"


# --------------------------------------------------------------------------
# Các phép biến đổi riêng lẻ
# --------------------------------------------------------------------------
def test_project_neutral_dat_ca_hai_rang_buoc():
    w = pd.Series([0.3, 0.2, -0.1, -0.4, 0.5], index=list("abcde"))
    beta = pd.Series([1.5, 0.8, 1.2, 0.5, 1.9], index=list("abcde"))
    out = project_neutral(w, beta)
    assert abs(out.sum()) < 1e-12, "tổng trọng số != 0"
    assert abs(float((out * beta).sum())) < 1e-12, "beta danh mục != 0"


def test_project_neutral_la_phep_chieu_gan_nhat():
    """Nghiệm phải gần `w` hơn mọi nghiệm hợp lệ khác (tính chất của phép chiếu)."""
    rng = np.random.default_rng(1)
    w = pd.Series(rng.normal(0, 1, 8), index=[f"s{i}" for i in range(8)])
    beta = pd.Series(rng.uniform(0.5, 2.0, 8), index=w.index)
    out = project_neutral(w, beta)
    d0 = float(((out - w) ** 2).sum())
    # thêm một vector bất kỳ nằm trong không gian ràng buộc -> phải xa hơn
    other = project_neutral(w + pd.Series(rng.normal(0, 0.5, 8), index=w.index), beta)
    assert abs(other.sum()) < 1e-10
    assert float(((other - w) ** 2).sum()) >= d0 - 1e-12


def test_beta_neutralize_khu_dung_phan_hinh_chieu():
    w = pd.Series([0.5, -0.5], index=["a", "b"])
    beta = pd.Series([2.0, 1.0], index=["a", "b"])
    out = beta_neutralize(w, beta)
    assert abs(float((out * beta).sum())) < 1e-12


def test_inverse_vol_ap_san_bien_dong():
    """Tài sản gần như không biến động không được nuốt trọn danh mục."""
    raw = pd.Series([1.0, 1.0], index=["a", "b"])
    vol = pd.Series([1e-12, 0.05], index=["a", "b"])
    out = inverse_vol_weights(raw, vol, vol_floor=0.01)
    assert out["a"] == pytest.approx(100.0)   # bị kẹp về sàn 0.01
    assert out["b"] == pytest.approx(20.0)


def test_vung_dem_giam_turnover(market):
    """Vùng đệm phải giảm turnover so với không đệm, cùng số vị thế."""
    close, signal = market
    plain = build_weights(signal, close, PortfolioSpec(mode="rank_binary", n_positions=10,
                                                       max_weight=1.0))
    buffered = build_weights(signal, close, PortfolioSpec(
        mode="rank_binary", n_positions=10, max_weight=1.0,
        top_frac=0.10, exit_frac=0.20))
    t_plain = (plain - plain.shift(1)).abs().sum(axis=1).mean()
    t_buf = (buffered - buffered.shift(1)).abs().sum(axis=1).mean()
    assert t_buf < t_plain, f"đệm {t_buf:.3f} không thấp hơn không đệm {t_plain:.3f}"


def test_spec_tu_choi_tham_so_vo_ly():
    with pytest.raises(ValueError):
        PortfolioSpec(mode="khong_ton_tai")
    with pytest.raises(ValueError):
        PortfolioSpec(top_frac=0.8)
    with pytest.raises(ValueError):
        PortfolioSpec(top_frac=0.2, exit_frac=0.1)
    with pytest.raises(ValueError):
        PortfolioSpec(n_positions=2)
