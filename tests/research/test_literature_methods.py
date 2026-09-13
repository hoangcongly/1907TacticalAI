"""
Test hai tầng gộp dựng theo tài liệu nghiên cứu: hồi quy mặt cắt ngang và LambdaRank.

Cả hai ĐỀU KHÔNG thắng được tầng gộp hiện có trên dữ liệu này (xem
`docs/upgrade_v3_report.md` §11). Nhưng chúng được giữ lại và test đầy đủ vì lý do
cụ thể: ràng buộc khiến chúng thua là ĐỘ RỘNG MẶT CẮT NGANG (~41 tài sản mỗi kỳ),
không phải sai sót cài đặt. Nếu universe mở rộng đủ lớn, chúng trở nên dùng được —
và khi đó không ai phải viết lại từ đầu.

Tính chất bắt buộc và được khoá ở đây là NHÂN QUẢ. Một tầng gộp rò rỉ tương lai sẽ
cho Sharpe rất đẹp và hoàn toàn vô giá trị.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.research.rank_model import RankModelSpec, build_dataset, combine_lambdarank
from aegis.research.xs_regression import (
    XSRegressionSpec, combine_xs_regression, cross_sectional_betas,
)

BAR = 14_400_000


@pytest.fixture
def market():
    """30 tài sản, 400 kỳ. Tín hiệu `good` CÓ sức dự báo thật, `noise` thì không."""
    rng = np.random.default_rng(3)
    n, k = 400, 30
    idx = np.arange(n) * BAR
    syms = [f"S{i:02d}" for i in range(k)]

    good = pd.DataFrame(rng.normal(0, 1, (n, k)), index=idx, columns=syms)
    noise = pd.DataFrame(rng.normal(0, 1, (n, k)), index=idx, columns=syms)
    # lợi suất kỳ sau phụ thuộc `good` tại kỳ này -> quan hệ nhân quả cài sẵn
    rets = 0.01 * good.shift(1).fillna(0.0) + rng.normal(0, 0.02, (n, k))
    close = pd.DataFrame(100 * np.cumprod(1 + rets.to_numpy(), axis=0), index=idx, columns=syms)
    return {"good": good, "noise": noise}, close


# ---------------------------------------------------------------- hồi quy XS
def test_hoi_quy_tim_ra_tin_hieu_that(market):
    """Hệ số của tín hiệu có sức dự báo phải DƯƠNG và lớn hơn hẳn tín hiệu nhiễu."""
    sigs, close = market
    betas = cross_sectional_betas(sigs, close, XSRegressionSpec(min_names=10))
    assert len(betas) > 300
    assert betas["good"].mean() > 0
    assert abs(betas["good"].mean()) > abs(betas["noise"].mean()) * 3


def test_hoi_quy_khong_nhin_truoc(market):
    """
    Đổi giá ở nến CUỐI không được làm đổi điểm số ở bất kỳ kỳ nào trước đó.
    Đây là phép thử nhân quả trực tiếp nhất.
    """
    sigs, close = market
    base = combine_xs_regression(sigs, close, XSRegressionSpec(min_names=10))
    tampered = close.copy()
    tampered.iloc[-1] *= 3.0
    after = combine_xs_regression(sigs, tampered, XSRegressionSpec(min_names=10))
    pd.testing.assert_frame_equal(base.iloc[:-2], after.iloc[:-2])


def test_he_so_duoc_dich_mot_nhip(market):
    """
    Hệ số tại hàng t học từ lợi suất t->t+1, nên dùng để dự báo TẠI t là nhìn trước.
    Điểm số phải dùng trung bình các hàng <= t-1.
    """
    sigs, close = market
    spec = XSRegressionSpec(beta_window=10, min_periods=5, min_names=10)
    score, betas = combine_xs_regression(sigs, close, spec, return_betas=True)
    # Trước khi tích đủ `min_periods` hệ số thì chưa được có điểm số.
    first_scored = score.dropna(how="all").index[0]
    assert first_scored >= betas.index[spec.min_periods]


@pytest.mark.parametrize("method,alpha", [("ols", 0.0), ("ridge", 1e-3), ("elastic_net", 1e-4)])
def test_moi_phuong_phap_hoi_quy_chay_duoc(market, method, alpha):
    sigs, close = market
    out = combine_xs_regression(sigs, close, XSRegressionSpec(
        method=method, alpha=alpha, min_names=10))
    assert out.shape == close.shape
    assert out.notna().any().any()


def test_winsorize_chan_mot_coin_quyet_dinh_ca_he_so():
    """Một coin tăng 300% không được một mình định đoạt hệ số của cả kỳ."""
    from aegis.research.xs_regression import _winsorize
    x = np.array([0.01] * 50 + [3.0])
    assert _winsorize(x, 0.02).max() < 3.0


def test_hoi_quy_tu_choi_method_la(market):
    sigs, close = market
    with pytest.raises(ValueError):
        cross_sectional_betas(sigs, close, XSRegressionSpec(method="khong_ton_tai"))


# ---------------------------------------------------------------- LambdaRank
def test_nhan_la_phan_vi_TRONG_KY(market):
    """
    Nhãn phải là thứ hạng trong cùng mặt cắt ngang, không phải lợi suất tuyệt đối.
    Nếu dùng lợi suất tuyệt đối, model chỉ học 'kỳ nào thị trường tăng' — vô dụng
    với danh mục trung lập.
    """
    sigs, close = market
    spec = RankModelSpec(n_labels=8, min_names=10)
    X, y, ts, names = build_dataset(sigs, close, spec)
    assert set(names) == {"good", "noise"}
    assert y.min() >= 0 and y.max() <= spec.n_labels - 1
    # mỗi kỳ phải phủ gần hết dải nhãn, vì đó là phân vị trong kỳ
    per = pd.DataFrame({"y": y, "ts": ts}).groupby("ts")["y"].nunique()
    assert per.median() >= spec.n_labels - 2


def test_lambdarank_khong_nhin_truoc(market):
    """Đổi dữ liệu tương lai không được làm đổi điểm số quá khứ."""
    sigs, close = market
    spec = RankModelSpec(train_periods=150, retrain_every=50, min_train_periods=100,
                         min_names=10, n_estimators=30, num_leaves=7)
    base = combine_lambdarank(sigs, close, spec)
    cut = len(close) // 2
    tampered = close.copy()
    tampered.iloc[cut:] *= 1.5
    after = combine_lambdarank(sigs, tampered, spec)

    a = base.iloc[:cut - spec.purge - 1].dropna(how="all")
    b = after.reindex(a.index)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_lambdarank_khong_cham_diem_khi_chua_du_train(market):
    sigs, close = market
    spec = RankModelSpec(min_train_periods=200, train_periods=300, min_names=10,
                         n_estimators=20, num_leaves=7)
    out = combine_lambdarank(sigs, close, spec)
    scored = out.dropna(how="all")
    assert len(scored) > 0
    assert scored.index[0] >= close.index[spec.min_train_periods]


def test_cau_hinh_tu_mau_thuan_nem_loi_ngay():
    """
    `train_periods < min_train_periods` khiến điều kiện huấn luyện không bao giờ
    thoả -> toàn NaN, không báo gì. Test này được viết TRƯỚC khi có bản vá và chính
    nó phát hiện ra chế độ hỏng im lặng đó.
    """
    with pytest.raises(ValueError, match="train_periods"):
        RankModelSpec(train_periods=150, min_train_periods=200)
    with pytest.raises(ValueError, match="n_labels"):
        RankModelSpec(n_labels=1)
    with pytest.raises(ValueError, match="purge"):
        RankModelSpec(purge=-1)


def test_lambdarank_chon_universe_khong_dua_vao_tuong_lai(market):
    """
    Tập tài sản được chấm điểm phải do ĐỘ PHỦ TÍN HIỆU quyết định, không do lợi suất
    tương lai. Bản đầu lọc theo lợi suất tương lai -> universe live khác backtest,
    và backtest âm thầm loại đúng các cặp bị huỷ niêm yết.
    """
    sigs, close = market
    tampered = close.copy()
    tampered.iloc[-1, :5] = np.nan          # 5 cặp "ngừng giao dịch" ở kỳ cuối
    spec = RankModelSpec(train_periods=150, retrain_every=50, min_train_periods=100,
                         min_names=10, n_estimators=30, num_leaves=7)
    out = combine_lambdarank(sigs, tampered, spec)
    scored = out.dropna(how="all")
    # kỳ áp chót vẫn phải chấm đủ, không bị ảnh hưởng bởi việc kỳ cuối thiếu dữ liệu
    assert scored.iloc[-1].notna().sum() >= 10
