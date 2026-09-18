"""
Khoá tính NHÂN QUẢ và tính nhất quán của engine đánh dấu theo thị trường.

`simulate_marked_to_market` đo cùng một chiến lược ở độ phân giải nến thay vì độ phân
giải tái cân bằng. Nó tồn tại vì lưới thô không nhìn thấy sụt giảm trong kỳ. Nhưng một
engine mịn hơn cũng có nhiều chỗ để nhìn trước hơn, nên đây là phần phải khoá chặt.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.research.backtest_v2 import (
    CostModel, simulate, simulate_marked_to_market,
)


def _panel(n=60, n_sym=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64) * 14_400_000
    cols = [f"S{i}" for i in range(n_sym)]
    px = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=(n, n_sym)), axis=0)
    return pd.DataFrame(px, index=idx, columns=cols)


def test_khong_nhin_truoc_mot_nen():
    """
    Hồi quy cho lỗi ĐÃ XẢY RA: đặt trọng số ở mốc `ts` rồi áp ngay lợi suất của chính
    nến vừa đóng tại `ts`. Một nhịp thôi cũng đủ thổi Sharpe holdout từ 1,28 lên 1,66.

    Cách bắt: dựng một cú nhảy +50% xảy ra ĐÚNG tại nến mà danh mục vào lệnh. Engine
    nhân quả không được ăn một xu nào của cú nhảy đó.
    """
    close = _panel(n=30, n_sym=2)
    jump_at = close.index[10]
    close.loc[jump_at:, "S0"] *= 1.5          # giá nhảy khi đóng nến 10

    W = pd.DataFrame(0.0, index=[jump_at], columns=close.columns)
    W.loc[jump_at, "S0"] = 1.0                # vào 100% S0 ở đúng nến đó

    res = simulate_marked_to_market(W, close, None, CostModel(min_bps=0.0,
                                    taker_fee_bps=0.0, maker_fee_bps=0.0,
                                    half_spread_bps=0.0), bar_hours=4.0)
    at_entry = float(res.returns.loc[jump_at])
    assert abs(at_entry) < 1e-9, f"ăn được {at_entry:+.4f} của cú nhảy — đang nhìn trước"


def test_khong_giao_dich_giua_hai_lan_tai_can_bang():
    """Chỉ có phí ở các mốc trong `weights`; giữa hai mốc trọng số TRÔI, không bị kéo về."""
    close = _panel(n=40, n_sym=3)
    marks = close.index[::10]
    W = pd.DataFrame(1.0 / 3.0, index=marks, columns=close.columns)

    res = simulate_marked_to_market(W, close, None, CostModel(min_bps=5.0), bar_hours=4.0)
    traded = res.turnover[res.turnover > 1e-12].index
    assert set(traded).issubset(set(marks)), "có giao dịch ngoài mốc tái cân bằng"
    assert (res.cost_drag[~res.cost_drag.index.isin(marks)] == 0.0).all()


def test_moc_tai_can_bang_phai_nam_tren_luoi_gia():
    close = _panel(n=20)
    W = pd.DataFrame(0.25, index=[close.index[-1] + 1], columns=close.columns)
    with pytest.raises(ValueError, match="không có trong lưới giá"):
        simulate_marked_to_market(W, close, None, CostModel(), bar_hours=4.0)


def test_cong_don_luoi_min_khop_luoi_tho_khi_khong_co_phi():
    """
    Cùng trọng số, cùng dữ liệu, không phí: lợi suất cộng dồn của lưới mịn phải bằng
    lợi suất "mua và giữ" của lưới thô cho MỘT kỳ.

    Đây là phép nối giữa hai engine. Nếu nó gãy thì một trong hai đang tính sai lợi
    suất danh mục, và mọi so sánh maxDD giữa hai lưới trở thành vô nghĩa.
    """
    close = _panel(n=25, n_sym=3, seed=7)
    t0, t1 = close.index[0], close.index[10]
    w = pd.Series([0.5, -0.3, 0.2], index=close.columns)

    # lưới thô: mua-và-giữ từ t0 tới t1
    coarse = float((w * (close.loc[t1] / close.loc[t0] - 1.0)).sum())

    # lưới mịn: vào ở t0, để trôi, cộng dồn tới t1
    W = pd.DataFrame([w.to_numpy()], index=[t0], columns=close.columns)
    res = simulate_marked_to_market(W, close, None, CostModel(min_bps=0.0,
                                    taker_fee_bps=0.0, maker_fee_bps=0.0,
                                    half_spread_bps=0.0), bar_hours=4.0)
    seg = res.returns.loc[t0:t1]
    fine = float(np.prod(1.0 + seg.to_numpy()) - 1.0)
    assert fine == pytest.approx(coarse, rel=1e-9, abs=1e-12)


def test_drawdown_luoi_min_khong_bao_gio_nho_hon_luoi_tho():
    """
    Bất biến: lưới mịn thấy MỌI điểm lưới thô thấy, cộng thêm các điểm giữa. Nên sụt
    giảm nó đo được không thể nhỏ hơn. Đo được trên dữ liệu thật: 36,9% -> 45,5%.
    """
    close = _panel(n=200, n_sym=5, seed=3)
    marks = close.index[::18]
    rng = np.random.default_rng(1)
    W = pd.DataFrame(rng.normal(0, 0.3, size=(len(marks), close.shape[1])),
                     index=marks, columns=close.columns)
    W = W.div(W.abs().sum(axis=1), axis=0)     # gross = 1.0

    free = CostModel(min_bps=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0, half_spread_bps=0.0)
    fine = simulate_marked_to_market(W, close, None, free, bar_hours=4.0)
    coarse = simulate(W, close.reindex(marks), None, free, bar_hours=72.0, rebalance_every=1)

    def mdd(r):
        e = np.cumprod(1.0 + r.dropna().to_numpy())
        return float((1.0 - e / np.maximum.accumulate(e)).max())

    assert mdd(fine.returns) >= mdd(coarse.returns) - 1e-12
