"""
Test basis trade — trọng tâm là QUY ƯỚC DẤU và TÍNH NHÂN QUẢ.

Lớp chiến lược này dễ tạo ra kết quả đẹp giả tạo hơn mọi thứ khác trong repo, vì
nguồn lợi nhuận (funding) gần như luôn dương còn nguồn rủi ro (basis) thì bị giấu.
Trong lúc dựng module đã mắc đúng ba lỗi đó, và mỗi lỗi giờ có một test canh:

  1. giả định phòng hộ hoàn hảo  -> đo Sharpe của funding thay vì của chiến lược
  2. chọn theo funding CÙNG THỜI ĐIỂM -> nhìn trước, 25-43%/năm thay vì 5-9%
  3. đuổi theo funding cao nhất mỗi nến -> phí ăn -150%/năm
"""
import numpy as np
import pandas as pd
import pytest

from aegis.research.basis_trade import BasisSpec, backtest_basis, basis_series

BAR = 14_400_000


def _frame(vals, cols, n=None):
    n = n or len(vals)
    idx = np.arange(n) * BAR
    if np.ndim(vals) == 1:
        vals = np.tile(np.asarray(vals).reshape(-1, 1), (1, len(cols)))
    return pd.DataFrame(np.asarray(vals, dtype=float), index=idx, columns=cols)


# ---------------------------------------------------------------- basis
def test_basis_dung_dinh_nghia():
    P = _frame([101.0, 102.0], ["A"])
    S = _frame([100.0, 100.0], ["A"])
    b = basis_series(P, S)
    assert b.loc[0, "A"] == pytest.approx(0.01)
    assert b.loc[BAR, "A"] == pytest.approx(0.02)


def test_basis_am_khi_perp_re_hon_spot():
    assert basis_series(_frame([99.0], ["A"]), _frame([100.0], ["A"])).iloc[0, 0] < 0


# ---------------------------------------------------------------- quy ước dấu
def test_lai_khi_basis_THU_HEP():
    """
    SHORT perp + LONG spot: ta lãi khi perp rẻ đi TƯƠNG ĐỐI so với spot.
    Đây là bất biến quan trọng nhất — đảo dấu ở đây thì mọi con số đảo ngược.
    """
    n = 30
    cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    # spot đứng yên; perp từ +2% basis thu hẹp về 0 -> ta phải LÃI
    S = _frame(np.full((n, len(cols)), 100.0), cols)
    perp = np.linspace(102.0, 100.0, n)
    P = _frame(np.tile(perp.reshape(-1, 1), (1, len(cols))), cols)
    F = _frame(np.zeros((n, len(cols))), cols)           # tắt funding để cô lập basis

    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=1,
                                           funding_window=2, round_trip_bps=0.0,
                                           min_universe=5))
    assert df["basis"].sum() > 0, "basis thu hẹp mà lại lỗ — quy ước dấu ngược"


def test_lo_khi_basis_NOI_RONG():
    n = 30
    cols = [f"S{i}" for i in range(10)]
    S = _frame(np.full((n, len(cols)), 100.0), cols)
    perp = np.linspace(100.0, 102.0, n)
    P = _frame(np.tile(perp.reshape(-1, 1), (1, len(cols))), cols)
    F = _frame(np.zeros((n, len(cols))), cols)
    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=1,
                                           funding_window=2, round_trip_bps=0.0,
                                           min_universe=5))
    assert df["basis"].sum() < 0


def test_thu_funding_khi_rate_duong():
    """Bên SHORT perp NHẬN funding khi rate dương. Nến 4h = nửa mốc funding 8h."""
    n = 20
    cols = [f"S{i}" for i in range(10)]
    P = _frame(np.full((n, len(cols)), 100.0), cols)
    S = _frame(np.full((n, len(cols)), 100.0), cols)
    F = _frame(np.full((n, len(cols)), 0.001), cols)     # +0.1% mỗi mốc
    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=1,
                                           funding_window=2, round_trip_bps=0.0,
                                           min_universe=5))
    active = df["funding"][df["funding"] != 0]
    assert len(active) > 0
    assert active.iloc[-1] == pytest.approx(0.001 * 0.5), "sai hệ số nửa mốc funding"


def test_tra_funding_khi_rate_am():
    n = 20
    cols = [f"S{i}" for i in range(10)]
    P = _frame(np.full((n, len(cols)), 100.0), cols)
    S = P.copy()
    F = _frame(np.full((n, len(cols)), -0.001), cols)
    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=1,
                                           funding_window=2, round_trip_bps=0.0,
                                           min_universe=5))
    assert df["funding"].sum() < 0


# ---------------------------------------------------------------- nhân quả
def test_xep_hang_khong_dung_funding_cung_thoi_diem():
    """
    Lỗi đã mắc: chọn top-k theo funding TẠI mốc t cho 25-43%/năm; dùng trung bình
    trượt nhân quả chỉ còn 5-9%. Chênh lệch đó hoàn toàn là nhìn trước.

    Kiểm chứng: đổi funding ở mốc CUỐI không được làm đổi kết quả các mốc trước.
    """
    rng = np.random.default_rng(5)
    n, cols = 60, [f"S{i}" for i in range(12)]
    P = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(cols))), axis=0), cols)
    S = _frame(P.to_numpy() * (1 + rng.normal(0, 0.0005, (n, len(cols)))), cols)
    F = _frame(rng.normal(0.0002, 0.0004, (n, len(cols))), cols)

    spec = BasisSpec(n_positions=3, rebalance_bars=5, funding_window=4, min_universe=5)
    base = backtest_basis(P, S, F, spec)

    F2 = F.copy()
    F2.iloc[-1] = 0.05                       # cú sốc khổng lồ ở mốc cuối
    after = backtest_basis(P, S, F2, spec)
    pd.testing.assert_frame_equal(base.iloc[:-2], after.iloc[:-2])


def test_gia_tuong_lai_khong_anh_huong_qua_khu():
    rng = np.random.default_rng(9)
    n, cols = 60, [f"S{i}" for i in range(12)]
    P = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(cols))), axis=0), cols)
    S = _frame(P.to_numpy() * 0.999, cols)
    F = _frame(rng.normal(0.0002, 0.0003, (n, len(cols))), cols)
    spec = BasisSpec(n_positions=3, rebalance_bars=5, funding_window=4, min_universe=5)

    base = backtest_basis(P, S, F, spec)
    P2 = P.copy(); P2.iloc[-1] *= 2.0
    after = backtest_basis(P2, S, F, spec)
    pd.testing.assert_frame_equal(base.iloc[:-2], after.iloc[:-2])


# ---------------------------------------------------------------- chi phí
def test_phi_ty_le_voi_thay_doi_danh_muc():
    """
    Lỗi đã mắc: tái cân bằng mỗi nến 4h khiến phí ăn -150%/năm. Lớp chiến lược này
    chỉ sống khi NẮM GIỮ. Test khoá lại: giữ lâu hơn thì tổng phí phải nhỏ hơn.
    """
    rng = np.random.default_rng(2)
    n, cols = 400, [f"S{i}" for i in range(15)]
    P = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(cols))), axis=0), cols)
    S = _frame(P.to_numpy() * 0.999, cols)
    F = _frame(rng.normal(0.0002, 0.0005, (n, len(cols))), cols)

    fast = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=2, min_universe=5))
    slow = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=100, min_universe=5))
    assert abs(fast["fee"].sum()) > abs(slow["fee"].sum()) * 5


def test_phi_bang_khong_khi_khong_giao_dich():
    n, cols = 50, [f"S{i}" for i in range(10)]
    P = _frame(np.full((n, len(cols)), 100.0), cols)
    S = P.copy()
    F = _frame(np.full((n, len(cols)), 0.0001), cols)
    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=10**6,
                                           min_universe=5))
    assert df["fee"].sum() == pytest.approx(0.0)


def test_tong_bang_cong_ba_thanh_phan():
    """Kế toán phải cộng đúng: total = funding + basis + fee."""
    rng = np.random.default_rng(1)
    n, cols = 120, [f"S{i}" for i in range(12)]
    P = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(cols))), axis=0), cols)
    S = _frame(P.to_numpy() * 0.999, cols)
    F = _frame(rng.normal(0.0002, 0.0004, (n, len(cols))), cols)
    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=20, min_universe=5))
    resid = (df["total"] - df["funding"] - df["basis"] - df["fee"]).abs().max()
    assert resid < 1e-12


# ---------------------------------------------------------------- vùng đệm
def test_vung_dem_giam_turnover():
    rng = np.random.default_rng(4)
    n, cols = 400, [f"S{i}" for i in range(20)]
    P = _frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, len(cols))), axis=0), cols)
    S = _frame(P.to_numpy() * 0.999, cols)
    F = _frame(rng.normal(0.0002, 0.0006, (n, len(cols))), cols)

    tight = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=20,
                                              exit_multiple=1, min_universe=5))
    buffered = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=20,
                                                 exit_multiple=4, min_universe=5))
    assert abs(buffered["fee"].sum()) <= abs(tight["fee"].sum())


def test_universe_qua_nho_thi_khong_mo_vi_the():
    n, cols = 40, ["A", "B"]
    P = _frame(np.full((n, 2), 100.0), cols)
    S = P.copy()
    F = _frame(np.full((n, 2), 0.001), cols)
    df = backtest_basis(P, S, F, BasisSpec(n_positions=3, rebalance_bars=5,
                                           min_universe=10))
    assert df["total"].abs().sum() == pytest.approx(0.0)
