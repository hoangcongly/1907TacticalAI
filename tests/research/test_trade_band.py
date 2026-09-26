"""
Dải không giao dịch trong backtest phải ra ĐÚNG các lệnh mà live đặt.

`research/trade_band.plan_band_trades` là bản sao trên trọng số của
`execution/portfolio_rebalancer.build_rebalance_plan`. Test gọi thẳng code live (giá 1,
bước khối lượng rất nhỏ để làm tròn không che mất logic) và so tập lệnh + sổ sau lệnh.
Hai đường tính khác nhau cho cùng một quyết định là họ lỗi F3.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.execution.portfolio_rebalancer import SymbolFilters, build_rebalance_plan
from aegis.research.backtest_v2 import CostModel, simulate_marked_to_market
from aegis.research.small_capital import drop_below_min_notional
from aegis.research.trade_band import plan_band_trades, smooth_signal

EQUITY = 3_000_000 / 26_300


def _book(rng, n=12, frac_new=0.3):
    """Sổ đang giữ (đã trôi giá) + đích mới trung lập, tổng |đích| = 1."""
    half = n // 2
    t = np.concatenate([rng.lognormal(0, 0.5, half), -rng.lognormal(0, 0.5, half)])
    t[:half] *= 0.5 / t[:half].sum()
    t[half:] *= 0.5 / -t[half:].sum()
    held = t * rng.lognormal(0, 0.25, n)              # trôi giá quanh đích
    swap = rng.random(n) < frac_new
    held[swap] = 0.0                                  # cặp mới vào
    out = rng.random(n) < frac_new / 2
    t[out] = 0.0                                      # cặp bị loại
    return held, t


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("band", [0.2, 0.5])
@pytest.mark.parametrize("lev", [1.5, 3.0])
def test_trung_lenh_cua_build_rebalance_plan(seed, band, lev):
    rng = np.random.default_rng(seed)
    held, t = _book(rng)
    budget = EQUITY * lev
    syms = [f"S{i:02d}USDT" for i in range(len(t))]
    # Live chuẩn hoá đích về tổng |w| = 1 rồi mới bỏ vị thế dưới min notional (F54).
    tn = t / np.abs(t).sum()
    tn = drop_below_min_notional(pd.DataFrame([tn], columns=syms), EQUITY, lev).iloc[0].to_numpy()
    d = np.abs(tn - held)
    live_cells = (tn != 0) & (d > 0)                 # nơi hai ngưỡng thật sự được so
    near = np.minimum(np.abs(d - band * np.abs(tn)), np.abs(d * budget - 5.0))[live_cells]
    if (near < 1e-7).any():
        pytest.skip("rơi đúng biên — chuyện làm tròn, không phải logic")

    filters = {s: SymbolFilters(s, 1e-6, 1e-9, 1e-9, 5.0) for s in syms}
    plan = build_rebalance_plan(dict(zip(syms, t)), dict(zip(syms, held * budget)),
                                {s: 1.0 for s in syms}, filters, equity=EQUITY, leverage=lev,
                                no_trade_band=band, neutrality_tolerance=0.02)
    live = {o.symbol for o in plan.orders}

    new, traded = plan_band_trades(held, tn, band, 5.0 / budget, 0.02)
    research = {s for s, x in zip(syms, traded) if x}
    assert research == live
    # Sổ sau lệnh: cặp có lệnh về đích, cặp không lệnh giữ nguyên.
    np.testing.assert_allclose(new[traded], tn[traded])
    np.testing.assert_allclose(new[~traded], held[~traded])


def test_khong_dai_khong_min_la_tai_can_bang_toan_phan():
    rng = np.random.default_rng(0)
    held, t = _book(rng)
    new, traded = plan_band_trades(held, t, 0.0, 0.0)
    np.testing.assert_allclose(new, t)
    assert traded.sum() == (np.abs(t - held) > 0).sum()


def test_dai_rong_hon_it_lenh_hon():
    rng = np.random.default_rng(1)
    books = [_book(rng) for _ in range(200)]           # CÙNG các sổ cho mọi độ rộng dải
    counts = [sum(plan_band_trades(h, t, band, 0.0)[1].sum() for h, t in books)
              for band in (0.0, 0.2, 0.5, 1.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_can_trung_lap_nhan_lai_lenh_bi_dai_bo():
    held = np.array([0.30, 0.30, -0.25, -0.25])       # net +0,10 sau trôi giá
    t = np.array([0.25, 0.25, -0.25, -0.25])
    new, traded = plan_band_trades(held, t, 0.5, 0.0, tolerance=0.02)
    assert traded[:2].any(), "sổ lệch 10% phải được cân lại dù điều chỉnh nằm trong dải"
    assert abs(new.sum()) / np.abs(new).sum() <= 0.02 + 1e-12


def test_mo_phong_dem_lenh_va_giam_turnover():
    rng = np.random.default_rng(2)
    n, m = 400, 20
    idx = pd.Index(1_600_000_000_000 + np.arange(n) * 4 * 3_600_000)
    cols = [f"C{i}" for i in range(m)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(0.01 * rng.standard_normal((n, m)), 0)),
                         index=idx, columns=cols)
    marks = idx[::18]
    W = pd.DataFrame(0.0, index=marks, columns=cols)
    for ts in marks:
        pick = rng.permutation(m)[:10]
        W.loc[ts, cols[pick[0]]] = W.loc[ts, cols[pick[1]]] = 0.0
        W.loc[ts, [cols[i] for i in pick[:5]]] = 0.1
        W.loc[ts, [cols[i] for i in pick[5:]]] = -0.1
    cost = CostModel(flat_bps=15.7)
    full = simulate_marked_to_market(W, close, None, cost)
    band = simulate_marked_to_market(W, close, None, cost, no_trade_band=0.5)
    assert band.meta["orders"].sum() < full.meta["orders"].sum()
    assert band.turnover.sum() <= full.turnover.sum() + 1e-12
    # Mặc định giữ nguyên hành vi cũ: đếm lệnh = số cặp đổi trọng số.
    assert full.meta["orders"].sum() > 0


def test_lam_muot_nhan_qua_va_giu_nan():
    idx = pd.RangeIndex(30)
    sig = pd.DataFrame(np.random.default_rng(3).standard_normal((30, 4)), index=idx)
    sig.iloc[10, 2] = np.nan
    sm = smooth_signal(sig, 2.0)
    assert np.isnan(sm.iloc[10, 2])
    sig2 = sig.copy()
    sig2.iloc[20:] = 99.0
    pd.testing.assert_frame_equal(smooth_signal(sig2, 2.0).iloc[:20], sm.iloc[:20])
