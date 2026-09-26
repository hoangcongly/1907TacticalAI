"""
[FIX F54] Backtest vốn nhỏ phải bỏ vị thế dưới min notional ĐÚNG như live.

Khoá ba điều:
1. PARITY với live: tập vị thế `drop_below_min_notional` quy về 0 TRÙNG tập mà
   `build_rebalance_plan` bỏ vì min_notional — gọi thẳng code live, không chép lại.
2. Không co giãn lại: phần còn lại giữ nguyên giá trị (live không co giãn), nên gross co.
3. Trọng số ĐỀU ở đòn bẩy sàn 6n/vốn không mất vị thế nào — lý do cấu hình vốn nhỏ
   dùng `rank_binary`.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.execution.portfolio_rebalancer import SymbolFilters, build_rebalance_plan
from aegis.research.small_capital import drop_below_min_notional, drop_stats, min_leverage

CAPITAL = 1_000_000 / 26_300          # 1 triệu VND


def _zscore_like(rng, n=12):
    """Trọng số kiểu zscore_riskparity: chênh nhau nhiều lần, trung lập, tổng |w| = 1."""
    half = n // 2
    long_ = rng.lognormal(0.0, 0.6, half)
    short = rng.lognormal(0.0, 0.6, half)
    w = np.concatenate([long_ / long_.sum() * 0.5, -short / short.sum() * 0.5])
    return pd.Series(w, index=[f"S{i:02d}USDT" for i in range(n)])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("lev", [1.9, 2.5, 4.0])
def test_trung_tap_bo_cua_build_rebalance_plan(seed, lev):
    rng = np.random.default_rng(seed)
    w = _zscore_like(rng)
    # Đẩy khỏi đúng biên $5: làm tròn XUỐNG theo step_size của live có thể đẩy một vị
    # thế $5,0000001 xuống dưới ngưỡng — đó là chuyện làm tròn, không phải phép lọc.
    usd = w.abs() * CAPITAL * lev
    if ((usd - 5.0).abs() < 0.01).any():
        pytest.skip("trọng số rơi đúng biên $5")
    prices = {s: 1.0 for s in w.index}
    filters = {s: SymbolFilters(s, 1e-6, 1e-8, 1e-8, 5.0) for s in w.index}
    plan = build_rebalance_plan(w.to_dict(), {}, prices, filters, equity=CAPITAL, leverage=lev)
    live_dropped = {s for s, why in plan.skipped.items() if "min_notional" in why}

    got = drop_below_min_notional(w.to_frame().T, CAPITAL, lev).iloc[0]
    research_dropped = {s for s in w.index if got[s] == 0.0}
    assert research_dropped == live_dropped
    assert {o.symbol for o in plan.orders} == set(w.index) - live_dropped


def test_khong_co_gian_lai_nen_gross_co_va_net_lech():
    rng = np.random.default_rng(1)
    w = _zscore_like(rng).to_frame().T
    after = drop_below_min_notional(w, CAPITAL, 1.9)
    kept = after.iloc[0] != 0
    assert kept.sum() < w.shape[1], "phải có vị thế bị bỏ, nếu không test không đo gì"
    pd.testing.assert_series_equal(after.iloc[0][kept], w.iloc[0][kept])
    st = drop_stats(w, after)
    assert st["gross_kept"] < 1.0
    assert st["net_over_gross"] > 0.0


def test_trong_so_deu_o_don_bay_san_khong_mat_vi_the():
    for n in (6, 8, 12, 20, 50):
        lev = min_leverage(n, CAPITAL)
        w = pd.DataFrame([[1.0 / n] * (n // 2) + [-1.0 / n] * (n // 2)])
        after = drop_below_min_notional(w, CAPITAL, lev)
        assert (after != 0).all().all(), f"n={n}: trọng số đều ở {lev:.2f}x không được mất vị thế"


def test_von_lon_khong_bo_gi():
    rng = np.random.default_rng(2)
    w = _zscore_like(rng).to_frame().T
    after = drop_below_min_notional(w, 5_000.0, 2.0)
    pd.testing.assert_frame_equal(after, w)


def test_hang_rong_va_nan_giu_nguyen():
    w = pd.DataFrame([[np.nan, np.nan], [0.0, 0.0], [0.5, -0.5]])
    after = drop_below_min_notional(w, CAPITAL, 2.0)
    assert after.iloc[0].isna().all()
    assert (after.iloc[1] == 0).all()
    assert (after.iloc[2] == w.iloc[2]).all()


# ------------------------------------------------------------------ ngắt mạch như live
def test_kill_la_hap_thu_va_von_dung_yen():
    from aegis.research.small_capital import equity_with_breakers
    r = np.zeros((1, 40))
    r[0, 2] = -0.05          # 6x -> sụt 30% -> kill
    r[0, 3:] = 0.05          # thị trường hồi mạnh, nhưng đã kill thì không được hưởng
    out = equity_with_breakers(r, 6.0)
    assert out["killed"][0]
    assert out["final"][0] == pytest.approx(0.70)


def test_tier1_cat_nua_vi_the_tu_moc_tai_can_bang_ke_tiep():
    from aegis.research.small_capital import equity_with_breakers
    r = np.zeros((1, 36))
    r[0, 0] = -0.06          # 2x -> sụt 12% -> TIER1 ở mốc 18
    r[0, 20] = 0.01          # sau mốc 18: chỉ ăn 0,5 x 2x = 1x
    out = equity_with_breakers(r, 2.0, rebalance_every=18)
    assert not out["killed"][0]
    assert out["final"][0] == pytest.approx(0.88 * 1.01)
    assert out["tier1_share"][0] == pytest.approx(0.5)


def test_khong_ngat_mach_khi_khong_sut():
    from aegis.research.small_capital import equity_with_breakers
    r = np.full((3, 50), 0.001)
    out = equity_with_breakers(r, 3.0)
    assert not out["killed"].any()
    np.testing.assert_allclose(out["final"], 1.003 ** 50)
