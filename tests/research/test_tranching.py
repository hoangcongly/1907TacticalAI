"""
Chia lô tái cân bằng + [FIX F53] lưới neo ở nến mới nhất.

Khoá bốn điều, theo thứ tự quan trọng:
1. PARITY: đích live (chỉ hàng cuối, `tranched_target_weights`) TRÙNG trọng số backtest
   (toàn lịch sử, `tranched_weight_panel`) tới sai số máy — K=1 lẫn K=3. Hai đường tính
   khác nhau cho cùng một quyết định là họ lỗi F3.
2. F53: mốc cuối của lưới LUÔN là nến mới nhất, không phải mốc cũ tới 68h.
3. Nhân quả: sửa dữ liệu SAU mốc neo không đổi đích của lô neo ở mốc đó.
4. Bất biến sổ: trung lập đô-la, gross <= 1 (netting giữa các lô, không phóng đại).
"""
import numpy as np
import pandas as pd
import pytest

from aegis.research.adaptive_combiner import CombinerSpec
from aegis.research.tranching import (
    anchored_marks, target_weights_at, tranche_offsets, tranched_target_weights,
    tranched_weight_panel,
)
from aegis.risk.portfolio import PortfolioSpec

STEP = 18
COMB = CombinerSpec(lookback=40, min_periods=8, t_threshold=1.0,
                    max_abs_weight=0.5, max_step=0.05)
SPEC = PortfolioSpec(mode="zscore_riskparity", n_positions=10, max_weight=0.2,
                     beta_neutral=False, vol_window=10)


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(3)
    n, m = 900, 30
    idx = pd.Index(1_600_000_000_000 + np.arange(n) * 4 * 3_600_000, name="ts")
    cols = [f"C{i:02d}USDT" for i in range(m)]
    alpha = rng.normal(0, 0.0004, m)
    r = alpha + 0.01 * rng.standard_normal((n, m))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    ret = close.pct_change()
    sigs = {
        "mom": ret.rolling(30, min_periods=10).sum(),
        "rev": -ret.rolling(6, min_periods=3).sum(),
        "noise": pd.DataFrame(rng.standard_normal((n, m)), index=idx, columns=cols),
    }
    return sigs, close


# ------------------------------------------------------------------ lưới
def test_luoi_neo_moc_cuoi_dung_nen_chi_dinh_f53():
    idx = pd.RangeIndex(100)
    for a in (0, 17, 18, 50, 99):
        m = anchored_marks(idx, STEP, a)
        assert m[-1] == a
        assert (np.diff(m) == STEP).all()
        assert m[0] == a % STEP                     # cùng pha, bắt đầu sớm nhất có thể


def test_luoi_neo_trung_luoi_research_cung_pha():
    idx = pd.RangeIndex(200)
    a = 157
    full = idx[a % STEP::STEP]
    assert list(anchored_marks(idx, STEP, a)) == [x for x in full if x <= a]


def test_do_lech_lo_phai_chia_het():
    assert tranche_offsets(18, 3) == [0, 6, 12]
    assert tranche_offsets(18, 1) == [0]
    with pytest.raises(ValueError):
        tranche_offsets(18, 4)
    with pytest.raises(ValueError):
        tranche_offsets(18, 0)


# ------------------------------------------------------------------ parity live / backtest
@pytest.mark.parametrize("k", [1, 3])
def test_dich_live_trung_backtest_toan_lich_su(market, k):
    sigs, close = market
    last = len(close) - 1
    gap = STEP // k
    live = tranched_target_weights(sigs, close, STEP, k, COMB, SPEC, 0.1, tail_only=True)
    panel = tranched_weight_panel(sigs, close, STEP, k, COMB, SPEC, 0.1, phase=last % gap)
    bt = panel.loc[close.index[last]]
    common = live.index.union(bt.index)
    diff = (live.reindex(common).fillna(0) - bt.reindex(common).fillna(0)).abs().max()
    assert diff < 1e-9, f"K={k}: live lệch backtest {diff:.2e}"


def test_k1_la_dung_trong_so_tai_nen_moi_nhat_khong_phai_moc_cu_f53(market):
    """Bản cũ lấy mốc cuối của lưới neo ở ĐẦU panel — có thể cũ tới 17 nến."""
    sigs, close = market
    last = len(close) - 1
    assert last % STEP != 0, "chọn độ dài panel sao cho lưới cũ KHÔNG trùng nến cuối"
    stale_mark = (last // STEP) * STEP                # mốc cuối của lưới close.index[::18]
    new = tranched_target_weights(sigs, close, STEP, 1, COMB, SPEC, 0.1)
    old = target_weights_at(sigs, close, stale_mark, STEP, COMB, SPEC, 0.1)
    assert last - stale_mark > 0
    assert not np.allclose(new.reindex(old.index).fillna(0), old, atol=1e-9), \
        "đích mới phải khác đích của mốc cũ — nếu trùng thì test không phân biệt được gì"


# ------------------------------------------------------------------ nhân quả
def test_sua_du_lieu_sau_moc_neo_khong_doi_dich_lo_do(market):
    sigs, close = market
    a = len(close) - 40
    base = target_weights_at(sigs, close, a, STEP, COMB, SPEC, 0.1)
    close2 = close.copy()
    close2.iloc[a + 1:] *= 3.0
    sigs2 = {k: v.copy() for k, v in sigs.items()}
    for v in sigs2.values():
        v.iloc[a + 1:] = -v.iloc[a + 1:] * 5
    after = target_weights_at(sigs2, close2, a, STEP, COMB, SPEC, 0.1)
    pd.testing.assert_series_equal(base, after, check_exact=False, atol=1e-12)


# ------------------------------------------------------------------ bất biến sổ
def test_so_gop_trung_lap_va_khong_phong_dai_gross(market):
    sigs, close = market
    w = tranched_target_weights(sigs, close, STEP, 3, COMB, SPEC, 0.1)
    assert abs(w.sum()) < 1e-9
    assert w.abs().sum() <= 1.0 + 1e-9
    assert (w > 0).any() and (w < 0).any()


def test_k_lon_hon_thi_so_gop_la_trung_binh_cac_lo(market):
    sigs, close = market
    last = len(close) - 1
    lots = [target_weights_at(sigs, close, last - off, STEP, COMB, SPEC, 0.1)
            for off in (0, 6, 12)]
    avg = pd.concat(lots, axis=1).fillna(0).sum(axis=1) / 3
    got = tranched_target_weights(sigs, close, STEP, 3, COMB, SPEC, 0.1)
    pd.testing.assert_series_equal(got.sort_index(), avg.sort_index(), check_names=False)


def test_bo_dem_theo_pha_khong_doi_ket_qua(market):
    """Quét nhiều K dùng chung bộ đệm pha — kết quả phải y hệt tính lại từ đầu."""
    sigs, close = market
    cache = {}
    tranched_weight_panel(sigs, close, STEP, 1, COMB, SPEC, 0.1, phase=6, cache=cache)
    cached = tranched_weight_panel(sigs, close, STEP, 3, COMB, SPEC, 0.1, cache=cache)
    fresh = tranched_weight_panel(sigs, close, STEP, 3, COMB, SPEC, 0.1)
    pd.testing.assert_frame_equal(cached, fresh)
    assert set(cache) == {0, 6, 12}
