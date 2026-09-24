"""
Động lượng PHẦN DƯ (`rmom_*`) — khoá CƠ CHẾ, không chỉ khoá hình dạng đầu ra.

Lý do tồn tại của họ tín hiệu này là một mệnh đề kiểm chứng được: trong thị trường
mà các coin chỉ khác nhau về BETA, động lượng thô xếp hạng theo beta còn động lượng
phần dư thì không. Nếu mệnh đề đó gãy, tín hiệu chỉ còn là một bản sao tốn kém của
`mom_*`, và mọi so sánh trong `scripts/residual_study.py` đo nhầm thứ.
"""
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from aegis.research.signal_library import (
    FAMILIES, SIGNAL_REGISTRY, build_signal, residual_returns,
)
from aegis.research.strategy_v3 import V3_SIGNALS

RMOM = ("rmom_fast", "rmom_mid", "rmom_slow", "rmom_vlong")


def _beta_market(n_bars=2400, n_assets=30, drift=0.0, idio=0.004, seed=0):
    """Panel mà mỗi coin = beta_i x thị trường + nhiễu riêng, KHÔNG có drift riêng."""
    rng = np.random.default_rng(seed)
    betas = np.linspace(0.4, 2.2, n_assets)
    m = drift + 0.01 * rng.standard_normal(n_bars)
    r = m[:, None] * betas[None, :] + idio * rng.standard_normal((n_bars, n_assets))
    idx = pd.RangeIndex(n_bars)
    cols = [f"C{i:02d}" for i in range(n_assets)]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)
    panel = {"close": close, "high": close * 1.002, "low": close * 0.998,
             "volume": pd.DataFrame(1e6, index=idx, columns=cols)}
    funding = pd.DataFrame(0.0, index=idx, columns=cols)
    return panel, funding, pd.Series(betas, index=cols)


def test_phan_du_truc_giao_voi_thi_truong():
    panel, _, _ = _beta_market(seed=1)
    r = np.log(panel["close"] / panel["close"].shift(1))
    mkt = r.mean(axis=1)
    e = residual_returns(panel["close"]).iloc[400:]
    raw_corr = r.iloc[400:].corrwith(mkt.iloc[400:]).abs().median()
    res_corr = e.corrwith(mkt.iloc[400:]).abs().median()
    assert raw_corr > 0.8, "panel thử phải do thị trường chi phối, nếu không test vô nghĩa"
    assert res_corr < 0.15, f"phần dư còn dính thị trường: |corr| trung vị {res_corr:.3f}"


def test_dong_luong_tho_xep_theo_beta_con_phan_du_thi_khong():
    """
    ĐÚNG cơ chế Blitz-Huij-Martens: sau một đợt thị trường tăng mạnh, động lượng thô
    đứng đầu là coin beta cao — không vì thông tin riêng nào cả. Phần dư phải miễn nhiễm.
    """
    panel, funding, betas = _beta_market(drift=0.002, seed=2)   # thị trường có xu hướng
    rows = slice(800, None)

    def mean_rank_corr(sig):
        s = sig.iloc[rows].dropna(how="all")
        return s.apply(lambda row: row.corr(betas, method="spearman"), axis=1).mean()

    raw = mean_rank_corr(build_signal("mom_slow", panel, funding))
    res = mean_rank_corr(build_signal("rmom_slow", panel, funding))
    # Đo trên 4 seed: thô 0,60-0,97, phần dư 0,00-0,07.
    assert raw > 0.6, f"động lượng thô phải xếp theo beta ở panel này, đo {raw:.2f}"
    assert abs(res) < 0.15, f"động lượng phần dư vẫn xếp theo beta: {res:.2f}"


def test_phan_du_giu_duoc_thong_tin_rieng():
    """Nếu có drift RIÊNG thật sự, phần dư phải nhìn thấy nó — không được khử quá tay."""
    panel, funding, betas = _beta_market(drift=0.002, seed=3)
    rng = np.random.default_rng(3)
    alpha = pd.Series(rng.permutation(np.linspace(-1, 1, len(betas))), index=betas.index)
    r = np.log(panel["close"] / panel["close"].shift(1)).fillna(0.0) + 0.0008 * alpha
    close = 100.0 * np.exp(r.cumsum())
    panel = {**panel, "close": close, "high": close * 1.002, "low": close * 0.998}
    s = build_signal("rmom_slow", panel, funding).iloc[800:].dropna(how="all")
    corr = s.apply(lambda row: row.corr(alpha, method="spearman"), axis=1).mean()
    assert corr > 0.6, f"phần dư đánh mất tín hiệu riêng: corr với alpha thật {corr:.2f}"


@pytest.mark.parametrize("name", RMOM)
def test_nhan_qua_khi_sua_nhieu_nen_tuong_lai(name):
    """Sửa 20 nến CUỐI (cả thị trường lẫn từng coin) không được đổi quá khứ."""
    panel, funding, _ = _beta_market(n_bars=1200, seed=4)
    tampered = {k: v.copy() for k, v in panel.items()}
    for k in ("close", "high", "low"):
        tampered[k].iloc[-20:] *= np.linspace(0.5, 3.0, tampered[k].shape[1])
    a = build_signal(name, panel, funding).iloc[:-20]
    b = build_signal(name, tampered, funding).iloc[:-20]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_ho_phan_du_khong_lot_vao_v3_hay_families():
    """
    Thêm vào registry KHÔNG được đổi bản đã kiểm định (F38) hay các script dựng theo
    họ (`build_all_families` duyệt `FAMILIES`).
    """
    assert all(n in SIGNAL_REGISTRY for n in RMOM)
    assert not set(RMOM) & set(V3_SIGNALS)
    assert len(V3_SIGNALS) == 26
    assert "residual_momentum" not in FAMILIES


def test_script_nghien_cuu_khong_duyet_ca_registry():
    """
    [FIX F50] Script dựng tín hiệu bằng cách duyệt CẢ `SIGNAL_REGISTRY` sẽ tự đổi kết
    quả mỗi khi registry có thêm tín hiệu — đó là cách `validate_v3.py` và 4 script
    khác từng lặng lẽ chạy 36 tín hiệu thay vì 26. Registry là kho, không phải cấu hình.
    """
    pat = re.compile(r"build_signal\([^)]*\)[^}\n]*\bfor\s+\w+\s+in\s+SIGNAL_REGISTRY\b")
    root = pathlib.Path(__file__).resolve().parents[2]
    offenders = [p.name for p in sorted((root / "scripts").glob("*.py"))
                 if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"script duyệt cả registry khi dựng tín hiệu: {offenders}"
