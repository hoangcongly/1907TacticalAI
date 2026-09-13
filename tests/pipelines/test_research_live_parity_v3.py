"""
[FIX F3] Test PARITY: trọng số live phải TRÙNG KHỚP trọng số research.

Đây là test quan trọng nhất của toàn bộ đợt nâng cấp v3. Lỗi F3 của hệ thống cũ
không phải sai công thức mà là CÓ HAI ĐƯỜNG TÍNH: research tự tính feature một kiểu,
live tính một kiểu, cặp nào thiếu thì live âm thầm điền 0.0. Không có test nào bắt
được điều đó vì cả hai đường đều "chạy đúng" theo tiêu chuẩn riêng của nó.

Test này so trực tiếp hai đầu ra trên cùng dữ liệu. Nếu ai đó sau này viết lại công
thức ở tầng live "cho nhanh", test sẽ đỏ ngay.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from aegis.data.panel_v2 import load_funding_panel_v2, load_panel_v2
from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights

UNIVERSE_FILE = pathlib.Path("artifacts/universe_wide.json")
DATA_ROOT = pathlib.Path("data/binance")

pytestmark = pytest.mark.skipif(
    not UNIVERSE_FILE.is_file() or not any(DATA_ROOT.glob("*_1h.parquet")),
    reason="cần dữ liệu thật: chạy scripts/download_wide_universe.py",
)

REBAL = 18
N_POS = 12


@pytest.fixture(scope="module")
def market():
    syms = json.load(open(UNIVERSE_FILE, encoding="utf-8"))
    panel = load_panel_v2(syms, "4h", source_interval="1h", min_coverage=0.15)
    funding = load_funding_panel_v2(syms, panel["close"].index).reindex(
        columns=panel["close"].columns)
    return panel, funding


def _weights_research(panel, funding, tail_only: bool):
    """Đường research; `tail_only=True` mô phỏng đúng cách live cắt đuôi chuỗi."""
    close = panel["close"]
    sigs = {n: build_signal(n, panel, funding) for n in SIGNAL_REGISTRY}
    marks = close.index[::REBAL]
    combined = combine_adaptive(
        {k: v.reindex(marks) for k, v in sigs.items()}, close.reindex(marks),
        CombinerSpec(lookback=500, min_periods=120, t_threshold=2.0,
                     max_abs_weight=0.20, max_step=0.05), top_frac=0.10)
    spec = PortfolioSpec(mode="zscore_riskparity", n_positions=N_POS,
                         max_weight=0.20, beta_neutral=False)
    idx = marks[-(spec.vol_window + 5):] if tail_only else marks
    return build_weights(combined.reindex(idx), close.reindex(idx), spec)


def test_cat_duoi_chuoi_khong_doi_trong_so_hang_cuoi(market):
    """
    Live chỉ dựng trọng số cho đuôi chuỗi để tiết kiệm thời gian. Việc cắt đó chỉ
    hợp lệ nếu hàng cuối KHÔNG đổi — tức cửa sổ ước lượng biến động đã đủ dài.
    """
    panel, funding = market
    full = _weights_research(panel, funding, tail_only=False).iloc[-1]
    tail = _weights_research(panel, funding, tail_only=True).iloc[-1]

    common = full.index.intersection(tail.index)
    diff = float((full[common] - tail[common]).abs().max())
    assert diff < 1e-9, f"cắt đuôi làm đổi trọng số tới {diff:.2e} — cửa sổ chưa đủ dài"


def test_live_khop_research(market, monkeypatch):
    """Trọng số của `CrossSectionalLivePipeline` engine=v3 phải trùng research."""
    from aegis.pipelines import xs_live_pipeline as xlp

    panel, funding = market
    expected = _weights_research(panel, funding, tail_only=True).iloc[-1]
    expected = {s: float(w) for s, w in expected.items() if abs(w) > 1e-12}

    cfg = xlp.LiveConfig.from_artifacts("artifacts/strategy_v3.json")
    pipe = xlp.CrossSectionalLivePipeline.__new__(xlp.CrossSectionalLivePipeline)
    pipe.config = cfg
    # Bỏ qua bước lọc universe theo sàn: test này chỉ kiểm chứng phép TÍNH.
    monkeypatch.setattr(pipe, "resolve_universe",
                        lambda: list(json.load(open(UNIVERSE_FILE, encoding="utf-8"))),
                        raising=False)

    got, ts = pipe._compute_target_weights_v3()

    assert set(got) == set(expected), (
        f"tập cặp khác nhau: chỉ live {sorted(set(got)-set(expected))}, "
        f"chỉ research {sorted(set(expected)-set(got))}")
    for sym in expected:
        assert abs(got[sym] - expected[sym]) < 1e-9, (
            f"{sym}: live {got[sym]:.10f} != research {expected[sym]:.10f}")
    assert ts == int(panel["close"].index[-1])


def test_trong_so_trung_lap_va_dung_so_vi_the(market, monkeypatch):
    """Bất biến của danh mục: net ~ 0, gross ~ 1, đúng số vị thế."""
    from aegis.pipelines import xs_live_pipeline as xlp

    cfg = xlp.LiveConfig.from_artifacts("artifacts/strategy_v3.json")
    pipe = xlp.CrossSectionalLivePipeline.__new__(xlp.CrossSectionalLivePipeline)
    pipe.config = cfg
    monkeypatch.setattr(pipe, "resolve_universe",
                        lambda: list(json.load(open(UNIVERSE_FILE, encoding="utf-8"))),
                        raising=False)

    w, _ = pipe._compute_target_weights_v3()
    vals = np.array(list(w.values()))

    assert abs(vals.sum()) < 1e-6, f"net exposure {vals.sum():.2e} != 0"
    assert abs(np.abs(vals).sum() - 1.0) < 1e-6, f"gross {np.abs(vals).sum():.4f} != 1.0"
    assert (vals > 0).any() and (vals < 0).any(), "phải có cả long lẫn short"
    assert len(vals) <= cfg.n_positions, f"{len(vals)} vị thế > trần {cfg.n_positions}"

    # Trần trọng số là XẤP XỈ, và điều đó có chủ ý — xem
    # `tests/risk/test_portfolio_v2.py::test_tran_trong_so_duoc_ap_trong_dung_sai_da_cong_bo`.
    # `build_weights` kết thúc bằng phép chiếu trung lập chứ không bằng kẹp trần, vì
    # thứ tự ngược lại để sổ ra khỏi hàm với net exposure khác 0 (đo được tới 20%
    # gross). Đánh đổi: trung lập TUYỆT ĐỐI (khẳng định ở trên, sai số 1e-6), trần
    # XẤP XỈ. Đây là đánh đổi đúng hướng — trần chỉ chống tập trung, còn trung lập
    # là toàn bộ lý do chiến lược này tồn tại.
    worst = float(np.abs(vals).max())
    assert worst <= cfg.max_weight * 1.10, (
        f"vượt trần {worst/cfg.max_weight - 1:.2%} — quá nhiều, không còn là sai số chiếu")
