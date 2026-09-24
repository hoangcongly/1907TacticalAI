"""
[FIX F53] + chia lô, kiểm ở tầng LIVE mà không cần dữ liệu thật.

`test_research_live_parity_v3.py` bị skip cả module khi thiếu `data/` — nên chính nơi
code được sửa lại là nơi parity không được kiểm. Test này gọi THẲNG
`CrossSectionalLivePipeline._compute_target_weights_v3` trên panel tổng hợp đủ 26 tín
hiệu, rồi so với đường BACKTEST toàn lịch sử (`tranched_weight_panel`) — một đường tính
khác hẳn (dựng trọng số trên toàn lưới, ffill theo pha) cho cùng một quyết định.
"""
import numpy as np
import pandas as pd
import pytest

import aegis.pipelines.xs_live_pipeline as xlp
from aegis.research.adaptive_combiner import CombinerSpec
from aegis.research.signal_library import build_signal
from aegis.research.strategy_v3 import V3_SIGNALS
from aegis.research.tranching import tranched_weight_panel
from aegis.risk.portfolio import PortfolioSpec

N_BARS, N_SYM, STEP = 1100, 24, 18


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(11)
    idx = pd.Index(1_700_000_000_000 + np.arange(N_BARS) * 4 * 3_600_000, name="ts")
    cols = [f"S{i:02d}USDT" for i in range(N_SYM)]
    r = rng.normal(0, 0.0004, N_SYM) + 0.012 * rng.standard_normal((N_BARS, N_SYM))
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    panel = {
        "close": close,
        "high": close * (1 + np.abs(r) + 0.002),
        "low": close * (1 - np.abs(r) - 0.002),
        "volume": pd.DataFrame(np.exp(rng.normal(14, 1, r.shape)), index=idx, columns=cols),
        "ofi": pd.DataFrame(np.tanh(40 * r + rng.standard_normal(r.shape)), index=idx,
                            columns=cols),
    }
    funding = pd.DataFrame(1e-4 * rng.standard_normal(r.shape), index=idx, columns=cols)
    return panel, funding


def _cfg(k):
    return xlp.LiveConfig(universe=[], engine="v3", n_positions=10, max_weight=0.2,
                          rebalance_bars=STEP, n_tranches=k, min_history_bars=200,
                          combiner_lookback=40, combiner_min_periods=8)


def _live(market, monkeypatch, k):
    panel, funding = market
    pipe = xlp.CrossSectionalLivePipeline.__new__(xlp.CrossSectionalLivePipeline)
    pipe.config = _cfg(k)
    cols = list(panel["close"].columns)
    monkeypatch.setattr(pipe, "resolve_universe", lambda *_, **__: cols, raising=False)
    monkeypatch.setattr(xlp, "load_panel_v2", lambda *a, **k: panel)
    monkeypatch.setattr(xlp, "load_funding_panel_v2", lambda *a, **k: funding)
    return pipe._compute_target_weights_v3()


@pytest.mark.parametrize("k", [1, 3])
def test_live_trung_backtest_chia_lo(market, monkeypatch, k):
    panel, funding = market
    got, ts = _live(market, monkeypatch, k)

    c = _cfg(k)
    close = panel["close"]
    sigs = {n: build_signal(n, panel, funding) for n in V3_SIGNALS}
    comb = CombinerSpec(lookback=c.combiner_lookback, min_periods=c.combiner_min_periods,
                        t_threshold=c.combiner_t_threshold,
                        max_abs_weight=c.combiner_max_abs_weight,
                        max_step=c.combiner_max_step)
    spec = PortfolioSpec(mode=c.weight_mode, n_positions=c.n_positions, top_frac=c.top_frac,
                         max_weight=c.max_weight, beta_neutral=False)
    last = len(close) - 1
    W = tranched_weight_panel(sigs, close, STEP, k, comb, spec, c.top_frac,
                              phase=last % (STEP // k))
    exp = W.loc[close.index[last]]
    exp = exp[exp.abs() > 1e-12]

    assert set(got) == set(exp.index)
    worst = max(abs(got[s] - exp[s]) for s in exp.index)
    assert worst < 1e-9, f"K={k}: live lệch backtest {worst:.2e}"
    assert ts == int(close.index[-1])          # mốc báo cáo NAY đúng là mốc của tín hiệu


def test_chia_lo_3_giu_nhieu_cap_hon_mot_lo_va_van_trung_lap(market, monkeypatch):
    one, _ = _live(market, monkeypatch, 1)
    three, _ = _live(market, monkeypatch, 3)
    assert len(three) >= len(one)
    v = np.array(list(three.values()))
    assert abs(v.sum()) < 1e-9 and np.abs(v).sum() <= 1 + 1e-9


def test_from_artifacts_doc_n_tranches_va_chia_nhip(tmp_path):
    import json
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps(["AAAUSDT"]))
    art = tmp_path / "c.json"
    art.write_text(json.dumps({"config": {"universe_file": str(uni), "n_positions": 50,
                                          "rebalance_every": 18, "period_hours": 72,
                                          "n_tranches": 3}}))
    cfg = xlp.LiveConfig.from_artifacts(str(art))
    assert cfg.n_tranches == 3
    assert cfg.rebalance_hours == pytest.approx(24.0)
