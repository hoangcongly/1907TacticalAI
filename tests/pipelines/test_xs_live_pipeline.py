"""
Test các CHỐT CHẶN AN TOÀN của pipeline live cross-sectional.

Trọng tâm không phải "chiến lược có lãi không" (backtest lo việc đó) mà là
"hệ thống có từ chối giao dịch đúng lúc cần từ chối không". Mỗi test dưới đây
tương ứng một cách hệ thống có thể mất tiền vì hành động trên trạng thái sai.
"""
import time

import pytest

from aegis.core.state_store import LiveState, StateStore
from aegis.pipelines.xs_live_pipeline import (
    TIER1_DD, TIER2_DD, TIER3_DD, CrossSectionalLivePipeline, LiveConfig,
)


class _FakeClient:
    """Sàn giả — kiểm soát hoàn toàn số dư và vị thế trả về."""
    credentials = None

    def __init__(self, wallet=1000.0, upnl=0.0, positions=None):
        self._wallet, self._upnl = wallet, upnl
        self._positions = positions or {}

    def balance_usdt(self):
        return {"wallet_balance": self._wallet, "available_balance": self._wallet,
                "unrealized_pnl": self._upnl}

    def position_risk(self, symbol=None):
        return [{"symbol": s, "positionAmt": str(q), "unRealizedProfit": "0"}
                for s, q in self._positions.items()]

    def open_orders(self, symbol=None):
        return []

    def symbol_filters(self, symbol):
        return {"tick_size": 0.001, "step_size": 0.1, "min_qty": 0.1, "min_notional": 5.0}

    def exchange_info(self, symbol=None):
        return {"symbols": [
            {"symbol": s, "status": "TRADING", "quoteAsset": "USDT",
             "contractType": "PERPETUAL"}
            for s in ("AAAUSDT", "BBBUSDT")
        ]}

    def _request(self, method, path, params=None, signed=False):
        if "ticker" in path:
            return [{"symbol": s, "quoteVolume": "1e12"} for s in ("AAAUSDT", "BBBUSDT")]
        return {}


def _pipe(tmp_path, client=None, **state_kw):
    store = StateStore(str(tmp_path / "s.json"))
    if state_kw:
        store.save(LiveState(**state_kw))
    cfg = LiveConfig(universe=["AAAUSDT", "BBBUSDT"], leverage=2.0)
    p = CrossSectionalLivePipeline(config=cfg, client=client or _FakeClient(),
                                   state_store=store, dry_run=True)
    return p


# ------------------------------------------------------------ ngắt mạch rủi ro
def test_kill_switch_blocks_all_trading(tmp_path):
    """Kill switch đã bật thì KHÔNG được giao dịch, kể cả sau restart."""
    p = _pipe(tmp_path, is_dead=True, peak_equity=1000.0)
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HALT"
    assert res["reason"] == "CIRCUIT_BREAKER"
    assert "KILL_SWITCH" in res["detail"]


def test_freeze_window_blocks_trading(tmp_path):
    future = int(time.time() * 1000) + 3_600_000
    p = _pipe(tmp_path, frozen_until_ms=future, peak_equity=1000.0)
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HALT" and "ĐÓNG BĂNG" in res["detail"]


def test_tier3_drawdown_triggers_permanent_kill(tmp_path):
    """Drawdown vượt Tier 3 phải kích hoạt kill switch VĨNH VIỄN và ghi xuống đĩa."""
    client = _FakeClient(wallet=1000.0 * (1 - TIER3_DD - 0.01))
    p = _pipe(tmp_path, client=client, peak_equity=1000.0)
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HALT" and "TIER 3" in res["detail"]
    assert p.store.load().is_dead is True, "kill switch phải được lưu bền vững"


def test_tier2_drawdown_freezes_for_24h(tmp_path):
    client = _FakeClient(wallet=1000.0 * (1 - TIER2_DD - 0.01))
    p = _pipe(tmp_path, client=client, peak_equity=1000.0)
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HALT" and "TIER 2" in res["detail"]
    state = p.store.load()
    assert state.frozen_until_ms > int(time.time() * 1000)
    assert state.is_dead is False, "Tier 2 chỉ đóng băng, không giết vĩnh viễn"


def test_tier1_halves_position_size(tmp_path):
    p = _pipe(tmp_path, peak_equity=1000.0)
    state = p.store.load()
    assert p._position_multiplier(state, 1000.0 * (1 - TIER1_DD - 0.01)) == 0.5
    assert p._position_multiplier(state, 1000.0) == 1.0


def test_peak_equity_ratchets_up_only(tmp_path):
    """Đỉnh vốn chỉ tăng — nếu tụt theo equity thì drawdown mất hết ý nghĩa."""
    p = _pipe(tmp_path, client=_FakeClient(wallet=800.0), peak_equity=1000.0)
    p.run_once(skip_data_refresh=True)
    assert p.store.load().peak_equity == 1000.0


# ------------------------------------------------------------ đối chiếu sổ sách
def test_position_mismatch_halts_trading(tmp_path):
    """
    Sàn có vị thế mà trạng thái nội bộ không biết -> DỪNG.
    Với danh mục market-neutral, một vị thế lạ phá vỡ tính trung lập.
    """
    client = _FakeClient(positions={"ZZZUSDT": 100.0})
    p = _pipe(tmp_path, client=client, peak_equity=1000.0,
              positions={"AAAUSDT": 1.0}, rebalance_count=5)
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HALT" and res["reason"] == "RECONCILIATION_FAILED"


def test_first_run_adopts_existing_positions(tmp_path, monkeypatch):
    """Lần chạy đầu (chưa có trạng thái) nhận vị thế sẵn có làm mốc, không DỪNG."""
    client = _FakeClient(positions={"AAAUSDT": 10.0})
    p = _pipe(tmp_path, client=client, peak_equity=0.0, rebalance_count=0)
    monkeypatch.setattr(p, "compute_target_weights",
                        lambda **kw: ({"AAAUSDT": 0.5, "BBBUSDT": -0.5}, int(time.time() * 1000)))
    res = p.run_once(skip_data_refresh=True)
    assert res.get("reason") != "RECONCILIATION_FAILED"


# ------------------------------------------------------------- độ tươi dữ liệu
def test_stale_data_halts_trading(tmp_path, monkeypatch):
    """Dữ liệu cũ -> không giao dịch. Bỏ lỡ một lượt an toàn hơn giao dịch mù."""
    p = _pipe(tmp_path, peak_equity=1000.0)
    old_ts = int(time.time() * 1000) - 48 * 3_600_000
    monkeypatch.setattr(p, "compute_target_weights", lambda **kw: ({"AAAUSDT": 1.0}, old_ts))
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HALT" and res["reason"] == "STALE_DATA"


def test_fresh_data_passes_the_gate(tmp_path, monkeypatch):
    p = _pipe(tmp_path, peak_equity=1000.0)
    now = int(time.time() * 1000)
    monkeypatch.setattr(p, "compute_target_weights", lambda **kw: ({"AAAUSDT": 0.5, "BBBUSDT": -0.5}, now))
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "DRY_RUN"


# ------------------------------------------------------------- sức chứa vốn
def test_targets_capped_by_capital(tmp_path, monkeypatch):
    """Vốn nhỏ -> cắt bớt vị thế, giữ lại tín hiệu MẠNH NHẤT."""
    p = _pipe(tmp_path, client=_FakeClient(wallet=20.0), peak_equity=20.0)
    now = int(time.time() * 1000)
    many = {f"S{i}USDT": (0.1 * (i + 1)) * (1 if i % 2 else -1) for i in range(20)}
    monkeypatch.setattr(p, "compute_target_weights", lambda **kw: (many, now))
    res = p.run_once(skip_data_refresh=True)
    assert res["n_targets"] <= res["capacity"]
    assert res["n_targets"] < 20


def test_dry_run_never_submits(tmp_path, monkeypatch):
    p = _pipe(tmp_path, peak_equity=1000.0)
    now = int(time.time() * 1000)
    monkeypatch.setattr(p, "compute_target_weights", lambda **kw: ({"AAAUSDT": 0.5, "BBBUSDT": -0.5}, now))
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "DRY_RUN" and "submitted" not in res
