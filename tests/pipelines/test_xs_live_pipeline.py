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
    # `_position_multiplier` trả về (hệ số, chi tiết) — chi tiết cần cho kiểm toán
    # sau sự cố; một con số trần trụi không nói được vì sao nó là con số đó.
    assert p._position_multiplier(state, 1000.0 * (1 - TIER1_DD - 0.01))[0] == 0.5
    assert p._position_multiplier(state, 1000.0)[0] == 1.0


def test_goal_overlay_tat_thi_khong_doi_hanh_vi(tmp_path):
    """Không cấu hình tầng phủ => hệ số y hệt hành vi cũ. Bất biến không hồi quy."""
    p = _pipe(tmp_path, peak_equity=1000.0)
    state = p.store.load()
    assert p.config.goal_overlay is None
    mult, info = p._position_multiplier(state, 1000.0)
    assert mult == 1.0
    assert info["goal_overlay"] is None


def test_goal_overlay_thieu_artifact_thi_bao_loi_va_giu_hanh_vi_cu(tmp_path):
    """
    Bật tầng phủ mà thiếu file chính sách: phải GIỮ hành vi cũ và GHI lỗi ra kết quả.

    Im lặng chạy tiếp là chế độ hỏng tệ nhất — người vận hành tưởng đã bật một thứ
    chưa từng được nạp.
    """
    from aegis.risk.goal_overlay import GoalOverlayConfig

    p = _pipe(tmp_path, peak_equity=1000.0)
    p.config.goal_overlay = GoalOverlayConfig(
        enabled=True, policy_path=str(tmp_path / "khong_ton_tai.npz"),
        start_equity=1000.0, deadline_ms=2 ** 62)
    mult, info = p._position_multiplier(state=p.store.load(), equity=1000.0)
    assert mult == 1.0
    assert "error" in info["goal_overlay"]


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


# ---------------------------------------------------------------------------
# Cò giảm rủi ro ngoài chu kỳ (tầng phủ mục tiêu)
# ---------------------------------------------------------------------------
def _goal_pipe(tmp_path, wallet, target=0.05):
    """Pipeline có tầng phủ mục tiêu đã bật, vốn gốc 1000, hạn chót còn xa."""
    import numpy as np

    from aegis.research.goal_dp import GoalSpec, solve_goal_dp
    from aegis.risk.goal_overlay import GoalOverlayConfig, save_policy

    rng = np.random.default_rng(11)
    r = rng.standard_t(df=4, size=2500) * 0.009 + 0.00035
    pol = solve_goal_dp(r, GoalSpec(target_return=target, horizon_periods=42,
                                    n_wealth=161, min_executable_leverage=1.58))
    path = str(tmp_path / "goal_policy.npz")
    save_policy(pol, path)

    # Vị thế trên sàn giả PHẢI khớp trạng thái nội bộ, nếu không `reconcile` sẽ HALT
    # trước khi tới được chốt nhịp — và ta sẽ test nhầm một đường khác.
    pos = {"AAAUSDT": 1.0}
    p = _pipe(tmp_path, client=_FakeClient(wallet=wallet, positions=pos), peak_equity=wallet)
    p.config.goal_overlay = GoalOverlayConfig(
        enabled=True, policy_path=path, start_equity=1000.0,
        deadline_ms=int(time.time() * 1000) + 5 * 24 * 3_600_000)
    st = p.store.load()
    st.positions = dict(pos)
    st.rebalance_count = 3
    st.last_rebalance_ms = int(time.time() * 1000) - 3_600_000   # mới 1h, CHƯA đến hạn
    p.store.save(st)
    return p


def test_dat_dich_thi_mo_chot_nhip_ngay_khong_cho_72h(tmp_path):
    """
    Chạm +5% giữa chu kỳ: phải giảm rủi ro NGAY, không chờ mốc tái cân bằng kế tiếp.

    Chờ tới mốc sau có thể muộn gần ba ngày, và biến động 72h của chiến lược (~3,7%)
    đủ để xoá sạch khoản lãi vừa chạm đích.
    """
    p = _goal_pipe(tmp_path, wallet=1080.0)          # +8% so với vốn gốc 1000
    res = p.run_once(skip_data_refresh=True)
    assert res.get("derisk_out_of_cycle") is True
    # Chốt nhịp đã mở => KHÔNG thể rơi vào nhánh HEARTBEAT (nhánh đó nằm trong `if not due`).
    assert res.get("action") != "HEARTBEAT", "đã đạt đích mà vẫn chỉ giám sát"
    # Và nó phải chạy tới nơi DÙ sàn giả không có panel nào: đường giảm rủi ro không
    # được phụ thuộc tầng dữ liệu. `data_age_hours is None` là dấu hiệu đã đi đường đó.
    assert res.get("data_age_hours") is None
    assert res.get("n_targets") == 0, "trọng số đích khi chốt lời phải là RỖNG"


def test_chua_dat_dich_thi_van_ton_trong_nhip_72h(tmp_path):
    """
    Cò chỉ mở theo hướng ĐÓNG. Chưa đạt đích thì không được phá nhịp — phá nhịp làm
    nhịp tín hiệu (18 nến) lệch nhịp thực thi, và cấu hình đang chạy sẽ không còn là
    cấu hình nào đã được kiểm định.
    """
    p = _goal_pipe(tmp_path, wallet=1010.0)          # +1%, chưa tới đích
    res = p.run_once(skip_data_refresh=True)
    assert res.get("derisk_out_of_cycle") is not True
    assert res["action"] == "HEARTBEAT"


def test_overlay_tat_thi_khong_co_co_nao(tmp_path):
    pos = {"AAAUSDT": 1.0}
    p = _pipe(tmp_path, client=_FakeClient(wallet=1080.0, positions=pos), peak_equity=1000.0)
    st = p.store.load()
    st.positions = dict(pos)
    st.rebalance_count = 3
    st.last_rebalance_ms = int(time.time() * 1000) - 3_600_000
    p.store.save(st)
    res = p.run_once(skip_data_refresh=True)
    assert res["action"] == "HEARTBEAT"
    assert "derisk_out_of_cycle" not in res
