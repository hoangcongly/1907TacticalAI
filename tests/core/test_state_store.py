"""Test trạng thái bền vững — thứ giữ cho ngắt mạch rủi ro không mất trí nhớ."""
import json
import pathlib

import pytest

from aegis.core.state_store import LiveState, StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(str(tmp_path / "state.json"))


def test_roundtrip_preserves_everything(store):
    s = LiveState(peak_equity=1234.5, is_dead=True, frozen_until_ms=999,
                  positions={"XRPUSDT": -12.5}, rebalance_count=7)
    store.save(s)
    loaded = store.load()
    assert loaded.peak_equity == 1234.5
    assert loaded.is_dead is True
    assert loaded.positions == {"XRPUSDT": -12.5}
    assert loaded.rebalance_count == 7


def test_missing_file_gives_fresh_state(store):
    s = store.load()
    assert s.peak_equity == 0.0 and s.positions == {}


def test_peak_equity_survives_restart(store):
    """
    LÕI CỦA VẤN ĐỀ: đỉnh vốn phải sống sót qua restart.
    Nếu không, bot bật lại giữa lúc đang lỗ sẽ coi mức lỗ hiện tại là đỉnh mới
    và ngắt mạch drawdown mất hiệu lực đúng lúc cần nhất.
    """
    store.save(LiveState(peak_equity=1000.0, last_equity=1000.0))
    restarted = store.load()          # mô phỏng tiến trình mới
    assert restarted.peak_equity == 1000.0
    assert restarted.drawdown(850.0) == pytest.approx(0.15)


def test_kill_switch_is_sticky(store):
    """Kill switch phải dai dẳng — restart không được xoá nó."""
    store.save(LiveState(is_dead=True))
    assert store.load().is_dead is True


def test_corrupt_file_is_quarantined_not_guessed(store):
    """File hỏng phải được cách ly để người xem, không được đoán nội dung."""
    pathlib.Path(store.path).write_text("{ đây không phải json")
    s = store.load()
    assert s.peak_equity == 0.0
    assert list(store.path.parent.glob("*.broken.*")), "file hỏng phải được giữ lại"


def test_save_is_atomic_no_temp_left_behind(store):
    store.save(LiveState(peak_equity=1.0))
    leftovers = list(store.path.parent.glob("*.tmp"))
    assert not leftovers, f"còn sót file tạm: {leftovers}"


def test_update_rejects_unknown_field(store):
    store.save(LiveState())
    with pytest.raises(AttributeError):
        store.update(khong_ton_tai=1)


def test_unknown_json_keys_ignored(store):
    """Field lạ từ phiên bản tương lai không được làm sập tiến trình."""
    store.path.write_text(json.dumps({"schema_version": 1, "peak_equity": 5.0, "tu_tuong_lai": "x"}))
    assert store.load().peak_equity == 5.0
