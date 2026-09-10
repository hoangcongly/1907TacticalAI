"""Test máy trạng thái OMS — chốt cứng vòng đời lệnh."""
import pytest

from aegis.oms.state_machine import (
    InvalidTransitionError, ManagedOrder, OrderBook, OrderState, TERMINAL_STATES,
)


def _order(**kw):
    base = dict(client_order_id="c1", symbol="XRPUSDT", side="BUY", qty=10.0, price=2.0)
    base.update(kw)
    return ManagedOrder(**base)


def test_happy_path_lifecycle():
    o = _order()
    o.transition(OrderState.SUBMITTED)
    o.apply_exchange_status("NEW")
    o.apply_exchange_status("PARTIALLY_FILLED", filled_qty=4.0, avg_price=2.0)
    o.apply_exchange_status("FILLED", filled_qty=10.0, avg_price=2.0)
    assert o.state is OrderState.FILLED
    assert o.is_terminal and o.remaining_qty == 0.0


def test_terminal_state_is_absorbing():
    """Trạng thái CUỐI không bao giờ rời đi — chống 'hồi sinh' lệnh đã xong."""
    o = _order()
    o.transition(OrderState.SUBMITTED)
    o.apply_exchange_status("CANCELED")
    with pytest.raises(InvalidTransitionError, match="CUỐI"):
        o.apply_exchange_status("FILLED", filled_qty=10.0)


def test_duplicate_terminal_event_is_tolerated():
    """
    Sự kiện lặp trên trạng thái CUỐI phải được bỏ qua êm.
    Kết nối lại WebSocket thường phát lại sự kiện — không được coi là lỗi.
    """
    o = _order()
    o.transition(OrderState.SUBMITTED)
    o.apply_exchange_status("FILLED", filled_qty=10.0)
    assert o.apply_exchange_status("FILLED").state is OrderState.FILLED


def test_illegal_transition_rejected():
    o = _order()
    with pytest.raises(InvalidTransitionError):
        o.transition(OrderState.FILLED)   # CREATED không thể nhảy thẳng sang FILLED


def test_filled_qty_can_never_decrease():
    """Khối lượng khớp giảm = sự kiện tới lệch thứ tự. Phải chặn."""
    o = _order()
    o.transition(OrderState.SUBMITTED)
    o.apply_exchange_status("PARTIALLY_FILLED", filled_qty=6.0)
    with pytest.raises(InvalidTransitionError, match="GIẢM"):
        o.apply_exchange_status("PARTIALLY_FILLED", filled_qty=3.0)


def test_overfill_rejected():
    o = _order()
    o.transition(OrderState.SUBMITTED)
    with pytest.raises(InvalidTransitionError, match="vượt quá"):
        o.apply_exchange_status("FILLED", filled_qty=11.0)


def test_unknown_exchange_status_rejected():
    o = _order()
    o.transition(OrderState.SUBMITTED)
    with pytest.raises(InvalidTransitionError, match="không nhận diện"):
        o.apply_exchange_status("TRẠNG_THÁI_LẠ")


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        _order(side="LONG")
    with pytest.raises(ValueError):
        _order(qty=0.0)


def test_order_book_rejects_duplicate_ids():
    book = OrderBook()
    book.add(_order())
    with pytest.raises(ValueError, match="trùng"):
        book.add(_order())


def test_net_position_counts_only_filled():
    """Vị thế ròng chỉ tính phần ĐÃ KHỚP — lệnh treo chưa phải vị thế."""
    book = OrderBook()
    a = book.add(_order(client_order_id="a", side="BUY", qty=10.0))
    b = book.add(_order(client_order_id="b", side="SELL", qty=4.0))
    pending = book.add(_order(client_order_id="c", side="BUY", qty=100.0))

    a.transition(OrderState.SUBMITTED); a.apply_exchange_status("FILLED", filled_qty=10.0)
    b.transition(OrderState.SUBMITTED); b.apply_exchange_status("FILLED", filled_qty=4.0)
    pending.transition(OrderState.SUBMITTED); pending.apply_exchange_status("NEW")

    assert book.net_position("XRPUSDT") == pytest.approx(6.0)
    assert len(book.open_orders()) == 1


def test_every_terminal_state_has_no_exits():
    from aegis.oms.state_machine import VALID_TRANSITIONS
    for st in TERMINAL_STATES:
        assert VALID_TRANSITIONS[st] == set(), f"{st} phải là trạng thái cuối"
