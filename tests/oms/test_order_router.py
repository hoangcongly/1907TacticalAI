"""Test router lệnh — trọng tâm là các chốt chặn an toàn TRƯỚC khi gửi."""
import pytest

from aegis.execution.portfolio_rebalancer import RebalanceOrder, SymbolFilters
from aegis.oms.order_router import (
    BinanceOrderRouter, OrderRejected, make_client_order_id,
)
from aegis.oms.state_machine import OrderBook, OrderState


class _FakeClient:
    """Client giả — không chạm mạng."""
    credentials = None
    def __init__(self): self.calls = []
    def _request(self, method, path, params=None, signed=False):
        self.calls.append((method, path, params))
        return {"orderId": 111, "status": "NEW", "executedQty": "0", "avgPrice": "0"}


def _filt(): return SymbolFilters("XRPUSDT", 0.0001, 0.1, 0.1, 5.0)
def _order(qty=10.0, price=2.0, side="BUY", reason="mở"):
    return RebalanceOrder("XRPUSDT", side, qty, price, reason, qty * price)


def _router(**kw):
    return BinanceOrderRouter(client=_FakeClient(), order_book=OrderBook(), **kw)


# ------------------------------------------------------------- idempotency
def test_client_order_id_is_deterministic():
    """Cùng ý định + cùng chu kỳ -> cùng id. Đây là lá chắn chống lệnh trùng."""
    a = make_client_order_id("XRPUSDT", "BUY", 10.0, 999)
    assert a == make_client_order_id("XRPUSDT", "BUY", 10.0, 999)
    assert a != make_client_order_id("XRPUSDT", "SELL", 10.0, 999)
    assert a != make_client_order_id("XRPUSDT", "BUY", 10.0, 1000)


def test_client_order_id_fits_binance_limit():
    assert len(make_client_order_id("XRPUSDT", "BUY", 12345.6789, 1700000000)) <= 36


def test_resubmitting_same_intent_does_not_duplicate():
    """Gửi lại cùng ý định trong cùng chu kỳ KHÔNG được tạo lệnh thứ hai."""
    r = _router()
    first = r.submit(_order(), _filt(), epoch_bucket=42)
    second = r.submit(_order(), _filt(), epoch_bucket=42)
    assert first is second
    assert len([c for c in r.client.calls if c[1] == "/fapi/v1/order"]) == 1


# --------------------------------------------------------------- an toàn
def test_notional_cap_blocks_oversized_order():
    """Trần notional là chốt chặn cuối chống lỗi logic sizing."""
    r = _router(max_order_notional=100.0)
    with pytest.raises(OrderRejected, match="vượt trần"):
        r.submit(_order(qty=1000.0, price=2.0), _filt())


def test_below_min_notional_blocked_locally():
    """Chặn tại chỗ, không tốn một vòng mạng để bị sàn từ chối."""
    r = _router()
    with pytest.raises(OrderRejected, match="min_notional"):
        r.submit(_order(qty=1.0, price=1.0), _filt())


def test_qty_not_on_step_size_blocked():
    r = _router()
    with pytest.raises(OrderRejected, match="step_size"):
        r.submit(_order(qty=10.05), _filt())


def test_invalid_side_blocked():
    r = _router()
    with pytest.raises(OrderRejected, match="side"):
        r.submit(_order(side="LONG"), _filt())


# --------------------------------------------------------------- hành vi
def test_post_only_uses_gtx_time_in_force():
    """
    Chiến lược đã kiểm định giả định phí MAKER. Khớp taker phá vỡ giả định đó,
    nên lệnh phải là post-only (GTX).
    """
    r = _router()
    r.submit(_order(), _filt(), post_only=True)
    params = r.client.calls[-1][2]
    assert params["type"] == "LIMIT" and params["timeInForce"] == "GTX"


def test_plan_closes_before_opening():
    """Đóng trước mở sau: giải phóng ký quỹ trước khi cần dùng."""
    r = _router()
    orders = [
        RebalanceOrder("XRPUSDT", "BUY", 10.0, 2.0, "mở", 20.0),
        RebalanceOrder("XRPUSDT", "SELL", 5.0, 2.0, "đóng", 10.0),
    ]
    r.submit_plan(orders, {"XRPUSDT": _filt()})
    sides = [c[2]["side"] for c in r.client.calls if c[1] == "/fapi/v1/order"]
    assert sides[0] == "SELL", "lệnh đóng phải được gửi trước"


def test_dry_run_sends_nothing():
    r = _router(dry_run=True)
    o = r.submit(_order(), _filt())
    assert o.state is OrderState.SUBMITTED
    assert not [c for c in r.client.calls if c[1] == "/fapi/v1/order"]
