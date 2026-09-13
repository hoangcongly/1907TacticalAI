"""
Test hồi quy cho SỰ CỐ THỰC THI 10/09/2026 — lỗi F21 đến F25.

DIỄN BIẾN THẬT (dựng lại từ nhật ký sàn, không phải suy đoán):

  10:30:47  đặt loạt lệnh post-only cho 12 cặp
  10:31     một ngoại lệ KHÔNG ĐƯỢC BẮT thoát ra khỏi `submit_plan` giữa chừng
            -> SANDUSDT, XMRUSDT (cuối danh sách MỞ) chưa từng được đặt
            -> bước dọn dẹp không bao giờ chạy: lệnh COTIUSDT vẫn sống trên sàn
            -> bước lưu trạng thái không bao giờ tới: `rebalance_count` đứng ở 1
  14:10:07  lệnh COTIUSDT nằm chờ 3 giờ 39 phút rồi TỰ KHỚP, không ai theo dõi
  11/09     hệ thống tự khoá vì lệch sổ sách (chốt chặn này hoạt động ĐÚNG)
            danh mục lúc đó: 10/12 vị thế, lệch +34.95% khỏi trung lập

Bài học và cũng là điều mỗi test dưới đây khoá lại:
  F21 huỷ lệnh phải được XÁC NHẬN; huỷ không xác nhận = lệnh vẫn sống
  F22 một cặp lỗi không được giết cả lượt
  F23 dọn dẹp phải chạy kể cả khi vòng chờ hỏng
  F24 sổ nội bộ phải đồng bộ từ sàn kể cả khi thực thi ném lỗi
  F25 lệch hướng sau thực thi phải bị phát hiện, không được im lặng
"""
import pytest

from aegis.execution.portfolio_rebalancer import RebalanceOrder, SymbolFilters
from aegis.oms.order_router import BinanceOrderRouter, OrderRejected
from aegis.oms.state_machine import OrderBook


def _filt(sym="XRPUSDT"):
    return SymbolFilters(sym, 0.0001, 0.1, 0.1, 5.0)


def _order(sym="XRPUSDT", qty=100.0, price=2.0, side="BUY", reason="mở"):
    return RebalanceOrder(sym, side, qty, price, reason, qty * price)


class _Client:
    """
    Client giả có thể được lập trình để hỏng ở đúng chỗ ta muốn.

    `fail_on_order`: các cặp sẽ ném lỗi khi đặt lệnh.
    `open_orders_left`: số lệnh treo `open_orders()` báo còn lại (mô phỏng huỷ hụt).
    """

    credentials = None

    def __init__(self, fail_on_order=(), open_orders_left=0, cancel_raises=False):
        self.calls = []
        self.fail_on_order = set(fail_on_order)
        self.open_orders_left = open_orders_left
        self.cancel_raises = cancel_raises
        self.cancelled = []

    def _request(self, method, path, params=None, signed=False):
        self.calls.append((method, path, dict(params or {})))
        sym = (params or {}).get("symbol")

        if path == "/fapi/v1/allOpenOrders" and method == "DELETE":
            self.cancelled.append(sym)
            if self.cancel_raises:
                raise RuntimeError("mạng lỗi khi huỷ")
            return {}
        if path == "/fapi/v1/openOrders":
            return [{"symbol": sym}] * self.open_orders_left
        if path == "/fapi/v1/order" and method == "GET":
            return {"status": "NEW", "executedQty": "0", "avgPrice": "0"}
        if path == "/fapi/v1/order":
            if sym in self.fail_on_order:
                raise RuntimeError(f"sàn trả 500 cho {sym}")
            return {"orderId": 1, "status": "NEW", "executedQty": "0", "avgPrice": "0"}
        return {}

    def open_orders(self, symbol=None):
        return [{"symbol": symbol}] * self.open_orders_left

    def best_bid_ask(self, symbol):
        return (1.999, 2.001)


def _router(client, **kw):
    return BinanceOrderRouter(client=client, order_book=OrderBook(), **kw)


# =====================================================================
# F21 — huỷ lệnh phải được XÁC NHẬN
# =====================================================================
def test_f21_cancel_xac_nhan_thanh_cong():
    c = _Client(open_orders_left=0)
    assert _router(c).cancel_all("XRPUSDT") is True
    assert "XRPUSDT" in c.cancelled


def test_f21_cancel_that_bai_khi_lenh_van_con_tren_san():
    """
    Đây chính là sự cố COTIUSDT: lệnh huỷ được gửi nhưng lệnh VẪN SỐNG.
    Bản cũ trả về None và chương trình chạy tiếp trong ảo tưởng sổ đã sạch.
    """
    c = _Client(open_orders_left=1)          # sàn vẫn báo còn lệnh treo
    assert _router(c).cancel_all("COTIUSDT", retries=2, delay=0.0) is False


def test_f21_cancel_that_bai_khi_api_nem_loi():
    c = _Client(cancel_raises=True, open_orders_left=1)
    assert _router(c).cancel_all("COTIUSDT", retries=2, delay=0.0) is False


def test_f21_khong_dat_lenh_moi_khi_chua_xac_nhan_huy():
    """Huỷ chưa xác nhận mà vẫn báo giá lại sẽ tạo HAI lệnh sống cùng lúc."""
    c = _Client(open_orders_left=1)
    r = _router(c)
    passive = r.submit_plan([_order()], {"XRPUSDT": _filt()}, post_only=True)
    before = len([x for x in c.calls if x[1] == "/fapi/v1/order" and x[0] != "GET"])

    r._requote_one(0, passive[0], passive, {"XRPUSDT": _order()},
                   {"XRPUSDT": _filt()}, {"XRPUSDT": 2.0}, 50.0, {"requotes": 0})

    after = len([x for x in c.calls if x[1] == "/fapi/v1/order" and x[0] != "GET"])
    assert after == before, "đã đặt lệnh mới dù chưa xác nhận huỷ lệnh cũ"
    assert "XRPUSDT" in r.uncancelled


# =====================================================================
# F22 — một cặp lỗi không được giết cả lượt
# =====================================================================
def test_f22_mot_lenh_loi_khong_giet_ca_lo():
    """
    Tái hiện chính xác nguyên nhân gốc: cặp thứ 2 trong danh sách ném lỗi.
    Bản cũ -> 1 lệnh được đặt, 3 cặp còn lại biến mất trong im lặng.
    Bản mới -> 3 lệnh được đặt, cặp hỏng được ghi nhận tường minh.
    """
    syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    c = _Client(fail_on_order={"BBBUSDT"})
    r = _router(c)
    orders = [_order(sym=s) for s in syms]
    filters = {s: _filt(s) for s in syms}

    placed = r.submit_plan(orders, filters, post_only=True)

    assert len(placed) == 3, "các cặp sau cặp hỏng phải vẫn được đặt"
    assert {m.symbol for m in placed} == {"AAAUSDT", "CCCUSDT", "DDDUSDT"}
    assert len(r.last_plan_failures) == 1
    assert r.last_plan_failures[0]["symbol"] == "BBBUSDT"
    assert "RuntimeError" in r.last_plan_failures[0]["reason"]


def test_f22_thieu_bo_loc_duoc_ghi_nhan_chu_khong_im_lang():
    c = _Client()
    r = _router(c)
    placed = r.submit_plan([_order(sym="AAAUSDT"), _order(sym="BBBUSDT")],
                           {"AAAUSDT": _filt("AAAUSDT")}, post_only=True)
    assert len(placed) == 1
    assert [f["symbol"] for f in r.last_plan_failures] == ["BBBUSDT"]


def test_f22_bao_cao_chua_danh_sach_lenh_that_bai():
    c = _Client(fail_on_order={"BBBUSDT"})
    r = _router(c)
    syms = ["AAAUSDT", "BBBUSDT"]
    rep = r.execute_with_fallback([_order(sym=s) for s in syms],
                                  {s: _filt(s) for s in syms},
                                  passive_wait_s=0.0, allow_taker_fallback=False)
    assert [f["symbol"] for f in rep["plan_failures"]] == ["BBBUSDT"]


# =====================================================================
# F23 — dọn dẹp phải chạy kể cả khi vòng chờ hỏng
# =====================================================================
def test_f23_von_cho_hong_van_don_dep():
    """
    Nếu vòng chờ ném lỗi, lệnh post-only PHẢI vẫn được huỷ. Bỏ qua bước này chính
    là cách COTIUSDT sống sót 3 giờ 39 phút trên sàn.
    """
    c = _Client()
    r = _router(c)

    def _no(*a, **kw):
        raise RuntimeError("vòng chờ hỏng")
    r._passive_wait_loop = _no

    rep = r.execute_with_fallback([_order()], {"XRPUSDT": _filt()},
                                  passive_wait_s=0.0, allow_taker_fallback=False)
    assert rep["passive_loop_error"] is True
    assert "XRPUSDT" in c.cancelled, "không dọn dẹp sau khi vòng chờ hỏng"


def test_f23_bao_cao_liet_ke_lenh_khong_huy_duoc():
    c = _Client(open_orders_left=1)
    rep = _router(c).execute_with_fallback(
        [_order()], {"XRPUSDT": _filt()}, passive_wait_s=0.0, allow_taker_fallback=False)
    assert "XRPUSDT" in rep["uncancelled"]


# =====================================================================
# F27 — nhân đôi vị thế do đếm trùng phần đã khớp khi báo giá lại
# (Sự cố 11/09/2026)
# =====================================================================
class _PartialFillClient(_Client):
    """
    Tái hiện đúng chuỗi VETUSDT: lệnh đầu khớp MỘT PHẦN rồi bị huỷ, lệnh đặt lại
    khớp phần còn lại. Nếu router quên phần đã khớp của lệnh đầu, nó sẽ đặt lại
    TOÀN BỘ khối lượng gốc và vị thế nhân đôi.
    """

    def __init__(self, first_fill, mid=0.007):
        super().__init__()
        self.first_fill = first_fill
        self.mid = mid
        self.order_seq = 0
        self.fills = {}          # clientOrderId -> (đã khớp, tổng)

    def best_bid_ask(self, symbol):
        # Sổ lệnh phải nằm SÁT giá đặt, nếu không chốt chặn đuổi giá
        # (`max_chase_bps`) sẽ từ chối báo giá lại và test đo nhầm thứ khác.
        return (self.mid * 0.999, self.mid * 1.001)

    def _request(self, method, path, params=None, signed=False):
        self.calls.append((method, path, dict(params or {})))
        sym = (params or {}).get("symbol")

        if path == "/fapi/v1/allOpenOrders" and method == "DELETE":
            self.cancelled.append(sym)
            return {}
        if path == "/fapi/v1/openOrders":
            return []
        if path == "/fapi/v1/order" and method == "GET":
            coid = (params or {}).get("origClientOrderId")
            filled, total = self.fills.get(coid, (0.0, 0.0))
            status = ("PARTIALLY_FILLED" if 0 < filled < total
                      else "FILLED" if filled >= total > 0 else "NEW")
            return {"status": status, "executedQty": str(filled), "avgPrice": "2.0"}
        if path == "/fapi/v1/order":
            self.order_seq += 1
            coid = (params or {}).get("newClientOrderId")
            total = float((params or {}).get("quantity", 0.0))
            # Lệnh ĐẦU TIÊN khớp một phần; các lệnh sau chưa khớp gì.
            filled = min(self.first_fill, total) if self.order_seq == 1 else 0.0
            self.fills[coid] = (filled, total)
            status = ("PARTIALLY_FILLED" if 0 < filled < total
                      else "FILLED" if filled >= total > 0 else "NEW")
            return {"orderId": self.order_seq, "status": status,
                    "executedQty": str(filled), "avgPrice": "2.0"}
        return {}


def test_f27_bao_gia_lai_khong_dat_lai_phan_da_khop():
    """
    Ý định 210_356; lệnh đầu khớp 110_276 rồi bị huỷ.
    Lệnh đặt lại PHẢI là 100_080, KHÔNG phải 210_356.
    Bản cũ đặt lại toàn bộ -> tổng khớp 320_632 và danh mục chạy 3.69x đòn bẩy.
    """
    total, first = 210356.0, 110276.0
    c = _PartialFillClient(first_fill=first)
    r = _router(c, max_order_notional=10_000.0)   # lệnh thật $1.472, nới trần mặc định
    filt = SymbolFilters("VETUSDT", 0.000001, 1.0, 1.0, 5.0)
    intent = _order(sym="VETUSDT", qty=total, price=0.007)

    passive = r.submit_plan([intent], {"VETUSDT": filt}, post_only=True)
    r._cycle_orders = {"VETUSDT": list(passive)}

    assert r.filled_for("VETUSDT") == pytest.approx(first)
    assert r.remaining_for("VETUSDT", {"VETUSDT": intent}) == pytest.approx(total - first)

    r._requote_one(0, passive[0], passive, {"VETUSDT": intent},
                   {"VETUSDT": filt}, {"VETUSDT": 0.007}, 10_000.0, {"requotes": 0})

    news = [x for x in c.calls if x[1] == "/fapi/v1/order" and x[0] == "POST"]
    assert len(news) == 2, "phải có đúng lệnh gốc + một lệnh đặt lại"
    requote_qty = float(news[-1][2]["quantity"])
    assert requote_qty == pytest.approx(total - first, rel=1e-6), (
        f"đặt lại {requote_qty:,.0f} thay vì {total-first:,.0f} -> vị thế sẽ nhân đôi")


def test_f27_tong_khop_cong_qua_moi_lenh_cua_cap():
    """Phần đã khớp phải cộng qua MỌI lệnh của cặp, không chỉ lệnh mới nhất."""
    c = _PartialFillClient(first_fill=50.0)
    r = _router(c)
    filt = SymbolFilters("VETUSDT", 0.000001, 1.0, 1.0, 5.0)
    intent = _order(sym="VETUSDT", qty=100.0, price=2.0)

    first = r.submit_plan([intent], {"VETUSDT": filt}, post_only=True)   # khớp 50/100
    r._cycle_orders = {"VETUSDT": list(first)}
    assert r.filled_for("VETUSDT") == pytest.approx(50.0)

    # đặt lại phần còn thiếu 50, lần này chưa khớp gì
    second = r.submit(_order(sym="VETUSDT", qty=50.0, price=2.0), filt,
                      post_only=True, epoch_bucket=999)
    r._cycle_orders["VETUSDT"].append(second)

    assert r.filled_for("VETUSDT") == pytest.approx(50.0)
    assert r.remaining_for("VETUSDT", {"VETUSDT": intent}) == pytest.approx(50.0)


def test_f27_khong_bao_gio_tra_ve_so_am():
    """
    Nếu tổng đã khớp bằng hoặc vượt ý định, phần còn thiếu phải là 0 — không được
    ra số âm, vì số âm sẽ bị diễn giải thành một lệnh NGƯỢC CHIỀU.
    """
    c = _PartialFillClient(first_fill=100.0)
    r = _router(c)
    filt = SymbolFilters("VETUSDT", 0.000001, 1.0, 1.0, 5.0)
    intent = _order(sym="VETUSDT", qty=100.0, price=2.0)
    passive = r.submit_plan([intent], {"VETUSDT": filt}, post_only=True)  # khớp đủ 100
    r._cycle_orders = {"VETUSDT": list(passive)}

    assert r.filled_for("VETUSDT") == pytest.approx(100.0)
    assert r.remaining_for("VETUSDT", {"VETUSDT": intent}) == 0.0
