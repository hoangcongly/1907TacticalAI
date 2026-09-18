"""
Van trung lập trong lúc khớp thụ động.

Lý do tồn tại nằm trong lịch sử sự cố của chính hệ thống này: lệnh maker khớp KHÔNG
đồng đều, chân long xong trước chân short, và một danh mục market-neutral khớp một nửa
không còn trung lập — nó là cược có hướng mà không ai chủ ý. Đã xảy ra hai lần (lệch
33% và 35% gross).

Bản trước chặn rủi ro đó GIÁN TIẾP bằng cách để thời gian chờ thật ngắn (300s). Cái
giá là tỷ lệ maker thấp (đo được 0,378-0,49) và vì thế chi phí cao. Van này chặn rủi
ro TRỰC TIẾP, nên thời gian chờ có thể dài ra mà vẫn an toàn hơn trước.
"""
import pytest

from aegis.oms.order_router import BinanceOrderRouter
from aegis.oms.state_machine import ManagedOrder, OrderState


class _StubClient:
    """Sàn giả tối thiểu — van trung lập chỉ đọc sổ nội bộ, không gọi sàn."""
    credentials = None


def _router():
    return BinanceOrderRouter(client=_StubClient(), dry_run=True)


def _mo(symbol, side, qty, price, filled):
    m = ManagedOrder(client_order_id=f"c_{symbol}", symbol=symbol, side=side,
                     qty=qty, price=price)
    m.filled_qty = filled
    m.avg_fill_price = price
    return m


def test_khop_dung_ty_le_thi_khong_lech():
    """Mọi cặp khớp cùng một tỷ lệ => trung lập được giữ nguyên ở mọi thời điểm."""
    r = _router()
    r._cycle_orders = {
        "A": [_mo("A", "BUY", 10, 100.0, 5)],      # khớp 50%
        "B": [_mo("B", "SELL", 10, 100.0, 5)],     # khớp 50%
    }
    out = r.fill_imbalance(plan_net=0.0, plan_gross=2000.0)
    assert out["drift"] == pytest.approx(0.0, abs=1e-12)
    assert out["progress"] == pytest.approx(0.5)


def test_mot_chan_khop_truoc_thi_bao_lech():
    """Chân BUY khớp hết còn chân SELL treo => sổ đang thành long, phải báo."""
    r = _router()
    r._cycle_orders = {
        "A": [_mo("A", "BUY", 10, 100.0, 10)],     # khớp 100%
        "B": [_mo("B", "SELL", 10, 100.0, 0)],     # chưa khớp
    }
    out = r.fill_imbalance(plan_net=0.0, plan_gross=2000.0)
    assert out["drift"] == pytest.approx(1.0), "khớp một chiều hoàn toàn => lệch 100%"


def test_ke_hoach_von_di_lech_thi_khong_bao_dong_gia():
    """
    Khi vốn tăng, MỌI vị thế nở ra cùng lúc nên kế hoạch vốn dĩ có net khác 0. So độ
    lệch với 0 sẽ báo động ngay từ lệnh đầu; phải so với đường thực thi THEO TỶ LỆ.
    """
    r = _router()
    # Kế hoạch: mua thêm cả hai chân (một chân long nở ra, một chân short thu lại)
    r._cycle_orders = {
        "A": [_mo("A", "BUY", 10, 100.0, 5)],
        "B": [_mo("B", "BUY", 10, 100.0, 5)],
    }
    plan_net = 2000.0        # cả hai lệnh đều là BUY
    out = r.fill_imbalance(plan_net=plan_net, plan_gross=2000.0)
    assert out["drift"] == pytest.approx(0.0, abs=1e-12), \
        "khớp đúng tỷ lệ trên một kế hoạch lệch sẵn vẫn phải cho drift = 0"


def test_chua_khop_gi_thi_khong_lech():
    r = _router()
    r._cycle_orders = {"A": [_mo("A", "BUY", 10, 100.0, 0)]}
    out = r.fill_imbalance(plan_net=0.0, plan_gross=1000.0)
    assert out["drift"] == 0.0 and out["filled_gross"] == 0.0


def test_lech_co_dau_dung_chieu():
    """Dấu phải nói được sổ đang nghiêng về phía nào — nếu không thì không sửa được."""
    r = _router()
    r._cycle_orders = {"A": [_mo("A", "SELL", 10, 100.0, 10)],
                       "B": [_mo("B", "BUY", 10, 100.0, 0)]}
    out = r.fill_imbalance(plan_net=0.0, plan_gross=2000.0)
    assert out["drift"] < 0, "chân SELL khớp trước => sổ nghiêng short => drift âm"
