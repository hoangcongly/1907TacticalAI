"""
[FIX F45] Van trung lập bắn quá sớm vì SAI MẪU SỐ, không phải vì thiếu cổng tiến độ.

SỰ CỐ ĐÃ ĐO (lượt 19/09/2026, 50 lệnh): `requotes=0`,
`requote_blocked_by_chase_cap=0`, `requote_skipped_no_move=0`,
`max_drift_bps_seen=0.0`. Cả BỐN bộ đếm bằng 0 — nếu `_requote_one` từng được gọi
thì ít nhất một trong ba bộ đếm đầu phải khác 0. Bằng 0 hết nghĩa là đường đuổi giá
chưa từng chạy. Kết quả: 31/50 lệnh hết giờ, maker 39%, và trần đuổi co giãn của
F44 thành code chết.

NGUYÊN NHÂN GỐC: van so `drift` — mẫu số là gross ĐÃ KHỚP — với ngưỡng 10%. Khi mới
khớp vài lệnh, mẫu số cực nhỏ nên tỷ lệ này nhiễu hơn chính ngưỡng. Mô phỏng 20.000
lượt cho độ lệch THUẦN NGẪU NHIÊN (không có rủi ro thật) là 42,9% ở tiến độ 10% và
26,1% ở tiến độ 25%. Van vì thế bắn 100% số lượt, ở tiến độ trung vị 2%.

CHẨN ĐOÁN ĐẦU TIÊN ĐÃ SAI và test này khoá luôn điều đó: bản vá đầu thêm
`min_gross_for_neutrality_check = 0,25 x gross kế hoạch`. Đo lại: ở ngưỡng tiến độ
25% van VẪN bắn 99,4% số lượt. Phải tới 95% mới im — mà ở đó van không còn tác dụng.
Cổng tiến độ không sửa được một phép đo sai đơn vị.

BẢN VÁ ĐÚNG: đo `drift_plan` — mẫu số là gross KẾ HOẠCH, cố định — nên nhiễu bị
chặn trên ở ~7,6% bất kể tiến độ. Ngưỡng 0,25 cho báo nhầm 0,3% và bắt đúng 100%.

Vì sao 558 test cũ không bắt được: `early_exit` không xuất hiện trong bất kỳ test
nào. Đường thoát sớm hoàn toàn không được phủ.
"""
import time
from dataclasses import dataclass, fields

import pytest

from aegis.oms.order_router import BinanceOrderRouter
from aegis.oms.state_machine import ManagedOrder


class _StubClient:
    credentials = None


@dataclass
class _Intent:
    """Ý định đặt lệnh — chỉ cần đủ cho `plan_net` / `plan_gross`."""
    symbol: str
    side: str
    notional: float


def _mo(symbol, side, qty, price, filled):
    m = ManagedOrder(client_order_id=f"c_{symbol}", symbol=symbol, side=side,
                     qty=qty, price=price)
    m.filled_qty = filled
    m.avg_fill_price = price
    return m


def _run_loop(max_fill_imbalance: float, filled_legs: int = 1):
    """
    Chạy vài nhịp poll của vòng chờ, trả về (report, số lần đuổi giá).

    Sổ: 10 lệnh cỡ bằng nhau ($1.000/lệnh, kế hoạch $10.000, plan_net = 0).
    `filled_legs` chân BUY đã khớp, phần còn lại treo -> lệch một chiều.
    """
    r = BinanceOrderRouter(client=_StubClient(), dry_run=True)
    r.poll_status = lambda m, retries=1: None          # sàn giả: không đổi trạng thái

    calls = []
    r._requote_one = lambda *a, **k: calls.append(a[1].symbol)

    passive, cycle = [], {}
    for i in range(5):                                  # 5 BUY
        m = _mo(f"B{i}", "BUY", 10, 100.0, 10 if i < filled_legs else 0)
        cycle[f"B{i}"] = [m]; passive.append(m)
    for i in range(5):                                  # 5 SELL, không khớp gì
        m = _mo(f"S{i}", "SELL", 10, 100.0, 0)
        cycle[f"S{i}"] = [m]; passive.append(m)
    r._cycle_orders = cycle

    intents = {m.symbol: _Intent(m.symbol, m.side, 1000.0) for m in passive}
    report: dict = {}
    r._passive_wait_loop(
        passive, intents, filters={}, deadline=time.time() + 0.03,
        poll_interval_s=0.01, requote=True, max_chase_bps=15.0,
        anchor={m.symbol: 100.0 for m in passive}, report=report,
        max_fill_imbalance=max_fill_imbalance, chase_caps={},
    )
    return report, calls


def test_mot_lenh_khop_khong_duoc_lam_van_ban():
    """
    LỖI GỐC F45: 1/10 lệnh khớp -> `drift` = 100% (nhiễu thuần), `drift_plan` = 10%.

    Van đo `drift_plan` nên im, và vòng báo giá lại được chạy — đó là toàn bộ điểm
    của bản vá. Đo theo `drift` thì mọi ngưỡng <= 100% đều bắn ngay tại đây.
    """
    report, calls = _run_loop(max_fill_imbalance=0.25)
    imb = report["fill_imbalance"]
    assert imb["drift"] == pytest.approx(1.0)         # nhiễu: 100% gross đã khớp
    assert imb["drift_plan"] == pytest.approx(0.10)   # rủi ro thật: 10% sổ định cầm
    assert "early_exit" not in report
    assert calls, "vòng báo giá lại phải được chạy khi độ lệch thật còn nhỏ"


def test_lech_mot_chieu_that_thi_van_van_ban():
    """Van phải GIỮ được tác dụng: 3/10 lệnh cùng chiều = 30% sổ, vượt trần 25%."""
    report, calls = _run_loop(max_fill_imbalance=0.25, filled_legs=3)
    assert report["fill_imbalance"]["drift_plan"] == pytest.approx(0.30)
    assert report.get("early_exit") == "neutrality_drift"
    assert calls == [], "bắn van thì phải thoát ngay, không báo giá lại thêm"


def test_cong_tien_do_khong_cuu_duoc_thuoc_do_sai():
    """
    Khoá lại chẩn đoán SAI đầu tiên, để không ai thử lại hướng đó.

    Sổ 50 lệnh (25 BUY / 25 SELL), đã khớp 13 -> tiến độ 26%, tức ĐÃ QUA cổng 25%
    mà bản vá đầu đề xuất. Trong 13 lệnh đó có 9 BUY và 4 SELL.

    9/4 không phải kịch bản dựng cho dễ thắng: ở k=13 rút từ 25/25, độ lệch chuẩn của
    (số BUY − số SELL) là sqrt(13x37/49) = 3,13 đơn vị, nên chênh 5 đơn vị chỉ là
    1,6 sigma — một lượt khớp hoàn toàn bình thường, KHÔNG có rủi ro thật.
    """
    r = BinanceOrderRouter(client=_StubClient(), dry_run=True)
    cycle = {}
    for i in range(50):
        side = "BUY" if i < 25 else "SELL"
        filled = 10 if (i < 9 or 25 <= i < 29) else 0    # 9 BUY + 4 SELL đã khớp
        cycle[f"X{i}"] = [_mo(f"X{i}", side, 10, 100.0, filled)]
    r._cycle_orders = cycle
    imb = r.fill_imbalance(plan_net=0.0, plan_gross=50 * 1000.0)

    assert imb["progress"] == pytest.approx(0.26)
    assert abs(imb["drift"]) == pytest.approx(5 / 13, abs=1e-6)
    assert abs(imb["drift"]) > 0.10, "thước đo cũ vượt ngưỡng dù đã qua cổng 25%"
    assert abs(imb["drift_plan"]) == pytest.approx(0.10)
    assert abs(imb["drift_plan"]) < 0.25, "thước đo mới nằm trong trần — không báo nhầm"


def test_pipeline_thuc_su_truyen_tran_va_dung_don_vi():
    """
    Chốt phần NỐI DÂY — đây mới là nơi lỗi F45 thật sự nằm.

    Router vốn đã có van; lỗi là pipeline truyền sai thước đo. Test này bắt đúng
    khoảng trống đó, và bắt luôn việc tên cũ quay lại (tên cũ = đơn vị cũ).
    """
    import inspect

    from aegis.pipelines.xs_live_pipeline import LiveConfig, CrossSectionalLivePipeline

    assert hasattr(LiveConfig, "max_fill_imbalance")
    assert not hasattr(LiveConfig, "neutrality_min_progress"), \
        "cổng tiến độ đã bị bác bỏ bằng đo đạc — đừng dựng lại"
    cfg = LiveConfig(universe=[])
    assert 0.0 < cfg.max_fill_imbalance < 1.0

    src = inspect.getsource(CrossSectionalLivePipeline)
    assert "max_fill_imbalance=self.config.max_fill_imbalance" in src, \
        ("pipeline phải TRUYỀN max_fill_imbalance cho execute_with_fallback; "
         "bỏ trống là để router dùng mặc định thay vì cấu hình đã kiểm định")


def test_ban_ghi_luu_du_de_chan_doan_lan_sau():
    """
    `fill_drift` một mình KHÔNG đủ để chẩn đoán — đó là lý do lượt 19/09 mất 3 ngày.

    Thiếu tiến độ thì không phân biệt được "lệch thật" với "mới khớp 2% nên nhiễu".
    """
    from aegis.core.execution_log import ExecutionRecord

    have = {f.name for f in fields(ExecutionRecord)}
    for f in ("fill_drift", "fill_drift_plan", "fill_progress"):
        assert f in have, f"ExecutionRecord thiếu {f}"
