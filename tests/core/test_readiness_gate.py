"""
Test CỔNG CHẤT LƯỢNG — điều kiện được phép bơm tiền thật.

Cổng này tồn tại vì một lý do cụ thể: tính tới 11/09/2026, số lượt tái cân bằng
chạy đúng mà không cần can thiệp tay là 0 trên 4, trong khi tốc độ phát hiện lỗi T0
vẫn là 10 lỗi trong 2 ngày. Mắt người nhìn nhật ký rồi kết luận "trông ổn" sẽ luôn
kết luận là ổn — nhất là khi đang sốt ruột muốn vào tiền. Nên tiêu chí phải nhị phân
và do máy chấm.
"""
import sys

import pytest

sys.path.insert(0, "scripts")

from aegis.core.execution_log import MAKER_FEE, TAKER_FEE, ExecutionRecord


def _rec(**kw):
    base = dict(timestamp_ms=1, equity=5000.0, n_targets=12, n_orders=12,
                gross_notional=10000.0, planned_turnover=10000.0,
                maker_notional=9000.0, taker_notional=1000.0, unfilled_count=0)
    base.update(kw)
    return ExecutionRecord(**base)


# --------------------------------------------------------------- định nghĩa "sạch"
def test_luot_hoan_hao_la_sach():
    assert _rec(net_ok=True, gross_ok=True).clean is True


def test_ban_ghi_cu_thieu_truong_van_duoc_coi_la_sach():
    """Bản ghi trước khi có cổng không có net_ok/gross_ok — không được coi là hỏng."""
    assert _rec().clean is True


@pytest.mark.parametrize("field,value", [
    ("plan_failures", 1),        # F22 — lệnh không đặt được
    ("uncancelled", 1),          # F21 — lệnh sống ngoài tầm kiểm soát
    ("exec_error", "RuntimeError: x"),   # F24
    ("manual_intervention", True),
    ("net_ok", False),           # F25
    ("gross_ok", False),         # F29
])
def test_moi_dieu_kien_deu_du_de_danh_truot(field, value):
    assert _rec(**{field: value}).clean is False


def test_su_co_thuc_te_1009_bi_danh_truot():
    """Lượt 10/09: lệnh mồ côi COTIUSDT + phải kill switch."""
    assert _rec(uncancelled=1, manual_intervention=True).clean is False


def test_su_co_thuc_te_1109_bi_danh_truot():
    """Lượt 11/09: nhân đôi vị thế -> 3.69x, phải reduceOnly sửa tay."""
    assert _rec(leverage=3.69, target_leverage=2.0, net_ok=False,
                gross_ok=False, manual_intervention=True).clean is False


# --------------------------------------------------------------- chuỗi liên tiếp
def _streak(records):
    n = 0
    for r in reversed(records):
        if r.clean:
            n += 1
        else:
            break
    return n


def test_chuoi_tinh_tu_cuoi_va_dut_khi_gap_luot_hong():
    recs = [_rec(), _rec(), _rec(plan_failures=1), _rec(), _rec()]
    assert _streak(recs) == 2, "một lượt hỏng phải cắt đứt chuỗi"


def test_luot_hong_gan_nhat_dua_chuoi_ve_khong():
    assert _streak([_rec(), _rec(), _rec(), _rec(uncancelled=1)]) == 0


def test_du_ba_luot_sach_thi_dat():
    assert _streak([_rec(plan_failures=1)] + [_rec()] * 3) == 3


# --------------------------------------------------------------- phí thật [F31]
def test_phi_dung_binance_vip0():
    """
    Hằng số cũ (1bp/4bp) khiến chính module 'đo chi phí thật' báo thấp hơn thực tế.
    Một thước đo bị lệch nguy hiểm hơn không đo, vì nó tạo cảm giác đã kiểm soát.
    """
    assert MAKER_FEE == pytest.approx(0.0002)
    assert TAKER_FEE == pytest.approx(0.0005)


def test_chi_phi_thuc_te_tinh_dung():
    r = _rec(maker_notional=8000.0, taker_notional=2000.0)
    expected_usd = 8000 * 0.0002 + 2000 * 0.0005
    assert r.realized_cost_usd == pytest.approx(expected_usd)
    assert r.realized_cost_bps == pytest.approx(expected_usd / 10000 * 1e4)
    assert r.maker_ratio == pytest.approx(0.8)


def test_khop_maker_cao_thi_chi_phi_thap_hon():
    hi = _rec(maker_notional=9500.0, taker_notional=500.0)
    lo = _rec(maker_notional=3800.0, taker_notional=6200.0)
    assert hi.realized_cost_bps < lo.realized_cost_bps


# --------------------------------------------------------------- đọc nhật ký
def test_load_records_bo_qua_dong_hong(tmp_path):
    from readiness_gate import load_records
    f = tmp_path / "log.jsonl"
    f.write_text('{"timestamp_ms":1,"equity":1,"n_targets":1,"n_orders":1,'
                 '"gross_notional":1,"planned_turnover":1,"maker_notional":1,'
                 '"taker_notional":0,"unfilled_count":0}\n'
                 'KHÔNG PHẢI JSON\n'
                 '\n')
    assert len(load_records(str(f))) == 1


def test_load_records_bo_qua_truong_la(tmp_path):
    """Nhật ký cũ/mới lệch schema không được làm sập cổng."""
    from readiness_gate import load_records
    f = tmp_path / "log.jsonl"
    f.write_text('{"timestamp_ms":1,"equity":1,"n_targets":1,"n_orders":1,'
                 '"gross_notional":1,"planned_turnover":1,"maker_notional":1,'
                 '"taker_notional":0,"unfilled_count":0,"truong_khong_ton_tai":9}\n')
    assert len(load_records(str(f))) == 1
