"""
`scripts/shortfall_report.py` — khoá DẤU và TRỌNG SỐ của phép đo chi phí khớp lệnh.

Một phép đo chi phí sai dấu còn tệ hơn không đo: nó biến khoản lỗ thành "lời nhờ khớp
tốt" và dạy người vận hành tin rằng thực thi đang ổn. Các test dưới đây dùng số tay
tính được để mọi con số có đáp án đúng tuyệt đối.
"""
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "scripts")
from shortfall_report import last_closed_bar_open, shortfall_table, summarize  # noqa: E402

H4 = 4 * 3_600_000


def _trades(rows):
    return pd.DataFrame(rows, columns=["symbol", "side", "price", "qty", "commission",
                                       "commissionAsset", "maker"])


def test_dau_chi_phi_mua_dat_ban_re_deu_duong():
    t = shortfall_table(_trades([
        ("A", "BUY", 101.0, 1.0, 0.0, "USDT", True),     # mua đắt hơn tham chiếu 1%
        ("B", "SELL", 99.0, 1.0, 0.0, "USDT", False),    # bán rẻ hơn tham chiếu 1%
    ]), ref_decision={"A": 100.0, "B": 100.0}, ref_bar={"A": 100.0, "B": 100.0})
    assert t["exec_bps"].tolist() == pytest.approx([100.0, 100.0])
    assert t["exec_usd"].tolist() == pytest.approx([1.0, 1.0])


def test_khop_tot_hon_tham_chieu_la_am():
    t = shortfall_table(_trades([("A", "BUY", 99.0, 1.0, 0.0, "USDT", True)]),
                        {"A": 100.0}, {"A": 100.0})
    assert t["exec_bps"].iloc[0] == pytest.approx(-100.0)


def test_tach_tre_va_khop():
    """Nến 4h đóng 100, lúc quyết định 102, khớp mua 103: trễ 200bp, khớp ~98bp."""
    t = shortfall_table(_trades([("A", "BUY", 103.0, 2.0, 0.0, "USDT", False)]),
                        {"A": 102.0}, {"A": 100.0})
    assert t["delay_bps"].iloc[0] == pytest.approx(200.0)
    assert t["exec_bps"].iloc[0] == pytest.approx(1 / 102 * 1e4)
    # Cộng USD của hai khoản = tổng chênh so với giá backtest giả định.
    assert t["delay_usd"].iloc[0] + t["exec_usd"].iloc[0] == pytest.approx((103 - 100) * 2)


def test_gop_theo_notional_khong_theo_so_lenh():
    """Lệnh $10.000 tốn 10bp và lệnh $100 tốn 100bp: gộp đúng ~10,9bp, không phải 55bp."""
    t = shortfall_table(_trades([
        ("A", "BUY", 100.1, 100.0, 0.0, "USDT", True),   # $10.010, 10bp
        ("B", "BUY", 101.0, 1.0, 0.0, "USDT", False),    # $101, 100bp
    ]), {"A": 100.0, "B": 100.0}, {"A": 100.0, "B": 100.0})
    s = summarize(t)
    assert s["exec_bps"] == pytest.approx((10.0 + 1.0) / (10010.0 + 101.0) * 1e4)
    assert s["exec_bps"] < 12.0
    assert s["exec_maker_bps"] == pytest.approx(10.0 / 10010.0 * 1e4)
    assert s["exec_taker_bps"] == pytest.approx(1.0 / 101.0 * 1e4)


def test_phi_bps_va_bo_qua_phi_khong_phai_usdt():
    t = shortfall_table(_trades([
        ("A", "BUY", 100.0, 10.0, 0.4, "USDT", True),    # 4bp
        ("B", "BUY", 100.0, 10.0, 0.01, "BNB", True),    # không quy đổi được -> bỏ khỏi phí
    ]), {"A": 100.0, "B": 100.0}, {"A": 100.0, "B": 100.0})
    s = summarize(t)
    assert s["fee_bps"] == pytest.approx(4.0)
    assert np.isnan(t["fee_usd"].iloc[1])


def test_thieu_tham_chieu_thi_bo_chu_khong_tinh_bang_0():
    """Điền giá khớp làm tham chiếu sẽ cho chi phí 0 và kéo trung bình về 0 giả tạo."""
    t = shortfall_table(_trades([
        ("A", "BUY", 101.0, 1.0, 0.0, "USDT", True),
        ("B", "BUY", 150.0, 1.0, 0.0, "USDT", True),     # B không có giá tham chiếu
    ]), {"A": 100.0}, {"A": 100.0})
    assert t["symbol"].tolist() == ["A"]


def test_nen_4h_cuoi_da_dong():
    t = 5 * H4 + 123_456                         # giữa nến thứ 5 -> nến đã đóng là nến 4
    assert last_closed_bar_open(t) == 4 * H4
    assert last_closed_bar_open(5 * H4) == 4 * H4   # đúng mốc mở nến 5: nến 4 vừa đóng
