"""
`CostModel.flat_bps` — chi phí một chiều ĐO THẬT thay cho mô hình phí + spread.

Mặc định `None` phải giữ NGUYÊN mọi con số cũ (mọi Sharpe đã kiểm định dựa trên nó);
đặt giá trị thì mọi cặp chịu đúng con số đó, không cộng thêm spread từng cặp — nếu
cộng thêm thì khoản spread bị tính HAI LẦN, vì con số đo thật đã chứa nó.
"""
import pandas as pd
import pytest

from aegis.research.backtest_v2 import CostModel


def test_mac_dinh_giu_nguyen_mo_hinh_cu():
    per = pd.Series({"A": 1.0, "B": 3.0})
    old = CostModel(maker_ratio=0.39, per_symbol_bps=per).bps_for(["A", "B"])
    new = CostModel(maker_ratio=0.39, per_symbol_bps=per, flat_bps=None).bps_for(["A", "B"])
    pd.testing.assert_series_equal(old, new)


def test_chi_phi_do_that_thay_ca_mo_hinh_khong_cong_don():
    per = pd.Series({"A": 1.0, "B": 3.0})
    out = CostModel(maker_ratio=0.39, per_symbol_bps=per, flat_bps=15.7).bps_for(["A", "B"])
    assert out.tolist() == pytest.approx([15.7, 15.7])
