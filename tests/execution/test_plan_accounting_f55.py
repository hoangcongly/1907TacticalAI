"""
[FIX F55] Lệnh TĂNG bị bỏ vì min notional: vị thế cũ vẫn phải nằm trong gross/net kế hoạch.

Ở vốn nhỏ nhánh này chạy thường xuyên (mọi điều chỉnh dưới $5). Bản cũ để vị thế biến mất
khỏi sổ kế toán của kế hoạch, nên `_repair_neutrality` quyết định trên một sổ sai.
"""
from aegis.execution.portfolio_rebalancer import SymbolFilters, build_rebalance_plan


def _f(s):
    return SymbolFilters(s, 1e-6, 1e-9, 1e-9, 5.0)


def test_vi_the_giu_nguyen_vi_lenh_tang_duoi_min_van_nam_trong_so():
    # Vốn $100 x 1x: đích A long $52, B short $48; đang giữ A $49 (tăng $3 < $5), B −$48.
    plan = build_rebalance_plan({"AUSDT": 0.52, "BUSDT": -0.48},
                                {"AUSDT": 49.0, "BUSDT": -48.0},
                                {"AUSDT": 1.0, "BUSDT": 1.0},
                                {"AUSDT": _f("AUSDT"), "BUSDT": _f("BUSDT")},
                                equity=100.0, leverage=1.0, no_trade_band=0.0)
    assert "AUSDT" in plan.skipped and "min_notional" in plan.skipped["AUSDT"]
    assert plan.gross_notional == 49.0 + 48.0
    assert plan.net_notional == 49.0 - 48.0


def test_mo_moi_duoi_min_khong_tinh_vao_so():
    plan = build_rebalance_plan({"AUSDT": 0.03, "BUSDT": -0.97},
                                {"BUSDT": -97.0},
                                {"AUSDT": 1.0, "BUSDT": 1.0},
                                {"AUSDT": _f("AUSDT"), "BUSDT": _f("BUSDT")},
                                equity=100.0, leverage=1.0, no_trade_band=0.0)
    assert "AUSDT" in plan.skipped
    assert plan.gross_notional == 97.0
