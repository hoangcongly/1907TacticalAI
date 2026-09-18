"""
[FIX F42] Dải không giao dịch KHÔNG được phá trung lập của kế hoạch.

Bối cảnh — vì sao test này tồn tại:

`portfolio.py` dựng trọng số trung lập chính xác (đo trên sổ thật 18/09/2026:
net/gross = 0,000%). `build_rebalance_plan` rồi áp DẢI KHÔNG GIAO DỊCH 20% để khỏi
trả phí cho điều chỉnh vụn. Nhưng cặp bị bỏ qua GIỮ KHỐI LƯỢNG CŨ chứ không phải
khối lượng ĐÍCH — nên mỗi lần bỏ qua là một sai số tới 20% cỡ vị thế, và các sai số
đó KHÔNG tự triệt tiêu theo chiều.

Hệ quả đo được trên sổ thật:
    18/09: bỏ 2 cặp NGƯỢC chiều  -> net +0,56%  ✅ lượt sạch
    16/09: bỏ các cặp CÙNG chiều -> net -2,96%  ❌ lượt bẩn
    11/09: cùng cơ chế           -> net -3,16%  ❌ lượt bẩn

Đó là RÒ NGẪU NHIÊN, không phải hỏng cố định — nên nó thoát mọi test cũ, và nó là
lý do tỉ lệ lượt sạch chỉ ~50%. Cổng "3 lượt sạch liên tiếp" ở p=50% cần 42 ngày;
ở p=17% (tỉ lệ lịch sử thật) cần 2,1 NĂM. Một hằng số duy nhất giữ cả dự án ở cửa.

Cổng F25 chấm trung lập SAU khi thực thi — lúc lệnh đã nằm trên sàn thì đã muộn.
Test này khoá bất biến ở tầng KẾ HOẠCH, tức trước khi mất một đồng phí nào.
"""
import pytest

from aegis.execution.portfolio_rebalancer import (
    DEFAULT_NEUTRALITY_TOLERANCE,
    SymbolFilters,
    build_rebalance_plan,
)


def _filters(symbols, step=0.001, min_notional=5.0):
    return {s: SymbolFilters(symbol=s, tick_size=0.01, step_size=step,
                             min_qty=step, min_notional=min_notional)
            for s in symbols}


def _net_ratio(plan):
    assert plan.gross_notional > 0
    return plan.net_notional / plan.gross_notional


def test_dai_khong_giao_dich_cung_chieu_van_giu_trung_lap():
    """Tái dựng lượt 16/09: các cặp bị dải bỏ qua CÙNG nằm một chân."""
    syms = [f"L{i}USDT" for i in range(6)] + [f"S{i}USDT" for i in range(6)]
    prices = {s: 100.0 for s in syms}
    weights = {s: (1 / 12 if s.startswith("L") else -1 / 12) for s in syms}
    equity, leverage = 10_000.0, 2.0

    # Mỗi vị thế đích = 20000/12 = $1666.67 -> 16.667 đơn vị @ $100.
    target_units = equity * leverage / 12 / 100.0
    held = {}
    for s in syms:
        sign = 1.0 if s.startswith("L") else -1.0
        # 3 cặp LONG đã nắm dư 15% (< dải 20%) -> bị bỏ qua, chân long phình lên.
        held[s] = sign * target_units * (1.15 if s in ("L0USDT", "L1USDT", "L2USDT") else 1.0)

    plan = build_rebalance_plan(
        target_weights=weights, current_qty=held, prices=prices,
        filters=_filters(syms), equity=equity, leverage=leverage,
    )
    r = _net_ratio(plan)
    assert abs(r) <= DEFAULT_NEUTRALITY_TOLERANCE, (
        f"kế hoạch lệch {r*100:+.2f}% gross, vượt trần "
        f"±{DEFAULT_NEUTRALITY_TOLERANCE*100:.0f}% — dải không giao dịch lại phá "
        f"trung lập (đúng cơ chế đã làm hỏng lượt 16/09)"
    )


def test_lech_nguoc_chieu_thi_van_bo_qua_ca_hai_khong_ton_phi():
    """Tái dựng lượt 18/09: bỏ qua ngược chiều thì triệt tiêu — KHÔNG được sinh lệnh thừa."""
    syms = ["L0USDT", "L1USDT", "S0USDT", "S1USDT"]
    prices = {s: 100.0 for s in syms}
    weights = {"L0USDT": 0.25, "L1USDT": 0.25, "S0USDT": -0.25, "S1USDT": -0.25}
    equity, leverage = 10_000.0, 2.0
    tu = equity * leverage / 4 / 100.0

    held = {"L0USDT": tu * 1.1, "L1USDT": tu,        # long dư 10%
            "S0USDT": -tu * 1.1, "S1USDT": -tu}      # short dư 10% -> triệt tiêu
    plan = build_rebalance_plan(
        target_weights=weights, current_qty=held, prices=prices,
        filters=_filters(syms), equity=equity, leverage=leverage,
    )
    assert abs(_net_ratio(plan)) <= DEFAULT_NEUTRALITY_TOLERANCE
    assert not any(o.reason == "cân trung lập" for o in plan.orders), (
        "sổ vốn đã trung lập mà vẫn phát sinh lệnh cân — đang trả phí vô ích"
    )
    assert len(plan.skipped) == 2, "phải giữ nguyên tác dụng tiết kiệm phí của dải"


def test_khong_bia_trong_so_ngoai_portfolio():
    """Lệnh cân trung lập chỉ được đưa vị thế về ĐÚNG khối lượng đích — không bịa thêm."""
    syms = ["L0USDT", "L1USDT", "S0USDT", "S1USDT"]
    prices = {s: 100.0 for s in syms}
    weights = {"L0USDT": 0.25, "L1USDT": 0.25, "S0USDT": -0.25, "S1USDT": -0.25}
    equity, leverage = 10_000.0, 2.0
    tu = equity * leverage / 4 / 100.0
    held = {"L0USDT": tu * 1.15, "L1USDT": tu * 1.15, "S0USDT": -tu, "S1USDT": -tu}

    plan = build_rebalance_plan(
        target_weights=weights, current_qty=held, prices=prices,
        filters=_filters(syms), equity=equity, leverage=leverage,
    )
    for o in plan.orders:
        if o.reason != "cân trung lập":
            continue
        final = held[o.symbol] + (o.qty if o.side == "BUY" else -o.qty)
        assert final == pytest.approx(weights[o.symbol] * equity * leverage / 100.0, rel=1e-6), (
            f"{o.symbol} sau khi cân không bằng khối lượng đích — tầng live đang tự "
            f"bịa trọng số, đúng họ lỗi F3"
        )
