"""
Test lớp tái cân bằng danh mục — cầu nối trọng số mục tiêu -> lệnh thật.

Trọng tâm: các ràng buộc CỨNG của sàn. Vi phạm bất kỳ điều nào cũng khiến lệnh bị
từ chối lúc chạy thật, và với danh mục 12 vị thế thì một lệnh hỏng làm lệch cả
trạng thái market-neutral.
"""
import pytest

from aegis.execution.portfolio_rebalancer import (
    RebalanceOrder,
    SymbolFilters,
    build_rebalance_plan,
    max_positions_for_capital,
)


def _filters(symbol="XRPUSDT", tick=0.0001, step=0.1, min_qty=0.1, min_notional=5.0):
    return SymbolFilters(symbol, tick, step, min_qty, min_notional)


def _std_setup():
    """3 cặp giá $1 với bước nhảy 0.1 — số học dễ kiểm bằng tay."""
    syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    return (
        {s: _filters(s) for s in syms},
        {s: 1.0 for s in syms},
    )


# ---------------------------------------------------------------- làm tròn
def test_round_qty_always_rounds_down():
    """Làm tròn XUỐNG: không bao giờ vượt quy mô dự tính (và vượt đòn bẩy)."""
    f = _filters(step=0.1)
    assert f.round_qty(1.99) == pytest.approx(1.9)
    assert f.round_qty(-1.99) == pytest.approx(-1.9)
    assert f.round_qty(0.05) == pytest.approx(0.0)


def test_round_price_snaps_to_tick():
    f = _filters(tick=0.01)
    assert f.round_price(1.2345) == pytest.approx(1.23)


def test_is_tradeable_enforces_both_limits():
    f = _filters(min_qty=1.0, min_notional=5.0)
    assert not f.is_tradeable(0.5, 100.0)   # dưới min_qty
    assert not f.is_tradeable(2.0, 1.0)     # dưới min_notional
    assert f.is_tradeable(6.0, 1.0)


# ------------------------------------------------------- ràng buộc min notional
def test_position_below_min_notional_is_skipped_not_shrunk():
    """
    Vị thế mục tiêu quá nhỏ phải bị BỎ HẲN, không được thu nhỏ rồi vẫn gửi.
    Gửi lệnh dưới min_notional là bị sàn từ chối, làm lệch trạng thái danh mục.
    """
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 1.0},
        current_qty={}, prices=prices, filters=filters,
        equity=3.0, leverage=1.0,   # gross $3 < min_notional $5
    )
    assert plan.orders == []
    assert "min_notional" in plan.skipped["AAAUSDT"]


def test_closing_is_always_allowed_even_below_min_notional():
    """
    Lệnh ĐÓNG luôn phải được phép — sàn cho đóng vị thế nhỏ hơn min_notional.
    Nếu chặn, ta sẽ mắc kẹt vĩnh viễn với vị thế không thể thoát.
    """
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={},                    # mục tiêu: phẳng hoàn toàn
        current_qty={"AAAUSDT": 2.0},         # đang giữ $2 < min_notional $5
        prices=prices, filters=filters, equity=100.0, leverage=1.0,
    )
    assert len(plan.orders) == 1
    assert plan.orders[0].side == "SELL"
    assert plan.orders[0].reason == "đóng"
    assert plan.orders[0].qty == pytest.approx(2.0)


# ---------------------------------------------------------------- long/short
def test_dollar_neutral_targets_produce_both_sides():
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 0.5, "BBBUSDT": -0.5},
        current_qty={}, prices=prices, filters=filters,
        equity=100.0, leverage=1.0,
    )
    sides = {o.symbol: o.side for o in plan.orders}
    assert sides["AAAUSDT"] == "BUY"
    assert sides["BBBUSDT"] == "SELL"
    assert plan.net_notional == pytest.approx(0.0, abs=1e-6)
    assert plan.gross_notional == pytest.approx(100.0, rel=1e-3)


def test_flipping_short_to_long_sizes_full_delta():
    """Đảo từ short sang long phải mua đủ cả phần đóng lẫn phần mở."""
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 1.0},
        current_qty={"AAAUSDT": -30.0},
        prices=prices, filters=filters, equity=50.0, leverage=1.0,
    )
    order = next(o for o in plan.orders if o.symbol == "AAAUSDT")
    assert order.side == "BUY"
    assert order.qty == pytest.approx(80.0)   # đóng 30 short + mở 50 long


def test_weights_are_normalised_to_gross_budget():
    """Trọng số không chuẩn hoá sẵn vẫn phải quy về đúng gross = equity*leverage."""
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 5.0, "BBBUSDT": -5.0},   # tổng |w| = 10
        current_qty={}, prices=prices, filters=filters,
        equity=100.0, leverage=2.0,
    )
    assert plan.gross_notional == pytest.approx(200.0, rel=1e-3)


# ------------------------------------------------------------- dải không giao dịch
def test_no_trade_band_suppresses_tiny_adjustments():
    """Lệch nhỏ phải bị bỏ qua — churn chỉ tạo phí, không tạo lợi nhuận."""
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 1.0},
        current_qty={"AAAUSDT": 95.0},        # mục tiêu 100, lệch 5%
        prices=prices, filters=filters, equity=100.0, leverage=1.0,
        no_trade_band=0.20,
    )
    assert plan.orders == []
    assert "dải" in plan.skipped["AAAUSDT"]


def test_large_deviation_does_trade():
    filters, prices = _std_setup()
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 1.0},
        current_qty={"AAAUSDT": 50.0},        # lệch 50% > dải 20%
        prices=prices, filters=filters, equity=100.0, leverage=1.0,
        no_trade_band=0.20,
    )
    assert len(plan.orders) == 1
    assert plan.orders[0].qty == pytest.approx(50.0)


# ---------------------------------------------------------------- an toàn vốn
def test_gross_notional_never_exceeds_equity_times_leverage():
    """Bất biến an toàn: tổng vị thế không được vượt vốn x đòn bẩy."""
    filters, prices = _std_setup()
    equity, lev = 38.0, 2.0
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 0.4, "BBBUSDT": -0.4, "CCCUSDT": 0.2},
        current_qty={}, prices=prices, filters=filters,
        equity=equity, leverage=lev,
    )
    assert plan.gross_notional <= equity * lev + 1e-6


def test_missing_price_or_filter_is_skipped_not_guessed():
    filters, prices = _std_setup()
    del prices["BBBUSDT"]
    plan = build_rebalance_plan(
        target_weights={"AAAUSDT": 0.5, "BBBUSDT": 0.5},
        current_qty={}, prices=prices, filters=filters,
        equity=100.0, leverage=1.0,
    )
    assert "BBBUSDT" in plan.skipped
    assert all(o.symbol != "BBBUSDT" for o in plan.orders)


def test_invalid_inputs_rejected():
    filters, prices = _std_setup()
    with pytest.raises(ValueError):
        build_rebalance_plan({}, {}, prices, filters, equity=0.0, leverage=1.0)
    with pytest.raises(ValueError):
        build_rebalance_plan({}, {}, prices, filters, equity=100.0, leverage=0.0)


# ---------------------------------------------------------------- sức chứa vốn
def test_max_positions_matches_capital_reality():
    """$38 (1 triệu VND) ở 2x phải đủ cho 12 vị thế — quy mô chiến lược đã kiểm định."""
    assert max_positions_for_capital(38.0, leverage=2.0, min_notional=5.0) >= 12
    assert max_positions_for_capital(38.0, leverage=1.0, min_notional=5.0) < 12


# ------------------------------------------------- hồi quy: lỗi precision thật
def test_rounding_produces_exchange_safe_decimals():
    """
    HỒI QUY — lỗi phát hiện khi chạy TESTNET THẬT:
        Binance -1111 "Precision is over the maximum defined for this asset"

    Nguyên nhân: `round(price/tick)*tick` để lại dư số dấu phẩy động
    (1.24263 -> 1.2426000000000002), vượt số chữ số thập phân sàn cho phép.
    Chỉ mock thì không bao giờ bắt được lỗi này.
    """
    import decimal

    cases = [
        # (tick_size, step_size, giá thô, qty thô)
        (0.0001, 0.1, 1.3807 * 0.90, 55.0),
        (0.1, 0.001, 78203.27, 0.0014999),
        (0.01, 0.001, 3456.789, 1.23456),
        (0.00001, 1.0, 0.0234567, 1234.9),
    ]
    for tick, step, raw_price, raw_qty in cases:
        f = _filters(tick=tick, step=step, min_qty=0.0, min_notional=0.0)

        price = f.round_price(raw_price)
        qty = f.round_qty(raw_qty)

        # Kiểm CHÍNH CHUỖI gửi lên sàn — đó mới là thứ Binance nhìn thấy.
        price_str, qty_str = f.format_price(price), f.format_qty(qty)
        price_dp = -decimal.Decimal(price_str).as_tuple().exponent
        qty_dp = -decimal.Decimal(qty_str).as_tuple().exponent

        assert price_dp <= f._decimals(tick), (
            f"chuỗi giá {price_str!r} có {price_dp} chữ số thập phân, tick {tick} chỉ cho {f._decimals(tick)}"
        )
        assert qty_dp <= f._decimals(step), (
            f"chuỗi qty {qty_str!r} có {qty_dp} chữ số thập phân, step {step} chỉ cho {f._decimals(step)}"
        )
        assert float(price_str) == pytest.approx(price)
        assert float(qty_str) == pytest.approx(qty)
        # vẫn phải là bội số hợp lệ của bước nhảy
        assert abs(price / tick - round(price / tick)) < 1e-6
        assert abs(qty / step - round(qty / step)) < 1e-6


def test_decimals_helper():
    f = _filters()
    assert f._decimals(0.0001) == 4
    assert f._decimals(0.1) == 1
    assert f._decimals(1.0) == 0
    assert f._decimals(0.00001) == 5
