"""
Test hồi quy cho FIX F7 / F11 / F12 — các lỗ hổng rủi ro trên đường tiền live.

F7: Circuit Breaker trước đây chỉ nhìn wallet_balance (tiền mặt ĐÃ chốt) nên một
    vị thế đang mở có thể lỗ sạch tài khoản mà vẫn báo NORMAL.
F12: Lệnh dừng trước đây khớp đúng bằng sl_current (không trượt giá).
"""
import json
import numpy as np
import pytest

from aegis.pipelines.live_pipeline import AegisLivePipeline
from aegis.risk.circuit_breaker import CircuitBreakerTier


class _ConstantModel:
    """Model giả luôn trả P(y=1) cao để ép pipeline mở lệnh."""

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.05), np.full(n, 0.95)])


def _flat_kelly_dict(f_value: float = 1.0, num_bins: int = 2):
    grid = [[f_value] * num_bins for _ in range(num_bins)]
    edges = list(np.linspace(0.0, 1.0, num_bins + 1))
    chop = [list(np.linspace(0.0, 1.0, num_bins + 1)) for _ in range(num_bins)]
    block = {"grid": grid, "p_edges": edges, "chop_edges": chop}
    return {"kelly_follow": block, "kelly_fade": json.loads(json.dumps(block))}


def _make_pipeline(initial_capital: float = 1000.0) -> AegisLivePipeline:
    return AegisLivePipeline(
        model=_ConstantModel(),
        selected_features=["ofi"],
        kelly_dict=_flat_kelly_dict(),
        metadata={},
        initial_capital=initial_capital,
        config={"fade_enabled": False, "max_safe_leverage": 3.0, "lot_step_size": 1.0},
    )


def _bar(idx: int, price: float, atr: float = 1.0, **kw):
    bar = {
        "bar_idx": idx,
        "timestamp_ms": 1_700_000_000_000 + idx * 60_000,
        "symbol": "BTCUSDT",
        "open": price,
        "high": price + 0.01,
        "low": price - 0.01,
        "close": price,
        "volume": 100.0,
        "atr_14": atr,
        "p_trend": 0.8,
        "p_chop": 0.2,
        "trend_score": 1.0,
        "ofi": 0.5,
        "is_toxic_flag": False,
        "insufficient_history": False,
    }
    bar.update(kw)
    return bar


def test_unrealized_pnl_is_marked_to_market_on_open_position():
    """uPnL phải được cập nhật mỗi nến khi còn vị thế mở (trước đây luôn = 0.0)."""
    pipe = _make_pipeline()

    # Đẩy giá tăng dần để kích hoạt CUSUM và mở lệnh Long.
    idx = 0
    for price in [100.0, 101.0, 103.0, 106.0, 110.0]:
        pipe.on_bar(_bar(idx, price))
        idx += 1

    assert pipe.current_position is not None, "Cần có vị thế mở để kiểm tra mark-to-market"
    pos = pipe.current_position
    entry = pos.entry_price

    # Giá nhích nhẹ có lợi và vẫn nằm TRÊN trailing SL hiện hành -> lệnh chưa đóng,
    # nên uPnL phải khác 0 và đúng dấu.
    nudge = pos.sl_current * 1.02
    pipe.on_bar(_bar(idx, nudge, high=nudge + 0.01, low=nudge - 0.01))
    assert pipe.current_position is not None, "Lệnh phải còn mở để kiểm tra uPnL"

    expected = ((nudge - entry) / entry) * pos.size_notional * pos.side
    assert pipe.account_tracker.unrealized_pnl == pytest.approx(expected, rel=1e-9)
    assert pipe.account_tracker.unrealized_pnl != 0.0
    assert pipe.account_tracker.margin_balance != pipe.account_tracker.wallet_balance


def test_circuit_breaker_sees_open_position_drawdown():
    """
    Sụt giảm equity do vị thế MỞ phải đẩy Circuit Breaker rời khỏi NORMAL.
    Trước FIX F7, CB chấm trên wallet_balance nên luôn đứng yên ở NORMAL.
    """
    pipe = _make_pipeline(initial_capital=1000.0)

    idx = 0
    for price in [100.0, 101.0, 103.0, 106.0, 110.0]:
        pipe.on_bar(_bar(idx, price))
        idx += 1
    assert pipe.current_position is not None

    pos = pipe.current_position
    peak_before = pipe.circuit_breaker.peak_equity

    # Ép giá giảm sâu nhưng SL bị đẩy ra rất xa (ATR khổng lồ) để vị thế KHÔNG bị
    # đóng — mô phỏng đúng kịch bản "lỗ nặng khi lệnh vẫn đang mở".
    pos.sl_current = 0.01
    pos.sl_initial = 0.01
    crash = pos.entry_price * 0.90
    pipe.on_bar(_bar(idx, crash, atr=1000.0, high=crash, low=crash))

    assert pipe.account_tracker.unrealized_pnl < 0.0
    # Equity mark-to-market đã sụt so với đỉnh -> CB phải phản ứng.
    dd = (peak_before - pipe.account_tracker.margin_balance) / peak_before
    assert dd > 0.05, f"Kịch bản test phải tạo drawdown > 5%, thực tế {dd:.2%}"

    state = pipe.circuit_breaker.update_equity(
        pipe.account_tracker.margin_balance, current_time_ms=1_700_000_999_000
    )
    assert state.tier != CircuitBreakerTier.NORMAL, (
        "Circuit Breaker phải rời NORMAL khi equity mark-to-market sụt quá ngưỡng Tier 1"
    )


def test_stop_loss_fill_includes_slippage():
    """Giá khớp lệnh dừng phải xấu hơn sl_current (trượt giá), không khớp đúng bằng."""
    pipe = _make_pipeline()

    idx = 0
    for price in [100.0, 101.0, 103.0, 106.0, 110.0]:
        pipe.on_bar(_bar(idx, price))
        idx += 1
    assert pipe.current_position is not None

    pos = pipe.current_position
    sl = pos.sl_current
    assert pipe._exit_slippage_pct() > 0.0

    # Nến quét thủng SL: open nằm trên SL nên giá khớp lý thuyết = sl_current.
    res = pipe.on_bar(_bar(idx, sl - 0.5, open=sl + 0.2, high=sl + 0.3, low=sl - 0.6))

    assert res["action"] == "CLOSE"
    fill = res["trade_record"]["fill_price_exit"]
    assert fill < sl, f"Long phải khớp THẤP hơn SL do trượt giá: fill={fill} sl={sl}"
    assert fill >= sl - 0.6, "Giá khớp không được thấp hơn low của nến"
