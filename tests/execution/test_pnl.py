import pytest
import inspect
from aegis.execution.pnl import compute_realized_pnl
import yaml
import os

@pytest.fixture(scope="session")
def canonical_registry():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "aegis_canonical_parameters.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_no_hardcoded_fee_in_signature(canonical_registry):
    """G0-2 / F3 / NT-3: So default THẬT của hàm với registry, không chép literal."""
    sig = inspect.signature(compute_realized_pnl)
    default_maker = sig.parameters["maker_fee_rate"].default
    default_taker = sig.parameters["taker_fee_rate"].default
    
    reg_maker = canonical_registry["execution_costs"]["maker_fee_rate"]["value"]
    reg_taker = canonical_registry["execution_costs"]["taker_fee_rate"]["value"]
    
    assert default_maker == reg_maker, f"Maker fee default {default_maker} != registry {reg_maker}"
    assert default_taker == reg_taker, f"Taker fee default {default_taker} != registry {reg_taker}"


@pytest.mark.parametrize("leverage", [3.0, 5.0, 10.0, 20.0, 50.0, 100.0])
def test_liquidation_return_leverage_invariant(leverage):
    """F1: cùng một cú sập giá phải cho cùng một realized_return, bất kể đòn bẩy."""
    base = compute_realized_pnl(
        entry_price=100.0, 
        exit_price=90.0, 
        side=1, 
        size_notional=1000.0, 
        leverage=10.0, 
        exit_reason="LIQUIDATION",
        entry_fill_type="taker",
        exit_fill_type="taker"
    )
    r = compute_realized_pnl(
        entry_price=100.0, 
        exit_price=90.0, 
        side=1, 
        size_notional=1000.0, 
        leverage=leverage, 
        exit_reason="LIQUIDATION",
        entry_fill_type="taker",
        exit_fill_type="taker"
    )
    assert abs(r["realized_return"] - base["realized_return"]) < 1e-9, (
        f"leverage={leverage} cho return={r['realized_return']:.6f}, "
        f"khác baseline={base['realized_return']:.6f} → Kelly sẽ bị đầu độc"
    )


def test_fee_by_fill_type():
    """F4: Maker/taker fee calculation based on fill type."""
    # Lệnh mở taker, thoát taker
    taker_res = compute_realized_pnl(
        entry_price=100.0, exit_price=110.0, side=1, size_notional=1000.0, leverage=1.0, 
        exit_reason="TRAIL", entry_fill_type="taker", exit_fill_type="taker"
    )
    # Lệnh mở maker, thoát maker
    maker_res = compute_realized_pnl(
        entry_price=100.0, exit_price=110.0, side=1, size_notional=1000.0, leverage=1.0, 
        exit_reason="TRAIL", entry_fill_type="maker", exit_fill_type="maker"
    )
    
    assert taker_res["fee_paid"] > maker_res["fee_paid"], (
        "Phí Taker phải lớn hơn phí Maker. Nếu không, Maker/Taker chưa được đấu vào logic."
    )
    # Taker fee 0.04% x 1000 = 0.4. Exit 0.04% x 1100 = 0.44. Total = 0.84.
    assert abs(taker_res["fee_paid"] - 0.84) < 1e-6
    # Maker fee 0.01% x 1000 = 0.1. Exit 0.01% x 1100 = 0.11. Total = 0.21.
    assert abs(maker_res["fee_paid"] - 0.21) < 1e-6


def test_liquidation_no_double_count_funding():
    """F2: Không thu đúp phí funding ở nhánh thanh lý."""
    res_pay = compute_realized_pnl(
        entry_price=100.0, exit_price=90.0, side=1, size_notional=1000.0, leverage=10.0,
        exit_reason="LIQUIDATION", entry_fill_type="taker", exit_fill_type="taker", funding_accrued_usd=15.0
    )
    # PnL Gross = -100 (margin). Fee = 0.4 (Taker on entry). Total Net PnL = -100.4
    # TUYỆT ĐỐI KHÔNG trừ đi 15.0 funding.
    assert abs(res_pay["net_pnl"] - (-100.4)) < 1e-6, f"Sai Net PnL, có thể đang bị đếm kép funding: {res_pay['net_pnl']}"
