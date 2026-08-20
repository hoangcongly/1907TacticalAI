import pytest
import inspect
from aegis.execution.pnl import compute_realized_pnl
import yaml
import os

@pytest.fixture(scope="session")
def canonical_registry():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "aegis_canonical_parameters.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
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


def test_pnl_normal_win():
    """[STREAMING_CHUNK: TEST_PNL_NORMAL] Lệnh thắng thông thường (đã vá Exit Fee)."""
    res = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 10.0, "TRAIL", "taker", "taker")
    # Gross = 10% của 1000 = +100 USD. Exit Notional = 1100 USD.
    # Fee Entry = 1000 * 0.0004 = 0.4 USD. Fee Exit = 1100 * 0.0004 = 0.44 USD -> Total Fee = 0.84 USD.
    # Net = 100.0 - 0.84 = 99.16 USD.
    assert abs(res["net_pnl"] - 99.16) < 1e-6
    # Realized return (Unleveraged) = 99.16 / 1000 = 0.09916 (9.916%)
    assert abs(res["realized_return"] - 0.09916) < 1e-6
    print("✅ [PNL ENGINE] Lệnh thắng thông thường PASSED!")


def test_pnl_exit_fee_accounting_flaw_fixed():
    """
    [Vá BỌ SỐ 1: Exit Fee Accounting Flaw] Kiểm chứng phí thoát lệnh tính đúng theo Exit Notional:
    - Khi thắng 50% (Gross = +5,000 USD trên 10,000 USD), Exit Notional = 15,000 USD -> Fee Exit = 7.5 USD.
    - Khi thua 10% (Gross = -1,000 USD trên 10,000 USD), Exit Notional = 9,000 USD -> Fee Exit = 4.5 USD.
    """
    # 1. Thắng 50%
    win_res = compute_realized_pnl(100.0, 150.0, 1, 10000.0, 5.0, "TRAIL", "taker", "taker", taker_fee_rate=0.0005)
    # Fee Entry = 5.0, Fee Exit = 15000 * 0.0005 = 7.5 -> Total fee = 12.5
    assert abs(win_res["fee_paid"] - 12.5) < 1e-6, f"Sai fee lệnh thắng to: {win_res['fee_paid']}"
    assert abs(win_res["net_pnl"] - (5000.0 - 12.5)) < 1e-6

    # 2. Thua 10%
    loss_res = compute_realized_pnl(100.0, 90.0, 1, 10000.0, 5.0, "SL", "taker", "taker", taker_fee_rate=0.0005)
    # Fee Entry = 5.0, Fee Exit = 9000 * 0.0005 = 4.5 -> Total fee = 9.5
    assert abs(loss_res["fee_paid"] - 9.5) < 1e-6, f"Sai fee lệnh lỗ: {loss_res['fee_paid']}"
    assert abs(loss_res["net_pnl"] - (-1000.0 - 9.5)) < 1e-6
    print("✅ [Vá BỌ SỐ 1] Exit Fee Accounting Flaw PASSED!")

def test_pnl_liquidation_branch():
    """
    [QĐ #7] Nhánh thanh lý xử lý phí THỐNG NHẤT với nhánh Normal.
    gross = -(margin) = -(1000/10) = -100
    fee_entry = 1000 * 0.0004 = 0.4
    net = -100 - 0.4 = -100.4
    """
    res = compute_realized_pnl(100.0, 90.0, 1, 1000.0, 10.0, "LIQUIDATION", "taker", "taker")
    # Gross = -(margin) = -100.0
    assert abs(res["gross_pnl"] - (-100.0)) < 1e-6, f"Gross sai: {res['gross_pnl']}"
    # Net = gross - fee_entry = -100.0 - 0.4 = -100.4
    assert abs(res["net_pnl"] - (-100.4)) < 1e-6, f"Net sai: {res['net_pnl']}"
    # Fee reported = 0.4
    assert abs(res["fee_paid"] - 0.4) < 1e-6, f"Fee sai: {res['fee_paid']}"
    print("✅ [QĐ #7] PnL nhánh Liquidation thống nhất phí PASSED!")

def test_pnl_leverage_does_not_affect_normal_exit():
    """Đòn bẩy KHÔNG ảnh hưởng PnL của lệnh thoát bình thường."""
    res_lev_2 = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 2.0, "TRAIL", "taker", "taker")
    res_lev_10 = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 10.0, "TRAIL", "taker", "taker")
    assert res_lev_2["net_pnl"] == res_lev_10["net_pnl"], (
        "Đòn bẩy đã rò rỉ vào PnL lệnh thoát bình thường!"
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
