from aegis.execution.pnl import compute_realized_pnl

def test_pnl_normal_win():
    """[STREAMING_CHUNK: TEST_PNL_NORMAL] Lệnh thắng thông thường (đã vá Exit Fee)."""
    res = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 10.0, "TRAIL")
    # Gross = 10% của 1000 = +100 USD. Exit Notional = 1100 USD.
    # Fee Entry = 1000 * 0.0005 = 0.5 USD. Fee Exit = 1100 * 0.0005 = 0.55 USD -> Total Fee = 1.05 USD.
    # Net = 100.0 - 1.05 = 98.95 USD.
    assert abs(res["net_pnl"] - 98.95) < 1e-6
    # Realized return (Unleveraged) = 98.95 / 1000 = 0.09895 (9.895%)
    assert abs(res["realized_return"] - 0.09895) < 1e-6
    print("✅ [PNL ENGINE] Lệnh thắng thông thường PASSED!")


def test_pnl_exit_fee_accounting_flaw_fixed():
    """
    [Vá BỌ SỐ 1: Exit Fee Accounting Flaw] Kiểm chứng phí thoát lệnh tính đúng theo Exit Notional:
    - Khi thắng 50% (Gross = +5,000 USD trên 10,000 USD), Exit Notional = 15,000 USD -> Fee Exit = 7.5 USD.
    - Khi thua 10% (Gross = -1,000 USD trên 10,000 USD), Exit Notional = 9,000 USD -> Fee Exit = 4.5 USD.
    """
    # 1. Thắng 50%
    win_res = compute_realized_pnl(100.0, 150.0, 1, 10000.0, 5.0, "TRAIL", fee_entry_rate=0.0005, fee_exit_rate=0.0005)
    # Fee Entry = 5.0, Fee Exit = 15000 * 0.0005 = 7.5 -> Total fee = 12.5
    assert abs(win_res["fee_paid"] - 12.5) < 1e-6, f"Sai fee lệnh thắng to: {win_res['fee_paid']}"
    assert abs(win_res["net_pnl"] - (5000.0 - 12.5)) < 1e-6

    # 2. Thua 10%
    loss_res = compute_realized_pnl(100.0, 90.0, 1, 10000.0, 5.0, "SL", fee_entry_rate=0.0005, fee_exit_rate=0.0005)
    # Fee Entry = 5.0, Fee Exit = 9000 * 0.0005 = 4.5 -> Total fee = 9.5
    assert abs(loss_res["fee_paid"] - 9.5) < 1e-6, f"Sai fee lệnh lỗ: {loss_res['fee_paid']}"
    assert abs(loss_res["net_pnl"] - (-1000.0 - 9.5)) < 1e-6
    print("✅ [Vá BỌ SỐ 1] Exit Fee Accounting Flaw PASSED!")

def test_pnl_liquidation_branch():
    """
    [QĐ #7] Nhánh thanh lý xử lý phí THỐNG NHẤT với nhánh Normal.
    gross = -(margin) = -(1000/10) = -100
    fee_entry = 1000 * 0.0005 = 0.5
    net = -100 - 0.5 = -100.5
    """
    res = compute_realized_pnl(100.0, 90.0, 1, 1000.0, 10.0, "LIQUIDATION")
    # Gross = -(margin) = -100.0
    assert abs(res["gross_pnl"] - (-100.0)) < 1e-6, f"Gross sai: {res['gross_pnl']}"
    # Net = gross - fee_entry = -100.0 - 0.5 = -100.5
    assert abs(res["net_pnl"] - (-100.5)) < 1e-6, f"Net sai: {res['net_pnl']}"
    # Fee reported = 0.5
    assert abs(res["fee_paid"] - 0.5) < 1e-6, f"Fee sai: {res['fee_paid']}"
    print("✅ [QĐ #7] PnL nhánh Liquidation thống nhất phí PASSED!")

def test_pnl_leverage_does_not_affect_normal_exit():
    """Đòn bẩy KHÔNG ảnh hưởng PnL của lệnh thoát bình thường."""
    res_lev_2 = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 2.0, "TRAIL")
    res_lev_10 = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 10.0, "TRAIL")
    assert res_lev_2["net_pnl"] == res_lev_10["net_pnl"], (
        "Đòn bẩy đã rò rỉ vào PnL lệnh thoát bình thường!"
    )
    print("✅ [QĐ #7] Leverage không ảnh hưởng PnL Normal PASSED!")


def test_pnl_liquidation_does_not_double_count_funding():
    """
    [Issue #4 / QĐ #7] Kiểm chứng nhánh Liquidation KHÔNG trừ thêm funding_accrued_usd
    để chống lỗi Đếm Kép (Double-Count) tiền Funding Fee đã bị trừ vào Ký quỹ trước khi thanh lý.
    """
    # Lệnh có funding_accrued_usd = 15.0 USD
    res_with_funding = compute_realized_pnl(
        entry_price=100.0, exit_price=90.0, side=1, size_notional=1000.0, leverage=10.0,
        exit_reason="LIQUIDATION", funding_accrued_usd=15.0
    )
    res_no_funding = compute_realized_pnl(
        entry_price=100.0, exit_price=90.0, side=1, size_notional=1000.0, leverage=10.0,
        exit_reason="LIQUIDATION", funding_accrued_usd=0.0
    )
    assert res_with_funding["net_pnl"] == res_no_funding["net_pnl"], (
        f"Lỗi Đếm Kép! Funding fee đã bị trừ lần hai vào nhánh Liquidation: {res_with_funding['net_pnl']} vs {res_no_funding['net_pnl']}"
    )
    print("✅ [Issue #4 / QĐ #7] Chống Đếm Kép Funding Fee nhánh Liquidation PASSED!")

