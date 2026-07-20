from aegis.execution.pnl import compute_realized_pnl

def test_pnl_normal_win():
    """[STREAMING_CHUNK: TEST_PNL_NORMAL] Lệnh thắng thông thường."""
    res = compute_realized_pnl(100.0, 110.0, 1, 1000.0, 10.0, "TRAIL")
    # Gross = 10% của 1000 = +100 USD. Fees = 2 * (1000 * 0.0005) = 1 USD. Net = +99 USD.
    assert abs(res["net_pnl"] - 99.0) < 1e-6
    # Realized return (Unleveraged) = 99 / 1000 = 0.099 (9.9%)
    assert abs(res["realized_return"] - 0.099) < 1e-6
    print("✅ [PNL ENGINE] Lệnh thắng thông thường PASSED!")

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
