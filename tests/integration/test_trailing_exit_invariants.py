import pytest
import numpy as np
from aegis.labeling.trailing_exit import compute_regime_aware_trailing_exit_v3_liquidation_aware

def test_trailing_stop_uses_previous_bar_extreme_not_current_bar():
    """
    [TASK B-1] (Phát hiện 4) Kiểm chứng rằng extreme_price được cập nhật SAU KHI kiểm tra Trail.
    Nếu bị cập nhật trước (Intrabar Path Ambiguity / Intra-bar Look-ahead Bias), nến hiện tại
    tự dùng râu của nó để đẩy Trail SL lên, và thoát lệnh ngay trong cùng nến một cách vô lý.
    """
    # Lệnh Long, entry = 100
    # Ngày 1: Nến đẩy lên 110, tạo extreme = 110
    # Ngày 2: Nến có râu rất dài đẩy lên 150, sau đó sập xuống 105
    # Nếu cập nhật extreme TRƯỚC: extreme = 150, Trail SL = 150 - (2 * 5) = 140. Giá Low=105 sẽ chạm 140 -> Cắt lỗ Trail ở nến 2.
    # Nếu cập nhật extreme SAU: extreme cũ = 110. Trail SL = 110 - (2 * 5) = 100. Giá Low=105 KHÔNG chạm 100 -> KHÔNG bị cắt lỗ.
    
    future_highs = np.array([110.0, 150.0])
    future_lows = np.array([105.0, 105.0])
    future_atr = np.array([5.0, 5.0])
    future_p_trend = np.array([1.0, 1.0]) # 1.0 -> Trend mạnh, không bị REGIME_FLIP
    
    result = compute_regime_aware_trailing_exit_v3_liquidation_aware(
        entry_price=100.0,
        side=1,
        trade_mode="follow",
        future_highs=future_highs,
        future_lows=future_lows,
        future_atr=future_atr,
        future_p_trend=future_p_trend,
        sl_initial=90.0,
        liquidation_price=80.0,
        t_max_live=10,
        m_trail_base=2.0, # 100 - 2.0 * 5 = 90 (SL)
        gamma=0.0
    )
    
    # Kỳ vọng: Không bị cắt lỗ Trail ở ngày 2 (do extreme=110, trail_sl=100 < low=105)
    # Lệnh sẽ tiếp tục giữ và bị TIME_STOP ở nến cuối.
    assert result["reason"] == "TIME_STOP", (
        f"Lỗi Intrabar Order! Bị dính Trail ảo do tự dùng râu nến hiện tại đẩy Trail SL lên. "
        f"Nhận được lý do exit: {result['reason']} tại bar {result['exit_idx']}"
    )

if __name__ == "__main__":
    pytest.main(["-v", __file__])
