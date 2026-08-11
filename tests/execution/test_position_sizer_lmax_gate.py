import pytest
import numpy as np
from aegis.execution.position_sizer import compute_position_size, AccountStateTracker

def test_position_sizer_lmax_gate():
    """
    F6 / G1: Mandatory Safety Gate (L_max).
    Bất biến cứng: size_notional / effective_equity <= max_safe_leverage + 1e-9
    """
    np.random.seed(42)
    for _ in range(100):
        f_star = np.random.uniform(0, 20.0)
        equity = np.random.uniform(100, 1e7)
        vol_ratio = np.random.uniform(0.1, 10.0)
        
        atr_hist_mean = 1.0
        atr_current = atr_hist_mean / vol_ratio
        
        max_safe_leverage = np.random.uniform(1.0, 50.0)
        
        size_notional = compute_position_size(
            f_star=f_star,
            current_equity=equity,
            lambda_kelly=0.5,
            atr_hist_mean_pct=atr_hist_mean,
            atr_current_pct=atr_current,
            max_safe_leverage=max_safe_leverage
        )
        
        implied_leverage = size_notional / equity
        assert implied_leverage <= max_safe_leverage + 1e-9, f"L_max gate failed! Implied: {implied_leverage}, Max: {max_safe_leverage}"

def test_lmax_missing_raises_error():
    """
    F6 / G1: Nếu không truyền max_safe_leverage, TypeError phải văng ra (do tham số bắt buộc).
    """
    with pytest.raises(TypeError):
        # Thiếu max_safe_leverage
        compute_position_size(f_star=2.0, current_equity=1000.0)
