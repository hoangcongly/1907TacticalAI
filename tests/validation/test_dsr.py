import pytest
import math
from aegis.validation.dsr import (
    compute_deflated_sharpe_ratio,
    compute_dsr_sensitivity,
    euler_mascheroni_approx_max_sr
)


def test_euler_mascheroni_approx_max_sr():
    """Kiểm chứng tính xấp xỉ kỳ vọng SR lớn nhất khi N thử nghiệm tăng lên."""
    sr_0 = 0.0
    var_sr = 0.5 ** 2  # std_sr = 0.5
    
    max_30 = euler_mascheroni_approx_max_sr(sr_0, 30, var_sr)
    max_100 = euler_mascheroni_approx_max_sr(sr_0, 100, var_sr)
    max_200 = euler_mascheroni_approx_max_sr(sr_0, 200, var_sr)

    # N tăng thì kỳ vọng SR tối đa dưới giả thuyết Null phải tăng theo (Hurdle rate cao hơn)
    assert max_30 < max_100 < max_200, f"Lỗi đơn điệu: {max_30} vs {max_100} vs {max_200}"
    print("✅ [DSR ENGINE] Euler-Mascheroni Approx Max SR PASSED!")


def test_dsr_sensitivity_analysis_directive():
    """
    [MODULE E SENSITIVITY DIRECTIVE — N = 30, 100, 200]:
    Kiểm chứng phân tích độ nhạy của Deflated Sharpe Ratio.
    Khi số lượng cấu hình backtest (N trials) tăng từ 30 lên 100, 200:
    -> SR kỳ vọng tối đa (SR_expected_max) tăng lên.
    -> DSR bị chiết khấu mạnh hơn (giảm xuống).
    """
    sr_est = 1.2       # Sharpe ước lượng của chiến lược = 1.2
    var_srs = 0.4 ** 2 # Phương sai các SR được thử nghiệm (std = 0.4)
    sample_len = 100   # 100 quan sát (tránh z-score bão hòa float64 ở 1.0)

    sensitivity = compute_dsr_sensitivity(
        sr_estimated=sr_est,
        variance_of_srs=var_srs,
        sample_length=sample_len,
        n_trials_list=[30, 100, 200]
    )

    dsr_30 = sensitivity[30]["dsr"]
    dsr_100 = sensitivity[100]["dsr"]
    dsr_200 = sensitivity[200]["dsr"]

    # Khẳng định DSR giảm dần khi số lần thử nghiệm N tăng lên
    assert dsr_30 > dsr_100 > dsr_200, (
        f"Lỗi chiết khấu multiple testing! DSR phải giảm khi N tăng: {dsr_30} vs {dsr_100} vs {dsr_200}"
    )

    # Khẳng định SR_expected_max tăng dần
    assert sensitivity[30]["sr_expected_max"] < sensitivity[100]["sr_expected_max"] < sensitivity[200]["sr_expected_max"]

    print(
        f"✅ [MODULE E SENSITIVITY] DSR(N=30)={dsr_30:.4f} | "
        f"DSR(N=100)={dsr_100:.4f} | DSR(N=200)={dsr_200:.4f} -> PASSED!"
    )


def test_dsr_armor_plated_guards():
    """Kiểm thử hải quan bọc thép cho DSR."""
    with pytest.raises(ValueError, match="sample_length phải là số nguyên > 1"):
        compute_deflated_sharpe_ratio(1.5, 0.1, sample_length=1)

    with pytest.raises(ValueError, match="num_trials phải là số nguyên >= 1"):
        compute_deflated_sharpe_ratio(1.5, 0.1, sample_length=252, num_trials=0)

    with pytest.raises(ValueError, match="không được âm"):
        compute_deflated_sharpe_ratio(1.5, -0.1, sample_length=252, num_trials=10)

    with pytest.raises(ValueError, match="không hợp lệ"):
        compute_deflated_sharpe_ratio(float('nan'), 0.1, sample_length=252)

    print("✅ [DSR ENGINE] Armor-plated guards PASSED!")
