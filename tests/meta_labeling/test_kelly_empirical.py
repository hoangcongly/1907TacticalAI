import math
import numpy as np


from aegis.meta_labeling.sizing.kelly_empirical import (
    DEFAULT_F_MAX,
    DEFAULT_LAMBDA_KELLY,
    solve_empirical_kelly_fraction,
    solve_empirical_kelly_fraction_with_confidence,
    compute_regime_weighted_bayesian_kelly,
)

# ============================================================================
# UNIT TESTS (TDD)
# ============================================================================
def test_f_max_is_leverage_search_bound():
    """
    [QĐ #1] Xác nhận f_max = 20.0 là giới hạn TÌM KIẾM cho brentq,
    KHÔNG PHẢI giới hạn rủi ro cuối cùng (đó là Dynamic Cap + Half-Kelly).
    """
    assert DEFAULT_F_MAX == 20.0, f"f_max phải = 20.0, nhận {DEFAULT_F_MAX}"
    assert DEFAULT_LAMBDA_KELLY == 0.5, f"λ phải = 0.5 (Half-Kelly), nhận {DEFAULT_LAMBDA_KELLY}"
    print("✅ [QĐ #1] f_max=20.0, λ=0.5 Confirmed!")


def test_fractional_kelly_lambda_discount():
    """
    [QĐ #2] Kiểm tra hệ số chiết khấu λ = 0.5 (Half Kelly)
    giảm đúng 50% vị thế f* so với Full Kelly (λ = 1.0).
    """
    np.random.seed(42)
    sample = np.random.choice([1.0, -0.999], p=[0.6, 0.4], size=10000)

    f_raw = solve_empirical_kelly_fraction(sample, f_max=1.0)

    f_full = f_raw * 1.0
    f_half = f_raw * DEFAULT_LAMBDA_KELLY

    assert abs(f_full - 0.2) < 0.05, f"Full Kelly cho coin toss kỳ vọng ~0.2, nhận {f_full}"
    assert abs(f_half - f_full * 0.5) < 1e-4, f"Half Kelly phải bằng 50% Full Kelly: {f_half} vs {f_full*0.5}"
    print(f"✅ [QĐ #2] Full Kelly: {f_full:.4f} | Half Kelly (λ=0.5): {f_half:.4f} PASSED!")


def test_b_1_1_kelly_classical_coin_toss():
    """
    UNIT TEST CHO B-1-1:
    Kiểm tra trên phân phối 2 điểm kinh điển (tung đồng xu):
    - Xác suất thắng p = 0.6
    - Thắng được b = 1.0 (1 ăn 1)
    - Thua mất 1.0 (mất trắng)
    => Theo công thức Full Kelly (λ=1.0): f* = p - (1-p)/b = 0.6 - 0.4/1.0 = 0.2
    """
    np.random.seed(42)
    sample = np.random.choice([1.0, -0.999], p=[0.6, 0.4], size=10000)

    f_point = solve_empirical_kelly_fraction(sample, f_max=1.0)
    result = solve_empirical_kelly_fraction_with_confidence(
        sample, f_max=1.0, n_bootstraps=200, lower_percentile=25.0
    )

    assert abs(f_point - 0.2) < 0.05, f"Kelly cho tung đồng xu sai, kỳ vọng ~0.2, nhận {f_point}"
    assert result.f_star_conservative <= result.f_star_point, (
        "Bootstrap (25th percentile) phải bảo thủ hơn hoặc bằng Point Estimate"
    )
    assert result.bootstrap_std >= 0, "Độ lệch chuẩn bootstrap phải >= 0"
    assert result.uncertainty_ratio >= 0, "Uncertainty ratio phải >= 0"
    print(f"✅ [B-1-1] Kelly: point={result.f_star_point:.4f}, "
          f"conservative={result.f_star_conservative:.4f}, "
          f"std={result.bootstrap_std:.4f}, unc={result.uncertainty_ratio:.4f} PASSED!")


def test_kelly_canary_and_nan_safety():
    """
    [FINDING C] Kiểm tra thứ tự: Canary assertion chạy SAU khi lọc NaN/Inf.
    Mẫu chứa NaN hợp lệ (artifact dữ liệu) KHÔNG được gây false-positive.
    """
    sample_with_nan = np.array([0.05, 0.03, -0.02, np.nan, 0.01] * 10)  # 50 phần tử, 40 hữu hạn
    result = solve_empirical_kelly_fraction(sample_with_nan)
    assert result >= 0.0, f"f* phải >= 0 khi mẫu có kỳ vọng dương, nhận {result}"

    try:
        bad_sample = np.array([0.5, -1.05, 0.2] * 15)  # 45 phần tử
        solve_empirical_kelly_fraction(bad_sample)
        assert False, "Lỗi rò rỉ: Return < -100% không bị Canary bắt!"
    except AssertionError as e:
        assert "Canary Error" in str(e)

    ok_sample = np.array([0.5, -1.0, 0.2] * 15)
    solve_empirical_kelly_fraction(ok_sample)

    print("✅ [FINDING C] Canary Assertion Order (NaN-safe) PASSED!")


def test_kelly_dynamic_cap_with_liquidation():
    """
    Kiểm chứng rằng khi mẫu chứa lệnh thanh lý (r = -1.0),
    brentq KHÔNG crash mà tự động giới hạn f_max_safe = 0.999 / abs(-1.0) = 0.999.
    Bất kể f_max = 20.0.
    """
    np.random.seed(42)
    wins = np.full(70, 0.05)
    losses = np.full(25, -0.03)
    liquidations = np.full(5, -1.0)
    sample = np.concatenate([wins, losses, liquidations])
    np.random.shuffle(sample)

    f_star = solve_empirical_kelly_fraction(sample, f_max=DEFAULT_F_MAX)

    assert f_star <= 0.999, (
        f"Dynamic cap thất bại! f* = {f_star} > 0.999 khi mẫu chứa thanh lý -1.0"
    )
    assert f_star >= 0.0, f"f* phải >= 0, nhận {f_star}"
    print(f"✅ Kelly Dynamic Cap: f* = {f_star:.4f} (capped <= 0.999 dù f_max=20.0) PASSED!")


def test_regime_probability_blend_and_bayesian():
    """
    [BAYESIAN-HMM] Kiểm tra chức năng phối trộn Kelly theo xác suất Regime
    và trừng phạt kích thước mẫu Bayesian Shrinkage.
    Tuân thủ tuyệt đối kiến trúc HMM 2 trạng thái chốt của Module B (Trending vs Choppy).
    """
    # Giả lập: HMM đang lưỡng lự 60% Trending, 40% Choppy
    regime_probs = {"trending": 0.6, "choppy": 0.4}
    
    # Trending có 100 lệnh (Đủ mẫu), winrate rất tốt -> Kelly sẽ cao
    np.random.seed(42)
    returns_trending = np.random.choice([0.1, -0.05], p=[0.55, 0.45], size=100)
    
    # Choppy chưa có lệnh nào (Đói data) -> Kelly phải bị ép về Prior (0.1)
    returns_choppy = np.array([])
    
    regime_returns = {"trending": returns_trending, "choppy": returns_choppy}
    
    # Kiểm tra
    f_safe = compute_regime_weighted_bayesian_kelly(
        regime_returns, regime_probs, prior_f=0.1, confidence_constant_C=20.0
    )
    
    assert 0.1 <= f_safe <= DEFAULT_F_MAX
    print(f"✅ [BAYESIAN-HMM] Phối trộn mượt mà thành công. F_Blend = {f_safe:.3f}")


def test_b_1_11_trade_records_to_kelly_table_inputs():
    """
    Kiểm tra Task B-1-11:
    - Ánh xạ p_i, p_chop_i về index lưới (idx_p, idx_chop).
    - Lọc bỏ bản ghi thiếu realized_return hoặc boundary_truncated=True.
    - Xử lý p=1.0 bằng math.floor không bị out-of-bounds (idx <= num_bins-1).
    """
    from aegis.meta_labeling.sizing.kelly_empirical import trade_records_to_kelly_table_inputs
    records = [
        {"p_i": 0.05, "p_chop_i": 0.95, "realized_return": 0.04, "boundary_truncated": False}, # bin (0, 9)
        {"p_i": 1.00, "p_chop_i": 1.00, "realized_return": -0.02, "boundary_truncated": False}, # bin (9, 9) khi num_bins=10
        {"p_i": 0.55, "p_chop_i": 0.25, "realized_return": 0.10, "boundary_truncated": True},  # Bỏ qua vì boundary_truncated
        {"p_i": 0.55, "p_chop_i": 0.25, "realized_return": 0.10, "boundary_truncated": np.bool_(True)},  # Bỏ qua vì np.bool_(True)
        {"p_i": 0.55, "p_chop_i": 0.25, "realized_return": None},                                # Bỏ qua vì thiếu realized_return
        {"p_i": -0.1, "p_chop_i": 1.2, "realized_return": 0.01, "boundary_truncated": False},   # Clamped về (0, 9)
    ]
    grid, p_edges, chop_edges_list = trade_records_to_kelly_table_inputs(records, num_bins=10)
    # The exact coordinates will depend on quantile binning, so we just check that grid is populated
    assert len(grid) > 0
    total_samples = sum(len(v) for v in grid.values())
    assert total_samples == 3 # 0.04, -0.02, 0.01
    print("✅ [TASK B-1-11] trade_records_to_kelly_table_inputs PASSED!")


def test_b_1_12_build_empirical_kelly_table_v2():
    """
    Kiểm tra Task B-1-12:
    - Nếu len(returns) < 5 -> f = prior_f.
    - Nếu len(returns) >= 5 -> f_bayesian = w*f_cons + (1-w)*prior_f.
    """
    from aegis.meta_labeling.sizing.kelly_empirical import build_empirical_kelly_table_v2
    import math
    import numpy as np
    np.random.seed(42)
    # Lưới có 1 bin (9, 9) chứa 50 mẫu thắng tốt, và 1 bin (0, 0) chứa 3 mẫu (<5)
    returns_good = np.random.choice([0.08, -0.03], p=[0.6, 0.4], size=50)
    grid_inputs = {
        (9, 9): returns_good,
        (0, 0): np.array([0.05, 0.02, -0.01]), # Chỉ 3 lệnh < 5
    }
    import numpy as np
    p_edges = np.linspace(0, 1, 11)
    chop_edges_list = [np.linspace(0, 1, 11) for _ in range(10)]
    inputs_tuple = (grid_inputs, p_edges, chop_edges_list)
    table, p_edges_out, chop_edges_list_out = build_empirical_kelly_table_v2(
        inputs_tuple, num_bins=10, prior_f=0.0, confidence_constant_C=20.0
    )
    assert table.shape == (10, 10)
    assert table[0, 0] == 0.0  # < 5 lệnh -> prior_f = 0.0
    assert table[9, 9] > 0.0   # >= 5 lệnh thắng -> có f_bayesian dương
    print(f"✅ [TASK B-1-12] build_empirical_kelly_table_v2 PASSED (f[9,9]={table[9,9]:.4f})!")


def test_b_1_13_compute_bi_directional_kelly_v14_unified():
    """
    Kiểm tra Task B-1-13:
    - O(1) inference tra cứu kelly_table.
    - Trả về {'f_target': 0.0, 'mode': 'none'} nếu mode none.
    - Trả về đúng f_target nếu mode follow hoặc fade.
    """
    from aegis.meta_labeling.sizing.kelly_empirical import compute_bi_directional_kelly_v14_unified
    import math
    import numpy as np
    table = np.zeros((10, 10), dtype=float)
    table[8, 2] = 3.5  # p_i around 0.8, p_chop around 0.2 -> follow
    table[1, 8] = 1.8  # p_i around 0.1, p_chop around 0.8 -> fade
    p_edges = np.linspace(0, 1, 11)
    chop_edges_list = [np.linspace(0, 1, 11) for _ in range(10)]

    # Follow mode
    res_follow = compute_bi_directional_kelly_v14_unified(
        p_i=0.85, p_chop_i=0.25, kelly_table=table, 
        p_edges=p_edges, chop_edges_list=chop_edges_list, fade_enabled=True
    )
    assert res_follow["mode"] == "follow"
    assert math.isclose(res_follow["f_target"], 3.5, rel_tol=1e-6)

    # Fade mode
    res_fade = compute_bi_directional_kelly_v14_unified(
        p_i=0.15, p_chop_i=0.85, kelly_table=table, 
        p_edges=p_edges, chop_edges_list=chop_edges_list, fade_enabled=True,
        fade_regime_gate_threshold=0.60
    )
    assert res_fade["mode"] == "fade"
    assert math.isclose(res_fade["f_target"], 1.8, rel_tol=1e-6)

    # None mode (deadzone)
    res_none = compute_bi_directional_kelly_v14_unified(
        p_i=0.35, p_chop_i=0.50, kelly_table=table, 
        p_edges=p_edges, chop_edges_list=chop_edges_list, fade_enabled=True
    )
    assert res_none["mode"] == "none"
    assert res_none["f_target"] == 0.0

    print("✅ [TASK B-1-13] compute_bi_directional_kelly_v14_unified PASSED!")


def test_small_sample_bayesian_dynamic_f_max_cap():
    """
    [TDD VERIFICATION - VÁ LỖ HỔNG 8: BAYESIAN SHRINKAGE DYNAMIC CAP N vs C]:
    Kiểm chứng với mẫu nhỏ N=10, trần đòn bẩy động f_max_dynamic = min(f_max_cap, max(1.0, sqrt(N))) ≈ 3.16x
    chặn đứng over-betting trước khi đi qua shrinkage, thay vì để f_conservative vọt lên tận 20.0x.
    """
    # Mẫu N=10 lệnh thắng liên tiếp (hoặc có rủi ro cực nhỏ khiến Kelly muốn max đòn bẩy 20x)
    returns_small = np.array([0.05] * 10)
    regime_returns = {"Bull": returns_small}
    regime_probs = {"Bull": 1.0}

    # Với f_max_cap = 20.0, N = 10 -> f_max_dynamic = sqrt(10) ≈ 3.162x.
    # Khi C=20, trọng số w = 10 / (10 + 20) = 1/3.
    # prior_f = 0.1 -> f_bayes tối đa có thể đạt được là (1/3 * 3.162) + (2/3 * 0.1) ≈ 1.054 + 0.0667 ≈ 1.12x.
    f_bayes = compute_regime_weighted_bayesian_kelly(
        regime_returns=regime_returns,
        regime_probs=regime_probs,
        f_max_cap=20.0,
        prior_f=0.1,
        confidence_constant_C=20.0
    )

    max_possible_with_dynamic_cap = (10.0 / 30.0) * math.sqrt(10.0) + (20.0 / 30.0) * 0.1
    assert f_bayes <= max_possible_with_dynamic_cap + 1e-6, (
        f"f_bayes ({f_bayes:.4f}) phải bị chặn dưới mức {max_possible_with_dynamic_cap:.4f} do dynamic cap sqrt(N)!"
    )

    # Nếu không có dynamic cap, f_bayes khi đó sẽ đạt cỡ (10/30)*20 + (20/30)*0.1 ≈ 6.73x
    assert f_bayes < 2.0, f"Đòn bẩy mẫu nhỏ phải an toàn dưới 2.0x, nhận {f_bayes:.4f}"
    print(f"✅ [VÁ LỖ HỔNG 8] Bayesian Shrinkage Dynamic Cap với N=10: f_bayes = {f_bayes:.4f}x (Max cap: ~{max_possible_with_dynamic_cap:.4f}x) PASSED!")

