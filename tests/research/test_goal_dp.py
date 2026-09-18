"""
Khoá các BẤT BIẾN của bài toán đạt mục tiêu trước hạn chót.

Các test ở đây không kiểm tra "DP cho ra con số đẹp" — con số phụ thuộc dữ liệu và sẽ
đổi. Chúng khoá những thứ PHẢI đúng bất kể dữ liệu nào, và mỗi test tương ứng với một
cách cụ thể mà module này từng sai hoặc có thể sai im lặng.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.research.goal_dp import (
    GoalSpec, constant_leverage_ceiling, dynamic_leverage_ceiling,
    evaluate_policy, solve_goal_dp,
)


@pytest.fixture(scope="module")
def returns():
    """Chuỗi lợi suất tổng hợp có edge dương và đuôi dày — đủ giống thật để test."""
    rng = np.random.default_rng(0)
    r = rng.standard_t(df=4, size=4000) * 0.009 + 0.00035
    return r


# ---------------------------------------------------------------------------
# Mốc chặn dạng đóng
# ---------------------------------------------------------------------------
def test_tran_dong_nam_trong_khoang_xac_suat():
    c = constant_leverage_ceiling(1.1, 0.55, 7 / 365, 0.05)
    d = dynamic_leverage_ceiling(1.1, 7 / 365, 0.05)
    assert 0.0 <= c["p_max_constant"] <= 1.0
    assert 0.0 <= d["p_max_dynamic"] <= 1.0


def test_tran_dong_tang_theo_sharpe():
    lo = constant_leverage_ceiling(0.5, 0.55, 7 / 365, 0.05)["p_max_constant"]
    hi = constant_leverage_ceiling(2.0, 0.55, 7 / 365, 0.05)["p_max_constant"]
    assert hi > lo


def test_gia_cua_muc_tieu_khong_phu_thuoc_chien_luoc():
    """
    `sqrt(2*ln(1+g))` chỉ phụ thuộc mục tiêu. Đây là điểm cốt lõi của công thức: cái
    giá của việc đặt ra một cái đích bị trừ thẳng vào z-score, và không chiến lược nào
    làm nó nhỏ đi. Nếu ai đó "cải tiến" công thức làm số hạng này phụ thuộc Sharpe thì
    họ đã phá mất thông điệp duy nhất đáng giá của nó.
    """
    a = constant_leverage_ceiling(0.3, 0.55, 7 / 365, 0.05)["goal_cost_term"]
    b = constant_leverage_ceiling(3.0, 0.20, 7 / 365, 0.05)["goal_cost_term"]
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(np.sqrt(2 * np.log(1.05)), rel=1e-12)


def test_dieu_khien_dong_khong_bao_gio_te_hon_don_bay_co_dinh():
    for S in (0.0, 0.5, 1.5, 3.0):
        c = constant_leverage_ceiling(S, 0.55, 7 / 365, 0.05)["p_max_constant"]
        d = dynamic_leverage_ceiling(S, 7 / 365, 0.05)["p_max_dynamic"]
        assert d >= c - 1e-12


# ---------------------------------------------------------------------------
# Cấu trúc chính sách
# ---------------------------------------------------------------------------
def test_tu_tim_ra_luat_chot_loi(returns):
    """
    Trên mức đích, đòn bẩy tối ưu phải là 0 ở MỌI thời điểm còn lại.

    Luật này KHÔNG được viết tay ở đâu trong `solve_goal_dp`. Nó phải rơi ra từ quy nạp
    lùi. Test này vì vậy là phép kiểm chứng tốt nhất rằng DP đang giải đúng bài toán:
    nếu hàm mục tiêu bị viết sai thành "tối đa vốn cuối kỳ" thay vì "tối đa xác suất
    chạm đích" thì chính sách ở đây sẽ KHÔNG bằng 0.
    """
    spec = GoalSpec(target_return=0.05, horizon_periods=20, n_wealth=201)
    pol = solve_goal_dp(returns, spec)
    for left in (1, 5, 10, 20):
        assert pol.leverage_for(1.08, left, 2.0) == 0.0


def test_xac_suat_tang_theo_von_va_theo_thoi_gian(returns):
    spec = GoalSpec(target_return=0.05, horizon_periods=20, n_wealth=201)
    pol = solve_goal_dp(returns, spec)
    for left in (5, 20):
        ps = [pol.p_success(w, left, 0.0) for w in (0.90, 0.97, 1.00, 1.03)]
        assert ps == sorted(ps), "vốn cao hơn phải cho xác suất cao hơn"
    for w in (0.95, 1.0, 1.03):
        ps = [pol.p_success(w, k, 0.0) for k in (2, 6, 12, 20)]
        assert ps == sorted(ps), "còn nhiều thời gian hơn phải cho xác suất cao hơn"


def test_truc_thoi_gian_khong_bi_lat_nguoc(returns):
    """
    Hồi quy cho lỗi ĐÃ XẢY RA: `policy` đánh chỉ số theo "số kỳ còn lại - 1" còn
    `value` theo "số kỳ còn lại", và hàm tra cứu dùng nhầm `horizon - k`. Bảng in ra
    vẫn TRÔNG hợp lý (liều khi sắp hết giờ) nên suýt lọt.

    Neo bằng một bất biến không đối xứng theo thời gian: ở ngay dưới đích, còn NHIỀU
    thời gian thì không cần liều; còn ÍT thời gian thì buộc phải liều hơn.
    """
    spec = GoalSpec(target_return=0.05, horizon_periods=30, n_wealth=241)
    pol = solve_goal_dp(returns, spec)
    som, muon = pol.leverage_for(1.045, 30, 2.0), pol.leverage_for(1.045, 2, 2.0)
    assert muon > som, f"sắp hết giờ phải liều hơn: còn 2 kỳ {muon} <= còn 30 kỳ {som}"


# ---------------------------------------------------------------------------
# Ràng buộc đặt được lệnh
# ---------------------------------------------------------------------------
def test_luoi_don_bay_bo_qua_vung_khong_dat_duoc_lenh():
    spec = GoalSpec(max_leverage=5.0, min_executable_leverage=1.58)
    g = spec.leverage_grid()
    assert g[0] == 0.0, "đứng ngoài thị trường luôn là hành động hợp lệ"
    assert not ((g > 1e-9) & (g < 1.58 - 1e-9)).any(), \
        "không được có mức đòn bẩy dương nào dưới sàn đặt lệnh"


def test_san_dat_lenh_tinh_dung():
    # $38, 12 vị thế, min notional $5 -> 12*5/38 = 1.578...
    assert GoalSpec.executable_floor(38.0, 12, 5.0) == pytest.approx(12 * 5 / 38.0)


# ---------------------------------------------------------------------------
# Đánh giá
# ---------------------------------------------------------------------------
def test_danh_sach_don_bay_rong_thi_chi_cham_dp(returns):
    """
    Hồi quy cho lỗi ĐÃ XẢY RA: `constant_leverages or [mặc định]` — list RỖNG là falsy
    trong Python nên nó âm thầm rơi về mặc định đúng lúc người gọi bảo "đừng chấm mức
    cố định nào". Hệ quả: bảng đánh đổi in ra cùng một con số ở mọi dòng.
    """
    spec = GoalSpec(target_return=0.05, horizon_periods=12, n_wealth=161)
    pol = solve_goal_dp(returns, spec)
    df = evaluate_policy(returns, pol, spec, constant_leverages=[], n_paths=2000, block=4)
    assert len(df) == 1 and df.iloc[0]["policy"] == "DP mục tiêu"


def test_dp_khong_te_hon_don_bay_co_dinh_tot_nhat(returns):
    """DP tối ưu theo đúng đại lượng này, nên nó không được thua — dù chấm ngoài mẫu."""
    spec = GoalSpec(target_return=0.05, horizon_periods=24, n_wealth=201)
    pol = solve_goal_dp(returns[:2000], spec)
    df = evaluate_policy(returns[2000:], pol, spec,
                         constant_leverages=[1, 2, 3, 4, 5], n_paths=6000, block=5)
    dp = df[df["policy"] == "DP mục tiêu"]["p_target"].iloc[0]
    best_const = df[df["policy"] != "DP mục tiêu"]["p_target"].max()
    assert dp >= best_const


def test_frac_flat_va_don_bay_khi_vao_nhat_quan(returns):
    """`avg_leverage` là trung bình trộn 0x với các mức dương — dễ đọc nhầm thành trạng thái."""
    spec = GoalSpec(target_return=0.05, horizon_periods=16, n_wealth=161,
                    min_executable_leverage=1.58)
    pol = solve_goal_dp(returns, spec)
    df = evaluate_policy(returns, pol, spec, constant_leverages=[2.0],
                         n_paths=3000, block=4)
    dp = df.iloc[-1]
    assert 0.0 <= dp["frac_flat"] <= 1.0
    if dp["frac_flat"] < 1.0:
        assert dp["avg_leverage_on"] >= 1.58 - 1e-9
        assert dp["avg_leverage"] <= dp["avg_leverage_on"] + 1e-9
