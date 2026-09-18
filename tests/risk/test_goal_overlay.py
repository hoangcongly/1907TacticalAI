"""
Khoá hành vi AN TOÀN của tầng phủ đòn bẩy theo mục tiêu.

Tầng này nằm trên đường tiền và bật/tắt bằng cấu hình, nên phần đáng test nhất không
phải "nó có tối ưu không" mà là "nó có làm điều bất ngờ không khi cấu hình sai".
"""
import numpy as np
import pytest

from aegis.research.goal_dp import GoalSpec, solve_goal_dp
from aegis.risk.goal_overlay import (
    GoalOverlay, GoalOverlayConfig, load_policy, save_policy,
)

HOUR_MS = 3_600_000


@pytest.fixture(scope="module")
def policy():
    rng = np.random.default_rng(3)
    r = rng.standard_t(df=4, size=3000) * 0.009 + 0.00035
    return solve_goal_dp(r, GoalSpec(target_return=0.05, horizon_periods=42,
                                     n_wealth=201, min_executable_leverage=1.58))


@pytest.fixture
def overlay(policy, tmp_path):
    path = str(tmp_path / "goal_policy.npz")
    save_policy(policy, path)
    cfg = GoalOverlayConfig(enabled=True, policy_path=path, start_equity=1000.0,
                            deadline_ms=7 * 24 * HOUR_MS)
    return GoalOverlay.from_config(cfg)


# ---------------------------------------------------------------------------
def test_luu_va_nap_giu_nguyen_chinh_sach(policy, tmp_path):
    path = str(tmp_path / "p.npz")
    save_policy(policy, path)
    back = load_policy(path)
    assert np.array_equal(back.policy, policy.policy)
    assert np.allclose(back.value, policy.value)
    assert back.spec.target_return == policy.spec.target_return
    assert back.spec.horizon_periods == policy.spec.horizon_periods


def test_thieu_artifact_thi_nem_loi_chu_khong_doan_mac_dinh(tmp_path):
    """Nạp mặc định âm thầm nghĩa là chạy một chính sách không ai duyệt."""
    cfg = GoalOverlayConfig(enabled=True, policy_path=str(tmp_path / "khong_co.npz"))
    with pytest.raises(FileNotFoundError):
        GoalOverlay.from_config(cfg)


def test_tat_thi_khong_lam_gi(policy, tmp_path):
    ov = GoalOverlay.from_config(GoalOverlayConfig(enabled=False))
    out = ov.multiplier(equity=900.0, now_ms=0, base_leverage=2.0,
                        current_leverage=2.0, period_hours=4.0)
    assert out["multiplier"] == 1.0
    assert ov.policy is None


def test_chua_mo_van_thi_khong_lam_gi(policy, tmp_path):
    path = str(tmp_path / "p.npz"); save_policy(policy, path)
    ov = GoalOverlay.from_config(GoalOverlayConfig(enabled=True, policy_path=path))
    out = ov.multiplier(equity=1000.0, now_ms=0, base_leverage=2.0,
                        current_leverage=2.0, period_hours=4.0)
    assert out["multiplier"] == 1.0 and "chưa mở ván" in out["reason"]


def test_dat_dich_thi_dong_sach(overlay):
    """+5% rồi thì mọi rủi ro tiếp theo chỉ có thể làm hỏng. Hệ số phải về 0."""
    out = overlay.multiplier(equity=1060.0, now_ms=0, base_leverage=2.0,
                             current_leverage=2.0, period_hours=4.0)
    assert out["multiplier"] == 0.0
    assert "ĐẠT ĐÍCH" in out["reason"]


def test_dem_chot_loi_chan_viec_chot_hut(overlay):
    """Ngay sát đích nhưng chưa qua đệm thì CHƯA chốt — phí đóng vị thế có thể kéo tụt lại."""
    just_under = 1000.0 * (1.0 + 0.05 + overlay.config.lock_in_buffer / 2)
    out = overlay.multiplier(equity=just_under, now_ms=0, base_leverage=2.0,
                             current_leverage=2.0, period_hours=4.0)
    assert "ĐẠT ĐÍCH" not in out["reason"]


def test_het_han_thi_dong_sach(overlay):
    out = overlay.multiplier(equity=980.0, now_ms=8 * 24 * HOUR_MS, base_leverage=2.0,
                             current_leverage=2.0, period_hours=4.0)
    assert out["multiplier"] == 0.0 and "hết hạn" in out["reason"]


def test_derisk_only_chan_moi_lenh_tang_don_bay(overlay):
    """
    Bất biến AN TOÀN quan trọng nhất của module: khi `derisk_only` bật, tầng phủ không
    bao giờ trả về hệ số > 1. DP khuyên liều khi đang thua và sắp hết giờ — đúng về
    toán, nhưng đó chính là lúc đường ống đang ở trạng thái xấu nhất.
    """
    assert overlay.config.derisk_only is True
    for equity in (700.0, 850.0, 950.0, 1000.0, 1040.0):
        for hours_left in (2, 24, 100, 160):
            now = (7 * 24 - hours_left) * HOUR_MS
            out = overlay.multiplier(equity=equity, now_ms=now, base_leverage=2.0,
                                     current_leverage=2.0, period_hours=4.0)
            assert out["multiplier"] <= 1.0 + 1e-12, \
                f"derisk_only bị thủng ở equity={equity}, còn {hours_left}h"


def test_tat_derisk_only_thi_van_bi_chan_boi_max_multiplier(policy, tmp_path):
    path = str(tmp_path / "p.npz"); save_policy(policy, path)
    ov = GoalOverlay.from_config(GoalOverlayConfig(
        enabled=True, policy_path=path, start_equity=1000.0,
        deadline_ms=7 * 24 * HOUR_MS, derisk_only=False, max_multiplier=1.5))
    seen_above_one = False
    for equity in (800.0, 900.0, 980.0):
        for hours_left in (8, 48, 150):
            out = ov.multiplier(equity=equity, now_ms=(7 * 24 - hours_left) * HOUR_MS,
                                base_leverage=2.0, current_leverage=2.0, period_hours=4.0)
            assert out["multiplier"] <= 1.5 + 1e-12
            seen_above_one |= out["multiplier"] > 1.0
    assert seen_above_one, "tắt derisk_only mà không bao giờ tăng thì cờ này vô nghĩa"


def test_so_ky_con_lai_lam_tron_xuong(overlay):
    """Làm tròn LÊN sẽ cho DP tưởng còn nhiều thời gian hơn thực tế — sai về phía nguy hiểm."""
    assert overlay.periods_left(now_ms=0, period_hours=4.0) == 42
    # còn 7h59p ở nhịp 4h = 1 kỳ trọn vẹn, không phải 2
    almost = 7 * 24 * HOUR_MS - (8 * HOUR_MS - 60_000)
    assert overlay.periods_left(now_ms=almost, period_hours=4.0) == 1


def test_luon_kem_theo_ly_do(overlay):
    for equity in (700.0, 1000.0, 1100.0):
        out = overlay.multiplier(equity=equity, now_ms=0, base_leverage=2.0,
                                 current_leverage=2.0, period_hours=4.0)
        assert isinstance(out["reason"], str) and out["reason"]
