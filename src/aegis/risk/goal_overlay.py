"""
Tầng phủ ĐÒN BẨY THEO MỤC TIÊU — đưa chính sách DP từ nghiên cứu vào đường chạy thật.

Chính sách tối ưu của `research/goal_dp.py` chỉ có giá trị nếu hệ thống thật chạy được
nó. Module này là toàn bộ cầu nối, và nó cố tình rất mỏng: nó KHÔNG tính lại gì cả, chỉ
tra bảng đã giải sẵn rồi trả về MỘT con số — hệ số nhân áp lên đòn bẩy gộp.

VÌ SAO CHỈ CO GIÃN GROSS CHỨ KHÔNG ĐỤNG THÀNH PHẦN DANH MỤC:
`xs_live_pipeline` và `research/strategy_v3` bắt buộc phải sinh ra trọng số giống hệt
nhau tới 1e-9 (`tests/pipelines/test_research_live_parity_v3.py`). Mọi thứ đụng vào
việc CHỌN cặp hay TỶ TRỌNG giữa các cặp đều phá bất biến đó và tái lập lỗi F3. Nhân
toàn bộ sổ với một vô hướng thì không: thành phần không đổi, chỉ độ lớn đổi. Đó là lý
do tầng này là một HỆ SỐ NHÂN chứ không phải một bộ dựng danh mục thứ hai.

AN TOÀN MẶC ĐỊNH — `derisk_only=True`:
DP khuyên tăng đòn bẩy khi đang thua và sắp hết giờ ("bạo phát"). Về toán học điều đó
đúng cho mục tiêu đã nêu. Về vận hành, nó nguy hiểm nhất đúng vào lúc hệ thống đang ở
trạng thái xấu nhất — và tính tới hôm nay cổng chất lượng mới đạt 1/3 lượt sạch, tức
đường ống CHƯA chứng minh là không vỡ. Vì vậy mặc định tầng này chỉ được phép GIẢM
rủi ro, không được phép tăng. Phần "bạo phát" phải bật tường minh, sau khi cổng mở.

Bất đối xứng đó không phải sự thận trọng mơ hồ: giảm đòn bẩy không thể gây ra sự cố
thực thi mới, còn tăng đòn bẩy thì có thể — và F21-F30 đều sinh ra từ lúc hệ thống cố
làm nhiều hơn chứ không phải ít hơn.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from aegis.research.goal_dp import GoalPolicy, GoalSpec

logger = logging.getLogger(__name__)

__all__ = ["GoalOverlayConfig", "GoalOverlay", "save_policy", "load_policy"]

DEFAULT_POLICY_PATH = "artifacts/goal_policy.npz"


# ---------------------------------------------------------------------------
# Lưu / nạp chính sách đã giải
# ---------------------------------------------------------------------------
def save_policy(policy: GoalPolicy, path: str = DEFAULT_POLICY_PATH) -> str:
    """
    Ghi chính sách ra `.npz` kèm `.meta.json`.

    Giải DP mất vài giây — không nhiều, nhưng đường chạy live tuyệt đối không nên tính
    lại một thứ có thể thay đổi. Nếu live tự giải mỗi lượt thì chính sách sẽ trôi theo
    dữ liệu mới mà không ai duyệt, và ta mất khả năng nói "hôm đó hệ thống dùng chính
    sách nào". Giải một lần, ghi ra đĩa, ký mốc thời gian.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, policy=policy.policy, value=policy.value,
                        wealth_grid=policy.wealth_grid, leverage_grid=policy.leverage_grid)
    meta = {"spec": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                     for k, v in policy.spec.__dict__.items()},
            "meta": policy.meta,
            "shape": {"policy": list(policy.policy.shape), "value": list(policy.value.shape)}}
    p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return str(p)


def load_policy(path: str = DEFAULT_POLICY_PATH) -> GoalPolicy:
    """Nạp chính sách + spec. Thiếu file `.meta.json` là lỗi, KHÔNG đoán spec mặc định."""
    p = pathlib.Path(path)
    mp = p.with_suffix(".meta.json")
    if not p.is_file() or not mp.is_file():
        raise FileNotFoundError(
            f"Thiếu {p} hoặc {mp}. Chạy `python scripts/build_goal_policy.py` trước.")
    z = np.load(p)
    spec_d = json.loads(mp.read_text())["spec"]
    spec = GoalSpec(**{k: v for k, v in spec_d.items()
                       if k in GoalSpec.__dataclass_fields__})
    return GoalPolicy(spec=spec, policy=z["policy"], value=z["value"],
                      wealth_grid=z["wealth_grid"], leverage_grid=z["leverage_grid"],
                      meta=json.loads(mp.read_text()).get("meta", {}))


# ---------------------------------------------------------------------------
# Cấu hình & tầng phủ
# ---------------------------------------------------------------------------
@dataclass
class GoalOverlayConfig:
    """Tham số vận hành của tầng phủ."""

    enabled: bool = False
    policy_path: str = DEFAULT_POLICY_PATH
    #: Vốn và hạn chót của "ván" hiện tại. `start_equity <= 0` nghĩa là chưa mở ván.
    start_equity: float = 0.0
    deadline_ms: int = 0
    #: CHỈ được phép giảm rủi ro. Xem docstring module.
    derisk_only: bool = True
    #: Trần hệ số nhân khi đã cho phép tăng — chặn cứng độc lập với DP.
    max_multiplier: float = 2.0
    #: Đạt đích rồi thì đóng sạch. Giữ riêng khỏi DP vì đây là luật ta MUỐN bảo đảm,
    #: không phải luật ta hy vọng DP suy ra (dù nó có suy ra thật).
    lock_in_at_target: bool = True
    #: Đệm trên đích trước khi chốt — tránh chốt rồi lại rơi xuống vì phí đóng vị thế.
    lock_in_buffer: float = 0.002


@dataclass
class GoalOverlay:
    """Tra chính sách DP -> hệ số nhân đòn bẩy. Không trạng thái, không tác dụng phụ."""

    config: GoalOverlayConfig
    policy: Optional[GoalPolicy] = None

    @classmethod
    def from_config(cls, config: GoalOverlayConfig) -> "GoalOverlay":
        pol = load_policy(config.policy_path) if config.enabled else None
        return cls(config=config, policy=pol)

    # ------------------------------------------------------------------
    def periods_left(self, now_ms: int, period_hours: float) -> int:
        """Số kỳ quyết định còn lại tới hạn chót, làm tròn XUỐNG (bi quan)."""
        if self.config.deadline_ms <= 0:
            return 0
        ms_left = max(0, self.config.deadline_ms - now_ms)
        return int(ms_left // int(period_hours * 3_600_000))

    def state(self, equity: float, now_ms: int, period_hours: float) -> Dict[str, float]:
        """Trạng thái hiện tại dưới dạng DP hiểu được — hữu ích để ghi nhật ký."""
        w0 = self.config.start_equity
        return {
            "wealth_ratio": float(equity / w0) if w0 > 0 else 1.0,
            "periods_left": float(self.periods_left(now_ms, period_hours)),
        }

    def multiplier(self, equity: float, now_ms: int, base_leverage: float,
                   current_leverage: float, period_hours: float) -> Dict[str, object]:
        """
        Hệ số nhân nên áp lên `base_leverage`, kèm lý do — luôn trả về lý do.

        Trả về dict chứ không trả về một float trần trụi là có chủ ý: một con số không
        giải thích được tại sao lại là nó thì không kiểm toán được sau sự cố, và mọi sự
        cố của hệ thống này tới nay đều được mổ xẻ từ nhật ký.
        """
        cfg = self.config
        out: Dict[str, object] = {"multiplier": 1.0, "reason": "overlay tắt",
                                  "wealth_ratio": None, "periods_left": None,
                                  "dp_leverage": None, "p_success": None}
        if not cfg.enabled or self.policy is None:
            return out
        if cfg.start_equity <= 0 or cfg.deadline_ms <= 0:
            out["reason"] = "chưa mở ván (thiếu start_equity/deadline)"
            return out

        wr = float(equity / cfg.start_equity)
        left = self.periods_left(now_ms, period_hours)
        out.update(wealth_ratio=wr, periods_left=left)

        target = 1.0 + self.policy.spec.target_return

        # --- 1. Đã đạt đích -> đóng sạch. Luật này đứng TRƯỚC DP và độc lập với nó.
        if cfg.lock_in_at_target and wr >= target + cfg.lock_in_buffer:
            out.update(multiplier=0.0,
                       reason=f"ĐẠT ĐÍCH: vốn {wr:.4f} >= {target:.4f} — đóng sạch, "
                              f"giữ lãi. Rủi ro thêm chỉ có thể làm hỏng.")
            return out

        # --- 2. Hết giờ -> không còn quyết định nào để ra.
        if left <= 0:
            out.update(multiplier=0.0, reason="hết hạn chót")
            return out

        # --- 3. Tra chính sách DP.
        dp_lev = self.policy.leverage_for(wr, left, current_leverage)
        p = self.policy.p_success(wr, left, current_leverage)
        out.update(dp_leverage=dp_lev, p_success=p)

        mult = dp_lev / base_leverage if base_leverage > 1e-9 else 0.0
        if cfg.derisk_only and mult > 1.0:
            out.update(multiplier=1.0,
                       reason=f"DP muốn {dp_lev:.2f}x (nhân {mult:.2f}) nhưng derisk_only "
                              f"đang bật — giữ nguyên {base_leverage:.2f}x")
            return out

        mult = float(np.clip(mult, 0.0, cfg.max_multiplier))
        out.update(multiplier=mult,
                   reason=f"DP: vốn {wr:.4f}, còn {left} kỳ -> {dp_lev:.2f}x "
                          f"(nhân {mult:.2f}), P(đạt đích)={p:.1%}")
        return out
