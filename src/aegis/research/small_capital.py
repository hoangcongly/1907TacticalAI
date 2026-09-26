"""
Vốn nhỏ: mô phỏng ĐÚNG thứ live làm với vị thế dưới min notional.

[FIX F54] BACKTEST GIẢ ĐỊNH MỌI VỊ THẾ ĐỀU ĐẶT ĐƯỢC. LIVE THÌ KHÔNG.

`build_rebalance_plan` (execution/portfolio_rebalancer.py) quy mọi vị thế MỤC TIÊU có
notional < min_notional của sàn về 0 — bỏ hẳn, KHÔNG co giãn phần còn lại. Còn mọi
backtest trong repo (`run_v3`, `run_v3_fine`, `run_v3_tranched`) giao dịch nguyên văn
trọng số của `portfolio.py`, kể cả những vị thế nhỏ tới vài cent.

Ở testnet $5.000 thì hai bên trùng nhau: vị thế nhỏ nhất vẫn cỡ hàng chục đô. Ở vốn
1 triệu VND (~$38) thì không. V3 dựng trọng số `zscore_riskparity` với trần 0,20 — các
vị thế chênh nhau nhiều lần — nên ở 12 vị thế, 2x, ngân sách gộp $76: vị thế nào dưới
5/76 = 6,6% gross (trung bình là 8,3%) đều bị bỏ. Phần bị bỏ KHÔNG chia đều hai chân,
nên sổ vừa nhỏ hơn thiết kế vừa lệch khỏi trung lập — và cổng F25 sau thực thi báo
NEUTRALITY_BREACH. Con số backtest cho "12 vị thế ở $38" đo một danh mục live không
dựng được.

`max_positions_for_capital` / F43 không cứu được: chúng chỉ đếm SỐ vị thế theo mức
TRUNG BÌNH ($6/vị thế), không nhìn từng trọng số.

Hàm ở đây là bản sao phép lọc của live, khoá bằng test đối chiếu trực tiếp với
`build_rebalance_plan` (`tests/research/test_small_capital_f54.py`). Vốn được coi là
CỐ ĐỊNH ở mức ban đầu — đúng câu hỏi "vốn cố định 1 triệu"; sụt vốn làm lọc gắt hơn
và F43 cắt bớt vị thế, nên đây là ước lượng LẠC QUAN của phần bị bỏ.
"""
from typing import Dict

import numpy as np
import pandas as pd

__all__ = ["drop_below_min_notional", "min_leverage", "drop_stats", "equity_with_breakers",
           "block_bootstrap_paths", "evaluate_paths", "TIER1_DD", "TIER3_DD"]

#: Cùng ngưỡng với `xs_live_pipeline` (TIER1 cắt nửa vị thế, TIER3 kill switch).
TIER1_DD, TIER3_DD = 0.10, 0.30


def drop_below_min_notional(
    weights: pd.DataFrame,
    capital_usd: float,
    leverage: float,
    min_notional_usd: float = 5.0,
) -> pd.DataFrame:
    """
    Quy về 0 mọi vị thế có notional mục tiêu < `min_notional_usd` — y hệt live.

    Live chuẩn hoá mỗi hàng về tổng |w| = 1 rồi nhân ngân sách `vốn × đòn bẩy`
    (`build_rebalance_plan`: `norm = w / total_abs`, `target = norm × gross_budget`),
    nên ngưỡng áp lên trọng số ĐÃ CHUẨN HOÁ. Phần còn lại giữ NGUYÊN giá trị — live
    không co giãn lại sau khi bỏ, nên gross co lại và net có thể lệch 0.
    """
    if capital_usd <= 0 or leverage <= 0:
        raise ValueError("capital_usd và leverage phải > 0")
    budget = capital_usd * leverage
    gross = weights.abs().sum(axis=1)
    norm = weights.div(gross.where(gross > 0), axis=0)
    small = (norm.abs() * budget < min_notional_usd) & weights.notna()
    return weights.mask(small, 0.0)


def min_leverage(n_positions: int, capital_usd: float, min_order_usd: float = 6.0) -> float:
    """Đòn bẩy tối thiểu để n vị thế ĐỀU nhau mỗi cái đạt `min_order_usd` (= $5 × an toàn 1,2)."""
    return float(min_order_usd * n_positions / capital_usd)


def drop_stats(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, float]:
    """Phần bị bỏ, trung bình trên các mốc có vị thế: gross còn lại, |net|/gross, số vị thế."""
    g0 = before.abs().sum(axis=1)
    live = g0 > 0
    if not live.any():
        return {"gross_kept": np.nan, "net_over_gross": np.nan, "positions_kept": np.nan}
    g1 = after.abs().sum(axis=1)[live]
    net = after.sum(axis=1)[live]
    n0 = (before[live].fillna(0) != 0).sum(axis=1)
    n1 = (after[live].fillna(0) != 0).sum(axis=1)
    return {"gross_kept": float((g1 / g0[live]).mean()),
            "net_over_gross": float((net.abs() / g1.where(g1 > 0)).mean()),
            "positions_kept": float((n1 / n0.where(n0 > 0)).mean())}


def equity_with_breakers(paths: np.ndarray, leverage: float, rebalance_every: int = 18,
                         tier1: float = TIER1_DD, tier3: float = TIER3_DD) -> Dict[str, np.ndarray]:
    """
    Đường vốn ở đòn bẩy `leverage` KÈM ngắt mạch như live, trên lợi suất 1x `paths`
    (n_đường, n_nến).

    Live chấm ngắt mạch trên SỤT GIẢM EQUITY, không quy theo đòn bẩy (CLAUDE.md, replay
    24/09, mục 3). Ở vốn nhỏ và đòn bẩy cao, đó mới là thứ quyết định vốn cuối cùng —
    không phải Sharpe. Mô phỏng ở đây:
      * TIER1 (sụt >= 10%): mỗi mốc tái cân bằng, hệ số vị thế = 0,5, về 1,0 khi hồi.
      * TIER3 (sụt >= 30%): kill switch — HẤP THỤ, vốn đứng yên từ nến đó.
      * TIER2 (đóng băng 24h) bỏ qua: vị thế vẫn giữ, chỉ lỡ tái cân bằng.

    ⚠️ Kill ở live (`_check_circuit_breaker`) chỉ đánh dấu `is_dead` và NGỪNG giao dịch,
    KHÔNG đóng vị thế — sổ cũ vẫn nằm trên sàn cho tới khi người vận hành `/kill`. Coi
    vốn đứng yên từ lúc kill là giả định LẠC QUAN (người vận hành đóng tay ngay).

    Trả về `final`, `killed` (bool), `tier1_share` (tỷ lệ mốc chạy nửa vị thế).
    """
    n_paths, n_bars = paths.shape
    eq = np.ones(n_paths)
    peak = np.ones(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    mult = np.ones(n_paths)
    halved = np.zeros(n_paths)
    marks = 0
    for t in range(n_bars):
        if t % rebalance_every == 0:
            mult = np.where(1.0 - eq / peak >= tier1, 0.5, 1.0)
            halved += (mult < 1.0) & alive
            marks += 1
        step = np.where(alive, leverage * mult * paths[:, t], 0.0)
        eq = eq * np.maximum(1.0 + step, 0.0)
        peak = np.maximum(peak, eq)
        alive &= eq / peak > 1.0 - tier3
    return {"final": eq, "killed": ~alive, "tier1_share": halved / max(marks, 1)}


def block_bootstrap_paths(r: np.ndarray, n_bars: int, n_paths: int, seed: int,
                          block: int = 18) -> np.ndarray:
    """Đường lợi suất (n_paths, n_bars) lấy mẫu theo KHỐI `block` nến — giữ cụm biến động."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n_bars / block))
    starts = rng.integers(0, len(r) - block, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)
    return r[idx[:, :n_bars]]


def evaluate_paths(paths: np.ndarray, lev: float, rebalance_every: int = 18) -> Dict[str, float]:
    """Hai phép chấm trên CÙNG các đường: không ngắt mạch (rủi ro gốc) và có ngắt mạch như live."""
    eq = np.cumprod(np.maximum(1.0 + lev * paths, 0.0), axis=1)
    dead = (eq <= 0.10).any(axis=1)                # cháy là HẤP THỤ, không bị lọc bỏ
    live = equity_with_breakers(paths, lev, rebalance_every=rebalance_every)
    fin = live["final"]
    return {"median_raw": float(np.median(np.where(dead, 0.0, eq[:, -1]))),
            "p_ruin": float(dead.mean()),
            "median_live": float(np.median(fin)),
            "p10_live": float(np.percentile(fin, 10)),
            "p_kill": float(live["killed"].mean()),
            "p_loss_live": float((fin < 1.0).mean()),
            "tier1_share": float(live["tier1_share"].mean())}
