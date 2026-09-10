"""
Chiến lược v2 — MỘT đường đi duy nhất từ dữ liệu tới trọng số vị thế.

VÌ SAO MODULE NÀY TỒN TẠI: lỗi F3 của hệ thống cũ (research tự tính feature một
kiểu, live tính một kiểu khác, thiếu thì âm thầm điền 0.0) không phải lỗi lập trình
mà là lỗi KIẾN TRÚC — hai đường đi tính cùng một thứ thì sớm muộn cũng lệch nhau.
Cách sửa duy nhất bền vững là chỉ có MỘT đường.

`StrategyV2.target_weights(panel, funding)` là đường đó. Backtest gọi nó trên toàn
bộ lịch sử; live gọi đúng nó trên cửa sổ dữ liệu gần nhất và lấy hàng cuối. Không
có nhánh `if live:` nào bên trong.

Bốn tầng, mỗi tầng một trách nhiệm:

    dữ liệu -> [1] thư viện tín hiệu   -> điểm số từng họ
            -> [2] gộp thích ứng       -> một điểm số tổng hợp
            -> [3] dựng danh mục       -> trọng số dollar/beta-neutral
            -> [4] đòn bẩy             -> trọng số cuối theo mục tiêu biến động

Tách bạch như vậy để mỗi tầng đo được RIÊNG (xem `scripts/lab.py`) — khi Sharpe
tụt, ta biết tầng nào gây ra.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.backtest_v2 import CostModel, estimate_cost_bps, simulate
from aegis.research.signal_library import FAMILIES, build_all_families
from aegis.research.leverage import LeverageSpec, apply_leverage
from aegis.risk.portfolio import PortfolioSpec, build_weights

__all__ = ["StrategyV2Config", "StrategyV2"]


@dataclass
class StrategyV2Config:
    """
    Toàn bộ tham số chiến lược — khai báo MỘT chỗ.

    Mọi con số ở đây phải đến từ một quyết định trên tập train và không được sửa
    sau khi đã chạm holdout. Ghi kèm `provenance` để sau này còn truy được vì sao.
    """

    universe_file: str = "artifacts/universe_filtered.json"
    interval: str = "4h"
    source_interval: str = "1h"
    rebalance_every: int = 6           # số nến giữa hai lần tái cân bằng
    families: List[str] = field(default_factory=lambda: list(FAMILIES))

    portfolio: PortfolioSpec = field(default_factory=lambda: PortfolioSpec(
        mode="zscore_riskparity", top_frac=0.10, beta_neutral=True, max_weight=0.20))
    combiner: CombinerSpec = field(default_factory=lambda: CombinerSpec(
        lookback=500, min_periods=90, t_threshold=1.5, max_step=0.10))
    leverage: LeverageSpec = field(default_factory=LeverageSpec)

    maker_ratio: float = 0.5
    notional_per_order: float = 20.0
    min_coverage: float = 0.15

    provenance: Dict = field(default_factory=dict)

    def bar_hours(self) -> float:
        from aegis.data.panel_v2 import interval_hours
        return interval_hours(self.interval)

    def period_hours(self) -> float:
        return self.bar_hours() * self.rebalance_every

    def periods_per_year(self) -> float:
        return 24 * 365.0 / self.period_hours()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["periods_per_year"] = self.periods_per_year()
        d["period_hours"] = self.period_hours()
        return d


class StrategyV2:
    """Chiến lược cross-sectional market-neutral thế hệ 2."""

    def __init__(self, config: Optional[StrategyV2Config] = None):
        self.cfg = config or StrategyV2Config()
        self.diagnostics: Dict = {}

    # -- tầng 1 + 2 ---------------------------------------------------------
    def score(self, panel: Dict[str, pd.DataFrame], funding: pd.DataFrame,
              marks: Optional[pd.Index] = None) -> pd.DataFrame:
        """
        Điểm số tổng hợp trên lưới tái cân bằng.

        Tín hiệu được tính trên TOÀN chuỗi nến (cần lịch sử warm-up) rồi mới lấy mẫu
        xuống lưới tái cân bằng. Làm ngược lại — lấy mẫu trước rồi tính — sẽ đổi ý
        nghĩa của mọi tham số lookback và làm research không khớp live.
        """
        fam_full = build_all_families(panel, funding, families=self.cfg.families)
        close = panel["close"]
        marks = marks if marks is not None else close.index[::self.cfg.rebalance_every]

        fam = {k: v.reindex(marks) for k, v in fam_full.items()}
        combined, weights, fac = combine_adaptive(
            fam, close.reindex(marks), self.cfg.combiner,
            top_frac=self.cfg.portfolio.top_frac, return_diagnostics=True)

        self.diagnostics["family_weights"] = weights
        self.diagnostics["factor_returns"] = fac
        self.diagnostics["family_signals"] = fam
        return combined

    # -- tầng 3 -------------------------------------------------------------
    def target_weights(self, panel: Dict[str, pd.DataFrame], funding: pd.DataFrame,
                       marks: Optional[pd.Index] = None) -> pd.DataFrame:
        """
        Trọng số mục tiêu (gross = 1.0, chưa đòn bẩy) — ĐIỂM VÀO DUY NHẤT cho cả
        backtest lẫn live. Live lấy `.iloc[-1]`.
        """
        close = panel["close"]
        marks = marks if marks is not None else close.index[::self.cfg.rebalance_every]
        sig = self.score(panel, funding, marks)
        return build_weights(sig, close.reindex(marks), self.cfg.portfolio)

    # -- tầng 4 + đo lường --------------------------------------------------
    def backtest(
        self,
        panel: Dict[str, pd.DataFrame],
        funding: pd.DataFrame,
        mask: Optional[np.ndarray] = None,
        cost: Optional[CostModel] = None,
        with_leverage: bool = True,
    ) -> Dict:
        """
        Chạy toàn bộ chuỗi tầng và trả về thống kê trước/sau đòn bẩy.

        `mask` cắt theo train/holdout SAU khi tín hiệu đã tính trên toàn chuỗi —
        hợp lệ vì mọi tín hiệu đều nhân quả, và tránh mất vài trăm nến warm-up ở
        đầu đoạn được cắt.
        """
        close = panel["close"]
        marks = close.index[::self.cfg.rebalance_every]
        if mask is not None:
            marks = marks[np.isin(marks, close.index[mask])]

        W = self.target_weights(panel, funding, marks)

        if cost is None:
            per_sym = estimate_cost_bps(
                panel, notional_usd=self.cfg.notional_per_order,
                bars_per_day=max(1, int(round(24 / self.cfg.bar_hours()))))
            cost = CostModel(maker_ratio=self.cfg.maker_ratio, half_spread_bps=0.0,
                             per_symbol_bps=per_sym * (1 - self.cfg.maker_ratio), min_bps=1.0)

        res = simulate(W, close.reindex(marks), funding.reindex(marks), cost,
                       bar_hours=self.cfg.period_hours(), rebalance_every=1)
        ppy = self.cfg.periods_per_year()
        out = {"unlevered": res.stats(ppy), "result": res, "weights": W,
               "cost_bps_mean": res.meta["cost_bps_mean"], "periods_per_year": ppy}

        if with_leverage:
            lev = apply_leverage(res.returns, self.cfg.leverage)
            r = lev["returns"]
            mu, sd = float(r.mean()), float(r.std(ddof=1))
            eq = lev["equity"]
            dd = float((1 - eq / eq.cummax()).max())
            out["levered"] = {
                "ann_return": mu * ppy,
                "ann_vol": sd * np.sqrt(ppy),
                "sharpe": mu / sd * np.sqrt(ppy) if sd > 1e-15 else 0.0,
                "max_dd": dd,
                "total_return": float(eq.iloc[-1] - 1.0),
                "avg_leverage": float(lev["leverage"].mean()),
                "max_leverage": float(lev["leverage"].max()),
            }
            out["levered_series"] = lev

        return out
