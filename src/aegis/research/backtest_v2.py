"""
Engine backtest cross-sectional thế hệ 2 — nhận TRỌNG SỐ dựng sẵn.

Khác engine cũ (`cross_sectional.backtest_cross_sectional`) ở bốn điểm, mỗi điểm
đều là chênh lệch giữa con số backtest và con số tài khoản thật:

1. TRỌNG SỐ TRÔI GIỮA HAI LẦN TÁI CÂN BẰNG. Danh mục thật không giữ nguyên trọng
   số — vị thế thắng phình ra, vị thế thua co lại. Turnover thật là khoảng cách từ
   trọng số ĐÃ TRÔI tới mục tiêu mới, không phải từ mục tiêu cũ. Engine cũ tính từ
   mục tiêu cũ nên ĐỘI chi phí (và với tín hiệu động lượng thì đội khá nhiều, vì
   trôi giá tự đẩy danh mục về đúng hướng tín hiệu muốn).

2. CHI PHÍ THEO TỪNG CẶP. Phí sàn thì như nhau nhưng SPREAD và tác động giá thì
   không: giao dịch $10 trên BTC gần như miễn phí trượt giá, cùng số tiền đó trên
   một memecoin vốn hoá nhỏ có thể mất 20bp. Một mức phí phẳng cho cả rổ luôn
   khiến chiến lược trông tốt hơn thực tế — vì tín hiệu thường mạnh nhất ở đúng
   những cặp đắt đỏ nhất.

3. FUNDING THEO MỐC QUYẾT TOÁN THẬT. Funding trả vào 00:00/08:00/16:00 UTC, không
   trải đều. Vị thế mở 4 tiếng có thể vắt qua 1 mốc hoặc 0 mốc.

4. LỢI SUẤT CỘNG DỒN THEO VỐN THẬT. Với đòn bẩy và mục tiêu biến động, chuỗi lợi
   suất số học không mô tả đúng đường vốn; engine trả cả hai để không nhầm lẫn.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    "CostModel",
    "BacktestV2Result",
    "simulate",
    "estimate_cost_bps",
    "drift_weights",
]

FUNDING_INTERVAL_MS = 8 * 3_600_000


# ---------------------------------------------------------------------------
# Mô hình chi phí
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    """
    Chi phí một chiều theo bp trên notional giao dịch.

    `taker_fee_bps` / `maker_fee_bps`: phí sàn. Binance USDⓈ-M VIP0 là 5.0 / 2.0 bp
    (0.0500% / 0.0200%); trả phí bằng BNB được giảm 10% -> 4.5 / 1.8 bp. Mặc định ở
    đây là VIP0 KHÔNG giảm giá — giả định bi quan là giả định đúng khi chưa đo được.
    `maker_ratio`: tỷ lệ notional khớp ở giá thụ động. Đo từ `core/execution_log.py`,
    KHÔNG được giả định — testnet đo được 0.378, mainnet thường cao hơn.
    `half_spread_bps`: nửa spread phải trả khi cắn giá. Ước lượng từ dữ liệu nếu có.
    `impact_coef`: hệ số tác động giá căn bậc hai — chi phí ≈ coef * sqrt(size/ADV).
    """

    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    maker_ratio: float = 0.5
    half_spread_bps: float = 1.0
    impact_coef_bps: float = 0.0
    per_symbol_bps: Optional[pd.Series] = None
    min_bps: float = 0.5

    def base_bps(self) -> float:
        """Chi phí phẳng khi không có thông tin từng cặp."""
        fee = self.maker_ratio * self.maker_fee_bps + (1 - self.maker_ratio) * self.taker_fee_bps
        slip = (1 - self.maker_ratio) * self.half_spread_bps
        return max(self.min_bps, fee + slip)

    def bps_for(self, symbols) -> pd.Series:
        """Vector chi phí một chiều (bp) cho từng cặp."""
        base = self.base_bps()
        out = pd.Series(base, index=list(symbols), dtype=np.float64)
        if self.per_symbol_bps is not None:
            extra = self.per_symbol_bps.reindex(out.index).fillna(0.0)
            out = out + extra
        return out.clip(lower=self.min_bps)


SPREAD_CACHE = "artifacts/live_spreads_bps.json"


def load_live_spreads(path: str = SPREAD_CACHE) -> Optional[pd.Series]:
    """
    Spread THẬT đo từ sổ lệnh Binance (`/fapi/v1/ticker/bookTicker`), lưu sẵn ra file.

    Đây là mỏ neo tốt nhất có được nếu không lưu sổ lệnh lịch sử. Sinh lại bằng
    `python scripts/refresh_spreads.py`.
    """
    import json
    import pathlib

    p = pathlib.Path(path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text())
    if not data:
        return None
    return pd.Series(data, dtype=np.float64)


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame,
                          bars_per_day: int = 6) -> pd.Series:
    """
    Ước lượng spread Corwin-Schultz (2012) — CHỈ dùng khi không có spread thật.

    Ý tưởng: biên độ high-low của một nến gồm phần biến động (tỷ lệ với thời gian)
    cộng phần spread (không tỷ lệ). So biên độ 1 nến với biên độ 2 nến gộp tách
    được hai phần.

    CẢNH BÁO ĐÃ KIỂM CHỨNG TRÊN DỮ LIỆU NÀY: ước lượng chỉ có nghĩa trên nến NGÀY.
    Áp thẳng lên nến 4h cho ra trung vị 24.8bp trong khi sổ lệnh thật là 0.65bp —
    sai số 40 lần, đủ để giết nhầm một chiến lược sống được. Vì vậy hàm này tự tổng
    hợp lên khung ngày trước, và kết quả vẫn bị kẹp trần.
    """
    n = max(1, int(bars_per_day))
    hi_d = high.rolling(n).max().iloc[::n]
    lo_d = low.rolling(n).min().iloc[::n]

    hi2 = hi_d.rolling(2).max()
    lo2 = lo_d.rolling(2).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (np.log(hi_d / lo_d) ** 2).rolling(2).sum()
        gamma = np.log(hi2 / lo2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)).clip(lower=0.0)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return (spread.median() * 1e4).clip(lower=0.2, upper=60.0)


def estimate_cost_bps(
    panel: Dict[str, pd.DataFrame],
    notional_usd: float = 20.0,
    lookback: int = 720,
    bars_per_day: int = 6,
    spread_path: str = SPREAD_CACHE,
    impact_coef: float = 1.0,
) -> pd.Series:
    """
    Chi phí TRƯỢT GIÁ một chiều theo từng cặp, tính bằng bp (chưa gồm phí sàn).

    Hai thành phần:

    1. NỬA SPREAD — lấy từ sổ lệnh thật nếu có (`artifacts/live_spreads_bps.json`),
       nếu không thì Corwin-Schultz trên nến ngày. Với lệnh cắn giá, đây là chi phí
       chính và ở quy mô vốn nhỏ nó gần như là chi phí DUY NHẤT ngoài phí sàn.

    2. TÁC ĐỘNG GIÁ theo luật căn bậc hai: `sigma_ngày * c * sqrt(N / ADV)`.
       Ở $20-100 mỗi lệnh trên perp có ADV hàng chục triệu đô, số hạng này ra
       khoảng 0.01-0.1bp — tức là KHÔNG ĐÁNG KỂ. Đây là kết luận quan trọng chứ
       không phải chi tiết vụn: ở vốn nhỏ, chi phí là phí sàn + spread, và cách duy
       nhất để giảm nó là khớp lệnh THỤ ĐỘNG, không phải giao dịch ít đi.
    """
    close = panel["close"]
    tail = slice(-lookback, None)

    live = load_live_spreads(spread_path)
    if live is not None:
        half_spread = (live.reindex(close.columns) * 0.5)
        missing = half_spread.isna()
        if missing.any():
            fallback = corwin_schultz_spread(panel["high"], panel["low"], bars_per_day) * 0.5
            half_spread[missing] = fallback.reindex(half_spread.index)[missing]
    else:
        half_spread = corwin_schultz_spread(panel["high"], panel["low"], bars_per_day) * 0.5

    # Tác động giá: sigma_ngày * sqrt(notional / ADV).
    dollar_vol = (panel["volume"] * close).iloc[tail]
    adv = (dollar_vol.median() * bars_per_day).replace(0.0, np.nan)
    sigma_day = (np.log(close / close.shift(1)).iloc[tail].std() * np.sqrt(bars_per_day))
    impact = (impact_coef * sigma_day * np.sqrt(notional_usd / adv) * 1e4).clip(lower=0.0, upper=20.0)

    total = (half_spread.fillna(half_spread.median()) + impact.fillna(0.0))
    return total.replace([np.inf, -np.inf], np.nan).fillna(total.median()).clip(lower=0.1)


# ---------------------------------------------------------------------------
# Trôi trọng số
# ---------------------------------------------------------------------------
def drift_weights(w: pd.Series, asset_return: pd.Series) -> pd.Series:
    """
    Trọng số sau một chu kỳ, do giá dịch chuyển chứ không do giao dịch.

    Với danh mục dollar-neutral có đòn bẩy, giá trị mỗi chân nhân với (1 + r), còn
    VỐN nhân với (1 + w.r). Trọng số mới = w(1+r) / (1 + w.r).
    """
    r = asset_return.reindex(w.index).fillna(0.0)
    port_ret = float((w * r).sum())
    denom = 1.0 + port_ret
    if abs(denom) < 1e-9:
        return w * 0.0
    return w * (1.0 + r) / denom


# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------
@dataclass
class BacktestV2Result:
    returns: pd.Series
    gross_returns: pd.Series
    cost_drag: pd.Series
    funding_pnl: pd.Series
    turnover: pd.Series
    n_positions: pd.Series
    net_exposure: pd.Series
    gross_exposure: pd.Series
    weights: pd.DataFrame
    meta: Dict = field(default_factory=dict)

    def stats(self, periods_per_year: float) -> Dict[str, float]:
        r = self.returns.dropna().to_numpy()
        if len(r) < 20:
            return {"n": float(len(r)), "viable": 0.0}

        mean, std = float(r.mean()), float(r.std(ddof=1))
        sharpe = mean / std * np.sqrt(periods_per_year) if std > 1e-15 else 0.0
        t_stat = mean / (std / np.sqrt(len(r))) if std > 1e-15 else 0.0

        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity)
        dd = 1.0 - equity / peak
        max_dd = float(dd.max())

        downside = r[r < 0]
        sortino = (mean / downside.std(ddof=1) * np.sqrt(periods_per_year)
                   if len(downside) > 2 and downside.std(ddof=1) > 1e-15 else 0.0)
        ann_ret = float(mean * periods_per_year)

        return {
            "n": float(len(r)),
            "ann_return": ann_ret,
            "ann_vol": float(std * np.sqrt(periods_per_year)),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "t_stat": float(t_stat),
            "max_dd": max_dd,
            "calmar": float(ann_ret / max_dd) if max_dd > 1e-9 else 0.0,
            "hit_rate": float((r > 0).mean()),
            "total_return": float(equity[-1] - 1.0),
            "cagr": float(equity[-1] ** (periods_per_year / len(r)) - 1.0) if equity[-1] > 0 else -1.0,
            "avg_turnover": float(self.turnover.dropna().mean()),
            "cost_drag_ann": float(self.cost_drag.dropna().mean() * periods_per_year),
            "funding_ann": float(self.funding_pnl.dropna().mean() * periods_per_year),
            "avg_positions": float(self.n_positions.dropna().mean()),
            "skew": float(pd.Series(r).skew()),
            "kurtosis": float(pd.Series(r).kurtosis()),
        }


# ---------------------------------------------------------------------------
# Mô phỏng
# ---------------------------------------------------------------------------
def simulate(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    funding: Optional[pd.DataFrame] = None,
    cost: Optional[CostModel] = None,
    bar_hours: float = 4.0,
    rebalance_every: int = 1,
    allow_drift: bool = True,
) -> BacktestV2Result:
    """
    Mô phỏng danh mục từ chuỗi trọng số MỤC TIÊU.

    `weights` phải nằm trên lưới thời gian của `close` và đã nhân quả (trọng số tại
    hàng t chỉ dùng thông tin <= t). Vị thế mở ở giá đóng nến t và ăn lợi suất
    t -> t+rebalance_every.

    `funding` là panel rate MỖI MỐC 8H đã căn về lưới nến (giá trị tại t = rate của
    mốc quyết toán gần nhất <= t). Vị thế LONG trả funding khi rate dương.
    """
    cost = cost or CostModel()
    marks = weights.index[::rebalance_every]
    W = weights.reindex(marks).reindex(columns=close.columns).fillna(0.0)
    px = close.reindex(marks)

    fwd = px.shift(-1) / px - 1.0
    cost_bps = cost.bps_for(close.columns) / 1e4

    period_hours = bar_hours * rebalance_every
    n_marks = len(marks)

    gross_r = np.full(n_marks, np.nan)
    cost_r = np.zeros(n_marks)
    fund_r = np.zeros(n_marks)
    turn = np.zeros(n_marks)
    npos = np.zeros(n_marks)
    net_exp = np.zeros(n_marks)
    gross_exp = np.zeros(n_marks)

    prev = pd.Series(0.0, index=close.columns)
    cost_vec = cost_bps.reindex(close.columns).to_numpy()

    fr = None
    if funding is not None and not funding.empty:
        fr = funding.reindex(marks).reindex(columns=close.columns).fillna(0.0)
    funding_periods = period_hours / 8.0

    for i, ts in enumerate(marks):
        target = W.loc[ts]
        traded = (target - prev).abs()
        turn[i] = float(traded.sum())
        cost_r[i] = float((traded.to_numpy() * cost_vec).sum())

        npos[i] = float((target != 0.0).sum())
        net_exp[i] = float(target.sum())
        gross_exp[i] = float(target.abs().sum())

        r = fwd.loc[ts]
        if r.notna().any():
            gross_r[i] = float((target * r.fillna(0.0)).sum())

        if fr is not None:
            fund_r[i] = -float((target * fr.loc[ts]).sum()) * funding_periods

        prev = drift_weights(target, r) if (allow_drift and r.notna().any()) else target

    net = pd.Series(np.nan_to_num(gross_r, nan=0.0) - cost_r + fund_r, index=marks)
    net.iloc[-1] = np.nan  # mốc cuối chưa có lợi suất tương lai
    net[np.isnan(gross_r)] = np.nan

    return BacktestV2Result(
        returns=net,
        gross_returns=pd.Series(gross_r, index=marks),
        cost_drag=pd.Series(cost_r, index=marks),
        funding_pnl=pd.Series(fund_r, index=marks),
        turnover=pd.Series(turn, index=marks),
        n_positions=pd.Series(npos, index=marks),
        net_exposure=pd.Series(net_exp, index=marks),
        gross_exposure=pd.Series(gross_exp, index=marks),
        weights=W,
        meta={"bar_hours": bar_hours, "rebalance_every": rebalance_every,
              "period_hours": period_hours, "cost_bps_mean": float(cost_bps.mean() * 1e4)},
    )
