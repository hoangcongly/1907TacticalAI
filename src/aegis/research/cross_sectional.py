"""
Engine backtest cross-sectional (market-neutral) đa tài sản.

Nguyên lý: tại mỗi thời điểm, XẾP HẠNG các tài sản theo tín hiệu rồi mua nhóm đầu /
bán nhóm cuối với tổng vị thế ròng bằng 0. Beta thị trường bị khử, nên tín hiệu yếu
vẫn có thể cho Sharpe cao — trái ngược với đặt cược hướng đi của một tài sản, nơi
beta lấn át tất cả.

Mọi tín hiệu đều được DỊCH MỘT NHỊP (shift) trước khi vào lệnh: giá trị tại t chỉ
được dùng để mở vị thế ăn lợi suất từ t đến t+1. Không có đường nào nhìn trước.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# Giờ mỗi nến theo khung — dùng để phân bổ funding (quyết toán mỗi 8h).
BAR_HOURS = {"1h": 1, "4h": 4, "8h": 8, "1d": 24}


@dataclass
class BacktestResult:
    """Kết quả backtest cross-sectional."""
    returns: pd.Series          # lợi suất ròng mỗi chu kỳ tái cân bằng
    gross_returns: pd.Series
    cost_drag: pd.Series
    funding_pnl: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame

    @property
    def n_periods(self) -> int:
        return int(self.returns.notna().sum())

    def stats(self, periods_per_year: float) -> Dict[str, float]:
        r = self.returns.dropna().to_numpy()
        if len(r) < 10:
            return {"n": len(r), "viable": 0.0}

        mean, std = float(r.mean()), float(r.std(ddof=1))
        sharpe = (mean / std * np.sqrt(periods_per_year)) if std > 1e-15 else 0.0
        # t-stat cho H0: lợi suất kỳ vọng = 0. Đây mới là thước đo "có edge hay không".
        t_stat = (mean / (std / np.sqrt(len(r)))) if std > 1e-15 else 0.0

        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity)
        max_dd = float((1.0 - equity / peak).max())

        return {
            "n": float(len(r)),
            "mean_per_period": mean,
            "sharpe": float(sharpe),
            "t_stat": float(t_stat),
            "ann_return": float(mean * periods_per_year),
            "ann_vol": float(std * np.sqrt(periods_per_year)),
            "max_dd": max_dd,
            "hit_rate": float((r > 0).mean()),
            "total_return": float(equity[-1] - 1.0),
            "avg_turnover": float(self.turnover.dropna().mean()),
        }


def rank_to_weights(
    signal: pd.DataFrame,
    top_frac: float = 0.25,
    neutral: bool = True,
) -> pd.DataFrame:
    """
    Biến tín hiệu thành trọng số dollar-neutral.

    Mỗi hàng: mua `top_frac` tài sản có tín hiệu cao nhất, bán `top_frac` thấp nhất,
    tổng |trọng số| = 1.0 (gross exposure = 1x), tổng trọng số = 0 (net = 0).
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for ts, row in signal.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n < 4:
            continue

        k = max(1, int(round(n * top_frac)))
        ordered = valid.sort_values()
        shorts, longs = ordered.index[:k], ordered.index[-k:]

        if neutral:
            weights.loc[ts, longs] = 0.5 / k
            weights.loc[ts, shorts] = -0.5 / k
        else:
            weights.loc[ts, longs] = 1.0 / k

    return weights


def rank_to_weights_buffered(
    signal: pd.DataFrame,
    entry_frac: float = 0.10,
    exit_frac: float = 0.25,
    neutral: bool = True,
) -> pd.DataFrame:
    """
    Xếp hạng có VÙNG ĐỆM — giảm turnover cho danh mục dựa trên thứ hạng.

    VÌ SAO CẦN: với danh mục top-decile, trọng số là nhị phân (trong hoặc ngoài),
    nên "dải không giao dịch" theo trọng số không bao giờ kích hoạt. Một cặp trượt
    từ hạng 6 xuống hạng 7 bị bán sạch rồi mua lại cặp khác — churn thuần tuý.

    Cơ chế đệm (chuẩn ngành cho danh mục nhân tố / chỉ số):
      - VÀO   khi lọt top `entry_frac`
      - GIỮ   cho tới khi rơi khỏi top `exit_frac` (rộng hơn)
    Cặp nằm giữa hai ngưỡng được giữ nguyên nếu đang nắm, không mở mới nếu chưa.
    """
    if not (0 < entry_frac <= exit_frac < 0.5):
        raise ValueError(f"Cần 0 < entry_frac <= exit_frac < 0.5, nhận {entry_frac}, {exit_frac}")

    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    held_long: set = set()
    held_short: set = set()

    for ts, row in signal.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n < 4:
            held_long, held_short = set(), set()
            continue

        k_in = max(1, int(round(n * entry_frac)))
        k_out = max(k_in, int(round(n * exit_frac)))
        ordered = valid.sort_values()

        enter_short, enter_long = set(ordered.index[:k_in]), set(ordered.index[-k_in:])
        keep_short, keep_long = set(ordered.index[:k_out]), set(ordered.index[-k_out:])

        # ƯU TIÊN VỊ THẾ ĐANG GIỮ. Đây là toàn bộ ý nghĩa của vùng đệm: cặp đang
        # nắm mà vẫn nằm trong vùng giữ thì KHÔNG bán, dù đã rơi khỏi top vào.
        # Chỉ khi còn chỗ trống mới nhận cặp mới. (Nếu cắt lại theo tín hiệu mạnh
        # nhất, ta sẽ chọn đúng tập `enter_*` và vùng đệm bị vô hiệu hoàn toàn.)
        survivors_long = held_long & keep_long
        survivors_short = held_short & keep_short
        survivors_long -= survivors_short
        survivors_short -= survivors_long

        room_long = max(0, k_in - len(survivors_long))
        room_short = max(0, k_in - len(survivors_short))

        cand_long = [x for x in reversed(ordered.index)
                     if x in enter_long and x not in survivors_long and x not in survivors_short]
        cand_short = [x for x in ordered.index
                      if x in enter_short and x not in survivors_short and x not in survivors_long]

        held_long = survivors_long | set(cand_long[:room_long])
        held_short = survivors_short | set(cand_short[:room_short])

        # Thừa chỗ (vùng giữ rộng hơn vùng vào) -> bỏ cặp có tín hiệu YẾU NHẤT.
        if len(held_long) > k_in:
            held_long = set(valid[list(held_long)].sort_values().index[-k_in:])
        if len(held_short) > k_in:
            held_short = set(valid[list(held_short)].sort_values().index[:k_in])

        if not held_long or not held_short:
            continue
        if neutral:
            weights.loc[ts, list(held_long)] = 0.5 / len(held_long)
            weights.loc[ts, list(held_short)] = -0.5 / len(held_short)
        else:
            weights.loc[ts, list(held_long)] = 1.0 / len(held_long)

    return weights


def backtest_cross_sectional(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    funding: Optional[pd.DataFrame] = None,
    interval: str = "4h",
    rebalance_every: int = 6,
    horizon: Optional[int] = None,
    top_frac: float = 0.25,
    cost_rate: float = 0.0004,
    neutral: bool = True,
    exit_frac: Optional[float] = None,
) -> BacktestResult:
    """
    Backtest danh mục cross-sectional có tính đủ chi phí.

    - `signal`: giá trị tại t (chỉ dùng dữ liệu <= t). Được shift nội bộ để vào lệnh.
    - `rebalance_every`: số nến giữa hai lần tái cân bằng.
    - `cost_rate`: phí một chiều (taker 0.0004, maker 0.0001) — nhân với turnover.
    - `funding`: panel funding rate; phân bổ theo số giờ nắm giữ / 8h.

    Vị thế LONG trả funding khi rate > 0, nên funding_pnl = -w * rate.
    """
    horizon = horizon or rebalance_every
    bar_hours = BAR_HOURS.get(interval, 4)

    # Chỉ tái cân bằng tại các mốc định kỳ.
    marks = signal.index[::rebalance_every]
    sig = signal.reindex(marks)
    px = close.reindex(marks)

    # Lợi suất từ mốc này tới mốc kế tiếp — đúng thứ vị thế sẽ ăn.
    fwd = px.shift(-1) / px - 1.0

    if exit_frac is not None:
        weights = rank_to_weights_buffered(sig, entry_frac=top_frac,
                                           exit_frac=exit_frac, neutral=neutral)
    else:
        weights = rank_to_weights(sig, top_frac=top_frac, neutral=neutral)

    gross = (weights * fwd).sum(axis=1, min_count=1)

    # Turnover = tổng thay đổi trọng số giữa hai kỳ (vòng đầu tính toàn bộ).
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * cost_rate

    if funding is not None and not funding.empty:
        fr = funding.reindex(marks).reindex(columns=weights.columns)
        periods_of_funding = (rebalance_every * bar_hours) / 8.0
        funding_pnl = -(weights * fr).sum(axis=1, min_count=1) * periods_of_funding
    else:
        funding_pnl = pd.Series(0.0, index=marks)

    net = gross.fillna(0.0) - cost.fillna(0.0) + funding_pnl.fillna(0.0)
    net.iloc[-1] = np.nan  # kỳ cuối chưa có lợi suất tương lai

    return BacktestResult(
        returns=net, gross_returns=gross, cost_drag=cost,
        funding_pnl=funding_pnl, turnover=turnover, weights=weights,
    )


# ============================================================================
# THƯ VIỆN TÍN HIỆU CROSS-SECTIONAL
# ============================================================================
def xs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá theo HÀNG (cross-sectional): khử mức chung, giữ thứ hạng tương đối."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def signal_momentum(close: pd.DataFrame, lookback: int = 30, skip: int = 1) -> pd.DataFrame:
    """Động lượng cross-sectional: lợi suất quá khứ, bỏ qua `skip` nến gần nhất."""
    return xs_zscore(close.shift(skip) / close.shift(skip + lookback) - 1.0)


def signal_reversal(close: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """Hồi quy ngắn hạn: NGƯỢC dấu lợi suất gần nhất."""
    return xs_zscore(-(close / close.shift(lookback) - 1.0))


def signal_funding_carry(funding: pd.DataFrame, smooth: int = 6) -> pd.DataFrame:
    """
    Carry funding: BÁN cặp có funding cao, MUA cặp có funding thấp.

    Vị thế short trên perp có funding dương sẽ NHẬN funding. Đây là dòng tiền cấu
    trúc của thị trường perp, không phải dự đoán hướng giá — nên độ bền cao hơn hẳn.
    """
    return xs_zscore(-funding.rolling(smooth, min_periods=1).mean())


def signal_low_vol(close: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """Bất thường biến động thấp: mua tài sản ít biến động, bán tài sản nhiều."""
    rets = close / close.shift(1) - 1.0
    return xs_zscore(-rets.rolling(lookback, min_periods=lookback // 2).std())


def signal_ofi(ofi: pd.DataFrame, smooth: int = 6) -> pd.DataFrame:
    """Mất cân bằng dòng lệnh cross-sectional (vi cấu trúc)."""
    return xs_zscore(ofi.rolling(smooth, min_periods=1).mean())


# ============================================================================
# TÍN HIỆU MỞ RỘNG — mỗi tín hiệu có cơ sở lý thuyết, KHÔNG dò tìm mù
# ============================================================================
def signal_risk_adjusted_momentum(
    close: pd.DataFrame, lookback: int = 90, skip: int = 1, vol_window: int = 30
) -> pd.DataFrame:
    """
    Động lượng chia cho biến động ("Sharpe momentum").

    Cơ sở: động lượng thô thiên vị các tài sản biến động mạnh — chúng có lợi suất
    quá khứ lớn chỉ vì rủi ro lớn, không phải vì xu hướng bền. Chuẩn hoá theo vol
    tách phần xu hướng thật ra khỏi phần chỉ là biến động.
    """
    mom = close.shift(skip) / close.shift(skip + lookback) - 1.0
    rets = close / close.shift(1) - 1.0
    vol = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    return xs_zscore(mom / vol.replace(0.0, np.nan))


def signal_long_term_reversal(close: pd.DataFrame, lookback: int = 360, skip: int = 90) -> pd.DataFrame:
    """
    Đảo chiều dài hạn: NGƯỢC dấu lợi suất rất dài hạn (bỏ qua đoạn gần đây).

    Cơ sở: động lượng hoạt động ở chân trời trung hạn nhưng ĐẢO NGƯỢC ở chân trời
    rất dài — tài sản tăng nhiều nhất trong 1-3 năm có xu hướng kém hơn sau đó.
    Hiệu ứng kinh điển, khác chân trời với momentum_90 nên ít tương quan.
    """
    return xs_zscore(-(close.shift(skip) / close.shift(skip + lookback) - 1.0))


def signal_negative_skew(close: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    """
    Cầu xổ số (lottery demand): BÁN tài sản có độ lệch DƯƠNG cao.

    Cơ sở: nhà đầu tư nhỏ lẻ trả giá quá cao cho tài sản có xác suất nhỏ thắng lớn
    (độ lệch dương). Phần bù đó âm. Hiệu ứng đặc biệt mạnh ở crypto do thành phần
    tham gia nghiêng về bán lẻ.
    """
    rets = close / close.shift(1) - 1.0
    return xs_zscore(-rets.rolling(lookback, min_periods=lookback // 2).skew())


def signal_funding_momentum(funding: pd.DataFrame, lookback: int = 42) -> pd.DataFrame:
    """
    Đà thay đổi funding: funding đang TĂNG báo hiệu vị thế long đang chen chúc.

    Khác với `signal_funding_carry` (dựa trên MỨC funding), tín hiệu này dựa trên
    THAY ĐỔI — bắt lúc đòn bẩy đang tích tụ, thường đi trước các đợt thanh lý.
    """
    smooth = funding.rolling(6, min_periods=1).mean()
    return xs_zscore(-(smooth - smooth.shift(lookback)))


def signal_idiosyncratic_vol(close: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Bất thường biến động riêng: MUA tài sản có biến động RIÊNG thấp.

    Khác `signal_low_vol` ở chỗ đã khử biến động chung của thị trường trước, nên
    chỉ còn phần rủi ro riêng của từng tài sản.
    """
    rets = close / close.shift(1) - 1.0
    market = rets.mean(axis=1)
    idio = rets.sub(market, axis=0)
    return xs_zscore(-idio.rolling(lookback, min_periods=lookback // 2).std())


def combine_inverse_vol(
    returns_by_signal: Dict[str, pd.Series],
    lookback: int = 90,
    min_periods: int = 30,
) -> pd.Series:
    """
    Gộp nhiều chiến lược theo trọng số NGHỊCH ĐẢO BIẾN ĐỘNG (risk parity đơn giản).

    Chia đều VỐN khiến chiến lược biến động mạnh chiếm phần lớn RỦI RO danh mục.
    Chia đều RỦI RO cho Sharpe cao hơn. Trọng số tính nhân quả (rolling), không
    dùng biến động của toàn mẫu.
    """
    df = pd.DataFrame(returns_by_signal).dropna(how="all")
    vol = df.rolling(lookback, min_periods=min_periods).std().shift(1)
    inv = (1.0 / vol.replace(0.0, np.nan))
    weights = inv.div(inv.sum(axis=1), axis=0)
    return (df * weights).sum(axis=1, min_count=1).dropna()


def volatility_target(
    returns: pd.Series,
    target_ann_vol: float = 0.20,
    lookback: int = 60,
    min_periods: int = 20,
    periods_per_year: float = 365.0,
    max_leverage: float = 3.0,
) -> pd.Series:
    """
    Điều tiết vị thế theo mục tiêu biến động.

    Cơ sở: biến động có tính CỤM và DỰ ĐOÁN ĐƯỢC (lợi suất thì không). Giảm vị thế
    khi biến động cao và tăng khi thấp giúp ổn định rủi ro, thường nâng Sharpe và
    cắt bớt đuôi trái — dù không làm tăng lợi suất thô.

    Hệ số đòn bẩy dùng biến động ƯỚC LƯỢNG TỚI HÔM QUA (shift 1) -> nhân quả.
    """
    realized = returns.rolling(lookback, min_periods=min_periods).std().shift(1)
    ann = realized * np.sqrt(periods_per_year)
    lev = (target_ann_vol / ann.replace(0.0, np.nan)).clip(upper=max_leverage).fillna(0.0)
    return (returns * lev).dropna()


def combine_signals_zscore(
    signals: Dict[str, pd.DataFrame],
    min_coverage: int = 2,
) -> pd.DataFrame:
    """
    Gộp nhiều tín hiệu cross-sectional thành MỘT điểm số tổng hợp.

    Đây là cách dựng danh mục đa nhân tố chuẩn: trung bình các z-score rồi xếp hạng
    MỘT LẦN, thay vì chạy N danh mục riêng rồi cộng lợi suất. Với vốn nhỏ đây là
    cách duy nhất khả thi — N danh mục riêng cần gấp N lần số vị thế.

    Hai chi tiết quyết định tính đúng đắn:

    1. GIỮ NaN, KHÔNG fillna(0). Điền 0 nghĩa là coi "không có dữ liệu" như "tín
       hiệu trung tính", khiến cặp thiếu dữ liệu vẫn được xếp hạng và lọt vào danh
       mục dựa trên thông tin không tồn tại.

    2. YÊU CẦU ĐỘ PHỦ TỐI THIỂU. Cặp chỉ có 1/4 tín hiệu sẽ được chấm bằng đúng
       tín hiệu đó — nhiễu đội lốt điểm tổng hợp. `min_coverage` chặn điều này.
    """
    if not signals:
        raise ValueError("Cần ít nhất một tín hiệu để gộp")
    if min_coverage < 1:
        raise ValueError(f"min_coverage phải >= 1, nhận {min_coverage}")

    frames = list(signals.values())
    stacked = pd.concat(frames)
    total = stacked.groupby(level=0).sum(min_count=1)
    count = pd.concat([f.notna().astype(float) for f in frames]).groupby(level=0).sum()

    combined = total / count.replace(0.0, np.nan)
    return combined.where(count >= min_coverage)
