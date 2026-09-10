"""
Phân tích hệ số thông tin (IC) — chẩn đoán tín hiệu TRƯỚC khi dựng danh mục.

VÌ SAO KHÔNG BACKTEST THẲNG: backtest trộn lẫn ba thứ khác nhau — chất lượng dự
báo, cách dựng danh mục, và chi phí giao dịch. Khi Sharpe thấp, backtest không nói
được thứ nào hỏng. IC tách riêng CHẤT LƯỢNG DỰ BÁO, nên nó trả lời được những câu
mà backtest không trả lời được:

  - Tín hiệu này dự báo được bao xa? (đường cong suy giảm IC -> chọn CHU KỲ TÁI
    CÂN BẰNG đúng, thay vì đoán "6 nến")
  - Dự báo còn đúng trong bao lâu nữa? (IC theo thời gian -> phát hiện edge chết)
  - Hai tín hiệu có bổ trợ nhau không? (tương quan IC, khác tương quan lợi suất)

ĐỊNH LUẬT CƠ BẢN CỦA QUẢN LÝ CHỦ ĐỘNG (Grinold):

        IR ≈ IC × √breadth

`breadth` = số quyết định độc lập mỗi năm ≈ số tài sản × số lần tái cân bằng. Đây
là công thức nói thẳng ra hai đòn bẩy mà hệ thống cũ đã tự chặn: nó xếp hạng 59 cặp
và ra quyết định 1 lần/ngày. Nhân đôi độ rộng theo mỗi chiều, cùng một IC, cho hệ
số 2 về IR — không cần model tốt hơn chút nào.

Vì các quyết định KHÔNG độc lập hoàn toàn (tín hiệu tự tương quan qua thời gian),
công thức này là TRẦN TRÊN. Hàm `effective_breadth` ở đây ước lượng phần chiết khấu.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ICResult",
    "information_coefficient",
    "ic_decay_curve",
    "ic_by_period",
    "signal_autocorrelation",
    "effective_breadth",
    "implied_sharpe",
    "ic_correlation_matrix",
    "quantile_returns",
    "decile_spread",
    "monotonicity",
]


@dataclass
class ICResult:
    """IC theo từng nến cộng phần tóm tắt."""

    series: pd.Series
    horizon: int

    @property
    def mean(self) -> float:
        return float(self.series.mean())

    @property
    def std(self) -> float:
        return float(self.series.std(ddof=1))

    @property
    def ir(self) -> float:
        """Tỷ lệ thông tin của chính IC: mean/std. Còn gọi là 'IC IR'."""
        return self.mean / self.std if self.std > 1e-15 else 0.0

    @property
    def t_stat(self) -> float:
        """
        t-stat cho H0: IC = 0.

        Đã hiệu chỉnh cho tự tương quan bằng số quan sát hiệu dụng — nếu không,
        tín hiệu chậm (động lượng 90 nến, quan sát chồng lấn nặng) sẽ cho t-stat
        thổi phồng gấp nhiều lần.
        """
        n_eff = _effective_n(self.series)
        return self.mean / (self.std / np.sqrt(n_eff)) if self.std > 1e-15 else 0.0

    @property
    def hit_rate(self) -> float:
        return float((self.series > 0).mean())

    def summary(self) -> Dict[str, float]:
        return {
            "horizon": float(self.horizon),
            "ic_mean": self.mean,
            "ic_std": self.std,
            "ic_ir": self.ir,
            "t_stat": self.t_stat,
            "hit_rate": self.hit_rate,
            "n": float(self.series.notna().sum()),
            "n_eff": float(_effective_n(self.series)),
        }


def _effective_n(x: pd.Series, max_lag: int = 50) -> float:
    """
    Số quan sát hiệu dụng sau khi trừ tự tương quan (Newey-West đơn giản hoá).

    n_eff = n / (1 + 2 Σ rho_k). Quan sát chồng lấn không mang thông tin mới; bỏ qua
    điều này là cách phổ biến nhất để một tín hiệu vô dụng trông có ý nghĩa thống kê.
    """
    v = x.dropna()
    n = len(v)
    if n < 10:
        return max(n, 1)
    arr = v.to_numpy()
    arr = arr - arr.mean()
    denom = float(arr @ arr)
    if denom <= 1e-18:
        return n
    factor = 1.0
    for k in range(1, min(max_lag, n // 4)):
        rho = float(arr[k:] @ arr[:-k]) / denom
        if rho <= 0:
            break
        factor += 2.0 * rho * (1.0 - k / n)
    return max(1.0, n / max(factor, 1.0))


def information_coefficient(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 6,
    method: str = "spearman",
    min_names: int = 8,
) -> ICResult:
    """
    Tương quan theo mặt cắt ngang giữa tín hiệu tại t và lợi suất t -> t+horizon.

    `method="spearman"` (mặc định) dùng thứ hạng — bền với đuôi dày của crypto.
    `method="pearson"` nhạy hơn với độ lớn, hữu ích khi ta định dùng trọng số liên tục.

    IC ~ 0.02-0.05 là bình thường và ĐỦ với độ rộng lớn. IC > 0.15 trên dữ liệu
    thật gần như luôn là dấu hiệu rò rỉ dữ liệu tương lai.
    """
    fwd = close.shift(-horizon) / close - 1.0
    common = signal.index.intersection(fwd.index)
    sig = signal.loc[common]
    ret = fwd.loc[common].reindex(columns=sig.columns)

    out = {}
    for ts in common:
        s, r = sig.loc[ts], ret.loc[ts]
        mask = s.notna() & r.notna()
        if int(mask.sum()) < min_names:
            continue
        a, b = s[mask], r[mask]
        if method == "spearman":
            a, b = a.rank(), b.rank()
        sa, sb = a.std(), b.std()
        if sa <= 1e-15 or sb <= 1e-15:
            continue
        out[ts] = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])

    return ICResult(pd.Series(out, name=f"ic_h{horizon}").sort_index(), horizon)


def ic_decay_curve(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    IC theo nhiều chân trời — công cụ chọn CHU KỲ TÁI CÂN BẰNG.

    Cách đọc: chân trời có IC×√(lần/năm) lớn nhất là chân trời tối ưu TRƯỚC CHI PHÍ.
    Cột `ic_per_sqrt_period` chuẩn hoá theo đúng nguyên tắc đó — IC cao ở chân trời
    dài không đáng bằng IC thấp hơn ở chân trời ngắn nếu chân trời ngắn cho nhiều
    lượt quyết định hơn.
    """
    horizons = horizons or [1, 2, 3, 6, 12, 24, 48, 90, 180]
    rows = []
    for h in horizons:
        res = information_coefficient(signal, close, horizon=h, method=method)
        s = res.summary()
        # Chuẩn hoá: quyết định giữ h nến thì mỗi năm có (periods/h) lượt.
        s["ic_per_sqrt_period"] = s["ic_mean"] / np.sqrt(h) if h > 0 else 0.0
        rows.append(s)
    return pd.DataFrame(rows).set_index("horizon")


def ic_by_period(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 6,
    freq: str = "YE",
) -> pd.DataFrame:
    """
    IC gộp theo năm/quý — phát hiện edge đang chết.

    Một tín hiệu có IC trung bình tốt nhưng chỉ đến từ 2020-2021 thì không dùng
    được: cấu trúc thị trường sinh ra nó đã biến mất. Đây là kiểm tra mà Sharpe
    toàn mẫu không bao giờ hiện ra.
    """
    res = information_coefficient(signal, close, horizon=horizon)
    s = res.series
    if s.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(s.index, unit="ms")
    grouped = s.groupby(pd.Series(idx, index=s.index).dt.to_period(
        {"YE": "Y", "QE": "Q", "ME": "M"}.get(freq, "Y")))
    return pd.DataFrame({
        "ic_mean": grouped.mean(),
        "ic_std": grouped.std(),
        "n": grouped.size(),
        "hit_rate": grouped.apply(lambda x: float((x > 0).mean())),
    })


def signal_autocorrelation(signal: pd.DataFrame, lags: Optional[List[int]] = None) -> pd.Series:
    """
    Tự tương quan mặt cắt ngang của tín hiệu qua các độ trễ.

    Đo TỐC ĐỘ THAY ĐỔI của tín hiệu, tức là turnover mà nó bắt danh mục phải chịu.
    Tự tương quan 0.98 ở lag 6 nghĩa là xếp hạng gần như không đổi — tái cân bằng
    dày hơn chỉ tạo phí chứ không tạo thông tin mới.
    """
    lags = lags or [1, 3, 6, 12, 24, 48, 90]
    out = {}
    for lag in lags:
        prev = signal.shift(lag)
        vals = []
        for ts in signal.index:
            a, b = signal.loc[ts], prev.loc[ts]
            mask = a.notna() & b.notna()
            if int(mask.sum()) < 8:
                continue
            x, y = a[mask].rank(), b[mask].rank()
            if x.std() <= 1e-15 or y.std() <= 1e-15:
                continue
            vals.append(float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1]))
        out[lag] = float(np.mean(vals)) if vals else np.nan
    return pd.Series(out, name="signal_autocorr")


def effective_breadth(
    signal: pd.DataFrame,
    rebalance_every: int,
    periods_per_year: float,
    n_positions: Optional[int] = None,
) -> Dict[str, float]:
    """
    Độ rộng HIỆU DỤNG sau chiết khấu vì các cược không độc lập.

    Độ rộng thô = số vị thế × số lần tái cân bằng mỗi năm. Nhưng nếu tín hiệu có
    tự tương quan rho ở đúng độ trễ bằng chu kỳ tái cân bằng thì cược ở kỳ này gần
    như lặp lại cược kỳ trước. Hệ số chiết khấu (1 - rho) là xấp xỉ chuẩn.

    Đây là con số nói cho ta biết tái cân bằng dày hơn có ĐÁNG hay không, trước khi
    tốn công viết thêm hạ tầng.
    """
    rho = float(signal_autocorrelation(signal, [rebalance_every]).iloc[0])
    rho = 0.0 if not np.isfinite(rho) else max(0.0, min(rho, 0.999))

    n_names = n_positions or int(signal.notna().sum(axis=1).median())
    rebals_per_year = periods_per_year / rebalance_every
    raw = n_names * rebals_per_year
    return {
        "n_names": float(n_names),
        "rebalances_per_year": float(rebals_per_year),
        "raw_breadth": float(raw),
        "signal_autocorr": rho,
        "independence_factor": float(1.0 - rho),
        "effective_breadth": float(raw * (1.0 - rho)),
    }


def implied_sharpe(ic_mean: float, breadth: float, transfer_coef: float = 0.6) -> float:
    """
    Sharpe kỳ vọng theo định luật cơ bản, đã tính hệ số truyền tải.

    `transfer_coef` (TC) là phần dự báo thực sự đi vào vị thế sau khi qua ràng buộc
    (trần trọng số, dollar/beta-neutral, giới hạn turnover). Danh mục thật hiếm khi
    vượt 0.5-0.7. Bỏ qua TC là lý do thường gặp khiến backtest hứa Sharpe 3 mà tài
    khoản thật chạy 1.
    """
    return float(transfer_coef * ic_mean * np.sqrt(max(breadth, 0.0)))


def ic_correlation_matrix(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizon: int = 6,
) -> pd.DataFrame:
    """
    Tương quan giữa các CHUỖI IC, không phải giữa các chuỗi lợi suất.

    Đây mới là đại lượng đúng để đánh giá tính bổ trợ. Hai tín hiệu có thể cho lợi
    suất tương quan thấp chỉ vì chúng chọn tài sản khác nhau, trong khi chất lượng
    dự báo lại lên xuống cùng nhau (cùng tốt lúc thị trường trend, cùng hỏng lúc
    sideway). Gộp chúng lại sẽ KHÔNG đa dạng hoá được như tương quan lợi suất hứa hẹn.
    """
    series = {}
    for name, sig in signals.items():
        res = information_coefficient(sig, close, horizon=horizon)
        if not res.series.empty:
            series[name] = res.series
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).corr()


# ---------------------------------------------------------------------------
# Phân tích theo PHÂN VỊ — cái mà IC không nhìn thấy
# ---------------------------------------------------------------------------
def quantile_returns(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 1,
    n_quantiles: int = 5,
    min_names: int = 10,
) -> pd.DataFrame:
    """
    Lợi suất trung bình theo từng NHÓM PHÂN VỊ của tín hiệu.

    VÌ SAO BẮT BUỘC PHẢI CÓ, ngoài IC: IC là tương quan hạng trên TOÀN mặt cắt
    ngang — nó đo mức độ ĐƠN ĐIỆU trung bình. Danh mục top/bottom decile lại chỉ
    giao dịch HAI ĐUÔI. Khi quan hệ không đơn điệu (hình chữ U hoặc chữ U ngược),
    hai đại lượng này có thể NGƯỢC DẤU nhau.

    Điều đó đã xảy ra thật trên dữ liệu này:
      - `low_idio_vol`: IC = +0.071 (t = 10.9) nhưng danh mục decile cho -29%/năm.
        Giữa phân phối thì biến động thấp thắng, nhưng đuôi biến động CAO nhất
        (coin mới niêm yết) tăng mạnh hơn tất cả — và đuôi mới là thứ ta mua/bán.
      - `mom_slow`: IC = -0.016 nhưng danh mục decile cho +49%/năm. Phần giữa hồi
        quy, hai đuôi thì tiếp diễn.

    Bài học phương pháp: CHỌN TÍN HIỆU THEO ĐẠI LƯỢNG MÌNH SẼ THỰC SỰ GIAO DỊCH.
    Nếu danh mục là decile thì tiêu chí là chênh lệch decile, không phải IC.
    """
    fwd = close.shift(-horizon) / close - 1.0
    common = signal.index.intersection(fwd.index)
    rows = []

    for ts in common:
        s, r = signal.loc[ts], fwd.loc[ts].reindex(signal.columns)
        m = s.notna() & r.notna()
        if int(m.sum()) < min_names:
            continue
        sv, rv = s[m], r[m]
        try:
            bucket = pd.qcut(sv.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        # Khử trung bình mặt cắt ngang: ta chỉ quan tâm lợi suất TƯƠNG ĐỐI, phần
        # chung của thị trường đã bị khử bởi thiết kế dollar-neutral.
        rel = rv - rv.mean()
        rows.append({f"Q{int(q)+1}": float(rel[bucket == q].mean()) for q in range(n_quantiles)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "mean_rel_return": df.mean(),
        "std": df.std(),
        "n": df.notna().sum(),
    })
    out["t_stat"] = out["mean_rel_return"] / (out["std"] / np.sqrt(out["n"].clip(lower=1)))
    return out


def decile_spread(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 1,
    top_frac: float = 0.10,
    min_names: int = 10,
) -> Dict[str, float]:
    """
    Chênh lệch lợi suất giữa nhóm đầu và nhóm cuối — ĐÚNG đại lượng danh mục kiếm được.

    Đây là tiêu chí chọn tín hiệu thay cho IC khi danh mục dựng theo top/bottom
    phân vị. Trả về trung bình, độ lệch chuẩn, t-stat (đã hiệu chỉnh tự tương quan)
    và Sharpe quy năm cho một chuỗi lợi suất chênh lệch.
    """
    fwd = close.shift(-horizon) / close - 1.0
    vals = {}
    for ts in signal.index.intersection(fwd.index):
        s, r = signal.loc[ts], fwd.loc[ts].reindex(signal.columns)
        m = s.notna() & r.notna()
        n = int(m.sum())
        if n < min_names:
            continue
        k = max(1, int(round(n * top_frac)))
        order = s[m].sort_values()
        rr = r[m]
        vals[ts] = float(rr[order.index[-k:]].mean() - rr[order.index[:k]].mean()) * 0.5

    ser = pd.Series(vals).sort_index()
    if len(ser) < 20:
        return {"n": float(len(ser)), "mean": 0.0, "t_stat": 0.0, "sharpe_per_period": 0.0}

    mu, sd = float(ser.mean()), float(ser.std(ddof=1))
    n_eff = _effective_n(ser)
    return {
        "n": float(len(ser)),
        "mean": mu,
        "std": sd,
        "t_stat": float(mu / (sd / np.sqrt(n_eff))) if sd > 1e-15 else 0.0,
        "sharpe_per_period": float(mu / sd) if sd > 1e-15 else 0.0,
        "hit_rate": float((ser > 0).mean()),
    }


def monotonicity(signal: pd.DataFrame, close: pd.DataFrame, horizon: int = 1,
                 n_quantiles: int = 5) -> float:
    """
    Mức độ đơn điệu: tương quan Spearman giữa số thứ tự nhóm và lợi suất nhóm.

    Gần +1 = tín hiệu đơn điệu tăng (IC và chênh lệch decile sẽ đồng thuận).
    Gần 0 hoặc âm khi IC dương = quan hệ hình chữ U — dấu hiệu bắt buộc phải nhìn
    `quantile_returns` trước khi kết luận bất cứ điều gì.
    """
    q = quantile_returns(signal, close, horizon, n_quantiles)
    if q.empty or len(q) < 3:
        return float("nan")
    ranks = np.arange(1, len(q) + 1)
    vals = q["mean_rel_return"].to_numpy()
    return float(pd.Series(ranks).corr(pd.Series(vals), method="spearman"))
