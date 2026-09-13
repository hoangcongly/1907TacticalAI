"""
Gộp tín hiệu bằng HỒI QUY MẶT CẮT NGANG — phương pháp Han-Zhou-Zhu, bản crypto CTREND.

NGUỒN: Han, Zhou & Zhu, "A Trend Factor..." (JF 2016) và bản áp dụng cho crypto
"A Trend Factor for the Cross Section of Cryptocurrency Returns" (JFQA 2025),
báo cáo long-short quintile ~3.87%/tuần trên 3.000+ coin, 2015-2022, dùng 28 tín
hiệu kỹ thuật gộp lại, tinh chỉnh bằng elastic net.

KHÁC GÌ SO VỚI TẦNG GỘP HIỆN CÓ (`adaptive_combiner`):

  Tầng cũ chấm mỗi HỌ tín hiệu bằng Sharpe của danh mục decile riêng của nó trong
  quá khứ, rồi lấy trọng số theo đó. Chỉ có 5 con số, và mỗi con số bỏ qua việc các
  họ tương quan với nhau — hai họ gần trùng nhau sẽ cùng nhận trọng số cao và danh
  mục bị đếm hai lần cùng một cược.

  Phương pháp này ước lượng ĐỒNG THỜI hệ số của TẤT CẢ tín hiệu bằng một hồi quy
  mặt cắt ngang tại mỗi thời điểm:

      r_{i,t+1} = sum_k  beta_{k,t} * S_{k,i,t} + e

  rồi dự báo bằng `beta` trung bình của các kỳ GẦN ĐÂY:

      Ehat[r_{i,t+1}] = sum_k  mean(beta_{k, t-w..t-1}) * S_{k,i,t}

  Hệ số hồi quy tự động chiết khấu phần thông tin trùng lặp giữa các tín hiệu —
  đúng thứ mà cách chấm điểm riêng lẻ không làm được.

VÌ SAO LẤY TRUNG BÌNH `beta` QUÁ KHỨ CHỨ KHÔNG DÙNG `beta` KỲ CUỐI: hệ số của MỘT
kỳ ước lượng từ một mặt cắt ngang (~120 quan sát) rất nhiễu. Trung bình qua w kỳ
giảm phương sai theo 1/sqrt(w), đổi lại phản ứng chậm hơn với đổi chế độ. Đây chính
là đánh đổi mà Han et al. giải quyết bằng cách chọn w đủ dài.

TÍNH NHÂN QUẢ: `beta` dùng tại t chỉ học từ các kỳ có lợi suất đã THỰC SỰ quan sát
được, tức là tới t-1. Hàm `_shift_for_causality` lo việc này và có test riêng.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["XSRegressionSpec", "cross_sectional_betas", "combine_xs_regression"]


@dataclass
class XSRegressionSpec:
    """Tham số tầng gộp bằng hồi quy mặt cắt ngang."""

    beta_window: int = 60        # số kỳ lấy trung bình hệ số
    min_periods: int = 20        # chưa đủ thì chưa dự báo
    min_names: int = 15          # số tài sản tối thiểu để hồi quy một kỳ
    method: str = "ols"          # "ols" | "ridge" | "elastic_net"
    alpha: float = 1e-3          # cường độ phạt cho ridge / elastic_net
    l1_ratio: float = 0.5        # chỉ dùng cho elastic_net
    winsorize: float = 0.02      # cắt đuôi lợi suất trước khi hồi quy
    standardize: bool = True     # chuẩn hoá lại tín hiệu theo hàng


def _winsorize(x: np.ndarray, frac: float) -> np.ndarray:
    """
    Cắt đuôi lợi suất trước khi hồi quy.

    Bắt buộc với crypto: một coin tăng 300% trong một kỳ sẽ một mình quyết định
    toàn bộ hệ số hồi quy của kỳ đó. Hồi quy bình phương tối thiểu không bền với
    đuôi dày, mà đuôi dày là đặc điểm cố hữu của lớp tài sản này.
    """
    if frac <= 0 or len(x) < 10:
        return x
    lo, hi = np.quantile(x, [frac, 1.0 - frac])
    return np.clip(x, lo, hi)


def _fit_one(X: np.ndarray, y: np.ndarray, spec: XSRegressionSpec) -> Optional[np.ndarray]:
    """Hệ số của MỘT kỳ. Trả None nếu không đủ dữ liệu hoặc ma trận suy biến."""
    if len(y) < spec.min_names:
        return None

    y = _winsorize(y, spec.winsorize)
    y = y - y.mean()                      # khử phần chung của thị trường

    if spec.method == "ols":
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return beta

    if spec.method == "ridge":
        k = X.shape[1]
        try:
            beta = np.linalg.solve(X.T @ X + spec.alpha * len(y) * np.eye(k), X.T @ y)
        except np.linalg.LinAlgError:
            return None
        return beta

    if spec.method == "elastic_net":
        from sklearn.linear_model import ElasticNet
        try:
            m = ElasticNet(alpha=spec.alpha, l1_ratio=spec.l1_ratio,
                           fit_intercept=False, max_iter=2000, tol=1e-5)
            m.fit(X, y)
        except Exception:
            return None
        return m.coef_

    raise ValueError(f"method không hợp lệ: {spec.method}")


def cross_sectional_betas(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: Optional[XSRegressionSpec] = None,
) -> pd.DataFrame:
    """
    Hệ số hồi quy mặt cắt ngang tại MỖI kỳ: index = thời gian, cột = tên tín hiệu.

    Hàng tại t là hệ số học từ lợi suất t -> t+1, tức là thông tin chỉ biết được
    SAU khi kỳ t kết thúc. Việc dịch cho đúng nhân quả do `combine_xs_regression`
    đảm nhiệm — hàm này cố tình trả về dạng "thô" để test kiểm chứng được.
    """
    spec = spec or XSRegressionSpec()
    names = list(signals)
    fwd = close.shift(-1) / close - 1.0

    rows, idx = [], []
    for ts in close.index:
        if ts not in fwd.index:
            continue
        r = fwd.loc[ts]
        cols = close.columns

        S = np.column_stack([signals[n].reindex(index=[ts], columns=cols).to_numpy()[0]
                             for n in names])
        y = r.reindex(cols).to_numpy(dtype=np.float64)

        ok = np.isfinite(y) & np.isfinite(S).all(axis=1)
        if ok.sum() < spec.min_names:
            continue

        X, yy = S[ok], y[ok]
        if spec.standardize:
            sd = X.std(axis=0)
            sd[sd < 1e-12] = 1.0
            X = (X - X.mean(axis=0)) / sd

        beta = _fit_one(X, yy, spec)
        if beta is None:
            continue
        rows.append(beta)
        idx.append(ts)

    if not rows:
        return pd.DataFrame(columns=names, dtype=np.float64)
    return pd.DataFrame(np.vstack(rows), index=idx, columns=names)


def combine_xs_regression(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: Optional[XSRegressionSpec] = None,
    return_betas: bool = False,
):
    """
    Điểm số tổng hợp = tín hiệu tại t nhân hệ số trung bình học từ các kỳ TRƯỚC t.

    Trả về DataFrame cùng shape với tín hiệu, đã chuẩn hoá theo hàng để tầng dựng
    danh mục dùng trực tiếp được.
    """
    spec = spec or XSRegressionSpec()
    names = list(signals)
    betas = cross_sectional_betas(signals, close, spec)
    if betas.empty:
        return (pd.DataFrame(np.nan, index=close.index, columns=close.columns),
                betas) if return_betas else pd.DataFrame(
                    np.nan, index=close.index, columns=close.columns)

    # NHÂN QUẢ: hệ số tại hàng t học từ lợi suất t->t+1. Muốn dùng để dự báo TẠI t
    # thì chỉ được lấy trung bình các hàng <= t-1. `shift(1)` trước rồi `rolling`.
    avg = betas.shift(1).rolling(spec.beta_window, min_periods=spec.min_periods).mean()
    avg = avg.reindex(close.index).ffill()

    score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cover = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for n in names:
        s = signals[n].reindex(index=close.index, columns=close.columns)
        w = avg[n]
        score = score.add(s.mul(w, axis=0).fillna(0.0), fill_value=0.0)
        cover = cover.add(s.notna().astype(float), fill_value=0.0)

    # Giữ NaN ở ô không đủ độ phủ tín hiệu, và ở giai đoạn chưa có hệ số.
    score = score.where(cover >= max(1, len(names) // 3))
    score = score.where(avg.notna().any(axis=1), np.nan)

    sd = score.std(axis=1).replace(0.0, np.nan)
    out = score.sub(score.mean(axis=1), axis=0).div(sd, axis=0)
    return (out, betas) if return_betas else out
