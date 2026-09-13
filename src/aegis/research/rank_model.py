"""
Gộp tín hiệu bằng HỌC-ĐỂ-XẾP-HẠNG (LambdaRank) — tối ưu trực tiếp thứ hạng.

NGUỒN: "LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction"
(arXiv 2605.00501). Ý tưởng: dự báo lợi suất bằng hồi quy rồi mới xếp hạng là tối
ưu SAI hàm mục tiêu. Danh mục long-short không quan tâm ta dự báo +3% hay +5%; nó
chỉ quan tâm THỨ TỰ. Vậy hãy tối ưu thẳng thứ tự.

VÌ SAO PHÙ HỢP ĐẶC BIỆT VỚI HỆ THỐNG NÀY:

Nghiên cứu trên chính dữ liệu này đã cho thấy IC toàn mặt cắt ngang và chênh lệch
decile có thể NGƯỢC DẤU nhau, vì quan hệ tín hiệu-lợi suất không đơn điệu (xem
`ic_analysis.quantile_returns`). Danh mục chỉ giao dịch hai ĐUÔI.

LambdaRank hợp với điều đó hơn hồi quy thường: hàm mục tiêu của nó chiết khấu theo
VỊ TRÍ trong bảng xếp hạng, nên nó dồn sức học cho phần đầu/cuối bảng — đúng chỗ ta
thực sự đặt lệnh — thay vì trải đều nỗ lực cho cả phần giữa mà ta không bao giờ chạm.

BA CHỐT CHẶN CHỐNG RÒ RỈ (mỗi cái đều có test riêng):
  1. HUẤN LUYỆN TIẾN VỀ PHÍA TRƯỚC: model dùng tại kỳ t chỉ học từ dữ liệu < t.
  2. PURGE: bỏ `purge` kỳ cuối của tập train, vì nhãn của chúng là lợi suất tương
     lai chồng lấn sang vùng dự báo.
  3. NHÃN THEO TỪNG KỲ: nhãn là phân vị lợi suất TRONG CÙNG một mặt cắt ngang,
     không phải lợi suất tuyệt đối — nếu không, model chỉ học "kỳ nào thị trường
     tăng", một thông tin vô dụng cho danh mục trung lập.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["RankModelSpec", "build_dataset", "combine_lambdarank"]


@dataclass
class RankModelSpec:
    """Tham số tầng xếp hạng."""

    n_labels: int = 8              # số bậc nhãn (phân vị lợi suất trong kỳ)
    train_periods: int = 400       # số kỳ dùng để huấn luyện mỗi lần
    retrain_every: int = 30        # cứ bao nhiêu kỳ thì huấn luyện lại
    purge: int = 2                 # số kỳ cuối của train bị loại bỏ
    min_train_periods: int = 150   # chưa đủ thì chưa dự báo
    min_names: int = 15
    num_leaves: int = 15
    n_estimators: int = 120
    learning_rate: float = 0.05
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 5.0
    seed: int = 7
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        # `train_periods` là ĐỘ DÀI cửa sổ huấn luyện; `min_train_periods` là độ dài
        # TỐI THIỂU chấp nhận được. Đặt cái sau lớn hơn cái trước thì điều kiện huấn
        # luyện không bao giờ thoả, hàm trả về toàn NaN và KHÔNG BÁO GÌ. Cấu hình
        # tự mâu thuẫn phải chết ngay tại chỗ, không được biến thành im lặng.
        if self.train_periods < self.min_train_periods:
            raise ValueError(
                f"train_periods ({self.train_periods}) phải >= min_train_periods "
                f"({self.min_train_periods}), nếu không model không bao giờ được huấn luyện")
        if self.purge < 0:
            raise ValueError(f"purge phải >= 0, nhận {self.purge}")
        if self.n_labels < 2:
            raise ValueError(f"n_labels phải >= 2, nhận {self.n_labels}")


def build_dataset(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: Optional[RankModelSpec] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Dựng bảng dữ liệu phẳng: mỗi hàng là (thời điểm, tài sản).

    Trả về `(X, y_label, timestamps, feature_names)` trong đó `y_label` là bậc phân
    vị của lợi suất TƯƠNG LAI trong cùng mặt cắt ngang — cao = thuộc nhóm tăng mạnh
    nhất kỳ đó.
    """
    spec = spec or RankModelSpec()
    names = list(signals)
    fwd = close.shift(-1) / close - 1.0

    Xs, ys, ts_list = [], [], []
    for ts in close.index:
        cols = close.columns
        S = np.column_stack([signals[n].reindex(index=[ts], columns=cols).to_numpy()[0]
                             for n in names])
        y = fwd.loc[ts].reindex(cols).to_numpy(dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(S).all(axis=1)
        if ok.sum() < spec.min_names:
            continue

        yy = y[ok]
        # Nhãn = bậc phân vị TRONG KỲ. Đây là chỗ tính trung lập được đưa vào nhãn:
        # model không thể ăn điểm bằng cách đoán hướng chung của thị trường.
        ranks = pd.Series(yy).rank(method="first", pct=True).to_numpy()
        labels = np.clip((ranks * spec.n_labels).astype(int), 0, spec.n_labels - 1)

        Xs.append(S[ok])
        ys.append(labels)
        ts_list.append(np.full(ok.sum(), ts, dtype=np.int64))

    if not Xs:
        return pd.DataFrame(columns=names), pd.Series(dtype=int), pd.Series(dtype=np.int64), names

    X = pd.DataFrame(np.vstack(Xs), columns=names)
    return X, pd.Series(np.concatenate(ys)), pd.Series(np.concatenate(ts_list)), names


def combine_lambdarank(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: Optional[RankModelSpec] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Điểm số tổng hợp do model xếp hạng sinh ra, huấn luyện tiến về phía trước.

    Trả về DataFrame cùng shape với `close`; kỳ chưa đủ dữ liệu huấn luyện để NaN.
    """
    import lightgbm as lgb

    spec = spec or RankModelSpec()
    X, y, ts, names = build_dataset(signals, close, spec)
    if X.empty:
        return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    uniq = np.array(sorted(pd.unique(ts)))
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    model = None
    trained_at = -10**9

    base = dict(objective="lambdarank", metric="ndcg",
                num_leaves=spec.num_leaves, n_estimators=spec.n_estimators,
                learning_rate=spec.learning_rate,
                min_child_samples=spec.min_child_samples,
                subsample=spec.subsample, subsample_freq=1,
                colsample_bytree=spec.colsample_bytree,
                reg_lambda=spec.reg_lambda, random_state=spec.seed,
                n_jobs=-1, verbose=-1,
                label_gain=list(range(spec.n_labels)))
    base.update(spec.params)

    ts_arr = ts.to_numpy()
    for i, t in enumerate(uniq):
        if i < spec.min_train_periods:
            continue

        if model is None or (i - trained_at) >= spec.retrain_every:
            lo = max(0, i - spec.train_periods)
            # PURGE: nhãn của `purge` kỳ cuối là lợi suất chồng lấn sang vùng dự báo.
            hi = i - spec.purge
            if hi - lo < spec.min_train_periods:
                continue
            window = uniq[lo:hi]
            mask = np.isin(ts_arr, window)
            if mask.sum() < spec.min_names * spec.min_train_periods // 4:
                continue

            Xtr, ytr, ttr = X[mask], y[mask].to_numpy(), ts_arr[mask]
            order = np.argsort(ttr, kind="stable")
            Xtr, ytr, ttr = Xtr.iloc[order], ytr[order], ttr[order]
            groups = pd.Series(ttr).value_counts(sort=False).reindex(
                pd.unique(ttr)).to_numpy()

            try:
                model = lgb.LGBMRanker(**base)
                model.fit(Xtr, ytr, group=groups)
                trained_at = i
                if verbose:
                    print(f"  huấn luyện lại tại kỳ {i}/{len(uniq)} "
                          f"({len(window)} kỳ, {mask.sum()} mẫu)", flush=True)
            except Exception as exc:
                if verbose:
                    print(f"  huấn luyện lỗi tại kỳ {i}: {exc}", flush=True)
                continue

        if model is None:
            continue

        # DỰ BÁO: tập tài sản được chấm điểm chỉ do ĐỘ PHỦ TÍN HIỆU quyết định.
        # Bản đầu của hàm này lọc thêm theo `np.isfinite(lợi_suất_tương_lai)` — tiện
        # vì dùng lại được ma trận đã dựng cho huấn luyện, nhưng đó là RÒ RỈ: live
        # không biết lợi suất tương lai, nên universe live sẽ khác universe backtest,
        # và backtest âm thầm loại bỏ đúng những cặp bị ngừng giao dịch/huỷ niêm yết.
        cols = close.columns
        S = np.column_stack([signals[n].reindex(index=[t], columns=cols).to_numpy()[0]
                             for n in names])
        ok = np.isfinite(S).all(axis=1)
        if ok.sum() < spec.min_names:
            continue
        out.loc[t, cols[ok]] = model.predict(pd.DataFrame(S[ok], columns=names))

    sd = out.std(axis=1).replace(0.0, np.nan)
    return out.sub(out.mean(axis=1), axis=0).div(sd, axis=0)
