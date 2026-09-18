"""
Gộp tín hiệu THÍCH ỨNG theo cửa sổ trượt — quyết định trọng số bằng dữ liệu QUÁ KHỨ.

VẤN ĐỀ MÀ MODULE NÀY GIẢI:

Chạy thư viện tín hiệu trên tập train cho ra một sự thật khó chịu: một số tín hiệu
hoạt động NGƯỢC với giả thuyết kinh tế nêu trước. Cụ thể, trên perp crypto:

    họ "volatility" (bán tài sản biến động mạnh / lệch dương — hiệu ứng "vé số"
    kinh điển ở cổ phiếu) cho chênh lệch decile ÂM: chính nhóm biến động MẠNH NHẤT
    mới là nhóm thắng (+0.176%/ngày ở Q1, mọi nhóm còn lại đều âm).

Có hai cách xử lý, và chỉ một cách là trung thực:

  (a) SAI — nhìn kết quả rồi đảo dấu bằng tay. Việc này biến train thành một tập
      đã dùng để chọn, và mọi Sharpe sau đó là ảo. Đây chính là data snooping ở
      dạng tinh vi nhất, vì nó luôn có vẻ "hợp lý sau khi biết đáp án".

  (b) ĐÚNG — để HỆ THỐNG tự học dấu và độ lớn từ dữ liệu quá khứ tại mỗi thời điểm.
      Nếu tại tháng 3/2022 dữ liệu trước đó nói họ volatility có dấu âm, hệ thống
      dùng dấu âm — và ta đo được kết quả của quyết định đó trên phần dữ liệu nó
      chưa nhìn. Không có bàn tay nào can thiệp.

Module này làm (b). Đổi lại, nó thêm một rủi ro mới — ĐUỔI THEO NHIỄU — nên có ba
lớp phòng vệ:

  1. NGƯỠNG t-stat: trọng số bằng 0 cho tới khi bằng chứng vượt ngưỡng.
  2. CO VỀ 0 kiểu James-Stein: trọng số bị co theo lượng bằng chứng, không nhảy
     thẳng từ 0 lên toàn phần.
  3. TRẦN THAY ĐỔI: trọng số không được đổi quá `max_step` mỗi kỳ, chống việc danh
     mục lộn nhào khi ước lượng dao động.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CombinerSpec",
    "factor_returns",
    "adaptive_weights",
    "combine_adaptive",
]


@dataclass
class CombinerSpec:
    """Tham số của tầng gộp thích ứng."""

    lookback: int = 250           # số kỳ dùng để ước lượng hiệu quả từng họ
    min_periods: int = 90         # chưa đủ bằng chứng thì dùng trọng số đều
    t_threshold: float = 1.0      # dưới ngưỡng này coi như không có bằng chứng
    max_abs_weight: float = 0.45  # trần tỷ trọng một họ (chống dồn hết vào một cược)
    max_step: float = 0.10        # thay đổi trọng số tối đa mỗi kỳ
    allow_sign_flip: bool = True  # cho phép học dấu ngược với giả thuyết ban đầu
    shrink: float = 1.0           # cường độ co; 0 = không co, 1 = co chuẩn
    warm_equal: bool = True       # CHƯA ĐỦ LỊCH SỬ -> trọng số đều thay vì bỏ trống
    #: KHI KHÔNG HỌ NÀO CÓ BẰNG CHỨNG (đủ lịch sử nhưng mọi |t| < ngưỡng) thì làm gì.
    #:
    #: Bản đầu dùng chung `warm_equal` cho cả hai tình huống. Chúng KHÁC NHAU về bản
    #: chất: "chưa đủ dữ liệu để biết" là vô tri, còn "đủ dữ liệu và dữ liệu nói không
    #: có gì" là một kết luận. Trả lời cả hai bằng "giao dịch đều tay, gross đầy đủ"
    #: nghĩa là ở tình huống thứ hai hệ thống đặt cược hết cỡ đúng lúc nó vừa tự kết
    #: luận rằng không tín hiệu nào chứng minh được điều gì.
    #:
    #: ĐÃ ĐO (14/09/2026): với 26 tín hiệu, nhánh này chạy 0/799 kỳ — luôn có ít nhất
    #: một họ vượt t = 2,0. Nên đây là RỦI RO TIỀM ẨN, không phải lỗi đang hoạt động,
    #: và mặc định giữ nguyên hành vi cũ để không đổi kết quả đã kiểm định. Nhưng nó
    #: sẽ chạy nếu ai đó thu hẹp thư viện tín hiệu hoặc nâng `t_threshold` — và lúc đó
    #: nó sẽ chạy IM LẶNG. Vì vậy nhánh này nay ghi cảnh báo mỗi khi kích hoạt.
    collapse_equal: bool = True


def factor_returns(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    top_frac: float = 0.10,
    min_names: int = 10,
) -> pd.Series:
    """
    Chuỗi lợi suất của "danh mục nhân tố" ứng với MỘT tín hiệu.

    Long đều nhóm `top_frac` cao nhất, short đều nhóm thấp nhất, gross = 1.0. Đây
    chính là đại lượng danh mục thật kiếm được — nên nó, chứ không phải IC, là
    thứ dùng để chấm điểm tín hiệu (xem `ic_analysis.quantile_returns` để biết vì
    sao hai đại lượng này có thể ngược dấu).

    Chuỗi trả về đã căn theo mốc t = lợi suất kiếm được TỪ t ĐẾN t+1, nên khi dùng
    để tính trọng số cho kỳ t phải cắt tới t-1 (hàm `adaptive_weights` lo việc này).
    """
    fwd = close.shift(-1) / close - 1.0
    out = {}
    for ts in signal.index.intersection(fwd.index):
        s = signal.loc[ts]
        r = fwd.loc[ts].reindex(signal.columns)
        m = s.notna() & r.notna()
        n = int(m.sum())
        if n < min_names:
            continue
        k = max(1, int(round(n * top_frac)))
        order = s[m].sort_values()
        rr = r[m]
        # gross = 1.0 -> mỗi chân 0.5, chia đều trong chân.
        out[ts] = float(rr[order.index[-k:]].mean() - rr[order.index[:k]].mean()) * 0.5
    return pd.Series(out, dtype=np.float64).sort_index()


def _score(window: np.ndarray, t_threshold: float, shrink: float,
           allow_flip: bool) -> float:
    """
    Điểm số của một họ từ cửa sổ lợi suất quá khứ.

    Dùng t-stat làm thước đo bằng chứng rồi CO theo kiểu soft-threshold:

        score = sign(t) * max(0, |t| - ngưỡng) / |t|   nhân với  Sharpe của cửa sổ

    Ý nghĩa: bằng chứng vừa đủ vượt ngưỡng chỉ được một phần nhỏ trọng số; bằng
    chứng áp đảo mới được gần như toàn phần. Đây là ước lượng co (shrinkage
    estimator) chứ không phải bộ lọc bật/tắt — bật/tắt gây nhảy trọng số và
    turnover không cần thiết.
    """
    w = window[np.isfinite(window)]
    if len(w) < 20:
        return 0.0
    mu, sd = float(w.mean()), float(w.std(ddof=1))
    if sd <= 1e-15:
        return 0.0

    t = mu / (sd / np.sqrt(len(w)))
    if not allow_flip and t < 0:
        return 0.0

    excess = abs(t) - t_threshold
    if excess <= 0:
        return 0.0
    factor = (excess / abs(t)) ** shrink if shrink > 0 else 1.0
    sharpe = mu / sd
    return float(np.sign(t) * abs(sharpe) * factor)


def adaptive_weights(
    fac_rets: Dict[str, pd.Series],
    spec: Optional[CombinerSpec] = None,
    return_diagnostics: bool = False,
):
    """
    Trọng số từng họ theo thời gian, tính HOÀN TOÀN từ dữ liệu quá khứ.

    Tại mốc t, cửa sổ dùng để chấm điểm là `[t-lookback, t)` — KHÔNG bao gồm t,
    vì lợi suất tại t là thứ trọng số tại t sẽ đi kiếm. Bao gồm nó là nhìn trước
    một nhịp, và một nhịp cũng đủ để Sharpe backtest tăng gấp đôi một cách giả tạo.
    """
    spec = spec or CombinerSpec()
    names = list(fac_rets)
    df = pd.DataFrame(fac_rets).sort_index()
    idx = df.index
    arr = df.to_numpy()

    weights = np.zeros((len(idx), len(names)))
    prev = np.zeros(len(names))
    equal = np.full(len(names), 1.0 / len(names))
    # Chẩn đoán: nhánh nào đã chạy, và TỔNG BẰNG CHỨNG thô trước khi chuẩn hoá.
    # `evidence` là đại lượng bị chuẩn hoá mất ở dòng `target = raw / total` — tức là
    # thông tin về ĐỘ MẠNH của bằng chứng bị vứt đi và chỉ giữ lại TỶ LỆ giữa các họ.
    # Ghi lại để đo xem sự mất mát đó có đáng kể không.
    branch = np.empty(len(idx), dtype=object)
    evidence = np.zeros(len(idx))
    n_active = np.zeros(len(idx))
    n_collapse = 0

    for i in range(len(idx)):
        lo = max(0, i - spec.lookback)
        window = arr[lo:i]  # LOẠI TRỪ hàng i

        if len(window) < spec.min_periods:
            target = equal.copy() if spec.warm_equal else np.zeros(len(names))
            branch[i] = "warmup"
        else:
            raw = np.array([_score(window[:, j], spec.t_threshold, spec.shrink,
                                   spec.allow_sign_flip) for j in range(len(names))])
            total = np.abs(raw).sum()
            evidence[i] = float(total)
            n_active[i] = float((np.abs(raw) > 1e-12).sum())
            if total <= 1e-12:
                target = equal.copy() if spec.collapse_equal else np.zeros(len(names))
                branch[i] = "collapse"
                n_collapse += 1
            else:
                branch[i] = "ok"
                target = raw / total
                over = np.abs(target) > spec.max_abs_weight
                if over.any():
                    target[over] = np.sign(target[over]) * spec.max_abs_weight
                    s = np.abs(target).sum()
                    if s > 1e-12:
                        target = target / s

        # Trần thay đổi mỗi kỳ.
        step = np.clip(target - prev, -spec.max_step, spec.max_step)
        cur = prev + step

        # [FIX F36] Chuẩn hoá về gross CỦA MỤC TIÊU, không phải về 1.0 vô điều kiện.
        #
        # Bản cũ luôn chia cho `|cur|.sum()`, tức luôn ép gross = 1.0. Hệ quả: khi mục
        # tiêu là "đứng ngoài" (vector 0 — điều mà `warm_equal=False` và
        # `collapse_equal=False` tồn tại để diễn đạt), việc chuẩn hoá KÉO NGAY trọng số
        # trở lại gross đầy đủ. Hai cờ đó vì thế là cờ GIẢ: chúng đổi tỷ lệ giữa các họ
        # nhưng không bao giờ đổi được quy mô, nên ý định "không có bằng chứng thì
        # không đặt cược" bị vô hiệu hoá trong im lặng.
        #
        # Test `test_collapse_equal_tat_thi_danh_muc_rong` đã phát hiện đúng điều này:
        # đặt `collapse_equal=False` vẫn cho gross = 1.0.
        #
        # KHÔNG ĐỔI HÀNH VI Ở TRẠNG THÁI BÌNH THƯỜNG: khi có bằng chứng, `target` luôn
        # có gross = 1.0 (nó là `raw/total`, hoặc `equal`), nên phép chia dưới đây cho
        # ra đúng kết quả cũ tới từng chữ số.
        tgt_gross = float(np.abs(target).sum())
        s = np.abs(cur).sum()
        if s > 1e-12 and tgt_gross > 1e-12:
            cur = cur * (tgt_gross / s)
        # tgt_gross = 0 -> để `cur` tự đi về 0 theo `max_step`; đứng ngoài phải đứng
        # ngoài thật, và đi ra từ từ để không tạo một cú xả toàn danh mục trong một kỳ.
        weights[i] = cur
        prev = cur

    if n_collapse:
        # KHÔNG im lặng. Đây là trạng thái "không tín hiệu nào chứng minh được gì" và
        # hệ thống vẫn đang giao dịch ở gross đầy đủ. Người vận hành phải biết.
        logger.warning(
            "[BẰNG CHỨNG SỤP ĐỔ] %d/%d kỳ không có họ tín hiệu nào vượt ngưỡng t=%.1f; "
            "đang dùng %s. Xem `CombinerSpec.collapse_equal`.",
            n_collapse, len(idx), spec.t_threshold,
            "trọng số ĐỀU ở gross đầy đủ" if spec.collapse_equal else "danh mục RỖNG")

    W = pd.DataFrame(weights, index=idx, columns=names)
    if not return_diagnostics:
        return W
    diag = pd.DataFrame({"branch": branch, "evidence": evidence,
                         "n_active": n_active}, index=idx)
    return W, diag


def combine_adaptive(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    spec: Optional[CombinerSpec] = None,
    top_frac: float = 0.10,
    return_diagnostics: bool = False,
):
    """
    Gộp nhiều tín hiệu thành MỘT điểm số tổng hợp bằng trọng số thích ứng.

    Trả về DataFrame điểm số cùng shape với các tín hiệu đầu vào; nếu
    `return_diagnostics=True` thì trả thêm bảng trọng số theo thời gian để soi
    xem hệ thống đã học được gì (và khi nào nó đảo dấu một họ).

    Ô thiếu dữ liệu được GIỮ NaN chứ không điền 0: điền 0 nghĩa là coi "không biết"
    thành "trung tính", khiến một cặp thiếu tín hiệu vẫn lọt vào danh mục dựa trên
    thông tin không tồn tại.
    """
    spec = spec or CombinerSpec()
    names = list(signals)

    fac = {n: factor_returns(signals[n], close, top_frac=top_frac) for n in names}
    W = adaptive_weights(fac, spec)

    idx = close.index
    W = W.reindex(idx).ffill().fillna(0.0)

    total = pd.DataFrame(0.0, index=idx, columns=close.columns)
    count = pd.DataFrame(0.0, index=idx, columns=close.columns)

    for n in names:
        sig = signals[n].reindex(index=idx, columns=close.columns)
        w = W[n]
        contrib = sig.mul(w, axis=0)
        total = total.add(contrib.fillna(0.0), fill_value=0.0)
        count = count.add(sig.notna().astype(float).mul(w.abs(), axis=0), fill_value=0.0)

    combined = total.where(count > 1e-9)
    # Chuẩn hoá lại theo hàng để độ lớn điểm số không phụ thuộc số họ đang hoạt động.
    sd = combined.std(axis=1).replace(0.0, np.nan)
    combined = combined.sub(combined.mean(axis=1), axis=0).div(sd, axis=0)

    if return_diagnostics:
        return combined, W, pd.DataFrame(fac)
    return combined
