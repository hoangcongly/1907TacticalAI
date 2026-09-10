"""
Sample Weights & Average Uniqueness for Overlapping Labels.
Module B-4-5 (AFML Chapter 4).

Fixes applied:
- #1: Guard division-by-zero in compute_sample_weights
- #2: Sweep-line O(N+T) algorithm replacing O(N×T) loop
"""

import pandas as pd
import numpy as np


def compute_num_concurrent_events(t1: pd.Series, bar_index: pd.Series = None) -> pd.Series:
    """
    Tính số lượng sự kiện (trades) đang diễn ra (active) tại mỗi thời điểm t.
    Ký hiệu: c_t

    Thuật toán Sweep-Line:
    - Với mỗi event [t0, t1]: +1 tại vị trí t0, -1 tại vị trí ngay SAU t1.
    - cumsum cho ra c_t chính xác tại mọi bar.
    - Độ phức tạp: O(N_events * log(T)) thay vì O(N_events * T).

    Args:
        t1: pd.Series, index = t0 (thời điểm bắt đầu), values = t1 (thời điểm kết thúc).
        bar_index: Tập hợp tất cả các mốc thời gian/index. Nếu None, tự tạo từ t1.

    Returns:
        pd.Series đếm số sự kiện chồng lấp tại mỗi thời điểm, index = bar_index.
    """
    t1_valid = t1.dropna()

    if bar_index is None:
        bar_index = t1_valid.index.append(pd.Index(t1_valid.values)).drop_duplicates().sort_values()

    if not bar_index.is_monotonic_increasing:
        raise ValueError("bar_index phải tăng nghiêm ngặt (monotonic increasing).")

    n_bars = len(bar_index)
    delta = np.zeros(n_bars + 1, dtype=int)  # +1 để chứa sentinel cuối

    for t0, t_end in t1_valid.items():
        # Tìm vị trí bắt đầu: bar đầu tiên >= t0
        start_loc = bar_index.searchsorted(t0, side='left')
        # Tìm vị trí kết thúc: bar đầu tiên > t_end (tức là ngay SAU t_end)
        end_loc = bar_index.searchsorted(t_end, side='right')

        if start_loc < n_bars:
            delta[start_loc] += 1
        if end_loc <= n_bars:  # <= vì delta có kích thước n_bars+1
            delta[end_loc] -= 1

    # cumsum trên delta[0:n_bars] cho ra c_t
    c_t_values = np.cumsum(delta[:n_bars])
    c_t = pd.Series(c_t_values, index=bar_index)

    return c_t


def compute_average_uniqueness(t1: pd.Series, c_t: pd.Series) -> pd.Series:
    """
    Tính Average Uniqueness cho mỗi sự kiện (trade).
    Ký hiệu: \\bar{u}_i = mean(1/c_t) cho t trong [t0_i, t1_i].

    Args:
        t1: pd.Series, index = t0, values = t1.
        c_t: pd.Series, số lượng sự kiện chồng lấp tại mỗi bar.

    Returns:
        pd.Series chứa Average Uniqueness cho mỗi sự kiện i.
    """
    out = pd.Series(index=t1.index, dtype=float)
    t1_valid = t1.dropna()

    for t0, t_end in t1_valid.items():
        mask = (c_t.index >= t0) & (c_t.index <= t_end)
        c_t_sub = c_t[mask]

        if len(c_t_sub) > 0:
            # Guard: c_t phải > 0 tại mọi bar có event (nếu = 0, có bug trong compute_num_concurrent_events)
            if (c_t_sub == 0).any():
                raise RuntimeError(
                    f"c_t chứa giá trị 0 trong khoảng [{t0}, {t_end}]. "
                    f"Đây là bug trong compute_num_concurrent_events."
                )
            u_t = 1.0 / c_t_sub
            out.loc[t0] = u_t.mean()
        else:
            out.loc[t0] = 0.0

    return out


def compute_sample_weights(t1: pd.Series, c_t: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Tính Sample Weights dựa trên Uniqueness và Absolute Returns.
    w_i = |R_i| * \\bar{u}_i

    FIX #1: Guard chia cho 0 khi tất cả returns = 0 hoặc tất cả u_i = 0.

    Args:
        t1: pd.Series, index = t0, values = t1.
        c_t: pd.Series, số lượng sự kiện chồng lấp.
        returns: pd.Series lợi nhuận của mỗi sự kiện.

    Returns:
        pd.Series trọng số w_i (đã chuẩn hóa tổng = n_samples).
    """
    u_i = compute_average_uniqueness(t1, c_t)
    w_i = returns.abs() * u_i

    # FIX #1: Guard division by zero
    total = w_i.sum()
    if total == 0 or np.isnan(total):
        # Fallback: trọng số đều 1.0 (không ưu tiên mẫu nào)
        w_i = pd.Series(1.0, index=w_i.index)
    else:
        w_i = w_i * (w_i.shape[0] / total)

    return w_i
