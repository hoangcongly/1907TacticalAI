"""
Dynamic HMM Triple-Barrier Labeling & compute_sl_initial đối xứng Long/Short.
Module C.2 (Tầng 1 Dán nhãn thống kê cho Meta-Labeler).
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def compute_dynamic_barriers(
    entry_price: float,
    side: int,
    p_trend: float,
    p_chop: float,
    sigma: float,
    is_toxic: bool = False,
    c_trade: float = 0.0004,
) -> Tuple[float, float]:
    """
    Tính rào cản Chốt lời (TP) và Cắt lỗ (SL) động theo xác suất regime HMM:
    m_pt = p_chop * 1.5 + p_trend * 3.0
    m_sl = p_chop * 1.5 + p_trend * 2.0
    c_trade_adj = c_trade * (1 + 0.5 * is_toxic)
    
    Trả về: (tp_price, sl_price)
    """
    p0 = float(entry_price)
    sig = max(float(sigma), 1e-6)
    pt = min(max(float(p_trend), 0.0), 1.0)
    pc = min(max(float(p_chop), 0.0), 1.0)

    m_pt = pc * 1.5 + pt * 3.0
    m_sl = pc * 1.5 + pt * 2.0
    c_adj = float(c_trade) * (1.5 if is_toxic else 1.0)

    if side > 0:
        tp = p0 * (1.0 + m_pt * sig)
        sl = p0 * (1.0 - m_sl * sig - c_adj)
    else:
        tp = p0 * (1.0 - m_pt * sig)
        sl = p0 * (1.0 + m_sl * sig + c_adj)

    return float(tp), float(sl)


def apply_triple_barrier_single_event(
    entry_idx: int,
    entry_price: float,
    side: int,
    tp_price: float,
    sl_price: float,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    future_closes: np.ndarray,
) -> Dict[str, Any]:
    """
    Kiểm tra sự kiện chạm rào cản nào đầu tiên (TP, SL hay TIME stop):
    - Causal Pre-Slice: future_bars bắt đầu từ entry_idx + 1.
    - Meta-labeling convention:
      + Hit TP: label = 1
      + Hit SL: label = 0
      + Hit TIME: label = 1 nếu lợi nhuận hướng theo side > 0, ngược lại 0.
    """
    n_lookahead = len(future_highs)
    p0 = float(entry_price)
    
    if n_lookahead == 0:
        # Biên cắt ngay lập tức
        return {
            "entry_idx": entry_idx,
            "exit_idx": entry_idx + 1,
            "hit_barrier": "TIME",
            "exit_price": p0,
            "realized_return": 0.0,
            "label": 0,
        }

    for k in range(n_lookahead):
        curr_high = future_highs[k]
        curr_low = future_lows[k]

        if side > 0:
            # Long position
            hit_sl = curr_low <= sl_price
            hit_tp = curr_high >= tp_price

            if hit_sl and hit_tp:
                # Nếu cùng chạm trong 1 bar, giả định xấu nhất: SL chạm trước
                return {
                    "entry_idx": entry_idx,
                    "exit_idx": entry_idx + 1 + k,
                    "hit_barrier": "SL",
                    "exit_price": sl_price,
                    "realized_return": (sl_price - p0) / p0,
                    "label": 0,
                }
            elif hit_sl:
                return {
                    "entry_idx": entry_idx,
                    "exit_idx": entry_idx + 1 + k,
                    "hit_barrier": "SL",
                    "exit_price": sl_price,
                    "realized_return": (sl_price - p0) / p0,
                    "label": 0,
                }
            elif hit_tp:
                return {
                    "entry_idx": entry_idx,
                    "exit_idx": entry_idx + 1 + k,
                    "hit_barrier": "TP",
                    "exit_price": tp_price,
                    "realized_return": (tp_price - p0) / p0,
                    "label": 1,
                }
        else:
            # Short position
            hit_sl = curr_high >= sl_price
            hit_tp = curr_low <= tp_price

            if hit_sl and hit_tp:
                return {
                    "entry_idx": entry_idx,
                    "exit_idx": entry_idx + 1 + k,
                    "hit_barrier": "SL",
                    "exit_price": sl_price,
                    "realized_return": (p0 - sl_price) / p0,
                    "label": 0,
                }
            elif hit_sl:
                return {
                    "entry_idx": entry_idx,
                    "exit_idx": entry_idx + 1 + k,
                    "hit_barrier": "SL",
                    "exit_price": sl_price,
                    "realized_return": (p0 - sl_price) / p0,
                    "label": 0,
                }
            elif hit_tp:
                return {
                    "entry_idx": entry_idx,
                    "exit_idx": entry_idx + 1 + k,
                    "hit_barrier": "TP",
                    "exit_price": tp_price,
                    "realized_return": (p0 - tp_price) / p0,
                    "label": 1,
                }

    # Vertical Barrier (Time Stop) reached
    last_idx = n_lookahead - 1
    exit_price = float(future_closes[last_idx])
    ret = (exit_price - p0) / p0 if side > 0 else (p0 - exit_price) / p0
    label = 1 if ret > 0.0 else 0

    return {
        "entry_idx": entry_idx,
        "exit_idx": entry_idx + 1 + last_idx,
        "hit_barrier": "TIME",
        "exit_price": exit_price,
        "realized_return": ret,
        "label": label,
    }


def generate_meta_labels_triple_barrier(
    events_idx: np.ndarray,
    sides: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    p_trend: np.ndarray,
    p_chop: np.ndarray,
    sigmas: np.ndarray,
    is_toxic: np.ndarray,
    t_window: int = 20,
    c_trade: float = 0.0004,
) -> pd.DataFrame:
    """
    Sinh tập nhãn Meta-Labeling hoàn chỉnh cho danh sách sự kiện:
    Trả về pd.DataFrame gồm các cột:
    - 't0': chỉ số entry_idx
    - 't1': chỉ số exit_idx (dùng cho sample_weights và purged k-fold)
    - 'side': chiều giao dịch nguyên bản
    - 'label': nhãn nhị phân y in {0, 1}
    - 'realized_return': tỷ suất lợi nhuận thực tế
    - 'hit_barrier': rào cản chạm ('TP', 'SL', 'TIME')
    """
    records = []
    n_total = len(closes)

    for idx, side in zip(events_idx, sides):
        if idx >= n_total - 1:
            continue

        p0 = float(closes[idx])
        sig = float(sigmas[idx]) if not np.isnan(sigmas[idx]) else 0.01
        pt = float(p_trend[idx]) if not np.isnan(p_trend[idx]) else 0.5
        pc = float(p_chop[idx]) if not np.isnan(p_chop[idx]) else 0.5
        toxic = bool(is_toxic[idx])

        tp, sl = compute_dynamic_barriers(p0, side, pt, pc, sig, is_toxic=toxic, c_trade=c_trade)

        slice_end = min(idx + 1 + t_window, n_total)
        future_highs = highs[idx + 1 : slice_end]
        future_lows = lows[idx + 1 : slice_end]
        future_closes = closes[idx + 1 : slice_end]

        res = apply_triple_barrier_single_event(
            entry_idx=idx,
            entry_price=p0,
            side=side,
            tp_price=tp,
            sl_price=sl,
            future_highs=future_highs,
            future_lows=future_lows,
            future_closes=future_closes,
        )

        records.append({
            "t0": int(res["entry_idx"]),
            "t1": int(res["exit_idx"]),
            "side": int(side),
            "label": int(res["label"]),
            "realized_return": float(res["realized_return"]),
            "hit_barrier": str(res["hit_barrier"]),
            "exit_price": float(res["exit_price"]),
        })

    return pd.DataFrame(records)
