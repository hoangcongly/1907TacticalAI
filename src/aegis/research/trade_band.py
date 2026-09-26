"""
Dải không giao dịch + cân trung lập — bản sao của `build_rebalance_plan` trên TRỌNG SỐ.

VÌ SAO CẦN: live không bao giờ kéo sổ về đúng trọng số mục tiêu. `build_rebalance_plan`
(execution/portfolio_rebalancer.py) bỏ mọi điều chỉnh nhỏ hơn `no_trade_band` × vị thế
đích (mặc định 20%), bỏ mọi lệnh MỞ/TĂNG dưới min notional, rồi nhận lại một phần lệnh bị
bỏ nếu sổ lệch trung lập quá trần (F42). Mọi backtest trước đây lại tái cân bằng TOÀN
PHẦN mỗi kỳ, tức đo một hệ thống giao dịch nhiều lệnh hơn live — và không đo được câu
hỏi "nới dải thì bớt bao nhiêu lệnh, mất bao nhiêu lời".

Đơn vị: trọng số = notional / (vốn × đòn bẩy), tức đúng `norm` của live. Khi đó
`min_trade = min_notional / (vốn × đòn bẩy)`. Test đối chiếu thẳng với
`build_rebalance_plan` ở `tests/research/test_trade_band.py`.
"""
from typing import Tuple

import numpy as np

__all__ = ["plan_band_trades", "smooth_signal"]


def plan_band_trades(
    held: np.ndarray,
    target: np.ndarray,
    band: float,
    min_trade: float,
    tolerance: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trả về (trọng số sau giao dịch, mặt nạ các cặp CÓ đặt lệnh), theo đúng thứ tự của live:

    1. Đóng hẳn (đích 0, đang giữ) — luôn giao dịch.
    2. |đích − đang giữ| < band × |đích| -> bỏ, giữ nguyên (nhớ lại để cân trung lập).
    3. Lệnh MỞ/TĂNG có |Δ| < min_trade -> bỏ, giữ nguyên.
    4. Còn lại -> giao dịch về đích.
    5. [F42] Sổ lệch quá `tolerance` thì nhận lại các lệnh bị dải bỏ ở phía kéo net về
       0, lệnh lớn nhất trước, mỗi lệnh phải tự vượt `min_trade` và phải làm sổ BỚT lệch.
    """
    held = np.asarray(held, dtype=float)
    target = np.nan_to_num(np.asarray(target, dtype=float))
    new = held.copy()
    traded = np.zeros(len(held), dtype=bool)
    band_skipped = []
    gross = net = 0.0

    for i in range(len(held)):
        h, t = held[i], target[i]
        d = t - h
        if d == 0.0:
            gross += abs(t)
            net += t
            continue
        full_close = h != 0.0 and t == 0.0
        if not full_close:
            ref = abs(t) if t != 0.0 else abs(h)
            if ref > 0 and abs(d) < band * ref:
                if t != 0.0:
                    gross += abs(h)
                    net += h
                    band_skipped.append(i)
                continue
        is_closing = full_close or (abs(t) < abs(h) and h * t >= 0)
        if not is_closing and abs(d) < min_trade:
            gross += abs(h)          # [F55] vị thế vẫn nằm trong sổ ở khối lượng cũ
            net += h
            continue
        new[i] = t
        traded[i] = True
        gross += abs(t)
        net += t

    if band_skipped and gross > 0 and abs(net) / gross > tolerance:
        cands = [i for i in band_skipped if (target[i] - held[i]) * net < 0]
        cands.sort(key=lambda i: -abs(target[i] - held[i]))
        for i in cands:
            if abs(net) / gross <= tolerance:
                break
            d = target[i] - held[i]
            if abs(d) < min_trade or abs(net + d) >= abs(net):
                continue
            new[i] = target[i]
            traded[i] = True
            gross += abs(target[i]) - abs(held[i])
            net += d
    return new, traded


def smooth_signal(sig, halflife: float):
    """
    Làm mượt điểm tổng hợp theo thời gian (EWM trên lưới tái cân bằng), NHÂN QUẢ.

    Gârleanu & Pedersen (2013): khi có chi phí giao dịch, nên nhắm vào trung bình có
    trọng số của mục tiêu hiện tại và tương lai kỳ vọng — thực hành là làm chậm tín hiệu
    để xếp hạng bớt nhảy, bớt đổi đồng mỗi kỳ. Cặp không có điểm ở kỳ hiện tại giữ NaN
    (không được giao dịch bằng điểm cũ của nó).

    ⚠️ CHỈ có ở nghiên cứu. Live chưa làm mượt — thắng ở đây thì phải nối vào
    `_compute_target_weights_v3` kèm test parity trước khi dùng.
    """
    if not halflife:
        return sig
    return sig.ewm(halflife=halflife, ignore_na=True).mean().where(sig.notna())
