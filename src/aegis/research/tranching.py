"""
CHIA LÔ TÁI CÂN BẰNG + NEO LƯỚI Ở NẾN MỚI NHẤT — một nguồn sự thật cho research và live.

═══════════════════════════════════════════════════════════════════════════════
VẤN ĐỀ 1 — TIMING LUCK (replay 24/09/2026)
═══════════════════════════════════════════════════════════════════════════════
Cùng hệ thống, cùng 28 ngày, chỉ khác GIỜ tái cân bằng: kết quả chạy từ −14% tới +231%
(61 giờ bắt đầu, đã vá funding). Giờ của daemon rơi vào nhóm 5% xấu nhất (−7,3%) trong
khi trung vị là +45%. Một sổ đổi TOÀN BỘ mỗi 72h là MỘT lần rút thăm giờ tái cân bằng,
và ở 5x lần rút thăm đó quyết định gần hết kết quả của cả tháng.

Lời giải chuẩn trong tài liệu là CHIA LÔ (Hoffstein, Faber & Braun 2020, "Rebalance
Timing Luck"; Blitz et al. 2010 đề xuất danh mục chồng lớp): chia vốn thành K lô, mỗi
lô vẫn giữ 72h như chiến lược đã kiểm định, nhưng các lô lệch nhau 72/K giờ. Sổ thật là
TRUNG BÌNH K lô, nên kết quả tiến về trung bình các giờ bắt đầu thay vì một giờ may
hay rủi. Kỳ vọng gần như không đổi; phương sai do chọn giờ giảm mạnh.

═══════════════════════════════════════════════════════════════════════════════
VẤN ĐỀ 2 — [FIX F53] LIVE GIAO DỊCH TÍN HIỆU CŨ TỚI 68 GIỜ
═══════════════════════════════════════════════════════════════════════════════
Bản cũ của `xs_live_pipeline._compute_target_weights_v3` dựng lưới
`close.index[::18]` — neo ở ĐẦU panel — rồi lấy trọng số ở mốc CUỐI của lưới đó. Mốc
cuối có thể cách nến mới nhất tới 17 nến 4h = 68 giờ, tuỳ vị trí daemon thức dậy so với
lưới. Hàm lại trả về mốc của NẾN MỚI NHẤT, nên cổng STALE_DATA báo "dữ liệu tươi"
trong khi tín hiệu đã cũ gần 3 ngày. Research thì luôn khớp lệnh ĐÚNG mốc với tín hiệu
của chính mốc đó — hai bên chạy hai chiến lược khác nhau (họ lỗi F3).

Vá: lưới NEO Ở NẾN MỚI NHẤT (`anchored_marks`). Mốc cuối LUÔN là nến mới nhất, nên
trọng số live là trọng số research tại đúng thời điểm khớp lệnh.

═══════════════════════════════════════════════════════════════════════════════
VÌ SAO KHÔNG CẦN LƯU TRẠNG THÁI TỪNG LÔ
═══════════════════════════════════════════════════════════════════════════════
Mọi hàm ở đây là hàm THUẦN của dữ liệu <= t. Đích của lô j là trọng số tại mốc cách
hiện tại j*(72/K) giờ, trên lưới neo ở chính mốc đó — tính lại được bất cứ lúc nào,
và ra ĐÚNG con số lô đó đã nhận khi nó tái cân bằng. Không có tệp trạng thái nào để
lệch, mất, hay hỏng giữa hai lần máy ngủ.

Lưới neo ở mốc a có pha `a mod step`, và `combine_adaptive` trên nó đi qua ĐÚNG các
mốc mà `tranched_weight_panel` dùng cho pha đó — nên đường research (toàn lịch sử) và
đường live (chỉ hàng cuối) trùng nhau tới sai số máy. Test:
`tests/research/test_tranching.py`.
"""

from typing import Dict, List, Optional

import pandas as pd

from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.risk.portfolio import PortfolioSpec, build_weights

__all__ = ["anchored_marks", "tranche_offsets", "target_weights_at",
           "tranched_target_weights", "tranched_weight_panel"]


def tranche_offsets(step: int, n_tranches: int) -> List[int]:
    """Độ lệch (số nến) của từng lô so với lô mới nhất: 0, step/K, 2*step/K, ..."""
    if n_tranches < 1:
        raise ValueError(f"n_tranches phải >= 1, nhận {n_tranches}")
    if step % n_tranches:
        raise ValueError(f"step={step} phải chia hết cho n_tranches={n_tranches} "
                         f"để các lô cách đều nhau")
    gap = step // n_tranches
    return [j * gap for j in range(n_tranches)]


def anchored_marks(index: pd.Index, step: int, anchor_pos: int) -> pd.Index:
    """Các mốc cách nhau `step` nến, mốc CUỐI rơi đúng `index[anchor_pos]`. [FIX F53]"""
    n = len(index)
    if anchor_pos < 0:
        anchor_pos += n
    if not 0 <= anchor_pos < n:
        raise IndexError(f"anchor_pos {anchor_pos} ngoài [0, {n})")
    return index[anchor_pos % step::step][: anchor_pos // step + 1]


def target_weights_at(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    anchor_pos: int,
    step: int,
    combiner: CombinerSpec,
    spec: PortfolioSpec,
    top_frac: float,
    tail_only: bool = True,
) -> pd.Series:
    """
    Trọng số MỘT lô tại mốc `anchor_pos`, trên lưới neo ở chính mốc đó.

    `tail_only=True` chỉ dựng trọng số cho đuôi lưới (đủ cửa sổ biến động) — cách live
    tiết kiệm thời gian. Hàng cuối không đổi so với dựng toàn bộ: test sẵn có
    `test_cat_duoi_chuoi_khong_doi_trong_so_hang_cuoi`.
    """
    marks = anchored_marks(close.index, step, anchor_pos)
    combined = combine_adaptive({k: v.reindex(marks) for k, v in signals.items()},
                                close.reindex(marks), combiner, top_frac=top_frac)
    idx = marks[-(spec.vol_window + 5):] if tail_only else marks
    W = build_weights(combined.reindex(idx), close.reindex(idx), spec)
    return W.iloc[-1]


def tranched_target_weights(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    step: int,
    n_tranches: int,
    combiner: CombinerSpec,
    spec: PortfolioSpec,
    top_frac: float,
    tail_only: bool = True,
) -> pd.Series:
    """
    Đích của CẢ SỔ tại nến mới nhất = trung bình K lô, lô j neo ở `cuối - j*step/K`.

    K = 1 là chiến lược một lô như cũ, nhưng đã vá F53 (tín hiệu tại nến mới nhất).

    Trung bình KHÔNG chuẩn hoá lại gross: K sổ trung lập đô-la cộng lại vẫn trung lập,
    nhưng một cặp long ở lô này và short ở lô kia thì bù trừ — đó là netting thật của
    việc chia lô, và chuẩn hoá lại sẽ ngầm tăng đòn bẩy của phần còn lại.
    """
    last = len(close.index) - 1
    rows = []
    for off in tranche_offsets(step, n_tranches):
        if last - off < 0:
            continue
        rows.append(target_weights_at(signals, close, last - off, step, combiner, spec,
                                      top_frac, tail_only=tail_only))
    if not rows:
        return pd.Series(dtype=float)
    # Chia cho K (không phải số lô dựng được): lô chưa đủ lịch sử là lô đứng ngoài.
    return pd.concat(rows, axis=1).fillna(0.0).sum(axis=1) / n_tranches


def tranched_weight_panel(
    signals: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    step: int,
    n_tranches: int,
    combiner: CombinerSpec,
    spec: PortfolioSpec,
    top_frac: float,
    phase: int = 0,
    cache: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Trọng số gộp trên TOÀN dòng thời gian, cho backtest.

    Mỗi lô j chạy trên lưới pha `(phase + j*step/K) mod step`, giữ đích của nó giữa hai
    mốc; sổ gộp đổi mỗi step/K nến (mỗi lần có một lô tái cân bằng). `n_tranches=1` với
    `phase=p` là chiến lược một lô bắt đầu ở pha p — dùng để đo phân tán theo giờ.

    `cache`: dict dùng lại trọng số từng PHA giữa nhiều lần gọi (quét 18 pha + nhiều K
    chỉ trả giá tầng gộp đúng 18 lần). CHỈ hợp lệ khi mọi lần gọi dùng CÙNG signals,
    close, step, combiner, spec, top_frac — người gọi chịu trách nhiệm điều đó.
    """
    gap = step // n_tranches
    tranche_offsets(step, n_tranches)                       # kiểm tra chia hết
    phases = sorted({(phase + j * gap) % step for j in range(n_tranches)})
    grid = close.index[phase % gap::gap]
    parts = []
    for p in phases:
        if cache is not None and p in cache:
            W = cache[p]
        else:
            marks = close.index[p::step]
            combined = combine_adaptive({k: v.reindex(marks) for k, v in signals.items()},
                                        close.reindex(marks), combiner, top_frac=top_frac)
            W = build_weights(combined, close.reindex(marks), spec)
            if cache is not None:
                cache[p] = W
        parts.append(W.reindex(grid).ffill().fillna(0.0))
    total = parts[0]
    for P in parts[1:]:
        total = total.add(P, fill_value=0.0)
    return total / n_tranches
