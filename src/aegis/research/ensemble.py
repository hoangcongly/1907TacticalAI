"""
Gộp DANH MỤC từ nhiều cấu hình — chống rủi ro chọn sai siêu tham số.

VÌ SAO CẦN, dựa trên đúng số liệu của hệ thống này:

Chạy 32 cấu hình siêu tham số trên tập holdout cho phân phối Sharpe:
    trung vị 0.84 | trung bình 0.62 | min -0.47 | max 1.44 | 75% dương

Cấu hình được chọn trên train (rebal 18, ngưỡng t 2.0) ra 0.71 — nằm quanh giữa
phân phối, không phải ở đuôi trên. Đó là điều PHẢI kỳ vọng: chọn cấu hình tốt nhất
trên train cho ta một mẫu ngẫu nhiên từ phân phối này, không cho ta cái tốt nhất.

Vậy nên thay vì cược vào một điểm trong không gian siêu tham số, ta nắm NHIỀU điểm
cùng lúc và trung bình TRỌNG SỐ VỊ THẾ của chúng. Ba hệ quả:

  1. Kết quả hội tụ về vùng giữa phân phối thay vì phụ thuộc may rủi của một lựa chọn.
  2. Biến động GIẢM: các cấu hình sai lệch theo hướng khác nhau và triệt tiêu lẫn nhau,
     nên Sharpe của tổ hợp thường CAO HƠN Sharpe trung vị của các thành phần.
  3. Turnover GIẢM: khi một cấu hình muốn vào và cấu hình khác muốn ra, phần bù trừ
     xảy ra ở tầng trọng số và không bao giờ trở thành lệnh thật.

Điều kiện để (2) đúng: các cấu hình phải khác nhau ở chiều CÓ Ý NGHĨA (chu kỳ tái
cân bằng, bộ tín hiệu), không phải khác vụn vặt. Gộp 5 cấu hình gần trùng nhau chỉ
tốn công tính.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from aegis.risk.portfolio import PortfolioSpec, build_weights, project_neutral

__all__ = ["EnsembleMember", "ensemble_weights", "align_to_grid"]


@dataclass
class EnsembleMember:
    """Một thành phần: tín hiệu trên lưới riêng + cách dựng danh mục riêng."""

    name: str
    signal: pd.DataFrame        # trên lưới tái cân bằng RIÊNG của nó
    spec: PortfolioSpec
    weight: float = 1.0


def align_to_grid(w: pd.DataFrame, grid: pd.Index) -> pd.DataFrame:
    """
    Đưa trọng số của một thành phần về lưới CHUNG.

    Thành phần tái cân bằng mỗi 48h, lưới chung mỗi 24h: giữa hai lần nó tái cân
    bằng thì nó vẫn ĐANG NẮM vị thế cũ, nên `ffill` mới đúng. Điền 0 sẽ mô tả một
    thành phần liên tục đóng rồi mở lại vị thế — sai cả về rủi ro lẫn chi phí.
    """
    return w.reindex(w.index.union(grid)).ffill().reindex(grid)


def ensemble_weights(
    members: List[EnsembleMember],
    close: pd.DataFrame,
    grid: pd.Index,
    max_weight: float = 0.20,
    gross: float = 1.0,
    max_positions: Optional[int] = None,
    vol: Optional[pd.DataFrame] = None,
    risk_parity: bool = True,
) -> pd.DataFrame:
    """
    Trọng số danh mục tổ hợp trên lưới chung `grid`.

    `max_positions` cắt bớt về đúng số vị thế mà vốn cho phép: sau khi trung bình,
    tổ hợp có thể chạm nhiều cặp hơn từng thành phần (mỗi thành phần chọn một tập
    hơi khác). Với tài khoản nhỏ đây là ràng buộc cứng — giữ lại các cặp có |trọng
    số| lớn nhất rồi chuẩn hoá lại.

    `risk_parity=True` KHÔI PHỤC cấu trúc chia đều rủi ro sau khi cắt. Bước này bắt
    buộc, và lý do đã đo được: bản đầu tiên bỏ qua nó và tổ hợp ra biến động 72.9%
    trong khi mọi thành phần đều quanh 50-55%. Nguyên nhân: từng thành phần đánh
    trọng số theo nghịch đảo biến động TRONG tập nó chọn, nhưng phép trung bình rồi
    cắt về 12 cặp rồi chuẩn hoá về gross 1.0 phá vỡ tỷ lệ đó — các cặp biến động mạnh
    được nhiều thành phần cùng chọn nên nổi lên trên và chiếm phần rủi ro lớn.
    Trung bình trọng số KHÔNG tự bảo toàn cấu trúc rủi ro.
    """
    if not members:
        raise ValueError("Cần ít nhất một thành phần")

    total_w = sum(m.weight for m in members)
    if total_w <= 0:
        raise ValueError("Tổng trọng số thành phần phải dương")

    acc = pd.DataFrame(0.0, index=grid, columns=close.columns)
    for m in members:
        marks = m.signal.index
        w = build_weights(m.signal, close.reindex(marks), m.spec)
        acc = acc.add(align_to_grid(w, grid).reindex(columns=close.columns).fillna(0.0)
                      * (m.weight / total_w), fill_value=0.0)

    out = pd.DataFrame(0.0, index=grid, columns=close.columns)
    for ts in grid:
        w = acc.loc[ts]
        if w.abs().sum() <= 1e-12:
            continue

        if max_positions is not None:
            keep = w.abs().nlargest(max_positions).index
            w = w.where(w.index.isin(keep), 0.0)

        if risk_parity and vol is not None and ts in vol.index:
            v = vol.loc[ts].reindex(w.index)
            floor = float(np.nanquantile(v.to_numpy(dtype=float), 0.10)) if v.notna().any() else np.nan
            if np.isfinite(floor) and floor > 0:
                v = v.where(v > floor, floor).fillna(floor)
                w = w / v

        w = project_neutral(w, None)          # ép net = 0 sau khi cắt bớt
        s = w.abs().sum()
        if s <= 1e-12:
            continue
        w = w / s * gross

        cap = max_weight * gross
        for _ in range(4):
            over = w.abs() > cap + 1e-12
            if not over.any():
                break
            w[over] = np.sign(w[over]) * cap
            free = ~over
            residual = gross - w[over].abs().sum()
            ft = w[free].abs().sum()
            if ft <= 1e-12 or residual <= 0:
                break
            w[free] = w[free] / ft * residual
            w = project_neutral(w, None)
            w = w / max(float(w.abs().sum()), 1e-12) * gross

        out.loc[ts, w.index] = w.to_numpy()

    return out
