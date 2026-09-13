"""
Basis trade (funding arbitrage): short perp + long spot cùng tài sản.

VÌ SAO LỚP CHIẾN LƯỢC NÀY KHÁC HẲN phần còn lại của hệ thống: nó KHÔNG dự báo giá.
Vị thế trung tính delta theo từng tài sản, nên nguồn lợi nhuận là dòng tiền funding —
một khoản cơ học, không phải một dự báo. Đổi lại, rủi ro chuyển từ "đoán sai hướng"
sang BASIS RISK: perp và spot lệch nhau.

ĐO ĐƯỢC TRÊN DỮ LIỆU NÀY (20 cặp có cả spot lẫn perp, 2020-2026):

  basis (perp/spot - 1): trung vị -0.048%, độ lệch chuẩn 0.081%
  thay đổi basis mỗi 4h: độ lệch chuẩn 0.066%, cực trị -1.17%..+1.14%

  8 vị thế, giữ dài:  train  +3.1%/năm, Sharpe 4.26, maxDD 0.3%
                      holdout +3.2%/năm, Sharpe 2.24, maxDD 0.6%

Đúng hình dạng mà tài liệu mô tả cho lớp này: Sharpe cao, drawdown cực nhỏ, lợi suất
tuyệt đối THẤP. Nó là một PHƯƠNG TIỆN ĐÒN BẨY, không phải một cỗ máy lợi nhuận: ở
maxDD 0.6%, đòn bẩy 30x vẫn chỉ cho drawdown ~18%.

HAI SAI LẦM ĐO LƯỜNG ĐÃ MẮC VÀ SỬA TRONG QUÁ TRÌNH DỰNG MODULE NÀY — ghi lại vì cả
hai đều tạo ra kết quả đẹp giả tạo:

  1. GIẢ ĐỊNH PHÒNG HỘ HOÀN HẢO. Bản đầu tính Sharpe của chính dòng funding (gần như
     luôn dương) và ra Sharpe 20-60. Đó là đặt rủi ro thật bằng 0 rồi đo rủi ro.
     Rủi ro thật của basis trade nằm ở basis, không ở funding.

  2. CHỌN THEO FUNDING CÙNG THỜI ĐIỂM. Lấy top-k theo funding tại chính mốc t cho
     25-43%/năm; dùng trung bình trượt NHÂN QUẢ chỉ còn 5-9%/năm. Chênh lệch đó
     hoàn toàn là nhìn trước.

  3. ĐUỔI THEO FUNDING CAO NHẤT MỖI 4H. Phí ăn -150%/năm. Lớp chiến lược này chỉ
     sống khi NẮM GIỮ: rebalance hàng tuần tới hàng tháng, có vùng đệm.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["BasisSpec", "basis_series", "backtest_basis"]


@dataclass
class BasisSpec:
    """Tham số basis trade."""

    n_positions: int = 8          # số cặp nắm cùng lúc
    rebalance_bars: int = 1008    # số nến 4h giữa hai lần xét lại (1008 = 168 ngày)
    funding_window: int = 12      # cửa sổ trung bình funding để xếp hạng (nhân quả)
    exit_multiple: int = 3        # vùng đệm: giữ tới khi rơi khỏi top n*exit_multiple
    round_trip_bps: float = 24.0  # phí một vòng đầy đủ 2 chân (perp maker + spot maker)
    min_universe: int = 8


def basis_series(perp_close: pd.DataFrame, spot_close: pd.DataFrame) -> pd.DataFrame:
    """`perp/spot - 1`. Dương nghĩa là perp đắt hơn spot (trạng thái thường gặp)."""
    s = spot_close.reindex(index=perp_close.index, columns=perp_close.columns)
    return perp_close / s.replace(0.0, np.nan) - 1.0


def backtest_basis(
    perp_close: pd.DataFrame,
    spot_close: pd.DataFrame,
    funding: pd.DataFrame,
    spec: Optional[BasisSpec] = None,
) -> pd.DataFrame:
    """
    Mô phỏng basis trade. Trả về DataFrame các cột:
    `total`, `funding`, `basis`, `fee` — lợi suất mỗi nến 4h trên vốn.

    Quy ước: SHORT perp + LONG spot. Lãi/lỗ giá = -(thay đổi basis); funding thu về
    khi rate dương (bên short nhận). Mỗi nến 4h ứng với nửa mốc funding 8h.
    """
    spec = spec or BasisSpec()
    cols = [c for c in perp_close.columns if c in spot_close.columns]
    P = perp_close[cols]
    S = spot_close.reindex(index=P.index)[cols]
    F = funding.reindex(index=P.index, columns=cols)

    rank_src = F.rolling(spec.funding_window,
                         min_periods=max(2, spec.funding_window // 3)).mean().shift(1)

    idx = P.index
    out = {}
    held: set = set()
    for i in range(1, len(idx)):
        ts, prev = idx[i], idx[i - 1]
        fee = 0.0

        if i % spec.rebalance_bars == 0:
            v = rank_src.loc[prev].dropna()
            if len(v) >= spec.min_universe:
                enter = set(v.nlargest(spec.n_positions).index)
                keep = set(v.nlargest(spec.n_positions * spec.exit_multiple).index)
                survivors = held & keep
                room = max(0, spec.n_positions - len(survivors))
                new = survivors | set([x for x in enter if x not in survivors][:room])
                if new:
                    fee = len(new ^ held) / max(len(new), 1) * spec.round_trip_bps / 1e4
                    held = new

        bas = fnd = 0.0
        if held:
            n = len(held)
            for s in held:
                vals = (P.loc[ts, s], P.loc[prev, s], S.loc[ts, s], S.loc[prev, s])
                if not all(pd.notna(x) for x in vals):
                    continue
                dp = P.loc[ts, s] / P.loc[prev, s] - 1.0
                ds = S.loc[ts, s] / S.loc[prev, s] - 1.0
                bas += -(dp - ds) / n                      # short perp, long spot
                fnd += float(F.loc[prev, s]) * 0.5 / n     # 4h = nửa mốc funding

        out[ts] = (bas + fnd - fee, fnd, bas, -fee)

    return pd.DataFrame(out, index=["total", "funding", "basis", "fee"]).T
