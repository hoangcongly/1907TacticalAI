"""
Dựng trọng số danh mục cross-sectional — tầng biến TÍN HIỆU thành VỊ THẾ.

VÌ SAO TẦNG NÀY QUYẾT ĐỊNH NHIỀU HƠN MODEL: hai nhà quản lý có cùng dự báo nhưng
khác cách dựng danh mục sẽ cho Sharpe chênh nhau 30-60%. Ba khuyết điểm của cách
xếp hạng nhị phân (mua top-decile, chia đều VỐN) mà module này sửa:

1. CHIA ĐỀU VỐN ≠ CHIA ĐỀU RỦI RO. Một memecoin biến động 200%/năm và BTC biến
   động 40%/năm nhận cùng số đô-la thì memecoin đóng góp gấp 5 lần rủi ro. Danh mục
   trên danh nghĩa có 12 vị thế, thực chất chỉ là cược vào 2-3 cái biến động nhất.
   -> `inverse_vol_weights`.

2. NHỊ PHÂN VỨT BỎ ĐỘ MẠNH TÍN HIỆU. Cặp hạng 1 và cặp hạng 6 nhận cùng trọng số,
   dù z-score có thể chênh 3 lần. Thông tin bị vứt đi không lấy lại được.
   -> chế độ `zscore` / `zscore_riskparity`.

3. DOLLAR-NEUTRAL KHÔNG PHẢI BETA-NEUTRAL. Long 6 alt beta 1.4 và short 6 large-cap
   beta 0.7 thì tổng đô-la bằng 0 nhưng danh mục vẫn long beta ròng ~0.35 — khi thị
   trường sập, "market-neutral" lỗ nặng. Đây là cách phần lớn quỹ neutral chết.
   -> `beta_neutralize`.

Mọi ước lượng (biến động, beta, hiệp phương sai) đều tính bằng dữ liệu tới t-1 và
được `shift(1)` — không có đường nào nhìn trước.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from aegis.risk.covariance_shrinkage import shrink_covariance

__all__ = [
    "PortfolioSpec",
    "build_weights",
    "inverse_vol_weights",
    "beta_neutralize",
    "realized_vol",
    "rolling_beta",
    "apply_buffer",
    "project_neutral",
]

# Chế độ dựng trọng số được hỗ trợ.
WEIGHT_MODES = ("rank_binary", "rank_riskparity", "zscore", "zscore_riskparity", "mvo")


@dataclass
class PortfolioSpec:
    """Toàn bộ tham số dựng danh mục — khai báo một chỗ, dùng chung research và live."""

    mode: str = "zscore_riskparity"
    top_frac: float = 0.10            # tỷ lệ tài sản mỗi bên khi `n_positions` không đặt
    n_positions: Optional[int] = None  # TỔNG số vị thế cố định (long + short)
    exit_frac: Optional[float] = None  # vùng đệm thứ hạng; None = không đệm
    gross: float = 1.0                # tổng |trọng số| mục tiêu
    max_weight: float = 0.20          # trần trọng số một cặp (chống tập trung)
    vol_window: int = 60              # cửa sổ ước lượng biến động (nến)
    vol_floor_pct: float = 0.10       # sàn biến động = phân vị này của mặt cắt ngang
    beta_neutral: bool = False        # khử beta thị trường ngoài dollar-neutral
    beta_window: int = 120
    z_clip: float = 2.5               # kẹp z-score trước khi biến thành trọng số
    min_names: int = 6                # số tài sản tối thiểu để mở danh mục
    cov_method: str = "constant_correlation"
    cov_window: int = 250             # chỉ dùng cho mode="mvo"
    risk_aversion: float = 5.0        # chỉ dùng cho mode="mvo"
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in WEIGHT_MODES:
            raise ValueError(f"mode phải thuộc {WEIGHT_MODES}, nhận {self.mode!r}")
        if not 0 < self.top_frac < 0.5:
            raise ValueError(f"top_frac phải trong (0, 0.5), nhận {self.top_frac}")
        if self.exit_frac is not None and self.exit_frac < self.top_frac:
            raise ValueError("exit_frac phải >= top_frac (vùng giữ rộng hơn vùng vào)")
        if self.n_positions is not None and self.n_positions < 4:
            raise ValueError(f"n_positions phải >= 4, nhận {self.n_positions}")

    def side_count(self, n_valid: int) -> int:
        """
        Số vị thế MỖI CHÂN.

        `n_positions` (số tuyệt đối) được ưu tiên hơn `top_frac` (tỷ lệ). Với tài
        khoản nhỏ, ràng buộc thật là min notional nhân số vị thế, chứ không phải một
        tỷ lệ phần trăm của universe — nếu để theo tỷ lệ thì mở rộng universe sẽ âm
        thầm làm số vị thế phình lên và mọi so sánh trở nên vô nghĩa.
        """
        if self.n_positions is not None:
            return max(1, min(self.n_positions // 2, n_valid // 2))
        return max(1, int(round(n_valid * self.top_frac)))

    def effective_min_names(self) -> int:
        """
        Số cặp tối thiểu để mở danh mục.

        `min_names` mặc định là 6 nhằm chặn một "sổ market-neutral" chỉ có 2 chân —
        khi đó nó là cược cặp đôi chứ không phải danh mục. Nhưng nếu người dùng CỐ Ý
        đặt `n_positions` nhỏ hơn thế, chặn theo mặc định sẽ khiến hàm im lặng không
        mở vị thế nào — im lặng là chế độ hỏng tệ nhất. Ưu tiên ý định tường minh.
        """
        if self.n_positions is not None:
            return min(self.min_names, self.n_positions)
        return self.min_names


# ---------------------------------------------------------------------------
# Ước lượng rủi ro (nhân quả)
# ---------------------------------------------------------------------------
def realized_vol(close: pd.DataFrame, window: int = 60, min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Biến động thực hiện theo nến, tính tới t-1.

    Dùng lợi suất log để cộng dồn đúng qua các khung thời gian và chịu được các cú
    nhảy giá lớn thường gặp ở altcoin.
    """
    min_periods = min_periods or max(5, window // 3)
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=min_periods).std().shift(1)


def rolling_beta(
    close: pd.DataFrame,
    window: int = 120,
    market: Optional[pd.Series] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Beta của từng cặp so với "thị trường" (mặc định: rổ đều trọng số của universe).

    Rổ đều trọng số tốt hơn dùng riêng BTC làm thị trường: khi dòng tiền xoay sang
    altcoin, BTC có thể đứng yên trong lúc cả rổ chạy — beta so với BTC khi đó
    ước lượng sai hẳn cấu trúc rủi ro.
    """
    min_periods = min_periods or max(20, window // 3)
    ret = close.pct_change()
    mkt = ret.mean(axis=1) if market is None else market.reindex(ret.index)

    cov = ret.rolling(window, min_periods=min_periods).cov(mkt)
    var = mkt.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var.replace(0.0, np.nan), axis=0)
    return beta.shift(1)


# ---------------------------------------------------------------------------
# Các phép biến đổi trọng số
# ---------------------------------------------------------------------------
def inverse_vol_weights(raw: pd.Series, vol: pd.Series, vol_floor: float) -> pd.Series:
    """
    Chia tỷ lệ trọng số theo NGHỊCH ĐẢO biến động — chia đều RỦI RO thay vì VỐN.

    `vol_floor` chặn trường hợp một cặp tạm thời gần như không biến động (nến thiếu
    thanh khoản, giá đứng im) nuốt trọn danh mục.
    """
    v = vol.reindex(raw.index).astype(np.float64)
    v = v.where(v > vol_floor, vol_floor)
    v = v.fillna(vol_floor)
    return raw / v


def beta_neutralize(w: pd.Series, beta: pd.Series) -> pd.Series:
    """
    Trừ đi phần hình chiếu của danh mục lên nhân tố thị trường.

    Giải bài toán: tìm `w' = w - lambda * beta` sao cho `w' . beta = 0`, tức
    `lambda = (w . beta) / (beta . beta)`. Đây là phép chiếu trực giao — thay đổi
    danh mục ít nhất có thể theo chuẩn L2 mà vẫn đạt beta ròng bằng 0.
    """
    b = beta.reindex(w.index).astype(np.float64)
    mask = b.notna() & (w != 0.0)
    if mask.sum() < 2:
        return w

    bb = float((b[mask] ** 2).sum())
    if bb <= 1e-12:
        return w

    lam = float((w[mask] * b[mask]).sum()) / bb
    out = w.copy()
    out[mask] = w[mask] - lam * b[mask]
    return out


def project_neutral(w: pd.Series, beta: Optional[pd.Series] = None) -> pd.Series:
    """
    Chiếu trọng số lên không gian thoả ĐỒNG THỜI: tổng trọng số = 0 và tổng
    (trọng số x beta) = 0.

    VÌ SAO PHẢI LÀM ĐỒNG THỜI: làm tuần tự (khử beta trước, ép dollar-neutral sau)
    thì bước sau phá bước trước. Đây chính là cơ chế khiến một sổ tự nhận là
    market-neutral vẫn ôm beta ròng — và nó chỉ lộ ra đúng vào ngày thị trường sập.

    Toán: với ràng buộc `A w = 0` trong đó A là ma trận 2 x n gồm hàng toàn 1 và
    hàng beta, nghiệm gần `w` nhất là phép chiếu trực giao
    `w' = w - A^T (A A^T)^{-1} A w`. Ma trận cần nghịch đảo chỉ là 2x2.

    Khi beta không khả dụng thì rút về ràng buộc dollar-neutral đơn thuần.
    """
    active = w != 0.0
    if int(active.sum()) < 3:
        return w * 0.0

    idx = w.index[active]
    x = w[idx].to_numpy(dtype=np.float64)
    ones = np.ones(len(idx))

    if beta is None:
        A = ones.reshape(1, -1)
    else:
        b = beta.reindex(idx).to_numpy(dtype=np.float64)
        if not np.isfinite(b).all():
            b = np.where(np.isfinite(b), b, np.nanmedian(b[np.isfinite(b)]) if np.isfinite(b).any() else 1.0)
        A = np.vstack([ones, b])

    G = A @ A.T
    try:
        lam = np.linalg.solve(G, A @ x)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(G, A @ x, rcond=None)[0]

    out = w.copy() * 0.0
    out[idx] = x - A.T @ lam
    return out


def _cap_and_scale(w: pd.Series, gross: float, max_weight: float) -> pd.Series:
    """
    Áp trần từng cặp rồi chuẩn hoá về tổng |trọng số| = `gross`.

    Lặp vì việc kẹp một cặp làm tăng tỷ trọng các cặp còn lại sau chuẩn hoá, có thể
    đẩy cặp khác vượt trần. Thực tế hội tụ sau 2-3 vòng.
    """
    total = w.abs().sum()
    if total <= 1e-12:
        return w * 0.0
    w = w / total * gross

    cap = max_weight * gross
    for _ in range(8):
        over = w.abs() > cap + 1e-12
        if not over.any():
            break
        w[over] = np.sign(w[over]) * cap
        free = ~over
        residual = gross - w[over].abs().sum()
        free_total = w[free].abs().sum()
        if free_total <= 1e-12 or residual <= 0:
            break
        w[free] = w[free] / free_total * residual
    return w


def _dollar_neutralize(w: pd.Series, sidewise: bool = True) -> pd.Series:
    """
    Ép tổng trọng số ròng về 0.

    `sidewise=True` (mặc định) chuẩn hoá RIÊNG mỗi chân về tổng |trọng số| = 0.5.
    Cách này trung lập đô-la theo đúng định nghĩa và KHÔNG BAO GIỜ đổi dấu một vị
    thế. Trừ trung bình chung (`sidewise=False`) tuy cũng cho net = 0 nhưng khi mặt
    cắt ngang lệch mạnh có thể lật dấu vị thế yếu nhất — biến một lệnh long đã chọn
    thành lệnh short mà không ai chủ ý.

    Chỉ dùng `sidewise=False` sau khi khử beta, khi cần trung hoà lại phần lệch nhỏ
    mà vẫn giữ nguyên cấu trúc vừa được chiếu.
    """
    active = w != 0.0
    if int(active.sum()) < 2:
        return w * 0.0

    if not sidewise:
        out = w.copy()
        out[active] = w[active] - w[active].mean()
        return out

    out = w.copy() * 0.0
    for sign, side in ((1.0, w > 0), (-1.0, w < 0)):
        total = w[side].abs().sum()
        if total > 1e-12:
            out[side] = w[side] / total * 0.5
    if (out > 0).sum() == 0 or (out < 0).sum() == 0:
        return w * 0.0
    return out


# ---------------------------------------------------------------------------
# Chọn tập tài sản
# ---------------------------------------------------------------------------
def apply_buffer(
    ranks: pd.DataFrame,
    entry_frac: float,
    exit_frac: float,
    n_positions: Optional[int] = None,
    exit_ratio: float = 1.5,
) -> pd.DataFrame:
    """
    Mặt nạ vùng đệm: +1 giữ long, -1 giữ short, 0 ngoài danh mục.

    VÀO khi lọt top `entry_frac`, GIỮ tới khi rơi khỏi top `exit_frac`. Giảm turnover
    mà gần như không mất tín hiệu — cặp trượt từ hạng 6 xuống 7 không đáng để trả
    hai lần phí cộng trượt giá.
    """
    mask = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    held_long: set = set()
    held_short: set = set()

    for ts, row in ranks.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n < 4:
            held_long, held_short = set(), set()
            continue

        if n_positions is not None:
            k_in = max(1, min(n_positions // 2, n // 2))
            k_out = max(k_in, min(int(round(k_in * exit_ratio)), n // 2))
        else:
            k_in = max(1, int(round(n * entry_frac)))
            k_out = max(k_in, int(round(n * exit_frac)))
        ordered = valid.sort_values()

        enter_short, enter_long = set(ordered.index[:k_in]), set(ordered.index[-k_in:])
        keep_short, keep_long = set(ordered.index[:k_out]), set(ordered.index[-k_out:])

        surv_long = (held_long & keep_long) - held_short
        surv_short = (held_short & keep_short) - surv_long

        room_long = max(0, k_in - len(surv_long))
        room_short = max(0, k_in - len(surv_short))

        cand_long = [x for x in reversed(ordered.index)
                     if x in enter_long and x not in surv_long and x not in surv_short]
        cand_short = [x for x in ordered.index
                      if x in enter_short and x not in surv_short and x not in surv_long]

        held_long = surv_long | set(cand_long[:room_long])
        held_short = surv_short | set(cand_short[:room_short])

        if len(held_long) > k_in:
            held_long = set(valid[list(held_long)].sort_values().index[-k_in:])
        if len(held_short) > k_in:
            held_short = set(valid[list(held_short)].sort_values().index[:k_in])

        if held_long:
            mask.loc[ts, list(held_long)] = 1.0
        if held_short:
            mask.loc[ts, list(held_short)] = -1.0

    return mask


def _selection_mask(signal: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    """Chọn tài sản vào danh mục: top/bottom `top_frac`, có hoặc không có vùng đệm."""
    if spec.exit_frac is not None:
        ratio = spec.exit_frac / spec.top_frac
        return apply_buffer(signal, spec.top_frac, spec.exit_frac,
                            n_positions=spec.n_positions, exit_ratio=ratio)

    mask = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for ts, row in signal.iterrows():
        valid = row.dropna()
        n = len(valid)
        if n < 4:
            continue
        k = spec.side_count(n)
        ordered = valid.sort_values()
        mask.loc[ts, ordered.index[:k]] = -1.0
        mask.loc[ts, ordered.index[-k:]] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Điểm vào chính
# ---------------------------------------------------------------------------
def build_weights(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    spec: Optional[PortfolioSpec] = None,
    vol: Optional[pd.DataFrame] = None,
    beta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Biến điểm số cross-sectional thành trọng số danh mục.

    `signal` đã là z-score theo hàng (xem `research.cross_sectional.xs_zscore`).
    `close` phải phủ CÙNG lưới thời gian với signal để ước lượng biến động/beta
    khớp nhịp tái cân bằng.

    Trả về DataFrame trọng số cùng shape với `signal`; hàng không mở danh mục toàn 0.
    """
    spec = spec or PortfolioSpec()
    signal = signal.reindex(columns=close.columns)

    if vol is None:
        vol = realized_vol(close, spec.vol_window)
    vol = vol.reindex(index=signal.index, columns=signal.columns)

    if spec.beta_neutral and beta is None:
        beta = rolling_beta(close, spec.beta_window)
    if beta is not None:
        beta = beta.reindex(index=signal.index, columns=signal.columns)

    if spec.mode == "mvo":
        return _build_weights_mvo(signal, close, spec, vol, beta)

    use_rank = spec.mode.startswith("rank")
    use_riskparity = spec.mode.endswith("riskparity")

    # Tập tài sản LUÔN được chọn theo thứ hạng, kể cả ở chế độ trọng số liên tục.
    # Không có bước này thì "liên tục" nghĩa là nắm TOÀN BỘ universe (120+ cặp) —
    # bất khả thi với vốn nhỏ (min notional $5 x 120 = $600 chỉ để mở một vòng),
    # và nó làm mọi so sánh với chế độ nhị phân trở nên bất công: phần lớn mức
    # Sharpe tăng thêm khi đó đến từ việc nắm nhiều cặp hơn, chứ không từ cách
    # đánh trọng số. Chọn theo hạng rồi mới đánh trọng số theo độ mạnh.
    mask = _selection_mask(signal, spec)
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    # Sàn biến động theo mặt cắt ngang: chống chia cho số gần 0 mà vẫn thích ứng
    # theo chế độ thị trường (giai đoạn cả thị trường lặng thì sàn cũng thấp theo).
    vol_floor = vol.quantile(spec.vol_floor_pct, axis=1)
    median_vol = vol.median(axis=1)

    for ts, row in signal.iterrows():
        floor = vol_floor.get(ts, np.nan)
        if not np.isfinite(floor) or floor <= 0:
            floor = median_vol.get(ts, np.nan)
        if not np.isfinite(floor) or floor <= 0:
            continue

        sel = mask.loc[ts]
        chosen = sel[sel != 0.0]
        if len(chosen) < spec.effective_min_names():
            continue

        if use_rank:
            # Nhị phân: mọi cặp trong nhóm nhận cùng độ lớn, chỉ khác dấu.
            raw = chosen
        else:
            # Liên tục: giữ ĐỘ MẠNH tín hiệu trong nhóm đã chọn. Chuẩn hoá bằng
            # thống kê của TOÀN mặt cắt ngang (không phải của riêng nhóm đã chọn)
            # để độ lớn phản ánh đúng vị trí trong phân phối gốc.
            valid = row.dropna()
            sd = valid.std()
            if not np.isfinite(sd) or sd <= 1e-12:
                continue
            z = ((valid - valid.mean()) / sd).clip(-spec.z_clip, spec.z_clip)
            raw = z.reindex(chosen.index).dropna()
            if len(raw) < spec.effective_min_names():
                continue
            # Ép dấu khớp với nhóm đã chọn: một cặp lọt nhóm long phải có trọng số
            # dương kể cả khi z-score của nó hơi âm (xảy ra khi mặt cắt ngang lệch).
            raw = raw.abs() * chosen.reindex(raw.index)

        if use_riskparity:
            raw = inverse_vol_weights(raw, vol.loc[ts], floor)

        w = _dollar_neutralize(raw)
        b_row = beta.loc[ts] if (spec.beta_neutral and beta is not None) else None
        # THỨ TỰ QUAN TRỌNG: kẹp trần TRƯỚC, chiếu SAU, và kết thúc bằng phép chiếu.
        # Kẹp trần làm hỏng tính trung lập (nó phân phối lại phần dư không đối xứng),
        # nên nếu kết thúc bằng kẹp trần thì sổ ra khỏi hàm với beta/net khác 0. Kết
        # thúc bằng phép chiếu đảm bảo trung lập, đổi lại một cặp có thể vượt trần
        # vài phần trăm — đánh đổi đúng hướng, vì trần chỉ chống tập trung còn trung
        # lập là toàn bộ lý do chiến lược này tồn tại.
        for _ in range(3):
            w = _cap_and_scale(w, spec.gross, spec.max_weight)
            w = project_neutral(w, b_row)
        w = w / max(float(w.abs().sum()), 1e-12) * spec.gross
        if w.abs().sum() <= 1e-12:
            continue
        weights.loc[ts, w.index] = w.to_numpy()

    return weights


def _build_weights_mvo(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    spec: PortfolioSpec,
    vol: pd.DataFrame,
    beta: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Tối ưu trung bình-phương sai với hiệp phương sai đã co.

    Giải `max  w'mu - (lambda/2) w'Sigma w` trên tập tài sản đã chọn, rồi ép
    dollar-neutral và áp trần. Dùng nghịch đảo trực tiếp trên tập nhỏ (12-30 cặp)
    nên chi phí không đáng kể.

    CẢNH BÁO: MVO khuếch đại sai số ước lượng. Chỉ dùng khi cửa sổ hiệp phương sai
    đủ dài so với số tài sản (`cov_window` >= 8x số vị thế) — nếu không, chế độ
    `zscore_riskparity` bền hơn.
    """
    ret = close.pct_change()
    mask = _selection_mask(signal, spec)
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    positions = list(signal.index)

    for i, ts in enumerate(positions):
        sel = mask.loc[ts]
        names = sel[sel != 0.0].index.tolist()
        if len(names) < spec.effective_min_names():
            continue

        lo = max(0, i - spec.cov_window)
        window = ret.iloc[lo:i][names].dropna(axis=1, how="all")
        names = [c for c in names if c in window.columns]
        window = window[names].dropna()
        if len(window) < max(30, 3 * len(names)) or len(names) < spec.effective_min_names():
            continue

        try:
            sigma = shrink_covariance(window, method=spec.cov_method)
        except (ValueError, np.linalg.LinAlgError):
            continue

        # mu tỷ lệ với tín hiệu nhân biến động: điểm số cao trên tài sản biến động
        # mạnh mang kỳ vọng lợi suất tuyệt đối lớn hơn cùng điểm số trên tài sản lặng.
        z = signal.loc[ts, names].fillna(0.0).clip(-spec.z_clip, spec.z_clip)
        sd = np.sqrt(np.diag(sigma))
        mu = z.to_numpy() * sd

        try:
            raw = np.linalg.solve(spec.risk_aversion * sigma, mu)
        except np.linalg.LinAlgError:
            continue

        w = pd.Series(raw, index=names)
        b_row = beta.loc[ts] if (spec.beta_neutral and beta is not None) else None
        for _ in range(3):
            w = _cap_and_scale(w, spec.gross, spec.max_weight)
            w = project_neutral(w, b_row)
        w = w / max(float(w.abs().sum()), 1e-12) * spec.gross
        if w.abs().sum() <= 1e-12:
            continue
        weights.loc[ts, w.index] = w.to_numpy()

    return weights
