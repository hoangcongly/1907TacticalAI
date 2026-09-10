"""
Đòn bẩy, mục tiêu biến động, và XÁC SUẤT ĐẠT MỤC TIÊU — trả lời bằng số, không bằng ý kiến.

Câu hỏi "làm sao lời 50% một tuần" có một câu trả lời định lượng chính xác, và nó
không phải "được" hay "không được". Nó là: VỚI ĐÒN BẨY L, xác suất đạt +50% trong
một tuần là p, và xác suất mất phần lớn tài khoản là q. Cả p lẫn q đều tính được từ
phân phối lợi suất của chiến lược. Việc chọn L là quyết định của người bỏ vốn; việc
đưa ra (p, q) trung thực là việc của hệ thống.

BA SỰ THẬT TOÁN HỌC CHI PHỐI MỌI THỨ Ở ĐÂY:

1. ĐÒN BẨY KHÔNG ĐỔI SHARPE. Nhân vị thế với L thì lợi suất kỳ vọng nhân L, độ lệch
   chuẩn cũng nhân L. Tỷ số không đổi. Đòn bẩy chỉ DI CHUYỂN ta dọc theo một đường
   thẳng đánh đổi, không nâng ta lên đường cao hơn. Muốn lên đường cao hơn phải tăng
   Sharpe — đó là việc của tín hiệu, danh mục và chi phí, không phải của đòn bẩy.

2. CÓ MỘT MỨC ĐÒN BẨY TỐI ƯU, VÀ VƯỢT QUÁ NÓ THÌ TĂNG ĐÒN BẨY LÀM GIẢM LỢI NHUẬN
   DÀI HẠN. Tốc độ tăng trưởng log là `L*mu - L^2*sigma^2/2`, một parabol úp ngược
   đạt đỉnh tại `L* = mu/sigma^2` (Kelly). Quá `2*L*` thì tốc độ tăng trưởng ÂM —
   tài khoản teo dần dù chiến lược có edge dương. Đây không phải lời khuyên thận
   trọng, đây là đại số.

3. LỖ VÀ LÃI KHÔNG ĐỐI XỨNG. Mất 50% cần lãi 100% để hoà. Vì vậy phân phối kết quả
   ở đòn bẩy cao có TRUNG VỊ thấp hơn nhiều so với TRUNG BÌNH: vài kịch bản cực tốt
   kéo trung bình lên trong khi phần lớn kịch bản đều tệ.

Module này KHÔNG dùng giả định phân phối chuẩn. Nó lấy mẫu lại (block bootstrap)
chính chuỗi lợi suất thật của chiến lược, nên giữ được đuôi dày và hiện tượng cụm
biến động — hai thứ mà công thức Gauss bỏ sót và luôn bỏ sót theo hướng lạc quan.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "LeverageSpec",
    "kelly_leverage",
    "volatility_target_series",
    "drawdown_throttle",
    "apply_leverage",
    "simulate_paths",
    "target_probability_table",
    "required_sharpe_for_target",
]


@dataclass
class LeverageSpec:
    """Tham số tầng đòn bẩy."""

    target_ann_vol: float = 0.30     # biến động mục tiêu của VỐN
    max_leverage: float = 3.0        # trần đòn bẩy gộp
    vol_window: int = 60             # cửa sổ ước lượng biến động thực hiện
    vol_min_periods: int = 20
    kelly_fraction: float = 0.5      # phần Kelly được dùng (0.5 = "nửa Kelly")
    dd_throttle_start: float = 0.10  # bắt đầu giảm đòn bẩy khi drawdown vượt mức này
    dd_throttle_stop: float = 0.25   # về 0 khi drawdown chạm mức này
    periods_per_year: float = 365.0


def kelly_leverage(mu_per_period: float, sigma_per_period: float,
                   fraction: float = 0.5, cap: float = 5.0) -> float:
    """
    Đòn bẩy tối ưu tăng trưởng (Kelly), nhân với `fraction`.

    `L* = mu / sigma^2`. Nửa Kelly (`fraction=0.5`) là chuẩn thực hành vì `mu` luôn
    được ước lượng với sai số lớn, và hàm tăng trưởng RẤT phẳng quanh đỉnh nhưng
    RẤT dốc phía bên phải: ước lượng `mu` cao gấp đôi khiến ta đặt đòn bẩy gấp đôi
    Kelly, và tại đúng `2L*` tốc độ tăng trưởng dài hạn rơi về 0.
    """
    if sigma_per_period <= 1e-15:
        return 0.0
    return float(np.clip(fraction * mu_per_period / sigma_per_period ** 2, 0.0, cap))


def volatility_target_series(
    returns: pd.Series,
    spec: Optional[LeverageSpec] = None,
) -> pd.Series:
    """
    Hệ số đòn bẩy theo thời gian để đưa biến động VỐN về mục tiêu.

    Biến động có tính cụm và DỰ ĐOÁN ĐƯỢC (lợi suất thì không) — đó là lý do mục
    tiêu biến động hiệu quả: nó dùng thứ duy nhất trong chuỗi lợi suất thực sự dự
    báo được. Ước lượng dùng dữ liệu tới t-1 (`shift(1)`), nên nhân quả.
    """
    spec = spec or LeverageSpec()
    realized = returns.rolling(spec.vol_window, min_periods=spec.vol_min_periods).std().shift(1)
    ann = realized * np.sqrt(spec.periods_per_year)
    lev = (spec.target_ann_vol / ann.replace(0.0, np.nan))
    return lev.clip(upper=spec.max_leverage).fillna(0.0)


def drawdown_throttle(equity_curve: pd.Series, spec: Optional[LeverageSpec] = None) -> pd.Series:
    """
    Hệ số giảm đòn bẩy khi đang trong drawdown, giảm tuyến tính về 0.

    LÝ DO KHÔNG PHẢI LÀ TÂM LÝ MÀ LÀ SỐ HỌC: sau khi mất 30%, mỗi đô-la còn lại
    phải làm việc nặng hơn 43% để về mốc cũ. Cắt vị thế trong drawdown làm chậm
    quá trình hồi phục kỳ vọng nhưng cắt hẳn phần đuôi trái — và với tài khoản nhỏ,
    phần đuôi trái là phần KẾT THÚC cuộc chơi, không phải một khoản lỗ tạm thời.

    Đánh đổi phải nói rõ: nếu drawdown chỉ là nhiễu, van tiết lưu này làm giảm lợi
    nhuận. Nó mua sự sống sót bằng lợi nhuận kỳ vọng.
    """
    spec = spec or LeverageSpec()
    peak = equity_curve.cummax()
    dd = 1.0 - equity_curve / peak.replace(0.0, np.nan)
    span = max(spec.dd_throttle_stop - spec.dd_throttle_start, 1e-9)
    factor = 1.0 - (dd - spec.dd_throttle_start) / span
    return factor.clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)


def apply_leverage(
    returns: pd.Series,
    spec: Optional[LeverageSpec] = None,
    use_vol_target: bool = True,
    use_dd_throttle: bool = True,
) -> Dict[str, pd.Series]:
    """
    Áp tầng đòn bẩy lên chuỗi lợi suất chưa đòn bẩy (gross = 1.0).

    Van drawdown phải chạy TUẦN TỰ vì đường vốn phụ thuộc chính đòn bẩy đã dùng —
    không thể vector hoá mà không nhìn trước.
    """
    spec = spec or LeverageSpec()
    r = returns.dropna()

    lev = volatility_target_series(r, spec) if use_vol_target else pd.Series(1.0, index=r.index)
    lev = lev.reindex(r.index).fillna(0.0).clip(upper=spec.max_leverage)

    out_r = np.zeros(len(r))
    out_l = np.zeros(len(r))
    equity = 1.0
    peak = 1.0
    span = max(spec.dd_throttle_stop - spec.dd_throttle_start, 1e-9)
    rv, lv = r.to_numpy(), lev.to_numpy()

    for i in range(len(r)):
        L = lv[i]
        if use_dd_throttle and peak > 0:
            dd = 1.0 - equity / peak
            if dd > spec.dd_throttle_start:
                L *= float(np.clip(1.0 - (dd - spec.dd_throttle_start) / span, 0.0, 1.0))
        step = L * rv[i]
        # Sàn -100%: tài khoản không âm được; chạm sàn là kết thúc.
        step = max(step, -0.999)
        equity *= (1.0 + step)
        peak = max(peak, equity)
        out_r[i], out_l[i] = step, L

    return {
        "returns": pd.Series(out_r, index=r.index),
        "leverage": pd.Series(out_l, index=r.index),
        "equity": pd.Series(np.cumprod(1.0 + out_r), index=r.index),
    }


# ---------------------------------------------------------------------------
# Mô phỏng phân phối kết quả
# ---------------------------------------------------------------------------
def simulate_paths(
    returns: pd.Series,
    leverage: float,
    n_periods: int,
    n_paths: int = 20000,
    block: int = 5,
    ruin_threshold: float = 0.30,
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """
    Block bootstrap phân phối kết quả sau `n_periods` kỳ ở đòn bẩy cố định.

    VÌ SAO BLOCK BOOTSTRAP CHỨ KHÔNG PHẢI CÔNG THỨC CHUẨN: lợi suất chiến lược có
    đuôi dày và biến động theo cụm. Lấy mẫu từng điểm riêng lẻ phá vỡ cụm và cho ra
    phân phối lạc quan giả tạo — đúng những chuỗi lỗ liên tiếp mới giết tài khoản
    lại là thứ bị xoá mất. Lấy mẫu theo KHỐI `block` kỳ liên tiếp giữ được cấu trúc đó.

    `ruin_threshold`: mức vốn còn lại bị coi là "cháy" (0.30 = mất 70%). Với vị thế
    có đòn bẩy trên futures, chạm ngưỡng này thường đồng nghĩa margin call hoặc
    thanh lý cưỡng bức, và đường về là không có.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < block * 3:
        raise ValueError(f"Cần ít nhất {block*3} quan sát, nhận {n}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_periods / block))
    starts = rng.integers(0, n - block, size=(n_paths, n_blocks))

    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n_periods]
    sample = r[idx] * leverage
    sample = np.maximum(sample, -0.999)

    equity = np.cumprod(1.0 + sample, axis=1)
    min_equity = np.minimum.accumulate(equity, axis=1)[:, -1]
    final = equity[:, -1]

    ruined = min_equity <= ruin_threshold
    # Cháy là hấp thụ: chạm ngưỡng thì dừng ở đó, không "hồi phục" trên giấy.
    final = np.where(ruined, ruin_threshold, final)

    return {"final": final, "min_equity": min_equity, "ruined": ruined}


def target_probability_table(
    returns: pd.Series,
    target_return: float,
    n_periods: int,
    leverages: Optional[List[float]] = None,
    n_paths: int = 20000,
    block: int = 5,
    ruin_threshold: float = 0.30,
    periods_per_year: float = 365.0,
) -> pd.DataFrame:
    """
    Bảng đánh đổi: mỗi mức đòn bẩy -> xác suất đạt mục tiêu, xác suất cháy, trung vị.

    Đây là bảng cần đọc trước khi chọn đòn bẩy. Ba cột quan trọng nhất:

      `p_target` — xác suất đạt bằng hoặc vượt mục tiêu trong khung thời gian.
      `p_ruin`   — xác suất mất tới ngưỡng cháy.
      `median`   — kết quả TRUNG VỊ, tức là điều xảy ra với một nửa số lần thử.

    Đọc cột `median` chứ đừng đọc cột `mean`. Ở đòn bẩy cao, trung bình bị kéo lên
    bởi một số ít đường cực tốt trong khi trung vị đi xuống — trung bình mô tả một
    kết quả mà hầu như không ai nhận được.
    """
    leverages = leverages or [1, 2, 3, 5, 8, 12, 20, 30]
    r = returns.dropna()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mu / sd * np.sqrt(periods_per_year) if sd > 1e-15 else 0.0
    kelly = kelly_leverage(mu, sd, fraction=1.0, cap=1e9)

    rows = []
    for L in leverages:
        sim = simulate_paths(r, L, n_periods, n_paths=n_paths, block=block,
                             ruin_threshold=ruin_threshold)
        final = sim["final"]
        rows.append({
            "leverage": float(L),
            "kelly_multiple": float(L / kelly) if kelly > 1e-9 else np.inf,
            "p_target": float((final >= 1.0 + target_return).mean()),
            "p_ruin": float(sim["ruined"].mean()),
            "p_loss": float((final < 1.0).mean()),
            "median": float(np.median(final) - 1.0),
            "mean": float(final.mean() - 1.0),
            "p05": float(np.percentile(final, 5) - 1.0),
            "p95": float(np.percentile(final, 95) - 1.0),
            "ann_vol_implied": float(L * sd * np.sqrt(periods_per_year)),
        })

    df = pd.DataFrame(rows)
    df.attrs["sharpe"] = sharpe
    df.attrs["kelly_leverage"] = kelly
    df.attrs["mu_per_period"] = mu
    df.attrs["sd_per_period"] = sd
    return df


def required_sharpe_for_target(
    target_return: float,
    n_periods: int,
    periods_per_year: float,
    confidence: float = 0.50,
    max_ann_vol: float = 0.60,
) -> Dict[str, float]:
    """
    Sharpe cần có để đạt mục tiêu với xác suất `confidence`, ở mức biến động cho trước.

    Đảo ngược câu hỏi: thay vì "chiến lược này cho bao nhiêu", hỏi "muốn con số kia
    thì cần chiến lược thế nào". Câu trả lời thường cho thấy mục tiêu đòi hỏi một
    Sharpe chưa từng tồn tại — và biết điều đó SỚM đáng giá hơn nhiều so với việc
    phát hiện nó bằng tiền thật.

    Với xấp xỉ chuẩn: `R = mu*T + z*sigma*sqrt(T)`, mà `mu = S*sigma/sqrt(ppy)`.
    """
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - confidence))
    T = n_periods
    years = T / periods_per_year
    sigma_p = max_ann_vol / np.sqrt(periods_per_year)

    # target = S*sigma_p/1 * T + z*sigma_p*sqrt(T)  (S theo đơn vị năm -> mu = S*sigma_ann/ppy)
    mu_needed = (target_return - z * sigma_p * np.sqrt(T)) / T
    sharpe_needed = mu_needed / sigma_p * np.sqrt(periods_per_year)

    return {
        "target_return": target_return,
        "horizon_periods": float(T),
        "horizon_years": float(years),
        "assumed_ann_vol": max_ann_vol,
        "confidence": confidence,
        "required_ann_sharpe": float(sharpe_needed),
        "required_ann_return": float(mu_needed * periods_per_year),
        "note": "Sharpe > 5 duy trì được là ngoài phạm vi mọi quỹ công khai; "
                "Sharpe > 10 chỉ tồn tại ở market-making tần suất cao với hạ tầng riêng.",
    }
