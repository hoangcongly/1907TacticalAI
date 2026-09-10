"""
Thư viện tín hiệu cross-sectional, tổ chức theo HỌ KINH TẾ.

PHƯƠNG PHÁP — vì sao tổ chức theo họ chứ không phải một danh sách phẳng:

Ném 30 tín hiệu vào dữ liệu rồi giữ lại cái nào Sharpe cao nhất là công thức chuẩn
để overfit. Với 30 phép thử độc lập trên nhiễu thuần tuý, t-stat lớn nhất kỳ vọng
đã khoảng 2.5 — tức là ta sẽ luôn tìm được "tín hiệu có ý nghĩa thống kê" kể cả khi
không có gì.

Cách làm ở đây, theo đúng quy trình các quỹ định lượng dùng:

1. Mỗi tín hiệu phải xuất phát từ một GIẢ THUYẾT KINH TẾ nêu trước, có dấu kỳ vọng
   nêu trước. Không có giả thuyết thì không đưa vào thư viện.
2. Các biến thể cùng một giả thuyết (động lượng 6/12/30/90/180 nến) được gộp thành
   MỘT tín hiệu cấp họ bằng cách trung bình z-score. Điều này khử luôn việc chọn
   tham số — thứ chiếm phần lớn overfit trong nghiên cứu nhân tố.
3. Chỉ ĐÁNH GIÁ ở cấp họ. Số phép thử độc lập rơi từ ~30 xuống ~6, và ngưỡng t-stat
   cần thiết giảm theo.
4. Sharpe cuối cùng phải bị CHIẾT KHẤU theo số phép thử (`validation/dsr.py`).

Mọi tín hiệu đều nhân quả: giá trị tại t chỉ dùng dữ liệu tới và bằng t. Việc dịch
một nhịp khi vào lệnh do engine backtest đảm nhiệm, không làm ở đây.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "SignalDef",
    "SIGNAL_REGISTRY",
    "FAMILIES",
    "xs_zscore",
    "xs_rank",
    "build_signal",
    "build_family",
    "build_all_families",
]


# ---------------------------------------------------------------------------
# Chuẩn hoá mặt cắt ngang
# ---------------------------------------------------------------------------
def xs_zscore(df: pd.DataFrame, clip: float = 4.0) -> pd.DataFrame:
    """
    Z-score theo HÀNG. Kẹp đuôi trước khi chuẩn hoá lại để một giá trị ngoại lai
    không kéo lệch trung bình/độ lệch chuẩn của cả mặt cắt ngang.
    """
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0.0, np.nan), axis=0)
    z = z.clip(-clip, clip)
    return z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1).replace(0.0, np.nan), axis=0)


def xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thứ hạng theo hàng, ánh xạ về [-1, 1].

    Bền hơn z-score khi phân phối có đuôi rất dày (đúng với hầu hết đại lượng
    crypto: khối lượng, độ lệch, funding). Đánh đổi: mất thông tin về ĐỘ LỚN.
    """
    r = df.rank(axis=1, pct=True)
    return (r - 0.5) * 2.0


# ---------------------------------------------------------------------------
# Khai báo tín hiệu
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalDef:
    """Một tín hiệu: giả thuyết, họ, hàm tính, và số nến cần khởi động."""

    name: str
    family: str
    hypothesis: str          # giả thuyết kinh tế, nêu TRƯỚC khi chạy
    fn: Callable[..., pd.DataFrame]
    warmup: int
    use_rank: bool = False   # chuẩn hoá bằng thứ hạng thay vì z-score


SIGNAL_REGISTRY: Dict[str, SignalDef] = {}


def _register(name: str, family: str, hypothesis: str, warmup: int, use_rank: bool = False):
    def deco(fn):
        SIGNAL_REGISTRY[name] = SignalDef(name, family, hypothesis, fn, warmup, use_rank)
        return fn
    return deco


def _ret(close: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return close / close.shift(n) - 1.0


def _log_ret(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close / close.shift(1))


# ===========================================================================
# HỌ 1 — CARRY (funding)
# Giả thuyết chung: funding perp là DÒNG TIỀN CƠ HỌC, không phải dự báo giá. Bên
# nào đông người chen chúc thì bên đó trả tiền. Đây là họ bền nhất vì nó không
# đòi hỏi dự đoán đúng hướng giá.
# ===========================================================================
@_register("carry_level", "carry",
           "Funding dương cao = long chen chúc = phải trả phí. Short thu funding đó.",
           warmup=48)
def _carry_level(panel, funding, smooth: int = 24):
    return -funding.rolling(smooth, min_periods=smooth // 3).mean()


@_register("carry_level_slow", "carry",
           "Cùng giả thuyết carry_level nhưng chân trời dài hơn — bắt trạng thái chen chúc dai dẳng.",
           warmup=180)
def _carry_level_slow(panel, funding, smooth: int = 120):
    return -funding.rolling(smooth, min_periods=smooth // 3).mean()


@_register("carry_zscore", "carry",
           "Funding lệch so với CHUẨN RIÊNG của từng cặp. Một số cặp luôn có funding cao "
           "về cấu trúc; cái đáng giao dịch là độ lệch, không phải mức tuyệt đối.",
           warmup=360)
def _carry_zscore(panel, funding, window: int = 240):
    m = funding.rolling(window, min_periods=window // 4).mean()
    s = funding.rolling(window, min_periods=window // 4).std()
    return -(funding - m) / s.replace(0.0, np.nan)


@_register("carry_momentum", "carry",
           "Funding đang TĂNG = đòn bẩy đang tích tụ, thường đi trước đợt thanh lý. "
           "Khác carry_level ở chỗ dùng thay đổi thay vì mức.",
           warmup=200)
def _carry_momentum(panel, funding, lookback: int = 168):
    sm = funding.rolling(24, min_periods=8).mean()
    return -(sm - sm.shift(lookback))


@_register("carry_vol", "carry",
           "Funding biến động mạnh = định vị bất ổn = phần bù rủi ro cao hơn cho bên cấp thanh khoản.",
           warmup=360, use_rank=True)
def _carry_vol(panel, funding, window: int = 240):
    return -funding.rolling(window, min_periods=window // 4).std()


# ===========================================================================
# HỌ 2 — MOMENTUM (xu hướng giá)
# Giả thuyết chung: dòng tiền vào tài sản crypto có quán tính (phản ứng chậm với
# thông tin, hành vi bầy đàn, tái cân bằng của quỹ). Tài sản THẮNG TƯƠNG ĐỐI có xu
# hướng tiếp tục thắng tương đối trong chân trời vài ngày tới vài tháng.
# LƯU Ý: ở crypto, khác cổ phiếu, chân trời NGẮN vẫn là động lượng chứ không đảo
# chiều — bằng chứng: `reversal_6` trong nghiên cứu cũ có Sharpe -1.73, tức là
# chiều ngược lại (động lượng 6 nến) mạnh dương.
# ===========================================================================
@_register("mom_fast", "momentum",
           "Động lượng chân trời 1 ngày: dòng tiền vào có quán tính ngắn hạn.",
           warmup=12)
def _mom_fast(panel, funding, lookback: int = 6, skip: int = 0):
    c = panel["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@_register("mom_mid", "momentum",
           "Động lượng chân trời 1 tuần.",
           warmup=48)
def _mom_mid(panel, funding, lookback: int = 42, skip: int = 1):
    c = panel["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@_register("mom_slow", "momentum",
           "Động lượng chân trời 2 tuần tới 1 tháng — chân trời kinh điển của nhân tố động lượng.",
           warmup=100)
def _mom_slow(panel, funding, lookback: int = 90, skip: int = 1):
    c = panel["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@_register("mom_vlong", "momentum",
           "Động lượng chân trời quý.",
           warmup=200)
def _mom_vlong(panel, funding, lookback: int = 180, skip: int = 1):
    c = panel["close"]
    return c.shift(skip) / c.shift(skip + lookback) - 1.0


@_register("mom_risk_adj", "momentum",
           "Động lượng chia cho biến động: tách xu hướng thật khỏi phần chỉ là biến động lớn. "
           "Động lượng thô luôn thiên vị tài sản rủi ro nhất.",
           warmup=120)
def _mom_risk_adj(panel, funding, lookback: int = 90, vol_window: int = 60):
    c = panel["close"]
    mom = c.shift(1) / c.shift(1 + lookback) - 1.0
    vol = _log_ret(c).rolling(vol_window, min_periods=vol_window // 3).std()
    return mom / vol.replace(0.0, np.nan)


@_register("mom_consistency", "momentum",
           "TỶ LỆ nến tăng, không phải tổng lợi suất. Xu hướng do nhiều bước nhỏ đều đặn "
           "bền hơn xu hướng do một cú nhảy — cú nhảy thường là tin một lần, đã phản ánh hết.",
           warmup=120)
def _mom_consistency(panel, funding, lookback: int = 90):
    r = _log_ret(panel["close"])
    return (r > 0).rolling(lookback, min_periods=lookback // 3).mean() - 0.5


@_register("mom_52w_high", "momentum",
           "Khoảng cách tới đỉnh cao nhất trong khung nhìn. Neo tâm lý: tài sản sát đỉnh "
           "gặp lực bán chốt lời chậm hơn, nên phản ứng chậm với tin tốt.",
           warmup=760)
def _mom_52w_high(panel, funding, window: int = 720):
    c = panel["close"]
    hi = panel["high"].rolling(window, min_periods=window // 4).max()
    return c / hi.replace(0.0, np.nan) - 1.0


# ===========================================================================
# HỌ 3 — FLOW (dòng lệnh / khối lượng)
# Giả thuyết chung: mất cân bằng lệnh chủ động để lại dấu vết. Bên chủ động là bên
# có nhu cầu gấp; nhu cầu gấp thường đến từ thông tin hoặc buộc phải thanh lý — cả
# hai đều dự báo được bước giá kế tiếp trong ngắn hạn.
# ===========================================================================
@_register("ofi_fast", "flow",
           "Mất cân bằng dòng lệnh chủ động, chân trời nửa ngày.",
           warmup=12)
def _ofi_fast(panel, funding, window: int = 6):
    return panel["ofi"].rolling(window, min_periods=2).mean()


@_register("ofi_mid", "flow",
           "Mất cân bằng dòng lệnh, chân trời 1-2 ngày.",
           warmup=48)
def _ofi_mid(panel, funding, window: int = 30):
    return panel["ofi"].rolling(window, min_periods=window // 3).mean()


@_register("ofi_slow", "flow",
           "Mất cân bằng dòng lệnh tích luỹ dài hạn — đại diện cho tích luỹ/phân phối.",
           warmup=180)
def _ofi_slow(panel, funding, window: int = 120):
    return panel["ofi"].rolling(window, min_periods=window // 3).mean()


@_register("ofi_persistence", "flow",
           "Tự tương quan dấu dòng lệnh. Dòng lệnh DAI DẲNG là dấu vết của lệnh lớn bị "
           "chia nhỏ (order splitting); dòng lệnh ngẫu nhiên chỉ là nhiễu.",
           warmup=120, use_rank=True)
def _ofi_persistence(panel, funding, window: int = 90):
    o = panel["ofi"]
    sign = np.sign(o)
    return (sign * sign.shift(1)).rolling(window, min_periods=window // 3).mean()


@_register("volume_shock", "flow",
           "Khối lượng vọt so với chuẩn riêng. Đột biến khối lượng đi kèm giá tăng "
           "xác nhận dòng tiền thật, không phải nhiễu thanh khoản mỏng.",
           warmup=240)
def _volume_shock(panel, funding, short: int = 24, long: int = 168):
    v = panel["volume"] * panel["close"]   # khối lượng quy USD
    s = v.rolling(short, min_periods=short // 3).mean()
    l = v.rolling(long, min_periods=long // 3).mean()
    ratio = np.log(s / l.replace(0.0, np.nan))
    direction = np.sign(_ret(panel["close"], short))
    return ratio * direction


@_register("amihud_illiq", "flow",
           "Kém thanh khoản (Amihud): giá dịch nhiều trên mỗi đô-la giao dịch. Phần bù "
           "thanh khoản dương — tài sản kém thanh khoản kiếm nhiều hơn để bù rủi ro đó. "
           "Ở vốn nhỏ ta CÓ THỂ thu phần bù này vì lệnh của ta không đủ lớn để trả nó.",
           warmup=240, use_rank=True)
def _amihud_illiq(panel, funding, window: int = 168):
    dv = (panel["volume"] * panel["close"]).replace(0.0, np.nan)
    illiq = _ret(panel["close"]).abs() / dv
    return illiq.rolling(window, min_periods=window // 3).median()


# ===========================================================================
# HỌ 4 — VOLATILITY / RISK
# Giả thuyết chung: nhà đầu tư nhỏ lẻ trả quá cao cho tài sản "vé số" (biến động
# mạnh, độ lệch dương, xác suất nhỏ thắng lớn). Phần bù cho những tài sản đó là ÂM.
# Đây là bất thường được ghi nhận rộng nhất ở mọi lớp tài sản, và crypto là nơi
# thành phần bán lẻ đậm đặc nhất nên hiệu ứng mạnh nhất.
# ===========================================================================
@_register("low_idio_vol", "volatility",
           "Bất thường biến động riêng: mua tài sản có biến động ĐÃ KHỬ THỊ TRƯỜNG thấp.",
           warmup=120, use_rank=True)
def _low_idio_vol(panel, funding, window: int = 90):
    r = _log_ret(panel["close"])
    idio = r.sub(r.mean(axis=1), axis=0)
    return -idio.rolling(window, min_periods=window // 3).std()


@_register("neg_skew", "volatility",
           "Cầu xổ số: bán tài sản có độ lệch DƯƠNG (đuôi phải béo), mua tài sản lệch âm.",
           warmup=240, use_rank=True)
def _neg_skew(panel, funding, window: int = 168):
    return -_log_ret(panel["close"]).rolling(window, min_periods=window // 3).skew()


@_register("low_max_ret", "volatility",
           "MAX-effect (Bali-Cakici-Whitelaw): bán tài sản có nến tăng mạnh nhất gần đây. "
           "Đại diện trực tiếp nhất cho 'vé số' và thường mạnh hơn cả độ lệch.",
           warmup=240, use_rank=True)
def _low_max_ret(panel, funding, window: int = 168, top_k: int = 5):
    r = _log_ret(panel["close"])
    return -r.rolling(window, min_periods=window // 3).max()


@_register("vol_of_vol", "volatility",
           "Bán tài sản có BIẾN ĐỘNG CỦA BIẾN ĐỘNG cao — rủi ro không định giá được, "
           "nhà đầu tư đòi phần bù nhưng thường trả quá cao cho khả năng bùng nổ.",
           warmup=360, use_rank=True)
def _vol_of_vol(panel, funding, short: int = 24, window: int = 240):
    v = _log_ret(panel["close"]).rolling(short, min_periods=short // 2).std()
    return -v.rolling(window, min_periods=window // 3).std() / v.rolling(
        window, min_periods=window // 3).mean().replace(0.0, np.nan)


@_register("downside_beta", "volatility",
           "Bán tài sản có beta CHIỀU XUỐNG cao. Beta hai chiều che mất thứ thật sự "
           "gây đau: tài sản rơi nhanh hơn thị trường nhưng không tăng nhanh hơn.",
           warmup=360)
def _downside_beta(panel, funding, window: int = 240):
    r = _log_ret(panel["close"])
    mkt = r.mean(axis=1)
    down = mkt < 0
    rd = r.where(down)
    md = mkt.where(down)
    cov = rd.rolling(window, min_periods=window // 4).cov(md)
    var = md.rolling(window, min_periods=window // 4).var()
    return -cov.div(var.replace(0.0, np.nan), axis=0)


# ===========================================================================
# HỌ 5 — MICROSTRUCTURE (hình dạng trong nến)
# Giả thuyết chung: vị trí giá đóng cửa trong biên độ nến cho biết ai KIỂM SOÁT
# cuối phiên. Đóng cửa sát đỉnh sau khi quét đáy = lực mua hấp thụ nguồn cung.
# Đây là thông tin không có trong chuỗi close-to-close.
# ===========================================================================
@_register("close_location", "microstructure",
           "Vị trí đóng cửa trong biên độ high-low. Đóng sát đỉnh = phe mua kiểm soát.",
           warmup=48)
def _close_location(panel, funding, window: int = 24):
    hi, lo, c = panel["high"], panel["low"], panel["close"]
    rng = (hi - lo).replace(0.0, np.nan)
    clv = ((c - lo) - (hi - c)) / rng
    return clv.rolling(window, min_periods=window // 3).mean()


@_register("range_compression", "microstructure",
           "Biên độ thu hẹp so với chuẩn riêng. Nén biên độ đi trước bùng nổ; kết hợp "
           "với chiều động lượng cho điểm vào tốt hơn động lượng đơn thuần.",
           warmup=240)
def _range_compression(panel, funding, short: int = 24, long: int = 168):
    tr = (panel["high"] - panel["low"]) / panel["close"].replace(0.0, np.nan)
    s = tr.rolling(short, min_periods=short // 3).mean()
    l = tr.rolling(long, min_periods=long // 3).mean()
    compression = -np.log(s / l.replace(0.0, np.nan))
    return compression * np.sign(_ret(panel["close"], long))


@_register("trade_intensity", "microstructure",
           "Số lệnh trên mỗi đơn vị khối lượng: kích thước lệnh trung bình NHỎ = nhiều "
           "nhà đầu tư nhỏ lẻ. Dòng tiền bán lẻ dồn dập thường đi trước đảo chiều.",
           warmup=240, use_rank=True)
def _trade_intensity(panel, funding, window: int = 168):
    if "tick_count" not in panel:
        return pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)
    dv = (panel["volume"] * panel["close"]).replace(0.0, np.nan)
    avg_size = dv / panel["tick_count"].replace(0, np.nan)
    z = np.log(avg_size.replace(0.0, np.nan))
    m = z.rolling(window, min_periods=window // 3).mean()
    s = z.rolling(window, min_periods=window // 3).std()
    return (z - m) / s.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Dựng tín hiệu
# ---------------------------------------------------------------------------
FAMILIES: List[str] = ["carry", "momentum", "flow", "volatility", "microstructure"]


def build_signal(name: str, panel: Dict[str, pd.DataFrame], funding: pd.DataFrame,
                 **kwargs) -> pd.DataFrame:
    """Tính MỘT tín hiệu và chuẩn hoá theo mặt cắt ngang."""
    if name not in SIGNAL_REGISTRY:
        raise KeyError(f"Tín hiệu không có trong registry: {name}. Có: {sorted(SIGNAL_REGISTRY)}")
    spec = SIGNAL_REGISTRY[name]
    raw = spec.fn(panel, funding, **kwargs)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return xs_rank(raw) if spec.use_rank else xs_zscore(raw)


def build_family(family: str, panel: Dict[str, pd.DataFrame], funding: pd.DataFrame,
                 members: Optional[List[str]] = None,
                 min_members: int = 1) -> pd.DataFrame:
    """
    Gộp mọi biến thể trong một họ thành MỘT tín hiệu cấp họ.

    Trung bình các z-score đã chuẩn hoá, yêu cầu tối thiểu `min_members` biến thể có
    dữ liệu tại mỗi ô. Đây là bước khử việc chọn tham số: ta không còn phải quyết
    định "động lượng 90 hay 180 nến" — cả hai đều vào với trọng số bằng nhau.
    """
    names = members or [n for n, s in SIGNAL_REGISTRY.items() if s.family == family]
    if not names:
        raise ValueError(f"Họ rỗng: {family}")

    frames = [build_signal(n, panel, funding) for n in names]
    stacked = pd.concat(frames)
    total = stacked.groupby(level=0).sum(min_count=1)
    count = pd.concat([f.notna().astype(float) for f in frames]).groupby(level=0).sum()

    avg = total / count.replace(0.0, np.nan)
    avg = avg.where(count >= min_members)
    return xs_zscore(avg)


def build_all_families(panel: Dict[str, pd.DataFrame], funding: pd.DataFrame,
                       families: Optional[List[str]] = None,
                       min_members: int = 1) -> Dict[str, pd.DataFrame]:
    """Dựng tín hiệu cấp họ cho mọi họ — đây là đầu vào của tầng gộp."""
    return {f: build_family(f, panel, funding, min_members=min_members)
            for f in (families or FAMILIES)}
