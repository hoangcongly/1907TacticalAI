"""
Nạp panel đa tài sản từ MỘT nguồn duy nhất (nến 1h) và tổng hợp lên khung bất kỳ.

VÌ SAO MỘT NGUỒN: hệ thống cũ tải riêng từng khung thời gian. Hệ quả là nến 4h và
nến 1h của cùng một cặp có thể lệch nhau về số lượng, về mốc biên, và — nghiêm
trọng nhất — OFI của khung này không tổng hợp được từ khung kia, nên nghiên cứu ở
hai khung không so sánh được với nhau. Tải 1h rồi tự tổng hợp làm mọi khung nhất
quán theo đúng nghĩa toán học.

QUY TẮC TỔNG HỢP (mỗi trường một quy tắc, không có cái nào là "trung bình"):
  open  -> giá trị ĐẦU     high -> lớn nhất    low -> nhỏ nhất    close -> giá trị CUỐI
  volume, tick_count -> tổng
  ofi   -> trung bình có TRỌNG SỐ KHỐI LƯỢNG. OFI là tỷ lệ, trung bình đơn giản sẽ
           cho nến mỏng thanh khoản cùng tiếng nói với nến dày — sai về bản chất.
"""

import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_ROOT = "data/binance"

INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}

__all__ = [
    "INTERVAL_MS",
    "resample_bars",
    "load_panel_v2",
    "load_funding_panel_v2",
    "align_panel",
    "interval_hours",
    "load_metrics_panel_v2",
    "METRICS_FIELDS",
]


def interval_hours(interval: str) -> float:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Khung không hỗ trợ: {interval}")
    return INTERVAL_MS[interval] / 3_600_000


def resample_bars(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    """
    Tổng hợp nến từ khung `source` lên khung `target`.

    Mốc nhóm neo vào epoch UTC (`timestamp_ms // step * step`) chứ không neo vào
    nến đầu tiên của chuỗi — nhờ vậy mọi cặp dùng CHUNG lưới thời gian dù ngày
    niêm yết khác nhau. Nếu neo theo chuỗi, hai cặp niêm yết lệch 1 giờ sẽ có nến
    4h lệch pha và mọi so sánh cross-sectional trở nên vô nghĩa.
    """
    src_ms, tgt_ms = INTERVAL_MS[source], INTERVAL_MS[target]
    if tgt_ms == src_ms:
        return df.copy()
    if tgt_ms % src_ms != 0:
        raise ValueError(f"{target} không chia hết cho {source}")

    d = df.copy()
    d["bucket"] = (d["timestamp_ms"] // tgt_ms) * tgt_ms
    g = d.groupby("bucket", sort=True)

    out = pd.DataFrame({
        "timestamp_ms": g["timestamp_ms"].first().index.to_numpy(),
        "open": g["open"].first().to_numpy(),
        "high": g["high"].max().to_numpy(),
        "low": g["low"].min().to_numpy(),
        "close": g["close"].last().to_numpy(),
        "volume": g["volume"].sum().to_numpy(),
    })
    if "tick_count" in d.columns:
        out["tick_count"] = g["tick_count"].sum().to_numpy()

    if "ofi" in d.columns:
        # OFI có trọng số khối lượng: sum(ofi_i * vol_i) / sum(vol_i).
        d["_ofi_vol"] = d["ofi"] * d["volume"]
        num = g["_ofi_vol"].sum()
        den = g["volume"].sum().replace(0.0, np.nan)
        out["ofi"] = (num / den).fillna(0.0).to_numpy()

    # Chỉ giữ nhóm ĐẦY ĐỦ ở cuối chuỗi: nến target cuối cùng thường mới hình thành
    # một phần, dùng nó là nhìn vào tương lai chưa xảy ra xong.
    expected = tgt_ms // src_ms
    counts = g.size().to_numpy()
    if len(counts) and counts[-1] < expected:
        out = out.iloc[:-1]

    return out.reset_index(drop=True)


def load_panel_v2(
    symbols: List[str],
    interval: str = "4h",
    source_interval: str = "1h",
    root: str = DATA_ROOT,
    fields: Optional[List[str]] = None,
    min_coverage: float = 0.15,
    min_bars: int = 200,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Nạp panel {trường: DataFrame(time x symbol)} ở khung `interval`.

    Ưu tiên tổng hợp từ `source_interval`; nếu cặp không có file nguồn thì rơi về
    file đúng khung đã tải sẵn. `min_coverage` loại cặp thiếu dữ liệu quá nhiều —
    vừa chống NaN lan, vừa chống thiên vị sống sót ngược (một cặp chỉ có dữ liệu ở
    giai đoạn nó chạy tốt).
    """
    fields = fields or ["open", "high", "low", "close", "volume", "ofi", "tick_count"]
    frames: Dict[str, Dict[str, pd.Series]] = {f: {} for f in fields}
    root_p = pathlib.Path(root)

    for symbol in symbols:
        src = root_p / f"{symbol}_{source_interval}.parquet"
        direct = root_p / f"{symbol}_{interval}.parquet"

        if src.is_file():
            raw = pd.read_parquet(src).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
            df = resample_bars(raw, source_interval, interval)
        elif direct.is_file():
            df = pd.read_parquet(direct).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
        else:
            continue

        if len(df) < min_bars:
            continue
        df = df.set_index("timestamp_ms")
        for f in fields:
            if f in df.columns:
                frames[f][symbol] = df[f].astype(np.float64)

    panel = {f: pd.DataFrame(cols).sort_index() for f, cols in frames.items() if cols}
    if "close" not in panel:
        raise RuntimeError("Không nạp được panel — chạy scripts/download_wide_universe.py trước")

    if start_ms is not None or end_ms is not None:
        lo = start_ms if start_ms is not None else -np.inf
        hi = end_ms if end_ms is not None else np.inf
        panel = {f: d[(d.index >= lo) & (d.index <= hi)] for f, d in panel.items()}

    coverage = panel["close"].notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    return {f: d[[c for c in keep if c in d.columns]] for f, d in panel.items()}


def load_funding_panel_v2(
    symbols: List[str],
    close_index: pd.Index,
    root: str = DATA_ROOT,
) -> pd.DataFrame:
    """
    Panel funding rate căn theo lưới giá.

    Funding là SỰ KIỆN RỜI RẠC mỗi 8h — giữ nguyên giá trị (ffill) tới mốc kế tiếp,
    tuyệt đối không nội suy. Nội suy sẽ cho tín hiệu biết trước một phần mức funding
    chưa được công bố.
    """
    series: Dict[str, pd.Series] = {}
    idx = pd.Index(close_index)
    for symbol in symbols:
        path = pathlib.Path(root) / f"{symbol}_funding.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path).drop_duplicates("funding_time").sort_values("funding_time")
        s = pd.Series(df["rate"].to_numpy(), index=df["funding_time"].to_numpy())
        union = s.index.union(idx)
        series[symbol] = s.reindex(union).ffill().reindex(idx)
    return pd.DataFrame(series)


def funding_last_ms(symbols: List[str], root: str = DATA_ROOT) -> pd.Series:
    """
    Mốc funding MỚI NHẤT có trên đĩa cho từng cặp (ms). Thiếu file -> NaN.

    [FIX F52] `load_funding_panel_v2` ffill không giới hạn, nên panel nó trả về KHÔNG
    phân biệt được "funding vừa công bố" với "funding đứng yên từ hai tuần trước".
    Độ tươi phải đo ở NGUỒN, trước khi ffill xoá mất thông tin đó.
    """
    out = {}
    for symbol in symbols:
        path = pathlib.Path(root) / f"{symbol}_funding.parquet"
        if path.is_file():
            ft = pd.read_parquet(path, columns=["funding_time"])["funding_time"]
            out[symbol] = float(ft.max()) if len(ft) else np.nan
        else:
            out[symbol] = np.nan
    return pd.Series(out, dtype=float)


def align_panel(panel: Dict[str, pd.DataFrame], funding: pd.DataFrame):
    """Ép mọi trường về cùng bộ cột và cùng thứ tự — tránh lệch cột âm thầm."""
    cols = [c for c in panel["close"].columns if c in funding.columns]
    if not cols:
        cols = list(panel["close"].columns)
    out = {f: d.reindex(columns=cols) for f, d in panel.items()}
    return out, funding.reindex(columns=cols)


METRICS_ROOT = "data/binance_metrics"

#: Trường vị thế lấy từ kho dump `data.binance.vision/.../metrics`.
#: Đây là những đại lượng mà GIÁ và KHỐI LƯỢNG không chứa: ai đang cầm gì.
METRICS_FIELDS = [
    "sum_open_interest",                 # OI theo đơn vị cơ sở
    "sum_open_interest_value",           # OI theo USD
    "count_toptrader_long_short_ratio",  # long/short theo SỐ TÀI KHOẢN của top trader
    "sum_toptrader_long_short_ratio",    # long/short theo VỊ THẾ của top trader
    "count_long_short_ratio",            # long/short của TOÀN BỘ tài khoản (đám đông)
    "sum_taker_long_short_vol_ratio",    # khối lượng taker mua/bán
]


def load_metrics_panel_v2(
    symbols: List[str],
    close_index: pd.Index,
    root: str = METRICS_ROOT,
    interval: str = "4h",
) -> Dict[str, pd.DataFrame]:
    """
    Panel dữ liệu VỊ THẾ, căn theo lưới giá — trường thiếu trả về khung rỗng.

    QUY ƯỚC THỜI GIAN PHẢI KHỚP `resample_bars`, nếu không sẽ nhìn trước cả một nến:
    mỗi trường ở đây là MỨC TỒN (số hợp đồng đang mở tại một thời điểm), nên giá trị
    của nến T là quan sát tại CUỐI nến T — y hệt `close`. Vì `simulate` cho trọng số
    tại T ăn lợi suất T -> T+1, dùng giá trị cuối-nến-T tại chỉ số T là nhân quả.

    Nếu ai đó đổi `_aggregate` trong `scripts/download_metrics.py` sang lấy giá trị
    ĐẦU kỳ thì quy ước gãy và mọi tín hiệu vị thế trễ đúng một nến — không crash, chỉ
    yếu đi. Hai nơi phải sửa cùng nhau.

    Thiếu dữ liệu KHÔNG phải lỗi: chưa tải metrics thì họ tín hiệu vị thế tự tắt
    (toàn NaN) và tầng gộp thích ứng bỏ qua nó. Hệ thống vẫn chạy y như trước.
    """
    idx = pd.Index(close_index)
    root_p = pathlib.Path(root)
    cols: Dict[str, Dict[str, pd.Series]] = {f: {} for f in METRICS_FIELDS}

    for symbol in symbols:
        path = root_p / f"{symbol}_{interval}.parquet"
        if not path.is_file():
            continue
        try:
            df = pd.read_parquet(path).drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
        except Exception:
            continue
        df = df.set_index("timestamp_ms")
        for f in METRICS_FIELDS:
            if f in df.columns:
                # reindex thẳng, KHÔNG ffill: thiếu là thiếu. Điền tới sẽ khiến một
                # cặp ngừng báo cáo vẫn có "vị thế" y như lần cuối, mãi mãi.
                cols[f][symbol] = df[f].astype(np.float64).reindex(idx)

    return {f: (pd.DataFrame(c).reindex(index=idx) if c
                else pd.DataFrame(index=idx, dtype=np.float64))
            for f, c in cols.items()}
