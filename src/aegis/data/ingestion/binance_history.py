"""
[FIX F2] Tải lịch sử nến + funding THẬT từ Binance và lưu parquet.

Thay thế `generate_synthetic_signal_bars()` — nguồn random walk mà `main.py` đang
dùng cho chế độ live. Mọi con số backtest chạy trên dữ liệu giả đều vô nghĩa.

Ghi parquet phân vùng theo cặp/khung thời gian, chống ghi trùng bằng khoá
`open_time` để tải lại nhiều lần vẫn cho kết quả giống hệt (idempotent).
"""

import logging
import pathlib
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from aegis.data.ingestion.binance_rest import BinanceFuturesREST

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "data/binance"

# Cột thô Binance trả về cho mỗi nến.
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "tick_count", "taker_buy_base", "taker_buy_quote", "ignore",
]

_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


def interval_to_ms(interval: str) -> int:
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Khung thời gian không hỗ trợ: {interval}. Có: {list(_INTERVAL_MS)}")
    return _INTERVAL_MS[interval]


def klines_to_frame(rows: List[list]) -> pd.DataFrame:
    """
    Chuyển nến thô Binance thành DataFrame OHLCV chuẩn cho signal_pipeline.

    Bao gồm `ofi` xấp xỉ từ taker buy/sell volume — mất cân bằng dòng lệnh, đại
    lượng vi cấu trúc thật sự có giá trị dự báo ở quy mô nhỏ.
    """
    if not rows:
        return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close",
                                     "volume", "tick_count", "ofi"])

    df = pd.DataFrame(rows, columns=_KLINE_COLUMNS)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base", "taker_buy_quote"]
    df[numeric] = df[numeric].astype(np.float64)

    out = pd.DataFrame({
        "timestamp_ms": df["open_time"].astype(np.int64),
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "tick_count": df["tick_count"].astype(np.int64),
    })

    # OFI = (mua chủ động - bán chủ động) / tổng khối lượng, kẹp về [-1, 1].
    taker_buy = df["taker_buy_base"].to_numpy()
    total = df["volume"].to_numpy()
    taker_sell = total - taker_buy
    with np.errstate(divide="ignore", invalid="ignore"):
        ofi = np.where(total > 0, (taker_buy - taker_sell) / total, 0.0)
    out["ofi"] = np.clip(np.nan_to_num(ofi, nan=0.0), -1.0, 1.0)

    return out


def download_klines(
    symbol: str,
    interval: str = "1m",
    days: float = 30.0,
    client: Optional[BinanceFuturesREST] = None,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Tải `days` ngày nến gần nhất, tự phân trang qua trần 1500 nến/request.

    Trả về DataFrame đã sắp xếp theo thời gian, loại bỏ trùng lặp.
    """
    client = client or BinanceFuturesREST()
    step_ms = interval_to_ms(interval)
    end = int(end_ms or time.time() * 1000)
    start = end - int(days * 86_400_000)

    chunks: List[pd.DataFrame] = []
    cursor = start
    n_requests = 0

    while cursor < end:
        rows = client.klines(symbol, interval=interval, start_ms=cursor, end_ms=end, limit=1500)
        n_requests += 1
        if not rows:
            break

        frame = klines_to_frame(rows)
        chunks.append(frame)

        last_open = int(rows[-1][0])
        if last_open <= cursor:  # không tiến thêm được -> dừng, tránh lặp vô hạn
            break
        cursor = last_open + step_ms

        if len(rows) < 1500:  # đã chạm cuối dữ liệu sẵn có
            break

    if not chunks:
        raise RuntimeError(f"Không tải được nến nào cho {symbol} {interval}")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp_ms", keep="last")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    logger.info("Tải %d nến %s %s qua %d request", len(df), symbol, interval, n_requests)
    return df


def download_funding_rates(
    symbol: str,
    days: float = 30.0,
    client: Optional[BinanceFuturesREST] = None,
    end_ms: Optional[int] = None,
) -> Dict[int, float]:
    """
    Lịch sử funding rate THẬT -> dict {timestamp_ms: rate}.

    Đây đúng định dạng `governance.funding_accrual.compute_funding_accrued_usd`
    nhận vào, thay cho hằng số mặc định DEFAULT_FUNDING_RATE.
    """
    client = client or BinanceFuturesREST()
    end = int(end_ms or time.time() * 1000)
    start = end - int(days * 86_400_000)

    rates: Dict[int, float] = {}
    cursor = start
    while cursor < end:
        rows = client.funding_rate_history(symbol, start_ms=cursor, end_ms=end, limit=1000)
        if not rows:
            break
        for row in rows:
            rates[int(row["fundingTime"])] = float(row["fundingRate"])
        last = int(rows[-1]["fundingTime"])
        if last <= cursor:
            break
        cursor = last + 1
        if len(rows) < 1000:
            break

    logger.info("Tải %d mốc funding cho %s", len(rates), symbol)
    return rates


# ============================================================================
# Lưu trữ parquet
# ============================================================================
def _parquet_path(symbol: str, interval: str, root: str = DEFAULT_DATA_ROOT) -> pathlib.Path:
    return pathlib.Path(root) / f"{symbol.upper()}_{interval}.parquet"


def save_klines(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "1m",
    root: str = DEFAULT_DATA_ROOT,
) -> pathlib.Path:
    """
    Ghi/gộp nến vào parquet. Gộp theo `timestamp_ms` nên tải chồng lặp vẫn an toàn:
    chạy lại nhiều lần cho ra đúng một kết quả (idempotent).
    """
    path = _parquet_path(symbol, interval, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)

    df = (
        df.drop_duplicates(subset="timestamp_ms", keep="last")
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )
    df.to_parquet(path, index=False)
    logger.info("Lưu %d nến -> %s", len(df), path)
    return path


def load_klines(
    symbol: str,
    interval: str = "1m",
    root: str = DEFAULT_DATA_ROOT,
) -> pd.DataFrame:
    path = _parquet_path(symbol, interval, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có dữ liệu {symbol} {interval} tại {path}. "
            f"Chạy: python scripts/download_data.py --symbol {symbol} --interval {interval}"
        )
    return pd.read_parquet(path)
