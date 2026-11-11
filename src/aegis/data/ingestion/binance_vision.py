"""Lịch sử perp USDT từ kho dump công khai data.binance.vision, GỒM CẢ cặp đã huỷ niêm yết.

REST `/fapi` chỉ liệt kê cặp đang giao dịch, nên một universe dựng từ nó chỉ chứa cặp còn
sống. Kho dump giữ lại mọi cặp từng niêm yết, đủ để backtest không dính thiên vị sống sót.
"""

from __future__ import annotations

import io
import re
import urllib.parse
import urllib.request
import zipfile
from typing import Iterator, List, Optional

import pandas as pd

from aegis.data.ingestion.binance_history import _KLINE_COLUMNS, klines_to_frame

LISTING_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILE_URL = "https://data.binance.vision/"
KLINES = "data/futures/um/{period}/klines/{symbol}/{interval}/"
FUNDING = "data/futures/um/monthly/fundingRate/{symbol}/"
PERP_USDT = re.compile(r"^[A-Z0-9]+USDT$")
_TAG = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def fetch(url: str, timeout: float = 60.0, retries: int = 4) -> bytes:
    import time

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def parse_listing(xml: bytes) -> tuple[List[str], List[str], Optional[str]]:
    """Trả về (thư mục con, khoá file, marker trang kế) của một trang liệt kê S3."""
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    prefixes = [
        node.findtext(f"{_TAG}Prefix") for node in root.findall(f"{_TAG}CommonPrefixes")
    ]
    keys = [node.findtext(f"{_TAG}Key") for node in root.findall(f"{_TAG}Contents")]
    marker = None
    if root.findtext(f"{_TAG}IsTruncated") == "true":
        marker = root.findtext(f"{_TAG}NextMarker") or (keys[-1] if keys else None)
    return prefixes, keys, marker


def list_prefix(
    prefix: str, marker: str = "", get=fetch
) -> Iterator[tuple[List[str], List[str]]]:
    while True:
        query = urllib.parse.urlencode(
            {"delimiter": "/", "prefix": prefix, "marker": marker}
        )
        prefixes, keys, next_marker = parse_listing(get(f"{LISTING_URL}?{query}"))
        yield prefixes, keys
        if not next_marker:
            return
        marker = next_marker


def is_perp_usdt(symbol: str) -> bool:
    """Loại hợp đồng kỳ hạn (`BTCUSDT_240329`) và cặp không định giá bằng USDT."""
    return bool(PERP_USDT.match(symbol))


def list_symbols(get=fetch) -> List[str]:
    root = "data/futures/um/monthly/klines/"
    names = [
        p[len(root) :].strip("/")
        for prefixes, _ in list_prefix(root, get=get)
        for p in prefixes
    ]
    return sorted(name for name in names if is_perp_usdt(name))


def zip_keys(prefix: str, marker: str = "", get=fetch) -> List[str]:
    return sorted(
        k
        for _, keys in list_prefix(prefix, marker, get=get)
        for k in keys
        if k.endswith(".zip")
    )


def kline_keys(symbol: str, interval: str = "1h", get=fetch) -> List[str]:
    """File tháng, cộng file ngày cho đoạn SAU tháng cuối cùng đã có file tháng."""
    monthly_prefix = KLINES.format(period="monthly", symbol=symbol, interval=interval)
    monthly = zip_keys(monthly_prefix, get=get)
    daily_prefix = KLINES.format(period="daily", symbol=symbol, interval=interval)
    marker = ""
    if monthly:
        last_month = monthly[-1].rsplit("-", 2)
        marker = f"{daily_prefix}{symbol}-{interval}-{last_month[-2]}-{last_month[-1][:2]}-99"
    return monthly + zip_keys(daily_prefix, marker, get=get)


def read_csv_rows(payload: bytes) -> List[List[str]]:
    """Các dòng dữ liệu của file CSV duy nhất trong zip; bỏ dòng tiêu đề nếu có."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8")
    rows = [line.split(",") for line in text.splitlines() if line.strip()]
    if rows and not rows[0][0].strip().lstrip("-").isdigit():
        rows = rows[1:]
    return rows


def klines_from_zip(payload: bytes) -> pd.DataFrame:
    rows = [row[: len(_KLINE_COLUMNS)] for row in read_csv_rows(payload)]
    frame = klines_to_frame(rows)
    if len(frame) and frame["timestamp_ms"].iloc[0] > 10**14:  # micro giây
        frame["timestamp_ms"] //= 1000
    return frame


def funding_from_zip(payload: bytes) -> pd.DataFrame:
    """Cột của dump: calc_time, funding_interval_hours, last_funding_rate."""
    rows = read_csv_rows(payload)
    return pd.DataFrame(
        {
            "funding_time": [int(row[0]) for row in rows],
            "rate": [float(row[2]) for row in rows],
        }
    )


def download_symbol(
    symbol: str, interval: str = "1h", get=fetch
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def merged(frames, key):
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame()
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(key, keep="last")
            .sort_values(key)
            .reset_index(drop=True)
        )

    bars = merged(
        [
            klines_from_zip(get(FILE_URL + key))
            for key in kline_keys(symbol, interval, get=get)
        ],
        "timestamp_ms",
    )
    funding_keys = zip_keys(FUNDING.format(symbol=symbol), get=get)
    funding = merged(
        [funding_from_zip(get(FILE_URL + key)) for key in funding_keys], "funding_time"
    )
    return bars, funding
