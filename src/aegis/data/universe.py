"""
Module K.3 — Dựng universe giao dịch được, chống thiên lệch sống sót.

Ba bộ lọc, mỗi bộ giải quyết một cách hỏng khác nhau:

1. GIAO DỊCH ĐƯỢC trên sàn thực thi
   Cặp có trên mainnet nhưng chưa có trên testnet sẽ bị bỏ âm thầm khi đặt lệnh,
   làm lệch cân bằng long/short mà không ai nhận ra. Phải loại TRƯỚC khi tính
   trọng số, không phải lúc gửi lệnh.

2. ĐỦ LỊCH SỬ
   `momentum_90` cần ít nhất 90 nến. Cặp mới niêm yết sẽ có tín hiệu tính từ
   vài điểm dữ liệu — nhiễu thuần tuý đội lốt alpha.

3. ĐỦ THANH KHOẢN
   Cặp mỏng có spread rộng và trượt giá lớn; backtest giả định phí maker sẽ
   không bao giờ đúng ở đó.

Các bộ lọc này là RÀNG BUỘC THỰC THI, không phải lựa chọn alpha — chúng loại bỏ
những thứ ta không thể giao dịch một cách trung thực.
"""

import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = "data/binance"


@dataclass
class UniverseFilter:
    """Tiêu chí lọc universe."""
    min_history_bars: int = 120       # > lookback dài nhất (momentum_90 = 90)
    min_quote_volume_24h: float = 5e6  # USD
    require_tradeable_on: Optional[Set[str]] = None  # tập cặp của sàn thực thi

    def describe(self) -> str:
        return (f"lịch sử >= {self.min_history_bars} nến, "
                f"khối lượng 24h >= ${self.min_quote_volume_24h/1e6:.0f}M"
                + (", có trên sàn thực thi" if self.require_tradeable_on else ""))


def tradeable_symbols(client) -> Set[str]:
    """Tập perp USDT đang giao dịch được trên sàn mà client đang trỏ tới."""
    info = client.exchange_info()
    return {
        s["symbol"] for s in info.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("contractType") == "PERPETUAL"
    }


def liquid_symbols(client, min_quote_volume_24h: float) -> Set[str]:
    """Tập cặp có khối lượng 24h vượt ngưỡng."""
    try:
        tickers = client._request("GET", "/fapi/v1/ticker/24hr")
    except Exception as exc:
        logger.warning("Không lấy được ticker 24h (%s) — bỏ qua lọc thanh khoản", exc)
        return set()
    return {t["symbol"] for t in tickers
            if float(t.get("quoteVolume", 0) or 0) >= min_quote_volume_24h}


def history_lengths(
    symbols: List[str], interval: str, root: str = DATA_ROOT
) -> Dict[str, int]:
    """Số nến đã tải được cho từng cặp."""
    out: Dict[str, int] = {}
    for symbol in symbols:
        path = pathlib.Path(root) / f"{symbol}_{interval}.parquet"
        if path.is_file():
            try:
                out[symbol] = len(pd.read_parquet(path, columns=["timestamp_ms"]))
            except Exception:
                out[symbol] = 0
    return out


def build_universe(
    candidates: List[str],
    interval: str,
    filt: UniverseFilter,
    execution_client=None,
    data_client=None,
    root: str = DATA_ROOT,
) -> tuple[List[str], Dict[str, str]]:
    """
    Lọc danh sách ứng viên xuống universe thực sự giao dịch được.

    Trả về `(danh_sách_giữ_lại, {symbol: lý_do_loại})` — lý do luôn được ghi lại
    để không có cặp nào biến mất một cách âm thầm.
    """
    rejected: Dict[str, str] = {}
    keep = list(candidates)

    if execution_client is not None:
        tradeable = tradeable_symbols(execution_client)
        for s in list(keep):
            if s not in tradeable:
                rejected[s] = "không giao dịch được trên sàn thực thi"
                keep.remove(s)

    if data_client is not None and filt.min_quote_volume_24h > 0:
        liquid = liquid_symbols(data_client, filt.min_quote_volume_24h)
        if liquid:
            for s in list(keep):
                if s not in liquid:
                    rejected[s] = f"khối lượng 24h < ${filt.min_quote_volume_24h/1e6:.0f}M"
                    keep.remove(s)

    lengths = history_lengths(keep, interval, root)
    for s in list(keep):
        n = lengths.get(s, 0)
        if n < filt.min_history_bars:
            rejected[s] = f"chỉ có {n} nến < {filt.min_history_bars} yêu cầu"
            keep.remove(s)

    logger.info("Universe: giữ %d / loại %d", len(keep), len(rejected))
    return sorted(keep), rejected
