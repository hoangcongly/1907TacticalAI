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
    #: [FIX F40] Notional tối đa mà MỘT vị thế được phép cần. Cặp có min notional của
    #: sàn cao hơn mức này sẽ bị loại.
    #:
    #: VÌ SAO Ở TẦNG UNIVERSE CHỨ KHÔNG Ở TẦNG ĐẶT LỆNH: min notional KHÔNG đồng nhất.
    #: Trên mainnet, 122/128 cặp cần $5 nhưng ETH/LTC/LINK/ETC/BCH cần $20 và BTC cần
    #: $50. Với vốn $38 ở 2x thì mỗi vị thế chỉ $6,34 — sáu cặp đó không mua nổi.
    #:
    #: Nếu để tới lúc gửi lệnh mới phát hiện, `build_rebalance_plan` bỏ lệnh đó vào
    #: `skipped` và danh mục MẤT MỘT CHÂN. Một sổ market-neutral thiếu một chân không
    #: còn trung lập — nó thành cược có hướng, đúng cơ chế đã làm sổ lệch +35%.
    #:
    #: `max_positions_for_capital` không cứu được: nó giả định MỌI cặp cần đúng $5.
    max_min_notional: Optional[float] = None

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
    symbols: List[str], interval: str, root: str = DATA_ROOT,
    source_interval: Optional[str] = "1h",
) -> Dict[str, int]:
    """
    Số nến khung `interval` THỰC SỰ DỰNG ĐƯỢC cho từng cặp.

    [FIX F37] Bản cũ chỉ đếm dòng trong `{symbol}_{interval}.parquet`. Nhưng hệ thống
    KHÔNG đọc file đó: `data/panel_v2.load_panel_v2` ưu tiên tổng hợp khung mục tiêu
    TỪ `source_interval` (1h) và chỉ rơi về file đúng khung khi không có nguồn. Vậy là
    bộ lọc universe đo một thứ còn đường chạy dùng một thứ khác.

    Hậu quả đo được ngày 14/09/2026: 89/170 cặp bị loại OAN. Ví dụ CRVUSDT có 52.869
    nến 1h (= 13.217 nến 4h) nhưng `CRVUSDT_4h.parquet` chỉ còn 186 dòng sót lại từ
    một lần tải cũ — nên nó bị loại vì "thiếu lịch sử". Live vì thế chạy **62 cặp**
    trong khi nghiên cứu kiểm định trên **127**.

    Đây đúng là họ lỗi F3: nghiên cứu và live đo CÙNG MỘT ĐẠI LƯỢNG bằng HAI ĐƯỜNG,
    và chúng lệch nhau mà không ai báo. Cách sửa bền vững không phải là tải lại file
    4h — file sẽ lại cũ đi — mà là đo đúng thứ mà đường chạy thật sẽ dùng.

    `source_interval=None` để đếm nguyên văn file đúng khung (hành vi cũ).
    """
    from aegis.data.panel_v2 import INTERVAL_MS

    out: Dict[str, int] = {}
    root_p = pathlib.Path(root)
    ratio = 1
    if source_interval and source_interval != interval:
        if interval not in INTERVAL_MS or source_interval not in INTERVAL_MS:
            source_interval = None
        else:
            ratio = INTERVAL_MS[interval] // INTERVAL_MS[source_interval]
            if ratio < 1:
                source_interval = None      # nguồn thô hơn đích -> không tổng hợp được

    for symbol in symbols:
        n = 0
        if source_interval:
            src = root_p / f"{symbol}_{source_interval}.parquet"
            if src.is_file():
                try:
                    n = len(pd.read_parquet(src, columns=["timestamp_ms"])) // ratio
                except Exception:
                    n = 0
        if n == 0:
            direct = root_p / f"{symbol}_{interval}.parquet"
            if direct.is_file():
                try:
                    n = len(pd.read_parquet(direct, columns=["timestamp_ms"]))
                except Exception:
                    n = 0
        if n or (root_p / f"{symbol}_{interval}.parquet").is_file() \
                or (source_interval and (root_p / f"{symbol}_{source_interval}.parquet").is_file()):
            out[symbol] = n
    return out


def symbol_min_notionals(client, symbols: Optional[Set[str]] = None) -> Dict[str, float]:
    """Notional tối thiểu của từng cặp, đọc MỘT LẦN từ `exchangeInfo`."""
    out: Dict[str, float] = {}
    for s in client.exchange_info().get("symbols", []):
        sym = s["symbol"]
        if symbols is not None and sym not in symbols:
            continue
        for f in s.get("filters", []):
            if f.get("filterType") == "MIN_NOTIONAL":
                out[sym] = float(f.get("notional", 0.0) or 0.0)
                break
    return out


def build_universe(
    candidates: List[str],
    interval: str,
    filt: UniverseFilter,
    execution_client=None,
    data_client=None,
    root: str = DATA_ROOT,
    source_interval: Optional[str] = "1h",
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

    # [FIX F40] Loại cặp mà vốn KHÔNG mua nổi — trước khi xếp hạng, không phải khi đặt lệnh.
    if execution_client is not None and filt.max_min_notional:
        try:
            mins = symbol_min_notionals(execution_client, set(keep))
        except Exception as exc:
            logger.warning("Không đọc được min notional, bỏ qua bộ lọc vốn: %s", exc)
            mins = {}
        for s in list(keep):
            need = mins.get(s)
            if need is not None and need > filt.max_min_notional:
                rejected[s] = (f"min notional ${need:.0f} > ${filt.max_min_notional:.2f} "
                               f"vốn cho phép mỗi vị thế")
                keep.remove(s)

    lengths = history_lengths(keep, interval, root, source_interval=source_interval)
    for s in list(keep):
        n = lengths.get(s, 0)
        if n < filt.min_history_bars:
            rejected[s] = f"chỉ có {n} nến < {filt.min_history_bars} yêu cầu"
            keep.remove(s)

    logger.info("Universe: giữ %d / loại %d", len(keep), len(rejected))
    return sorted(keep), rejected
