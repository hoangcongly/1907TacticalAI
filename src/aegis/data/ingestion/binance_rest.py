"""
[FIX F1/F2] REST client Binance USDⓈ-M Futures.

Thay thế stub `raise NotImplementedError`. Đây là lớp DUY NHẤT nói chuyện REST với
sàn — mọi module khác đi qua đây, không tự gọi HTTP.

Bao gồm:
- Endpoint công khai: ping, time, exchangeInfo, klines, fundingRate, premiumIndex
- Endpoint có ký HMAC-SHA256: account, positionRisk, order, leverage, marginType
- Backoff khi bị rate-limit (418/429) và lỗi 5xx
- Đồng bộ lệch đồng hồ với server (chống lỗi -1021 timestamp)

QUAN TRỌNG: mặc định trỏ TESTNET. Muốn dùng tiền thật phải chỉ định tường minh.
"""

import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

from aegis.core.credentials import ExchangeCredentials, load_binance_credentials

logger = logging.getLogger(__name__)

TESTNET_BASE = "https://testnet.binancefuture.com"
MAINNET_BASE = "https://fapi.binance.com"

# Binance trả 429 (rate limit) và 418 (bị cấm tạm thời) -> phải lùi lại, không spam.
_RETRY_STATUS = {418, 429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_KLINES_MAX_LIMIT = 1500  # trần của Binance cho mỗi request klines


class BinanceAPIError(RuntimeError):
    """Sàn trả về lỗi nghiệp vụ (có mã lỗi riêng của Binance)."""

    def __init__(self, status: int, code: Optional[int], message: str):
        self.status = status
        self.code = code
        super().__init__(f"HTTP {status} / Binance code {code}: {message}")


class BinanceFuturesREST:
    """Client REST cho Binance USDⓈ-M Futures."""

    def __init__(
        self,
        credentials: Optional[ExchangeCredentials] = None,
        testnet: Optional[bool] = None,
        timeout: float = 15.0,
        recv_window: int = 5000,
        public_only: bool = False,
    ):
        self.public_only = bool(public_only)
        if self.public_only:
            # Chế độ chỉ-đọc: không cần khoá, luôn trỏ MAINNET để lấy dữ liệu THẬT.
            self.credentials = None
            self.base_url = MAINNET_BASE
        else:
            self.credentials = credentials or load_binance_credentials(testnet=testnet)
            self.base_url = TESTNET_BASE if self.credentials.testnet else MAINNET_BASE

        self.timeout = float(timeout)
        self.recv_window = int(recv_window)

        self._session = requests.Session()
        if self.credentials is not None:
            self._session.headers.update({"X-MBX-APIKEY": self.credentials.api_key})
        self._time_offset_ms = 0

    @classmethod
    def public_mainnet(cls, timeout: float = 15.0) -> "BinanceFuturesREST":
        """
        Client chỉ-đọc trỏ MAINNET, không cần khoá API.

        VÌ SAO CẦN TÁCH RIÊNG: sổ lệnh testnet là thanh khoản giả — mỏng, lệch, OFI
        gần như luôn bão hoà ±1. Nghiên cứu trên dữ liệu đó cho ra backtest vô nghĩa.
        Kiến trúc đúng:
            - Dữ liệu nghiên cứu  -> MAINNET (công khai, giá/khối lượng/funding thật)
            - Đặt lệnh thử nghiệm -> TESTNET (an toàn, tiền ảo)
        """
        return cls(public_only=True, timeout=timeout)

    # ------------------------------------------------------------------
    # Hạ tầng
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}

        if signed:
            if self.credentials is None:
                raise BinanceAPIError(
                    0, None,
                    f"Không thể gọi endpoint có ký `{path}` ở chế độ public_only "
                    f"(client này chỉ dùng để đọc dữ liệu thị trường mainnet).",
                )
            params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            params["recvWindow"] = self.recv_window
            query = urllib.parse.urlencode(params)
            params["signature"] = hmac.new(
                self.credentials.api_secret.encode(),
                query.encode(),
                hashlib.sha256,
            ).hexdigest()

        url = f"{self.base_url}{path}"
        backoff = 1.0

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._session.request(
                    method, url, params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning("Lỗi mạng %s (lần %d), thử lại sau %.1fs", exc, attempt, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue

            if response.status_code == 200:
                return response.json()

            # Lệch đồng hồ: đồng bộ lại rồi thử ngay lần nữa.
            if response.status_code == 400 and '"code":-1021' in response.text:
                self.sync_time()
                if signed:
                    params.pop("signature", None)
                    params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
                    query = urllib.parse.urlencode(
                        {k: v for k, v in params.items() if k != "signature"}
                    )
                    params["signature"] = hmac.new(
                        self.credentials.api_secret.encode(), query.encode(), hashlib.sha256
                    ).hexdigest()
                continue

            if response.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                wait = float(response.headers.get("Retry-After", backoff))
                logger.warning(
                    "HTTP %d từ %s (lần %d), lùi %.1fs", response.status_code, path, attempt, wait
                )
                time.sleep(wait)
                backoff *= 2
                continue

            try:
                payload = response.json()
                code, message = payload.get("code"), payload.get("msg", response.text)
            except ValueError:
                code, message = None, response.text
            raise BinanceAPIError(response.status_code, code, str(message))

        raise BinanceAPIError(0, None, f"Hết {_MAX_RETRIES} lần thử cho {path}")

    def sync_time(self) -> int:
        """Đồng bộ lệch đồng hồ cục bộ với server. Trả về độ lệch (ms)."""
        server_ms = int(self._request("GET", "/fapi/v1/time")["serverTime"])
        self._time_offset_ms = server_ms - int(time.time() * 1000)
        logger.info("Lệch đồng hồ với Binance: %d ms", self._time_offset_ms)
        return self._time_offset_ms

    # ------------------------------------------------------------------
    # Endpoint công khai
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        self._request("GET", "/fapi/v1/ping")
        return True

    def exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Metadata sàn: bộ lọc giá/khối lượng, đòn bẩy, trạng thái giao dịch."""
        return self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol})

    def symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """
        Bộ lọc giao dịch của một cặp, đã bóc gọn thành dict phẳng.

        Đây là các RÀNG BUỘC CỨNG của sàn — mọi lệnh sai bước giá/khối lượng hoặc
        dưới notional tối thiểu sẽ bị từ chối thẳng. Bắt buộc dùng khi đặt lệnh.
        """
        info = self.exchange_info(symbol)
        entries = [s for s in info.get("symbols", []) if s["symbol"] == symbol.upper()]
        if not entries:
            raise BinanceAPIError(0, None, f"Không tìm thấy cặp {symbol} trên sàn")
        entry = entries[0]
        filters = {f["filterType"]: f for f in entry.get("filters", [])}

        return {
            "symbol": entry["symbol"],
            "status": entry.get("status"),
            "base_asset": entry.get("baseAsset"),
            "quote_asset": entry.get("quoteAsset"),
            "price_precision": int(entry.get("pricePrecision", 8)),
            "quantity_precision": int(entry.get("quantityPrecision", 8)),
            "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.0) or 0.0),
            "step_size": float(filters.get("LOT_SIZE", {}).get("stepSize", 0.0) or 0.0),
            "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", 0.0) or 0.0),
            "max_qty": float(filters.get("LOT_SIZE", {}).get("maxQty", 0.0) or 0.0),
            "market_step_size": float(
                filters.get("MARKET_LOT_SIZE", {}).get("stepSize", 0.0) or 0.0
            ),
            "market_max_qty": float(
                filters.get("MARKET_LOT_SIZE", {}).get("maxQty", 0.0) or 0.0
            ),
            "min_notional": float(
                filters.get("MIN_NOTIONAL", {}).get("notional", 0.0) or 0.0
            ),
        }

    def klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = _KLINES_MAX_LIMIT,
    ) -> List[list]:
        """Nến thô (tối đa 1500 mỗi lần gọi)."""
        return self._request(
            "GET",
            "/fapi/v1/klines",
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(int(limit), _KLINES_MAX_LIMIT),
            },
        )

    def funding_rate_history(
        self,
        symbol: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Lịch sử funding rate THẬT theo thời gian.

        Đây là thứ `governance.funding_accrual` cần (dict {timestamp_ms: rate}) để
        thay cho hằng số mặc định 0.01%.
        """
        return self._request(
            "GET",
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol.upper(),
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(int(limit), 1000),
            },
        )

    def book_ticker(self, symbol: Optional[str] = None) -> Any:
        """
        Giá mua/bán tốt nhất hiện tại (BBO).

        Bắt buộc cho lệnh post-only: đặt BUY ở giá MARK khi mark > ask sẽ khiến
        lệnh khớp ngay và bị sàn từ chối với -5022. Phải neo vào BBO thật.
        """
        return self._request("GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})

    def best_bid_ask(self, symbol: str) -> tuple:
        t = self.book_ticker(symbol)
        return float(t["bidPrice"]), float(t["askPrice"])

    def premium_index(self, symbol: str) -> Dict[str, Any]:
        """Mark price + funding rate hiện hành. Thanh lý tính theo MARK price."""
        return self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol.upper()})

    # ------------------------------------------------------------------
    # Endpoint có ký (cần khoá API)
    # ------------------------------------------------------------------
    def account(self) -> Dict[str, Any]:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def balance_usdt(self) -> Dict[str, float]:
        """Số dư USDT — nguồn sự thật cho AccountStateTracker khi khởi động."""
        for asset in self.account().get("assets", []):
            if asset.get("asset") == "USDT":
                return {
                    "wallet_balance": float(asset["walletBalance"]),
                    "available_balance": float(asset["availableBalance"]),
                    "unrealized_pnl": float(asset.get("unrealizedProfit", 0.0)),
                }
        return {"wallet_balance": 0.0, "available_balance": 0.0, "unrealized_pnl": 0.0}

    def position_risk(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Vị thế thật trên sàn — dùng để đối chiếu khi khởi động lại."""
        return self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)

    def open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._request("GET", "/fapi/v1/openOrders", {"symbol": symbol}, signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        return self._request(
            "POST", "/fapi/v1/leverage",
            {"symbol": symbol.upper(), "leverage": int(leverage)}, signed=True,
        )

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """
        Đặt chế độ ký quỹ. Hệ thống thiết kế cho ISOLATED (ký quỹ cô lập):
        lỗ tối đa của một vị thế bị chặn ở đúng phần margin đã cọc.
        """
        if margin_type not in ("ISOLATED", "CROSSED"):
            raise ValueError(f"margin_type phải là ISOLATED hoặc CROSSED, nhận {margin_type}")
        try:
            return self._request(
                "POST", "/fapi/v1/marginType",
                {"symbol": symbol.upper(), "marginType": margin_type}, signed=True,
            )
        except BinanceAPIError as exc:
            # -4046: "No need to change margin type" -> đã đúng chế độ rồi.
            if exc.code == -4046:
                return {"code": 200, "msg": "Đã ở đúng chế độ ký quỹ"}
            raise
