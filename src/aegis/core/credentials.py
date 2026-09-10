"""
Nạp thông tin xác thực sàn từ biến môi trường / file .env.

Nguyên tắc: KHOÁ API KHÔNG BAO GIỜ nằm trong code hay file cấu hình được commit.
`.env` đã nằm trong .gitignore. Module này cũng không bao giờ log giá trị khoá.
"""

import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional


def load_dotenv(path: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """
    Nạp file .env vào os.environ. Trả về dict các cặp đã đọc được.
    Không ghi đè biến môi trường sẵn có trừ khi override=True.
    """
    env_path = pathlib.Path(path or ".env")
    if not env_path.is_file():
        return {}

    parsed: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


@dataclass(frozen=True)
class ExchangeCredentials:
    """Khoá API của sàn. `__repr__` được che để khoá không lọt vào log/traceback."""
    api_key: str
    api_secret: str
    testnet: bool

    def __repr__(self) -> str:
        return f"ExchangeCredentials(api_key='***{self.api_key[-4:]}', testnet={self.testnet})"

    __str__ = __repr__


def load_binance_credentials(
    testnet: Optional[bool] = None,
    dotenv_path: Optional[str] = None,
) -> ExchangeCredentials:
    """
    Đọc khoá Binance Futures từ môi trường.

    Testnet: BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET
    Mainnet: BINANCE_API_KEY / BINANCE_API_SECRET

    Mặc định dùng testnet trừ khi biến BINANCE_TESTNET được đặt là false —
    mặc định AN TOÀN: nhầm lẫn sẽ rơi về tiền ảo, không phải tiền thật.
    """
    load_dotenv(dotenv_path)

    if testnet is None:
        testnet = os.environ.get("BINANCE_TESTNET", "true").strip().lower() not in (
            "false", "0", "no",
        )

    prefix = "BINANCE_TESTNET" if testnet else "BINANCE"
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    secret = os.environ.get(f"{prefix}_API_SECRET", "").strip()

    if not key or not secret:
        raise RuntimeError(
            f"Thiếu khoá API: cần {prefix}_API_KEY và {prefix}_API_SECRET "
            f"trong biến môi trường hoặc file .env. "
            f"Testnet tạo khoá miễn phí tại https://testnet.binancefuture.com"
        )

    return ExchangeCredentials(api_key=key, api_secret=secret, testnet=bool(testnet))
