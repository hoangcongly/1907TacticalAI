"""
Trạng thái bền vững qua các lần khởi động lại.

VÌ SAO CẦN: `CircuitBreaker` giữ `peak_equity` và `is_dead` trong RAM. Tiến trình
chết rồi bật lại giữa lúc đang lỗ sẽ khiến `peak_equity` reset về equity HIỆN TẠI —
xoá sạch trí nhớ drawdown và vô hiệu hoá toàn bộ cơ chế ngắt mạch đúng lúc nó cần
hoạt động nhất. Cùng lý do với vị thế đang mở: quên vị thế nghĩa là đối chiếu sổ
sách sẽ báo lệch, hoặc tệ hơn, mở chồng thêm vị thế mới.

Ghi NGUYÊN TỬ (file tạm + rename) để tiến trình bị giết giữa lúc ghi không để lại
file JSON hỏng — trường hợp đó ta sẽ mất trạng thái đúng vào lúc cần nó nhất.
"""

import json
import logging
import os
import pathlib
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "artifacts/live_state.json"
SCHEMA_VERSION = 1


@dataclass
class LiveState:
    """Toàn bộ trạng thái phải sống sót qua restart."""
    schema_version: int = SCHEMA_VERSION

    # --- Ngắt mạch rủi ro ---
    peak_equity: float = 0.0
    is_dead: bool = False           # kill switch vĩnh viễn
    frozen_until_ms: int = 0

    # --- Vị thế & lệnh ---
    positions: Dict[str, float] = field(default_factory=dict)   # symbol -> qty (âm = short)
    last_rebalance_ms: int = 0
    rebalance_count: int = 0

    # --- Nhật ký ---
    last_equity: float = 0.0
    last_error: Optional[str] = None
    updated_ms: int = 0

    def drawdown(self, current_equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - current_equity) / self.peak_equity)


class StateStore:
    """Đọc/ghi `LiveState` ra đĩa một cách nguyên tử."""

    def __init__(self, path: str = DEFAULT_STATE_PATH):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> LiveState:
        """Nạp trạng thái; file thiếu hoặc hỏng thì trả về trạng thái mới tinh."""
        if not self.path.is_file():
            logger.info("Chưa có state store tại %s — khởi tạo mới", self.path)
            return LiveState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Không tự đoán: đổi tên file hỏng để con người xem, rồi bắt đầu sạch.
            broken = self.path.with_suffix(f".broken.{int(time.time())}")
            self.path.rename(broken)
            logger.error("State store hỏng (%s) — đã chuyển sang %s", exc, broken)
            return LiveState()

        if data.get("schema_version") != SCHEMA_VERSION:
            logger.warning("Schema state store lệch: %s != %s", data.get("schema_version"), SCHEMA_VERSION)

        known = {f for f in LiveState.__dataclass_fields__}
        return LiveState(**{k: v for k, v in data.items() if k in known})

    def save(self, state: LiveState) -> None:
        """Ghi nguyên tử: file tạm cùng thư mục rồi rename (rename là atomic trên POSIX)."""
        state.updated_ms = int(time.time() * 1000)
        payload = json.dumps(asdict(state), indent=2, ensure_ascii=False)

        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise

    def update(self, **changes: Any) -> LiveState:
        """Đọc - sửa - ghi trong một nhịp."""
        state = self.load()
        for key, value in changes.items():
            if not hasattr(state, key):
                raise AttributeError(f"LiveState không có trường `{key}`")
            setattr(state, key, value)
        self.save(state)
        return state
