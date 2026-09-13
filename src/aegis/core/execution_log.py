"""
Nhật ký thực thi — đo chi phí THẬT thay vì tin vào giả định của backtest.

VÌ SAO CẦN: backtest giả định 100% khớp ở giá maker (1bp). Lần chạy testnet đầu
tiên cho thấy thực tế chỉ 34.5% khớp maker, còn lại phải cắn giá — chi phí thật
gần 3bp, gấp 3 lần giả định. Chênh lệch đó ăn ~8%/năm.

Không đo thì không biết mình đang trả bao nhiêu, và mọi con số backtest chỉ là
giả thuyết. Mỗi lượt tái cân bằng ghi một dòng JSONL để về sau đối chiếu
lợi nhuận thực tế với lợi nhuận kỳ vọng.
"""

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_LOG_PATH = "artifacts/execution_log.jsonl"

# [FIX F31] Phí THẬT của Binance USDⓈ-M VIP0: maker 0.0200%, taker 0.0500%.
# Hằng số cũ (1bp / 4bp) thấp hơn thực tế, nên chính module sinh ra để "đo chi phí
# thật" lại báo chi phí thấp hơn thật ~35%. Một thước đo bị lệch còn nguy hiểm hơn
# không đo, vì nó tạo cảm giác đã kiểm soát.
MAKER_FEE = 0.0002
TAKER_FEE = 0.0005


@dataclass
class ExecutionRecord:
    """Một lượt tái cân bằng."""
    timestamp_ms: int
    equity: float
    n_targets: int
    n_orders: int
    gross_notional: float
    planned_turnover: float
    maker_notional: float
    taker_notional: float
    unfilled_count: int
    net_exposure: float = 0.0
    testnet: bool = True
    note: Optional[str] = None

    # --- trường phục vụ CỔNG CHẤT LƯỢNG (mặc định để tương thích bản ghi cũ) ---
    plan_failures: int = 0        # lệnh trong kế hoạch KHÔNG đặt được [F22]
    uncancelled: int = 0          # cặp còn lệnh sống không xác nhận huỷ [F21]
    leverage: Optional[float] = None      # đòn bẩy gộp thật sau khi thực thi
    target_leverage: Optional[float] = None
    net_ok: Optional[bool] = None         # qua cổng trung lập [F25]
    gross_ok: Optional[bool] = None       # qua cổng đòn bẩy [F29]
    manual_intervention: bool = False     # con người phải vào sửa tay
    exec_error: Optional[str] = None      # thực thi ném lỗi [F24]

    @property
    def clean(self) -> bool:
        """
        Lượt này có SẠCH không — định nghĩa duy nhất, dùng cho cổng chất lượng.

        Sạch nghĩa là: mọi lệnh trong kế hoạch đều đặt được, không còn lệnh sống
        ngoài tầm kiểm soát, danh mục đúng thiết kế cả về hướng lẫn đòn bẩy, và
        không ai phải vào sửa tay. Thiếu bất kỳ điều nào thì lượt đó KHÔNG tính.
        """
        return (self.plan_failures == 0
                and self.uncancelled == 0
                and self.exec_error is None
                and not self.manual_intervention
                and self.net_ok is not False
                and self.gross_ok is not False)

    @property
    def filled_notional(self) -> float:
        return self.maker_notional + self.taker_notional

    @property
    def maker_ratio(self) -> float:
        total = self.filled_notional
        return (self.maker_notional / total) if total > 0 else 0.0

    @property
    def realized_cost_usd(self) -> float:
        return self.maker_notional * MAKER_FEE + self.taker_notional * TAKER_FEE

    @property
    def realized_cost_bps(self) -> float:
        total = self.filled_notional
        return (self.realized_cost_usd / total * 10000.0) if total > 0 else 0.0

    @property
    def fill_ratio(self) -> float:
        """Tỷ lệ khối lượng dự tính thực sự khớp được — dưới 1.0 nghĩa là danh mục lệch."""
        planned = self.planned_turnover
        return (self.filled_notional / planned) if planned > 0 else 0.0


class ExecutionLog:
    """Ghi/đọc nhật ký thực thi dạng JSONL (append-only)."""

    def __init__(self, path: str = DEFAULT_LOG_PATH):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExecutionRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def read_all(self) -> List[ExecutionRecord]:
        if not self.path.is_file():
            return []
        out: List[ExecutionRecord] = []
        known = set(ExecutionRecord.__dataclass_fields__)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(ExecutionRecord(**{k: v for k, v in data.items() if k in known}))
        return out

    def summary(self) -> Dict[str, Any]:
        """Thống kê chi phí thực tế — dùng để đối chiếu với giả định backtest."""
        records = self.read_all()
        if not records:
            return {"n_rebalances": 0}

        maker = sum(r.maker_notional for r in records)
        taker = sum(r.taker_notional for r in records)
        total = maker + taker
        cost = sum(r.realized_cost_usd for r in records)
        planned = sum(r.planned_turnover for r in records)

        return {
            "n_rebalances": len(records),
            "maker_ratio": (maker / total) if total > 0 else 0.0,
            "fill_ratio": (total / planned) if planned > 0 else 0.0,
            "realized_cost_bps": (cost / total * 10000.0) if total > 0 else 0.0,
            "backtest_assumed_bps": MAKER_FEE * 10000.0,
            "total_cost_usd": cost,
            "total_traded_usd": total,
            "avg_net_exposure_pct": (
                sum(abs(r.net_exposure) for r in records) / len(records) * 100.0
            ),
            "first_ms": records[0].timestamp_ms,
            "last_ms": records[-1].timestamp_ms,
        }
