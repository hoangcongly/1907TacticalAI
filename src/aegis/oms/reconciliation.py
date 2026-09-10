"""
Đối chiếu trạng thái nội bộ với trạng thái THẬT trên sàn.

NGUYÊN TẮC CỐT LÕI: sàn là NGUỒN SỰ THẬT. Trạng thái nội bộ chỉ là bản sao, và bản
sao có thể sai — do tiến trình chết giữa lúc gửi lệnh, WebSocket mất gói, hoặc thao
tác thủ công trên app.

Khi phát hiện lệch, hệ thống KHÔNG được đoán và giao dịch tiếp. Với danh mục
market-neutral, một vị thế lệch làm hỏng tính trung lập và biến chiến lược đã kiểm
định thành một cược có hướng mà không ai chủ ý.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aegis.data.ingestion.binance_rest import BinanceFuturesREST
from aegis.oms.state_machine import OrderBook

logger = logging.getLogger(__name__)

# Lệch khối lượng nhỏ hơn ngưỡng này coi như bụi làm tròn, không phải sai lệch thật.
DEFAULT_QTY_TOLERANCE = 1e-8


@dataclass
class Discrepancy:
    symbol: str
    internal_qty: float
    exchange_qty: float
    kind: str

    @property
    def delta(self) -> float:
        return self.exchange_qty - self.internal_qty


@dataclass
class ReconciliationReport:
    exchange_positions: Dict[str, float] = field(default_factory=dict)
    internal_positions: Dict[str, float] = field(default_factory=dict)
    discrepancies: List[Discrepancy] = field(default_factory=list)
    orphan_orders: List[str] = field(default_factory=list)
    wallet_balance: float = 0.0
    available_balance: float = 0.0

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies and not self.orphan_orders

    def summary(self) -> str:
        if self.is_clean:
            return (f"✅ Khớp sổ sách | {len(self.exchange_positions)} vị thế | "
                    f"ví ${self.wallet_balance:.2f}")
        lines = [f"❌ LỆCH SỔ SÁCH ({len(self.discrepancies)} vị thế, "
                 f"{len(self.orphan_orders)} lệnh mồ côi)"]
        for d in self.discrepancies:
            lines.append(f"   {d.symbol:<12} nội bộ {d.internal_qty:+.6f} | "
                         f"sàn {d.exchange_qty:+.6f} | lệch {d.delta:+.6f} ({d.kind})")
        for oid in self.orphan_orders:
            lines.append(f"   lệnh treo trên sàn không có trong sổ nội bộ: {oid}")
        return "\n".join(lines)


class ReconciliationError(RuntimeError):
    """Trạng thái lệch — TUYỆT ĐỐI không giao dịch tiếp cho tới khi giải quyết."""


def reconcile(
    client: BinanceFuturesREST,
    order_book: Optional[OrderBook] = None,
    expected_positions: Optional[Dict[str, float]] = None,
    qty_tolerance: float = DEFAULT_QTY_TOLERANCE,
) -> ReconciliationReport:
    """
    So trạng thái nội bộ với sàn.

    `expected_positions` là vị thế ta TIN là mình đang có (từ state store bền vững).
    Không truyền thì suy từ `order_book` của phiên hiện tại.
    """
    report = ReconciliationReport()

    balance = client.balance_usdt()
    report.wallet_balance = balance["wallet_balance"]
    report.available_balance = balance["available_balance"]

    for pos in client.position_risk():
        amt = float(pos.get("positionAmt", 0.0))
        if abs(amt) > qty_tolerance:
            report.exchange_positions[pos["symbol"]] = amt

    if expected_positions is not None:
        internal = {s: q for s, q in expected_positions.items() if abs(q) > qty_tolerance}
    elif order_book is not None:
        symbols = {o.symbol for o in order_book.all_orders()}
        internal = {}
        for s in symbols:
            q = order_book.net_position(s)
            if abs(q) > qty_tolerance:
                internal[s] = q
    else:
        internal = {}
    report.internal_positions = internal

    for symbol in set(report.exchange_positions) | set(internal):
        ex = report.exchange_positions.get(symbol, 0.0)
        it = internal.get(symbol, 0.0)
        if abs(ex - it) <= qty_tolerance:
            continue
        if it == 0.0:
            kind = "sàn có vị thế mà nội bộ không biết"
        elif ex == 0.0:
            kind = "nội bộ tưởng có vị thế nhưng sàn không có"
        else:
            kind = "lệch khối lượng"
        report.discrepancies.append(Discrepancy(symbol, it, ex, kind))

    if order_book is not None:
        known = {o.exchange_order_id for o in order_book.all_orders() if o.exchange_order_id}
        for od in client.open_orders():
            if int(od.get("orderId", 0)) not in known:
                report.orphan_orders.append(f"{od.get('symbol')}#{od.get('orderId')}")

    return report


def assert_clean_or_raise(report: ReconciliationReport) -> None:
    """
    Chốt chặn khởi động: lệch sổ sách thì DỪNG, không tự đoán.

    Tự động "sửa" trạng thái lệch là cách chắc chắn nhất để nhân đôi vị thế hoặc
    giao dịch mù. Con người phải nhìn và quyết định.
    """
    if not report.is_clean:
        raise ReconciliationError(
            report.summary()
            + "\n\nHệ thống DỪNG. Hãy kiểm tra thủ công trên sàn rồi:"
            + "\n  - đóng vị thế ngoài dự kiến, hoặc"
            + "\n  - cập nhật state store cho khớp thực tế, hoặc"
            + "\n  - chạy kill switch để đưa về trạng thái phẳng."
        )
