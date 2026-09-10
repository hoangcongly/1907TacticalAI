"""
OMS State Machine — vòng đời lệnh và các chuyển trạng thái HỢP LỆ.

VÌ SAO CẦN MÁY TRẠNG THÁI TƯỜNG MINH: lệnh trên sàn là bất đồng bộ. Xác nhận có
thể tới muộn, tới trùng, hoặc không bao giờ tới. Nếu không chốt cứng chuyển trạng
thái nào là hợp lệ, hệ thống sẽ âm thầm "khớp" một lệnh đã huỷ, hoặc mở lại một
lệnh đã đóng — và ta giao dịch dựa trên trạng thái sai.

Nguyên tắc: mọi chuyển trạng thái KHÔNG nằm trong bảng đều bị TỪ CHỐI kèm lỗi rõ
ràng. Thà dừng còn hơn hành động trên trạng thái sai.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class OrderState(Enum):
    """Trạng thái lệnh. Trạng thái CUỐI không bao giờ rời đi được."""
    CREATED = "CREATED"              # đã dựng cục bộ, chưa gửi
    SUBMITTED = "SUBMITTED"          # đã gửi, chờ sàn xác nhận
    ACKNOWLEDGED = "ACKNOWLEDGED"    # sàn đã nhận, nằm trên sổ
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"                # CUỐI
    CANCELED = "CANCELED"            # CUỐI
    REJECTED = "REJECTED"            # CUỐI
    EXPIRED = "EXPIRED"              # CUỐI


TERMINAL_STATES: Set[OrderState] = {
    OrderState.FILLED, OrderState.CANCELED,
    OrderState.REJECTED, OrderState.EXPIRED,
}

# Bảng chuyển trạng thái hợp lệ — nguồn sự thật duy nhất.
VALID_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.CREATED: {OrderState.SUBMITTED, OrderState.REJECTED, OrderState.CANCELED},
    OrderState.SUBMITTED: {
        OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED,
        OrderState.REJECTED, OrderState.CANCELED, OrderState.EXPIRED,
    },
    OrderState.ACKNOWLEDGED: {
        OrderState.PARTIALLY_FILLED, OrderState.FILLED,
        OrderState.CANCELED, OrderState.EXPIRED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED, OrderState.FILLED,
        OrderState.CANCELED, OrderState.EXPIRED,
    },
    OrderState.FILLED: set(),
    OrderState.CANCELED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
}

# Ánh xạ trạng thái Binance -> trạng thái nội bộ.
BINANCE_STATUS_MAP: Dict[str, OrderState] = {
    "NEW": OrderState.ACKNOWLEDGED,
    "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
    "FILLED": OrderState.FILLED,
    "CANCELED": OrderState.CANCELED,
    "REJECTED": OrderState.REJECTED,
    "EXPIRED": OrderState.EXPIRED,
    "EXPIRED_IN_MATCH": OrderState.EXPIRED,
}


class InvalidTransitionError(RuntimeError):
    """Chuyển trạng thái không hợp lệ — dấu hiệu logic sai hoặc sự kiện tới lệch thứ tự."""


@dataclass
class ManagedOrder:
    """Một lệnh cùng toàn bộ lịch sử vòng đời của nó."""
    client_order_id: str
    symbol: str
    side: str                     # BUY / SELL
    qty: float
    price: Optional[float] = None  # None = lệnh MARKET
    state: OrderState = OrderState.CREATED
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    exchange_order_id: Optional[int] = None
    history: List[OrderState] = field(default_factory=list)
    last_error: Optional[str] = None

    def __post_init__(self):
        if self.side not in ("BUY", "SELL"):
            raise ValueError(f"side phải là BUY hoặc SELL, nhận {self.side}")
        if self.qty <= 0:
            raise ValueError(f"qty phải > 0, nhận {self.qty}")
        if not self.history:
            self.history.append(self.state)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    def transition(
        self,
        new_state: OrderState,
        filled_qty: Optional[float] = None,
        avg_price: Optional[float] = None,
        error: Optional[str] = None,
    ) -> "ManagedOrder":
        """
        Chuyển sang trạng thái mới, TỪ CHỐI mọi chuyển đổi không hợp lệ.

        Trạng thái CUỐI không bao giờ rời đi — kể cả khi sàn gửi lại sự kiện trùng
        (điều rất hay xảy ra khi kết nối lại WebSocket).
        """
        if self.state in TERMINAL_STATES:
            if new_state == self.state:
                return self  # sự kiện lặp trên trạng thái cuối: bỏ qua, không lỗi
            raise InvalidTransitionError(
                f"Lệnh {self.client_order_id} đã ở trạng thái CUỐI {self.state.value}, "
                f"không thể chuyển sang {new_state.value}"
            )

        if new_state not in VALID_TRANSITIONS[self.state]:
            raise InvalidTransitionError(
                f"Chuyển trạng thái không hợp lệ cho {self.client_order_id}: "
                f"{self.state.value} -> {new_state.value}"
            )

        if filled_qty is not None:
            if filled_qty < self.filled_qty - 1e-12:
                raise InvalidTransitionError(
                    f"Khối lượng khớp không được GIẢM: {self.filled_qty} -> {filled_qty}"
                )
            if filled_qty > self.qty + 1e-9:
                raise InvalidTransitionError(
                    f"Khớp {filled_qty} vượt quá khối lượng đặt {self.qty}"
                )
            self.filled_qty = float(filled_qty)

        if avg_price is not None:
            self.avg_fill_price = float(avg_price)
        if error is not None:
            self.last_error = error

        self.state = new_state
        self.history.append(new_state)
        return self

    def apply_exchange_status(self, status: str, **kwargs) -> "ManagedOrder":
        """Áp trạng thái thô của Binance qua bảng ánh xạ."""
        mapped = BINANCE_STATUS_MAP.get(status.upper())
        if mapped is None:
            raise InvalidTransitionError(f"Trạng thái Binance không nhận diện được: {status}")
        return self.transition(mapped, **kwargs)


class OrderBook:
    """Sổ theo dõi toàn bộ lệnh của phiên, tra cứu theo client_order_id."""

    def __init__(self):
        self._orders: Dict[str, ManagedOrder] = {}

    def add(self, order: ManagedOrder) -> ManagedOrder:
        if order.client_order_id in self._orders:
            raise ValueError(f"client_order_id trùng: {order.client_order_id}")
        self._orders[order.client_order_id] = order
        return order

    def get(self, client_order_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(client_order_id)

    def open_orders(self) -> List[ManagedOrder]:
        return [o for o in self._orders.values() if not o.is_terminal]

    def all_orders(self) -> List[ManagedOrder]:
        return list(self._orders.values())

    def net_position(self, symbol: str) -> float:
        """Vị thế ròng theo phần ĐÃ KHỚP (không tính phần còn treo)."""
        total = 0.0
        for o in self._orders.values():
            if o.symbol != symbol or o.filled_qty <= 0:
                continue
            total += o.filled_qty if o.side == "BUY" else -o.filled_qty
        return float(total)
