"""
Cầu nối trọng số mục tiêu -> lệnh thật trên sàn (danh mục cross-sectional).

`live_pipeline.py` cũ điều khiển MỘT vị thế theo hướng dự đoán. Chiến lược đã kiểm
định là danh mục ~12 vị thế long/short tái cân bằng hằng ngày, nên cần lớp riêng.

Ba ràng buộc CỨNG của sàn phải tôn trọng, nếu không lệnh bị từ chối thẳng:
  1. `min_notional`  — giá trị lệnh tối thiểu (phần lớn perp Binance là $5)
  2. `step_size`     — bước nhảy khối lượng
  3. `tick_size`     — bước nhảy giá

Với vốn nhỏ, (1) là ràng buộc quyết định: nó giới hạn số vị thế có thể mở.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Không giao dịch nếu lệch trọng số nhỏ hơn ngưỡng này — chống churn phí vô ích.
DEFAULT_NO_TRADE_BAND = 0.20


@dataclass
class SymbolFilters:
    """Ràng buộc giao dịch của một cặp (lấy từ /fapi/v1/exchangeInfo)."""
    symbol: str
    tick_size: float
    step_size: float
    min_qty: float
    min_notional: float

    @staticmethod
    def _decimals(step: float) -> int:
        """
        Số chữ số thập phân mà một bước nhảy cho phép.

        [FIX F19] Bắt buộc vì phép nhân dấu phẩy động để lại dư số:
        `round(1.24263/0.0001)*0.0001` cho 1.2426000000000002, và Binance từ chối
        thẳng với lỗi -1111 "Precision is over the maximum defined for this asset".
        """
        if step <= 0:
            return 8
        text = f"{step:.10f}".rstrip("0")
        return len(text.split(".")[1]) if "." in text else 0

    def round_qty(self, qty: float) -> float:
        """Làm tròn XUỐNG theo step_size — không bao giờ vượt quy mô dự tính."""
        if self.step_size <= 0:
            return float(qty)
        sign = -1.0 if qty < 0 else 1.0
        steps = math.floor((abs(qty) + 1e-12) / self.step_size)
        return float(round(sign * steps * self.step_size, self._decimals(self.step_size)))

    def round_price(self, price: float) -> float:
        """Làm tròn về đúng bội số tick_size VÀ đúng số chữ số thập phân cho phép."""
        if self.tick_size <= 0:
            return float(price)
        snapped = round(price / self.tick_size) * self.tick_size
        return float(round(snapped, self._decimals(self.tick_size)))

    def format_qty(self, qty: float) -> str:
        """
        Chuỗi khối lượng gửi lên sàn, ĐÚNG số chữ số thập phân cho phép.

        [FIX F19] Chỉ làm tròn số thôi là chưa đủ: với step_size=1.0, giá trị
        float 1234.0 được tuần tự hoá thành "1234.0" và Binance từ chối vì
        quantityPrecision=0. Phải kiểm soát cả khâu định dạng chuỗi.
        """
        return f"{qty:.{self._decimals(self.step_size)}f}"

    def format_price(self, price: float) -> str:
        """Chuỗi giá gửi lên sàn, đúng số chữ số thập phân của tick_size."""
        return f"{price:.{self._decimals(self.tick_size)}f}"

    def is_tradeable(self, qty: float, price: float) -> bool:
        """Lệnh có vượt được cả min_qty lẫn min_notional không."""
        q = abs(qty)
        return q >= self.min_qty and q > 0 and (q * price) >= self.min_notional


@dataclass
class TargetPosition:
    symbol: str
    weight: float          # trọng số mục tiêu (âm = short), tổng |w| = gross
    price: float
    target_notional: float = 0.0
    target_qty: float = 0.0


@dataclass
class RebalanceOrder:
    symbol: str
    side: str              # "BUY" hoặc "SELL"
    qty: float
    price: float
    reason: str
    notional: float = 0.0


@dataclass
class RebalancePlan:
    orders: List[RebalanceOrder] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    gross_notional: float = 0.0
    net_notional: float = 0.0

    @property
    def total_turnover(self) -> float:
        return float(sum(abs(o.notional) for o in self.orders))


def build_rebalance_plan(
    target_weights: Dict[str, float],
    current_qty: Dict[str, float],
    prices: Dict[str, float],
    filters: Dict[str, SymbolFilters],
    equity: float,
    leverage: float = 1.0,
    no_trade_band: float = DEFAULT_NO_TRADE_BAND,
) -> RebalancePlan:
    """
    Dựng danh sách lệnh đưa danh mục hiện tại về trọng số mục tiêu.

    Tham số:
    - `target_weights`: {symbol: w}, âm = short. Chuẩn hoá sao cho sum|w| = 1.
    - `current_qty`: khối lượng đang nắm (âm = đang short).
    - `equity`, `leverage`: gross notional = equity * leverage.
    - `no_trade_band`: bỏ qua lệnh có notional < band * |notional mục tiêu|,
      tránh trả phí cho những điều chỉnh vụn vặt.

    Vị thế mục tiêu KHÔNG đạt min_notional sẽ bị bỏ (ghi vào `skipped`) — và nếu
    đang có vị thế ở đó thì lệnh đóng vẫn được tạo, vì đóng luôn được phép.
    """
    if equity <= 0:
        raise ValueError(f"equity phải > 0, nhận {equity}")
    if leverage <= 0:
        raise ValueError(f"leverage phải > 0, nhận {leverage}")

    gross_budget = float(equity) * float(leverage)

    total_abs = sum(abs(w) for w in target_weights.values())
    if total_abs <= 0:
        norm = {s: 0.0 for s in target_weights}
    else:
        norm = {s: w / total_abs for s, w in target_weights.items()}

    plan = RebalancePlan()
    symbols = set(norm) | set(current_qty)

    for symbol in sorted(symbols):
        price = prices.get(symbol)
        filt = filters.get(symbol)
        if price is None or price <= 0 or filt is None:
            plan.skipped[symbol] = "thiếu giá hoặc bộ lọc sàn"
            continue

        held = float(current_qty.get(symbol, 0.0))
        target_notional = norm.get(symbol, 0.0) * gross_budget
        target_qty = filt.round_qty(target_notional / price)

        # Vị thế mục tiêu quá nhỏ so với min_notional -> coi như bằng 0 (đóng nếu đang giữ).
        if target_qty != 0.0 and not filt.is_tradeable(target_qty, price):
            plan.skipped[symbol] = (
                f"mục tiêu ${abs(target_qty)*price:.2f} < min_notional ${filt.min_notional:.2f}"
            )
            target_qty = 0.0

        delta = target_qty - held
        if delta == 0.0:
            if target_qty != 0.0:
                plan.gross_notional += abs(target_qty) * price
                plan.net_notional += target_qty * price
            continue

        delta = filt.round_qty(delta)
        if delta == 0.0:
            continue

        is_closing = (held != 0.0 and target_qty == 0.0) or (abs(target_qty) < abs(held) and held * target_qty >= 0)

        # Dải không giao dịch: bỏ qua điều chỉnh vụn (nhưng LUÔN cho phép đóng hẳn).
        if not (held != 0.0 and target_qty == 0.0):
            ref = abs(target_qty) if target_qty != 0.0 else abs(held)
            if ref > 0 and abs(delta) < no_trade_band * ref:
                plan.skipped[symbol] = f"lệch {abs(delta)/ref:.0%} < dải {no_trade_band:.0%}"
                if target_qty != 0.0:
                    plan.gross_notional += abs(held) * price
                    plan.net_notional += held * price
                continue

        # Lệnh MỞ phải tự vượt min_notional; lệnh ĐÓNG thì luôn được phép.
        if not is_closing and not filt.is_tradeable(delta, price):
            plan.skipped[symbol] = (
                f"lệnh ${abs(delta)*price:.2f} < min_notional ${filt.min_notional:.2f}"
            )
            continue

        plan.orders.append(RebalanceOrder(
            symbol=symbol,
            side="BUY" if delta > 0 else "SELL",
            qty=abs(delta),
            price=filt.round_price(price),
            reason="đóng" if target_qty == 0.0 else ("mở" if held == 0.0 else "điều chỉnh"),
            notional=abs(delta) * price,
        ))
        if target_qty != 0.0:
            plan.gross_notional += abs(target_qty) * price
            plan.net_notional += target_qty * price

    return plan


def max_positions_for_capital(
    equity: float,
    leverage: float,
    min_notional: float = 5.0,
    safety: float = 1.2,
) -> int:
    """
    Số vị thế TỐI ĐA mà vốn cho phép, để mỗi lệnh vẫn vượt min_notional.

    `safety` là đệm: đặt sát min_notional sẽ khiến lệnh bị từ chối ngay khi giá
    nhích nhẹ hoặc bị làm tròn xuống theo step_size.
    """
    if equity <= 0 or leverage <= 0 or min_notional <= 0:
        raise ValueError("equity, leverage, min_notional đều phải > 0")
    return max(0, int((equity * leverage) // (min_notional * safety)))
