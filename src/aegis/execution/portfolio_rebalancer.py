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
#: Trần |net|/gross của KẾ HOẠCH, khớp `LiveConfig.max_net_exposure` [FIX F42].
DEFAULT_NEUTRALITY_TOLERANCE = 0.02


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
    neutrality_tolerance: float = DEFAULT_NEUTRALITY_TOLERANCE,
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

    # [FIX F42] các cặp bị DẢI KHÔNG GIAO DỊCH bỏ qua, có thể phải nhận lại.

    band_skipped: List[tuple] = []
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
            # [FIX F42] Điều chỉnh làm tròn về 0 nghĩa là vị thế ĐỨNG YÊN ở khối
            # lượng cũ — nó vẫn nằm trong danh mục và vẫn mang rủi ro. Bản cũ
            # `continue` thẳng, nên nó BIẾN MẤT khỏi gross/net của kế hoạch: đo thử
            # 12 vị thế thì 9 cái bốc hơi, gross còn 5.000 thay vì 20.000 và tỉ lệ
            # net/gross đọc ra +100% thay vì 0%. Sổ kế toán của kế hoạch phải tả
            # đúng danh mục SẼ tồn tại, không phải chỉ những cặp có phát lệnh.
            if target_qty != 0.0:
                plan.gross_notional += abs(held) * price
                plan.net_notional += held * price
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
                    # [FIX F42] Giữ lại để bước cân lại còn nhận về được nếu sổ lệch.
                    band_skipped.append((symbol, delta, price, target_qty, held, filt))
                continue

        # Lệnh MỞ phải tự vượt min_notional; lệnh ĐÓNG thì luôn được phép.
        if not is_closing and not filt.is_tradeable(delta, price):
            plan.skipped[symbol] = (
                f"lệnh ${abs(delta)*price:.2f} < min_notional ${filt.min_notional:.2f}"
            )
            # [FIX F55] Lệnh TĂNG bị bỏ thì vị thế vẫn đứng ở khối lượng cũ — nó vẫn
            # nằm trong sổ. Bản cũ `continue` thẳng nên nó biến mất khỏi gross/net của
            # kế hoạch (cùng họ lỗi thứ hai của F42), và `_repair_neutrality` quyết định
            # trên một sổ sai. Ở vốn nhỏ nhánh này chạy THƯỜNG XUYÊN: mọi điều chỉnh
            # dưới $5 đều rơi vào đây.
            if held != 0.0:
                plan.gross_notional += abs(held) * price
                plan.net_notional += held * price
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

    _repair_neutrality(plan, band_skipped, neutrality_tolerance)
    return plan


def _repair_neutrality(
    plan: RebalancePlan,
    band_skipped: List[tuple],
    tolerance: float,
) -> None:
    """
    [FIX F42] Nhận lại các lệnh bị DẢI KHÔNG GIAO DỊCH bỏ qua, cho tới khi kế hoạch
    về trong trần trung lập.

    VÌ SAO CẦN: `combine_adaptive` -> `portfolio.py` dựng trọng số trung lập CHÍNH XÁC
    (đo được: net/gross = 0,000%). Dải không giao dịch phá đúng bất biến đó: cặp bị bỏ
    qua giữ khối lượng CŨ chứ không phải khối lượng ĐÍCH, nên mỗi lần bỏ qua là một sai
    số tới 20% cỡ vị thế — và các sai số này KHÔNG tự triệt tiêu theo chiều.

    Đây là RÒ NGẪU NHIÊN: lượt nào các cặp bị bỏ qua tình cờ ngược chiều thì sổ sạch;
    lượt nào chúng cùng chiều thì sổ lệch. Đo thật:
        lượt 18/09: bỏ 2 cặp NGƯỢC chiều (SAND short, VTHO long) -> net +0,56% ✅
        lượt 16/09: bỏ các cặp CÙNG chiều                        -> net -2,96% ❌
        lượt 11/09: cùng cơ chế                                  -> net -3,16% ❌
    Vì thế tỉ lệ lượt sạch chỉ ~50%, và cổng "3 lượt sạch liên tiếp" không bao giờ
    tới: ở p=50% kỳ vọng là 42 ngày, ở p=17% (lịch sử) là 2,1 NĂM.

    CÁCH VÁ: giữ nguyên dải (nó tiết kiệm phí thật), nhưng nếu kế hoạch lệch quá trần
    thì nhận lại các lệnh ở CHÂN NẶNG — lệnh kéo net về 0 nhiều nhất trước — cho tới
    khi đạt trần. Lượt nào vốn đã sạch thì không phát sinh lệnh nào.

    Cố tình KHÔNG sửa bằng cách thu nhỏ chân nặng ngoài kế hoạch: làm thế là bịa ra
    trọng số không đến từ `portfolio.py`, tái lập đúng họ lỗi F3 (viết lại công thức
    ở tầng live).
    """
    if not band_skipped or plan.gross_notional <= 0:
        return
    if abs(plan.net_notional) / plan.gross_notional <= tolerance:
        return  # vốn đã sạch — không trả thêm một đồng phí nào

    # Ứng viên phải KÉO NET VỀ 0, tức ngược dấu với net đang lệch.
    cands = [c for c in band_skipped if c[1] * plan.net_notional < 0]
    # Lệnh kéo mạnh nhất trước: ít lệnh nhất, ít phí nhất.
    cands.sort(key=lambda c: -abs(c[1] * c[2]))

    for symbol, delta, price, target_qty, held, filt in cands:
        if abs(plan.net_notional) / plan.gross_notional <= tolerance:
            break
        # Nhận lại thì phải tự vượt min_notional như mọi lệnh MỞ khác.
        if not filt.is_tradeable(delta, price):
            continue
        # Chỉ nhận nếu thực sự làm sổ BỚT lệch — tránh nhận vào rồi vọt quá đầu kia.
        if abs(plan.net_notional + delta * price) >= abs(plan.net_notional):
            continue

        plan.orders.append(RebalanceOrder(
            symbol=symbol,
            side="BUY" if delta > 0 else "SELL",
            qty=abs(delta),
            price=filt.round_price(price),
            reason="cân trung lập",
            notional=abs(delta) * price,
        ))
        # Cặp này nay theo khối lượng ĐÍCH, không còn theo khối lượng CŨ.
        plan.gross_notional += abs(target_qty) * price - abs(held) * price
        plan.net_notional += delta * price
        plan.skipped.pop(symbol, None)


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
