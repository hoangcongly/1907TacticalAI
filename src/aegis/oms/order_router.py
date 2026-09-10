"""
Điều hướng lệnh sang Binance USDⓈ-M Futures.

An toàn là ưu tiên số một ở module này — đây là nơi duy nhất tiền thật rời tài khoản.

Bốn cơ chế bảo vệ:
  1. `newClientOrderId` TẤT ĐỊNH  -> thử lại sau timeout KHÔNG tạo lệnh trùng.
  2. Kiểm bộ lọc sàn TRƯỚC khi gửi -> lệnh sai bước giá/khối lượng bị chặn tại chỗ.
  3. Trần notional mỗi lệnh        -> lỗi logic không thể biến thành lệnh khổng lồ.
  4. Kill switch                   -> huỷ sạch lệnh treo và đóng toàn bộ vị thế.

Mặc định TESTNET. Muốn chạm tiền thật phải chỉ định tường minh.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from aegis.data.ingestion.binance_rest import BinanceAPIError, BinanceFuturesREST
from aegis.execution.portfolio_rebalancer import RebalanceOrder, SymbolFilters
from aegis.oms.state_machine import ManagedOrder, OrderBook, OrderState

logger = logging.getLogger(__name__)

# Trần tuyệt đối cho notional MỖI lệnh (USD). Chốt chặn cuối chống lỗi logic.
DEFAULT_MAX_ORDER_NOTIONAL = 1000.0


class OrderRejected(RuntimeError):
    """Lệnh bị từ chối TRƯỚC khi gửi (vi phạm kiểm tra an toàn cục bộ)."""


def make_client_order_id(symbol: str, side: str, qty: float, epoch_bucket: int) -> str:
    """
    Sinh client order id TẤT ĐỊNH.

    Cùng ý định giao dịch trong cùng chu kỳ tái cân bằng -> cùng một id. Nếu request
    bị timeout nhưng sàn ĐÃ nhận lệnh, lần thử lại sẽ bị sàn từ chối vì trùng id,
    thay vì tạo ra vị thế gấp đôi. Đây là lá chắn quan trọng nhất của module.
    """
    raw = f"{symbol}|{side}|{qty:.10f}|{epoch_bucket}"
    return "aegis" + hashlib.sha1(raw.encode()).hexdigest()[:26]


class BinanceOrderRouter:
    """Gửi lệnh tới Binance Futures và theo dõi vòng đời qua OrderBook."""

    def __init__(
        self,
        client: Optional[BinanceFuturesREST] = None,
        order_book: Optional[OrderBook] = None,
        max_order_notional: float = DEFAULT_MAX_ORDER_NOTIONAL,
        dry_run: bool = False,
        notifier: Optional[Any] = None,
    ):
        self.client = client or BinanceFuturesREST()
        self.book = order_book or OrderBook()
        self.max_order_notional = float(max_order_notional)
        self.dry_run = bool(dry_run)

        if notifier is None:
            try:
                from aegis.monitoring.alerts import TelegramNotifier
                self.notifier = TelegramNotifier()
            except Exception:
                self.notifier = None
        else:
            self.notifier = notifier

        if self.client.credentials is not None and not self.client.credentials.testnet and not dry_run:
            logger.warning("⚠️  ROUTER ĐANG TRỎ MAINNET — lệnh sẽ dùng TIỀN THẬT")

    # ------------------------------------------------------------------
    # Cấu hình tài khoản (gọi một lần lúc khởi động)
    # ------------------------------------------------------------------
    def configure_symbol(self, symbol: str, leverage: int, margin_type: str = "ISOLATED") -> None:
        """
        Đặt chế độ ký quỹ và đòn bẩy cho một cặp.

        Bắt buộc gọi TRƯỚC khi giao dịch: đòn bẩy mặc định của Binance có thể là 20x,
        và sàn dùng chính con số đó để tính ký quỹ lẫn giá thanh lý.
        """
        try:
            self.client.set_margin_type(symbol, margin_type)
        except BinanceAPIError as exc:
            logger.warning("Không đổi được margin type cho %s: %s", symbol, exc)
        self.client.set_leverage(symbol, leverage)

    # ------------------------------------------------------------------
    # Kiểm tra an toàn trước khi gửi
    # ------------------------------------------------------------------
    def _preflight(self, order: RebalanceOrder, filt: SymbolFilters) -> None:
        if order.qty <= 0:
            raise OrderRejected(f"{order.symbol}: qty phải > 0, nhận {order.qty}")
        if order.side not in ("BUY", "SELL"):
            raise OrderRejected(f"{order.symbol}: side không hợp lệ {order.side}")

        notional = order.qty * order.price
        if notional > self.max_order_notional:
            raise OrderRejected(
                f"{order.symbol}: notional ${notional:.2f} vượt trần ${self.max_order_notional:.2f}"
            )
        if not filt.is_tradeable(order.qty, order.price):
            raise OrderRejected(
                f"{order.symbol}: qty {order.qty} @ {order.price} không đạt "
                f"min_qty {filt.min_qty} / min_notional ${filt.min_notional}"
            )
        if filt.round_qty(order.qty) != order.qty:
            raise OrderRejected(
                f"{order.symbol}: qty {order.qty} không khớp step_size {filt.step_size}"
            )

    # ------------------------------------------------------------------
    # Gửi lệnh
    # ------------------------------------------------------------------
    def submit(
        self,
        order: RebalanceOrder,
        filt: SymbolFilters,
        post_only: bool = True,
        epoch_bucket: Optional[int] = None,
        retries_left: int = 3,
        backoff_ticks: int = 2,
    ) -> ManagedOrder:
        """
        Gửi một lệnh.

        `post_only=True` dùng GTX (post-only limit): lệnh bị HUỶ nếu sẽ khớp ngay,
        đảm bảo luôn là maker. Chiến lược đã kiểm định giả định phí maker, nên đây
        là mặc định — khớp taker sẽ phá vỡ giả định chi phí của backtest.
        """
        self._preflight(order, filt)

        bucket = epoch_bucket if epoch_bucket is not None else int(time.time() // 60)
        coid = make_client_order_id(order.symbol, order.side, order.qty, bucket)

        existing = self.book.get(coid)
        if existing is not None:
            logger.info("Lệnh %s đã tồn tại (%s) — bỏ qua, chống trùng", coid, existing.state.value)
            return existing

        managed = self.book.add(ManagedOrder(
            client_order_id=coid, symbol=order.symbol, side=order.side,
            qty=order.qty, price=order.price,
        ))

        # [FIX F19] Gửi CHUỖI đã định dạng theo đúng step_size/tick_size.
        # Truyền float thô khiến Python tuần tự hoá thành "1234.0" hoặc
        # "1.2426000000000002" -> Binance từ chối với -1111.
        params: Dict[str, Any] = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": filt.format_qty(order.qty),
            "newClientOrderId": coid,
        }
        if post_only:
            params.update({
                "type": "LIMIT", "timeInForce": "GTX",
                "price": filt.format_price(order.price),
            })
        else:
            params["type"] = "MARKET"

        if self.dry_run:
            logger.info("[DRY RUN] %s", params)
            managed.transition(OrderState.SUBMITTED)
            return managed

        managed.transition(OrderState.SUBMITTED)
        try:
            resp = self.client._request("POST", "/fapi/v1/order", params, signed=True)
        except BinanceAPIError as exc:
            # [FIX POST-ONLY-RETRY] Mã -5022 nghĩa là giá đặt sẽ khớp ngay nên lệnh
            # post-only bị huỷ. Sổ lệnh dịch chuyển giữa lúc đọc BBO và lúc lệnh tới
            # sàn — chuyện thường xuyên. Lùi giá ra xa thêm vài tick rồi thử lại,
            # thay vì bỏ lệnh và để danh mục lệch cân bằng long/short.
            if exc.code == -5022 and post_only and retries_left > 0:
                step = filt.tick_size * backoff_ticks
                new_price = order.price - step if order.side == "BUY" else order.price + step
                logger.info("Lệnh %s bị -5022, lùi giá %s -> %s rồi thử lại",
                            coid, order.price, filt.round_price(new_price))
                retry = RebalanceOrder(order.symbol, order.side, order.qty,
                                       filt.round_price(new_price), order.reason,
                                       order.notional)
                managed.transition(OrderState.REJECTED, error=str(exc))
                return self.submit(retry, filt, post_only=True,
                                   epoch_bucket=(bucket + retries_left + 1),
                                   retries_left=retries_left - 1,
                                   backoff_ticks=backoff_ticks * 2)
            managed.transition(OrderState.REJECTED, error=str(exc))
            logger.error("Lệnh %s bị từ chối: %s", coid, exc)
            if not self.dry_run and self.notifier and getattr(self.notifier, "is_configured", False):
                self.notifier.send_anomaly_alert(
                    title=f"Sàn từ chối lệnh {order.symbol}",
                    message=f"Lệnh {order.side} {order.qty} @ ${order.price} bị từ chối: {exc}",
                    level="ERROR",
                )
            return managed

        managed.exchange_order_id = int(resp.get("orderId", 0)) or None
        managed.apply_exchange_status(
            resp.get("status", "NEW"),
            filled_qty=float(resp.get("executedQty", 0.0) or 0.0),
            avg_price=float(resp.get("avgPrice", 0.0) or 0.0),
        )

        # [FIX F20] Lệnh MARKET thường trả về status=NEW với executedQty=0
        # vì phản hồi được gửi TRƯỚC khi khớp được ghi nhận. Nếu tin con số đó, sổ
        # nội bộ sẽ báo "chưa khớp" trong khi sàn đã có vị thế — đúng loại lệch mà
        # module reconciliation sinh ra để bắt. Truy vấn lại để lấy trạng thái thật.
        if not post_only and not managed.is_terminal:
            managed = self.poll_status(managed)

        if not self.dry_run and self.notifier and getattr(self.notifier, "is_configured", False):
            self.notifier.send_single_order_alert(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                price=order.price,
                notional=order.notional,
                reason=order.reason,
                status=managed.state.value,
                order_type="LIMIT" if post_only else "MARKET",
                client_order_id=coid,
            )

        return managed

    def poll_status(self, managed: ManagedOrder, retries: int = 3, delay: float = 0.4) -> ManagedOrder:
        """Hỏi lại sàn trạng thái thật của lệnh và cập nhật sổ nội bộ."""
        for attempt in range(retries):
            try:
                resp = self.client._request(
                    "GET", "/fapi/v1/order",
                    {"symbol": managed.symbol, "origClientOrderId": managed.client_order_id},
                    signed=True,
                )
            except BinanceAPIError as exc:
                logger.warning("Không truy vấn được lệnh %s: %s", managed.client_order_id, exc)
                return managed

            status = resp.get("status", "NEW")
            filled = float(resp.get("executedQty", 0.0) or 0.0)
            try:
                managed.apply_exchange_status(
                    status, filled_qty=filled,
                    avg_price=float(resp.get("avgPrice", 0.0) or 0.0),
                )
            except Exception as exc:
                logger.warning("Trạng thái lệch khi truy vấn %s: %s", managed.client_order_id, exc)

            if managed.is_terminal:
                return managed
            if attempt < retries - 1:
                time.sleep(delay)
        return managed

    def submit_plan(
        self,
        orders: List[RebalanceOrder],
        filters: Dict[str, SymbolFilters],
        post_only: bool = True,
    ) -> List[ManagedOrder]:
        """
        Gửi cả kế hoạch tái cân bằng.

        ĐÓNG TRƯỚC, MỞ SAU: giải phóng ký quỹ trước khi cần dùng, tránh bị từ chối
        vì thiếu margin giữa chừng và mắc kẹt ở trạng thái nửa vời.
        """
        bucket = int(time.time() // 60)
        closing = [o for o in orders if o.reason == "đóng"]
        opening = [o for o in orders if o.reason != "đóng"]

        results: List[ManagedOrder] = []
        for order in closing + opening:
            filt = filters.get(order.symbol)
            if filt is None:
                logger.error("Thiếu bộ lọc sàn cho %s — bỏ qua", order.symbol)
                continue
            try:
                results.append(self.submit(order, filt, post_only=post_only, epoch_bucket=bucket))
            except OrderRejected as exc:
                logger.error("Chặn tại chỗ: %s", exc)
                if not self.dry_run and self.notifier and getattr(self.notifier, "is_configured", False):
                    self.notifier.send_anomaly_alert(
                        title=f"Chặn lệnh {order.symbol}",
                        message=f"Vi phạm kiểm tra an toàn: {exc}",
                        level="WARNING",
                    )
        return results

    # ------------------------------------------------------------------
    # Thuật toán thực thi: THỤ ĐỘNG TRƯỚC, CHỦ ĐỘNG SAU
    # ------------------------------------------------------------------
    def execute_with_fallback(
        self,
        orders: List[RebalanceOrder],
        filters: Dict[str, SymbolFilters],
        passive_wait_s: float = 120.0,
        poll_interval_s: float = 10.0,
        allow_taker_fallback: bool = True,
        requote: bool = True,
        max_chase_bps: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Đặt lệnh maker trước, phần không khớp thì cắn giá sau.

        VÌ SAO CẦN: đây là bài học từ lần chạy testnet thật — 9/12 lệnh post-only
        nằm chờ với 0% khớp, khiến danh mục lệch 33% khỏi trung lập. Một chiến lược
        market-neutral chỉ khớp một nửa KHÔNG còn trung lập; nó biến thành cược có
        hướng mà không ai chủ ý, và toàn bộ kết quả kiểm định mất hiệu lực.

        Đánh đổi được cân nhắc:
          - Phí maker rẻ hơn taker 4 lần -> đáng chờ.
          - Nhưng lệch trung lập nguy hiểm hơn nhiều so với trả thêm phí.
        Nên: chờ `passive_wait_s` để ăn phí maker, rồi cắn giá phần còn lại để
        đảm bảo danh mục về đúng trạng thái mục tiêu.
        """
        report: Dict[str, Any] = {
            "passive_submitted": 0, "passive_filled_notional": 0.0,
            "taker_orders": 0, "taker_notional": 0.0, "unfilled": [],
        }

        intents = {o.symbol: o for o in orders}
        passive = self.submit_plan(orders, filters, post_only=True)
        report["passive_submitted"] = len(passive)
        report["requotes"] = 0

        # Chờ khớp thụ động, BÁO GIÁ LẠI theo sổ lệnh.
        # Lệnh đặt ở BBO cũ sẽ không bao giờ khớp khi sổ dịch đi — đây là lý do
        # lần chạy trước có 0% khớp maker. Định kỳ huỷ và đặt lại ở BBO hiện tại,
        # nhưng chỉ đuổi trong phạm vi `max_chase_bps` để không rượt theo một cú
        # chạy giá và mua đúng đỉnh.
        deadline = time.time() + max(0.0, passive_wait_s)
        anchor = {m.symbol: (m.price or 0.0) for m in passive}

        while time.time() < deadline:
            time.sleep(min(poll_interval_s, max(0.0, deadline - time.time())))
            for m in passive:
                if not m.is_terminal:
                    self.poll_status(m, retries=1)
            if all(m.is_terminal for m in passive):
                break

            if not requote:
                continue

            for idx, m in enumerate(passive):
                if m.is_terminal or m.remaining_qty <= 0:
                    continue
                filt = filters.get(m.symbol)
                base = anchor.get(m.symbol, 0.0)
                if filt is None or base <= 0:
                    continue
                try:
                    bid, ask = self.client.best_bid_ask(m.symbol)
                except Exception:
                    continue

                target = bid if m.side == "BUY" else ask
                if abs(target - (m.price or target)) < filt.tick_size:
                    continue  # sổ chưa dịch đủ để đáng báo giá lại

                # Chặn đuổi giá quá xa mốc ban đầu.
                drift_bps = abs(target - base) / base * 10000.0
                if drift_bps > max_chase_bps:
                    continue

                self.cancel_all(m.symbol)
                time.sleep(0.3)
                self.poll_status(m, retries=1)
                remaining = filt.round_qty(m.remaining_qty)
                if remaining <= 0 or not filt.is_tradeable(remaining, target):
                    continue

                intent = intents.get(m.symbol)
                new_order = RebalanceOrder(m.symbol, m.side, remaining,
                                           filt.round_price(target),
                                           intent.reason if intent else "điều chỉnh",
                                           remaining * target)
                replaced = self.submit(new_order, filt, post_only=True,
                                       epoch_bucket=int(time.time()))
                passive[idx] = replaced
                report["requotes"] += 1

        by_id = {m.client_order_id: m for m in passive}

        # Huỷ phần chưa khớp rồi tính lại khối lượng còn thiếu.
        for m in passive:
            if not m.is_terminal:
                self.cancel_all(m.symbol)
        time.sleep(0.8)
        for m in passive:
            self.poll_status(m, retries=1)
            report["passive_filled_notional"] += m.filled_qty * (m.price or 0.0)

        if not allow_taker_fallback:
            report["unfilled"] = [
                {"symbol": m.symbol, "remaining": m.remaining_qty}
                for m in passive if m.remaining_qty > 0
            ]
            return report

        for m in passive:
            remaining = m.remaining_qty
            if remaining <= 0:
                continue
            filt = filters.get(m.symbol)
            intent = intents.get(m.symbol)
            if filt is None or intent is None:
                continue

            qty = filt.round_qty(remaining)
            if qty <= 0 or not filt.is_tradeable(qty, m.price or intent.price):
                report["unfilled"].append({"symbol": m.symbol, "remaining": remaining,
                                           "reason": "dưới min_notional"})
                continue

            taker = RebalanceOrder(m.symbol, m.side, qty, intent.price,
                                   intent.reason, qty * intent.price)
            try:
                filled = self.submit(taker, filt, post_only=False,
                                     epoch_bucket=int(time.time() // 60) + 7919)
                report["taker_orders"] += 1
                report["taker_notional"] += filled.filled_qty * (filled.avg_fill_price or intent.price)
            except OrderRejected as exc:
                report["unfilled"].append({"symbol": m.symbol, "remaining": remaining,
                                           "reason": str(exc)})
        return report

    # ------------------------------------------------------------------
    # Huỷ / dừng khẩn cấp
    # ------------------------------------------------------------------
    def cancel_all(self, symbol: str) -> None:
        if self.dry_run:
            logger.info("[DRY RUN] huỷ toàn bộ lệnh %s", symbol)
            return
        try:
            self.client._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        except BinanceAPIError as exc:
            logger.error("Không huỷ được lệnh treo %s: %s", symbol, exc)

    def kill_switch(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        DỪNG KHẨN CẤP: huỷ sạch lệnh treo rồi đóng toàn bộ vị thế bằng MARKET.

        Dùng MARKET chứ không dùng maker — khi cần thoát, chắc chắn thoát được quan
        trọng hơn tiết kiệm phí.
        """
        report: Dict[str, Any] = {"canceled": [], "closed": [], "errors": []}
        positions = self.client.position_risk()
        live = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
        if symbols:
            live = [p for p in live if p["symbol"] in symbols]

        for pos in live:
            symbol = pos["symbol"]
            amt = float(pos["positionAmt"])
            try:
                self.cancel_all(symbol)
                report["canceled"].append(symbol)
                if self.dry_run:
                    report["closed"].append({"symbol": symbol, "qty": abs(amt), "dry_run": True})
                    continue
                self.client._request("POST", "/fapi/v1/order", {
                    "symbol": symbol,
                    "side": "SELL" if amt > 0 else "BUY",
                    "type": "MARKET",
                    "quantity": f"{abs(amt):.10f}".rstrip("0").rstrip("."),
                    "reduceOnly": "true",
                }, signed=True)
                report["closed"].append({"symbol": symbol, "qty": abs(amt)})
            except BinanceAPIError as exc:
                report["errors"].append({"symbol": symbol, "error": str(exc)})
                logger.error("KILL SWITCH lỗi trên %s: %s", symbol, exc)
        return report
