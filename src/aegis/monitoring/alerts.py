"""
Hệ thống cảnh báo và gửi thông báo qua Telegram / Discord / Webhook.

Đảm bảo an toàn:
- Mọi lỗi mạng khi gửi thông báo đều được bắt và log warning, KHÔNG BAO GIỜ làm crash luồng giao dịch.
- Token và chat ID được nạp từ biến môi trường hoặc .env:
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...
"""
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional
import requests

from aegis.core.credentials import load_dotenv

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Bộ gửi thông báo qua Telegram Bot API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_s: float = 6.0,
    ):
        load_dotenv()
        if bot_token is not None:
            self.bot_token = bot_token.strip()
        else:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

        if chat_id is not None:
            self.chat_id = chat_id.strip()
        else:
            self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        self.timeout_s = timeout_s

    @property
    def is_configured(self) -> bool:
        """Kiểm tra xem đã cấu hình token và chat id chưa."""
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Gửi tin nhắn văn bản đến Telegram.
        Trả về True nếu thành công, False nếu thất bại hoặc chưa cấu hình.
        """
        if not self.is_configured:
            logger.debug("TelegramNotifier chưa được cấu hình (thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID)")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            if resp.status_code == 200:
                logger.info("Đã gửi thông báo Telegram thành công")
                return True
            else:
                logger.warning(
                    "Gửi Telegram thất bại: HTTP %d - %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        except Exception as exc:
            logger.warning("Lỗi kết nối khi gửi tin nhắn Telegram: %s", exc)
            return False

    def send_order_alert(
        self,
        mode: str,
        equity: float,
        orders: List[Dict[str, Any]],
        turnover: float = 0.0,
        capacity: Optional[int] = None,
        upnl: Optional[float] = None,
        data_age_hours: Optional[float] = None,
    ) -> bool:
        """Gửi thông báo danh mục lệnh khi hoàn thành tái cân bằng."""
        if not self.is_configured:
            return False

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_mode = f"🟢 [{mode.upper()}]" if mode.lower() != "dry_run" else "🟡 [DRY RUN]"

        lines = [
            f"🚀 <b>AEGIS TRADING SYSTEM — {status_mode}</b>",
            f"⏰ <i>{now_str}</i>",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"💰 <b>Equity:</b> <code>${equity:,.2f}</code>",
        ]

        if upnl is not None:
            pnl_sign = "+" if upnl >= 0 else ""
            pnl_icon = "🟢" if upnl >= 0 else "🔴"
            lines.append(f"{pnl_icon} <b>uPnL:</b> <code>{pnl_sign}${upnl:,.2f}</code>")

        if turnover > 0:
            lines.append(f"🔄 <b>Turnover:</b> <code>${turnover:,.2f}</code>")
        if capacity:
            lines.append(f"📊 <b>Vị thế mục tiêu:</b> {len(orders)} (Sức chứa: {capacity})")
        if data_age_hours is not None:
            lines.append(f"⏳ <b>Tuổi dữ liệu:</b> {data_age_hours:.1f}h")

        lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📋 <b>Chi tiết lệnh ({len(orders)} lệnh):</b>")

        # Hiển thị tối đa 25 lệnh trong một tin nhắn để tránh vượt giới hạn 4096 ký tự
        for o in orders[:25]:
            side = o.get("side", "").upper()
            symbol = o.get("symbol", "")
            qty = o.get("qty", 0)
            price = o.get("price", 0)
            notional = o.get("notional", 0)
            reason = o.get("reason", "")

            icon = "🟢" if side == "BUY" else "🔴"
            tag = f"[{reason}]" if reason else ""
            lines.append(
                f"{icon} <b>{side}</b> <code>{symbol}</code>: {qty} @ ${price} = <b>${notional:,.2f}</b> {tag}"
            )

        if len(orders) > 25:
            lines.append(f"<i>... và {len(orders) - 25} lệnh khác</i>")

        return self.send_message("\n".join(lines))

    def send_circuit_breaker_alert(
        self,
        reason: str,
        detail: str,
        drawdown: float,
        equity: Optional[float] = None,
    ) -> bool:
        """Gửi cảnh báo khẩn cấp khi hệ thống ngắt mạch (Circuit Breaker)."""
        if not self.is_configured:
            return False

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            "🚨 <b>[CẢNH BÁO KHẨN CẤP] AEGIS CIRCUIT BREAKER ĐÃ KÍCH HOẠT</b> 🚨",
            f"⏰ <i>{now_str}</i>",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"⚠️ <b>Lý do ngắt mạch:</b> <code>{reason}</code>",
            f"📝 <b>Chi tiết:</b> {detail}",
            f"📉 <b>Drawdown hiện tại:</b> <code>{drawdown*100:.2f}%</code>",
        ]
        if equity is not None:
            lines.append(f"💰 <b>Equity còn lại:</b> <code>${equity:,.2f}</code>")
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
        lines.append("🛑 <i>Hệ thống đã dừng gửi lệnh mới để bảo toàn vốn!</i>")

    def send_single_order_alert(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        notional: float,
        reason: str = "",
        status: str = "SUBMITTED",
        order_type: str = "LIMIT",
        client_order_id: Optional[str] = None,
    ) -> bool:
        """Thông báo từng lệnh riêng lẻ ngay tức thì khi phát sinh."""
        if not self.is_configured:
            return False

        icon = "🟢" if side.upper() == "BUY" else "🔴"
        status_icon = "✅" if status in ("FILLED", "ACKNOWLEDGED", "SUBMITTED", "NEW") else "⚠️"
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        lines = [
            f"{icon} <b>[LỆNH {side.upper()}] {symbol}</b>",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"⚡ <b>Trạng thái:</b> {status_icon} <code>{status}</code>",
            f"📦 <b>Khối lượng:</b> <code>{qty}</code>",
            f"💵 <b>Giá đặt:</b> <code>${price:,.4f}</code>",
            f"💰 <b>Giá trị:</b> <code>${notional:,.2f}</code>",
            f"🏷️ <b>Loại / Mục đích:</b> {order_type} {f'({reason})' if reason else ''}",
            f"⏰ <i>{now_str}</i>",
        ]
        return self.send_message("\n".join(lines))

    def send_anomaly_alert(
        self,
        title: str,
        message: str,
        level: str = "WARNING",
    ) -> bool:
        """Thông báo bất thường (Lỗi sàn, lệch sổ, rớt mạng, nến cũ...)."""
        if not self.is_configured:
            return False

        level_icon = "🚨" if level == "ERROR" else "⚠️"
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        lines = [
            f"{level_icon} <b>[CẢNH BÁO BẤT THƯỜNG] {title}</b>",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"📝 <b>Nội dung:</b> {message}",
            f"⏰ <i>{now_str}</i>",
        ]
        return self.send_message("\n".join(lines))


def send_telegram_alert(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """Hàm tiện ích nhanh gửi tin nhắn Telegram."""
    notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
    return notifier.send_message(text)
