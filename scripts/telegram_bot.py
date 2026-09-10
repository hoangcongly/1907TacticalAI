#!/usr/bin/env python3
"""
Aegis Interactive Telegram AI Assistant.

Biến bot Telegram thành một trợ lý định lượng AI thông minh 2 chiều:
1. Trả lời mọi câu hỏi tự nhiên bằng tiếng Việt qua Google Gemini 2.5 Flash.
2. Nạp dữ liệu sống từ tài khoản Binance (Equity, uPnL, vị thế mở) vào ngữ cảnh AI.
3. Hỗ trợ các lệnh điều khiển từ xa: /status, /rebalance, /costs, /kill.

Chạy:
    python scripts/telegram_bot.py
    hoặc chạy ngầm:
    nohup python scripts/telegram_bot.py > logs/telegram_ai.log 2>&1 &
"""
from datetime import datetime, timezone
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional
import requests

# Tự động nạp src/ vào sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
src_path = str(ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from aegis.core.credentials import load_dotenv
from aegis.core.state_store import StateStore
from aegis.data.ingestion.binance_rest import BinanceFuturesREST
from aegis.monitoring.ai_agent import GeminiAIAssistant, build_live_context_snapshot
from aegis.monitoring.alerts import TelegramNotifier
from aegis.oms.order_router import BinanceOrderRouter
from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline, LiveConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("telegram_bot")


class AegisTelegramAIBot:
    """Bot đàm thoại 2 chiều tích hợp AI Gemini và điều khiển hệ thống Aegis."""

    def __init__(self, config_path: Optional[str] = None):
        load_dotenv()
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.authorized_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not self.bot_token:
            raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong file .env!")

        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.ai = GeminiAIAssistant()

        # Khởi tạo kết nối Binance & State
        self.client = BinanceFuturesREST(testnet=True)
        self.store = StateStore("artifacts/live_state.json")

        # Khởi tạo Pipeline
        cfg_path = config_path
        if cfg_path is None:
            v3 = ROOT / "artifacts/strategy_v3.json"
            cfg_path = str(v3) if v3.is_file() else str(ROOT / "artifacts/strategy_validated.json")

        self.cfg = LiveConfig.from_artifacts(cfg_path)
        self.router = BinanceOrderRouter(client=self.client, max_order_notional=5000.0, dry_run=False)
        self.pipe = CrossSectionalLivePipeline(
            config=self.cfg,
            client=self.client,
            router=self.router,
            state_store=self.store,
            dry_run=False,
        )

        self.last_update_id = 0

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Gửi tin nhắn về cho người dùng."""
        url = f"{self.api_url}/sendMessage"
        payload = {
            "chat_id": self.authorized_chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as exc:
            logger.error("Lỗi khi gửi tin nhắn Telegram: %s", exc)
            return False

    def send_chat_action(self, action: str = "typing") -> None:
        """Hiển thị trạng thái 'đang nhập...' trên Telegram."""
        try:
            requests.post(f"{self.api_url}/sendChatAction", json={
                "chat_id": self.authorized_chat_id,
                "action": action,
            }, timeout=3)
        except Exception:
            pass

    def handle_status(self) -> str:
        """Xử lý lệnh /status — trả về trạng thái tài khoản thời gian thực."""
        try:
            bal = self.client.balance_usdt()
            wallet = bal.get("wallet_balance", 0.0)
            upnl = bal.get("unrealized_pnl", 0.0)
            equity = wallet + upnl
            positions = self.client.position_risk()
            live = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
            state = self.store.load()

            pnl_icon = "🟢" if upnl >= 0 else "🔴"
            pnl_sign = "+" if upnl >= 0 else ""

            lines = [
                "📊 <b>BÁO CÁO TÀI KHOẢN THỜI GIAN THỰC</b>",
                "━━━━━━━━━━━━━━━━━━━━━",
                f"💰 <b>Ví USDT:</b> <code>${wallet:,.2f}</code>",
                f"{pnl_icon} <b>uPnL:</b> <code>{pnl_sign}${upnl:,.2f}</code>",
                f"💎 <b>Tổng tài sản (Equity):</b> <code>${equity:,.2f}</code>",
                f"📉 <b>Drawdown:</b> <code>{state.drawdown(equity)*100:.2f}%</code>",
                f"🛡️ <b>Kill switch:</b> {'🔴 ĐÃ DỪNG' if state.is_dead else '🟢 Bình thường'}",
                f"🔄 <b>Số lượt đã cân:</b> {state.rebalance_count}",
                "━━━━━━━━━━━━━━━━━━━━━",
                f"🎯 <b>Vị thế đang mở ({len(live)} cặp):</b>",
            ]

            for p in live:
                sym = p["symbol"]
                amt = float(p["positionAmt"])
                entry = float(p.get("entryPrice", 0))
                mark = float(p.get("markPrice", 0))
                pnl = float(p.get("unRealizedProfit", 0))
                side = "LONG" if amt > 0 else "SHORT"
                icon = "🟢" if side == "LONG" else "🔴"
                p_icon = "📈" if pnl >= 0 else "📉"
                p_sign = "+" if pnl >= 0 else ""

                lines.append(
                    f"{icon} <b>{sym}</b> ({side}): {abs(amt)} @ ${entry:,.4f}\n"
                    f"   └ Mark: ${mark:,.4f} | {p_icon} uPnL: <code>{p_sign}${pnl:,.2f}</code>"
                )

            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ Lỗi khi truy vấn trạng thái: {exc}"

    def handle_costs(self) -> str:
        """Xử lý lệnh /costs — trả về báo cáo chi phí và độ trung lập."""
        try:
            from aegis.core.execution_log import ExecutionLog
            s = ExecutionLog().summary()
            if not s.get("n_rebalances"):
                return "ℹ️ Chưa có lượt tái cân bằng nào được ghi nhận."

            lines = [
                "💸 <b>BÁO CÁO CHI PHÍ & ĐỘ LỆCH TRUNG LẬP</b>",
                "━━━━━━━━━━━━━━━━━━━━━",
                f"🔄 <b>Số lượt cân:</b> {s['n_rebalances']}",
                f"🎯 <b>Tỷ lệ khớp maker:</b> <code>{s['maker_ratio']*100:.1f}%</code>",
                f"⚡ <b>Tỷ lệ khớp/dự tính:</b> <code>{s['fill_ratio']*100:.1f}%</code>",
                f"💵 <b>Chi phí THỰC TẾ:</b> <code>{s['realized_cost_bps']:.2f} bp</code>",
                f"⚖️ <b>Lệch hướng TB:</b> <code>{s['avg_net_exposure_pct']:.2f}%</code> (0% = trung lập tuyệt đối)",
                f"📦 <b>Tổng volume:</b> <code>${s['total_traded_usd']:,.0f}</code>",
                f"🏷️ <b>Tổng phí sàn:</b> <code>${s['total_cost_usd']:,.2f}</code>",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"⚠️ Lỗi khi đọc báo cáo chi phí: {exc}"

    def handle_rebalance(self) -> str:
        """Xử lý lệnh /rebalance — chạy tái cân bằng ngay lập tức."""
        self.send_message("⏳ Đang tiến hành quét tín hiệu và tái cân bằng danh mục...")
        try:
            res = self.pipe.run_once(skip_data_refresh=False)
            orders = res.get("orders", [])
            lines = [
                f"✅ <b>TÁI CÂN BẰNG HOÀN TẤT — [{res.get('action')}]</b>",
                f"💰 <b>Equity:</b> ${res.get('equity', 0):,.2f}",
                f"🎯 <b>Số lệnh:</b> {len(orders)} | <b>Turnover:</b> ${res.get('turnover', 0):,.2f}",
            ]
            for o in orders[:15]:
                side = o.get("side", "")
                icon = "🟢" if side == "BUY" else "🔴"
                lines.append(f"{icon} {side} <b>{o.get('symbol')}</b>: {o.get('qty')} @ ${o.get('price')} (${o.get('notional')}) [{o.get('reason')}]")

            return "\n".join(lines)
        except Exception as exc:
            return f"❌ Lỗi khi thực hiện tái cân bằng: {exc}"

    def handle_kill(self) -> str:
        """Xử lý lệnh /kill — dừng khẩn cấp, đóng sạch vị thế."""
        self.send_message("🚨 <b>ĐANG KÍCH HOẠT DỪNG KHẨN CẤP (KILL SWITCH)...</b>")
        try:
            report = self.pipe.kill()
            closed = report.get("closed", [])
            return f"🛑 <b>ĐÃ ĐÓNG SẠCH TOÀN BỘ VỊ THẾ!</b>\n- Số vị thế đã đóng: {len(closed)}\n- Bot đã chuyển sang trạng thái dừng an toàn."
        except Exception as exc:
            return f"❌ Lỗi khi kích hoạt kill switch: {exc}"

    def process_message(self, text: str) -> str:
        """Xử lý tin nhắn từ người dùng (lệnh hoặc câu hỏi AI)."""
        cmd = text.strip().lower()

        if cmd in ("/start", "/help"):
            return (
                "👋 <b>Xin chào Sếp! Tôi là Aegis Tactical AI.</b>\n\n"
                "Tôi là trợ lý định lượng cá nhân của anh, quản lý danh mục Futures tự động 24/7.\n\n"
                "📌 <b>Các lệnh nhanh:</b>\n"
                "• /status — Xem số dư, lãi/lỗ và các vị thế đang mở\n"
                "• /costs — Xem chi phí giao dịch & độ trung lập\n"
                "• /rebalance — Kích hoạt tái cân bằng danh mục ngay\n"
                "• /kill — Dừng khẩn cấp, đóng sạch toàn bộ vị thế\n\n"
                "💬 <b>Trò chuyện tự nhiên:</b>\n"
                "Anh có thể nhắn bất kỳ câu hỏi nào bằng tiếng Việt (ví dụ: <i>'Tình hình lãi lỗ sao rồi em?', 'Tại sao lại Short BCH?', 'Thị trường hôm nay thế nào?'</i>). Tôi sẽ phân tích dựa trên dữ liệu thực tế và trả lời anh ngay!"
            )
        elif cmd == "/status":
            return self.handle_status()
        elif cmd == "/costs":
            return self.handle_costs()
        elif cmd == "/rebalance":
            return self.handle_rebalance()
        elif cmd == "/kill":
            return self.handle_kill()

        # Nếu là câu hỏi tự nhiên: gọi AI Gemini 2.5 Flash kèm dữ liệu thời gian thực
        self.send_chat_action("typing")
        live_context = build_live_context_snapshot(self.client, self.store)
        return self.ai.generate_reply(user_message=text, live_context=live_context)

    def start_polling(self) -> None:
        """Vòng lặp lắng nghe tin nhắn Telegram 24/7."""
        logger.info("Khởi động Aegis Telegram AI Assistant...")
        self.send_message(
            "⚡ <b>Aegis AI Assistant đã sẵn sàng!</b>\n"
            "Tôi đã được kết nối với bộ não Google Gemini và tài khoản Binance Futures của sếp.\n"
            "Gõ /status để kiểm tra hoặc hỏi tôi bất cứ điều gì!"
        )

        while True:
            try:
                url = f"{self.api_url}/getUpdates"
                params = {"offset": self.last_update_id, "timeout": 20}
                resp = requests.get(url, params=params, timeout=25)

                if resp.status_code != 200:
                    time.sleep(2)
                    continue

                data = resp.json()
                for update in data.get("result", []):
                    self.last_update_id = max(self.last_update_id, update["update_id"] + 1)
                    msg = update.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    sender_id = str(msg.get("from", {}).get("id", ""))
                    text = msg.get("text", "")

                    if not text:
                        continue

                    # Xác thực bảo mật: Chỉ nhận lệnh từ Chat ID đã cấu hình
                    if self.authorized_chat_id and chat_id != self.authorized_chat_id and sender_id != self.authorized_chat_id:
                        logger.warning("Bỏ qua tin nhắn từ nguồn không xác thực: chat_id=%s, text=%s", chat_id, text)
                        continue

                    logger.info("Nhận tin nhắn: %s", text)
                    reply = self.process_message(text)
                    self.send_message(reply)

            except KeyboardInterrupt:
                logger.info("Dừng bot Telegram.")
                break
            except Exception as exc:
                logger.error("Lỗi trong vòng lặp polling: %s", exc)
                time.sleep(3)


if __name__ == "__main__":
    bot = AegisTelegramAIBot()
    bot.start_polling()
