"""
Aegis AI Agent — Tích hợp Google Gemini 2.5 Flash làm bộ não định lượng cho Telegram Bot.
"""
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Dict, Optional
import requests

from aegis.core.credentials import load_dotenv

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là Aegis Tactical AI — trợ lý định lượng (Quantitative Trading AI) thông minh và quyền lực của hệ thống Aegis Trading System (perpetual futures USDⓈ-M).
Bạn đang giao tiếp trực tiếp với chủ sở hữu hệ thống qua Telegram.

Phong cách của bạn:
- Xưng hô: "tôi" hoặc "em", gọi người dùng là "anh" / "sếp" / "bạn" thân mật, chuyên nghiệp, tự tin và sắc sảo như một chuyên gia định lượng cấp cao tại quỹ đầu tư.
- Trả lời ngắn gọn, súc tích, đi thẳng vào vấn đề, dùng gạch đầu dòng và emoji để dễ đọc trên điện thoại.
- Bạn có toàn quyền truy cập dữ liệu thời gian thực của tài khoản (được cung cấp bên dưới). Hãy dựa vào số liệu thực tế này để trả lời chính xác, trung thực, không bịa đặt số liệu.

Kiến thức chiến lược cốt lõi:
- Chiến lược hiện tại: Aegis v3 Cross-Sectional Market-Neutral (Trung lập thị trường).
- Danh mục mục tiêu: 12 vị thế (6 Long mạnh nhất, 6 Short yếu nhất) xếp hạng theo Decile Spread của 26 tín hiệu Alpha (Momentum, Carry, Flow, Volatility, Microstructure).
- Chu kỳ tái cân bằng: 72 giờ / lượt.
- Đòn bẩy an toàn: 2.0x (Nửa Kelly).
- Mục tiêu: Kiếm lợi nhuận từ chênh lệch giữa coin mạnh và coin yếu, không phụ thuộc vào việc Bitcoin tăng hay giảm.
"""


class GeminiAIAssistant:
    """Giao tiếp với Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        timeout_s: float = 15.0,
    ):
        load_dotenv()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model_name = model_name
        self.timeout_s = timeout_s

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_reply(self, user_message: str, live_context: str = "") -> str:
        """Sinh câu trả lời từ Gemini dựa trên tin nhắn và dữ liệu thực tế."""
        if not self.is_configured:
            return "⚠️ Chưa cấu hình GEMINI_API_KEY trong file .env."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== DỮ LIỆU TÀI KHOẢN & HỆ THỐNG THỜI GIAN THỰC ===\n"
            f"{live_context}\n"
            f"====================================================\n\n"
            f"Người dùng hỏi: \"{user_message}\"\n\n"
            f"Hãy trả lời câu hỏi trên:"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return "Xin lỗi sếp, tôi không nhận được nội dung trả lời từ mô hình."
            else:
                logger.error("Gemini API error: %d - %s", resp.status_code, resp.text[:200])
                return f"⚠️ Lỗi kết nối Gemini API (HTTP {resp.status_code})."
        except Exception as exc:
            logger.error("Lỗi khi gọi Gemini: %s", exc)
            return f"⚠️ Không thể kết nối với bộ não AI: {exc}"


def build_live_context_snapshot(client: Any, state_store: Any) -> str:
    """Tạo bản chụp dữ liệu sống của tài khoản để nạp vào prompt của AI."""
    lines = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"Thời gian hiện tại: {now_str}")

    try:
        bal = client.balance_usdt()
        wallet = bal.get("wallet_balance", 0.0)
        upnl = bal.get("unrealized_pnl", 0.0)
        equity = wallet + upnl
        lines.append(f"- Ví USDT: ${wallet:,.2f}")
        lines.append(f"- Lợi nhuận chưa chốt (uPnL): ${upnl:+,.2f}")
        lines.append(f"- Tổng tài sản ròng (Equity): ${equity:,.2f}")
    except Exception as e:
        lines.append(f"- Lỗi đọc số dư: {e}")

    try:
        positions = client.position_risk()
        live = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
        lines.append(f"- Số vị thế đang mở: {len(live)}")
        for p in live:
            sym = p["symbol"]
            amt = float(p["positionAmt"])
            entry = float(p.get("entryPrice", 0))
            mark = float(p.get("markPrice", 0))
            pnl = float(p.get("unRealizedProfit", 0))
            side = "LONG" if amt > 0 else "SHORT"
            lines.append(f"  • {sym} ({side}): {abs(amt)} @ ${entry:,.4f} | Giá hiện tại: ${mark:,.4f} | uPnL: ${pnl:+,.2f}")
    except Exception as e:
        lines.append(f"- Lỗi đọc vị thế: {e}")

    try:
        state = state_store.load()
        lines.append(f"- Số lần đã tái cân bằng: {state.rebalance_count}")
        lines.append(f"- Đỉnh vốn (Peak Equity): ${state.peak_equity:,.2f}")
        if state.last_rebalance_ms > 0:
            hrs = (int(datetime.now(timezone.utc).timestamp() * 1000) - state.last_rebalance_ms) / 3_600_000
            lines.append(f"- Đã qua {hrs:.1f} giờ kể từ lượt tái cân bằng trước")
        lines.append(f"- Trạng thái Kill Switch: {'ĐÃ CHẾT/DỪNG' if state.is_dead else 'Bình thường'}")
    except Exception as e:
        lines.append(f"- Lỗi đọc trạng thái state_store: {e}")

    return "\n".join(lines)
