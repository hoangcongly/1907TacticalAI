"""Unit test cho TelegramNotifier và module alerts."""
from unittest.mock import MagicMock, patch
import pytest

from aegis.monitoring.alerts import TelegramNotifier, send_telegram_alert


def test_telegram_notifier_not_configured_by_default():
    """Khi chưa có token/chat_id thì is_configured=False và gửi trả về False không lỗi."""
    notifier = TelegramNotifier(bot_token="", chat_id="")
    assert not notifier.is_configured
    assert not notifier.send_message("test")


@patch("requests.post")
def test_telegram_notifier_send_message_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="12345:TOKEN", chat_id="98765")
    assert notifier.is_configured
    ok = notifier.send_message("Hello Aegis")

    assert ok is True
    assert mock_post.called
    url, kwargs = mock_post.call_args
    assert "12345:TOKEN" in url[0]
    assert kwargs["json"]["chat_id"] == "98765"
    assert kwargs["json"]["text"] == "Hello Aegis"


@patch("requests.post")
def test_telegram_notifier_send_order_alert(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    notifier = TelegramNotifier(bot_token="fake_token", chat_id="fake_chat")
    orders = [
        {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.5, "price": 60000.0, "notional": 30000.0, "reason": "mở"},
        {"symbol": "ETHUSDT", "side": "SELL", "qty": 5.0, "price": 3000.0, "notional": 15000.0, "reason": "đóng"},
    ]
    ok = notifier.send_order_alert(
        mode="TESTNET",
        equity=50000.0,
        orders=orders,
        turnover=45000.0,
        capacity=10,
        upnl=125.50,
    )
    assert ok is True
    text = mock_post.call_args[1]["json"]["text"]
    assert "BTCUSDT" in text
    assert "ETHUSDT" in text
    assert "$50,000.00" in text
    assert "+$125.50" in text


@patch("requests.post")
def test_telegram_notifier_network_failure_is_handled_gracefully(mock_post):
    mock_post.side_effect = Exception("Connection Timeout")

    notifier = TelegramNotifier(bot_token="token", chat_id="chat")
    ok = notifier.send_message("Should not crash")
    assert ok is False
