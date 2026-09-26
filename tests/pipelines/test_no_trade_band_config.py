"""Độ rộng dải không giao dịch đọc từ JSON; thiếu khoá thì chạy y như trước (0,20)."""
import json

from aegis.execution.portfolio_rebalancer import DEFAULT_NO_TRADE_BAND
from aegis.pipelines.xs_live_pipeline import LiveConfig


def test_thieu_khoa_giu_hang_so_cu():
    assert LiveConfig.from_artifacts("artifacts/strategy_v3.json").no_trade_band \
        == DEFAULT_NO_TRADE_BAND == 0.20


def test_doc_khoa_tu_json(tmp_path):
    d = json.load(open("artifacts/strategy_v3.json"))
    d["config"]["no_trade_band"] = 0.5
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(d))
    assert LiveConfig.from_artifacts(str(p)).no_trade_band == 0.5
