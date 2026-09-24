"""
[FIX F52] Funding đứng yên từ 10/09 mà daemon vẫn giao dịch như thường.

Ba mắt xích cùng hỏng: (1) `refresh_data` chỉ tải nến, (2) script tải ban đầu chỉ ghi
file funding MỘT LẦN, (3) `load_funding_panel_v2` ffill giá trị cuối tới vô hạn — nên 5
tín hiệu carry thấy một con số cũ trông y như số mới. Cổng STALE_DATA chỉ nhìn NẾN.

Các test dưới đây khoá cả đường vá (tải + gộp funding mỗi lượt) lẫn đường chặn
(funding cũ quá trần thì HALT có lý do, không im lặng giao dịch).
"""
import time

import numpy as np
import pandas as pd

import aegis.pipelines.xs_live_pipeline as xlp
from aegis.core.state_store import LiveState, StateStore
from aegis.data.ingestion.binance_history import save_funding_rates
from aegis.data.panel_v2 import funding_last_ms
from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline, LiveConfig

H = 3_600_000


# ------------------------------------------------------------------ lưu trữ
def test_luu_funding_gop_khong_ghi_de_lich_su(tmp_path):
    root = str(tmp_path)
    save_funding_rates({1000: 0.0001, 2000: 0.0002}, "AAAUSDT", root=root)
    save_funding_rates({2000: 0.0005, 3000: 0.0003}, "AAAUSDT", root=root)   # chồng lặp
    df = pd.read_parquet(tmp_path / "AAAUSDT_funding.parquet")
    assert df["funding_time"].tolist() == [1000, 2000, 3000]
    assert df["rate"].tolist() == [0.0001, 0.0005, 0.0003]      # bản mới thắng ở mốc trùng


def test_luu_funding_chay_lai_cho_cung_ket_qua(tmp_path):
    for _ in range(3):
        save_funding_rates({1000: 0.0001, 2000: 0.0002}, "AAAUSDT", root=str(tmp_path))
    assert len(pd.read_parquet(tmp_path / "AAAUSDT_funding.parquet")) == 2


def test_moc_funding_cuoi_doc_tu_nguon_thieu_file_la_nan(tmp_path):
    save_funding_rates({1000: 0.1, 5000: 0.2}, "AAAUSDT", root=str(tmp_path))
    last = funding_last_ms(["AAAUSDT", "ZZZUSDT"], root=str(tmp_path))
    assert last["AAAUSDT"] == 5000
    assert np.isnan(last["ZZZUSDT"])


# ------------------------------------------------------------------ refresh_data
class _Client:
    credentials = None

    def balance_usdt(self):
        return {"wallet_balance": 1000.0, "available_balance": 1000.0, "unrealized_pnl": 0.0}

    def position_risk(self, symbol=None):
        return []

    def open_orders(self, symbol=None):
        return []

    def symbol_filters(self, symbol):
        return {"tick_size": 0.001, "step_size": 0.1, "min_qty": 0.1, "min_notional": 5.0}

    def exchange_info(self, symbol=None):
        return {"symbols": [{"symbol": s, "status": "TRADING", "quoteAsset": "USDT",
                             "contractType": "PERPETUAL"} for s in ("AAAUSDT", "BBBUSDT")]}

    def _request(self, method, path, params=None, signed=False):
        if "ticker" in path:
            return [{"symbol": s, "quoteVolume": "1e12"} for s in ("AAAUSDT", "BBBUSDT")]
        return {}


def _pipe(tmp_path, engine="v3"):
    store = StateStore(str(tmp_path / "s.json"))
    store.save(LiveState(peak_equity=1000.0))
    cfg = LiveConfig(universe=["AAAUSDT", "BBBUSDT"], leverage=2.0, engine=engine)
    return CrossSectionalLivePipeline(config=cfg, client=_Client(), state_store=store,
                                      dry_run=True)


def test_refresh_data_tai_va_luu_funding_moi_cap(tmp_path, monkeypatch):
    p = _pipe(tmp_path)
    saved = {}
    monkeypatch.setattr(xlp, "download_klines", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(xlp, "save_klines", lambda *a, **k: None)
    monkeypatch.setattr(xlp, "download_funding_rates", lambda s, **k: {1000: 0.0001})
    monkeypatch.setattr(xlp, "save_funding_rates", lambda r, s, **k: saved.update({s: r}))
    p.refresh_data(days=3)
    assert set(saved) == {"AAAUSDT", "BBBUSDT"}


def test_funding_hong_khong_keo_nen_hong_theo(tmp_path, monkeypatch):
    p = _pipe(tmp_path)
    klines = []

    def boom(*a, **k):
        raise RuntimeError("funding endpoint chết")
    monkeypatch.setattr(xlp, "download_klines", lambda s, *a, **k: klines.append(s))
    monkeypatch.setattr(xlp, "save_klines", lambda *a, **k: None)
    monkeypatch.setattr(xlp, "download_funding_rates", boom)
    assert p.refresh_data(days=3) == 2          # cả hai cặp vẫn cập nhật NẾN
    assert klines == ["AAAUSDT", "BBBUSDT"]


# ------------------------------------------------------------------ cổng STALE_FUNDING
def _run_with_funding_age(tmp_path, monkeypatch, age_hours, engine="v3"):
    p = _pipe(tmp_path, engine=engine)
    now = int(time.time() * 1000)
    monkeypatch.setattr(p, "compute_target_weights",
                        lambda **kw: ({"AAAUSDT": 0.5, "BBBUSDT": -0.5}, now))
    monkeypatch.setattr(xlp, "funding_last_ms",
                        lambda syms, **k: pd.Series({s: now - age_hours * H for s in syms}))
    return p.run_once(skip_data_refresh=True)


def test_funding_dung_yen_14_ngay_thi_halt_co_ly_do(tmp_path, monkeypatch):
    """Đúng tình huống thật: nến tươi, funding đứng yên từ 10/09 (14 ngày)."""
    res = _run_with_funding_age(tmp_path, monkeypatch, age_hours=14 * 24)
    assert res["action"] == "HALT" and res["reason"] == "STALE_FUNDING"
    assert "2/2" in res["detail"]


def test_funding_tuoi_thi_qua_cong(tmp_path, monkeypatch):
    res = _run_with_funding_age(tmp_path, monkeypatch, age_hours=6)
    assert res.get("reason") != "STALE_FUNDING"
    assert res["funding_age_hours"] == 6.0


def test_khong_co_file_funding_nao_cung_la_halt(tmp_path, monkeypatch):
    """Thiếu SẠCH funding = họ carry tắt câm. Phải dừng, không được coi là 'tươi'."""
    res = _run_with_funding_age(tmp_path, monkeypatch, age_hours=np.nan)
    assert res["action"] == "HALT" and res["reason"] == "STALE_FUNDING"


def test_engine_v1_khong_bi_cong_funding_chan(tmp_path, monkeypatch):
    res = _run_with_funding_age(tmp_path, monkeypatch, age_hours=14 * 24, engine="v1")
    assert res.get("reason") != "STALE_FUNDING"
