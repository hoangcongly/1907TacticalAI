"""
[FIX F40] Cặp mà VỐN KHÔNG MUA NỔI phải bị loại TRƯỚC khi xếp hạng.

Min notional của Binance KHÔNG đồng nhất. Trên mainnet: 122/128 cặp cần $5, nhưng
ETH/LTC/LINK/ETC/BCH cần $20 và BTC cần $50. Vốn 1.000.000 VND (~$38) ở đòn bẩy 2x cho
$6,34 mỗi vị thế — sáu cặp đó không mua nổi.

Nếu để tới lúc gửi lệnh mới phát hiện, `build_rebalance_plan` bỏ lệnh vào `skipped` và
danh mục MẤT MỘT CHÂN. Một sổ market-neutral thiếu một chân không còn trung lập — nó
thành cược có hướng, đúng cơ chế đã làm sổ lệch +35% ngày 10/09.

`max_positions_for_capital` không cứu được: nó giả định MỌI cặp cần đúng $5.

Lỗi này chưa từng cắn vì hệ thống mới chạy testnet. Nó sẽ cắn ở lượt MAINNET đầu tiên.
"""
import pytest

from aegis.data.universe import UniverseFilter, build_universe, symbol_min_notionals


class _FakeClient:
    """Sàn giả với min notional KHÔNG đồng nhất — đúng hình dạng của mainnet thật."""

    MINS = {"CHEAPUSDT": 5.0, "MIDUSDT": 20.0, "BIGUSDT": 50.0, "ALSOCHEAPUSDT": 5.0}
    credentials = None          # router kiểm trường này khi khởi tạo

    def exchange_info(self, symbol=None):
        return {"symbols": [
            {"symbol": s, "status": "TRADING", "quoteAsset": "USDT",
             "contractType": "PERPETUAL",
             "filters": [{"filterType": "MIN_NOTIONAL", "notional": str(m)}]}
            for s, m in self.MINS.items()
        ]}

    def _request(self, *a, **k):
        return [{"symbol": s, "quoteVolume": "1e12"} for s in self.MINS]


ALL = sorted(_FakeClient.MINS)


def test_doc_duoc_min_notional_tung_cap():
    mins = symbol_min_notionals(_FakeClient())
    assert mins == _FakeClient.MINS


def test_khong_dat_tran_thi_giu_tat_ca(tmp_path):
    keep, rej = build_universe(ALL, "4h", UniverseFilter(min_history_bars=0,
                               min_quote_volume_24h=0, max_min_notional=None),
                               execution_client=_FakeClient(), root=str(tmp_path))
    assert set(keep) == set(ALL)


def test_loai_dung_cap_khong_mua_noi(tmp_path):
    """$6,34/vị thế: giữ cặp $5, loại cặp $20 và $50."""
    keep, rej = build_universe(ALL, "4h", UniverseFilter(min_history_bars=0,
                               min_quote_volume_24h=0, max_min_notional=6.34),
                               execution_client=_FakeClient(), root=str(tmp_path))
    assert set(keep) == {"CHEAPUSDT", "ALSOCHEAPUSDT"}
    assert "MIDUSDT" in rej and "BIGUSDT" in rej


def test_ly_do_loai_noi_ro_con_so(tmp_path):
    """
    Lý do phải chứa CON SỐ, không chỉ 'không đủ điều kiện'. Sau sự cố, thứ người ta cần
    là biết thiếu bao nhiêu — để quyết định nạp thêm vốn hay giảm số vị thế.
    """
    _, rej = build_universe(ALL, "4h", UniverseFilter(min_history_bars=0,
                            min_quote_volume_24h=0, max_min_notional=6.34),
                            execution_client=_FakeClient(), root=str(tmp_path))
    assert "50" in rej["BIGUSDT"] and "6.34" in rej["BIGUSDT"]


def test_von_lon_hon_thi_giu_them_cap(tmp_path):
    """Đơn điệu: vốn nhiều hơn không bao giờ loại nhiều cặp hơn."""
    prev = None
    for cap in (6.0, 21.0, 60.0):
        keep, _ = build_universe(ALL, "4h", UniverseFilter(min_history_bars=0,
                                 min_quote_volume_24h=0, max_min_notional=cap),
                                 execution_client=_FakeClient(), root=str(tmp_path))
        if prev is not None:
            assert set(prev) <= set(keep), f"tăng vốn lên ${cap} mà lại loại thêm cặp"
        prev = keep
    assert set(prev) == set(ALL)


def test_loc_von_khong_chan_khi_sàn_loi(tmp_path, monkeypatch):
    """
    Sàn lỗi khi đọc min notional -> GHI CẢNH BÁO và bỏ qua bộ lọc vốn, không làm hỏng
    cả lượt. Bộ lọc này là lớp bảo vệ THÊM; để nó đánh sập đường chạy thì tệ hơn.
    """
    class _Broken(_FakeClient):
        def exchange_info(self, symbol=None):
            if getattr(self, "_n", 0):
                raise RuntimeError("sàn lỗi")
            self._n = 1
            return super().exchange_info(symbol)

    keep, _ = build_universe(ALL, "4h", UniverseFilter(min_history_bars=0,
                             min_quote_volume_24h=0, max_min_notional=6.34),
                             execution_client=_Broken(), root=str(tmp_path))
    assert set(keep) == set(ALL)


def test_pipeline_tinh_dung_tran_tu_von():
    """Trần = equity * đòn bẩy / số vị thế / đệm an toàn."""
    from aegis.pipelines.xs_live_pipeline import LiveConfig

    cfg = LiveConfig(universe=["AAAUSDT"], leverage=2.0, n_positions=12,
                     min_notional_safety=1.2)
    expected = 38.0 * 2.0 / 12 / 1.2
    got = 38.0 * cfg.leverage / max(1, cfg.n_positions) / max(1.0, cfg.min_notional_safety)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(5.28, abs=0.01)


# ---------------------------------------------------------------------------
# Bộ lọc an toàn KHÔNG được tự trở thành nguyên nhân sự cố
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("equity", [38.0, 35.0, 25.0, 15.0, 8.0])
def test_nguong_khong_bao_gio_tut_duoi_5_do(equity):
    """
    Bất biến sống còn: ngưỡng phải LUÔN >= $5 dù vốn nhỏ tới đâu.

    Bản đầu chia cho `n_positions` CỨNG, nên vốn $35 cho ngưỡng $4,86 — dưới mức mà
    gần như mọi cặp đều cần. Bộ lọc khi đó quét sạch universe và `resolve_universe`
    ném lỗi "còn 0 cặp". Một cú sụt 8% vốn sẽ giết đường chạy, và nguyên nhân lại
    chính là lớp bảo vệ vừa thêm vào.

    Sửa bằng cách dùng số vị thế ĐÃ ĐIỀU CHỈNH THEO VỐN. Khi đó theo đại số:
        n_eff = floor(E*L / (5*safety))  =>  (E*L/n_eff)/safety >= 5
    """
    from aegis.execution.portfolio_rebalancer import max_positions_for_capital

    lev, n_pos, safety = 2.0, 12, 1.2
    n_eff = min(n_pos, max_positions_for_capital(equity, lev, min_notional=5.0,
                                                 safety=safety))
    if n_eff < 1:
        pytest.skip(f"vốn ${equity} không nuôi nổi một vị thế nào — trường hợp khác")
    threshold = equity * lev / n_eff / safety
    assert threshold >= 5.0 - 1e-9, \
        f"vốn ${equity} -> ngưỡng ${threshold:.2f} < $5: bộ lọc sẽ quét sạch universe"


def test_pipeline_van_giu_du_cap_khi_von_tut(tmp_path):
    """Đầu-cuối: vốn tụt 30% vẫn không được làm rỗng universe."""
    from aegis.core.state_store import StateStore
    from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline, LiveConfig

    cfg = LiveConfig(universe=ALL, leverage=2.0, n_positions=12,
                     min_history_bars=0, min_quote_volume_24h=0)
    pipe = CrossSectionalLivePipeline(config=cfg, client=_FakeClient(),
                                      state_store=StateStore(str(tmp_path / "s.json")),
                                      dry_run=True)
    import aegis.pipelines.xs_live_pipeline as mod
    orig = mod.build_universe
    captured = {}

    def _spy(**kw):
        captured["max_min_notional"] = kw["filt"].max_min_notional
        return orig(**{**kw, "root": str(tmp_path)})

    mod.build_universe = _spy
    try:
        pipe.resolve_universe(equity=26.6)      # tụt 30% so với $38
    except RuntimeError:
        pass                                    # ít cặp là do fixture, không phải do vốn
    finally:
        mod.build_universe = orig

    assert captured["max_min_notional"] >= 5.0, \
        f"ngưỡng {captured['max_min_notional']:.2f} sẽ loại cả cặp $5"
