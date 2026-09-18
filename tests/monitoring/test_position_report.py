"""
Test báo cáo vị thế gửi Telegram.

Hai nhóm tính chất được khoá ở đây, cả hai đều từ lỗi thật gặp khi dựng:

  1. KHÔNG GÂY HIỂU NHẦM. Đòn bẩy sàn (20x) và đòn bẩy thực (2x) khác nhau 10 lần;
     lãi trên ký quỹ (+582%) và mức giá đã chạy (+41%) khác nhau 14 lần. Hiển thị
     một con số mà không kèm con số kia sẽ khiến người đọc hiểu sai mức rủi ro.

  2. TIN PHẢI TỚI NƠI. Telegram từ chối tin > 4096 ký tự và thư viện chỉ trả về
     False, KHÔNG ném ngoại lệ. Báo cáo 12 vị thế dài 4482 ký tự — vượt ngay từ
     lần đầu, và nếu không chia tin thì nó im lặng không bao giờ đến.
"""
import pytest

from aegis.monitoring.position_report import (
    TELEGRAM_LIMIT, build_position_report, chunk_message, position_rows,
)


def _pos(symbol="AAAUSDT", amt=100.0, entry=10.0, mark=11.0, lev=20.0,
         liq=5.0, upnl=100.0, notional=None):
    return {
        "symbol": symbol, "positionAmt": str(amt), "entryPrice": str(entry),
        "markPrice": str(mark), "leverage": str(lev), "liquidationPrice": str(liq),
        "unRealizedProfit": str(upnl),
        "notional": str(notional if notional is not None else amt * mark),
    }


# ------------------------------------------------------------- chuẩn hoá
def test_phan_biet_long_short():
    rows = position_rows([_pos(amt=100.0), _pos(symbol="BBB", amt=-50.0)], equity=1000.0)
    assert {r["symbol"]: r["side"] for r in rows} == {"AAAUSDT": "LONG", "BBB": "SHORT"}


def test_von_vao_lenh_bang_notional_chia_don_bay():
    r = position_rows([_pos(amt=100.0, mark=11.0, lev=20.0)], equity=1000.0)[0]
    assert r["notional"] == pytest.approx(1100.0)
    assert r["margin"] == pytest.approx(55.0)


def test_phan_tram_tai_khoan_tinh_theo_notional():
    """Notional mới là rủi ro đang gánh; ký quỹ chỉ là tiền đặt cọc."""
    r = position_rows([_pos(amt=100.0, mark=11.0)], equity=2200.0)[0]
    assert r["pct_account"] == pytest.approx(50.0)


def test_gia_chay_dung_dau_cho_vi_the_SHORT():
    """
    SHORT lãi khi giá GIẢM. Nếu không nhân dấu theo hướng vị thế, báo cáo sẽ hiện
    'giá chạy -10%' cho một lệnh đang LÃI — hiểu ngược hoàn toàn.
    """
    long_r = position_rows([_pos(amt=100.0, entry=10.0, mark=11.0)], 1000.0)[0]
    short_r = position_rows([_pos(amt=-100.0, entry=10.0, mark=9.0)], 1000.0)[0]
    assert long_r["price_move_pct"] == pytest.approx(10.0)
    assert short_r["price_move_pct"] == pytest.approx(10.0), "short giá giảm là LÃI"


def test_short_gia_tang_thi_am():
    r = position_rows([_pos(amt=-100.0, entry=10.0, mark=11.0)], 1000.0)[0]
    assert r["price_move_pct"] == pytest.approx(-10.0)


def test_hai_cach_quy_doi_lai_lo_deu_co_mat():
    """Ký quỹ và notional cho hai con số rất khác nhau — phải có cả hai."""
    r = position_rows([_pos(amt=100.0, mark=11.0, lev=20.0, upnl=110.0)], 1000.0)[0]
    assert r["upnl_pct_margin"] == pytest.approx(200.0)   # 110 / 55
    assert r["upnl"] / r["notional"] * 100 == pytest.approx(10.0)


def test_gia_thanh_ly_bang_0_khong_hien_thi_thanh_sap_bi_thanh_ly():
    """Sàn trả 0 khi vị thế quá an toàn để tính. Hiện '0' trông như sắp cháy."""
    r = position_rows([_pos(liq=0.0)], 1000.0)[0]
    assert r["liq"] is None and r["liq_distance_pct"] is None


def test_bo_qua_vi_the_rong():
    assert position_rows([_pos(amt=0.0)], 1000.0) == []


def test_sap_xep_theo_notional_giam_dan():
    rows = position_rows([_pos(symbol="NHO", amt=10.0, mark=1.0),
                          _pos(symbol="TO", amt=100.0, mark=10.0)], 1000.0)
    assert [r["symbol"] for r in rows] == ["TO", "NHO"]


# ------------------------------------------------------------- nội dung báo cáo
def test_bao_cao_neu_ro_KHONG_co_TP_SL():
    """
    Người dùng hỏi 'giá chốt lời, giá chốt lỗ'. Chiến lược không có chúng. Báo cáo
    PHẢI nói rõ điều đó thay vì bịa ra một con số.
    """
    txt = build_position_report([_pos()], 1000.0, 900.0, 1000.0)
    assert "KHÔNG dùng TP/SL theo giá" in txt
    assert "tái cân bằng" in txt.lower()


def test_bao_cao_co_nguong_ngat_mach_lam_chot_lo_that():
    txt = build_position_report([_pos()], 1000.0, 900.0, 1000.0)
    for lvl in ("TIER1", "TIER2", "TIER3"):
        assert lvl in txt
    assert "900.00" in txt      # TIER1 = 1000 * 0.9


def test_bao_cao_phan_biet_hai_loai_don_bay():
    txt = build_position_report([_pos(amt=100.0, mark=11.0, lev=20.0)], 1000.0, 900.0, 1000.0)
    assert "đòn bẩy sàn 20x" in txt
    assert "Đòn bẩy THỰC" in txt and "1.10x" in txt   # 1100 notional / 1000 equity


def test_cach_dinh_von_hien_dau_am_khi_dang_lo():
    txt = build_position_report([_pos()], equity=900.0, wallet=900.0, peak_equity=1000.0)
    assert "-10.00%" in txt, "đang dưới đỉnh mà hiện dấu dương là hiểu ngược"


def test_cat_bot_khi_qua_nhieu_vi_the():
    pos = [_pos(symbol=f"S{i}", amt=10.0 + i) for i in range(20)]
    txt = build_position_report(pos, 10000.0, 9000.0, 10000.0, max_rows=5)
    assert "và 15 vị thế nữa" in txt


# ------------------------------------------------------------- chia tin
def test_tin_ngan_khong_bi_chia():
    assert chunk_message("ngắn") == ["ngắn"]


def test_tin_dai_bi_chia_va_moi_phan_duoi_gioi_han():
    text = "\n".join(f"dòng số {i} " + "x" * 100 for i in range(200))
    parts = chunk_message(text)
    assert len(parts) > 1
    assert all(len(p) <= TELEGRAM_LIMIT + 40 for p in parts)   # +40 cho nhãn (i/n)


def test_chia_tin_khong_mat_noi_dung():
    text = "\n".join(f"dòng {i}" for i in range(2000))
    joined = "".join(chunk_message(text))
    for i in (0, 999, 1999):
        assert f"dòng {i}" in joined


def test_chia_tin_cat_o_ranh_gioi_dong():
    """Cắt giữa dòng sẽ làm hỏng thẻ HTML và Telegram từ chối cả tin."""
    text = "\n".join(f"<b>dòng {i}</b>" for i in range(1000))
    for p in chunk_message(text):
        assert p.count("<b>") == p.replace("<i>", "").count("</b>")


def test_bao_cao_that_cua_12_vi_the_luon_duoi_gioi_han_sau_khi_chia():
    """Đây là kích thước thật đã làm Telegram từ chối: 12 vị thế, 4482 ký tự."""
    pos = [_pos(symbol=f"SYMBOL{i}USDT", amt=1000.0 + i, mark=0.00081) for i in range(12)]
    txt = build_position_report(pos, 5827.0, 5249.0, 6033.0,
                                next_rebalance="16/09 03:46 UTC")
    for p in chunk_message(txt):
        assert len(p) <= TELEGRAM_LIMIT + 40
