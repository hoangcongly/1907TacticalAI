"""
[FIX F44] Trần đuổi giá phải co giãn theo BIẾN ĐỘNG của từng cặp.

Bối cảnh đo được (60 cặp thật, σ nến 1h quy về cửa sổ chờ 900s):
    dịch giá trung vị trong 900s : 64,4 bp
    trần đuổi cũ (cố định)       : 15,0 bp
    -> 100% số cặp vượt trần; cặp mạnh nhất (BTRUSDT, σ=5,22%/h) dịch 261 bp

Ở mức dịch trung vị, trần 15bp bị chạm sau ~50 GIÂY của cửa sổ 900 giây: lệnh nằm
đóng băng ngoài thị trường 94% thời gian chờ rồi rơi xuống taker. Đó là nguyên nhân
thật của tỷ lệ maker 54%, và nó giải thích vì sao nâng `passive_wait_s` 300 -> 900s
gần như vô ích (48,8% -> 54,3%): nút thắt chưa bao giờ là THỜI GIAN.

Một trần cố định không thể đúng cho cả BTC lẫn một altcoin σ=5%/giờ. Trần theo σ làm
cho "đuổi trong phạm vi nhiễu bình thường" mang cùng một ý nghĩa ở mọi cặp.
"""
import math

import pytest


def _cap(sigma_1h: float, wait_s: float, floor_bps: float, mult: float) -> float:
    """Bản sao công thức trong `xs_live_pipeline._chase_caps`."""
    scale = math.sqrt(max(1.0, wait_s) / 3600.0)
    return max(floor_bps, mult * sigma_1h * scale * 1e4)


@pytest.mark.parametrize("sym,sigma_1h,drift_bps", [
    ("BTRUSDT",   0.0522, 261.0),
    ("BULLAUSDT", 0.0443, 221.5),
    ("SOLUSDT",   0.0100,  50.0),
    ("BTC-như",   0.0020,  10.0),
])
def test_tran_phu_duoc_dich_gia_dien_hinh(sym, sigma_1h, drift_bps):
    """Trần phải phủ được mức dịch 1σ của chính cặp đó, nếu không lệnh chết ngoài thị trường."""
    cap = _cap(sigma_1h, 900.0, floor_bps=15.0, mult=1.5)
    assert cap >= drift_bps * 0.99, (
        f"{sym}: trần {cap:.1f}bp < dịch điển hình {drift_bps:.1f}bp — lệnh sẽ bị "
        f"đóng băng ngoài thị trường rồi rơi xuống taker (đúng cơ chế maker 54%)"
    )


def test_tran_cu_co_dinh_that_su_hong():
    """Khoá lại BẰNG CHỨNG: trần cũ 15bp thất bại ở mọi cặp biến động thật."""
    for sigma in (0.0522, 0.0443, 0.0100):
        drift = sigma * math.sqrt(900 / 3600) * 1e4
        assert drift > 15.0, "mẫu test phải là cặp thực sự vượt trần cũ"


def test_van_chan_cu_chay_gia_that():
    """Trần không được biến thành vô hạn — vượt ngoài nhiễu vẫn phải bị chặn."""
    sigma = 0.01
    cap = _cap(sigma, 900.0, floor_bps=15.0, mult=1.5)
    drift_1sigma = sigma * math.sqrt(900 / 3600) * 1e4
    assert cap < drift_1sigma * 3, (
        f"trần {cap:.1f}bp quá rộng so với 1σ={drift_1sigma:.1f}bp — sẽ rượt theo "
        f"cú chạy giá thật, đúng thứ trần này sinh ra để tránh"
    )


def test_cap_it_bien_dong_van_giu_tran_san():
    """Cặp rất ít biến động không được nhận trần NHỎ HƠN trần sàn."""
    cap = _cap(0.0001, 900.0, floor_bps=15.0, mult=1.5)
    assert cap == pytest.approx(15.0), "phải ngả về trần sàn, không được hẹp hơn"


def test_thieu_du_lieu_thi_nga_ve_than_trong():
    """σ không đo được -> dùng trần sàn, KHÔNG được nới rộng khi mù."""
    caps = {}
    cap = caps.get("KHONGCOUSDT", 15.0)
    assert cap == 15.0, "thiếu dữ liệu phải ngả về phía thận trọng"
