"""
[FIX F37] Bộ lọc universe phải đo lịch sử THEO ĐÚNG ĐƯỜNG mà hệ thống sẽ dùng.

Đây là họ lỗi F3 dưới một lớp áo khác: nghiên cứu và live đo CÙNG MỘT đại lượng —
"cặp này có đủ lịch sử không" — bằng HAI đường khác nhau, rồi lệch nhau trong im lặng.

  * `data/panel_v2.load_panel_v2` tổng hợp khung 4h TỪ file 1h (và chỉ rơi về file 4h
    khi không có nguồn 1h).
  * `data/universe.history_lengths` (bản cũ) đếm dòng trong file 4h.

Hệ quả đo được ngày 14/09/2026: 89/170 cặp bị loại OAN. CRVUSDT có 52.869 nến 1h
(= 13.217 nến 4h) nhưng `CRVUSDT_4h.parquet` chỉ còn 186 dòng sót từ một lần tải cũ.
Live vì thế chạy 62 cặp trong khi nghiên cứu kiểm định trên 127 — và chênh lệch đó
đáng **0,36 Sharpe** (1,20 so với 1,56 trên cùng dữ liệu).

Bài học chung: tải lại file 4h KHÔNG phải cách sửa — file sẽ lại cũ đi và lỗi quay
lại. Cách sửa là đo đúng thứ đường chạy thật sẽ dùng.
"""
import numpy as np
import pandas as pd
import pytest

from aegis.data.universe import history_lengths


@pytest.fixture
def data_root(tmp_path):
    """
    Một cặp có 1h dài và 4h NGẮN (đúng hình dạng gây ra lỗi), một cặp chỉ có 4h.
    """
    def _write(name, n):
        df = pd.DataFrame({
            "timestamp_ms": np.arange(n, dtype=np.int64) * 3_600_000,
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        })
        df.to_parquet(tmp_path / name)

    _write("LONGUSDT_1h.parquet", 40_000)     # = 10.000 nến 4h
    _write("LONGUSDT_4h.parquet", 186)        # file cũ sót lại, ngắn
    _write("ONLY4HUSDT_4h.parquet", 5_000)    # không có nguồn 1h
    _write("SHORTUSDT_1h.parquet", 400)       # = 100 nến 4h, thật sự ngắn
    return str(tmp_path)


def test_dung_nguon_1h_de_dem_nen_4h(data_root):
    L = history_lengths(["LONGUSDT"], "4h", data_root)
    assert L["LONGUSDT"] == 10_000, \
        "phải đếm 40.000 nến 1h / 4 = 10.000, không phải 186 dòng của file 4h cũ"


def test_hanh_vi_cu_van_tai_tao_duoc(data_root):
    """`source_interval=None` giữ nguyên cách đếm cũ — để so sánh và để gỡ lỗi."""
    L = history_lengths(["LONGUSDT"], "4h", data_root, source_interval=None)
    assert L["LONGUSDT"] == 186


def test_roi_ve_file_dung_khung_khi_khong_co_nguon(data_root):
    L = history_lengths(["ONLY4HUSDT"], "4h", data_root)
    assert L["ONLY4HUSDT"] == 5_000


def test_cap_that_su_ngan_van_bi_loai(data_root):
    """Bản vá không được biến thành 'chấp nhận tất' — cặp thiếu thật vẫn phải thiếu."""
    L = history_lengths(["SHORTUSDT"], "4h", data_root)
    assert L["SHORTUSDT"] == 100


def test_cap_khong_co_du_lieu_thi_khong_xuat_hien(data_root):
    L = history_lengths(["KHONGCOUSDT"], "4h", data_root)
    assert "KHONGCOUSDT" not in L


def test_cung_khung_thi_khong_chia(data_root):
    """interval == source_interval: không được chia cho tỷ lệ nào cả."""
    L = history_lengths(["LONGUSDT"], "1h", data_root, source_interval="1h")
    assert L["LONGUSDT"] == 40_000


def test_nguon_tho_hon_dich_thi_bo_qua_nguon(data_root):
    """Nguồn 4h không dựng được nến 1h — phải rơi về file đúng khung, không chia 0."""
    L = history_lengths(["ONLY4HUSDT"], "1h", data_root, source_interval="4h")
    assert L.get("ONLY4HUSDT", 0) == 0      # không có file 1h -> 0, KHÔNG được crash
