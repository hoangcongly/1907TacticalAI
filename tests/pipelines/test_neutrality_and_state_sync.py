"""
Test hồi quy F24/F25 — hai bản vá ở tầng pipeline sau sự cố 10/09/2026.

F24: sổ nội bộ phải đồng bộ từ SÀN kể cả khi thực thi ném lỗi.
     Sự cố thật: lệnh đã khớp trên sàn nhưng bước lưu trạng thái không tới được,
     nên `rebalance_count` đứng yên ở 1 và sổ nội bộ "cũ hơn" sàn. Lượt sau hệ
     thống tự khoá vì lệch — đúng, nhưng đáng lẽ không được để lệch xảy ra.

F25: lệch hướng sau thực thi phải bị phát hiện.
     Sự cố thật: danh mục chạy 27 giờ ở mức long ròng +34.95% gross mà không có
     bất kỳ cảnh báo nào, vì không tồn tại phép kiểm nào trên đại lượng đó.
"""
import pytest

from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline, LiveConfig


def _pipe(**kw):
    """Pipeline rỗng, không chạm mạng — chỉ để gọi các hàm thuần."""
    p = CrossSectionalLivePipeline.__new__(CrossSectionalLivePipeline)
    p.config = LiveConfig(universe=["AAAUSDT"], **kw)
    p.dry_run = True
    return p


# =====================================================================
# F25 — cổng chặn lệch hướng
# =====================================================================
def test_f25_so_trung_lap_hoan_hao_thi_qua_cong():
    p = _pipe()
    r = p.check_neutrality({"A": 100.0, "B": -50.0}, {"A": 1.0, "B": 2.0})
    assert r["net"] == pytest.approx(0.0)
    assert r["net_ratio"] == pytest.approx(0.0)
    assert r["net_ok"] is True and r["ok"] is True


def test_f25_tai_hien_dung_so_lech_3495_phan_tram():
    """
    Dựng lại đúng danh mục đo được lúc 11/09: long $4.270 / short $2.058,
    net +$2.212 trên gross $6.328 = +34.95%.
    Cổng chặn PHẢI trượt. Đây là bài kiểm tra quan trọng nhất của cả bản vá.
    """
    p = _pipe()
    positions = {"LONG": 4270.0, "SHORT": -2058.0}
    prices = {"LONG": 1.0, "SHORT": 1.0}
    r = p.check_neutrality(positions, prices, tolerance=0.02)

    assert r["gross"] == pytest.approx(6328.0)
    assert r["net"] == pytest.approx(2212.0)
    assert r["net_ratio"] == pytest.approx(0.3495, abs=1e-3)
    assert r["ok"] is False, "sổ lệch 35% mà cổng chặn vẫn cho qua"
    assert r["long_notional"] == pytest.approx(4270.0)
    assert r["short_notional"] == pytest.approx(-2058.0)


@pytest.mark.parametrize("ratio,expected_ok", [
    (0.000, True), (0.019, True), (0.021, False), (0.100, False), (-0.350, False),
])
def test_f25_nguong_duoc_ton_trong_ca_hai_chieu(ratio, expected_ok):
    p = _pipe()
    # dựng long/short sao cho net/gross = ratio với gross = 100
    long_n = 100.0 * (1.0 + ratio) / 2.0
    short_n = -100.0 * (1.0 - ratio) / 2.0
    r = p.check_neutrality({"L": long_n, "S": short_n}, {"L": 1.0, "S": 1.0},
                           tolerance=0.02)
    assert r["net_ratio"] == pytest.approx(ratio, abs=1e-9)
    assert r["ok"] is expected_ok


def test_f25_so_rong_khong_gay_chia_cho_khong():
    p = _pipe()
    r = p.check_neutrality({}, {})
    assert r["net_ratio"] == 0.0 and r["ok"] is True


def test_f25_thieu_gia_thi_coi_notional_bang_khong():
    """Thiếu giá của một cặp không được âm thầm biến nó thành trung lập giả."""
    p = _pipe()
    r = p.check_neutrality({"A": 100.0, "B": -100.0}, {"A": 1.0})   # thiếu giá B
    assert r["net_ratio"] == pytest.approx(1.0), "phải lộ ra là lệch hoàn toàn"
    assert r["ok"] is False


def test_f25_nguong_lay_tu_cau_hinh():
    assert LiveConfig(universe=["X"]).max_net_exposure == 0.02
    assert LiveConfig(universe=["X"], max_net_exposure=0.10).max_net_exposure == 0.10


# =====================================================================
# F24 — đọc vị thế tới khi ỔN ĐỊNH
# =====================================================================
class _LaggyClient:
    """
    Mô phỏng tính nhất quán kiểu eventual của `/fapi/v2/positionRisk`:
    vài lần đọc đầu còn thiếu vị thế vừa khớp.
    """
    def __init__(self, settle_after=2):
        self.n = 0
        self.settle_after = settle_after

    def position_risk(self, symbol=None):
        self.n += 1
        if self.n <= self.settle_after:
            return [{"symbol": "AAAUSDT", "positionAmt": "10"}]
        return [{"symbol": "AAAUSDT", "positionAmt": "10"},
                {"symbol": "COTIUSDT", "positionAmt": "-21269"}]


def test_f24_doc_toi_khi_on_dinh_bat_duoc_vi_the_ve_muon():
    """
    Bản cũ `sleep(1.5)` rồi chụp MỘT lần -> bỏ sót vị thế về muộn, đúng cơ chế
    khiến COTIUSDT không vào sổ. Bản mới đọc tới khi hai lần liên tiếp trùng nhau.
    """
    p = _pipe()
    p.client = _LaggyClient(settle_after=2)
    pos = p._settle_positions(tries=6, delay=0.0)
    assert pos == {"AAAUSDT": 10.0, "COTIUSDT": -21269.0}


def test_f24_loc_vi_the_bang_khong():
    class _C:
        def position_risk(self, symbol=None):
            return [{"symbol": "A", "positionAmt": "5"},
                    {"symbol": "B", "positionAmt": "0"}]
    p = _pipe()
    p.client = _C()
    assert p._settle_positions(tries=3, delay=0.0) == {"A": 5.0}


def test_f24_doc_that_bai_hoan_toan_tra_ve_rong_chu_khong_nem_loi():
    """Không đọc được sàn thì trả về rỗng + log lỗi; KHÔNG được ném ra ngoài,
    vì hàm này chạy trong `finally` và ngoại lệ ở đó sẽ nuốt mất lỗi gốc."""
    class _Broken:
        def position_risk(self, symbol=None):
            raise RuntimeError("sàn không trả lời")
    p = _pipe()
    p.client = _Broken()
    assert p._settle_positions(tries=2, delay=0.0) == {}


# =====================================================================
# F26 — làm mới đúng khung dữ liệu mà engine thực sự đọc
# =====================================================================
def test_f26_v3_lam_moi_khung_nguon_khong_phai_khung_tong_hop():
    """
    Engine v3 đọc file 1h rồi tự tổng hợp lên 4h. Nếu làm mới khung 4h thì file 1h
    không bao giờ mới, và mọi lượt chạy đều dừng ở chốt STALE_DATA — hệ thống tự
    khoá vĩnh viễn mà không có gì thật sự hỏng.
    """
    p = _pipe(engine="v3", interval="4h", source_interval="1h")
    assert p.data_interval() == "1h"


def test_f26_v1_van_lam_moi_khung_cu():
    p = _pipe(engine="v1", interval="4h", source_interval="1h")
    assert p.data_interval() == "4h"


def test_f26_refresh_data_goi_dung_khung():
    """Kiểm chứng hành vi thật, không chỉ kiểm chứng hàm trả về chuỗi gì."""
    import aegis.pipelines.xs_live_pipeline as xlp

    called = []

    def fake_download(symbol, interval, days=30.0, client=None):
        called.append((symbol, interval))
        return "bars"

    p = _pipe(engine="v3", interval="4h", source_interval="1h")
    p.config.universe = ["AAAUSDT", "BBBUSDT"]
    p.data_client = None
    orig_dl, orig_save = xlp.download_klines, xlp.save_klines
    xlp.download_klines = fake_download
    xlp.save_klines = lambda bars, sym, interval: None
    try:
        n = p.refresh_data(days=5.0)
    finally:
        xlp.download_klines, xlp.save_klines = orig_dl, orig_save

    assert n == 2
    assert {i for _, i in called} == {"1h"}, f"đã tải nhầm khung: {called}"


# =====================================================================
# F29 — cổng chặn phải kiểm CẢ đòn bẩy gộp, không chỉ độ lệch hướng
# =====================================================================
def test_f29_so_nhan_doi_van_trung_lap_nhung_phai_bi_bat():
    """
    Tái hiện sự cố 11/09: lỗi báo giá lại nhân đôi mọi vị thế. Sổ vẫn TRUNG LẬP
    hoàn hảo (nhân đôi cả hai chân) nhưng đòn bẩy 4x thay vì 2x. Cổng chặn chỉ
    kiểm net sẽ cho qua — đó chính là điều đã xảy ra.
    """
    p = _pipe()
    eq = 5000.0
    # nhân đôi cả hai chân: net vẫn 0, gross gấp đôi
    r = p.check_neutrality({"L": 10000.0, "S": -10000.0}, {"L": 1.0, "S": 1.0},
                           tolerance=0.02, equity=eq, target_leverage=2.0)
    assert r["net_ok"] is True, "sổ nhân đôi đối xứng vẫn trung lập"
    assert r["leverage"] == pytest.approx(4.0)
    assert r["gross_ok"] is False, "đòn bẩy 4x so với mục tiêu 2x phải bị bắt"
    assert r["ok"] is False


def test_f29_don_bay_dung_thi_qua_cong():
    p = _pipe()
    r = p.check_neutrality({"L": 5000.0, "S": -5000.0}, {"L": 1.0, "S": 1.0},
                           tolerance=0.02, equity=5000.0, target_leverage=2.0)
    assert r["leverage"] == pytest.approx(2.0)
    assert r["gross_ok"] is True and r["ok"] is True


def test_f29_sai_lech_nho_duoc_chap_nhan():
    """Trượt giá và làm tròn khiến gross không bao giờ khớp tuyệt đối."""
    p = _pipe()
    r = p.check_neutrality({"L": 5250.0, "S": -5250.0}, {"L": 1.0, "S": 1.0},
                           tolerance=0.02, equity=5000.0, target_leverage=2.0,
                           gross_tolerance=0.15)
    assert r["leverage"] == pytest.approx(2.1)
    assert r["gross_ok"] is True


def test_f29_khong_co_equity_thi_bo_qua_kiem_don_bay():
    """Thiếu thông tin thì KHÔNG được kết luận bừa là đạt hay không đạt."""
    p = _pipe()
    r = p.check_neutrality({"L": 100.0, "S": -100.0}, {"L": 1.0, "S": 1.0})
    assert r["leverage"] is None and r["gross_ok"] is True


# =====================================================================
# F28 — khoá chống chạy chồng
# =====================================================================
def test_f28_khoa_chan_tien_trinh_thu_hai(tmp_path):
    from aegis.core.process_lock import ProcessLockBusy, acquire_lock
    lock = str(tmp_path / "test.lock")
    h = acquire_lock(lock)
    try:
        with pytest.raises(ProcessLockBusy):
            acquire_lock(lock)
    finally:
        h.close()


def test_f28_nha_khoa_thi_chiem_lai_duoc(tmp_path):
    from aegis.core.process_lock import acquire_lock, exclusive_lock
    lock = str(tmp_path / "test.lock")
    with exclusive_lock(lock):
        pass
    h = acquire_lock(lock)     # phải chiếm lại được sau khi nhả
    h.close()


def test_f28_ghi_pid_vao_file_khoa(tmp_path):
    import os
    from aegis.core.process_lock import acquire_lock
    lock = tmp_path / "test.lock"
    h = acquire_lock(str(lock))
    try:
        assert lock.read_text().strip() == str(os.getpid())
    finally:
        h.close()


# =====================================================================
# F30 — chốt nhịp tái cân bằng phải có ở MỌI đường chạy, không chỉ daemon
# =====================================================================
class _MiniPipe:
    """Pipeline tối giản để kiểm riêng logic 'đã đến hạn chưa'."""

    def __init__(self, rebalance_hours=72.0):
        from aegis.pipelines.xs_live_pipeline import LiveConfig
        self.config = LiveConfig(universe=["A"], rebalance_hours=rebalance_hours)

    def due(self, hours_since, rebalance_count, force=False):
        return (force or rebalance_count == 0
                or hours_since >= self.config.rebalance_hours)


@pytest.mark.parametrize("hours,count,force,expected", [
    (0.0,   0, False, True),    # lần chạy đầu -> luôn cân
    (2.7,   1, False, False),   # cron 07:00 ngay sau khi mở sổ -> KHÔNG được cân
    (24.0,  1, False, False),   # cron ngày hôm sau -> vẫn chưa tới hạn 72h
    (48.0,  1, False, False),
    (71.9,  1, False, False),
    (72.0,  1, False, True),    # đúng hạn
    (99.0,  1, False, True),
    (2.7,   1, True,  True),    # can thiệp thủ công --force
])
def test_f30_chot_nhip_dung_han(hours, count, force, expected):
    """
    `crontab` chạy 07:00 hằng ngày. Không có chốt nhịp thì hệ thống tái cân bằng
    mỗi 24h trong khi chiến lược được kiểm định ở 72h — tín hiệu tính trên lưới 18
    nến còn vị thế bị đặt lại mỗi ngày, tức đang chạy một cấu hình CHƯA TỪNG được
    kiểm định, với chi phí giao dịch gấp ~3 lần.
    """
    assert _MiniPipe().due(hours, count, force) is expected


def test_f30_chu_ky_lay_tu_cau_hinh_v3():
    from aegis.pipelines.xs_live_pipeline import LiveConfig
    cfg = LiveConfig.from_artifacts("artifacts/strategy_v3.json")
    assert cfg.rebalance_hours == pytest.approx(72.0), (
        "chu kỳ live phải khớp cấu hình đã kiểm định (18 nến x 4h)")


def test_f30_run_once_nhan_co_force():
    import inspect
    from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline
    sig = inspect.signature(CrossSectionalLivePipeline.run_once)
    assert "force" in sig.parameters
    assert sig.parameters["force"].default is False, "mặc định PHẢI là không ép"
