"""
[FIX F47] Cảnh báo phải phân biệt "máy ngủ" với "daemon chết".

VÌ SAO: `com.aegis.trading.plist` ghi rõ máy này là "laptop gập nắp, ngủ gần như
liên tục". `time.sleep()` bị treo theo giấc ngủ máy, nên một vòng lặp 30 phút giãn
thành nhiều giờ đồng hồ tường. ĐÃ ĐO ngày 22/09/2026: nhịp tim đứng 5,6 giờ trong
khi `pmset -g log` cho thấy máy ngủ 4,3/6 giờ gần nhất, và daemon hoàn toàn khoẻ.

Healthcheck cũ báo ĐỎ y hệt nhau cho hai tình huống khác hẳn. Một cảnh báo không
phân biệt được chúng sẽ bị người vận hành học cách bỏ qua — rồi khi nó kêu THẬT thì
không ai nghe. Đó là bài học F41 lặp lại ở tầng cảnh báo thay vì tầng tiến trình.
"""
import sys
import time
import types

import pytest

sys.path.insert(0, "scripts")
import healthcheck as H  # noqa: E402


def _fake_pmset(monkeypatch, body: str):
    """Thay `pmset -g log` bằng nội dung dựng sẵn."""
    def run(cmd, **kw):
        return types.SimpleNamespace(stdout=body)
    monkeypatch.setattr(H.subprocess if hasattr(H, "subprocess") else __import__("subprocess"),
                        "run", run, raising=False)
    import subprocess
    monkeypatch.setattr(subprocess, "run", run)


def _stamp(offset_s: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + offset_s))


def test_may_ngu_duoc_tru_ra_khoi_tuoi_nhip_tim(monkeypatch):
    """Ngủ 2 giờ trong 3 giờ qua -> `_slept_since` phải trả về ~2 giờ."""
    body = "\n".join([
        f"{_stamp(-10800)} +0700 Sleep               Entering Sleep state",
        f"{_stamp(-3600)} +0700 Wake from Normal Sleep [CDNVA]",
    ])
    _fake_pmset(monkeypatch, body)
    assert H._slept_since(time.time() - 4 * 3600) == pytest.approx(2.0, abs=0.05)


def test_khong_ngu_thi_khong_tru_gi(monkeypatch):
    """Log không có giấc ngủ nào -> 0 giờ, nhịp tim đứng vẫn là ĐỎ."""
    _fake_pmset(monkeypatch, f"{_stamp(-3600)} +0700 Assertions  PID 1 Summary")
    assert H._slept_since(time.time() - 4 * 3600) == 0.0


def test_doc_pmset_that_bai_thi_nga_ve_phia_than_trong(monkeypatch):
    """
    Không đọc được `pmset` phải trả 0.0, KHÔNG phải một con số đoán.

    0.0 nghĩa là "coi như máy không ngủ", tức nhịp tim đứng vẫn bị báo ĐỎ. Thiếu dữ
    liệu mà ngả về phía im lặng là tự tay tắt cảnh báo — đúng thứ F41 đã trả giá.
    """
    import subprocess

    def boom(cmd, **kw):
        raise OSError("pmset không tồn tại")
    monkeypatch.setattr(subprocess, "run", boom)
    assert H._slept_since(time.time() - 4 * 3600) == 0.0


def test_giac_ngu_chua_ket_thuc_van_duoc_tinh(monkeypatch):
    """Máy ngủ rồi thức dậy để chạy healthcheck: giấc ngủ cuối chưa có dòng Wake."""
    _fake_pmset(monkeypatch, f"{_stamp(-7200)} +0700 Sleep   Entering Sleep state")
    assert H._slept_since(time.time() - 3 * 3600) == pytest.approx(2.0, abs=0.05)
