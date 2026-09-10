import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import main as aegis_main
from main import main


def test_main_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "mode" in capsys.readouterr().out


def test_research_and_cpcv_modes_run_on_sample_data():
    """
    research/cpcv được phép dùng dữ liệu mẫu (chúng chỉ nghiên cứu, không đặt lệnh).
    Chỉ live/paper mới bị chặn cứng.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        assert main(["--mode", "research", "--artifacts-dir", temp_dir, "--bars-count", "100"]) == 0
        assert os.path.exists(os.path.join(temp_dir, "model.pkl"))
        assert os.path.exists(os.path.join(temp_dir, "metadata.json"))

        assert main(["--mode", "cpcv", "--bars-count", "80"]) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_paper_mode_refuses_to_run_without_artifacts():
    """
    [FIX F2] Thiếu artifacts thì phải DỪNG, tuyệt đối không tự huấn luyện.

    Bản cũ sinh random walk (`np.random.seed(42)`), fit model production lên chính
    nhiễu đó rồi đem đi giao dịch — đây là lỗ hổng nguy hiểm nhất của main.py.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(SystemExit) as exc_info:
            main(["--mode", "paper", "--artifacts-dir", os.path.join(temp_dir, "trong_rong")])
        assert "TỪ CHỐI CHẠY" in str(exc_info.value)
        assert not os.path.exists(os.path.join(temp_dir, "trong_rong", "model.pkl")), (
            "Không được sinh model khi bị từ chối chạy"
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_paper_mode_refuses_synthetic_market_data():
    """[FIX F2] Có artifacts nhưng KHÔNG có dữ liệu thật -> vẫn phải từ chối."""
    temp_dir = tempfile.mkdtemp()
    try:
        assert main(["--mode", "research", "--artifacts-dir", temp_dir, "--bars-count", "100"]) == 0

        with pytest.raises(SystemExit) as exc_info:
            main(["--mode", "paper", "--artifacts-dir", temp_dir, "--bars-count", "20"])
        assert "TỪ CHỐI CHẠY" in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_loader_allows_synthetic_only_when_explicitly_permitted():
    """Cổng chặn nằm ở tham số `allow_synthetic`, mặc định research vẫn dùng được."""
    df = aegis_main.load_data_or_synthetic(None, n_bars=50, allow_synthetic=True)
    assert len(df) == 50

    with pytest.raises(SystemExit) as exc_info:
        aegis_main.load_data_or_synthetic(None, n_bars=50, allow_synthetic=False)
    assert "TỪ CHỐI CHẠY" in str(exc_info.value)
