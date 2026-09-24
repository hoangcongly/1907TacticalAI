#!/usr/bin/env python3
"""
[FIX F41] MỘT LỆNH NÓI THẬT hệ thống còn sống hay đã chết.

    python scripts/healthcheck.py

Vì sao cần script này: `launchctl list` chỉ cho biết tiến trình CÒN TỒN TẠI, không
cho biết nó CÒN LÀM VIỆC. Sự cố 16-18/09/2026: `.venv` bị xoá, mọi kết nối TLS chết,
daemon ném lỗi 58 chu trình liên tiếp — nhưng `launchctl list` vẫn hiện PID bình
thường suốt 2 ngày 3 giờ. Nhìn vào đó mà yên tâm là nhìn nhầm.

Sự thật nằm ở NHỊP TIM GHI RA ĐĨA (`logs/daemon_health.json`), vì đĩa vẫn sống khi
mạng đã chết. Nhịp tim cũ hơn 2 chu kỳ = daemon không còn làm việc, bất kể PID.

Mã thoát: 0 = XANH, 1 = ĐỎ (tiện cho cron/alert sau này).
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEALTH_FILE = ROOT / "logs" / "daemon_health.json"
DOWN_FLAG = ROOT / "logs" / "DAEMON_DOWN.txt"
VENV_PY = ROOT / ".venv" / "bin" / "python"
LOOP_INTERVAL_MINS = 30.0
STALE_AFTER_S = LOOP_INTERVAL_MINS * 60 * 2 + 300  # lỡ 2 chu kỳ + 5 phút dư


def _slept_since(epoch_s: float) -> float:
    """
    Số giờ máy đã NGỦ kể từ `epoch_s`, đọc từ `pmset -g log`.

    [FIX F47] VÌ SAO CẦN: plist ghi rõ "laptop gập nắp, ngủ gần như liên tục", và
    `time.sleep()` bị treo theo giấc ngủ máy — 30 phút ngủ thành 5,6 giờ đồng hồ
    tường. Nhịp tim đứng vì thế là chuyện BÌNH THƯỜNG trên máy này.

    Nhưng healthcheck cũ báo ĐỎ y hệt nhau cho hai chuyện khác hẳn: "máy vừa ngủ
    dậy" và "daemon đã chết". Một cảnh báo không phân biệt được hai thứ đó thì
    người vận hành sẽ học cách bỏ qua nó — và khi nó kêu THẬT thì không ai nghe.
    Đó đúng là bài học F41, lặp lại ở tầng cảnh báo thay vì tầng tiến trình.

    Trả về 0.0 nếu không đọc được — thiếu dữ liệu phải ngả về phía THẬN TRỌNG,
    tức coi như máy KHÔNG ngủ và nhịp tim đứng là đáng lo.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(["pmset", "-g", "log"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return 0.0

    total, sleep_at = 0.0, None
    for line in out.splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if not m:
            continue
        try:
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        except Exception:
            continue
        if ts < epoch_s:
            continue
        if "Entering Sleep" in line or "Sleep  " in line:
            sleep_at = ts
        elif ("Wake from" in line or "DarkWake" in line) and sleep_at is not None:
            total += max(0.0, ts - sleep_at)
            sleep_at = None
    if sleep_at is not None:
        total += max(0.0, time.time() - sleep_at)
    return total / 3600.0


def _launchctl_status(label: str):
    """Trả về (pid, mã thoát lần cuối) hoặc None nếu job chưa nạp."""
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == label:
            return parts[0], parts[1]
    return None


def main() -> int:
    problems, warnings = [], []
    print("=" * 66)
    print("  AEGIS HEALTHCHECK")
    print("=" * 66)

    # --- 1. Môi trường: chính thứ đã chết ngày 16/09 ---
    if not VENV_PY.exists():
        problems.append(f"THIẾU .venv — {VENV_PY} không tồn tại. Dựng lại:\n"
                        f"    uv venv --python /opt/homebrew/bin/python3.12 .venv\n"
                        f"    uv pip install -e . requests joblib")
        print("  .venv            : ❌ KHÔNG CÓ")
    else:
        try:
            r = subprocess.run(
                [str(VENV_PY), "-c",
                 "import certifi,os,sys;sys.exit(0 if os.path.exists(certifi.where()) else 3)"],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0:
                print("  .venv + certifi  : ✅ OK")
            else:
                problems.append("certifi/cacert.pem hỏng — mọi kết nối TLS sẽ chết "
                                "(đúng sự cố 16/09). Chạy: uv pip install --force-reinstall certifi")
                print("  .venv + certifi  : ❌ HỎNG")
        except Exception as exc:
            problems.append(f"Không chạy nổi python trong .venv: {exc}")
            print("  .venv + certifi  : ❌ HỎNG")

    # --- 2. launchd: tồn tại, KHÔNG đồng nghĩa đang làm việc ---
    for label in ("com.aegis.trading", "com.aegis.telegram"):
        st = _launchctl_status(label)
        if st is None:
            problems.append(f"Job {label} CHƯA ĐƯỢC NẠP vào launchd.")
            print(f"  {label:<17}: ❌ chưa nạp")
        else:
            pid, rc = st
            if pid == "-":
                problems.append(f"Job {label} không chạy (mã thoát lần cuối {rc}).")
                print(f"  {label:<17}: ❌ không chạy (exit {rc})")
            else:
                print(f"  {label:<17}: ✅ PID {pid}  (⚠️ mới chỉ là CÒN SỐNG)")

    # --- 3. Nhịp tim — đây mới là sự thật ---
    if not HEALTH_FILE.exists():
        warnings.append("Chưa có logs/daemon_health.json — daemon chưa chạy lần nào "
                        "kể từ bản vá F41. Bình thường nếu vừa khởi động lại.")
        print("  nhịp tim         : ⚠️ chưa có")
    else:
        try:
            h = json.loads(HEALTH_FILE.read_text())
            age = time.time() - float(h.get("ts", 0))
            status = h.get("status", "?")
            print(f"  nhịp tim         : {h.get('ts_human','?')} "
                  f"({age/60:.0f} phút trước) — {status}")
            if age > STALE_AFTER_S:
                # [FIX F47] Trừ phần máy ngủ trước khi kết luận daemon chết.
                slept_h = _slept_since(float(h.get("ts", 0)))
                awake_h = age / 3600.0 - slept_h
                if awake_h * 3600.0 > STALE_AFTER_S:
                    problems.append(
                        f"NHỊP TIM ĐỨNG {age/3600:.1f} GIỜ, trong đó {awake_h:.1f}h máy "
                        f"THỨC (ngưỡng {STALE_AFTER_S/3600:.1f}h). Daemon KHÔNG còn làm "
                        f"việc dù có thể vẫn còn PID.")
                else:
                    warnings.append(
                        f"Nhịp tim {age/3600:.1f}h trước, nhưng máy đã NGỦ {slept_h:.1f}h "
                        f"trong khoảng đó — chỉ {awake_h:.1f}h thức, dưới ngưỡng "
                        f"{STALE_AFTER_S/3600:.1f}h. Daemon còn sống, lượt sẽ chạy bù khi "
                        f"máy thức (launchd + kiểm 'đã đủ 72h chưa').")
            if status in ("ERROR", "DEAD"):
                problems.append(f"Trạng thái nhịp tim = {status}: {h.get('detail','')}")
        except Exception as exc:
            problems.append(f"Không đọc được nhịp tim: {exc}")

    if DOWN_FLAG.exists():
        problems.append(f"CÓ CỜ BÁO CHẾT:\n    {DOWN_FLAG.read_text().strip()}")

    # --- Kết luận ---
    print("=" * 66)
    if problems:
        print(f"  🔴 ĐỎ — {len(problems)} vấn đề:\n")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")
        print("\n  Khởi động lại sau khi sửa:")
        print("   for j in trading telegram; do")
        print("     launchctl unload ~/Library/LaunchAgents/com.aegis.$j.plist")
        print("     launchctl load   ~/Library/LaunchAgents/com.aegis.$j.plist")
        print("   done")
        print("=" * 66)
        return 1

    for w in warnings:
        print(f"  ⚠️ {w}")
    print("  🟢 XANH — daemon đang thực sự làm việc.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
