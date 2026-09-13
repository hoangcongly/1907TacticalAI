"""
[FIX F28] Khoá loại trừ cho mọi tiến trình chạm tiền.

VÌ SAO CẦN, từ sự cố thật 11/09/2026: `crontab` của máy vận hành có ĐÚNG một dòng
bị lặp hai lần, nên mỗi 07:00 có hai tiến trình cùng chạy `run_daily.py --live`.
Hai lượt tái cân bằng chồng nhau đặt lệnh hai lần cho cùng một mục tiêu.

Vì sao id lệnh tất định KHÔNG cứu được: `newClientOrderId` chống trùng bằng
`epoch_bucket = int(time.time() // 60)` — nó chỉ khử được lệnh lặp TRONG CÙNG MỘT
PHÚT. Hai tiến trình lệch nhau vài giây có thể rơi vào hai phút khác nhau, và khi
đó chúng sinh ra hai id khác nhau cho cùng một ý định. Chống trùng ở tầng id là
chống LẶP LỆNH, không phải chống CHẠY CHỒNG. Đó là hai bài toán khác nhau.

Khoá theo file dùng `flock`: hệ điều hành tự nhả khi tiến trình chết, nên không bao
giờ để lại khoá mồ côi sau một lần crash — khác hẳn khoá tự cài bằng file cờ.
"""

import errno
import fcntl
import logging
import os
import pathlib
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LOCK = "artifacts/.aegis_trading.lock"

__all__ = ["ProcessLockBusy", "acquire_lock", "exclusive_lock"]


class ProcessLockBusy(RuntimeError):
    """Một tiến trình khác đang giữ khoá."""


def acquire_lock(path: str = DEFAULT_LOCK):
    """
    Chiếm khoá độc quyền, KHÔNG chờ. Trả về đối tượng file phải giữ nguyên tham
    chiếu suốt thời gian cần khoá — đóng file là nhả khoá.

    Ném `ProcessLockBusy` nếu tiến trình khác đang giữ.
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    handle = open(p, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.seek(0)
        holder = handle.read().strip() or "không rõ"
        handle.close()
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise ProcessLockBusy(
                f"Tiến trình khác đang giữ khoá {path} (pid {holder})") from exc
        raise

    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


@contextmanager
def exclusive_lock(path: str = DEFAULT_LOCK, required: bool = True):
    """
    Dùng theo kiểu `with`:

        with exclusive_lock():
            ...phần chạm tiền...

    `required=False` cho phép chạy tiếp khi không chiếm được khoá — CHỈ dùng cho
    thao tác chỉ đọc (xem trạng thái, xuất báo cáo). Mọi thao tác đặt lệnh phải
    để `required=True`.
    """
    try:
        handle = acquire_lock(path)
    except ProcessLockBusy:
        if required:
            raise
        logger.warning("Không chiếm được khoá %s — chạy ở chế độ chỉ đọc", path)
        yield None
        return
    try:
        yield handle
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        except Exception:
            logger.warning("Không nhả được khoá %s", path)
