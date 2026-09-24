"""
[FIX F48/F49] Công cụ giám sát phải nói ĐÚNG mức nghiêm trọng, và đo đủ để hành động.

F48 — CẮT CỤT BIẾN CẢNH BÁO THÀNH SỰ CỐ KHÁC HẲN. `readiness_gate.py` cắt thông điệp
lỗi ở 60 ký tự, biến:

    "... long $13,184 vs short $13,730."      (lệch 2%, sổ vẫn đủ hai chân)
 -> "... long $13,184 vs short $1"            (đọc ra: sổ MỘT CHIỀU hoàn toàn)

Đo thật 23/09/2026: sổ lúc đó long $13.061 / short $13.668 — hoàn toàn bình thường.
Một công cụ giám sát BỊA RA sự cố nặng hơn thực tế cũng nguy hiểm ngang việc nó bỏ
sót: cả hai đều dạy người vận hành ngừng tin nó. Bài học F41/F47 ở khâu HIỂN THỊ.

F49 — ĐẾM SỐ LẦN CHẶN KHÔNG ĐỦ ĐỂ QUYẾT ĐỊNH. Cùng lượt đó, trần đuổi giá chặn 358
lần so với 40 lần đuổi thành công. Nhưng chặn vì vượt trần 1,1 lần và chặn vì vượt 20
lần đòi hai hành động khác hẳn nhau, và bản ghi cũ không phân biệt được.
"""
import re
from dataclasses import fields

import pytest

from aegis.core.execution_log import ExecutionRecord
from aegis.pipelines.xs_live_pipeline import _pct


def test_readiness_gate_khong_duoc_cat_cut_thong_diep_loi():
    src = open("scripts/readiness_gate.py", encoding="utf-8").read()
    m = re.search(r"lỗi treo: \{([^}]*)\}", src)
    assert m, "không tìm thấy dòng in lỗi treo"
    assert "[:" not in m.group(1), (
        "thông điệp lỗi bị cắt cụt — cắt giữa một con số tiền biến cảnh báo thành "
        "một sự cố khác hẳn (xem docstring)")


def test_ban_ghi_do_duoc_MUC_vuot_tran_chu_khong_chi_so_lan():
    have = {f.name for f in fields(ExecutionRecord)}
    assert "requote_blocked_by_chase_cap" in have, "vẫn cần số lần chặn"
    for f in ("chase_block_p50_ratio", "chase_block_p90_ratio"):
        assert f in have, f"thiếu {f} — không có nó thì chỉnh trần là phỏng đoán"


def test_router_ghi_ty_le_vuot_tran():
    src = open("src/aegis/oms/order_router.py", encoding="utf-8").read()
    assert "chase_block_ratios" in src
    assert "drift_bps / max_chase_bps" in src, \
        "phải ghi TỶ LỆ drift/trần, vì đó mới là đại lượng chuyển thành hành động"


@pytest.mark.parametrize("vals,q,want", [
    (None, 50, 0.0),          # không có lần chặn nào -> 0, không phải lỗi
    ([], 90, 0.0),
    ([1.0, 2.0, 3.0, 10.0], 50, 2.5),
    ([2.0], 90, 2.0),
])
def test_pct_chiu_duoc_rong(vals, q, want):
    assert _pct(vals, q) == pytest.approx(want)
