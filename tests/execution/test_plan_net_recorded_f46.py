"""
[FIX F46] Bản ghi phải lưu net của KẾ HOẠCH cạnh net THỰC TẾ.

SỰ CỐ: lượt 19/09/2026 ra `net_exposure = +4,00%` (trần 2%) trong khi `n_orders=50`,
`unfilled_count=0`, `plan_failures=0` — tức khớp đủ mọi lệnh. Với bản ghi cũ, KHÔNG
có cách nào phân biệt hai giả thuyết:

    (a) kế hoạch vốn đã lệch  -> lỗi nằm ở `build_rebalance_plan` / `_repair_neutrality`
    (b) kế hoạch cân, thực thi làm lệch -> lỗi nằm ở `order_router`

Hai nguyên nhân đó sửa ở hai file khác nhau. Không đo được thì không sửa được, và
đó là lý do lỗi này còn sống sau cả F25 lẫn F42.

Đây cũng đúng bài học F42: mọi bất biến phải đo được ở tầng CÒN SỬA ĐƯỢC MIỄN PHÍ,
không chỉ sau khi lệnh đã nằm trên sàn.
"""
from dataclasses import fields

from aegis.core.execution_log import ExecutionRecord


def test_ban_ghi_co_ca_hai_so_do():
    have = {f.name for f in fields(ExecutionRecord)}
    assert "plan_net_exposure" in have, "thiếu net của KẾ HOẠCH"
    assert "net_exposure" in have, "thiếu net THỰC TẾ sau thực thi"


def test_pipeline_thuc_su_ghi_net_ke_hoach():
    """Có trường mà không ai ghi vào thì vẫn là mù — khoá luôn phần nối dây."""
    import inspect

    from aegis.pipelines.xs_live_pipeline import CrossSectionalLivePipeline

    src = inspect.getsource(CrossSectionalLivePipeline)
    assert "plan_net_exposure=" in src
    assert "plan.net_notional" in src and "plan.gross_notional" in src
