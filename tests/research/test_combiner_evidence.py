"""
Tầng gộp: tách bạch "chưa đủ dữ liệu" khỏi "dữ liệu nói không có gì".

Hai trạng thái này từng dùng chung một nhánh xử lý. Chúng khác nhau về bản chất:
vô tri khác với một kết luận. Trả lời cả hai bằng "giao dịch đều tay ở gross đầy đủ"
nghĩa là ở trạng thái thứ hai, hệ thống đặt cược hết cỡ đúng vào lúc nó vừa tự kết
luận rằng không tín hiệu nào chứng minh được điều gì.

Đo được trên dữ liệu thật: nhánh sụp đổ chạy 0/799 kỳ với 26 tín hiệu. Nên đây là rủi
ro TIỀM ẨN. Test dưới đây giữ nó ở trạng thái tiềm ẩn — và ồn ào nếu nó thành hiện thực.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from aegis.research.adaptive_combiner import CombinerSpec, adaptive_weights


def _noise(n=400, k=3, seed=0):
    """Lợi suất nhân tố THUẦN NHIỄU — không họ nào có thể vượt ngưỡng t."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64) * 14_400_000
    return {f"f{j}": pd.Series(rng.normal(0.0, 0.01, n), index=idx) for j in range(k)}


def test_chan_doan_phan_loai_dung_ba_nhanh():
    fac = _noise()
    spec = CombinerSpec(lookback=200, min_periods=100, t_threshold=3.5)
    _, diag = adaptive_weights(fac, spec, return_diagnostics=True)
    assert set(diag["branch"].unique()) <= {"warmup", "collapse", "ok"}
    assert (diag["branch"].iloc[:100] == "warmup").all(), "100 kỳ đầu phải là warm-up"


def test_nguong_t_cao_tren_nhieu_thuan_thi_sup_do():
    """Ngưỡng t = 6 trên nhiễu thuần: không gì vượt nổi => phải rơi vào nhánh sụp đổ."""
    fac = _noise()
    spec = CombinerSpec(lookback=200, min_periods=100, t_threshold=6.0)
    _, diag = adaptive_weights(fac, spec, return_diagnostics=True)
    assert (diag["branch"] == "collapse").sum() > 0


def test_sup_do_phai_ghi_canh_bao(caplog):
    """
    Im lặng là chế độ hỏng tệ nhất. Nếu hệ thống rơi vào trạng thái "không có bằng
    chứng nào" mà vẫn giao dịch ở gross đầy đủ, người vận hành PHẢI biết.
    """
    fac = _noise()
    spec = CombinerSpec(lookback=200, min_periods=100, t_threshold=6.0)
    with caplog.at_level(logging.WARNING, logger="aegis.research.adaptive_combiner"):
        adaptive_weights(fac, spec)
    assert any("BẰNG CHỨNG SỤP ĐỔ" in r.message for r in caplog.records)


def test_collapse_equal_tat_thi_danh_muc_rong():
    """`collapse_equal=False` = "không có bằng chứng thì không đặt cược"."""
    fac = _noise()
    spec = CombinerSpec(lookback=200, min_periods=100, t_threshold=6.0,
                        collapse_equal=False)
    W, diag = adaptive_weights(fac, spec, return_diagnostics=True)
    late = diag.index[diag["branch"] == "collapse"]
    assert len(late) > 0
    # [FIX F36] max_step làm trọng số đi về 0 DẦN chứ không nhảy, nhưng nó PHẢI về 0.
    # Bản trước bị dòng chuẩn hoá vô điều kiện kéo lại gross 1.0 — cờ này khi đó là
    # cờ giả. Kiểm ở cuối chuỗi, sau khi đã có đủ kỳ để đi hết quãng đường.
    assert W.loc[late[-1]].abs().sum() < 0.05, \
        "collapse_equal=False phải cho danh mục THỰC SỰ rỗng, không chỉ đổi tỷ lệ"


def test_warm_equal_va_collapse_equal_doc_lap():
    """Hai cờ phải điều khiển hai nhánh KHÁC NHAU — nếu không thì việc tách là vô nghĩa."""
    fac = _noise()
    a = CombinerSpec(lookback=200, min_periods=100, t_threshold=6.0,
                     warm_equal=True, collapse_equal=False)
    b = CombinerSpec(lookback=200, min_periods=100, t_threshold=6.0,
                     warm_equal=False, collapse_equal=True)
    Wa, da = adaptive_weights(fac, a, return_diagnostics=True)
    Wb, _ = adaptive_weights(fac, b, return_diagnostics=True)
    warm = da.index[da["branch"] == "warmup"]
    assert Wa.loc[warm[-1]].abs().sum() > Wb.loc[warm[-1]].abs().sum(), \
        "warm_equal=True phải giữ gross trong warm-up, warm_equal=False thì không"


def test_do_manh_bang_chung_duoc_ghi_lai():
    """
    `evidence` là đại lượng bị `target = raw / total` chuẩn hoá mất. Ghi lại nó là
    điều kiện để trả lời được câu hỏi "bằng chứng mạnh hơn có lãi hơn không".
    (Đã đo trên dữ liệu thật: KHÔNG — ngũ phân vị yếu nhất lại có Sharpe cao nhất.)
    """
    fac = _noise(seed=5)
    spec = CombinerSpec(lookback=200, min_periods=100, t_threshold=1.0)
    _, diag = adaptive_weights(fac, spec, return_diagnostics=True)
    ok = diag[diag["branch"] == "ok"]
    assert len(ok) > 0
    assert (ok["evidence"] > 0).all()
    assert (ok["n_active"] >= 1).all()
