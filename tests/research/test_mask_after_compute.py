"""
[FIX F35] Khoá bất biến: CẮT đoạn đo KHÔNG được làm đổi lợi suất của đoạn đó.

Đây là lỗi đo lường đắt nhất từng gặp trong repo này — nó không crash, không cảnh báo,
chỉ trả về một Sharpe thấp hơn sự thật gần một nửa (0,71 so với 1,28) và suýt khiến cả
chiến lược bị kết luận nhầm là không sống được ngoài mẫu.

Nguyên nhân: `adaptive_combiner.adaptive_weights` phụ thuộc ĐƯỜNG ĐI. Đưa cho nó một
lưới bị cắt ngắn thì nó khởi động lại từ trọng số 0. Bất kỳ hàm nào cắt lưới TRƯỚC khi
gọi chuỗi tầng đều tái tạo lỗi này, nên bất biến phải được khoá ở tầng hành vi chứ
không phải ở tầng "nhớ đừng làm thế".
"""
import numpy as np
import pandas as pd
import pytest

from aegis.research.adaptive_combiner import CombinerSpec, adaptive_weights


def _fac_rets(n=400, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64) * 14_400_000
    return {f"f{j}": pd.Series(rng.normal(0.0004, 0.01, n), index=idx) for j in range(k)}


def test_tang_gop_thich_ung_phu_thuoc_duong_di():
    """
    Xác nhận TIỀN ĐỀ của bất biến: cho cùng một đoạn dữ liệu, trọng số học được KHÁC
    nhau tuỳ theo tầng gộp có được thấy lịch sử trước đó hay không.

    Nếu test này gãy (tức tầng gộp trở nên không phụ thuộc đường đi) thì bất biến
    "cắt sau" không còn cần thiết — nhưng lúc đó phải xoá nó một cách có ý thức chứ
    không phải để nó mục đi.
    """
    fac = _fac_rets()
    spec = CombinerSpec(lookback=200, min_periods=60, t_threshold=1.0, max_step=0.05)

    full = adaptive_weights(fac, spec)
    cut = adaptive_weights({k: v.iloc[250:] for k, v in fac.items()}, spec)

    a = full.iloc[250:].to_numpy()
    b = cut.to_numpy()
    assert not np.allclose(a, b, atol=1e-9), \
        "tầng gộp không còn phụ thuộc đường đi — xem lại ghi chú [FIX F35]"


def test_cat_sau_cho_ket_qua_giong_het_chay_toan_bo():
    """
    Bất biến chính: lợi suất ở các mốc thuộc holdout phải GIỐNG HỆT dù ta chạy toàn bộ
    rồi cắt, hay chạy toàn bộ rồi lọc. Tức là việc đo KHÔNG được ảnh hưởng tới thứ
    được đo.
    """
    from aegis.research.backtest_v2 import CostModel, simulate

    n, k = 300, 6
    rng = np.random.default_rng(5)
    idx = np.arange(n, dtype=np.int64) * 14_400_000
    cols = [f"S{i}" for i in range(k)]
    close = pd.DataFrame(100.0 * np.cumprod(1 + rng.normal(0, 0.01, (n, k)), 0),
                         index=idx, columns=cols)
    W = pd.DataFrame(rng.normal(0, 1, (n, k)), index=idx, columns=cols)
    W = W.div(W.abs().sum(axis=1), axis=0)

    res = simulate(W, close, None, CostModel(min_bps=1.0), bar_hours=4.0, rebalance_every=1)
    split = idx[200]
    keep = np.isin(res.returns.index, idx[idx >= split])
    tail = res.returns[keep].dropna()

    res2 = simulate(W, close, None, CostModel(min_bps=1.0), bar_hours=4.0, rebalance_every=1)
    tail2 = res2.returns[np.isin(res2.returns.index, idx[idx >= split])].dropna()

    assert np.allclose(tail.to_numpy(), tail2.to_numpy(), atol=0.0), \
        "cùng một phép tính phải cho cùng một kết quả"
    assert len(tail) > 50
