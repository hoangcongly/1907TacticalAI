# Bất biến hệ thống + lệnh kiểm chứng

Vi phạm bất kỳ dòng nào dưới đây = mất tiền thật. Mỗi bất biến có lệnh kiểm chứng
chạy được ngay — dùng nó thay vì đọc code.

## I1. Nhân quả (causality) — không được nhìn tương lai
Mọi feature tại nến `t` chỉ được dùng dữ liệu `<= t`.
- ❌ CẤM: `.rolling(...).mean()` rồi dùng ở chính nến đó mà không `.shift(1)`
- ❌ CẤM: `center=True`, backward smoothing của HMM, `fit()` trên toàn bộ data rồi predict in-sample
- ✅ HMM chỉ được dùng forward pass (`hmm_causal.py:47 step()`), KHÔNG dùng Viterbi/smoothing
```bash
grep -rn "center=True\|bfill\|backfill\|shift(-" src/aegis/features/
```

## I2. Parity research ↔ live — CÙNG MỘT code tính feature
Feature lúc train và lúc inference phải sinh ra từ CÙNG một hàm.
Vi phạm hiện tại = **F3**.
```bash
# Feature nào model cần mà live không có -> đó là bug
python -c "
import json;from pathlib import Path
f=json.load(open('artifacts/selected_features.json'))
print('model cần:',f)"
```

## I3. Chỉ MỘT engine PnL
Mọi tính tiền đi qua `execution/pnl.py:compute_realized_pnl`. Không được tự viết
công thức PnL ở bất kỳ đâu khác.
```bash
grep -rn "gross_pnl\s*=\|net_pnl\s*=" src/ --include="*.py" | grep -v "execution/pnl.py"
```

## I4. Chi phí futures phải đủ 4 khoản
`phí (maker/taker) + funding (8h) + thanh lý + trượt giá`. Thiếu một khoản = backtest nói dối.
```bash
python -m pytest tests/execution/test_pnl.py tests/risk/ -q
```

## I5. Ký quỹ cô lập — lỗ không vượt quá margin đã cọc
`net_pnl >= -(size_notional / leverage)`. Ép ở `live_pipeline.py` sau `compute_realized_pnl`.

## I6. Circuit breaker chấm trên equity MARK-TO-MARKET
Phải là `margin_balance` (gồm uPnL), KHÔNG phải `wallet_balance`.
```bash
grep -n "update_equity" src/aegis/pipelines/live_pipeline.py   # phải thấy margin_balance
```

## I7. SL phải nằm TRONG vùng an toàn trước giá thanh lý
Đệm tối thiểu `safety_buffer_pct = 0.15`. Ép ở pre-flight `validate_leverage_against_sl`.

## I8. Purge/embargo phải phủ TOÀN BỘ thời gian giữ lệnh
`t1 - t0 >= t_max_live`. Vi phạm hiện tại = **F16**.

## I9. Bảng Kelly index bằng ĐÚNG đại lượng dùng lúc inference
Dựng bảng bằng đại lượng X thì tra bảng cũng phải bằng X. Vi phạm hiện tại = **F4**.

## I10. Canonical config là nguồn sự thật DUY NHẤT
Tham số đọc từ `config/aegis_canonical_parameters.yaml` qua `load_canonical_config()`
(đã làm phẳng). Không hardcode hằng số chiến lược trong code.
```bash
python -c "
from aegis.core.config_loader import load_canonical_config as L
c=L()
[print(f'{k:22} = {c.get(k,\"<<MISSING>>\")}') for k in
 ['c_trade','m_sl','t_max_live','embargo_bars','taker_fee_rate','safety_buffer_pct']]"
```

---

## Lệnh kiểm chứng nhanh
```bash
python -m pytest -q                      # toàn bộ (~60s)
python -m pytest tests/execution tests/risk tests/pipelines -q   # đường tiền
python scripts/gen_codemap.py            # sinh lại bản đồ sau refactor
grep -rn "\[FIX F" src/                  # các lỗ hổng đã vá
```

## Bẫy đã biết
- `fast_mode` mặc định `True` (`main.py`) -> **bỏ qua hoàn toàn** feature selection,
  chỉ lấy `available_cols[:5]`.
- `load_canonical_config()` trả về **cả** nhánh lồng nhau **lẫn** khoá phẳng.
  Khoá phẳng đặt tường minh ở cấp gốc luôn thắng.
- Bar là **dollar-volume bar**, không phải nến thời gian -> "120 nến" KHÔNG phải 120 phút.
  Mọi tính toán theo thời gian (funding!) phải dùng `timestamp_ms`, không dùng số nến.
- `CPCVPipeline` hardcode `fade_enabled=True` (:198) trong khi live mặc định `False`.
