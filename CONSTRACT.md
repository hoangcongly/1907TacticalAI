# AFML ENGINE v11.9 — DATA CONTRACTS

Bản hợp đồng này quy định giao thức giao tiếp dữ liệu nghiêm ngặt giữa Track A (Data & Signal) và Track B (Labeling & Decision). Bất kỳ thay đổi nào đối với schema dưới đây đều cần đồng thuận và review chéo của cả hai Track — thực thi bằng CI (`tests/integration/test_data_contracts.py`), không chỉ bằng quy ước bằng lời.

Lưu ý về đánh số phiên bản: số hiệu tài liệu (v11.9) và `schema_version` gắn trong dữ liệu là hai không gian đánh số độc lập. Đây là lần đầu schema được đánh version chính thức — bắt đầu lại từ `1.0.0`.

## 1. SIGNAL_BAR_SCHEMA (Track A → Track B)
Output của `build_signal_bars()`, `schema_version = "1.0.0"`. `timestamp_ms` LÀ `knowledge_time` (thời điểm bar sẵn sàng để dùng).

1. `bar_idx` (int64): >= 0, duy nhất, vị trí tuyệt đối trong mảng.
2. `symbol` (str)
3. `timestamp_ms` (int64): epoch ms (knowledge_time).
4. `open` (float64): > 0
5. `high` (float64): > 0, >= open/close/low
6. `low` (float64): > 0, <= open/close/high
7. `close` (float64): > 0
8. `volume` (float64): >= 0
9. `ofi` (float64): [-1.0, 1.0]
10. `tick_count` (int64): >= 0
11. `is_toxic_flag` (bool)
12. `is_tail_event` (bool)
13. `insufficient_history` (bool): True trong W bar đầu sau mọi gap.
14. `trend_score` (float64): nullable (null khi insufficient_history=True)
15. `p_trend` (float64): [0,1], nullable
16. `p_chop` (float64): [0,1], nullable
17. `atr_14` (float64): >= 0, nullable
18. `hurst_value` (float64): [0,1], nullable
19. `d_star_used` (float64): [0,1], nullable

## 2. TRADE_RECORD_SCHEMA (Nội bộ Track B & Module F/G)
`schema_version = "1.0.0"`.

1. `schema_version` (str): "1.0.0"
2. `dataset_manifest_hash` (str): SHA-256 khớp với mảng nến sinh ra nó.
3. `fold_id` (str, nullable)
4. `symbol` (str)
5. `entry_idx` (int): Chỉ số bar_idx tuyệt đối.
6. `entry_price` (float): > 0
7. `p_i` (float): [0,1]
8. `p_chop_i` (float): [0,1]
9. `mode` (str): "follow" | "fade"
10. `side` (int): +1 | -1
11. `sl_initial` (float): > 0
12. `size_notional` (float): > 0
13. `exit_idx_relative` (int): >= 0
14. `exit_idx_absolute` (int): = entry_idx + 1 + exit_idx_relative
15. `exit_reason` (str): "SL" | "TRAIL" | "REGIME_FLIP" | "TIME_STOP" | "LIQUIDATION"
16. `fill_price_exit` (float): > 0
17. `boundary_truncated` (bool)
18. `fee_entry` (float): >= 0
19. `fee_exit` (float): >= 0
20. `funding_accrued` (float)
21. `gross_pnl` (float)
22. `realized_return` (float)

## 3. Hợp đồng hành vi (Behavioral Contract)
- Track B PHẢI bỏ qua bar có `is_toxic_flag=True` hoặc `insufficient_history=True` khi sinh sự kiện CUSUM.
- `is_tail_event=True` KHÔNG đồng nghĩa với bỏ qua.
- Khi `insufficient_history=True`, các cột tín hiệu bắt buộc là null. Track A không được điền giá trị "tạm".
- `exit_idx_absolute` là chỉ số DUY NHẤT được phép dùng để tra `fill_price_exit` và `funding_accrued`.
```eof

### 2. Thực thi Contract bằng Pandera (Module Kiểm Tra)

*(Bố nhớ cài `pip install pandera pytest` trước khi chạy nhé)*

http://googleusercontent.com/immersive_entry_chip/0

### 3. CI Integration Tests (Gatekeeper của Hệ thống)

http://googleusercontent.com/immersive_entry_chip/1

