# Registry lỗi F1–F16

Nguồn sự thật DUY NHẤT về lỗi đã biết. Sửa xong -> đổi STATUS + xoá khỏi mục "CÒN LẠI".
Mọi lỗi đã vá đều có marker `[FIX Fxx]` trong code: `grep -rn "\[FIX F" src/`

| ID | Lỗi | Mức | STATUS | Vị trí |
|---|---|---|---|---|
| F1 | Không có kết nối sàn | T0 | ✅ VÁ | ingestion + OMS đầy đủ, đã kiểm thử đầu-cuối trên testnet |
| F2 | Live chạy trên dữ liệu giả; tự train trên nhiễu | T0 | ✅ VÁ | `main.py` `[FIX F2]` + `data/ingestion/binance_*.py` |
| F3 | Feature vector inference bị zero-hoá (4/5 = 0.0) | T0 | ✅ VÁ | `features/feature_spec.py` + `feature_engine.py` `[FIX F3]` |
| F4 | Bảng Kelly index bằng `p_trend`, tra bằng model proba | T0 | ✅ VÁ | `cpcv_pipeline.py` `[FIX F4]` |
| F5 | CPCV không fit gì; `train_idx` bỏ không -> "OOS" hư cấu | T0 | ✅ VÁ | `cpcv_pipeline.py` `[FIX F5]` |
| F6 | HMM không bao giờ fit; tham số hardcode | T0 | ✅ VÁ | `features/regime/efficiency_regime.py` `[FIX F6]` |
| F7 | Circuit breaker mù với vị thế mở | T1 | ✅ VÁ | `live_pipeline.py` `[FIX F7]` |
| F8 | Không kiểm tra thanh lý ở live | T1 | ✅ VÁ | `live_pipeline.py` `[FIX F8]` |
| F9 | Tầng config chết (nested vs flat key) | T1 | ✅ VÁ | `config_loader.py` `[FIX F9]` |
| F10 | Hai engine PnL khác nhau | T1 | ✅ VÁ | `live_pipeline.py` `[FIX F10]` |
| F11 | Phí exit sai cho lệnh Short | T1 | ✅ VÁ | `pnl.py` `[FIX F11]` |
| F12 | Lệnh dừng khớp không trượt giá | T1 | ✅ VÁ | `live_pipeline.py` `[FIX F12]` |
| F13 | Fade tra nhầm bảng Kelly | T1 | ✅ VÁ | `live_pipeline.py` `[FIX F13]` |
| F14 | `high_watermark` tính nhưng không dùng | T1 | ✅ VÁ | `live_pipeline.py` `[FIX F14]` |
| F15 | Sharpe/DSR sai công thức | T1 | ✅ VÁ | `cpcv_pipeline.py` `[FIX F15]` |
| F6b | Ngưỡng Regime-Flip tuyệt đối 0.35 sai sau khi F6 đổi phân phối p_trend | T1 | ✅ VÁ | `trailing_exit.py:resolve_regime_exit_threshold` |
| F19 | Làm tròn để lại dư số dấu phẩy động -> Binance -1111 | T1 | ✅ VÁ | `portfolio_rebalancer.py` `format_qty/format_price` |
| F20 | Lệnh MARKET báo khớp=0 -> sổ nội bộ lệch với sàn | T1 | ✅ VÁ | `order_router.py` `poll_status` |
| F18 | CPCV không trừ funding perp -> backtest thổi phồng edge | T1 | ✅ VÁ | `cpcv_pipeline.py` `[FIX F18]` |
| F17 | DSR đọc trạng thái toàn cục -> KHÔNG tái lập được | T1 | ✅ VÁ | `validation/dsr.py` `[FIX F17]` |
| F16 | Chân trời purge = 10 nhưng lệnh giữ 120 | T1 | ✅ VÁ | `cpcv_pipeline.py` `[FIX F16]` |
| — | Funding perp không được tính ở đâu cả | T1 | ✅ VÁ | `funding_accrual.py` + `[FIX FUTURES-1]` |

Đường dẫn rút gọn: `src/aegis/pipelines/`, `src/aegis/features/`, `src/aegis/execution/`, `src/aegis/core/`.

---

## Chi tiết lỗi CÒN LẠI

### F1 — OMS vẫn rỗng (ingestion đã xong) `T0`
**Đã xong**: `data/ingestion/binance_rest.py` (REST có ký HMAC, backoff, exchangeInfo
filters), `binance_history.py` (tải nến + funding thật, lưu parquet idempotent),
`core/credentials.py` (khoá từ .env, repr che kín).
**KIẾN TRÚC QUAN TRỌNG**: dữ liệu nghiên cứu lấy từ MAINNET công khai
(`BinanceFuturesREST.public_mainnet()`), đặt lệnh mới qua TESTNET. Lý do: 43.3% nến
testnet có |OFI| bão hoà ±1 (thanh khoản bot giả) so với 0.0% ở mainnet.

**CÒN LẠI**: `oms/order_router.py`, `oms/state_machine.py`, `oms/reconciliation.py`
vẫn là 1 dòng docstring. Xem `plan.md` P0.3.

### F1-cũ — mô tả gốc `T0`
`grep -rn "websocket|ccxt|binance|requests" src/` -> **0 kết quả**. `oms/order_router.py`,
`oms/state_machine.py`, `oms/reconciliation.py` mỗi file 1 dòng docstring.
`data/ingestion/*` đều `raise NotImplementedError`. Hệ thống KHÔNG THỂ đặt lệnh.
Sửa: xem `plan.md` P0.1 + P0.3.

## Lỗi phát hiện thêm khi chạy trên DỮ LIỆU THẬT
- **F6b**: F6 đổi phân phối `p_trend` (Efficiency Ratio, trung vị 0.203) nhưng ngưỡng
  thoát Regime-Flip vẫn là hằng số tuyệt đối **0.35** — đúng ở 80% số nến, giết 53.8%
  số lệnh. Nay lấy theo **phân vị của p_trend TRONG TRAIN FOLD** (không rò rỉ).
  *Bài học*: mọi ngưỡng tuyệt đối đặt trên đại lượng có phân phối phụ thuộc mô hình
  đều là lỗi chờ phát nổ.
- **F17**: `dsr.py` gọi `ExperimentTracker().get_total_trials()` (đếm dòng log) rồi
  `max()` với `num_trials` của caller. DSR do đó **không tái lập được** — cùng input
  ra kết quả khác nhau tuỳ máy đã chạy bao nhiêu thí nghiệm. Nay là hàm thuần tuý;
  muốn dùng tracker phải bật `use_experiment_tracker=True` tường minh.
- **F18**: `finalize_trade_record` trong CPCV không nhận `funding_accrued` -> mọi
  backtest bỏ qua chi phí đặc trưng của perp. BTC funding TB +0.0101%/8h; lệnh giữ
  5 ngày = 15 chu kỳ = **0.152% notional** — đủ xoá sạch một "edge" +0.209%/lệnh.
  Nay dùng lịch funding THẬT `{ts_ms: rate}` tải từ sàn.

## Bẫy phương pháp luận (KHÔNG phải bug, nhưng gây tự lừa mình)
CPCV `C(6,2)=15` fold khiến mỗi sự kiện xuất hiện trong test fold ~3.5-5 lần.
`clean_records` do đó ĐẾM TRÙNG. Dùng thẳng `len(records)` làm cỡ mẫu sẽ thổi phồng
ý nghĩa thống kê ~2x. **Luôn gộp về `entry_idx` duy nhất trước khi tính t-stat.**
Tương tự, KHÔNG được nhân dồn `(1+r)` qua các bản ghi CPCV — chúng chồng lấn thời
gian, không phải một đường vốn tuần tự.

## Lỗi chỉ lộ ra khi chạy SÀN THẬT (mock không bắt được)
- **F19 precision**: `round(price/tick)*tick` cho `1.2426000000000002`; và với
  `step_size=1.0`, float `1234.0` tuần tự hoá thành `"1234.0"` trong khi
  `quantityPrecision=0`. Cả hai đều bị Binance từ chối với `-1111`.
  Vá: `SymbolFilters.format_qty/format_price` kiểm soát khâu ĐỊNH DẠNG CHUỖI,
  và router gửi chuỗi thay vì float.
- **F20 theo dõi khớp**: lệnh MARKET trả về `status=NEW, executedQty=0` vì phản hồi
  đi TRƯỚC khi khớp được ghi nhận. Sổ nội bộ báo "chưa khớp" trong khi sàn đã có
  vị thế. Vá: `poll_status()` truy vấn lại sau khi gửi lệnh không post-only.

**Bài học**: hai lỗi này không thể phát hiện bằng mock. Bắt buộc chạy testnet thật.
