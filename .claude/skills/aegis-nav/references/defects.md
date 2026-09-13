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
| F21 | Huỷ lệnh không xác nhận -> lệnh post-only sống tiếp trên sàn | T0 | ✅ VÁ | `oms/order_router.py` `[FIX F21]` |
| F22 | Một lệnh lỗi giết cả lượt tái cân bằng | T0 | ✅ VÁ | `oms/order_router.py` `[FIX F22]` |
| F23 | Vòng chờ hỏng thì bỏ qua luôn bước dọn dẹp | T0 | ✅ VÁ | `oms/order_router.py` `[FIX F23]` |
| F24 | Sổ nội bộ không đồng bộ khi thực thi ném lỗi; chụp vị thế một lần | T0 | ✅ VÁ | `pipelines/xs_live_pipeline.py` `[FIX F24]` |
| F25 | Không có phép kiểm lệch hướng sau thực thi | T0 | ✅ VÁ | `pipelines/xs_live_pipeline.py` `[FIX F25]` |
| F26 | `refresh_data` làm mới sai khung -> v3 kẹt STALE_DATA vĩnh viễn | T0 | ✅ VÁ | `pipelines/xs_live_pipeline.py` `[FIX F26]` |
| F27 | Báo giá lại đếm trùng phần đã khớp -> vị thế NHÂN ĐÔI | T0 | ✅ VÁ | `oms/order_router.py` `[FIX F27]` |
| F28 | Không có khoá chống hai tiến trình cùng tái cân bằng | T0 | ✅ VÁ | `core/process_lock.py` `[FIX F28]` |
| F29 | Cổng chặn chỉ kiểm lệch hướng, không kiểm đòn bẩy gộp | T0 | ✅ VÁ | `pipelines/xs_live_pipeline.py` `[FIX F29]` |
| F30 | Chốt nhịp tái cân bằng chỉ có ở daemon -> cron cân sai chu kỳ | T0 | ✅ VÁ | `pipelines/xs_live_pipeline.py` `[FIX F30]` |
| F31 | Phí hardcode 1bp/4bp trong module ĐO chi phí thật (thật: 2bp/5bp) | T1 | ✅ VÁ | `core/execution_log.py` `[FIX F31]` |
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


---

# Sự cố thực thi 10/09/2026 — nguồn gốc F21–F26

Một ngoại lệ KHÔNG ĐƯỢC BẮT trong `submit_plan` đã sinh ra toàn bộ sáu lỗi trên.
Đây là ví dụ mẫu về việc một lỗi nhỏ ở tầng thấp lan thành hỏng hóc hệ thống.

**Diễn biến dựng lại từ nhật ký sàn (không suy đoán):**

| thời điểm | việc xảy ra |
|---|---|
| 10:30:47 | đặt loạt lệnh post-only cho 12 cặp |
| ~10:31 | ngoại lệ thoát ra khỏi `submit_plan` giữa chừng |
| | -> SANDUSDT, XMRUSDT (cuối danh sách MỞ) **chưa từng được đặt** |
| | -> bước dọn dẹp không chạy: lệnh COTIUSDT **vẫn sống trên sàn** |
| | -> bước lưu trạng thái không tới: `rebalance_count` đứng ở 1 |
| 14:10:07 | lệnh COTIUSDT nằm chờ **3 giờ 39 phút** rồi tự khớp, không ai theo dõi |
| 11/09 00:00 | hệ thống tự khoá vì lệch sổ sách — **chốt chặn này hoạt động ĐÚNG** |

**Thiệt hại đo được lúc 11/09:** danh mục còn 10/12 vị thế, lệch **+34.95%** khỏi
trung lập (long $4.270 / short $2.058). Trong 27 giờ thị trường giảm 6.09%: riêng
phần lệch hướng mất **2.61% equity**, trong khi phần chọn cặp LÃI. Nói cách khác,
toàn bộ khoản lỗ đến từ đường ống, không đến từ chiến lược.

**Vì sao đáng ghi lại:** lỗi không nằm ở công thức nào cả. Nó nằm ở giả định ngầm
rằng "mọi thứ sẽ chạy tới hết". Ba nguyên tắc rút ra, đã đưa vào code:

1. Vòng lặp qua N đối tượng phải cách ly lỗi từng đối tượng. Một cặp hỏng không
   được kéo theo 11 cặp còn lại.
2. Dọn dẹp phải nằm trong `finally`. Lệnh sống trên sàn mà không ai theo dõi là
   trạng thái nguy hiểm nhất mà hệ thống này có thể rơi vào.
3. Bất biến quan trọng nhất phải được ĐO sau khi thực thi, không phải giả định.
   Sổ lệch 35% chạy 27 giờ mà không có gì báo động — vì không ai đo.

Test hồi quy: `tests/oms/test_execution_incident_20260910.py`,
`tests/pipelines/test_neutrality_and_state_sync.py`.


---

# Sự cố nhân đôi vị thế 11/09/2026 — nguồn gốc F27, F28

Xảy ra ngay trong lượt mở sổ đầu tiên sau khi vá F21–F26. Lần này nguyên nhân nằm
ở chỗ khác hẳn, và nó cho thấy tại sao phải ĐO sau khi thực thi chứ không giả định.

**Nhật ký sàn VETUSDT, một lượt duy nhất:**

| thời điểm | lệnh | khớp |
|---|---|---|
| 18:46:24 | LIMIT GTX 210.356 | 0, bị huỷ |
| 18:46:55 | LIMIT GTX 210.356 (báo giá lại) | **110.276**, bị huỷ |
| 18:48:48 | LIMIT GTX 210.213 (báo giá lại) | **210.213** |
| 18:48:52 | MARKET (cắn giá) | **100.080** |
| | **tổng** | **420.569 = gấp đôi mục tiêu 210.213** |

**Nguyên nhân:** mỗi lần báo giá lại, `passive[idx]` bị GHI ĐÈ bằng đối tượng lệnh
mới. Phần đã khớp của lệnh cũ biến mất khỏi danh sách, nên "còn thiếu" được tính
theo riêng lệnh mới nhất thay vì theo Ý ĐỊNH GỐC trừ TỔNG đã khớp.

**Hậu quả:** toàn danh mục chạy **3,69x đòn bẩy** thay vì 2,0x — gần gấp đôi rủi ro
dự kiến. Cổng chặn F25 bắt được lệch hướng +6,4% nhưng KHÔNG bắt được lệch đòn bẩy,
vì lúc đó chưa có phép kiểm nào trên gross.

**F28 phát hiện kèm:** `crontab` của máy có đúng một dòng bị lặp hai lần, nên mỗi
07:00 có hai tiến trình cùng chạy `--live`. Id lệnh tất định chỉ chống trùng TRONG
một phút (`epoch_bucket = time//60`), không chống được hai tiến trình lệch nhịp.
Đã dọn crontab và thêm khoá file (`artifacts/.run_daily.lock`).

**F29 phát hiện kèm:** cổng chặn F25 bắt được lệch hướng +6,4% nhưng KHÔNG thấy
đòn bẩy đã là 3,69x. Một sổ nhân đôi ĐỐI XỨNG vẫn trung lập hoàn hảo — net và gross
là hai đại lượng độc lập và phải kiểm riêng. Đã bổ sung kiểm đòn bẩy gộp.

**Nguyên tắc bổ sung rút ra:**

4. Đại lượng tích luỹ (khối lượng đã khớp) phải theo dõi ở cấp Ý ĐỊNH, không ở cấp
   đối tượng lệnh. Đối tượng lệnh có vòng đời ngắn hơn ý định.
5. Mọi tiến trình chạm tiền phải có khoá loại trừ. Giả định "chỉ có một bản chạy"
   là giả định, không phải bảo đảm.
6. Một cổng chặn chỉ bắt được đúng đại lượng nó đo. Trung lập và đòn bẩy độc lập
   nhau; kiểm một cái không nói gì về cái kia.

Test hồi quy: `tests/oms/test_execution_incident_20260910.py` (mục F27).


---

# F30 — cron tái cân bằng sai chu kỳ (phát hiện 11/09/2026, chưa kịp gây hại)

Chốt "đã đến hạn tái cân bằng chưa" tồn tại trong `run_daily.py --loop` (chế độ
daemon) nhưng KHÔNG có ở `run_once`, tức đường mà `crontab` đang dùng.

`crontab` chạy `run_daily.py --live` lúc 07:00 hằng ngày. Không có chốt nhịp thì hệ
thống tái cân bằng **mỗi 24h** trong khi cấu hình đã kiểm định là **72h** (18 nến 4h).

Đây không phải "chạy dày hơn một chút". Tín hiệu được tính trên lưới 18 nến nhưng
vị thế bị đặt lại mỗi ngày, nên **nhịp tín hiệu và nhịp thực thi lệch nhau** — cấu
hình đang chạy không trùng với bất kỳ cấu hình nào trong lưới kiểm định. Kèm theo
chi phí giao dịch gấp khoảng 3 lần.

Đã chuyển chốt nhịp vào `run_once` để MỌI điểm vào đều tuân thủ. Lượt chạy chưa đến
hạn giờ trả về `HEARTBEAT`: không giao dịch, nhưng vẫn chạy cổng chặn F25/F29 — tức
là giám sát hằng ngày miễn phí. Cờ `--force` dành cho can thiệp thủ công.

**Nguyên tắc bổ sung:**

7. Chốt chặn đặt ở tầng điểm-vào sẽ bị bỏ qua bởi điểm vào khác. Bất biến phải nằm
   ở tầng thực thi chung, nơi mọi đường đi đều phải chạy qua.


---

# Cổng chất lượng trước khi bơm tiền thật (11/09/2026)

`scripts/readiness_gate.py` — chấm bằng máy, không bằng mắt.

**Vì sao cần:** tính tới 11/09/2026, số lượt tái cân bằng chạy đúng mà không cần
can thiệp tay là **0 trên 4**, và tốc độ phát hiện lỗi T0 vẫn là **10 lỗi trong 2
ngày** (F21–F30) — đường cong chưa đi ngang. Mắt người nhìn nhật ký rồi kết luận
"trông ổn" sẽ luôn kết luận là ổn, nhất là khi đang sốt ruột muốn vào tiền.

**Điều kiện:** 3 lượt LIÊN TIẾP đều sạch — mọi lệnh đặt được (F22), không lệnh sống
ngoài kiểm soát (F21), thực thi không ném lỗi (F24), trung lập trong trần (F25),
đòn bẩy đúng (F29), không ai sửa tay. Ở chu kỳ 72h là ~9 ngày.

**Cổng này KHÔNG nói gì về lợi nhuận** — chỉ nói đường ống đã thôi vỡ. Lợi nhuận cần
~891 ngày mới có ý nghĩa thống kê (`docs/upgrade_v3_report.md` §7).

Tiến độ được in ở mọi lượt chạy `run_daily.py`, nên không ai phải tự nhớ.

**Nguyên tắc bổ sung:**

8. Tiêu chí "đã sẵn sàng chưa" phải nhị phân và do máy chấm. Tiêu chí định tính sẽ
   luôn được diễn giải theo hướng ta đang muốn đi.
