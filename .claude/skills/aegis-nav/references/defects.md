# Registry lỗi F1–F16

Nguồn sự thật DUY NHẤT về lỗi đã biết. Sửa xong -> đổi STATUS + xoá khỏi mục "CÒN LẠI".
Mọi lỗi đã vá đều có marker `[FIX Fxx]` trong code: `grep -rn "\[FIX F" src/`
Lỗi đánh dấu 🔧 VÁ HẠ TẦNG nằm ngoài mã nguồn (plist, crontab) nên không có marker.

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
| F32 | cron macOS bỏ lượt khi máy ngủ; daemon chạy code cũ hàng ngày | T0 | 🔧 VÁ HẠ TẦNG | `~/Library/LaunchAgents/com.aegis.trading.plist` |
| F33 | Log daemon bị đệm -> giám sát mù suốt 2 ngày 16 giờ | T1 | 🔧 VÁ HẠ TẦNG | plist: `-u` + `PYTHONUNBUFFERED` |
| F34 | Telegram từ chối tin > 4096 ký tự, chỉ trả về False -> báo cáo im lặng không tới | T1 | ✅ VÁ | `monitoring/position_report.py` `[FIX F34]` |
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


---

# Sự cố hạ tầng 13/09/2026 — F32, F33

Không phải lỗi thuật toán. Lỗi VẬN HÀNH, và cả hai đều thuộc loại im lặng.

## F32 — daemon chạy code cũ, cron bỏ lượt

Phát hiện khi kiểm tra định kỳ:

    daemon khởi động : 10/09 17:59
    các bản vá F21-F31: 11/09 18:56 - 21:32

Tiến trình `run_daily.py --live --loop` chạy liên tục 2 ngày 16 giờ, tức nạp code
TRƯỚC toàn bộ bản vá. Nó không có F22 (một lệnh lỗi giết cả lượt), F27 (nhân đôi vị
thế), F21/F23 (lệnh mồ côi), F25/F29 (cổng chặn). Nhật ký của nó có đúng dòng
`ERROR ... 'rejected'` — chính lỗi đã vá.

**Lượt tái cân bằng kế tiếp còn 32 giờ.** Nếu không phát hiện, nó sẽ chạy bằng đúng
bản đã gây ra cả hai sự cố trước.

Song song: `crontab` 07:00 hằng ngày KHÔNG chạy hai ngày liền, không một dòng log.
Nguyên nhân xác định bằng `pmset -g log`: máy là laptop gập nắp, ngủ gần như liên
tục (Clamshell Sleep). **cron của macOS không chạy bù lượt đã lỡ.**

Đã thay bằng launchd với `KeepAlive` — khởi động lại tiến trình khi máy thức, và
daemon tự kiểm tra "đã đủ 72h chưa" nên bắt kịp lượt bị lỡ.

## F33 — log bị đệm, giám sát mù

Daemon chạy 2 ngày 16 giờ mà log chỉ có dòng `ERROR`/`WARNING`, không một heartbeat.
Nguyên nhân: Python đệm stdout khi đầu ra là file; stderr thì không. Nên ta chỉ thấy
được lỗi, không bao giờ thấy trạng thái bình thường — và "không có tin" bị hiểu nhầm
thành "không có vấn đề".

Đã thêm `-u` và `PYTHONUNBUFFERED=1` vào plist.

## Nguyên tắc bổ sung

9. Tiến trình dài hạn KHÔNG tự nạp lại code. Sau mỗi lần vá đường tiền, phải khởi
   động lại mọi daemon đang chạy — nếu không, bản vá chỉ tồn tại trên đĩa.
10. Bộ lập lịch phải chịu được máy ngủ. Trên macOS dùng launchd, không dùng cron.
11. Log không đệm, nếu không "im lặng" và "hỏng" trông giống hệt nhau.


---

# F34 — báo cáo Telegram im lặng không tới nơi (13/09/2026)

Thêm báo cáo chi tiết vị thế theo yêu cầu vận hành (tên cặp, vốn vào lệnh, % tài
khoản, giá vào, giá thanh lý). Báo cáo 12 vị thế dài **4482 ký tự**; Telegram từ
chối mọi tin quá 4096 và thư viện chỉ trả về `False`, KHÔNG ném ngoại lệ.

Nghĩa là báo cáo sẽ không bao giờ tới nơi và không ai biết. Cùng một họ lỗi với
F33: "im lặng" và "hoạt động bình thường" trông giống hệt nhau.

Đã thêm `chunk_message` cắt ở ranh giới DÒNG (cắt giữa dòng làm hỏng thẻ HTML và
Telegram từ chối cả tin), và phía gọi phải KIỂM TRA giá trị trả về, ghi log lỗi nếu
một phần bị từ chối.

## Ghi chú thiết kế: vì sao không có chốt lời / chốt lỗ theo giá

Câu hỏi vận hành thường gặp. Chiến lược này là danh mục cross-sectional
market-neutral: vị thế mở vì tài sản xếp hạng cao/thấp trong mặt cắt ngang, đóng khi
rơi khỏi nhóm ở lần tái cân bằng. Điều kiện thoát là THỨ HẠNG và THỜI GIAN.

Đặt chốt lỗ theo giá cho từng chân sẽ PHÁ VỠ trung lập: chân long bị cắt còn chân
short vẫn giữ thì danh mục thành cược có hướng — đúng cơ chế làm sổ lệch +35% ngày
10/09. Chốt lỗ thật của chiến lược nằm ở CẤP DANH MỤC: ngắt mạch TIER1/2/3 theo
drawdown, cộng giá thanh lý từng vị thế do sàn tính.

`monitoring/position_report.py` hiển thị đúng những thứ đó thay vì bịa ra TP/SL.

---

# F35 — cắt đoạn đo TRƯỚC khi tính, làm Sharpe holdout đọc nhầm 0,71 thay vì 1,28 (14/09/2026)

`StrategyV2.backtest(mask=...)` cắt lưới tái cân bằng về đoạn cần đo RỒI mới gọi
`target_weights`. Nhưng `adaptive_combiner.adaptive_weights` phụ thuộc ĐƯỜNG ĐI —
nó đi tới từ chỉ số 0 với `prev = 0` và trần `max_step` mỗi kỳ. Đưa cho nó một lưới
đã cắt ngắn nghĩa là bắt nó học lại từ đầu: `min_periods = 120` kỳ đầu chạy trọng số
đều, và cửa sổ học 500 kỳ không bao giờ đầy.

Live KHÔNG gặp handicap đó — live luôn có toàn bộ lịch sử trong tay. Nên con số đo
theo cách cũ không mô tả điều live sẽ trải qua.

`scripts/validate_v3.py` đã phát hiện và né lỗi này từ 10/09 (nó tự tính toàn dòng
thời gian rồi cắt sau), nhưng **bản vá chỉ nằm trong script, không nằm trong thư
viện**. `StrategyV2.backtest` vẫn nguyên lỗi tới 14/09. Bất kỳ ai gọi nó với `mask`
đều nhận một con số thấp hơn sự thật gần một nửa, không có cảnh báo nào.

Bài học chung, đắt hơn bản thân lỗi: **vá một lỗi đo lường trong script mà không vá
trong thư viện thì lỗi vẫn còn sống.** Script là nơi lỗi được PHÁT HIỆN, không phải
nơi nó được SỬA.

Đã vá bằng cách đảo thứ tự (tính toàn bộ -> cắt sau) và khoá bằng
`tests/research/test_mask_after_compute.py`, trong đó có một test khẳng định TIỀN ĐỀ
(tầng gộp thật sự phụ thuộc đường đi) để nếu sau này tầng gộp đổi bản chất thì bất
biến được xoá một cách có ý thức chứ không mục đi trong im lặng.

---

# Hai lỗ hổng TÁI LẬP phát hiện cùng ngày (14/09/2026) — không đánh số F vì không phải lỗi logic

## (a) `strategy_v3.json` không ghi MỐC DỮ LIỆU nên không tái lập được

`load_panel_v2` lọc universe bằng `min_coverage` tính TRÊN CHÍNH panel được nạp.
Panel dài thêm -> tỷ lệ phủ của mọi cặp đổi -> cặp mới vượt ngưỡng và vào rổ. Đã xảy
ra: rổ là **127 cặp tới 12/09 20:00, thành 128 cặp ngay sau đó**. Thêm một cặp làm
đổi THỨ HẠNG của toàn mặt cắt ngang, nên mọi con số — kể cả đoạn train năm 2021 —
đều dịch đi.

Hệ quả: chạy lại `validate_v3.py` hôm nay ra Sharpe khác hôm qua, và không phân biệt
được "code hỏng" với "dữ liệu dài ra". Đã ghim bằng `strategy_v3.V3_VINTAGE_MS` và
`load_v3_data(end_ms=...)`; `scripts/export_returns_v3.py` kiểm chứng parity với
artifact đã ghi (nhóm cấu trúc khớp **0.00e+00**, nhóm phụ thuộc chi phí < 0,72%).

Hệ quả thứ hai, tinh vi hơn và CHƯA sửa: rổ được lọc bằng dữ liệu tới HÔM NAY rồi áp
ngược cho quá khứ — một dạng thiên vị sống sót nhẹ. Ghim mốc không xoá được nó,
nhưng làm nó đứng yên và đo được.

## (b) `estimate_cost_bps` đọc 720 nến CUỐI panel

Nên ước lượng chi phí trôi theo dữ liệu mới, và nó vào thẳng lợi suất ròng của MỌI
kỳ kể cả train. Đây là phần dư duy nhất không tái lập được chính xác (biết ngày chạy
nhưng không biết giờ). Đã xử lý bằng ngưỡng parity hai tầng: nhóm thống kê thuần từ
trọng số khoá tuyệt đối, nhóm phụ thuộc chi phí cho phép 1% tương đối.

---

# F36 — `warm_equal` / `collapse_equal` là CỜ GIẢ: chuẩn hoá gross vô hiệu hoá chúng (14/09/2026)

`adaptive_weights` kết thúc mỗi kỳ bằng `cur = cur / |cur|.sum()`, tức ép gross = 1.0
VÔ ĐIỀU KIỆN. Hệ quả: khi mục tiêu là "đứng ngoài" (vector 0 — chính là thứ mà
`warm_equal=False` và `collapse_equal=False` tồn tại để diễn đạt), phép chuẩn hoá kéo
ngay trọng số trở lại gross đầy đủ.

Hai cờ đó vì vậy chưa bao giờ làm được việc chúng hứa. Chúng đổi TỶ LỆ giữa các họ tín
hiệu nhưng không bao giờ đổi được QUY MÔ. Ý định "không có bằng chứng thì không đặt
cược" bị vô hiệu hoá trong im lặng — không lỗi, không cảnh báo, chỉ là một cờ không có
tác dụng.

Đã vá: chuẩn hoá về gross CỦA MỤC TIÊU thay vì về 1.0. Khi có bằng chứng, `target`
luôn có gross = 1.0 nên kết quả cũ được tái tạo tới sai số dấu phẩy động — parity với
`strategy_v3.json` sau khi vá lệch **2,22e-16** ở nhóm thống kê thuần trọng số.

Khoá bằng `tests/research/test_combiner_evidence.py`.

## Ghi chú liên quan: hai trạng thái bị gộp làm một

Cùng file, nhánh xử lý "chưa đủ lịch sử" (warm-up) và "đủ lịch sử nhưng không họ nào
vượt ngưỡng t" (bằng chứng sụp đổ) dùng chung cờ `warm_equal`. Chúng khác nhau về bản
chất: vô tri khác với một kết luận. Nay tách thành hai cờ, và nhánh sụp đổ GHI CẢNH
BÁO mỗi khi kích hoạt.

Đo được: với 26 tín hiệu, nhánh sụp đổ chạy **0/799 kỳ**. Đây là rủi ro tiềm ẩn, không
phải lỗi đang hoạt động — nhưng nó sẽ kích hoạt nếu ai đó thu hẹp thư viện tín hiệu
hoặc nâng `t_threshold`, và trước bản vá này nó sẽ kích hoạt im lặng.

---

# F37 — bộ lọc universe đo lịch sử SAI FILE: live chạy 62 cặp thay vì 127 (14/09/2026)

## Lỗi

`data/universe.history_lengths` đếm dòng trong `{symbol}_{interval}.parquet`. Nhưng hệ
thống KHÔNG đọc file đó: `data/panel_v2.load_panel_v2` **ưu tiên tổng hợp khung mục
tiêu TỪ `source_interval` (1h)** và chỉ rơi về file đúng khung khi không có nguồn.

Vậy là bộ lọc universe đo một thứ còn đường chạy dùng một thứ khác — đúng họ lỗi F3,
chỉ khác lớp áo.

## Hậu quả đo được

- **89/170 cặp bị loại OAN.** Ví dụ: CRVUSDT có 52.869 nến 1h (= 13.217 nến 4h) nhưng
  `CRVUSDT_4h.parquet` chỉ còn **186 dòng** sót lại từ một lần tải cũ -> bị loại vì
  "thiếu lịch sử".
- Live chạy **62 cặp**, nghiên cứu kiểm định trên **127**.
- Chi phí thật, đo trên cùng dữ liệu và cùng mọi tham số khác:

| rổ | cặp | Sharpe | ann | trung vị fold | fold tệ nhất |
|---|---|---|---|---|---|
| CŨ (lỗi F37) | 60 | **1,20** | 45,5% | 1,04 | −0,62 |
| MỚI (đã vá) | 101 | **1,56** | 64,8% | 1,31 | −0,20 |
| nghiên cứu | 127 | 1,74 | 79,4% | 1,61 | +0,30 |

**0,36 Sharpe và 19,3 điểm phần trăm lợi suất năm.** Quy ra tiền ở nửa Kelly:
~10.450 -> ~17.660 VND/tuần, **+69%**.

Xác nhận mạnh nhất: sau khi vá, `build_universe` trên **MAINNET trả về đúng 127 cặp** —
khớp chính xác cấu hình nghiên cứu. Trước khi vá thì không.

## Vì sao KHÔNG sửa bằng cách tải lại file 4h

File 4h rồi sẽ lại cũ đi và lỗi quay lại — lần sau không ai nhớ vì sao. Cách sửa bền
vững là ĐO ĐÚNG THỨ ĐƯỜNG CHẠY THẬT SẼ DÙNG. `history_lengths` nay nhận
`source_interval` và đếm theo đúng thứ tự ưu tiên của `load_panel_v2`.

Khoá bằng `tests/data/test_universe_history_path.py`.

## Bài học chung — đắt hơn bản thân lỗi

Cả một chiến dịch nghiên cứu trong ngày (độ rộng, mục tiêu biến động, bể chọn, cân theo
bằng chứng, vùng đệm thứ hạng) tìm được đúng **+7,7%** lợi nhuận. Một lỗi ĐÚNG/SAI phát
hiện tình cờ khi viết `scripts/preflight.py` đáng **+69%**.

Ở một hệ thống chưa từng được kiểm tra đầu-cuối, alpha lớn nhất nằm ở chỗ hệ thống
KHÔNG chạy đúng thứ mình tưởng nó đang chạy — không nằm ở mô hình.

---

# F38 — live duyệt `SIGNAL_REGISTRY` thay vì bộ tín hiệu ĐÃ KIỂM ĐỊNH (14/09/2026)

`xs_live_pipeline._compute_target_weights_v3` dựng tín hiệu bằng
`for n in SIGNAL_REGISTRY`. Nhưng registry là nơi CHỨA mọi tín hiệu từng được viết, kể
cả tín hiệu đang thử nghiệm — nó không phải danh sách "những gì đang được giao dịch".

Hệ quả: thêm một họ tín hiệu mới vào registry (việc hoàn toàn hợp lệ khi nghiên cứu)
sẽ âm thầm đổi thứ LIVE đặt lệnh. Và vì `adaptive_weights` chuẩn hoá theo SỐ HỌ
(`equal = 1/n`, `raw / total`), mọi trọng số dịch đi — kể cả khi họ mới toàn NaN và
không bao giờ nhận trọng số. Không crash, không cảnh báo.

Phát hiện ngay khi thêm họ `positioning` (10 tín hiệu): registry thành 36 trong khi
bản đã kiểm định holdout là 26.

**Test parity trọng số KHÔNG bắt được lỗi này** — nó tự dựng cả hai phía bằng cùng một
danh sách, nên research và live có thể đã lệch mà test vẫn xanh. Cùng khe hở mà F37 lọt
qua ở một tham số khác: công thức thì khoá, CẤU HÌNH thì không.

Đã vá: `LiveConfig.v3_signals` (mặc định `V3_SIGNALS` — bộ 26 đóng băng), và live NÉM
LỖI nếu cấu hình yêu cầu tín hiệu không có trong registry thay vì lặng lẽ bỏ qua.

Thêm hai test ở tầng CẤU HÌNH trong `test_research_live_parity_v3.py`: bộ tín hiệu
trùng nhau, và 12 tham số chiến lược của `LiveConfig.from_artifacts` trùng `V3`.

## Quy tắc rút ra sau F37 + F38

Mọi đại lượng mà research và live CÙNG quyết định phải có **đúng một nguồn sự thật**,
và phải có test so hai phía ở tầng cấu hình chứ không chỉ ở tầng công thức. Khoá công
thức mà để hai phía đọc hai cấu hình khác nhau thì test chỉ chứng minh rằng hai hàm
giống nhau — không chứng minh rằng hệ thống đang chạy đúng chiến lược.

---

# F39 — `combined_signal` dùng cả `V3Data.signals` thay vì lọc theo cấu hình (14/09/2026)

Anh em sinh đôi của F38, ở tầng NGHIÊN CỨU thay vì tầng live.

`V3Data` được nạp một lần rồi tái dùng cho nhiều cấu hình — đó vừa là điểm mạnh (nạp
panel mất 20 giây) vừa là cái bẫy. `combined_signal` lấy nguyên `data.signals.items()`,
nên nếu `V3Data` được nạp với 36 tín hiệu thì MỌI cấu hình chạy trên nó đều dùng 36 —
kể cả cấu hình khai báo chỉ 26.

Hệ quả cụ thể: thí nghiệm "26 tín hiệu so với 26+10" chạy 36 ở CẢ HAI phía và cho ra
hai con số y hệt nhau. Không crash, không cảnh báo — chỉ là một thí nghiệm không đo cái
nó tưởng đang đo, và ta sẽ kết luận "thêm tín hiệu không có tác dụng" trong khi chưa hề
thử.

Đã vá: `combined_signal` lọc theo `cfg.signal_names()` và NÉM LỖI nếu `V3Data` thiếu
tín hiệu mà cấu hình yêu cầu — thay vì im lặng chạy với bộ khác.

Parity với `strategy_v3.json` sau khi vá: **2,22e-16** (không đổi).

## Quy tắc chung rút ra từ F37 + F38 + F39

Ba lỗi, một cơ chế: **một đối tượng dùng chung được đọc như thể nó là cấu hình.**
`{symbol}_4h.parquet` bị đọc như "lịch sử khả dụng"; `SIGNAL_REGISTRY` bị đọc như "bộ
tín hiệu đang chạy"; `V3Data.signals` bị đọc như "tín hiệu cấu hình này muốn".

Trong cả ba, thứ được đọc là một CÁI KHO — nó chứa mọi thứ từng có. Cấu hình là một
LỰA CHỌN từ cái kho đó. Nhầm kho với lựa chọn không bao giờ gây lỗi; nó chỉ làm hệ
thống chạy một thứ khác thứ ta nghĩ, và im lặng.

---

# F40 — `min_notional` được giả định ĐỒNG NHẤT $5; sẽ cắn ở lượt MAINNET đầu tiên (14/09/2026)

`max_positions_for_capital(equity, leverage, min_notional=5.0, ...)` giả định MỌI cặp
cần đúng $5. Trên mainnet điều đó SAI: 122/128 cặp cần $5, nhưng ETH/LTC/LINK/ETC/BCH
cần **$20** và BTCUSDT cần **$50**.

Vốn 1.000.000 VND (~$38) ở 2x cho $6,34 mỗi vị thế. Nếu một trong sáu cặp đó lọt vào
top-6, `build_rebalance_plan` bỏ lệnh vào `skipped` và danh mục **MẤT MỘT CHÂN**. Một
sổ market-neutral thiếu một chân không còn trung lập — nó thành cược có hướng, đúng cơ
chế đã làm sổ lệch +35% ngày 10/09 (F25).

Lỗi này CHƯA TỪNG cắn vì hệ thống mới chạy testnet, nơi bộ lọc khác. Nó sẽ cắn ở lượt
mainnet đầu tiên — tức đúng lúc có tiền thật.

Đã vá: `UniverseFilter.max_min_notional` + `symbol_min_notionals()`, lọc TRƯỚC khi xếp
hạng (cùng nguyên tắc mà `resolve_universe` đã áp cho tính giao dịch được).

## Bẫy thứ hai, phát hiện khi viết test cho bản vá thứ nhất

Bản vá đầu chia cho `n_positions` CỨNG: ngưỡng = 38*2/12/1,2 = **$5,28**. Vốn tụt còn
$35 thì ngưỡng thành **$4,86 — dưới mức gần như mọi cặp đều cần**, bộ lọc quét sạch
universe và `resolve_universe` ném lỗi "còn 0 cặp". Một cú sụt 8% vốn sẽ giết đường
chạy, và nguyên nhân chính là lớp bảo vệ vừa thêm vào.

Sửa bằng cách dùng số vị thế ĐÃ ĐIỀU CHỈNH THEO VỐN. Khi đó ngưỡng luôn >= $5:

    n_eff = floor(E*L / (5*safety))  =>  E*L/n_eff >= 5*safety
    ngưỡng = (E*L/n_eff)/safety >= 5

Bài học: **một bộ lọc an toàn có thể tự trở thành nguyên nhân sự cố.** Mọi lớp bảo vệ
mới phải được hỏi "nó hỏng thế nào khi điều kiện xấu đi" — chứ không chỉ "nó có chặn
đúng thứ cần chặn không".

Khoá bằng `tests/data/test_universe_affordability.py` (13 test, gồm bất biến ngưỡng
>= $5 quét qua nhiều mức vốn).
