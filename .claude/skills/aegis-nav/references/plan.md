# Kế hoạch P0 / P1 / P2

Thứ tự BẮT BUỘC: P0 → P1 → P2. Làm P2 trước P0 là tối ưu một hệ thống không chạy được.

Ký hiệu: `[Fxx]` = vá lỗ hổng tương ứng trong `defects.md`.

---

# P0 — XƯƠNG SỐNG (không có = không thể kiếm tiền)

Mục tiêu thoát P0: **một lệnh thật khớp trên testnet, đối chiếu khớp sổ sách.**

## P0.1 — Nạp dữ liệu Binance USDⓈ-M thật `[F2]`
Thay `data/ingestion/*` (đang `raise NotImplementedError`).

| File tạo/sửa | Nội dung |
|---|---|
| `data/ingestion/binance_rest.py` | REST: `/fapi/v1/klines` (lịch sử), `/fapi/v1/exchangeInfo` (bộ lọc: tickSize, stepSize, minNotional), `/fapi/v1/fundingRate` (lịch sử funding), `/fapi/v1/premiumIndex` (mark price) |
| `data/ingestion/binance_ws.py` | WebSocket: `aggTrade`, `kline`, `bookTicker`, `markPrice`. Tự kết nối lại + heartbeat |
| `data/ingestion/store.py` | Ghi parquet phân vùng theo ngày, chống ghi trùng bằng khoá `(symbol, open_time)` |

**Then chốt (acceptance):**
- Tải được ≥ 90 ngày lịch sử cho ≥ 3 cặp, không thủng lỗ dữ liệu
- WS chạy 24h liên tục không mất kết nối vĩnh viễn
- `exchangeInfo` filters được lưu và **thực sự dùng** để làm tròn giá/khối lượng
- Test: dữ liệu nạp lại 2 lần cho ra kết quả byte-identical

⚠️ Funding phải lấy **rate thật theo thời gian**, không dùng hằng số mặc định
(`funding_accrual.py` đã nhận `dict {ts: rate}` — đấu nối vào đây).

## P0.2 — Module feature parity `[F3]` ← **QUAN TRỌNG NHẤT VỀ KIẾN TRÚC**
Nguồn gốc F3: research tự tính `hl_spread/ret_1/ret_5/vol_ratio` trong
`research_pipeline._ensure_candidate_features`, live thì không → `bar.get(col, 0.0)` trả 0.0.

**Không được vá bằng cách copy công thức sang live** — đó là cách lỗi tái phát.

| File tạo | Nội dung |
|---|---|
| `features/feature_spec.py` | Khai báo từng feature MỘT LẦN: tên, số nến warm-up, hàm tính nhân quả |
| `features/online_features.py` | Bộ tính trạng thái tăng dần (streaming), phát 1 vector / 1 nến |
| `features/batch_features.py` | Bộ tính theo lô cho research — **gọi lại đúng hàm** trong `feature_spec` |

**Then chốt:**
- Test parity bắt buộc: `online(bars) == batch(bars)` từng phần tử, sai số `< 1e-9`
- Research và live cùng nạp `selected_features.json`; nếu live thiếu feature nào →
  **ném lỗi**, tuyệt đối không im lặng điền 0.0
- Xoá `_ensure_candidate_features` khỏi `research_pipeline.py`

## P0.3 — OMS thật `[F1]`
Thay 3 file rỗng trong `oms/`.

| File | Nội dung |
|---|---|
| `oms/order_router.py` | Đặt/huỷ/sửa lệnh, ký HMAC-SHA256, `newClientOrderId` idempotent, xử lý rate limit + backoff |
| `oms/state_machine.py` | `CREATED → SUBMITTED → PARTIAL → FILLED / CANCELED / REJECTED`. Chuyển trạng thái chỉ theo sự kiện từ `userDataStream` |
| `oms/reconciliation.py` | Khi khởi động: đọc `/fapi/v2/positionRisk` + `/fapi/v1/openOrders`, so với trạng thái nội bộ. Lệch → **dừng, không tự đoán** |

**Cấu hình futures bắt buộc gọi lúc khởi động:**
- `POST /fapi/v1/marginType` → `ISOLATED`
- `POST /fapi/v1/leverage` → khớp `max_safe_leverage`
- Xác nhận position mode (one-way vs hedge)

**Then chốt:** chạy testnet, đặt→khớp→đóng 1 lệnh, sổ sách nội bộ khớp sàn 100%.
Kill switch huỷ sạch lệnh treo + đóng vị thế trong < 5s.

## P0.4 — Trạng thái bền vững `[F8]`
Hiện `peak_equity` / `is_dead` / vị thế mở đều mất khi restart → xoá sạch trí nhớ drawdown.

- `core/state_store.py`: ghi SQLite/JSON sau **mỗi** thay đổi trạng thái
- Khởi động lại: khôi phục `peak_equity`, `is_dead`, `frozen_until_ms`, vị thế đang mở
- **Then chốt:** giết tiến trình giữa lúc có vị thế mở → khởi động lại → trạng thái nguyên vẹn

## P0.5 — Chặn cứng dữ liệu giả `[F2]`
`main.py:72,135`: live/paper phải **từ chối chạy** nếu nguồn không phải feed thật.
Xoá hẳn nhánh tự-train-trên-nhiễu (`main.py:135`).

---

# P1 — LÀM CHO NGHIÊN CỨU TRUNG THỰC

Mục tiêu thoát P1: **một con số OOS Sharpe mà ta dám tin.**

## P1.1 — CPCV fit thật từng fold `[F5]`
`cpcv_pipeline.py:166` — `train_idx` hiện không bao giờ được dùng.

Trong mỗi fold phải: fit model trên `train_idx` → predict trên `test_idx` →
dùng xác suất OOS đó để mô phỏng giao dịch. Feature selection + calibration
cũng phải nằm **trong** vòng lặp fold (hiện đang chạy trên toàn bộ data → rò rỉ lựa chọn).

**Then chốt:** thay `test_pipeline_refit_per_fold.py` (đang dùng MockEstimator) bằng
test chạy pipeline THẬT, xác nhận số lần fit == số fold và các tập train khác nhau.

## P1.2 — Bảng Kelly index đúng đại lượng `[F4]`
`cpcv_pipeline.py:192` dựng bảng bằng `p_trend`; `live_pipeline.py:504` tra bằng
`model.predict_proba()`. Sau P1.1 sẽ có xác suất OOS thật → dùng **chính nó** để dựng bảng.

**Then chốt:** test khẳng định phân phối `p_i` lúc dựng bảng và lúc tra cứu
đến từ cùng một nguồn (so sánh min/max/mean).

## P1.3 — Chân trời purge `[F16]`
`cpcv_pipeline.py:149`: `t1 = t0 + 10` → đổi thành `t0 + t_max_live_follow` (=120, canonical).
**Then chốt:** tỷ lệ truncation < 15% (hiện 17–37%).

## P1.4 — Sharpe / DSR đúng công thức `[F15]`
- `:242` annualize bằng **tần suất giao dịch thật** (`n_trades / số năm dữ liệu`), bỏ `sqrt(252)` cứng
- `:250` `variance_of_srs` = phương sai **của các Sharpe qua các trial**, không phải của lợi suất

**Then chốt:** test với chuỗi lợi suất đã biết Sharpe giải tích, sai số < 1%.

## P1.5 — Trailing dùng high-watermark `[F14]`
`live_pipeline.py:364,378` cập nhật `high_watermark` rồi vứt; trailing dùng `close ± 2*ATR`.
Đổi sang chandelier `high_watermark - k*ATR`, **đồng bộ với `labeling/trailing_exit.py`**
(nơi đã dựng ra bảng Kelly) — nếu không, calibration và inference lệch nhau.

## P1.6 — HMM: fit hoặc bỏ `[F6]`
`signal_pipeline.py:39-43` hardcode `means=[0.001,0.0]`, `stds=[0.02,0.005]` —
không hiệu chỉnh theo tài sản/khung thời gian; thực chất chỉ là phân loại "biến động cao".

Hai lựa chọn:
- **(a)** Cài `fit_hmm_2state_loglik` bằng Baum-Welch EM trên cửa sổ trượt, refit định kỳ.
  Bắt buộc **fit nhân quả**: tham số dùng ở nến `t` chỉ được học từ dữ liệu `< t`.
- **(b)** **Bỏ HMM** — thay bằng phân loại chế độ đơn giản, bền hơn (ví dụ tỷ lệ
  biến động thực hiện, hoặc phân vị ATR). Ít bậc tự do hơn = ít overfit hơn.

**Khuyến nghị: (b) trước.** HMM 2 trạng thái với tham số học được rất dễ overfit trên
mẫu nhỏ, và nó không phải nguồn edge — vi cấu trúc mới là (P2.3).

---

# P2 — NÂNG CẤP CÔNG NGHỆ

Xếp theo **tỷ lệ lợi ích / công sức**, không theo độ "hiện đại".

## P2.1 — Lệnh maker thay taker ⭐ ROI CAO NHẤT
0.02% → 0.01%… hoặc taker 0.04% → maker 0.01% = **giảm chi phí tới 4 lần**.
Ở quy mô vốn nhỏ, điều này đáng giá hơn *bất kỳ* cải tiến model nào.
`execution/limit_queue_sim.py` đã có sẵn để mô phỏng vị trí hàng đợi.

Việc cần làm: đặt `POST_ONLY` limit tại/gần BBO, có timeout rồi mới rơi về market;
mô hình hoá rủi ro không khớp trong backtest (đây là chi phí thật của maker).

## P2.2 — Thay model bằng gradient boosting
`WeightedBootstrapForestClassifier` → LightGBM/XGBoost có `sample_weight`.
Hiệu chỉnh tốt hơn, nhanh hơn, chuẩn ngành cho dữ liệu bảng.
**Giữ nguyên** sample-weighting kiểu AFML (`labeling/sample_weights.py`) — phần đó đúng.

## P2.3 — Feature vi cấu trúc ⭐ NƠI CÓ EDGE THẬT
Ở quy mô nhỏ, edge nằm ở vi cấu trúc chứ không ở chế độ thị trường:
- Order flow imbalance (đã có `data/bars/tick_rule_ofi.py`)
- Mất cân bằng độ sâu sổ lệnh (`governance/l2_depth.py` — đang rỗng)
- Funding rate + basis (perp vs spot) — tín hiệu riêng của futures
- Tự tương quan dấu lệnh

## P2.4 — Kiểm định làm cho đúng ⭐ GIÁ TRỊ HƠN MỌI NÂNG CẤP MODEL
Đây là chỗ hầu hết chiến lược chết. Hoàn thiện `validation/`:
`flat_plateau.py`, `structural_break_cusum.py` (đang rỗng), PBO/CSCV.
Một chiến lược qua được PBO < 50% đáng giá hơn mười model tinh vi.

---

# ⭐ KẾT QUẢ NGHIÊN CỨU (đã chạy trên dữ liệu thật)

## Directional một tài sản: KHÔNG có edge
12 cấu hình (3 khung x 2 chiều x 2 mức phí), 6 năm BTC thật, funding đã trừ,
đã gộp trùng lặp CPCV. Tốt nhất: 4h/follow/maker +0.205%/lệnh, **t-stat 1.29**.
DSR sau chiết khấu = 0. Mean-reversion **lỗ có ý nghĩa** (t = -2.0 đến -4.5).
=> Đừng tối ưu tiếp hướng này.

## Cross-sectional market-neutral: CÓ EDGE
`research/cross_sectional.py` + `data/panel.py`. 61 cặp perp, 6 năm, 4h, phí maker.

| Tín hiệu | ann | Sharpe | t-stat |
|---|---|---|---|
| funding_carry | +24.0% | 0.87 | 2.14 |
| momentum_90 | +45.0% | 1.54 | 3.78 |
| ofi_flow | +35.7% | 1.43 | 3.51 |
| **chia đều vốn** | **+34.9%** | **1.93** | **4.74** |

Tương quan giữa tín hiệu chỉ 0.02-0.29 -> bổ trợ mạnh. Dương **7/7 năm**.
Tập trung `top_frac=0.10` (12 vị thế): **+54.9%/năm, Sharpe 1.71**, hợp vốn nhỏ
($9.50/lệnh ở 3x, trên min notional $5).

**Vì sao hướng này thắng**: khử beta thị trường. Directional phải thắng nhiễu của
toàn thị trường; cross-sectional chỉ cần đúng THỨ HẠNG TƯƠNG ĐỐI.

## Bài học THỰC THI (chỉ lộ ra khi chạy sàn thật)
1. **Lệnh maker không đảm bảo khớp.** Lần chạy đầu: 9/12 lệnh nằm chờ 0% khớp,
   danh mục lệch **33% khỏi trung lập** — market-neutral khớp một nửa KHÔNG còn
   trung lập, nó thành cược có hướng. Giải: `execute_with_fallback` (thụ động
   trước, cắn giá sau) -> lệch hướng còn **0,1-0,5%**.
2. **Đặt lệnh ở mark price bị từ chối `-5022`.** Mark thường nằm TRÊN ask. Phải
   neo vào BBO thật + đệm 1 tick, kèm thử lại lùi giá dần.
3. **Kill switch chạy dry-run là vô dụng** — nút dừng khẩn cấp chỉ giả vờ đóng
   còn nguy hiểm hơn không có nút nào. Nay `--kill` luôn thật.
4. **Chi phí thật gấp ~3x giả định** (2,87bp vs 1,00bp). Đo bằng
   `core/execution_log.py`; chiến lược vẫn sống ở mức 4bp (holdout Sharpe 0,96).
5. **Vùng đệm thứ hạng** giảm turnover 19% và drawdown 26,2%->21,3% mà không mất
   Sharpe. Dải-không-giao-dịch theo trọng số KHÔNG dùng được (trọng số nhị phân).

## Việc còn lại
Execution cross-sectional: `live_pipeline` hiện là single-asset directional, KHÔNG
dùng lại được. Cần portfolio rebalancer 12 vị thế + OMS (`oms/*.py` vẫn rỗng).

# ═══ CẬP NHẬT v3 (2026-09-10) — thay thế phần kỳ vọng bên dưới ═══

Báo cáo đầy đủ: `docs/upgrade_v3_report.md`. Tóm tắt điều đã đổi:

**Con số cũ trong file này (Sharpe 1.71-2.36) dùng giả định phí 1bp/chiều.**
Chi phí thật đo từ sổ lệnh Binance là **4-5bp/chiều**. Sau khi tính đúng, cộng với
kiểm định holdout, kỳ vọng hợp lý là:

    Sharpe 0.7-1.1 | 35-55%/năm ở gross 1.0x | drawdown 25-40%

**Nâng cấp DUY NHẤT được xác nhận ngoài mẫu: mở rộng universe** (59 -> 127 cặp).
Sharpe holdout của universe rộng dương ở mọi cấu hình (0.34-0.82); universe hẹp có
cấu hình âm (-0.21). Đã tải 170 cặp khung 1h.

**Các hướng đã thử và KHÔNG có tác dụng** (đừng thử lại): MVO, khử beta, trọng số
liên tục sau khi cố định số vị thế, tăng tần suất tái cân bằng, gộp theo IC, tổ hợp
đa cấu hình. Lý do từng cái ở §8 báo cáo.

**P2.1 (lệnh maker) vẫn đúng nhưng nhỏ hơn tưởng:** từ maker 38% lên 85% đáng
+0.07 Sharpe trên holdout (2.76bp so với 5.15bp/chiều). Vẫn nên làm, chỉ đừng kỳ
vọng nó cứu được chiến lược.

**P2.3 (vi cấu trúc) là hướng còn lại duy nhất để tăng độ rộng.** Tăng tần suất trên
bộ tín hiệu hiện tại đã được kiểm chứng là phản tác dụng (chi phí 27%/năm ở nến 1h).

**Việc tiếp theo bắt buộc:** giao dịch giấy tiến về phía trước 4-8 tuần. Holdout đã
dùng 2 lần, không còn giá trị chứng cứ.

# Về kỳ vọng lợi nhuận (bản gốc — đọc phần cập nhật ở trên trước)

Vốn 1.000.000 VND ≈ **$38**. Mục tiêu 100k/tuần = **10%/tuần = 142 lần/năm** khi cộng dồn.

Hệ thống định lượng crypto thuộc nhóm tốt nhất chạy Sharpe 1.5–2.5 ≈ **0.5–1%/tuần**.
Mục tiêu này cao hơn 10–20 lần. Đạt được trong một tuần may mắn thì có; duy trì thì không —
kết cục *trung vị* của chiến lược đòn bẩy cao là cháy tài khoản trong vài tuần.

**Ràng buộc thật là vốn, không phải thuật toán.** Với $38:
- Min notional Binance futures: BTCUSDT ~$100 (không vừa), ETHUSDT ~$20, phần lớn alt ~$5
  → **phải chọn cặp có min notional $5**, nếu không Kelly sizing mất hết ý nghĩa
- Phí khứ hồi taker 0.08% — với maker (P2.1) giảm còn ~0.02%

**Thứ tự đúng:** P0 → paper trên dữ liệu THẬT 2–4 tuần → đo edge thật →
chỉ khi edge dương đã chứng minh mới nạp tiền, và nạp đủ để sizing có ý nghĩa.
Tiền đến từ việc mở rộng vốn trên edge đã chứng minh, không từ siết đòn bẩy trên $38.


# ═══ CẬP NHẬT 14/09/2026 — MỤC TIÊU CÓ HẠN CHÓT ═══

Mục tiêu "vốn 1.000.000 VND, lời 50.000 VND trong 7 ngày" KHÔNG phải bài toán tối đa
Sharpe và cũng không phải Kelly. Nó là bài toán **tối đa P(chạm đích trước hạn chót)**,
và lời giải của nó khác hẳn: đòn bẩy tối ưu phụ thuộc VỐN HIỆN TẠI và THỜI GIAN CÒN
LẠI, không phải một hằng số.

Công cụ: `research/goal_dp.py` (lý thuyết + quy hoạch động), `risk/goal_overlay.py`
(đưa vào live), `scripts/goal_plan.py` (trả lời), `scripts/build_goal_policy.py` (giải
sẵn ra artifact).

## Ba con số cần nhớ

| | P(đạt +5% trong 7 ngày) |
|---|---|
| Trần dạng đóng với đòn bẩy CỐ ĐỊNH | **43,6%** (tại L\* = 3,55x) |
| Đòn bẩy 2,0x đang chạy | 38,5% |
| Chính sách DP, chấm trên holdout | **77,8%** |

`P_max_const = Phi( S*sqrt(T) - sqrt(2*ln(1+g)) )`. Số hạng thứ hai KHÔNG chứa S:
cái giá của mục tiêu là hằng số trừ thẳng vào z-score. **Tăng đòn bẩy không bao giờ
vượt được 43,6%** — nó nhân cả mu lẫn sigma.

## Điều quan trọng nhất, và dễ tự lừa nhất

Trong 77,8% đó, **72,9 điểm là HÌNH HỌC** (đích +5% gần hơn sàn cháy -30% rất nhiều)
và chỉ **4,8 điểm là do EDGE**. Kiểm chứng: chấm lại chính sách trên chuỗi holdout đã
TRỪ TRUNG BÌNH (edge = 0, giữ nguyên biến động và cụm) vẫn ra 72,9%.

Nghĩa là chiến lược này biến một trò chơi gần công bằng thành "thắng nhỏ 73% số lần,
thua lớn 15% số lần" — hình dạng của việc BÁN BẢO HIỂM. Kỳ vọng ở kịch bản edge=0 là
+0,17%, tức bằng 0 trong sai số. Xác suất thắng cao KHÔNG đồng nghĩa kỳ vọng dương.

## Hai con số bị đo sai trước đây, nay đã sửa

* **maxDD thật là 45,5%, không phải 36,9%.** Lưới 72h chỉ thấy giá đóng ba ngày một
  lần nên không nhìn thấy sụt giảm trong kỳ — nhưng sàn, margin và van drawdown chạy
  mỗi 30 phút thì thấy. Dùng `strategy_v3.run_v3_fine`.
* **Sharpe ở độ phân giải sàn thật sự nhìn thấy là 1,10**, không phải 1,28.

## Ràng buộc đặt được lệnh — chỗ lý thuyết gãy nếu bỏ qua

Bản DP đầu tiên chọn trung bình 1,27x, tương đương **$4,01/vị thế — dưới min notional
$5**, tức tối ưu trên giấy mà sàn từ chối từng lệnh. Lưới hành động phải là
`{0} ∪ [1,58x, 5x]`: "đóng sạch đứng ngoài" đặt được, "0,25x" thì không.
Sau khi sửa, chính sách đứng ngoài **65% thời gian** và chạy **3,76x khi vào** = $11,92
mỗi vị thế. Đó là lời giải "bạo phát" kinh điển: không cược thì thôi, cược thì cược lớn.

# ═══ 14/09/2026 — CHIẾN DỊCH TÌM SHARPE: 5 HƯỚNG ĐÓNG, 1 HƯỚNG XÁC NHẬN ═══

## Con số neo mọi thứ

Tốc độ tăng trưởng BỀN VỮNG tối đa ở đòn bẩy Kelly là `S²/2` mỗi năm. Đảo ngược:

| muốn mỗi tuần | cần Sharpe năm |
|---|---|
| 1,0% (10.000 VND) | 1,02 |
| 2,0% (20.000 VND) | 1,44 |
| 3,0% (30.000 VND) | 1,75 |
| **5,0% (50.000 VND)** | **2,25** |

Sharpe holdout đo ở lưới 4h (độ phân giải sàn nhìn thấy) là **1,10** -> bền vững
**~11.700 VND/tuần**. Muốn 50.000 VND/tuần phải NHÂN ĐÔI Sharpe. Đó là toàn bộ cuộc
chơi; mọi thứ khác chỉ là hệ quả.

## Đã thử và KHÔNG có tác dụng — ĐỪNG LÀM LẠI

| hướng | kết quả | vì sao |
|---|---|---|
| **Độ rộng** `n_positions` 8→40 | Sharpe không tăng; n=12 tốt nhất về độ ổn định (100% fold dương, tệ nhất +0,30) | Tín hiệu nằm ở HAI CỰC của mặt cắt ngang. Pha loãng vào hạng 13-40 là pha loãng edge: vol giảm đơn điệu nhưng lợi suất giảm nhanh hơn. Thêm nữa, n cao ÉP gross cao (n=40 cần 5,26x) — đúng lúc không nên. `scripts/breadth_study.py` |
| **Mục tiêu biến động + van drawdown** | Sharpe 1,74 -> 1,75. Van drawdown luôn làm tệ đi | `zscore_riskparity` ĐÃ chia đều rủi ro theo biến động ở cấp TÀI SẢN. Chồng thêm tầng cấp danh mục là thừa. `scripts/risk_overlay_study.py` |
| **Mở rộng bể chọn** (`min_coverage` 0,15→0,10, 127→152 cặp) | Trung vị fold 1,61→1,91 NHƯNG toàn bộ mức tăng nằm ở fold CUỐI (25/11: 1,66→2,23), mọi fold khác ±0,10 | Hạ ngưỡng chỉ nạp thêm cặp MỚI NIÊM YẾT — chúng chỉ tồn tại ở giai đoạn gần đây, và giai đoạn đó nằm trong holdout. Hiệu ứng giai đoạn, không phải cải tiến bền. **Cần kiểm chứng tiến về phía trước mới được áp dụng.** |
| **Cân vị thế theo ĐỘ MẠNH bằng chứng** | Không có quan hệ. Ngũ phân vị bằng chứng YẾU nhất (Q1) lại có Sharpe CAO nhất (2,02 so với 1,26 ở Q5) | Độ mạnh bằng chứng quá khứ không dự báo lợi suất tương lai. Cân theo nó sẽ làm TỆ ĐI. |
| **Vùng đệm thứ hạng** (`exit_frac`) | Sharpe 1,74 -> 1,48 ở `exit_frac=0,15` | ⚠️ Ghi chú cũ "giảm turnover 19% mà không mất Sharpe" là từ cấu hình **v1** (4 tín hiệu, trọng số nhị phân, 61 cặp). KHÔNG chuyển sang v3 được. Turnover giảm 13% thật, chi phí giảm thật, nhưng lợi suất giảm nhiều hơn. **Ai bật theo ghi chú cũ sẽ tự làm hại mình.** |

## Hướng DUY NHẤT xác nhận có tác dụng: CHI PHÍ

| tỷ lệ maker | bp/chiều | Sharpe | chi phí/năm | VND/tuần (nửa Kelly) |
|---|---|---|---|---|
| 0% | 6,81 | 1,67 | 8,79% | 20.416 |
| **0,378 (đo được)** | 4,99 | 1,73 | 6,43% | 21.711 |
| 0,50 (giả định) | 4,41 | 1,74 | 5,67% | 22.137 |
| **0,85 (mục tiêu)** | **2,72** | **1,79** | **3,49%** | **23.384** |
| 1,00 | 2,00 | 1,81 | 2,56% | 23.930 |

**+0,06 Sharpe, +7,7% lợi nhuận.** Thật, chắc chắn, và thực hiện được bằng kỹ thuật
thực thi chứ không cần tín hiệu mới.

## Đã làm để lấy phần đó

1. **Van trung lập trong lúc khớp thụ động** (`order_router.fill_imbalance`). Vòng chờ
   cũ hoàn toàn MÙ về trung lập nên `passive_wait_s` buộc phải ngắn (300s). Nguy hiểm
   thật không nằm ở chờ lâu mà ở KHỚP LỆCH MỘT CHIỀU. Đo trực tiếp độ lệch so với
   đường thực thi theo tỷ lệ -> chờ lâu hơn mà AN TOÀN HƠN bản cũ.
2. **`passive_wait_s` 300s -> 900s.** Trong chu kỳ 72h, 15 phút là 0,3% thời gian.
3. **Chẩn đoán tỷ lệ maker** vào `ExecutionRecord` + `readiness_gate.py`: đếm lệnh hết
   giờ, lần bị trần đuổi giá chặn, độ trôi tối đa. Lượt sạch tới sẽ TỰ TRẢ LỜI "maker
   thấp vì chờ chưa đủ hay vì `max_chase_bps` quá hẹp" — hai nguyên nhân cần hai bản
   vá khác nhau, và đoán sai thì vá sai.

# ═══ 14/09/2026 (phần 2) — HỌ VỊ THẾ: TÍN HIỆU TỐT, TÍCH HỢP THẤT BẠI ═══

## Đã làm

Tải dữ liệu vị thế từ `data.binance.vision/.../metrics` — 5 phút/điểm, từ 2021, miễn
phí, KHÔNG bị giới hạn 30 ngày như endpoint `/futures/data/`. Sáu trường: open interest
(đơn vị + USD), tỷ lệ long/short của top trader (theo số tài khoản VÀ theo vị thế), tỷ
lệ long/short toàn sàn, tỷ lệ khối lượng taker. `scripts/download_metrics.py`.

Dựng 10 tín hiệu trong họ `positioning` (`signal_library.py`), đánh giá bằng
`scripts/positioning_study.py`.

## Từng tín hiệu: TỐT (đo trên TRAIN 2023-02 -> 2024-11, 212 kỳ, 87 cặp)

| tín hiệu | chênh decile | t-stat | Sharpe/năm |
|---|---|---|---|
| `oi_price_confirm` | **+0,473%** | **2,01** | 1,53 |
| `crowd_ls` | **−0,441%** | **−2,27** | 1,78 |
| `crowd_ls_momentum` | **−0,439%** | **−2,04** | 1,62 |
| `smart_count_vs_size` | +0,325% | 1,91 | 1,59 |
| — mốc so sánh: `carry_level` | +0,243% | 1,22 | 0,80 |
| — mốc so sánh: `mom_slow` | +0,568% | 2,80 | 1,42 |

Ba tín hiệu vượt ngưỡng t = 2,0, và cả ba có hiệu ứng LỚN HƠN `carry_level` — một
trong những tín hiệu lõi. Trực giao: |tương quan| lớn nhất với họ cũ chỉ 0,269.

t-stat tăng đơn điệu theo độ phủ mặt cắt ngang: 39 cặp -> 0/10 đạt ngưỡng; 70 -> 1/10;
87 -> 3/10. Tức chúng là tín hiệu thật, chỉ cần đủ độ rộng để đo.

## Đầu-cuối: THẤT BẠI. KHÔNG ĐƯA VÀO.

| bộ tín hiệu | Sharpe | ann | trung vị fold | %fold dương |
|---|---|---|---|---|
| 26 đã kiểm định | **1,77** | 75,0% | 1,02 | 83% |
| 26 + 3 vượt ngưỡng | 1,36 | 56,9% | 0,80 | 67% |
| 26 + 5 mạnh nhất | 1,39 | 58,2% | 1,02 | 67% |
| 26 + cả 10 | 1,36 | 57,3% | 0,98 | 67% |

Mọi tập con đều tệ hơn. Hai giả thuyết đã KIỂM CHỨNG VÀ BÁC BỎ:

* *"Do phủ lệch — cặp có metrics chấm 6 họ, cặp không có chấm 5 họ."*
  Cắt rổ về đúng 87 cặp có metrics: vẫn tệ (0,80 -> 0,63).
* *"Do 7 tín hiệu yếu pha loãng."* Chỉ giữ 3 tín hiệu vượt ngưỡng — chọn TRÊN CHÍNH
  cửa sổ đo, tức thiên vị theo hướng có lợi — vẫn tệ (1,77 -> 1,36).

## Vì sao — cơ chế, không phải phỏng đoán

`combine_adaptive` ràng buộc **gross = 1,0** trên các họ. Thêm họ thứ 6 BẮT BUỘC lấy
trọng số khỏi 5 họ đã chứng minh. Để có lời, họ mới phải đóng góp NHIỀU HƠN phần nó
chiếm chỗ — chứ không chỉ cần "dương khi đứng riêng".

Chênh lệch decile đo thông tin của một tín hiệu KHI ĐỨNG MỘT MÌNH. Tương quan 0,269
với momentum là dấu hiệu: vị thế đám đông phần lớn đã nằm sẵn trong động lượng. Tín
hiệu trực giao 73% vẫn có thể dư thừa CÓ ĐIỀU KIỆN.

## Việc còn để ngỏ (không làm bây giờ)

Dữ liệu và 10 tín hiệu ĐÃ NẰM TRONG REPO và có test. Ba hướng nếu quay lại:
1. Tải đủ 170 cặp (đang dở ~110) rồi đo lại — t-stat tăng đơn điệu theo độ phủ.
2. Gộp ở tầng TÍN HIỆU thay vì tầng HỌ, để một tín hiệu mạnh không bị trần 1/6 của họ.
3. Chiến lược PHỤ riêng chạy song song, thay vì nhét vào bộ gộp hiện có.
