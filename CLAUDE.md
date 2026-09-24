# Aegis Trading System

Hệ thống giao dịch **perpetual futures USDⓈ-M** (KHÔNG phải margin spot).
Python 3.10+, `src/aegis/`, ~20k dòng, 558 test.

## 🆕 Nâng cấp v3 (2026-09-10) — ĐỌC `docs/upgrade_v3_report.md` TRƯỚC KHI SỬA CHIẾN LƯỢC
Ba điều bắt buộc biết trước khi chạm vào code chiến lược:
1. **Chi phí thật là 4-5bp/chiều, không phải 1bp.** Nghiên cứu cũ giả định 1bp và vì
   thế cao hơn thực tế 0.1-0.35 Sharpe. Dùng `backtest_v2.estimate_cost_bps`.
2. **IC và chênh lệch decile có thể NGƯỢC DẤU** khi quan hệ không đơn điệu — đã xảy ra
   thật ở đây. Chọn tín hiệu bằng `ic_analysis.decile_spread`, không bằng IC.
3a. **Basis trade** (`research/basis_trade.py`): edge ĐÃ CHẾT từ 2025 — funding đổi
   dấu (2024: +12,7% -> 2025: −2,2%). Trên 119 cặp, Sharpe holdout trung vị −0,54,
   chỉ 17% cấu hình dương. ĐỪNG bật lại nếu chưa phân rã theo năm. §12.
3b. **ĐÃ THỬ VÀ THUA, đừng làm lại**: hồi quy mặt cắt ngang kiểu CTREND
   (`research/xs_regression.py`) và LambdaRank (`research/rank_model.py`). Cả hai
   thua tầng gộp hiện có vì universe chỉ có ~41 tài sản/kỳ — ràng buộc là ĐỘ RỘNG,
   không phải thuật toán. Chi tiết `docs/upgrade_v3_report.md` §11.
3. **Kỳ vọng hợp lý là Sharpe 1.0-1.3 / 55-75%/năm ở gross 1.0x**, không phải 1.99 của
   train. Holdout v3: Sharpe **1.28**, ann **70.8%**, maxDD 36.9%, giữ 64% Sharpe train.
   Qua 32 cấu hình siêu tham số, holdout **100% dương**, trung vị 1.24.
4. **Đánh giá holdout phải tính NHÂN QUẢ trên toàn dòng thời gian rồi CẮT SAU.** Cắt
   lưới về holdout trước sẽ khởi động lại tầng gộp thích ứng — handicap mà live không
   gặp. Sai lệch đo được: Sharpe 0.71 (sai) so với 1.28 (đúng).

## 🆕 14/09/2026 — MỤC TIÊU CÓ HẠN CHÓT (`research/goal_dp.py`)

Câu hỏi "vốn 1 triệu, lời 50 nghìn trong 7 ngày" là bài toán **tối đa P(chạm đích
trước hạn chót)**, KHÁC hẳn tối đa Sharpe và khác cả Kelly. Kelly không quan tâm hạn
chót; mục tiêu có hạn chót không thưởng thêm xu nào cho việc vượt đích. Hệ quả: đòn
bẩy tối ưu phụ thuộc VỐN HIỆN TẠI và THỜI GIAN CÒN LẠI.

| | P(đạt +5% trong 7 ngày) | ghi chú |
|---|---|---|
| Trần dạng đóng, đòn bẩy CỐ ĐỊNH | **43,6%** | `Phi(S*sqrt(T) - sqrt(2*ln(1+g)))`, đạt tại 3,55x |
| 2,0x đang chạy | 38,5% | |
| Chính sách DP, chấm trên holdout | **77,8%** | ngoài thị trường 65% thời gian, 3,76x khi vào |
| — trong đó do HÌNH HỌC (edge = 0) | 72,9% | đích gần, sàn cháy xa |
| — trong đó do EDGE | **4,8%** | |

**Điều quan trọng nhất: xác suất thắng cao KHÔNG có nghĩa kỳ vọng dương.** Chấm lại
chính sách trên holdout đã trừ trung bình (edge = 0) vẫn ra 72,9%, với kỳ vọng +0,17%
tức bằng 0. Hình dạng của nó là BÁN BẢO HIỂM: thắng nhỏ thường xuyên, thua lớn hiếm
khi (P(cháy -30%) = 11,4%).

**Hai con số cũ bị đo sai, nay đã sửa:**
- **maxDD thật 45,5%**, không phải 36,9% — lưới 72h không thấy sụt giảm trong kỳ.
- **Sharpe ở độ phân giải sàn nhìn thấy là 1,10**, không phải 1,28.

Tầng phủ live (`risk/goal_overlay.py`) mặc định **`derisk_only=True`** — chỉ được
GIẢM rủi ro. Phần "bạo phát" của DP (tăng đòn bẩy khi đang thua và sắp hết giờ) đúng
về toán nhưng nguy hiểm nhất đúng lúc đường ống xấu nhất, nên phải bật tường minh sau
khi `readiness_gate.py` mở.

## 🚫 24/09/2026 — "+20% MỖI TUẦN, BẰNG MỌI CÁCH": ĐÃ ĐO. KHÔNG BẢO ĐẢM ĐƯỢC. `scripts/weekly_target_study.py`

+20%/tuần gộp lại là **×13.105/năm**, và muốn nó BỀN VỮNG thì cần Sharpe **4,35**
(tăng trưởng Kelly = S²/2). Đang có: 1,10 holdout, 1,34 toàn lịch sử, 2,23 in-sample
kỷ nguyên rộng. Đòn bẩy không thay được Sharpe. Nó nhân lãi tuyến tính nhưng nhân rủi
ro bậc hai.

Đòn bẩy cố định, cơ sở THẬN TRỌNG, có thanh lý giữa tuần:

| đòn bẩy | P(+20% trong 1 tuần) | trung vị sau 52 tuần | P(cháy trong năm) |
|---|---|---|---|
| 5x (daemon) | 15% | **+1,9%/tuần** | 2,5% |
| 7,89x | 25% | +1,0%/tuần | 8,5% |
| 10x | 30% | −0,8%/tuần | 19% |
| 20x | 37% | **cháy** | **82%** |

Càng đẩy đòn bẩy thì xác suất có MỘT tuần +20% càng tăng, nhưng số tiền cuối năm lại
giảm. Ở 20x, lãi TRUNG BÌNH tuần là +18% (trông như đạt mục tiêu), còn trung vị sau
một năm là 0.

**Chính sách DP tối ưu cho đích +20%** (`solve_goal_dp`, trần 20x, cơ sở lạc quan)
đạt 58–66% trong MỘT tuần. Nhưng khi edge = 0 nó vẫn đạt **48–54%**, tức edge chỉ góp
khoảng 11 điểm, phần còn lại là bán bảo hiểm. Đặt sàn −30% thì **27% số tuần chạm
sàn**. Đòi đạt MỌI tuần thì xác suất là 11–19% cho 4 tuần, <1% cho 12 tuần, và dưới
1e-9 cho 52 tuần.

⚠️ Số liệu này đo trên lợi suất TỔNG HỢP hiệu chỉnh theo thống kê đã đo (đuôi −5,4σ so
với −5,19σ thật), vì container không có dữ liệu thật. Kết luận không phụ thuộc vào
chi tiết: phần dạng đóng (Sharpe cần có, trần xác suất) không dùng mô phỏng.
**Đừng nâng đòn bẩy vượt khoảng [4..6]x để đuổi mục tiêu tuần.** Hai đòn bẩy có thật
là VỐN (nhân tiền tuyến tính, không bị phạt) và tỷ lệ maker (F49, +21%/năm ở 7,89x).

## 🔁 24/09/2026 — CHIA LÔ TÁI CÂN BẰNG (chống timing luck) + F53. `research/tranching.py`

**F53 — live giao dịch tín hiệu cũ tới 68 giờ. ĐÃ VÁ (có hiệu lực khi NẠP LẠI daemon).**
`_compute_target_weights_v3` dựng lưới `close.index[::18]` neo ở ĐẦU panel rồi lấy hàng
cuối của lưới. Mốc đó có thể cách nến mới nhất tới 17 nến 4h, nhưng hàm vẫn trả mốc nến
mới nhất nên STALE_DATA không thấy gì. Test parity cũ XANH vì phía research cũng dựng
lưới y hệt: hai bản sao của cùng một lỗi. Độ cũ phụ thuộc lúc máy thức dậy, và đây
nhiều khả năng là "độ trễ tín hiệu thay đổi theo giờ" mà replay thấy. Nay lưới neo ở
NẾN MỚI NHẤT (`anchored_marks`).

**Chia lô** (Hoffstein-Faber-Braun 2020). Sổ là trung bình K lô; mỗi lô vẫn giữ 72h
như chiến lược đã kiểm định, nhưng các lô lệch nhau 72/K giờ. Kết quả tiến về TRUNG
BÌNH các giờ bắt đầu, thay vì phụ thuộc một giờ may hay rủi (replay: −14% đến +231%
chỉ do giờ). Hàm là THUẦN nên không cần lưu trạng thái từng lô: đích của lô j tính
lại được từ dữ liệu. Live và backtest gọi CÙNG `tranched_target_weights` /
`tranched_weight_panel`, và test khoá hai đường trùng nhau tới 1e-9, ở K=1 lẫn K=3,
không cần dữ liệu thật (`tests/research/test_tranching.py`,
`tests/pipelines/test_tranching_live_f53.py`).

- `LiveConfig.n_tranches` và `StrategyV3Config.n_tranches` đọc từ khoá `"n_tranches"`
  trong JSON. Nhịp cả hệ = `period_hours / n_tranches`. Test F51 khoá parity.
- `run_v3_fine` tự chia lô theo cấu hình. `run_v3` (lưới 72h, một lô) **ném lỗi** khi
  gặp cấu hình chia lô, thay vì âm thầm đo sai. Script nào cố ý đo một lô phải
  `replace(cfg, n_tranches=1)` tường minh.
- Chia lô thêm cặp (hợp của K lô), nên ở vốn $38 sẽ dưới min notional và bị cắt theo
  F43. Chỉ hợp lý từ vốn khoảng vài trăm USD trở lên; testnet $5.000 thì thoải mái.

⚠️ **CHƯA BẬT trong `strategy_v3_wide.json`.** Sửa file cấu hình daemon đang chạy là
thay đổi production, nên phải do người vận hành làm. Cách bật: thêm
`"n_tranches": 3` vào khối `"config"`, rồi nạp lại daemon. Nên chạy trước:

```bash
python scripts/tranche_study.py --cost-bps 15.7       # 18 pha một lô vs K=3/6/18, luật chốt trước
python scripts/recent_backtest.py --cost-bps 15.7 --tranches 3   # 28 ngày của hệ thống chia 3 lô
```

## 🎯 24/09/2026 — REPLAY LIVE 28 NGÀY: −4,1% Ở GIỜ DAEMON, NHƯNG GIỜ BẮT ĐẦU QUYẾT ĐỊNH TẤT CẢ

Replay gọi ĐÚNG code live (`xs_live_pipeline`: trọng số, ngắt mạch, giảm đòn bẩy, lập
lệnh), point-in-time (mỗi quyết định chỉ thấy dữ liệu có trước nó, universe lọc lại
tại chỗ). Vốn $5.000, 5x, chi phí **15,7bp/chiều đo từ lệnh khớp thật trên testnet**.
Kiểm chứng: ngày 23/09 trùng 48/49 cặp cùng hướng với sổ thật. Từ 19/09, replay ra
−12,6% còn tài khoản thật −17,1%; khoảng chênh ≈ lệnh không khớp + lệch trung lập.
⚠️ `replay_live.py` chạy trong phiên cục bộ và **CHƯA được commit**.

| giờ daemon (05:00 UTC), funding đóng băng | 28 ngày | 14 ngày | 7 ngày |
|---|---|---|---|
| lời/lỗ | **−4,1%** | −1,5% | **−12,9%** |

maxDD −21,7%. Chạy lại cùng hệ thống ở 61 giờ bắt đầu khác nhau, 28 ngày:

| | trung vị | 10% tệ / tốt | tệ nhất / tốt nhất | % có lời |
|---|---|---|---|---|
| như đang chạy | +34,8% | +1,8% / +75% | −16% / +175% | 92% |
| đã vá funding | +45,2% | +7,8% / +110% | −14% / +231% | 95% |

Giờ của daemon rơi vào **nhóm 5% xấu nhất**, và sau khi vá funding thì giờ đó ra −7,3%.
Cải thiện +10 điểm trung vị nằm trong nhiễu. **Một tháng không đánh giá được hệ thống**:
cả tháng dựa vào tuần 10–17/09 (+33%), còn tuần gần nhất chỉ 16% số giờ có lời.

**Năm phát hiện, và trạng thái:**
1. **F52 — funding đứng yên từ 10/09. ĐÃ VÁ** (xem dưới). Phải NẠP LẠI daemon.
2. **Chi phí thật 15,7bp/chiều, gấp ~3,5 lần mô hình.** Mô hình thiếu khoản trôi giá
   trong cửa sổ chờ 900s. Đã thêm `CostModel.flat_bps` và cờ `--cost-bps` cho
   `recent_backtest`, `residual_study`, `leverage_study`; mặc định không đổi. Ước lượng
   THÔ (giả định turnover ~1,1 lần gross mỗi kỳ, tức ~134/năm): chi phí thêm khoảng
   15%/năm ở 1x. Theo đó Sharpe kỷ nguyên ≥100 cặp từ 2,23 còn ~1,6; toàn lịch sử từ
   1,34 còn ~0,8; Kelly toàn lịch sử từ ~4,7x còn ~2,9x. **Khoảng đòn bẩy "4–6x" đo ở
   4,5bp, ở chi phí thật có thể không còn đứng.** Chạy
   `leverage_study.py --cost-bps 15.7`. Lưu ý 15,7bp đo trên testnet (sổ mỏng), nên
   mainnet có thể thấp hơn.
3. **Ngắt mạch tính trên EQUITY, không theo đòn bẩy. CHƯA ĐỔI, cần người quyết.**
   10/20/30% ở 5x chỉ tương đương 2/4/6% sụt giảm chiến lược, trong khi maxDD 1x lịch
   sử là 45,5%. Replay: sụt giảm trung vị trong tháng −26%, 30% số giờ bắt đầu chạm
   đóng băng, kill switch ở rất gần. TIER1 (cắt nửa vị thế ở −10%) gần như luôn bật,
   mà repo đã đo "van drawdown luôn làm tệ đi". Có ba lựa chọn: (a) quy ngưỡng theo
   đòn bẩy, (b) hạ đòn bẩy, (c) giữ nguyên và chấp nhận dừng thường xuyên.
   Sổ thật lúc 23/09 +6h: 6.088 → 5.082, tức **DD 16,5%, còn 3,5 điểm tới đóng băng**.
4. **Timing luck.** Giờ tái cân bằng làm kết quả 28 ngày chạy từ −16% tới +175%.
   KHÔNG được chọn "giờ tốt nhất" từ một tháng, vì đó là overfit. Chia lô tái cân bằng
   giảm được nó nhưng cần vốn để vượt min notional.
5. **Vốn 1 triệu VND (~$38) chưa replay.** Ở 5x chỉ nuôi được khoảng 30 vị thế, nên
   cắt theo F43 và kết quả sẽ khác bảng trên.

### F52 — funding đứng yên 14 ngày, daemon vẫn giao dịch như thường. ĐÃ VÁ.

Ba mắt xích cùng hỏng:
- `refresh_data` chỉ tải NẾN.
- `download_wide_universe.py` chỉ ghi file funding khi file CHƯA có.
- `load_funding_panel_v2` ffill không giới hạn, nên 5/26 tín hiệu carry thấy con số từ
  10/09 như số mới.

Cổng STALE_DATA chỉ nhìn nến, nên cả 14 ngày không có cảnh báo nào.

**Vá:**
- `save_funding_rates()` gộp an toàn.
- `refresh_data` tải funding mỗi lượt, với `try` riêng để funding hỏng không kéo nến hỏng theo.
- Cổng **`STALE_FUNDING`** cho engine v3: đo tuổi funding ở NGUỒN (`funding_last_ms`,
  trước khi ffill), trung vị quá `max_funding_age_hours = 24` thì HALT có lý do. Thiếu
  sạch file funding cũng HALT, vì đó là trường hợp họ carry tắt câm.
- Khoá bằng `tests/pipelines/test_funding_refresh_f52.py` (9 test).

## 🔴 F51 (24/09/2026) — DAEMON CHẠY MỘT CẤU HÌNH CHƯA TỪNG ĐƯỢC BACKTEST

`strategy_v3_wide.json`, file daemon chạy (lượt 23/09 có 50 vị thế), ghi tầng gộp
`max_step = 0,025`. **Mọi** nghiên cứu cấu hình wide (`breadth_beyond_50`,
`sizing_mode_study`, `maker_ratio_study`, `leverage_study`) lại tự dựng
`PortfolioSpec` n=50 nhưng GIỮ tầng gộp của `V3`, tức `max_step = 0,05`.
(`residual_study` từng mắc cùng lỗi, nay đã dựng từ `config_from_json`.) Vì vậy
mọi Sharpe 2,23–2,26 báo cho "cấu hình đang chạy" đều đo một cấu hình KHÁC, và không
có ghi chép nào giải thích vì sao giá trị 0,025 có mặt.

Khe hở lọt qua vì test parity cấu hình chỉ kiểm `strategy_v3.json` (n=12), tức file
daemon không còn chạy. Thêm nữa, test đó nằm trong module bị skip TOÀN BỘ khi thiếu
`data/`. Cùng họ F38: công thức thì khoá, cấu hình thì không.

**Vá phía đo, CHƯA đổi phía live:** `strategy_v3.config_from_json()` dựng cấu hình
nghiên cứu từ ĐÚNG file JSON, bằng đúng phép ánh xạ của `LiveConfig.from_artifacts`
(kể cả việc live bỏ qua `beta_neutral`/`vol_window` trong JSON). Test parity mới
`tests/pipelines/test_config_parity_wide_f51.py` chạy KHÔNG cần dữ liệu.
`run_v3_fine` nay lọc tín hiệu theo cấu hình (phần F39 còn sót).

⚠️ **Quyết định còn treo:** nên giữ 0,025 hay về 0,05 thì phải để
`recent_backtest.py` (mục 3) trả lời trên dữ liệu thật. Đổi JSON là đổi đường tiền,
nên phải NẠP LẠI daemon.

```bash
python scripts/recent_backtest.py --days 28 --leverage 5.0   # ⭐ lời lỗ cấu hình LIVE + so 0,025/0,05
```

## 📉 24/09/2026 — LÃI/LỖ THẬT (testnet) VÀ CHI PHÍ KHỚP LỆNH CHƯA TỪNG ĐƯỢC ĐO

Nguồn là các ảnh chụp equity trong `artifacts/execution_log.jsonl`, chụp ở ĐẦU mỗi
lượt. Chỉ có 13 ngày dữ liệu, vì v3 chạy từ 10/09.

| giai đoạn | equity | thay đổi | cấu hình |
|---|---|---|---|
| 10/09 → 13/09 (code còn lỗi F21-F31) | 5.112 → 6.033 | +18,0% | có sửa tay, từng chạy 3,67x; riêng 11/09→13/09 **+17,5% trong 40h, chưa rõ có phải nạp tiền testnet** |
| 13/09 → 19/09 (code đã vá) | 6.033 → 6.088 | +0,9% | n=12, ~2x |
| 19/09 → 23/09 05:29 | 6.088 → 5.499 | −9,7% | n=50, 7,62x gross |
| 23/09 05:29 → +6h | 5.499 → 5.082 | −7,6% | n=50, 4,89x |
| **code đã vá, cộng dồn** | **6.033 → 5.082** | **−15,8%** | |

**13 ngày KHÔNG nói được gì về edge.** Ở 5-7,6x, σ của 4 ngày vào khoảng 17-22%, nên
−16,5% chỉ là khoảng −0,8σ. Muốn t = 2 thì cần 0,8 năm (nếu Sharpe là 2,23) hoặc 2,2
năm (nếu Sharpe là 1,34). Thứ đo được sớm là CHI PHÍ, và chi phí khớp lệnh chưa từng
được đo: `ExecutionRecord` không ghi giá khớp so với giá lúc quyết định. Lượt 23/09
mới ghi nhận phí $16, trong khi equity mất $417 trong 6h.

```bash
python scripts/pnl_report.py --days 28     # sổ kế toán sàn: tách NẠP TIỀN khỏi lãi thật
python scripts/shortfall_report.py         # ⭐ chi phí TRỄ + KHỚP + PHÍ so với 4,5bp backtest
```

`shortfall_report.py` (7 test ở `tests/ops/test_shortfall_report.py`) lấy giá tham
chiếu từ CÙNG sàn với lệnh. **TRỄ** = giá lúc quyết định so với giá đóng nến 4h, tức
giá backtest giả định khớp. **KHỚP** = giá khớp so với giá lúc quyết định, tách
maker/taker. Nếu tổng vượt xa 4,5bp thì mọi Sharpe backtest đều lạc quan, và thực thi
là đòn bẩy lợi nhuận số 1, đứng trước mọi tín hiệu mới.

⚠️ Container cloud không kết nối được sàn (proxy trả 403 cho `data.binance.vision`,
`fapi.binance.com`, `testnet.binancefuture.com`). Hai script trên chạy trên máy Mac,
hoặc trong session mới sau khi mở Network access và thêm khoá API chỉ-đọc vào biến
môi trường.

## 🔬 24/09/2026 — ĐỘNG LƯỢNG PHẦN DƯ + KHỬ BETA ở n=50: ĐÃ CÀI, **CHƯA ĐO TRÊN DỮ LIỆU THẬT**

```bash
python scripts/residual_study.py     # ⭐ chạy trên máy có data/ (~10-15 phút)
```

**Rà tài liệu trước khi chọn.** Lợi suất crypto cao nhất đã công bố là CTREND
(+3,87%/tuần) và momentum Liu-Tsyvinski-Wu (+4,2%/tuần), cả hai ở 1x và trên dữ liệu
thời kỳ đầu. Momentum crypto thất bại toàn bộ trong 2022-2023. CTREND và LambdaRank đã
thử và thua ở đây (§11 báo cáo v3), vì ràng buộc là ĐỘ RỘNG. Vì vậy chỉ chọn cơ chế
**không cần mặt cắt ngang rộng**:

| hướng | nguồn | vì sao chọn / loại |
|---|---|---|
| **Động lượng phần dư** `rmom_*` | Blitz-Huij-Martens 2011; Blitz et al. 2013 | ~2 lần Sharpe động lượng thô ở cổ phiếu, giữ được ngoài mẫu. Ở crypto động lượng thô xếp theo `beta × thị trường`, nên sổ trung lập đô-la vẫn ngầm cược thị trường |
| **Khử beta ở n=50** | — | §8 báo cáo v3 loại nó (−0,04) **ở n=12**, lý do "12 vị thế thì ràng buộc beta đòi dịch trọng số quá nhiều". Lý do đó không còn ở n=50 |
| chia lô tái cân bằng | Hoffstein-Faber-Braun | **loại**: sinh lệnh nhỏ, dưới min notional ở vốn $38 |
| giao dịch từng phần | Gârleanu-Pedersen 2013 | **loại**: cùng lý do, và để lại vị thế vụn |

Test khoá **cơ chế** (`tests/research/test_residual_momentum.py`, 9 test): trên panel
chỉ có beta, động lượng thô tương quan hạng 0,60–0,97 với beta, còn phần dư chỉ
0,00–0,07. Ngoài ra phần dư vẫn giữ được alpha riêng thật, và có test nhân quả khi sửa
20 nến tương lai.

**Luật quyết định CHỐT TRƯỚC khi chạy** (in ra trong script): thắng khi đủ cả ba điều
kiện. (1) ΔSharpe ≥ +0,10 ở CẢ HAI cơ sở. (2) Bootstrap ghép cặp cho P(hơn) ≥ 90% ở
kỷ nguyên ≥100 cặp. (3) Fold tệ nhất không tệ hơn. Thắng rồi vẫn phải giao dịch giấy
tiến về phía trước, vì kỷ nguyên ≥100 chồng lên holdout.

⚠️ `--synthetic` chỉ chứng minh script chạy hết đường. Panel giả được dựng sẵn cấu
trúc beta + alpha bền, nên phần dư thắng là đương nhiên. **Đừng trích số của nó.**

⚠️ Họ `residual_momentum` cố ý KHÔNG nằm trong `FAMILIES` hay `V3_SIGNALS`.

**F50** — `validate_v3.py`, `stability.py`, `diagnose_oos.py`, `capital_reality.py`,
`validate_ensemble.py` dựng tín hiệu bằng cách duyệt CẢ `SIGNAL_REGISTRY`, tức đang
chạy 36 tín hiệu chứ không phải 26. Chưa gây hại vì 10 tín hiệu positioning toàn NaN.
Nhưng thêm bất kỳ tín hiệu có giá trị nào sẽ âm thầm đổi kết quả của cả năm script.
Nay chúng duyệt `V3_SIGNALS`, và có test chặn script nào duyệt lại registry.

## 🆕 14/09/2026 — HỌ TÍN HIỆU THỨ 6: VỊ THẾ (open interest + long/short)

26 tín hiệu cũ đều dựng từ GIÁ, KHỐI LƯỢNG, FUNDING. Không cái nào thấy AI ĐANG CẦM GÌ.
Với perp, vị thế là biến trạng thái quan trọng nhất mà giá không chứa: squeeze và thanh
lý dây chuyền sinh ra từ việc quá nhiều người đứng cùng một bên với đòn bẩy.

Nguồn: kho dump `data.binance.vision/.../metrics` — 5 phút một điểm, từ 2021, miễn phí,
KHÔNG bị giới hạn 30 ngày như endpoint `/futures/data/`. Tải bằng
`scripts/download_metrics.py` -> `data/binance_metrics/`.

**KẾT QUẢ: tín hiệu TỐT, tích hợp THẤT BẠI. KHÔNG đưa vào V3.**

3/10 tín hiệu vượt ngưỡng t = 2,0 và cả ba có hiệu ứng LỚN HƠN `carry_level` (tín hiệu
lõi): `oi_price_confirm` (+0,473%, t=2,01), `crowd_ls` (−0,441%, t=−2,27),
`crowd_ls_momentum` (−0,439%, t=−2,04). Trực giao (|corr| ≤ 0,269).

Nhưng thêm vào thì Sharpe đầu-cuối **1,77 -> 1,36**. Mọi tập con đều tệ hơn, kể cả khi
chỉ giữ 3 cái vượt ngưỡng và chọn chúng TRÊN CHÍNH cửa sổ đo (thiên vị có lợi).

Cơ chế: `combine_adaptive` ràng buộc gross = 1,0 trên các HỌ, nên họ thứ 6 bắt buộc lấy
trọng số khỏi 5 họ đã chứng minh. Chênh decile đo thông tin khi tín hiệu ĐỨNG RIÊNG;
trực giao 73% vẫn có thể dư thừa CÓ ĐIỀU KIỆN. Chi tiết + 2 giả thuyết đã bác bỏ:
`references/plan.md`.

⚠️ **V3 bị ĐÓNG BĂNG ở đúng 26 tín hiệu** (`V3_SIGNALS`). `adaptive_weights` chuẩn hoá
theo SỐ họ nên thêm một họ — kể cả họ toàn NaN — vẫn làm dịch trọng số warm-up. Thêm
tín hiệu vào registry KHÔNG được phép âm thầm đổi bản đã kiểm định. Đánh giá bằng
`scripts/positioning_study.py`.

## 💥 18/09/2026 — GẤP ĐÔI TIỀN LỜI: `n_positions=12` ĐƯỢC CHỐT DƯỚI MỘT UNIVERSE KHÔNG CÒN TỒN TẠI

**Kết quả: 20.393 -> 46.811 VND/tuần (2,30x) ở vốn $38 KHÔNG ĐỔI, và ở 0,79 Kelly
so với 0,75 hiện tại — tức không tăng tỷ lệ quá liều.** Cấu hình:
`artifacts/strategy_v3_wide.json` (n_positions 12 -> **50**, max_weight 0,20 -> **0,02**,
đòn bẩy 2,0x -> **7,89x**).

### Lỗi gốc: một con số trung vị bị dùng như một hằng số

Mọi quyết định về độ rộng trong repo đều dựa trên "universe chỉ có ~41 tài sản/kỳ".
Con số 41 là **TRUNG VỊ CỦA TOÀN LỊCH SỬ 2020-2026**, bị kéo xuống bởi những năm perp
crypto còn rất ít cặp:

| 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | **2026** |
|---|---|---|---|---|---|---|
| 20 | 37 | **41** | 51 | 78 | 111 | **127** |

Hệ thống sẽ giao dịch trong môi trường **127 cặp**. Chấm một cấu hình cần độ rộng bằng
trung bình toàn lịch sử là bắt nó gánh những năm mà nó KHÔNG THỂ tồn tại, rồi kết luận
nó không hoạt động. `IR = IC x sqrt(breadth)`: từ 41 lên 127 là hệ số **1,76x**.

### Điểm giao cắt có thật và đơn điệu

| kỷ nguyên | n=12 | n=50 | thắng |
|---|---|---|---|
| toàn bộ (2020+) | 1,74 | 1,53 | n=12 |
| >=40 cặp (2021-12) | 1,32 | 1,01 | n=12 |
| >=60 cặp (2023-11) | 1,61 | **1,93** | **n=50** |
| >=80 cặp (2024-09) | 1,37 | **1,80** | **n=50** |
| >=100 cặp (2025-02) | 1,50 | **1,87** | **n=50** |
| >=120 cặp (2025-08) | 1,75 | **2,19** | **n=50** |

Giao cắt ở ~60 cặp, khoảng cách NỚI RỘNG theo độ rộng. Có cơ chế, không phải lát cắt may.

### Mảnh thứ hai, và nó lớn hơn mảnh thứ nhất: TRỌNG SỐ ĐỀU

`max_weight = 0,20` cho phép MỘT vị thế chiếm 20% gross. Với 12 vị thế thì hợp lý; với
50 vị thế thì một cặp được phép nặng bằng 10 cặp khác. Siết về **0,02 = 1/50, tức trọng
số đều**:

| n=50, kỷ nguyên >=100 | Sharpe | vol | Kelly | xKelly | VND/tuần |
|---|---|---|---|---|---|
| max_w = 0,20 | 1,83 | 26% | 7,03x | 0,94 | 32.187 |
| **max_w = 0,02 (đều)** | **2,26** | **22%** | **10,00x** | **0,66** | **43.249** |

Đây KHÔNG phải tham số dò ra: `1/n` là trọng số đều, một lựa chọn cấu trúc. Quét quanh
nó cho **cao nguyên trơn** (0,022 -> 2,14 | 0,020 -> 2,26 | 0,015 -> 2,31 | 0,010 -> 2,11),
không phải điểm nhọn.

⚠️ `_cap_and_scale` kẹp trần rồi chuẩn hoá về gross 1,0, nên khi trần đặt ĐÚNG BẰNG
trung bình (1/n) vài trọng số vượt nhẹ (đo trên live: dải 0,0148-0,0202). Đó là hành vi
hội tụ bình thường, không phải lỗi.

### Độ ổn định — chữ ký mạnh nhất đo được trong repo

| n=50 đều | Sharpe | trung vị fold | %fold dương | fold tệ nhất | maxDD |
|---|---|---|---|---|---|
| >=60 cặp (344 kỳ) | 2,36 | 2,31 | 83% | −0,69 | 61% |
| >=100 cặp (187 kỳ) | 2,26 | 2,10 | **100%** | **+0,17** | 61% |
| >=120 cặp (127 kỳ) | 2,33 | 2,05 | **100%** | **+0,45** | 61% |

**Fold tệ nhất DƯƠNG.** Cấu hình hiện tại: 67% dương, tệ nhất −1,76.

### Vì sao phải đổi n VÀ đòn bẩy CÙNG LÚC

Ở cùng 2,0x thì n=12 THẮNG (105% so với 86%/năm) — vì ở đòn bẩy thấp, cấu hình
lời-cao/biến-động-cao có lợi. Lợi thế của n=50 nằm ở chỗ biến động 22% cho phép nó
CHỊU được đòn bẩy mà n=12 không chịu nổi. **Đổi mỗi `n_positions` mà giữ 2,0x sẽ làm
hệ thống TỆ ĐI**, và sẽ bị hiểu nhầm là phát hiện này sai.

7,89x chọn theo ràng buộc vật lý: $38 x 7,89 / 50 = **$6,00/vị thế**, đúng bằng
min_notional $5 x an toàn 1,2. Không phải số tối ưu hoá, là số vừa khít.

### Ba hướng đã thử và THUA trong cùng chiến dịch này — đừng làm lại

1. **Họ tín hiệu vị thế**: nghi ngờ nó bị đóng oan do hiện vật kiến trúc
   (`equal = 1/len(names)` cấp trọng số cho họ rỗng). **Bác bỏ**: thêm 3 và 10 họ TOÀN
   NaN cho Sharpe **1,7425 y hệt** tới 4 chữ số. Kiến trúc sạch; ghi chú cảnh báo cũ
   trong CLAUDE.md đã lỗi thời. Họ vị thế thua THẬT.
2. **Hồi quy mặt cắt ngang**: nghi ngờ nó bị đóng vì 26 hệ số / 41 quan sát là suy
   biến, và 127 quan sát sẽ khác. **Bác bỏ**: kể cả ở độ rộng cao, Sharpe 0,27-0,72 so
   với 1,83 của tầng gộp thích ứng. Thử ols / ridge / elastic_net đều thua.
3. **Nới universe** (`min_coverage` 0,15 -> 0,05, 127 -> 161 cặp): nhiễu, không thắng
   rõ trong Kelly. n=50 ở min_cov 0,15 vẫn tốt hơn.

### ⚠️ Giới hạn của bằng chứng này

Đây là **ĐỘ ỔN ĐỊNH qua các giai đoạn con**, KHÔNG phải kiểm định ngoài mẫu. Holdout đã
dùng 2 lần cho v3 — chạm lần ba là tự chấm bài mình, nên `strategy_v3_wide.json` đã
XOÁ hẳn khối `holdout`/`train` (chúng thuộc về n=12, mang sang là gian lận). Kiểm định
sạch duy nhất còn lại: **giao dịch giấy tiến về phía trước**.

maxDD mô phỏng **62% -> 69%**. Gấp đôi tiền không miễn phí.

```bash
python scripts/breadth_era_study.py      # tái lập bảng độ rộng theo kỷ nguyên
python scripts/preflight.py --config artifacts/strategy_v3_wide.json
```

## 🧪 18/09/2026 — HAI HƯỚNG NỮA ĐÓNG: TẬP TRUNG DANH MỤC và CHU KỲ TÁI CÂN BẰNG

Giả thuyết được đề xuất: *ít lệnh hơn, chỉ vào lệnh uy tín nhất, rồi đẩy đòn bẩy lên
20–100x.* Đã đo cả ba vế. **Cả ba đóng, và hai vế đầu mâu thuẫn với vế thứ ba.**

**1. Tập trung danh mục (`breadth_study.py --grid 4,5,6,8,10,12`)**

| n | trung vị Sharpe | %fold dương | fold tệ nhất | **Kelly** | VND/tuần |
|---|---|---|---|---|---|
| 4 | 1,31 | 88% | **−0,15** | **1,44x** | 12.379 |
| 6 | 1,34 | 88% | **−0,46** | **1,94x** | 12.944 |
| 12 | **1,61** | **100%** | **+0,30** | **3,54x** | **18.987** |

Càng tập trung thì đòn bẩy an toàn càng **THẤP**: Kelly = μ/σ², bỏ đa dạng hoá làm σ
tăng nên mẫu số phình nhanh hơn tử số. **"Ít lệnh hơn VÀ đòn bẩy cao hơn" tự mâu
thuẫn — vế đầu phá vế sau.** `n_positions >= 4` là chặn cứng trong `portfolio.py`
(sổ "trung lập" 2 chân không còn là trung lập).

**2. Chu kỳ tái cân bằng (`scripts/rebalance_period_study.py` — script MỚI)**

| chu kỳ | trung vị | %dương | tệ nhất | VND/tuần |
|---|---|---|---|---|
| 24h | 1,35 | 75% | −0,10 | 16.231 |
| 48h | 1,66 | 88% | −0,15 | 20.762 |
| **72h** | **1,61** | **100%** | **+0,30** | **21.896** |
| 96h | 1,06 | 75% | −0,35 | 8.147 |
| 144h | 1,24 | 100% | +0,05 | 11.421 |

**72h đã là tối ưu.** ⚠️ `lab_grid.json` cho 96h -> Sharpe 1,92 và suýt dẫn tới kết
luận sai: lưới đó chấm **chỉ trên train** và trên **StrategyV2** (top_frac, 4 tín
hiệu, beta_neutral=True), không phải v3. Trên v3 thì 96h là cấu hình **tệ nhất**
trong lưới. Kết quả của một cấu hình khác không chuyển thẳng sang được.

**3. Đòn bẩy 20–100x — chấm trên CẢ HAI cơ sở để kết luận không phụ thuộc cách đo:**

| | holdout (thận trọng) | toàn dòng thời gian |
|---|---|---|
| Sharpe / vol | 1,28 / 55,2% | 1,72 / 46% |
| Kelly | 2,32x | 3,73x |
| 2,0x đang chạy | **0,86 Kelly** | 0,54 Kelly |
| g(L) = 0 tại | 4,65x | 7,47x |
| **g(20x)** | **−4.676%/năm** | **−2.652%/năm** |
| **g(100x)** | **−145.231%/năm** | **−97.900%/năm** |

20x cháy khi giá lệch 5%; 100x cháy khi lệch 1%. Ở 20x, biến động **58,6%/NGÀY**.

**Điều đáng nhớ hơn các con số:** VỐN nhân lợi nhuận **tuyến tính** và không kèm hình
phạt. ĐÒN BẨY nhân lợi nhuận tuyến tính nhưng nhân rủi ro **bậc hai** — `g = μL −
(σL)²/2`. Vì thế gấp đôi vốn luôn gấp đôi tiền, còn gấp đôi đòn bẩy thì không, và quá
2×Kelly thì thêm đòn bẩy LÀM GIẢM tiền. Đây là lý do ràng buộc thật luôn là vốn.

⚠️ **Dư địa đòn bẩy còn lại phụ thuộc tin cơ sở nào.** Holdout nói 2,0x đã là 0,86
Kelly — gần như hết dư địa. Toàn dòng thời gian nói 0,54 Kelly — còn tới ~2,8x. Cơ sở
thứ hai GỒM CẢ đoạn đã dùng để chọn cấu hình, nên nó rộng rãi có thiên vị. Đừng nâng
quá 2,4x nếu chưa có giao dịch giấy tiến về phía trước xác nhận.

## 🔴 F44 — TRẦN ĐUỔI GIÁ CỐ ĐỊNH 15bp LÀ NÚT THẮT THẬT CỦA TỶ LỆ MAKER. ĐÃ VÁ.

Ghi chú 14/09 nói maker 0,378 -> 0,85 đáng **+0,06 Sharpe, +7,7% lợi nhuận**, rồi vá
bằng cách nâng `passive_wait_s` 300 -> 900s. **Vá sai chỗ.** Thực đo: 48,8% -> 54,3%.

Đo trên 60 cặp thật (σ nến 1h, quy về cửa sổ chờ theo căn bậc hai thời gian):

| | |
|---|---|
| dịch giá trung vị trong 900s | **64,4 bp** |
| trần đuổi cũ (cố định) | 15,0 bp |
| số cặp vượt trần | **100%** |
| cặp mạnh nhất (BTRUSDT, σ=5,22%/h) | 261 bp |

Ở mức dịch trung vị, trần 15bp bị chạm sau **~50 giây của cửa sổ 900 giây**. Lệnh
đóng băng ngoài thị trường suốt 94% thời gian chờ rồi rơi xuống taker. **Nút thắt
chưa bao giờ là THỜI GIAN** — nên kéo dài thời gian chờ không thể sửa được nó.

**Điều sắc nhất, và nó lật ngược lý do tồn tại của trần này:** khi
`allow_taker_fallback=True`, vượt trần KHÔNG làm hệ thống bỏ lệnh — nó vẫn giao dịch,
chỉ là bằng taker ở đúng mức giá đã trôi đó. Cả hai đường đều trả khoản trôi giá như
nhau; đường taker trả THÊM spread + 3bp phí. Trần đuổi vì thế không bảo vệ khỏi bất
cứ thứ gì, nó thuần tuý làm giao dịch đắt hơn. Nó chỉ có nghĩa nếu vượt trần đồng
nghĩa BỎ HẲN lệnh — mà bỏ lệnh thì phá trung lập, tức tệ hơn nhiều.

**Vá:** trần co giãn theo biến động RIÊNG của từng cặp trong đúng cửa sổ chờ —
`max(max_chase_bps, chase_sigma_mult x σ_cửa-sổ)`, `chase_sigma_mult = 1.5`. Đo trên
12 cặp đang giữ: trung vị trần mới **124,7bp (rộng hơn 8,3x)**, nhưng PAXGUSDT (vàng)
chỉ 20,9bp còn BULLAUSDT (σ=4,43%/h) 448,7bp. **Một con số duy nhất không thể đúng
cho cả hai** — đó mới là lỗi gốc, không phải giá trị 15 to hay nhỏ.

Cặp nào không đo được σ thì dùng trần sàn: thiếu dữ liệu phải ngả về phía THẬN TRỌNG.
Khoá bằng `tests/oms/test_chase_cap_volatility_f44.py` (8 test).

⚠️ Đây là cải thiện theo CƠ CHẾ, chưa phải theo backtest khớp lệnh. Bộ đếm
`requote_blocked_by_chase_cap` + `max_drift_bps_seen` đã có sẵn trong `ExecutionRecord`
— lượt sạch tới sẽ xác nhận bằng số thật. Đừng ghi nhận +7,7% cho tới khi có số đó.

## 🔴 F45/F46/F47 (22/09/2026) — F44 LÀ CODE CHẾT VÌ VAN TRUNG LẬP SAI MẪU SỐ. ĐÃ VÁ.

Lượt 19/09 (lượt 50 vị thế đầu tiên) để lại bốn bộ đếm bằng 0: `requotes=0`,
`requote_blocked_by_chase_cap=0`, `requote_skipped_no_move=0`, `max_drift_bps_seen=0.0`.
Nếu `_requote_one` từng chạy thì ít nhất một trong ba cái đầu phải khác 0. **Vòng đuổi
giá chưa từng được gọi lần nào — trần co giãn của F44 chưa bao giờ có hiệu lực.**

Van trung lập bắn ở `early_exit=neutrality_drift` rồi `return`, và vòng báo giá lại
nằm ngay bên dưới nên không bao giờ tới lượt. 31/50 lệnh hết giờ, maker 39%.

### Nguyên nhân: một tỷ lệ có mẫu số CO LẠI

`drift = (net − net kỳ vọng) / gross ĐÃ KHỚP`. Lúc mới khớp vài lệnh, mẫu số cực nhỏ
nên tỷ lệ này **không đo rủi ro, nó đo việc ta mới khớp được ít**. Mô phỏng 20.000
lượt (50 lệnh, cỡ lognormal σ=0,35) cho độ lệch THUẦN NGẪU NHIÊN — không có rủi ro thật:

| tiến độ | /gross đã khớp | /gross kế hoạch |
|---|---|---|
| 10% | **42,9%** | 4,8% |
| 25% | **26,1%** | 6,8% |
| 50% | **14,9%** | 7,6% |
| 80% | 7,2% | 5,9% |

Ngưỡng 10% nằm DƯỚI mức nhiễu ở mọi tiến độ hữu ích -> van bắn **100% số lượt**, ở
tiến độ trung vị **2%**.

⚠️ **Chẩn đoán đầu tiên đã SAI, và suýt được commit.** Bản vá đầu thêm
`min_gross_for_neutrality_check = 0,25 × gross kế hoạch`. Đo lại: ở ngưỡng đó van
**vẫn bắn 99,4%**; phải tới 95% mới im, mà lúc đó van vô dụng. **Cổng tiến độ không
sửa được một phép đo sai đơn vị.** Đừng dựng lại hướng này — đã khoá bằng test.

**Vá:** van đo `drift_plan` (mẫu số = gross KẾ HOẠCH, cố định) nên nhiễu bị chặn trên
ở ~7,6%. Ngưỡng `max_fill_imbalance = 0,25` chọn bằng đo, 8.000 lượt/kịch bản:

| ngưỡng | báo nhầm (khớp lành) | bắt được (một chân kẹt) |
|---|---|---|
| 0,10 | 67,0% | 100% |
| 0,20 | 3,6% | 100% |
| **0,25** | **0,3%** | **100%** |
| 0,40 | 0,0% | **61%** |

Đổi TÊN tham số (`neutrality_tolerance` -> `max_fill_imbalance`) là cố ý: đơn vị đã
đổi, nên chỗ gọi nào sót phải nổ bằng TypeError thay vì âm thầm chạy sai — đúng khe
hở F38.

### Bản vá này đáng bao nhiêu tiền — và vì sao con số cũ thấp gấp 4 lần

`scripts/maker_ratio_study.py` (MỚI) đo trên chính cấu hình đang chạy, kỷ nguyên >=100 cặp:

| maker | Sharpe | +ann% @1x | +ann% @2x | **+ann% @7,89x** |
|---|---|---|---|---|
| 0,39 (live 19/09) | 2,229 | — | — | — |
| 0,50 (giả định artifact) | 2,257 | +0,6 | +1,3 | +5,1 |
| 0,70 | 2,309 | +1,8 | +3,6 | +14,3 |
| **0,85 (mục tiêu)** | **2,348** | **+2,7** | +5,4 | **+21,2** |

Ghi chú 14/09 định giá "+7,7% lợi nhuận" — đo ở **n=12 / 2,0x**. Chi phí = turnover ×
đơn giá, và turnover tỷ lệ THUẬN với đòn bẩy, nên cùng một cải thiện ở 7,89x đáng
**gấp ~4 lần**. Dùng lại con số cũ là xếp sai thứ tự ưu tiên công việc.

⚠️ `strategy_v3_wide.json` ghi `maker_ratio_assumed: 0.50` còn live đo **0,39** —
Sharpe đã kiểm định vốn đã lạc quan hơn thực tế, trước cả khi bàn tới cải thiện.

**F46** — bản ghi nay lưu `plan_net_exposure` cạnh `net_exposure`. Lượt 19/09 ra net
+4,00% dù khớp đủ 50/50 lệnh, và bản ghi cũ KHÔNG phân biệt được "kế hoạch vốn đã
lệch" với "thực thi làm nó lệch" — hai nguyên nhân sửa ở hai file khác nhau. Thêm
`fill_drift_plan` + `fill_progress` vì `fill_drift` một mình vô nghĩa nếu không biết
lúc đó đã khớp bao nhiêu.

**F47** — `healthcheck.py` trừ thời gian MÁY NGỦ trước khi kết luận daemon chết.
Plist ghi rõ máy này "gập nắp, ngủ gần như liên tục", `time.sleep` treo theo giấc ngủ
nên nhịp tim đứng là BÌNH THƯỜNG. Đo 22/09: nhịp tim đứng 5,6h trong khi `pmset` cho
thấy máy ngủ 4,3/6h gần nhất — daemon hoàn toàn khoẻ. Cảnh báo không phân biệt được
hai tình huống đó sẽ bị học cách bỏ qua, rồi câm khi cần nhất. Bài học F41, lặp lại
ở tầng cảnh báo.

## ✅ 23/09/2026 — F45 XÁC NHẬN BẰNG LƯỢT THẬT. Nút thắt ĐÃ DỊCH sang trần đuổi giá.

Lượt 23/09 12:29 là lần đầu F45 chạy production. So với lượt 19/09 (cùng cấu hình,
trước bản vá):

| chỉ số | 19/09 (trước) | 23/09 (sau) | |
|---|---|---|---|
| `early_exit` | `neutrality_drift` | **`None`** | van hết bắn sớm ✅ |
| `requotes` | **0** | **40** | đường đuổi giá SỐNG ✅ |
| `max_drift_bps_seen` | **0,0** | **778,4** | xác nhận `_requote_one` có chạy |
| `requote_blocked_by_chase_cap` | 0 | **358** | nút thắt MỚI |
| tỷ lệ maker | 39,0% | **43,6%** | |
| net sau thực thi | +4,00% | −2,03% | |
| **net của KẾ HOẠCH** | *(chưa đo)* | **0,0050%** | F46 |

**F46 trả lời dứt điểm câu hỏi treo từ 19/09:** kế hoạch trung lập tới 5 phần trăm
nghìn, còn sổ sau thực thi lệch −2,03%. **Toàn bộ sai lệch đến từ THỰC THI, không phải
từ kế hoạch.** `_repair_neutrality` của F42 làm đúng việc của nó; đừng đi sửa tầng lập
kế hoạch nữa. Cơ chế còn lại: giá trôi tới 778bp trong cửa sổ 900s làm notional thực
tế lệch khỏi notional lúc lập kế hoạch.

### Nút thắt mới, và lần này công cụ TỰ chỉ ra

`readiness_gate.py` đổi kết luận từ "Nút thắt là THỜI GIAN CHỜ" sang **"Nút thắt là
TRẦN ĐUỔI GIÁ"** — chẩn đoán cũ sai vì lúc đó ba bộ đếm đều bằng 0.

Trần chặn **358/398 = 90%** số lần thử đuổi. Tức trần co giãn 1,5σ của F44 nay là thứ
giữ tỷ lệ maker ở 43,6%.

⚠️ Lập luận của chính F44 dẫn tới kết luận F44 chưa đi hết: khi
`allow_taker_fallback=True`, vượt trần KHÔNG làm bỏ lệnh — lệnh vẫn khớp bằng taker ở
đúng mức giá đã trôi đó, cộng thêm spread + 3bp. **Trần không tránh được cú trôi giá,
nó chỉ làm cú trôi đó đắt hơn.** Nhưng ĐỪNG nới vội: xem F49.

**F48** — `readiness_gate.py` cắt thông điệp lỗi ở 60 ký tự, biến
`"long $13.184 vs short $13.730"` thành `"long $13.184 vs short $1"` — đọc ra thành sổ
MỘT CHIỀU hoàn toàn, một sự cố thuộc hạng khác hẳn. Sổ thật lúc đó long $13.061 /
short $13.668, bình thường. **Công cụ giám sát bịa ra sự cố nặng hơn thực tế cũng nguy
hiểm ngang việc bỏ sót** — cả hai đều dạy người vận hành ngừng tin nó.

**F49** — bản ghi nay lưu `chase_block_p50_ratio` / `chase_block_p90_ratio`: vượt trần
bao nhiêu LẦN, không chỉ đếm số lần vượt. "Chặn 358 lần" không nói được phải nới bao
nhiêu — chặn vì vượt 1,1 lần và vượt 20 lần đòi hai hành động khác hẳn. Lượt tới sẽ
cho hai phân vị này, và khi đó việc chỉnh trần là PHÉP ĐO chứ không phải phỏng đoán.
**Đừng chỉnh `chase_sigma_mult` trước khi có hai con số đó.**

### Tiền: phí KHÔNG phải thứ làm tụt ví

Ví $5.301,55 -> $4.782,37 (−$519,18) qua lượt này. Phân rã:

| | |
|---|---|
| phí sàn (3,62bp trên $44.444 turnover) | **−$16,10  (3%)** |
| lỗ đã chốt khi đóng vị thế cũ | −$503,08  (97%) |

97% khoản trừ ví chỉ là uPnL chuyển thành lỗ thực hiện — kế toán, không phải mất mới.
Đại lượng cần theo dõi vẫn là EQUITY: $5.499 (lúc tái cân bằng) -> $5.082 (6h sau).

## 🧪 22/09/2026 — ĐỘ RỘNG VƯỢT 50: ĐÓNG. n=50 nằm trên CAO NGUYÊN, không phải may.

Bảng độ rộng cũ dừng ở n=50 vì ràng buộc vốn ($38 × 7,89 / 50 = $6,00), nên "50 tốt
nhất" chưa từng được kiểm — chỉ có "50 là lớn nhất mà $38 mua nổi". Nay đã quét
(`scripts/breadth_beyond_50.py`, kỷ nguyên >=100 cặp, maker 0,39 mức thật):

| n | max_w | Sharpe | ann% | vol% | g(Kelly)% | trung vị fold |
|---|---|---|---|---|---|---|
| 30 | 0,033 | 1,36 | 45,0 | 33,1 | 93 | 1,60 |
| 40 | 0,025 | 2,19 | 58,3 | 26,6 | 239 | 2,27 |
| **50** | 0,020 | **2,23** | 50,3 | 22,6 | 248 | 2,27 |
| 60 | 0,017 | **2,26** | 44,2 | 19,6 | 255 | **2,60** |
| 70 | 0,014 | 2,09 | 35,7 | 17,1 | 218 | 2,31 |
| 100 | 0,010 | 1,75 | 22,6 | 12,9 | 153 | 2,01 |

n=60 hơn n=50 đúng **+0,03 Sharpe** — nhiễu, và đòi 9,5x thay vì 7,9x ở vốn $38. Từ
n=70 suy giảm rõ: pha loãng vào tín hiệu yếu làm lợi suất tụt nhanh hơn biến động.
**Ràng buộc vốn và điểm tối ưu tình cờ trùng nhau** — nay đã đo chứ không còn là giả định.

## ⚖️ 22/09/2026 — ĐÒN BẨY: KHOẢNG HAI CƠ SỞ CÙNG ĐỒNG Ý LÀ **4–6x**. `scripts/leverage_study.py`

Daemon chạy `--leverage 5.0` (cờ CLI), cấu hình wide chốt **7,89x**. Trước đây câu hỏi
này được trả lời bằng `g = μL − (σL)²/2`. **Công thức đó là khai triển Taylor bậc 2 và
nó SAI theo hướng lạc quan** — giả định lợi suất gần Gauss, bỏ qua đuôi trái nơi
`log(1+Lr)` phân kỳ về âm vô cùng. Nay đo bằng compounding thật `E[log(1+Lr)]`.

### Điều quan trọng nhất: kỳ 72h TỆ NHẤT trong lịch sử là −13,41%

Tức **cháy sạch vốn từ 7,46x trở lên** — cấu hình 7,89x nằm NGAY TRÊN ngưỡng đó. Một
lần lặp lại kỳ 20/02/2021 là mất trắng tài khoản, không phải sụt giảm.

| | toàn lịch sử (799 kỳ) | kỷ nguyên >=100 cặp (187 kỳ) |
|---|---|---|
| Sharpe | 1,34 | 2,23 |
| kỳ tệ nhất | **−13,41%** | −6,21% |
| cháy từ | **7,46x** | 16,09x |
| L* theo g THẬT | 4,5x | 10,0x |
| L* bootstrap khối, KTC 90% | **[2,0 .. 6,0]x** | **[4,0 .. 14,0]x** |
| g(5,0x) | 89%/năm, P(cháy) 0% | 193%/năm, P(cháy) 0% |
| g(7,89x) | 42%/năm, **P(cháy) 65%** | 264%/năm, P(cháy) 0% |

**Giao của hai khoảng tin cậy là [4,0 .. 6,0]x.** 5,0x nằm giữa; **7,89x nằm ngoài
khoảng của toàn lịch sử.** Đây không phải thoả hiệp — đó là vùng mà kết luận KHÔNG phụ
thuộc vào việc tin cơ sở nào.

### Đuôi trái biến mất THẬT hay chưa kịp xuất hiện?

Kỳ −13,41% rơi vào **20/02/2021, lúc universe chỉ có 34 cặp** — nên phần nào do độ
rộng thật. Nhưng so với mức worst-of-n kỳ vọng nếu Gauss:

| kỷ nguyên | n kỳ | σ kỳ | tệ nhất | = mấy σ | Gauss kỳ vọng |
|---|---|---|---|---|---|
| toàn bộ | 799 | 2,58% | −13,41% | **−5,19σ** | −9,45% |
| >=100 cặp | 187 | 2,04% | −6,21% | **−3,05σ** | −6,60% |

Kỷ nguyên rộng có cú tệ nhất **đúng bằng mức Gauss dự đoán** — tức nó **CHƯA GẶP** cú
đuôi dày nào, chứ không phải đã chứng minh là không có. Mẫu 187 kỳ = 1,5 năm.

**Phép thử căng thẳng** — giả định 5,19σ lặp lại ở σ hiện tại (2,04%) = **−10,59%/72h**:

| đòn bẩy | mất trong 72h |
|---|---|
| 2,5x | 26,5% |
| 5,0x | **52,9%** |
| 7,0x | 74,1% |
| **7,89x** | **83,5%** |
| 9,45x | **CHÁY SẠCH** |

### Walk-forward: chọn đòn bẩy theo quá khứ KHÔNG chuyển sang tương lai

Chọn L tối ưu trên 1 năm quá khứ rồi chấm trên quý kế tiếp:

| | L* theo quá khứ | cố định 5,0x | cố định 7,89x |
|---|---|---|---|
| toàn lịch sử (216 cửa sổ) | +16% | **+21%** | **−111%** |
| kỷ nguyên >=60 (88 cửa sổ) | +42% | +136% | **+170%** |

Tối ưu hoá L theo quá khứ THUA cả việc cố định 5,0x ở cả hai cơ sở — cùng một cơ chế
đã đo ở tương quan Sharpe(quá khứ) vs Sharpe(tương lai) = **−0,015**. Không dự báo
được Sharpe thì không dự báo được Kelly.

⚠️ `g5 = [x for x in g5 if np.isfinite(x)]` — **đừng bao giờ lọc như thế**. Bản nháp
đầu của script này lọc `isfinite` trong bootstrap, tức vứt đúng những lần CHÁY rồi lấy
trung vị phần còn lại. Nó biến P(cháy)=65% thành một con số 42%/năm trông lành lặn.
Cháy phải được báo RIÊNG như một xác suất, không được lẫn vào phân phối tăng trưởng.

## 🧪 22/09/2026 — CỠ VỊ THẾ KHÁC NHAU CHO TỪNG LỆNH: ĐÓNG CẢ BA DẠNG. `scripts/sizing_mode_study.py`

Giả thuyết: mỗi lệnh nên có đòn bẩy/cỡ riêng thay vì giống nhau toàn bộ.

**Điều cần biết trước:** với ký quỹ CHÉO trên perp USDⓈ-M, "đòn bẩy từng lệnh" KHÔNG
phải đại lượng rủi ro độc lập — cài 20x cho một cặp chỉ đổi bậc ký quỹ ban đầu của cặp
đó, rủi ro danh mục vẫn là TỔNG NOTIONAL / VỐN. Thứ phân biệt được là TRỌNG SỐ.

Hệ thống **đã** phân biệt qua `zscore_riskparity`, nhưng `max_weight = 0,02 = 1/50`
kẹp phẳng gần hết (live đo dải 0,0148–0,0202, chênh 1,37 lần). Nới trần để phân biệt
bung ra, kỷ nguyên >=100 cặp:

| chế độ | trần 0,02 | trần 0,10 | kỳ tệ nhất @0,10 |
|---|---|---|---|
| `rank_binary` (đều hoàn toàn) | 2,25 | **2,25** | −6,21% |
| `rank_riskparity` (theo biến động) | 2,26 | 2,16 | −5,79% |
| `zscore` (theo độ mạnh tín hiệu) | 2,26 | **1,79** | **−7,50%** |
| `zscore_riskparity` (cả hai) | 2,23 | 1,85 | −6,06% |

Phân biệt theo tín hiệu thua **cả Sharpe lẫn đuôi** — đòn bẩy an toàn tụt 5,7x -> 3,9x.

**Dạng thứ ba, theo CHI PHÍ** — dạng duy nhất có cơ chế rõ (chi phí đáng 21%/năm ở
7,89x). Cũng thua, và thua nặng nhất:

| bể chọn | n cặp | Sharpe | ann% |
|---|---|---|---|
| TOÀN BỘ (đang chạy) | 127 | **2,23** | 50,3 |
| bỏ 10% đắt nhất | 114 | 1,91 | 34,1 |
| bỏ 25% đắt nhất | 95 | 1,60 | 28,0 |
| chỉ 50% rẻ nhất | 64 | 0,95 | 14,5 |
| chỉ 25% rẻ nhất | 32 | **0,59** | 8,3 |

Cặp ĐẮT chính là cặp NHỎ và KÉM HIỆU QUẢ — tức nơi alpha sống. Tiết kiệm vài bp không
bù nổi alpha mất đi.

**Lý do chung cho cả ba, và nó đã nằm sẵn trong repo:** `IR = IC × √độ_rộng`. Edge của
chiến lược này là ĐỘ RỘNG, không phải độ tin cậy từng lệnh. Mọi phép TẬP TRUNG — theo
tín hiệu, theo biến động, hay theo chi phí — đều bán đi chính nguồn sinh ra edge. Cùng
một cơ chế đã đóng `n_positions` nhỏ (18/09: n=4 cho Kelly 1,44x so với n=12 cho 3,54x).

## 🔴 F43 — CẮT THEO SỨC CHỨA VỐN BIẾN QUỸ TRUNG LẬP THÀNH CƯỢC MỘT CHIỀU. ĐÃ VÁ.

`xs_live_pipeline` cắt danh mục khi vốn không đủ `n_positions` vị thế. Bản cũ lấy
top-N theo **|trọng số| bất kể dấu** — chú thích ghi "giữ mạnh nhất ở cả hai chiều"
nhưng code không hề làm thế.

**Chưa từng cắn vì testnet có $6.226 nên `cap` luôn >> 12. Nó cắn đúng lúc chạm tiền
thật.** Vốn $38 ở 2x cho `cap` = 12 — vừa khít. Sụt 8% xuống $35 là `cap` = 11, và
cắt lẻ thì hai chân không thể bằng nhau. DD kỳ vọng 24,7%/năm nên đây là đường đi
MẶC ĐỊNH của tài khoản thật, không phải biên hiếm.

Mô phỏng 20.000 lượt với độ mạnh tín hiệu ngẫu nhiên (trần ±2%):

| vốn | cap | P(vi phạm trần) | net xấu nhất |
|---|---|---|---|
| $35 | 11 | **100%** | 9,1% |
| $28 | 9 | **100%** | 33,3% |
| $25 | 8 | 55% | 50,0% |
| $20 | 6 | 57% | **100%** |

Net 100% nghĩa là danh mục **một chiều hoàn toàn ở đòn bẩy 2x** — đúng thứ chiến lược
này sinh ra để không bao giờ làm.

**Vá:** cắt cân hai chân (`cap // 2` mỗi chân, giữ tín hiệu mạnh nhất từng chân), rồi
co giãn mỗi chân về đúng nửa gross -> net = 0 **chính xác** (test khoá ở 1e-12, không
phải ở trần ±2%: phép cắt là thao tác toán học thuần, để ngưỡng lỏng là tự cho phép
một chỗ rò mới núp dưới trần). Không đủ 1 cặp mỗi chân thì **HALT** — giao dịch một
chiều còn tệ hơn không giao dịch.

Cố tình KHÔNG dùng `project_neutral`: phép chiếu trực giao trừ đi trung bình nên có
thể ĐẢO DẤU một trọng số nhỏ, biến một cặp long thành short ngược với tín hiệu sinh
ra nó. Co giãn theo chân giữ nguyên mọi dấu và mọi thứ hạng trong chân.

Khoá bằng `tests/pipelines/test_capacity_truncation_f43.py` (18 test, quét toàn dải
vốn $12-$38).

## 🔴 F42 — DẢI KHÔNG GIAO DỊCH PHÁ TRUNG LẬP. ĐÂY LÀ THỨ GIỮ CỔNG ĐÓNG. ĐÃ VÁ.

`portfolio.py` dựng trọng số trung lập **chính xác 0,000%** (đo trên sổ thật 18/09).
Rồi `build_rebalance_plan` áp `DEFAULT_NO_TRADE_BAND = 0.20` để khỏi trả phí cho
điều chỉnh vụn — nhưng cặp bị bỏ qua **giữ khối lượng CŨ, không phải khối lượng
ĐÍCH**. Mỗi lần bỏ qua là sai số tới 20% cỡ vị thế, và các sai số đó KHÔNG tự triệt
tiêu theo chiều.

**RÒ NGẪU NHIÊN — đó là lý do nó thoát mọi test cũ.** Lượt nào các cặp bị bỏ qua
tình cờ ngược chiều thì sổ sạch; cùng chiều thì sổ lệch:

| lượt | các cặp bị bỏ | net | kết quả |
|---|---|---|---|
| 18/09 | NGƯỢC chiều (SAND short, VTHO long) | +0,56% | ✅ sạch |
| 16/09 | CÙNG chiều | −2,96% | ❌ bẩn |
| 11/09 | cùng cơ chế | −3,16% | ❌ bẩn |

**Hệ quả là toàn bộ dự án đứng ở cửa.** Cổng cần 3 lượt sạch LIÊN TIẾP, và
`E[lượt] = 1/p + 1/p² + 1/p³`:

| tỉ lệ lượt sạch | thời gian tới cổng |
|---|---|
| 17% (lịch sử thật: 1 sạch / 6 lượt) | **2,1 NĂM** |
| 50% | 42 ngày |
| 90% (sau vá) | **11 ngày** |

⚠️ **"9 ngày" từng báo là SÀN TUYỆT ĐỐI, không phải kỳ vọng.** Nó chỉ đúng nếu 3
lượt liên tiếp đều sạch. Báo cáo con số tốt nhất như thể nó là con số kỳ vọng đã
che mất việc cổng này thực tế không bao giờ tới.

**Vá:** giữ nguyên dải (nó tiết kiệm phí thật), thêm `_repair_neutrality()` — nếu kế
hoạch lệch quá `max_net_exposure` thì NHẬN LẠI các lệnh đã bỏ ở chân nặng, lệnh kéo
mạnh nhất trước, tới khi về trong trần. Lượt vốn đã sạch thì không phát sinh lệnh nào
(kiểm chứng trên sổ thật 18/09: 0 lệnh cân thừa). Trần lấy từ `LiveConfig.max_net_exposure`,
**cùng một con số** cổng F25 dùng để chấm sau thực thi — trước đây tầng kế hoạch hoàn
toàn mù về trung lập và mãi tới bước 6b, khi lệnh ĐÃ nằm trên sàn, mới có thứ đo nó.

**Lỗi thứ hai, lộ ra khi viết test cho lỗi thứ nhất:** điều chỉnh làm tròn về 0 thì
`continue` thẳng, nên vị thế **biến mất khỏi `gross`/`net` của kế hoạch**. Đo thử 12
vị thế thì 9 cái bốc hơi — gross đọc ra $5.000 thay vì $20.000, net/gross ra **+100%
thay vì 0%**. Sổ kế toán của kế hoạch phải tả đúng danh mục SẼ tồn tại, không chỉ
những cặp có phát lệnh.

Khoá bằng `tests/execution/test_neutrality_no_trade_band_f42.py` (3 test, ở tầng KẾ
HOẠCH — tức trước khi mất một đồng phí nào).

**Bài học:** F25 chấm trung lập SAU thực thi. Một cổng chỉ biết hét sau khi lệnh đã
lên sàn thì không ngăn được gì — nó chỉ ghi biên bản. Mọi bất biến phải được chặn ở
tầng còn sửa được miễn phí.

## 🔴 F41 — DAEMON CHẾT 2 NGÀY 3 GIỜ MÀ `launchctl list` VẪN KHOE PID. ĐÃ VÁ.

16/09 18:40 -> 18/09 21:35: `.venv` biến mất -> `certifi/cacert.pem` không còn ->
MỌI kết nối TLS chết. Daemon ném lỗi **58 chu trình liên tiếp**, không gọi nổi sàn,
không đặt nổi lệnh. Nhưng `launchctl list` vẫn hiện PID bình thường suốt 2 ngày.

**Hai lỗi thiết kế cộng hưởng:**
1. `run_daily.py` bắt `Exception` -> ghi log -> `sleep` -> lặp **vô hạn**. Hỏng vĩnh
   viễn và hỏng thoáng qua được đối xử y hệt nhau.
2. Kênh báo động duy nhất là Telegram, mà Telegram đi qua **đúng cái TLS vừa chết**.
   Đây là HỎNG TƯƠNG QUAN: kênh báo động dùng chung hạ tầng với thứ nó canh, nên
   đúng lúc cần kêu nhất thì nó câm.

**Thiệt hại thật = 0 đồng, và đó là MAY, không phải nhờ thiết kế.** Cửa sổ chết nằm
trọn trong một chu kỳ giữ 72h (lượt cuối 16/09 15:43, lượt kế 19/09 15:43) nên không
lỡ lượt nào. Dựng lại đường vốn: đáy $5,734.61 lúc 16/09 21:00 = DD 5,74%, còn cách
van TIER1 (10%) 4,3 điểm phần trăm. Mất là mất **BẢO HIỂM**, không mất lợi nhuận —
giá rơi thêm 4,3pp nữa thì van cắt nửa vị thế đã không ai bấm.

**Vá:**
- `_write_health()` ghi nhịp tim ra `logs/daemon_health.json` mỗi chu trình. Đĩa còn
  sống khi mạng đã chết, nên đây là nguồn sự thật duy nhất tin được.
- `_local_alert()` báo động qua `osascript` + cờ `logs/DAEMON_DOWN.txt` — **không đi
  qua mạng**.
- Hỏng >= 3 chu trình liên tiếp thì `return 1` để launchd khởi động lại; nếu vẫn
  hỏng, `launchctl list` hiện mã thoát khác 0 — nói THẬT thay vì khoe PID của xác.
- `scripts/healthcheck.py` — **một lệnh** trả lời XANH/ĐỎ bằng tuổi nhịp tim.

**Vá luôn nguyên nhân venv không tái lập được:** `pyarrow`, `requests`, `joblib`
trước đây chỉ cài tay vào `.venv`, không khai báo trong `pyproject.toml`. Dựng lại
venv ra một môi trường THIẾU chúng: không pyarrow -> không đọc nổi parquet ->
`universe` về **0 cặp**. Và `.gitignore` có `venv/` nhưng thiếu `.venv/`, nên `.venv`
là untracked — `git clean -fd` xoá sạch. Cả hai đã sửa.

**Bài học:** `launchctl list` chỉ chứng minh tiến trình CÒN TỒN TẠI, không chứng minh
nó CÒN LÀM VIỆC. Mọi kênh giám sát phải trả lời được câu "nó hỏng thế nào khi chính
hạ tầng của nó hỏng" — cùng câu hỏi đã học ở F40.

```bash
python scripts/healthcheck.py    # ⭐ CHẠY ĐẦU TIÊN mỗi lần vào xem hệ thống
```

## 🔴 F40 — min notional bị giả định ĐỒNG NHẤT $5. Sẽ cắn ở lượt MAINNET ĐẦU TIÊN. ĐÃ VÁ.

`max_positions_for_capital` giả định mọi cặp cần $5. Mainnet: 122/128 cặp đúng $5,
nhưng **ETH/LTC/LINK/ETC/BCH cần $20 và BTCUSDT cần $50**. Vốn $38 ở 2x cho $6,34/vị
thế — sáu cặp đó không mua nổi. Nếu một cái lọt top-6, lệnh bị bỏ âm thầm, danh mục
**mất một chân** và hết trung lập — đúng cơ chế F25 (sổ lệch +35%).

Chưa từng cắn vì mới chạy testnet. Sẽ cắn đúng lúc có tiền thật.

⚠️ Bẫy thứ hai, phát hiện khi viết test cho bản vá thứ nhất: chia cho `n_positions`
CỨNG làm ngưỡng tụt theo vốn; vốn $35 -> ngưỡng $4,86 -> **quét sạch universe** ->
`resolve_universe` ném lỗi "còn 0 cặp". Sụt 8% vốn là chết đường chạy, và thủ phạm là
chính lớp bảo vệ vừa thêm. Nay dùng số vị thế đã điều chỉnh theo vốn, ngưỡng luôn >= $5.

**Một bộ lọc an toàn có thể tự trở thành nguyên nhân sự cố.** Mọi lớp bảo vệ mới phải
được hỏi "nó hỏng thế nào khi điều kiện xấu đi", không chỉ "nó chặn đúng thứ cần chặn".

## 🔴 F38 — live duyệt `SIGNAL_REGISTRY` thay vì bộ đã kiểm định. ĐÃ VÁ.

Registry là nơi CHỨA mọi tín hiệu từng viết, không phải danh sách đang giao dịch. Thêm
họ mới vào registry sẽ âm thầm đổi thứ live đặt lệnh. **Test parity trọng số không bắt
được** — nó dựng cả hai phía bằng cùng một danh sách. Cùng khe hở F37 lọt qua: công
thức thì khoá, cấu hình thì không. Nay có 2 test parity ở tầng CẤU HÌNH.

## 🔴 14/09/2026 — F37: LIVE ĐANG CHẠY 62 CẶP THAY VÌ 127. ĐÃ VÁ.

`data/universe.history_lengths` đếm nến trong file `{sym}_4h.parquet`, nhưng hệ thống
dựng khung 4h **TỪ file 1h** (`load_panel_v2(source_interval="1h")`). Bộ lọc universe
vì thế đo một thứ còn đường chạy dùng một thứ khác — họ lỗi F3, lớp áo khác.

**89/170 cặp bị loại OAN.** CRVUSDT có 52.869 nến 1h (=13.217 nến 4h) nhưng file 4h
chỉ còn 186 dòng sót từ lần tải cũ -> bị loại vì "thiếu lịch sử".

| rổ | cặp | Sharpe | ann | trung vị fold |
|---|---|---|---|---|
| CŨ (lỗi F37) | 60 | **1,20** | 45,5% | 1,04 |
| MỚI (đã vá) | 101 | **1,56** | 64,8% | 1,31 |
| nghiên cứu | 127 | 1,74 | 79,4% | 1,61 |

**0,36 Sharpe = +69% lợi nhuận tuần** (~10.450 -> ~17.660 VND). Sau khi vá,
`build_universe` trên MAINNET trả về **đúng 127 cặp** — khớp chính xác cấu hình đã
kiểm định. Trước khi vá thì không.

⚠️ **F37 nằm trên đường tiền và chỉ có hiệu lực sau khi NẠP LẠI daemon.**

**Bài học đắt hơn bản thân lỗi:** cả chiến dịch nghiên cứu hôm nay (5 hướng) tìm được
+7,7%. Một lỗi đúng/sai phát hiện tình cờ khi viết `scripts/preflight.py` đáng +69%.
Ở hệ thống chưa từng kiểm tra đầu-cuối, alpha lớn nhất nằm ở chỗ nó KHÔNG chạy đúng
thứ mình tưởng — không nằm ở mô hình.

## 💰 14/09/2026 — KỲ VỌNG LỢI NHUẬN THẬT, và vì sao nó là con số đó

Tăng trưởng BỀN VỮNG tối đa ở đòn bẩy Kelly là `S²/2`/năm. Đảo ngược cho vốn 1 triệu:

| muốn mỗi tuần | cần Sharpe năm | hiện có |
|---|---|---|
| 1,0% (10.000 VND) | 1,02 | ✅ |
| 2,0% (20.000 VND) | 1,44 | ⚠️ |
| 3,0% (30.000 VND) | 1,75 | ❌ |
| 5,0% (50.000 VND) | **2,25** | ❌ (đang 1,10) |

**Sharpe holdout ở lưới 4h = 1,10 -> bền vững ~11.700 VND/tuần.** Muốn 50.000 phải
nhân đôi Sharpe. Đã quét 6 hướng để tìm phần nhân đôi đó:

**5 hướng ĐÓNG — đừng làm lại** (chi tiết + số liệu ở `references/plan.md`):
độ rộng `n_positions` · mục tiêu biến động + van drawdown · mở rộng bể chọn
(`min_coverage`) · cân theo độ mạnh bằng chứng · **vùng đệm thứ hạng `exit_frac`**.

⚠️ Riêng `exit_frac`: ghi chú cũ nói "giảm turnover mà không mất Sharpe" là từ cấu
hình **v1**. Trên v3 nó làm Sharpe **1,74 -> 1,48**. Đừng bật theo ghi chú cũ.

**1 hướng XÁC NHẬN — chi phí:** maker 0,378 -> 0,85 đáng **+0,06 Sharpe, +7,7% lợi
nhuận**. Đã làm: van trung lập trong lúc khớp thụ động (`order_router.fill_imbalance`),
`passive_wait_s` 300s -> **900s**, và chẩn đoán tỷ lệ maker vào `ExecutionRecord` +
`readiness_gate.py` để lượt sạch tới tự chỉ ra nút thắt.

⚠️ **Các thay đổi trên nằm trên ĐƯỜNG TIỀN và chỉ có hiệu lực sau khi NẠP LẠI daemon.**

## 🧭 Điều hướng — ĐỌC TRƯỚC KHI TÌM CODE
Repo này có skill điều hướng riêng. **Đừng grep dò dẫm** — nạp nó rồi tra bảng:

```
.claude/skills/aegis-nav/SKILL.md
```

Nó chứa router "triệu chứng → file:line" cho mọi vùng của hệ thống, cộng 5 file
reference nạp theo nhu cầu: `defects.md` (lỗi F1–F16), `money-path.md` (đường tiền),
`invariants.md` (bất biến + lệnh kiểm chứng), `codemap.md` (bản đồ tự sinh),
`plan.md` (kế hoạch P0/P1/P2).

## ⚠️ Trạng thái thật của hệ thống
- **558 test xanh KHÔNG có nghĩa chiến lược sinh lời.** Test khoá tính ĐÚNG (nhân quả,
  kế toán, bất biến danh mục), không khoá được EDGE. Edge chỉ đo được ngoài mẫu.
- F1–F44 đã vá. (Ghi chú cũ nói F14–F16 còn lại là SAI — `references/defects.md`
  ghi rõ cả ba đã vá từ lâu ở `live_pipeline.py` và `cpcv_pipeline.py`.)
- **F35 (14/09)**: `StrategyV2.backtest(mask=...)` cắt lưới TRƯỚC khi tính -> khởi động
  lại tầng gộp -> Sharpe holdout đọc nhầm 0,71 thay vì 1,28. `validate_v3.py` đã né lỗi
  này từ 10/09 nhưng **bản vá chỉ nằm trong script, không nằm trong thư viện**.
- **Sự cố thực thi 10/09/2026** sinh ra F21–F26: một ngoại lệ không bắt trong
  `submit_plan` làm hỏng cả lượt, để lại lệnh post-only sống 3h39 trên sàn và danh
  mục lệch **+35%** khỏi trung lập suốt 27 giờ.
- **Sự cố nhân đôi 11/09/2026** sinh ra F27–F28: báo giá lại đếm trùng phần đã khớp
  -> danh mục chạy **3,69x** thay vì 2,0x (F29: cổng chặn chỉ kiểm net nên không thấy). Đọc hai mục cuối `references/defects.md`
  trước khi sửa bất cứ thứ gì trong `oms/` hoặc `xs_live_pipeline.py`.
- **~20 file vẫn là stub** chỉ có docstring (`governance/l2_depth.py`,
  `validation/flat_plateau.py`, ...). Danh sách ở đầu `references/codemap.md`.
- **CHƯA ĐƯỢC BƠM TIỀN THẬT.** `scripts/readiness_gate.py` đang báo 0/3 lượt sạch.
  Chỉ mở cổng khi 3 lượt tái cân bằng liên tiếp đều sạch (~9 ngày ở chu kỳ 72h).
- **Holdout đã dùng 2 lần** (lần 2 chạy lại sau khi sửa lỗi đo lường, không đổi tham
  số nào). Bước tiếp theo bắt buộc: giao dịch giấy tiến về phía trước 4–8 tuần
  (`python scripts/run_daily.py --live` trên testnet), rồi `--costs` để đo tỷ lệ maker thật.

## Đường đi chiến lược v3 (một đường DUY NHẤT cho research và live)
```
dữ liệu -> research/signal_library.py     (26 tín hiệu / 5 họ, mỗi tín hiệu 1 giả thuyết)
        -> research/adaptive_combiner.py  (học dấu + trọng số từ QUÁ KHỨ, không đảo dấu tay)
        -> risk/portfolio.py              (chọn theo hạng -> trọng số -> trung lập -> trần)
        -> research/leverage.py           (mục tiêu biến động + van drawdown)
```
Backtest gọi `research/strategy_v2.py`; live gọi `xs_live_pipeline._compute_target_weights_v3`
— cả hai dùng **chung các hàm trên**. `tests/pipelines/test_research_live_parity_v3.py` khoá
bất biến này (sai số < 1e-9). Viết lại công thức ở tầng live = tái lập lỗi F3.

## Lệnh
```bash
python -m pytest -q                    # 558 test, ~167s
python scripts/healthcheck.py          # ⭐ CHẠY ĐẦU TIÊN — daemon còn SỐNG hay đã chết
python scripts/download_metrics.py     # tải dữ liệu vị thế (OI + long/short)
python scripts/positioning_study.py    # họ vị thế có đáng thêm không
# ⭐ CHẠY TRƯỚC MỖI LƯỢT — chặn lượt hỏng trước khi nó hỏng.
# PHẢI truyền cả --config lẫn --leverage: đòn bẩy không nằm trong file cấu hình,
# `run_daily.py` gán đè từ cờ CLI. Bỏ trống -> preflight kiểm bằng 2.0x mặc định
# trong khi daemon chạy 5.0x, và ba phép kiểm vốn/ký quỹ đo sai đại lượng.
python scripts/preflight.py --config artifacts/strategy_v3_wide.json --leverage 5.0
python scripts/breadth_study.py        # độ rộng danh mục (kết quả: ĐÓNG)
python scripts/risk_overlay_study.py   # mục tiêu biến động (kết quả: ĐÓNG)
python scripts/export_returns_v3.py    # tái tạo v3 + KIỂM CHỨNG parity, cache lợi suất
python scripts/goal_plan.py            # mục tiêu +5%/7 ngày: trả lời bằng số
python scripts/build_goal_policy.py    # giải sẵn chính sách DP ra artifact
python scripts/attribution.py          # quy kết từng nâng cấp (train)
python scripts/stability.py            # chọn cấu hình theo ĐỘ ỔN ĐỊNH, không theo đỉnh
python scripts/diagnose_oos.py         # chẩn đoán suy giảm ngoài mẫu
python scripts/refresh_spreads.py      # đo lại spread sổ lệnh thật
python scripts/trial_sweep.py          # đo variance_of_srs — chạy lại lưới cấu hình, ghi Sharpe từng lần thử
python scripts/dsr_report.py           # ⭐ PHÁN QUYẾT: Sharpe có vượt nhiễu chọn lọc của 250 lần thử không
python scripts/readiness_gate.py       # ĐÃ ĐƯỢC PHÉP BƠM TIỀN THẬT CHƯA?
launchctl list | grep aegis            # 2 job: com.aegis.trading + com.aegis.telegram
tail -f logs/aegis_daemon.log          # heartbeat mỗi 30 phút

# SAU MỖI LẦN VÁ CODE ĐƯỜNG TIỀN, BẮT BUỘC khởi động lại daemon:
for j in trading telegram; do
  launchctl unload ~/Library/LaunchAgents/com.aegis.$j.plist
  launchctl load   ~/Library/LaunchAgents/com.aegis.$j.plist
done
python scripts/gen_codemap.py          # sinh lại bản đồ code sau refactor
python scripts/check_docs.py           # kiểm tra tham chiếu file:line trong docs còn đúng
make agent-sync                        # chạy cả hai (làm sau mỗi lần refactor)
grep -rn "\[FIX F" src/                # xem các lỗ hổng đã vá và vá ở đâu
```

## Quy ước
- Tham số chiến lược **chỉ** đến từ `config/aegis_canonical_parameters.yaml` qua
  `load_canonical_config()`. Không hardcode hằng số trong code.
- Mọi tính tiền đi qua **đúng một** engine: `src/aegis/execution/pnl.py:compute_realized_pnl`.
- Chi phí futures phải đủ 4 khoản: **phí + funding + thanh lý + trượt giá**.
- Feature phải nhân quả (chỉ dùng dữ liệu `<= t`) và dùng **chung một hàm** cho
  train lẫn inference.
- Bình luận/tài liệu viết bằng tiếng Việt, khớp với code hiện có.
- Vá lỗi thì đánh dấu `[FIX Fxx]` ngay tại chỗ sửa.
- **Số vị thế là ràng buộc CỨNG** (`PortfolioSpec.n_positions`), không phải `top_frac`.
  Vốn $38 x min notional $5 => tối đa ~12 vị thế ở đòn bẩy 2x. Mọi so sánh backtest phải
  cố định số vị thế, nếu không ta chỉ đang đo tác dụng của đa dạng hoá.
- **Không tinh chỉnh thêm trên holdout hiện tại** — đã dùng 2 lần cho v3. Kiểm định sạch
  còn lại duy nhất là giao dịch giấy tiến về phía trước.
