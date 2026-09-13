# Nâng cấp v3 — báo cáo nghiên cứu và kết quả

Ngày: 2026-09-10. Toàn bộ số liệu tái lập được bằng các script trong `scripts/`.

---

## 0. Tóm tắt cho người bận

| Câu hỏi | Trả lời |
|---|---|
| Hệ thống cũ có thổi phồng kết quả không? | **Có.** Giả định phí 1bp/chiều; chi phí thật đo được **4–5bp**. Riêng điều này lấy đi **0.1–0.35 Sharpe**. |
| Nâng cấp nào thật sự có tác dụng? | **(1) Mở rộng universe** (27→41 cặp xếp hạng mỗi kỳ) — mạnh nhất và ổn định nhất ngoài mẫu (trung vị Sharpe holdout 0.51 → 0.94). **(2) Gộp tín hiệu thích ứng** (holdout 1.10 → 1.28 so với 4 tín hiệu cũ, cùng universe rộng). **(3) Chi phí đo thật** — không tăng lợi nhuận nhưng khiến mọi quyết định sau đó đúng. |
| Nâng cấp nào KHÔNG có tác dụng? | Trọng số liên tục/chia đều rủi ro (≈ hoà), khử beta (≈ hoà), MVO (**âm**), tổ hợp đa cấu hình (**âm**). Chi tiết ở §5. |
| Kết quả holdout của cấu hình v3? | Sharpe **1.28**, lợi suất **70.8%/năm**, maxDD 36.9%. Giữ lại **64%** Sharpe của train — mức lành mạnh. |
| Có phải một cấu hình may mắn? | Không. Trên **32 cấu hình** siêu tham số: Sharpe holdout **trung vị 1.24**, **100% dương**, 56% vượt 1.0, lợi suất năm trung vị **70.3%**. |
| So với hệ thống cũ, cùng điều kiện? | Cùng 12 vị thế, cùng chi phí thật, cùng chu kỳ: **0.63 → 1.28** (+103% Sharpe), lợi suất **28.3% → 70.8%/năm**. |
| Kỳ vọng hợp lý đi tới? | **Sharpe 1.0–1.3**, lợi suất **55–75%/năm** ở gross 1.0x, drawdown **35–45%**. Không phải 1.99 của train. |
| Mục tiêu +50%/tuần? | Cần Sharpe năm ≈ **51**. Ở đòn bẩy 10x: **P(đạt) = 17%**, P(cháy) = 5%, **trung vị = +0.3%**. Chi tiết §7. |

---

## 1. Vì sao con số cũ cao hơn thực tế

### 1.1 Giả định chi phí

Nghiên cứu cũ dùng `cost_rate = 0.0001` (1bp/chiều). Đo lại từ sổ lệnh Binance thật
(`scripts/refresh_spreads.py`, lấy mẫu 20 lần trong 30 phút trên 766 cặp):

| Thành phần | Giá trị thật |
|---|---|
| Phí sàn VIP0 | maker 2.0bp / taker 5.0bp |
| Nửa spread (trung vị universe) | ~0.65–2.1bp |
| Tác động giá ở $10–30/lệnh | **< 0.1bp** (không đáng kể) |
| **Tổng ở maker 50%** | **~4.1–4.5bp/chiều** |

Kết luận quan trọng về vốn nhỏ: **tác động giá bằng không**. Chi phí là phí sàn cộng
spread, và cần duy nhất để giảm nó là **khớp lệnh thụ động**, không phải giao dịch
ít đi.

### 1.2 Một sai lầm đo lường mà chính đợt này suýt mắc phải

Bản đầu của `estimate_cost_bps` dùng ước lượng spread Corwin-Schultz áp thẳng lên
nến 4h, cho ra trung vị **24.8bp** trong khi sổ lệnh thật là **0.65bp** — sai **40
lần**, đủ để kết luận nhầm rằng chiến lược đã chết. Corwin-Schultz chỉ có nghĩa trên
nến NGÀY. Hàm hiện tại tự tổng hợp lên khung ngày và chỉ dùng làm phương án dự phòng
khi không có spread thật (`backtest_v2.corwin_schultz_spread`).

---

## 2. Phát hiện phương pháp quan trọng nhất: IC và chênh lệch decile có thể **ngược dấu**

Kiểm tra quy ước dấu bằng tín hiệu ORACLE (đặt tín hiệu = đúng lợi suất tương lai):
IC = **+1.0000** và backtest Sharpe = **+27.15**; đảo dấu cho **−1.0000** và **−27.15**.
Cả hai engine đều đúng dấu. Vậy mà trên dữ liệu thật:

| Tín hiệu | IC | Chênh lệch decile | Kết luận |
|---|---|---|---|
| `low_idio_vol` | **+0.071** (t = 10.9) | **−0.191%**/ngày | ngược dấu nhau |
| `mom_slow` | **−0.016** (t = −2.3) | **+0.312%**/ngày | ngược dấu nhau |

Nguyên nhân: quan hệ tín hiệu–lợi suất **không đơn điệu**. IC đo độ đơn điệu trung
bình trên toàn mặt cắt ngang; danh mục top/bottom decile chỉ giao dịch **hai đuôi**.

Xem `low_idio_vol` theo nhóm phân vị (Q1 = biến động cao nhất):

```
Q1 +0.176%   Q2 -0.041%   Q3 -0.029%   Q4 -0.093%   Q5 -0.016%
```

Toàn bộ hiệu ứng nằm ở Q1. Nói cách khác, trên perp crypto, **bất thường "biến động
thấp thắng" của cổ phiếu bị đảo ngược ở đuôi**: nhóm biến động MẠNH NHẤT mới là nhóm
thắng.

**Hệ quả cho quy trình:** chọn tín hiệu bằng đúng đại lượng mình sẽ giao dịch. Danh mục
decile ⇒ tiêu chí là chênh lệch decile, không phải IC. Công cụ:
`ic_analysis.quantile_returns` / `decile_spread` / `monotonicity`.

---

## 3. Xử lý việc tín hiệu chạy ngược giả thuyết — không đảo dấu bằng tay

Có hai cách, chỉ một cách trung thực:

- **(a) Sai** — nhìn kết quả rồi đảo dấu. Biến train thành tập đã dùng để chọn; mọi
  Sharpe sau đó là ảo.
- **(b) Đúng** — để hệ thống tự học dấu và độ lớn từ dữ liệu QUÁ KHỨ tại mỗi thời điểm
  (`research/adaptive_combiner.py`).

Trọng số họ mà hệ thống tự học được (âm = nó tự đảo chiều):

| năm | carry | momentum | flow | volatility | microstructure |
|---|---|---|---|---|---|
| 2021 | 0.000 | 0.443 | 0.539 | 0.011 | 0.000 |
| 2022 | 0.035 | 0.352 | 0.278 | **−0.216** | 0.056 |
| 2023 | 0.148 | 0.329 | 0.165 | 0.163 | 0.165 |
| 2024 | −0.074 | 0.158 | 0.664 | −0.033 | 0.014 |

Ba lớp phòng vệ chống đuổi theo nhiễu: ngưỡng t-stat, co kiểu James-Stein, trần thay
đổi trọng số mỗi kỳ.

---

## 4. Tần suất: nhanh hơn KHÔNG tốt hơn

Đường cong suy giảm IC (chuẩn hoá `IC/√chân trời` — chân trời tối ưu là chỗ đạt cực đại):

| chân trời (nến 4h) | 1 | 3 | **6** | 12 | 24 | 48 |
|---|---|---|---|---|---|---|
| `IC/√h` | 0.0105 | 0.0130 | **0.0138** | 0.0118 | 0.0102 | 0.0083 |

Và chi phí thì tăng **tuyến tính** theo tần suất:

| chu kỳ tái cân bằng | 4h | 8h | 12h | 24h | 48h |
|---|---|---|---|---|---|
| Sharpe | 0.46 | 0.92 | 1.19 | **1.42** | 1.34 |
| chi phí/năm | **24.7%** | 16.6% | 12.8% | 8.2% | 4.8% |

Đã tải 170 cặp khung 1h (`scripts/download_wide_universe.py`) để kiểm tra giả thuyết
"tăng tần suất ⇒ tăng độ rộng ⇒ tăng IR". **Giả thuyết bị bác bỏ trên dữ liệu này**:
tái cân bằng theo nến 1h cho chi phí 27%/năm và Sharpe ≈ 0. Bộ tín hiệu hiện tại vốn
chậm (lookback 90–720 nến); muốn khai thác tần suất cao phải có tín hiệu vi cấu trúc
thật (sổ lệnh L2), không phải chạy nhanh hơn cùng một tín hiệu.

---

## 5. Bảng quy kết — số vị thế giữ CỐ ĐỊNH ở 12

Ràng buộc thật của tài khoản $38: min notional $5 × 12 vị thế = $60 ⇒ cần đòn bẩy ≥ 2x.
Mọi so sánh phải cùng số vị thế, nếu không ta chỉ đang đo tác dụng của đa dạng hoá.

```
cấu hình                                 vịthế     ann     vol  sharpe    t   maxDD  phí/năm     Δ
A. hệ thống cũ (phí 1bp GIẢ ĐỊNH)           11   34.8%   33.9%    1.03  2.24   30.7%     1.0%
B. + chi phí ĐO THẬT theo từng cặp          11   31.6%   33.9%    0.93  2.04   31.5%     4.2%  -0.09
C. + universe rộng (27->41 cặp)             11   54.4%   34.9%    1.56  3.39   21.6%     5.0%  +0.62
D. + thư viện 26 tín hiệu (gộp đều)         11   40.8%   36.1%    1.13  2.47   39.1%     4.2%  +0.20
E. + gộp THÍCH ỨNG                          11   60.9%   39.2%    1.55  3.38   38.4%     4.2%  +0.62
F. + trọng số liên tục & chia đều rủi ro    11   62.3%   43.0%    1.45  3.16   43.4%     4.3%  +0.52
G. + khớp thụ động 85%                      11   62.5%   39.2%    1.59  3.47   38.4%     2.6%  +0.66
```

### Một lỗi đo lường đã bị bắt và sửa

Bản đầu tiên của bảng này cho thấy trọng số liên tục nâng Sharpe **1.40 → 1.85**.
Kiểm tra lại: chế độ liên tục đang nắm **toàn bộ 59 cặp** thay vì 12 — gần như toàn
bộ phần tăng đến từ việc nắm nhiều vị thế hơn, thứ mà tài khoản $38 không mua được.
Đã sửa trong `risk/portfolio.py`: **tập tài sản LUÔN chọn theo thứ hạng trước**, mọi
chế độ trọng số chỉ khác nhau ở cách đánh trọng số TRONG tập đó. Sau khi sửa, phần
lợi ích của trọng số liên tục còn ≈ 0.

### Lỗi trung lập hoá đã sửa

Kẹp trần trọng số phá vỡ tính trung lập (đo được net exposure tới **20% gross** — đúng
cơ chế từng khiến danh mục live lệch 33% khỏi trung lập, ghi trong `plan.md`). Đã thêm
`portfolio.project_neutral`: chiếu ĐỒNG THỜI hai ràng buộc `Σw = 0` và `Σw·β = 0` bằng
phép chiếu trực giao, và **kết thúc bằng phép chiếu chứ không bằng kẹp trần**. Kết quả:
net exposure còn **1e-16**, đổi lại một cặp có thể vượt trần ~1% tương đối.

---

## 6. Kiểm định holdout

### 6.0 Một lỗi ĐO LƯỜNG đã suýt dẫn tới kết luận sai hoàn toàn

Lần chạy đầu cho holdout Sharpe **0.71** và kết luận "không qua kiểm định". Kiểm tra
lại thì lỗi nằm ở **chính script kiểm định**, không nằm ở chiến lược:

script cắt lưới thời gian về holdout **TRƯỚC** rồi mới tính tín hiệu. Hệ quả là tầng
gộp thích ứng khởi động lại từ con số 0 ở đầu holdout — 120 kỳ đầu chạy bằng trọng số
đều, và cửa sổ học chỉ còn ≤219 mốc thay vì 500. **Live không bao giờ gặp handicap
đó**: khi chạy thật, toàn bộ lịch sử luôn nằm trong tay.

Cách đúng là tính nhân quả trên toàn dòng thời gian rồi **CẮT SAU**. Đã kiểm chứng
cách này không rò rỉ tương lai: đổi **toàn bộ** dữ liệu sau một mốc bất kỳ làm sai
khác lợi suất trước mốc đó đúng bằng **0.00e+00**.

| cách đánh giá | Sharpe holdout | ann |
|---|---|---|
| cắt TRƯỚC (sai — gộp khởi động lại) | 0.71 | 35.0% |
| **cắt SAU (đúng — như live)** | **1.28** | **70.8%** |

Bài học: **một kết quả xấu bất thường phải nghi ngờ dụng cụ đo trước khi kết luận về
đối tượng đo.** Nếu dừng ở con số 0.71, ta đã vứt bỏ một chiến lược tốt.

### 6.1 Kết quả (đã sửa lỗi đo lường)

```
                          kỳ     ann     vol  sharpe  t-stat   maxDD  thắng
TRAIN  (đã dùng chốt)    580   82.6%   41.4%    1.99    4.35   28.7%  53.6%
HOLDOUT (chưa từng nhìn) 219   70.8%   55.2%    1.28    1.72   36.9%  52.5%

Sharpe giữ lại ngoài mẫu: 64%   (suy giảm 30-50% là bình thường và lành mạnh)
Deflated Sharpe (chiết khấu 250 phép thử):  DSR = 0.68   (dưới chuẩn vàng 0.95)
```

**Holdout theo năm — dương cả ba năm:**

| 2024 (15 kỳ) | 2025 (121 kỳ) | 2026 (83 kỳ) |
|---|---|---|
| +8.7% (Sharpe 1.78) | +28.6% (Sharpe 0.80) | +97.1% (Sharpe 1.72) |

### Chẩn đoán (`scripts/diagnose_oos.py`)

**H4 — holdout quá ngắn:** KTC 95% của Sharpe holdout là **[−0.18, 2.75]**, chồng lấn
với KTC train **[1.09, 2.90]**. Với 219 kỳ, khoảng tin cậy rộng tới mức chênh lệch
train/holdout **chưa có ý nghĩa thống kê**. Đừng đọc quá nhiều vào một con số đơn lẻ.

**H1 — có phải một cấu hình may mắn không? KHÔNG.** Phân phối Sharpe holdout qua 32
cấu hình siêu tham số:

```
trung vị 1.24 | trung bình 1.09 | min 0.08 | max 1.85
100% dương | 88% > 0.5 | 56% > 1.0 | lợi suất năm trung vị 70.3%
```

**Không có cấu hình nào âm.** Đây là bằng chứng mạnh hơn nhiều so với một con số đơn
lẻ: edge tồn tại trên cả một vùng tham số, không chỉ ở một điểm.

Theo chu kỳ tái cân bằng (Sharpe holdout trung vị): 48h → **1.48**, 72h → 1.32,
96h → 0.89, 144h → 1.10. Chu kỳ 48h tốt hơn lựa chọn 72h chốt trên train — nhưng
điều đó chỉ biết được SAU khi mở holdout, nên không được dùng để đổi cấu hình.

**H3 — universe rộng có phản tác dụng ngoài mẫu không?** Ngược lại, nó là nâng cấp
mạnh nhất và ổn định nhất:

| universe | Sharpe holdout (4 bộ tín hiệu × 2 chu kỳ) | trung vị |
|---|---|---|
| hẹp (59 cặp) | 0.17 … 1.77 (phương sai rất lớn) | 0.51 |
| **rộng (127 cặp)** | **0.52 … 1.28 (ổn định)** | **0.94** |

**H2 — đổi chế độ thị trường:** biến động holdout ~55%/năm so với train ~41%. Chiến
lược vẫn giữ được 64% Sharpe qua cú đổi chế độ đó.

### Chi phí — chiến lược sống được tới đâu (trên holdout)

| tỷ lệ maker | bp/chiều | ann | sharpe |
|---|---|---|---|
| 100% | 2.00 | 74.7% | 1.35 |
| 85% | 2.76 | 73.6% | 1.33 |
| 50% | 4.54 | 70.8% | 1.28 |
| 38% (testnet đo) | 5.15 | 69.9% | 1.27 |
| 0% (toàn taker) | 7.07 | 66.9% | 1.21 |

Chiến lược **không chết vì chi phí** ở mọi mức đã thử.

---

## 7. Mục tiêu +50% mỗi tuần — trả lời bằng số

Chiến lược toàn mẫu: Sharpe 1.56, lợi suất 68.3%/năm, biến động 43.7%/năm ở gross 1.0x.
Một tuần = 2.3 kỳ tái cân bằng. Đòn bẩy Kelly toàn phần = **3.6x**.

Mô phỏng block bootstrap 20.000 đường trên chính phân phối lợi suất thật (giữ đuôi dày
và cụm biến động; "cháy" = mất 70% vốn):

| đòn bẩy | ×Kelly | P(đạt +50%) | P(cháy) | **trung vị** | p5 | p95 |
|---|---|---|---|---|---|---|
| 1.8 (nửa Kelly) | 0.50 | 0.2% | 0.0% | **+0.6%** | −10.7% | +17.5% |
| 3.6 (Kelly) | 1.00 | 1.9% | 0.4% | **+1.1%** | −21.7% | +36.2% |
| 5 | 1.40 | 5.3% | 1.0% | **+1.3%** | −29.6% | +51.8% |
| 10 | 2.79 | 16.2% | 4.5% | **+0.1%** | −63.1% | +106.9% |
| 15 | 4.19 | 23.9% | 9.7% | **+0.0%** | −70.0% | +172.6% |
| 25 | 6.99 | 31.7% | 21.8% | **−0.0%** | −70.0% | +319.2% |
| 40 | 11.18 | 34.7% | 34.8% | **−9.7%** | −70.0% | +600.1% |

**Đọc cột trung vị, đừng đọc cột trung bình.** Ở đòn bẩy cao, trung bình bị kéo lên bởi
vài đường cực tốt trong khi trung vị đi xuống — trung bình mô tả một kết quả mà hầu như
không ai nhận được.

Ba sự thật toán học chi phối bảng này (`research/leverage.py`):

1. **Đòn bẩy không đổi Sharpe.** Nhân vị thế với L thì cả lợi suất kỳ vọng lẫn độ lệch
   chuẩn đều nhân L. Đòn bẩy chỉ di chuyển ta dọc một đường thẳng, không nâng lên đường
   cao hơn. Muốn lên đường cao hơn phải tăng Sharpe.
2. **Vượt quá Kelly thì tăng đòn bẩy làm GIẢM lợi nhuận dài hạn.** Tốc độ tăng trưởng log
   là `L·μ − L²σ²/2`, parabol úp ngược đỉnh tại `L* = μ/σ²`. Tại `2L*` tốc độ tăng trưởng
   bằng **0**; quá đó thì **âm**. Bảng trên cho thấy đúng điều đó: trung vị đạt đỉnh quanh
   5x rồi đi xuống, và ở 40x thì âm.
3. **Lỗ và lãi không đối xứng.** Mất 50% cần lãi 100% để hoà.

**Đảo ngược câu hỏi:** để đạt +50%/tuần với xác suất 50% ở biến động 60%/năm, cần Sharpe
năm ≈ **51** (≈ 3.042%/năm). Quỹ định lượng tốt nhất công khai chạy Sharpe 2–4. Sharpe
trên 10 chỉ tồn tại ở market-making tần suất cao với hạ tầng đặt cạnh sàn.

Cách duy nhất thật sự tồn tại để có +50%/tuần **lặp lại được** là mở rộng vốn trên một
edge đã chứng minh, không phải siết đòn bẩy trên $38.

---

## 8. Những gì KHÔNG có tác dụng (kết quả âm, ghi lại để không thử lại)

| Ý tưởng | Kết quả | Vì sao |
|---|---|---|
| MVO với hiệp phương sai co | −0.23 Sharpe | Khuếch đại sai số ước lượng; 12 vị thế thì cửa sổ hiệp phương sai không bao giờ đủ dài |
| Khử beta thị trường | −0.04 Sharpe | Beta danh mục chỉ giảm 0.159 → 0.125; với 12 vị thế, ràng buộc beta cứng đòi dịch chuyển trọng số quá nhiều |
| Trọng số liên tục / chia đều rủi ro | ≈ 0 | Sau khi cố định số vị thế, không còn lợi ích |
| Tăng tần suất tái cân bằng | Âm mạnh | Chi phí tăng tuyến tính, IC/√h giảm |
| Gộp trọng số theo IC | −2.18 Sharpe | Vì IC ngược dấu chênh lệch decile (§2) |
| Gộp 5 họ với trọng số ĐỀU | thua gộp thích ứng 0.4-0.7 Sharpe | Trung bình đều pha loãng họ tốt bằng họ đang chạy ngược |
| Tổ hợp đa cấu hình (ensemble) | 0.71 vs trung vị thành phần 0.77 | Các thành phần tương quan quá cao (cùng tín hiệu gốc) nên không đa dạng hoá được |

---

## 9. Việc còn phải làm

1. **Giao dịch giấy tiến về phía trước 4–8 tuần** trên dữ liệu chưa tồn tại. Holdout đã
   dùng 2 lần (lần 2 là chạy lại sau khi sửa lỗi đo lường, **không đổi tham số nào**);
   đây là kiểm định sạch duy nhất còn lại.
   ```bash
   python scripts/run_daily.py --skip-refresh          # dry-run, xem kế hoạch lệnh
   python scripts/run_daily.py --live                  # testnet, tiền giả
   python scripts/run_daily.py --costs                 # đo chi phí + tỷ lệ maker THẬT
   ```
2. **Đo tỷ lệ maker thật trên mainnet** (`core/execution_log.py`). Testnet đo 0.378 nhưng
   thanh khoản testnet là giả. Chênh giữa 38% và 85% đáng ~0.07 Sharpe và ~2.4bp/chiều.
3. **Tín hiệu vi cấu trúc thật** (`governance/l2_depth.py` vẫn rỗng) nếu muốn khai thác
   tần suất cao — đây là hướng duy nhất còn lại để tăng độ rộng.
4. **Không tinh chỉnh thêm trên holdout hiện tại.** Mỗi lần dùng lại làm giá trị chứng cứ
   giảm; đã dùng 2 lần cho v3.

---

## 10. Tái lập

```bash
python scripts/download_wide_universe.py --interval 1h --days 2400 --top 200 --min-qv 8000000
python scripts/refresh_spreads.py          # spread sổ lệnh thật
python scripts/attribution.py              # bảng §5
python scripts/lab.py --stage all --source 4h   # §1, §2, §4
python scripts/stability.py                # chọn cấu hình theo độ ổn định
python scripts/validate_v3.py              # §6, §7  (CHỈ CHẠY MỘT LẦN)
python scripts/diagnose_oos.py             # chẩn đoán §6
python -m pytest -q                        # 341 test
```


---

## 11. Áp dụng tài liệu nghiên cứu mới (11/09/2026) — KẾT QUẢ ÂM, có giải thích

Hai phương pháp được lấy từ tài liệu, cài đầy đủ, test nhân quả đầy đủ, và **cả hai
đều thua tầng gộp sẵn có**. Ghi lại để không ai tốn công làm lại.

### 11.1 CTREND / hồi quy mặt cắt ngang (Han-Zhou-Zhu; JFQA 2025)

Bài báo dùng 28 tín hiệu kỹ thuật trên **3.000+ coin**, gộp bằng hồi quy mặt cắt
ngang rồi lấy trung bình hệ số, tinh chỉnh bằng elastic net. Báo cáo long-short
quintile **3,87%/tuần**. Cài ở `research/xs_regression.py`.

| Cấu hình | Sharpe (train) |
|---|---|
| **Tầng gộp hiện có — 26 tín hiệu** | **2,05** |
| Tầng gộp hiện có — 5 họ | 1,32 |
| Hồi quy XS, 26 tín hiệu, ols / ridge / elastic-net | 0,06 / 0,18 / 0,34 |
| Hồi quy XS, 5 họ, cửa sổ beta 60 / 120 / 250 | 0,60 / 0,81 / **1,30** |

**Vì sao thua, và đây là điều đáng giá nhất rút ra:** phương pháp đòi ước lượng một
hệ số cho mỗi tín hiệu, tại mỗi kỳ, từ mặt cắt ngang của kỳ đó. Universe này có
**trung vị 41 tài sản giao dịch được mỗi kỳ** trên tập train. Hồi quy 26 hệ số từ 41
quan sát là bài toán gần bão hoà. Bằng chứng: kết quả cải thiện ĐƠN ĐIỆU khi cửa sổ
trung bình hệ số dài ra (60→0,60; 120→0,81; 250→1,30) — dấu hiệu kinh điển của ước
lượng quá nhiễu cần co mạnh. Mà co đủ mạnh thì hội tụ về gần chia đều, tức là quay
lại chỗ xuất phát.

Đã kiểm chứng giả thuyết "do độ phủ tín hiệu kém": **sai** — 40/41 tài sản có đủ cả
26 tín hiệu. Ràng buộc là ĐỘ RỘNG MẶT CẮT NGANG, không phải dữ liệu thiếu.

### 11.2 LambdaRankIC (arXiv 2605.00501)

Tối ưu trực tiếp thứ hạng thay vì hồi quy lợi suất, dùng LightGBM `lambdarank`.
Về lý thuyết rất hợp với hệ thống này, vì hàm mục tiêu chiết khấu theo vị trí nên
dồn sức học cho hai ĐUÔI — đúng chỗ danh mục đặt lệnh, và đúng chỗ mà §2 cho thấy
IC toàn mặt cắt ngang đánh lừa. Cài ở `research/rank_model.py`.

Kết quả: **Sharpe 0,19** (26 tín hiệu) và **−0,22** (5 họ). Cùng nguyên nhân: ~24k
hàng huấn luyện từ 41 tài sản × 580 kỳ là quá ít cho cây tăng cường, và mỗi kỳ chỉ
có 41 phần tử để xếp hạng.

### 11.3 Kết luận

**Ràng buộc đang chặn hệ thống là ĐỘ RỘNG, không phải độ tinh vi của thuật toán.**
Mọi phương pháp hiện đại trong tài liệu đều ngầm giả định một mặt cắt ngang rộng
(hàng nghìn tài sản). Ở 41 tài sản/kỳ, ước lượng viên càng nhiều tham số càng thua
ước lượng viên được co mạnh — và tầng gộp hiện có (5 con số, học từ chuỗi 580 quan
sát) chính là một ước lượng viên co mạnh, phù hợp với kích thước mẫu thật sự có.

Hai module được GIỮ LẠI kèm test đầy đủ: ràng buộc khiến chúng thua là kích thước
universe, không phải sai sót cài đặt. Universe rộng ra thì chúng dùng được ngay.

### 11.4 Hướng duy nhất còn lại có tiềm năng bậc độ lớn

Tăng số tài sản xếp hạng mỗi kỳ. Ba cách, theo thứ tự công sức:
1. Hạ ngưỡng thanh khoản / mở rộng universe Binance (41 → có thể 80-100 ở giai đoạn gần đây)
2. Thêm sàn: Bybit, OKX perp — mỗi sàn thêm vài trăm cặp
3. Dữ liệu sổ lệnh L2 cho tín hiệu vi cấu trúc (`governance/l2_depth.py` vẫn rỗng)

Đây là dự án HẠ TẦNG, không phải đổi thuật toán. Và ngay cả khi thành công, định luật
cơ bản nói IR tăng theo CĂN của độ rộng: gấp 4 lần số tài sản mới gấp đôi IR.


---

## 12. Basis trade / funding arbitrage (11/09/2026) — có edge, KHÔNG dùng được ở vốn này

Lớp chiến lược thứ hai, hoàn toàn khác cross-sectional: short perp + long spot cùng
tài sản, delta trung tính, thu funding. Tài liệu ghi nhận Sharpe 5-10, drawdown 0,6%.
Module: `research/basis_trade.py`. Kiểm định: `scripts/validate_basis.py`.

### 12.1 Đo được gì

36 cặp có đủ cả perp lẫn spot. Basis (perp/spot − 1): trung vị −0,048%, độ lệch 0,116%.

Lưới 36 cấu hình, **Sharpe holdout: trung vị 0,35 | min −1,38 | max 4,62 | 64% dương**.

Cấu trúc rõ ràng và hợp lý: **càng nhiều vị thế càng ổn định**, vì basis trade là thu
dòng tiền nhỏ đều và cần phân tán để triệt tiêu nhiễu basis.

| số vị thế | Sharpe holdout (trung vị) | vốn tối thiểu |
|---|---|---|
| 3 | **−0,45** | $30 |
| 5 | −0,37 | $50 |
| 8 | +1,03 | $80 |
| 12 | **+1,19** (max 4,62) | $120 |

### 12.2 Ba sai lầm đã mắc và tự bắt trong quá trình này

Ghi lại vì cả ba đều tạo ra kết quả đẹp giả tạo, và cả ba đều suýt được báo cáo:

1. **Sharpe 20-60 giả.** Tính Sharpe của chính dòng funding (gần như luôn dương) sau
   khi GIẢ ĐỊNH phòng hộ hoàn hảo — tức xoá rủi ro thật rồi đo rủi ro. Rủi ro của
   basis trade nằm ở basis, không ở funding.
2. **Nhìn trước.** Chọn top-k theo funding tại chính mốc t cho 25-43%/năm; dùng trung
   bình trượt nhân quả chỉ còn 5-9%/năm.
3. **Báo cáo một đỉnh nhọn.** Báo "Sharpe 3,88" từ MỘT cấu hình. Chạy hết lưới thì
   trung vị chỉ 0,35. Đúng lỗi đã cảnh báo ở §6 rồi tự mắc lại.

Và một sai lầm thứ tư về đòn bẩy: bảng "basis 20x = 55%/năm" là SAI. Chân spot phải
trả đủ tiền mặt, không lên đòn bẩy được. Vay margin USDT (5-15%/năm) lớn hơn cả lợi
suất 3,2%. Portfolio Margin yêu cầu vốn lớn.

### 12.3 Kết luận trên UNIVERSE ĐẦY ĐỦ: EDGE ĐÃ CHẾT TỪ 2025

Kết quả trên 36 cặp (trung vị Sharpe 0,35) là do chọn mẫu. Chạy lại trên **119 cặp**
có đủ perp + spot:

    Sharpe holdout: trung vị **−0,54** | min −1,33 | max 0,83 | **chỉ 17% dương**
    lợi suất năm holdout: trung vị −2,4%

Phân rã theo năm cho biết chính xác chuyện gì đã xảy ra:

| năm | tổng | funding | basis | Sharpe |
|---|---|---|---|---|
| 2023 | +4,8% | +5,2% | −0,3% | **5,85** |
| 2024 | +12,6% | +12,7% | +0,3% | **7,99** |
| 2025 | −0,3% | **−2,2%** | +2,3% | −0,06 |
| 2026 | −0,7% | **−1,4%** | +1,0% | −0,33 |

**Funding đổi dấu.** Chiến lược này hoạt động xuất sắc 2023-2024 rồi chết. Stress chi
phí xác nhận: Sharpe 0,05 ở 12bp/vòng, 0,00 ở 24bp — không còn gì để chi phí ăn.

Điều đáng nói nhất: tài liệu ĐÃ NÓI TRƯỚC điều này. Câu trích ngay khi bắt đầu hướng
nghiên cứu này: *"crypto carry Sharpe 6.45 (2020-2025), rơi còn 4.06 từ 2024, và ÂM
trong 2025."* Đã đọc, rồi vẫn đi đo như thể nó không tồn tại — và suýt kết luận
"chờ đủ $120 rồi bật".

**Bài học phương pháp:** khi tài liệu nói một edge đã suy giảm, việc đầu tiên phải làm
là PHÂN RÃ THEO NĂM, không phải đo trung bình toàn mẫu. Trung bình toàn mẫu của một
edge đã chết vẫn dương, vì quá khứ kéo nó lên.

**Kết luận:** đây KHÔNG phải ràng buộc vốn như §12.2 sơ bộ kết luận. Kể cả có $120
hay $10.000 thì chiến lược này vẫn âm từ 2025. Module giữ lại kèm 13 test để nếu
funding quay lại chế độ dương thì dùng được — nhưng phải KIỂM TRA LẠI THEO NĂM trước
khi bật, không được tin con số toàn mẫu.
