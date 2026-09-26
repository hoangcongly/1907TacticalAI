# Forex với vốn vài triệu VND: nghiên cứu và plan (26/09/2026)

Câu hỏi: vốn chỉ vài triệu VND, nên đánh forex bằng chiến lược nào để lời nhiều nhất?

**Trả lời ngắn.** Không có chiến lược forex nào có bằng chứng sinh lời đáng kể ở vốn
này. Ba lý do độc lập, và chỉ cần MỘT lý do là đủ để dừng:

1. **Pháp lý và đối tác.** NHNN không cấp phép sàn forex nào; chuyển tiền ra nước ngoài
   để giao dịch forex không thuộc giao dịch được phép; tranh chấp thì luật không bảo
   vệ. Lừa đảo "sàn forex" là loại vụ án tài chính lớn nhất VN gần đây.
2. **Edge đã chết.** Trên dữ liệu tỷ giá thật của Fed 1976–09/2026, bốn chiến lược FX
   kinh điển (lấy nguyên tham số từ bài báo, không dò) đều có Sharpe **ÂM** sau chi
   phí bán lẻ từ 1999 tới nay, và âm cả TRƯỚC chi phí từ 2010.
3. **Lô tối thiểu.** 0,01 lot ≈ $1.000–1.300 notional. Với 3 triệu VND (~$114), MỘT lệnh
   nhỏ nhất đã là 8,8x vốn. Danh mục đa dạng hoá ở 10%/năm cần ~383 triệu VND (lô 0,01)
   hoặc ~38 triệu VND (lô 0,001). Đánh 1 lệnh micro duy nhất thì biến động 76%/năm và
   **mất trắng tài khoản** trong giai đoạn 2010–nay.

Tái lập: `python scripts/fx_small_capital_study.py` (cần
`git clone --depth 1 https://github.com/datasets/exchange-rates ../datasets/exchange-rates`).

---

## 1. Pháp lý và rủi ro đối tác ở Việt Nam

- NHNN: không cấp phép bất kỳ sàn forex nào; thanh toán, chuyển tiền ra nước ngoài cho
  sàn forex là giao dịch không được phép; pháp luật không bảo vệ khi tranh chấp.
  Nguồn: [VnExpress](https://vnexpress.net/ngan-hang-nha-nuoc-khong-cap-phep-bat-ky-san-forex-nao-tai-viet-nam-4882559.html),
  [Luật sư Việt Nam](https://lsvn.vn/ngan-hang-nha-nuoc-khong-cap-phep-bat-ky-san-forex-nao-tai-viet-nam-a157359.html),
  [Thanh Niên — chặn thanh toán cho sàn forex](https://thanhnien.vn/chan-thanh-toan-hoat-dong-cua-san-giao-dich-ngoai-hoi-forex-185250505161320207.htm).
- Nghị định 88/2019/NĐ-CP: sử dụng tài khoản ngoại tệ ở nước ngoài không đúng quy định,
  hoặc chuyển tiền cho giao dịch không đúng quy định, bị phạt 80–100 triệu đồng, có thể
  tịch thu tiền. Nguồn: [Thư viện pháp luật](https://thuvienphapluat.vn/van-ban/Tien-te-Ngan-hang/Nghi-dinh-88-2019-ND-CP-xu-phat-vi-pham-hanh-chinh-trong-linh-vuc-tien-te-va-ngan-hang-428666.aspx).
- Lừa đảo: vụ Mr Pips (Phó Đức Nam) — 2.661 bị hại, thu giữ/phong toả hơn 5.200 tỷ
  đồng, đề nghị truy tố 75 bị can (03/2026); vụ Shark Bình khởi tố 10/2025. Nguồn:
  [Công an Hà Nội](https://congan.hanoi.gov.vn/phan-hoi-thong-tin-bao-chi/cong-an-ha-noi-triet-pha-bang-nhom-lua-28868),
  [Nhân Dân](https://nhandan.vn/khoi-to-bi-can-doi-voi-nguyen-hoa-binh-shark-binh-va-cac-doi-tuong-co-lien-quan-ve-hanh-vi-rua-tien-post953081.html).

Hệ quả cho plan: rủi ro lớn nhất ở vốn nhỏ KHÔNG phải chọn sai chiến lược, mà là mất
toàn bộ tiền vào một "sàn" không rút được.

## 2. Tỷ lệ thắng thua của nhà giao dịch bán lẻ

| nguồn | kết quả |
|---|---|
| ESMA 2018, đo trên nhà cung cấp CFD/forex ở EU | **74–89%** tài khoản bán lẻ thua lỗ; từ đó EU giới hạn đòn bẩy 30:1 cho cặp chính ([ESMA](https://www.esma.europa.eu/sites/default/files/library/esma50-162-215_product_intervention_analysis_cfds.pdf)) |
| Chague, De-Losso & Giovannetti (2019), toàn bộ day trader hợp đồng tương lai Brazil | **97%** người kiên trì >300 ngày thua lỗ; 1,1% kiếm hơn lương tối thiểu; **không có bằng chứng học được** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101)) |

Giao dịch ngắn hạn/scalping là nơi chi phí ăn hết edge. Plan này loại chúng ngay từ đầu.

## 3. Chiến lược FX có bằng chứng trong tài liệu

| chiến lược | nguồn | bằng chứng gốc | tình trạng |
|---|---|---|---|
| Quy tắc kỹ thuật (MA, filter) | Neely, Weller & Ulrich (2009), JFQA | lời thật ở 1970s–80s | **biến mất từ đầu 1990s** ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1403345)) |
| Theo xu hướng (TSMOM) | Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017) | Sharpe TB ~0,4/thị trường, 67 thị trường, 1880–2016 ([AQR](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing)) | đa tài sản; riêng FX suy yếu (xem §4) |
| Momentum mặt cắt ngang | Menkhoff, Sarno, Schmeling & Schrimpf (2012), JFE | chênh tới 10%/năm | **nhạy chi phí**, dồn vào đồng đắt để giao dịch ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1809776)) |
| Carry (lãi suất cao − thấp) | Lustig, Roussanov & Verdelhan (2011), RFS; Brunnermeier, Nagel & Pedersen (2008) | có phần bù rủi ro | **đuôi trái dày**: sập khi thanh khoản cạn ([NBER](https://www.nber.org/papers/w14473)) |

Bản tái lập công khai trên 6 đồng G10, 1995–2024 ([philippweder/fx-factor-strategies](https://github.com/philippweder/fx-factor-strategies)),
TRƯỚC chi phí bán lẻ: carry Sharpe 0,39–0,47 (skew −0,8 đến −1,0), momentum 0,03–0,24,
gộp các nhân tố ngoài mẫu 0,18–0,52. Carry không kiểm được ở §4 vì không có dữ liệu
lãi suất; ở tài khoản bán lẻ nó đến qua swap qua đêm, mà sàn cộng thêm phần chênh.

## 4. Kiểm chứng trên dữ liệu thật: Fed H.10, 9 đồng chính so với USD, 1976–09/2026

Không dò tham số. Cùng biến động 10%/năm. Chi phí bán lẻ: 1–1,5bp cặp chính, 2,5bp NZD,
5bp SEK/NOK; swap cộng thêm 1%/năm trên notional. Sharpe **sau** chi phí (trong ngoặc:
trước chi phí):

| chiến lược | 1976–nay | 1999–nay | 2010–nay | 2020–nay |
|---|---|---|---|---|
| TSMOM 12 tháng | 0,17 (0,38) | −0,10 (0,09) | −0,43 (−0,23) | −0,44 (−0,25) |
| TREND 1/3/12 tháng | 0,28 (0,47) | −0,11 (0,07) | −0,47 (−0,28) | −0,51 (−0,34) |
| XS-MOM 1 tháng | −0,09 (0,32) | −0,60 (−0,20) | −0,50 (−0,06) | −0,65 (−0,20) |
| MA 50/200 | 0,26 (0,48) | −0,01 (0,18) | −0,34 (−0,12) | −0,38 (−0,16) |

MaxDD của mọi dòng 41–84% ở 10%/năm. Độ nhạy của TREND 2010–nay: trước chi phí −0,28,
chỉ spread −0,32, thêm swap 0,5% là −0,40, thêm swap 1% là −0,47. **Chi phí không phải
nguyên nhân chính. Edge theo giá của FX chính đã hết từ 2010.** Kết quả khớp với
Neely et al. (2009): lợi nhuận có thật ở giai đoạn đầu, rồi biến mất.

⚠️ Giới hạn: lợi suất giá giao ngay, không gồm chênh lãi suất (xem §3). Dữ liệu ngày,
nên không đo được chiến lược trong ngày. Kết luận không phụ thuộc vào hai điểm này: trong
ngày còn tệ hơn vì chi phí (§2), và carry có đuôi sập.

## 5. Bài toán lô tối thiểu ở vốn nhỏ (TREND, 10%/năm, 2010–nay)

Vốn 3 triệu VND (~$114). Danh mục muốn trung bình **$19/cặp**, trong khi 0,01 lot ≈
$1.000–1.300.

| cỡ lệnh nhỏ nhất | số cặp giữ được | đòn bẩy | biến động thật | maxDD |
|---|---|---|---|---|
| chia nhỏ tuỳ ý (lý tưởng) | 9 | 1,5x | 10% | −59% |
| 0,001 lot (100 đơn vị) | 0,2 | 0,2x | 4% | −13% |
| 0,01 lot (micro) | 0 | 0 | 0 | 0 |
| **ép vào 1 lệnh 0,01 lot, cặp mạnh nhất** | 1 | **9,4x** | **76%** | **−100%** |

Một lệnh micro duy nhất: 38% số năm mất quá nửa vốn, vốn cuối/đầu 2010–nay = **0**.
Ở 10 triệu VND: 2,8x, biến động 23%, maxDD −85%, vốn còn 0,26.

Vốn tối thiểu để giữ đủ 9 cặp ở 10%/năm: **~383 triệu VND** với lô 0,01, **~38 triệu
VND** với lô 0,001.

## 6. Ra tiền, trường hợp TỐT NHẤT

Giả sử lạc quan rằng carry + trend gộp lại cho Sharpe 0,5 SAU phí bán lẻ (cao hơn mọi
con số đo được ở §4 từ 1999). Vốn 3 triệu VND:

| biến động | lãi kỳ vọng/năm | VND/tuần | P(lỗ sau 1 năm) |
|---|---|---|---|
| 10% | 5,0% | ~2.600 | 33% |
| 20% | 10,0% | ~4.600 | 34% |
| 50% (Kelly, trần lý thuyết) | 25,0% | ~7.200 | 40% |

Với Sharpe đo được từ 2010 (−0,47): không mức đòn bẩy nào cho kỳ vọng dương.

## 7. So với hệ thống crypto đang có

Hệ thống v3 (crypto perp, trung lập thị trường) có Sharpe đo được 1,1–1,7 ở chi phí mô
hình, ước lượng ~0,8–1,6 ở chi phí thật 15,7bp — vẫn cao hơn hẳn mọi chiến lược FX ở §4. Ở vốn 1 triệu VND, ước lượng trung vị
~4.000–16.000 VND/tuần (CLAUDE.md, mục 26/09). Forex không cải thiện được con số đó.

⚠️ Crypto cũng có rủi ro pháp lý: Nghị quyết 05/2025/NQ-CP (hiệu lực 09/09/2025) thí
điểm thị trường tài sản mã hoá, và theo các phân tích pháp lý, giao dịch phải tập trung
tại đơn vị được Việt Nam cấp phép ([Thư viện pháp luật](https://thuvienphapluat.vn/van-ban/Tien-te-Ngan-hang/Nghi-quyet-05-2025-NQ-CP-trien-khai-thi-diem-thi-truong-tai-san-ma-hoa-tai-Viet-Nam-672252.aspx),
[ATA Legal](https://ata-legal.com/nghi-quyet-05-2025-nq-cp-tien-so-tien-ao-chinh-thuc-duoc-cong-nhan-la-tai-san-nhung-phai-duoc-giao-dich-tap-trung-tai-don-vi-do-viet-nam-cap-phep)).
Nên hỏi luật sư trước khi bơm tiền thật vào bất kỳ sàn nước ngoài nào.

---

## PLAN

### Khuyến nghị chính: KHÔNG chuyển vốn vài triệu sang forex

Mọi bằng chứng ở trên cùng chỉ một hướng. Cách dùng vốn nhỏ tốt hơn:
1. Giữ hệ thống v3 hiện có ở cấu hình đúng cho vốn nhỏ (`artifacts/strategy_v3.json`,
   ~1,9x; chạy `scripts/small_capital_study.py` trên máy có dữ liệu), sau khi đã làm rõ
   pháp lý.
2. Đòn bẩy thật duy nhất là VỐN: lợi nhuận tăng tuyến tính theo vốn, không kèm hình
   phạt. Tăng vốn từ thu nhập đều đặn mạnh hơn mọi chiến lược.

### Nếu vẫn muốn thử forex: các bước có cổng dừng

**Bước 0 — loại trừ (bắt buộc).**
- Không nạp tiền vào "sàn" giới thiệu qua Zalo/Telegram/Facebook, "chuyên gia", nhóm
  tín hiệu, quỹ uỷ thác, copy-trade hứa lợi nhuận. Đây đúng là kịch bản vụ Mr Pips.
- Không dùng EA/robot bán trên mạng, martingale, lưới (grid), scalping, đánh tin.
- Hiểu rõ: NHNN không cấp phép sàn forex nào; chuyển tiền ra nước ngoài để giao dịch
  forex không được phép; có tranh chấp thì tự chịu.

**Bước 1 — demo 3 tháng, không tiền thật.**
- Chiến lược duy nhất đáng thử: danh mục đa dạng carry + trend trên các cặp chính, tái
  cân bằng tuần hoặc tháng, biến động mục tiêu ≤10%/năm. Không đánh một cặp đơn lẻ.
- Ghi lại chi phí THẬT của sàn: spread lúc vào lệnh, swap qua đêm hai chiều từng cặp.

**Bước 2 — cổng quyết định. Chỉ đi tiếp khi đủ CẢ BA:**
1. Sàn cho đặt lệnh đủ nhỏ để giữ đủ danh mục ở ≤10%/năm với vốn của bạn: lô 0,001
   cần ~38 triệu VND, lô 0,01 cần ~383 triệu VND (§5).
2. Chạy lại `scripts/fx_small_capital_study.py` với chi phí và swap đo ở bước 1 mà
   Sharpe giai đoạn 2010–nay vẫn > 0,3. Hiện tại là −0,47.
3. Kết quả demo không tệ hơn backtest cùng kỳ quá 1 độ lệch chuẩn.

**Bước 3 — tiền thật (chỉ khi qua bước 2).**
- Biến động mục tiêu 10%/năm, tổng đòn bẩy ≤ 3x, không bao giờ tăng cỡ lệnh sau thua.
- Dừng hẳn khi sụt 25% từ đỉnh và quay lại bước 1.
- Kỳ vọng hợp lý: vài nghìn VND/tuần trên 3 triệu VND (§6).
