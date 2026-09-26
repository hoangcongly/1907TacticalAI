# Ít lệnh hơn, ít đồng hơn, lời nhiều hơn ở vốn vài triệu VND — nghiên cứu và plan (26/09/2026)

Câu hỏi: vốn chỉ vài triệu VND, vào nhiều lệnh và nhiều đồng thì mỗi lệnh rất nhỏ — có
nên giảm số lệnh và số đồng để lời nhiều hơn không?

## Trả lời ngắn

1. **Lệnh nhỏ KHÔNG tự làm lời ít đi.** Phí Binance tính theo tỷ lệ (bp), không có phí cố
   định mỗi lệnh. Lệnh $10 và lệnh $1.000 trả cùng một tỷ lệ phí và ăn cùng một phần trăm
   lời. Ở vốn nhỏ, thứ thật sự gắn với cỡ lệnh chỉ là min notional $5: nó đặt sàn đòn bẩy
   (6n/vốn), bỏ vị thế và lệnh dưới $5 (F54, F55), và gây sai số làm tròn.
2. **Ít đồng hơn làm lời ít đi** ở mọi mức đòn bẩy (bảng dưới). Lý do đã đo trong repo:
   edge của chiến lược là ĐỘ RỘNG (`IR = IC × √n`); bỏ đồng làm biến động tăng nhanh hơn
   lời, đòn bẩy an toàn tụt, và ngắt mạch −30% bắn nhiều hơn.
3. **Ít lệnh hơn thì làm được, nhưng phải chọn đúng cách.** Giảm tần suất tái cân bằng
   (72h -> 144h) giảm một nửa số lệnh nhưng mất ~60% tiền lời. Cách rẻ nhất là nới **dải
   không giao dịch**: bỏ các lệnh chỉnh vị thế nhỏ, chỉ giữ lệnh vào/ra. Nay đã đo được
   đúng như live; con số cuối cùng cần chạy trên máy có dữ liệu.
4. **Đòn bẩy vừa phải lời nhiều hơn đòn bẩy cao.** Ở vốn 3 triệu, 12 vị thế ở 1,0–1,5x
   cho trung vị cao nhất vì ngắt mạch −30% ít bắn. Ở 2x nó bắn 81% số đường trong một năm.

## Ước lượng — vốn 3 triệu VND (~$114), chi phí thật 15,7bp/chiều, ngắt mạch như live

Nguồn: `artifacts/breadth_study.csv`, `artifacts/rebalance_period_study.csv` (toàn lịch
sử, đo ở chi phí mô hình), trừ thêm chi phí thật theo turnover đo ở live (0,85 × gross mỗi
kỳ), lợi suất Student-t df=4. **Đây là ước lượng, không phải backtest** (container không
có `data/`). Kill 30% coi là hấp thụ: chạm là dừng hẳn.

| cấu hình | đòn bẩy | có kill 30%: x vốn/năm | VND/tuần | P(kill 1 năm) | không ngắt mạch: x/năm | ~lệnh/tuần |
|---|---|---|---|---|---|---|
| 4 đồng, 72h | 1,0x | 0,99 | −500 | 90% | 1,46 | 7 |
| 6 đồng, 72h | 1,0x | 1,13 | ~6.900 | 62% | 1,54 | 10 |
| 8 đồng, 72h | 1,0x | 1,27 | ~14.000 | 35% | 1,61 | 13 |
| 10 đồng, 72h | 1,0x | 1,38 | ~18.700 | 17% | 1,69 | 16 |
| **12 đồng, 72h** | **1,0x** | **1,46** | **~22.100** | **10%** | 1,78 | 20 |
| **12 đồng, 72h** | **1,5x** | **1,44** | **~21.300** | 47% | 2,19 | 20 |
| 12 đồng, 72h | 2,0x | 1,21 | ~11.000 | 81% | 2,57 | 20 |
| 12 đồng, 48h | 1,5x | 1,29 | ~14.800 | 62% | 2,10 | 29 |
| 12 đồng, 144h | 1,5x | 1,15 | ~7.800 | 72% | 1,79 | 10 |

Đọc bảng:
- Mỗi hàng bớt đồng đều mất tiền lời. 8 đồng mất ~37%, 6 đồng mất ~69%, 4 đồng lỗ.
- 144h cắt một nửa lệnh nhưng mất ~63% tiền lời ở 1,5x.
- "Không ngắt mạch" lời nhiều hơn nhiều ở đòn bẩy cao, nhưng đó là chạy xuyên qua sụt giảm
  30–50% — quyết định của người vận hành, xem bước 5.
- VND/tuần tỷ lệ thuận với vốn: vốn 5 triệu thì nhân 5/3.

## Việc đã làm trong code

- **Mô phỏng giao dịch như live** (`research/trade_band.py`): dải không giao dịch, bỏ lệnh
  mở/tăng dưới min notional, cân lại trung lập (F42). Test đối chiếu thẳng với
  `build_rebalance_plan` trên 48 sổ ngẫu nhiên (`tests/research/test_trade_band.py`).
  `simulate_marked_to_market(..., no_trade_band=, min_trade=)` đếm số lệnh mỗi kỳ.
- **[FIX F55]** `build_rebalance_plan`: lệnh TĂNG bị bỏ vì dưới $5 thì vị thế cũ biến mất
  khỏi gross/net của kế hoạch, nên bước cân trung lập quyết định trên sổ sai. Ở vốn nhỏ
  nhánh này chạy thường xuyên. Đã vá, có test.
- **Dải không giao dịch cấu hình được**: khoá `"no_trade_band"` trong JSON. Thiếu khoá thì
  vẫn 0,20 như cũ, nên không đổi hành vi live.
- **`scripts/fewer_trades_study.py`**: đo mọi cách giảm lệnh và giảm đồng ở đúng vốn, chi
  phí thật, ngắt mạch như live, với luật chọn chốt trước: **ít lệnh nhất trong các cấu hình
  giữ được ≥95% tiền lời của cấu hình tốt nhất**. Ghi `artifacts/strategy_v3_lean.json`.

## PLAN

**Bước 1 — ngay hôm nay, không cần dữ liệu mới.** Ngừng chạy cấu hình 50 đồng ở vốn nhỏ.
Chuyển daemon sang V3 12 đồng đã kiểm định, đòn bẩy 1,5x:
```bash
# trong plist com.aegis.trading: --config artifacts/strategy_v3.json --leverage 1.5
python scripts/preflight.py --config artifacts/strategy_v3.json --leverage 1.5
```
rồi nạp lại daemon. Ở 3 triệu VND, mỗi vị thế trung bình ~$14, trên mức $5. Đây là
thay đổi đường tiền, bạn tự làm.

**Bước 2 — đo trên dữ liệu thật (máy Mac, ~30–60 phút).**
```bash
git pull
python scripts/fewer_trades_study.py --capital-vnd <vốn của bạn>
```
Giai đoạn A trả lời: nới dải không giao dịch tới đâu thì số lệnh/tuần giảm bao nhiêu và
mất bao nhiêu lời; 144h có còn thua ở chi phí thật không; làm mượt tín hiệu có đáng không.
Giai đoạn B trả lời: số đồng tối thiểu còn giữ được lời ở đúng vốn của bạn.

**Bước 3 — áp kết quả.** Script ghi `artifacts/strategy_v3_lean.json` và in đòn bẩy. Đổi
daemon sang file đó kèm `--leverage` đã in, chạy `preflight.py`, nạp lại daemon. Nếu làm
mượt tín hiệu thắng, script báo; nó chưa có ở live và cần nối code trước.

**Bước 4 — giảm chi phí mỗi lệnh** (lợi nhiều hơn giảm số lệnh). Chi phí thật 15,7bp gấp
~3,5 lần mô hình, phần lớn là trôi giá trong 900s chờ khớp và rơi xuống taker. Chờ số
`chase_block_p50_ratio` / `p90_ratio` của lượt tới (F49) rồi mới chỉnh `chase_sigma_mult`.
Kiểm tra thêm trong cài đặt phí của Binance: trả phí bằng BNB có được giảm không.

**Bước 5 — quyết định ngắt mạch (cần bạn chọn).** Kill 30% tính trên equity là thứ quyết
định lời ở vốn nhỏ. Lưu ý kill tự động KHÔNG đóng vị thế; khi có báo động phải `/kill`.
Giữ 30% thì đòn bẩy nên ≤1,5x. Muốn đòn bẩy cao hơn thì phải chấp nhận sụt giảm sâu hơn.

**Không nên làm** (đã đo, thua): giảm xuống dưới 8 đồng; chỉ giữ đồng "chắc nhất" theo độ
mạnh tín hiệu (`sizing_mode_study`: Sharpe 2,26 -> 1,79); chỉ đánh đồng rẻ phí (Sharpe
2,23 -> 0,59); vùng đệm thứ hạng `exit_frac` (1,74 -> 1,48, turnover chỉ giảm 13%); đòn
bẩy cao để bù ít đồng (n=4 có Kelly 1,44x so với 3,54x của n=12).

**Đòn bẩy lời lớn nhất vẫn là VỐN.** Lời tăng tuyến tính theo vốn và không kèm hình phạt;
đòn bẩy thì nhân rủi ro bậc hai.
