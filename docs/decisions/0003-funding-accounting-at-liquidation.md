# ADR 0003: Funding Accounting at Liquidation

## Lịch sử
- **Trạng thái:** Đã chấp thuận (Approved)
- **Ngày:** 2026-08-10

## Ngữ cảnh (Context)
Trong hệ thống tính toán PnL, hàm `compute_liquidation_loss` ở tầng `liquidation_layer` tính toán khoản lỗ tối đa khi thanh lý bằng `-margin`. Tuy nhiên, trong quá trình tính `net_pnl`, hệ thống đã thực hiện thêm phép trừ `funding_accrued_usd`. Điều này dẫn đến sự không thống nhất và có nguy cơ đếm kép (double-count) khoản phí funding đã được cấu thành vào quá trình bòn rút ký quỹ trước khi giá chạm mức thanh lý.

## Quyết định (Decision)
- **Không thu thêm phí funding ở nhánh thanh lý (Liquidation Branch).** Lỗ khi thanh lý chính là phần margin bị bốc hơi (bao gồm cả các khoản funding đã bào mòn margin). Trừ thêm một lần nữa ở bước cuối sẽ là đếm kép.
- Phí Taker (`fee_entry_cost`) vẫn được khấu trừ vì phí này phát sinh ngay từ lúc mở lệnh và được thu độc lập. 
- Không thu phí thoát lệnh (`fee_exit`) trong nhánh thanh lý để tránh đếm kép với Phí Thanh Lý Cưỡng Chế (Liquidation Clearance Fee). Phí thanh lý cưỡng chế (hiện tại được set 0.5% trong Canonical Registry) đã được cấu trúc trực tiếp vào giá thanh lý (Liquidation Price).

## Ví dụ bằng số (Numerical Example)
- Ký quỹ mở lệnh (Margin): 1000 USD (Leverage 10x, Notional 10,000 USD)
- Lệnh giữ trong 10 ngày, mỗi ngày trả funding 10 USD -> Tổng funding đã trả: 100 USD.
- Sàn liên tục trừ 100 USD này từ số dư ví (hoặc margin). Do đó, điểm thanh lý sẽ bị đẩy lùi lại gần giá mở lệnh hơn.
- Khi giá chạm ngưỡng thanh lý, khoản lỗ thực sự ghi nhận trên hệ thống là toàn bộ số margin còn lại bị quét sạch. Tổng lỗ từ đầu đến cuối chiến dịch trade này chính là 1000 USD (tương đương Margin cọc ban đầu).
- Nếu ta tính PnL = -(Margin) - Funding = -1000 - 100 = -1100 USD thì ta đã tính khoản 100 USD hai lần (vì 100 USD này vốn đã nằm trong 1000 USD Margin ban đầu bị bòn rút).

## Hệ quả (Consequences)
- `test_pnl.py` phải tuân thủ đúng logic này.
- Bất biến PnL được đảm bảo. Lỗi lệch PnL trong Kelly do funding đã được loại bỏ.
