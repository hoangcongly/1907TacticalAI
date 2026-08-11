# ADR 0004: Embargo Unit - Bar vs Time

## Lịch sử
- **Trạng thái:** Đã chấp thuận (Approved)
- **Ngày:** 2026-08-10

## Ngữ cảnh (Context)
Trong quá trình chia fold (Purged K-Fold Cross Validation), thuật toán cần cách ly (embargo) các điểm dữ liệu sau dải test (test set) để ngăn rò rỉ thông tin (đặc biệt là nhãn HMM hoặc biến động giá) sang tập huấn luyện (training set). Trong sách Marcos Lopez de Prado, khoảng thời gian embargo thường được định nghĩa bằng một lượng thời gian cố định ($h$). Tuy nhiên, đối với thị trường crypto hoạt động 24/7 và biến động mạnh theo số lượng giao dịch, mật độ thông tin có thể thay đổi rất nhiều trong một khoảng thời gian cố định.

## Quyết định (Decision)
- **Đơn vị Embargo (Embargo Unit) bắt buộc tính bằng nến (bars/indices), không tính bằng thời gian tuyệt đối (ms/hours/days).**
- Tham số `embargo_bars` được lưu trong Canonical Registry và có **mức độ ưu tiên cao nhất**.
- Bất kỳ module chia fold nào (như `PurgedKFold`) cũng phải tịnh tiến số index của nhãn theo `embargo_bars` thay vì dùng timedelta cộng vào timestamp.

## Hệ quả (Consequences)
- `PurgedKFold` sẽ sử dụng trực tiếp số lượng bars để loại bỏ nến sau tập test.
- Ngăn chặn lỗi khi có khoảng trống (gaps) dữ liệu (ví dụ bảo trì sàn) làm sai lệch kích thước cách ly.
- Cải thiện tốc độ tính toán (chỉ cần thao tác trên mảng số nguyên của index thay vì date arithmetic).
