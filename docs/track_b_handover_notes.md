# TRACK B HANDOVER NOTES
*Tài liệu này lưu trữ các lưu ý quan trọng từ Track A (Data & Signal) gửi cho Track B (Labeling & Decision) để ngăn chặn lỗi tích hợp (Integration Errors) giữa các phân hệ.*

---

## 📌 Cập nhật Giai đoạn P(-1): Hạ tầng Lõi (Core Infrastructure)
*Ngày bàn giao: 19/07/2026*

### 1. Sử dụng chung Enum `TrialClass`
*   **File tham chiếu:** `src/aegis/core/trial_classes.py`
*   **Hành động cần làm:** Khi Track B thực thi `kelly_empirical.py` hoặc các thuật toán tính PnL, nếu có lưu log thử nghiệm, hãy **bắt buộc import** `TrialClass` từ đường dẫn trên. Tuyệt đối không tự định nghĩa lại các cờ trạng thái dưới dạng string thô để tránh bộ lọc DSR đếm sai số lần thử nghiệm (gây ra lỗi PBO penalty sai lệch).

### 2. Ghi Log qua Cổng duy nhất `ExperimentTracker`
*   **File tham chiếu:** `src/aegis/core/experiment_tracker.py`
*   **Hành động cần làm:** Lớp này đã được bọc kiến trúc `Singleton`. Khi Track B chạy Walk-Forward đa luồng (Multi-threading), các bạn chỉ cần khởi tạo `tracker = ExperimentTracker()`. Hệ thống sẽ tự động điều phối luồng ghi file bằng Thread Lock mà không sợ kẹt I/O. Vui lòng **không tự viết hàm `open(file, 'a')` thủ công** để tránh làm hỏng file JSONL chung.

### 3. Bổ sung Data Fields trong File Log JSONL (Tính Truy Xuất Nguồn Gốc - Reproducibility)
*   **Thay đổi Interface:** Theo chuẩn SOP mới nhất, file log `JSONL` xuất ra giờ đây sẽ **gắn cứng thêm 2 keys bắt buộc** là `"git_commit"` và `"env_versions"` (chứa version của các thư viện lõi như numpy, polars, numba). 
*   **Hành động cần làm:** Nếu Track B đang hoặc sẽ viết các đoạn script đọc/parse file log JSONL để tính toán hoặc trực quan hóa lên Dashboard, vui lòng cập nhật lại schema đọc (Reader Schema) để đón nhận 2 trường dữ liệu mới này mà không bị quăng lỗi `KeyError` hoặc lệch định dạng cấu trúc JSON.

---
*(Các lưu ý mới sẽ được Track A tiếp tục bổ sung vào đây sau mỗi đợt Push/Release)*
