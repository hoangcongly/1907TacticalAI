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

## 📌 Cập nhật Giai đoạn (Hợp thức hóa Kiến trúc Kelly 2D)
*Ngày bàn giao: Cập nhật mới nhất*

### 4. Giải Thích Sự Chuyển Dịch Kiến Trúc Lưới Kelly: Từ 1D (Uniform Binning) sang 2D (Conditional Quantile Grid)
*   **File tham chiếu:** `src/aegis/meta_labeling/sizing/kelly_empirical.py`
*   **Lý do và Động lực Chuyển Dịch:**
    *   Trong tài liệu cũ (Mục 3.5.3 của Blueprint gốc), hàm `trade_records_to_kelly_table_inputs` được thiết kế dưới dạng trả về 3 mảng phẳng 1D vô hướng (không binning sẵn): `(ndarray, ndarray, ndarray)`. Việc bọc thô sơ bằng cách chia đều (Uniform Binning) như `min(int(math.floor(p * num_bins)), num_bins - 1)` gặp hạn chế chí mạng khi đối diện với đặc thù dòng tín hiệu từ HMM và CUSUM: dữ liệu trong thị trường thực tế phân phối phi tuyến cực độ (dính nháy ngập tràn ở các rìa xác suất gần $0.0$ và $1.0$, trong khi vùng ở giữa bị thưa thớt hoặc bỏ trống).
    *   **Giải pháp 2D Conditional Quantile Grid:** Thay thế hoàn toàn phân chia chia đều (Uniform) bằng **Lưới Lượng Tử Đồng Điều Kiện 2D (Conditional Quantile Binning)**. Lúc này, thị trường được cắt dọc theo phân vị ngang của tín hiệu xu hướng $p_i$, sau đó trong từng lát cắt, hệ thống tiếp tục phân vùng lượng tử theo tín hiệu nhiễu ngang $p_{\text{chop}\_i}$. Mỗi ô trong lưới 2D đều bảo đảm mật độ mẫu thực nghiệm ngang nhau (Equiprobable Bins), loại bỏ hoàn toàn các hố đen rỗng lệnh.
*   **Thay đổi Chữ ký & Cống Hiến Nhất Quán (API Breaking Change):**
    *   Hàm `trade_records_to_kelly_table_inputs` giờ đây trả về `Tuple[Dict[Tuple[int, int], np.ndarray], np.ndarray, List[np.ndarray]]` (gồm Từ điển ma trận lợi suất, mảng biên lượng tử của $p$, và Danh sách các mảng biên lượng tử của $p_{\text{chop}}$).
    *   **Cảnh báo tích hợp cho Track B:** Các hàm `build_empirical_kelly_table_v2` và `compute_bi_directional_kelly_v14_unified` BẮT BUỘC tiếp nhận thêm 2 tham số `p_edges` và `chop_edges_list`. Vui lòng truyền trọn vẹn cả 3 thành phần này từ bước pre-compute sang execution engine nhằm loại trừ hoàn toàn rò rỉ dữ liệu ngoài fold (Data Leakage) khi tra cứu inference O(1).

### 5. Cập nhật Blueprint (Master Architecture v11.9)
*   **File tham chiếu:** `docs/architecture.md`
*   **Hành động cần làm:** Toàn bộ Blueprint đã được đồng bộ hóa chi tiết với mã nguồn thực thi hiện kim (PnL, CUSUM, Execution, Kalman...). Khóa rào cản đạo hàm theo chỉ thị SOP Phase 2 đã được thẩm định an toàn tại `trade_mode.py`.

---

## 📌 Cập nhật Giai đoạn (Tiêm Chủng Toàn Diện 5 Lỗ Hổng Chí Mạng - Fatal Flaws Remediation)
*Ngày bàn giao: Cập nhật sau kỳ rà soát toàn hệ thống*

### 6. Đồng Bộ Ngưỡng Kích Thước Mẫu (Sample Size Threshold $= 5$)
*   **File tham chiếu:** `src/aegis/meta_labeling/sizing/kelly_empirical.py`
*   **Vấn đề & Giải pháp:** Trước đây hàm `solve_empirical_kelly_fraction` ràng buộc cứng `if len(returns_sample) < 30: return 0.0`. Khóa cản này vô tình tàn phá lưới Lượng Tử 2D (chẳng hạn lưới $10 \times 10$ yêu cầu số lượng mẫu lên đến hàng ngàn lệnh mới lọt được qua con số 30 cho từng bin con). Đã gia cố lại chuẩn mức ngưỡng `len(returns_sample) < 5`, hài hòa hoàn toàn với quy tắc thu nhỏ niềm tin tiên nghiệm (Bayesian Shrinkage, $C=20$).

### 7. Rào Chắn Lượng Tử Đơn Điệu Ngặt (Strictly Monotonic Quantile Boundaries)
*   **File tham chiếu:** `src/aegis/meta_labeling/sizing/kelly_empirical.py`
*   **Hành động cần làm:** Nhằm khắc phục hiện tượng trùng lặp ranh giới khi tín hiệu model đè trúng các mốc xác suất cực đại/cực tiểu (dẫn tới việc `np.searchsorted` gom cụm toàn bộ mẫu về 1 ô duy nhất), nay hệ thống đã trang bị vi sai tăng ngặt ($\text{edge}[k] > \text{edge}[k-1] + 10^{-12}$). Tra cứu O(1) nay vĩnh viễn chuẩn xác mà không gặp lỗi lệch ô hay bỏ thớt mẫu.

### 8. Hợp Đồng Dữ Liệu Ranh Giới Fold OOS (`boundary_truncated = True`)
*   **File tham chiếu:** `src/aegis/labeling/trailing_exit.py`
*   **Hành động cần làm:** Đã dỡ bỏ rào cản ném `ValueError` ngoại lệ đối với các lệnh tiến hành sát ranh giới cuối cùng của Fold thử nghiệm (nơi mảng `future_highs` bị cắt rỗng do ranh giới Pre-Slice Zero-Leakage). Theo đúng **Data Contract v11.9**, các sự kiện cạn kiệt chân trời thời gian này trả về ngay bản ghi TIME_STOP mang cờ `boundary_truncated = True`. Điều này bảo toàn chuỗi PnL liên tục cho hệ thống chấm điểm OOS PBO/DSR bên Track B mà không làm gián đoạn luồng thực thi.

---
*(Các lưu ý mới sẽ được Track A tiếp tục bổ sung vào đây sau mỗi đợt Push/Release)*

