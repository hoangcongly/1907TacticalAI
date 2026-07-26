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
---
*(Cac luu y moi se duoc Track A tiep tuc bo sung vao day sau moi dot Push/Release)*


---

## CANH BAO KHAN CAP: 4 LO HONG TIEM AN DA DUOC VAC BINH (Bug Patch Round 2)
*Ngay ban giao: 26/07/2026*

> [!CAUTION]
> Phan nay mo ta 4 loi da duoc phat hien va sua trong phien kiem toan tiep theo (26/07/2026).
> Track B can doc ky va cap nhat bat ky code nao phu thuoc vao cac interface duoi day.

---

### Bug Fix #1 — Xoa Import Kep & Import Thua (`kelly_empirical.py`)
**File tham chieu:** `src/aegis/meta_labeling/sizing/kelly_empirical.py`
**Dong da sua:** 5-6

**Van de da sua:**
- Import kep `brentq` hai lan lien tiep (dong 5 va dong 6) gay nhap nhem khi bao tri.
- `minimize_scalar` duoc import nhung khong duoc su dung o bat ky dau trong toan bo file.

**Hanh dong can lam:** Khong co breaking change nao voi Track B. Tuy nhien, neu bat ky script nao cua Track B dang import
`minimize_scalar` gian tiep tu `kelly_empirical.py` (vi du `from aegis.meta_labeling.sizing.kelly_empirical import minimize_scalar`),
hay xoa dong do va import truc tiep tu `scipy.optimize` thay the.

---

### Bug Fix #2 — Double-Counting Mau Soft Mode Gay Sai Lech Bayesian Kelly (`kelly_empirical.py`)
**File tham chieu:** `src/aegis/meta_labeling/sizing/kelly_empirical.py`
**Ham bi anh huong:** `build_regime_returns_dict(..., assignment_mode='soft')`
**Dong da sua:** 183-186

> [!WARNING]
> Day la loi toan hoc nghiem trong nhat trong dot kiem toan nay. Toan bo he thong dinh gia von Bayesian
> dua tren ket qua cua ham nay co the bi sai lech neu goi theo soft mode.

**Van de da sua:**
Khi `p_trend_val == 0.5` (thi truong luong lu hoan toan), logic cu `if ... if ...` (hai menh de if rieng biet)
khien mot lenh bi chen vao **ca hai** `trending_list` va `choppy_list` cung mot luc. Ket qua:

| Tac dong | Mo ta |
|----------|-------|
| `n_samples` bi phong ao | Ca hai bucket bao cao nhieu mau hon thuc te |
| Trong so Bayesian `w = N/(N+C)` bi boc pham sai | Tin tuong "gia tao" vao du lieu thi truong dong |
| `f_bayesian` cuoi cung bi lenh khoi pham | Phep tron `f = sum(prob * f_bayesian)` cho ket qua sai |

**Hanh dong can lam:** Neu Track B co bat ky script rieng nao goi `build_regime_returns_dict` voi `assignment_mode='soft'`,
ket qua tu phien chay truoc co the bi sai lech. Nen chay lai toan bo qua trinh xay dung bang Kelly tren du lieu huan luyen
moi nhat de dam bao ket qua f_bayesian hoi tu chinh xac.

**Hanh vi moi (chinh xac):** Khi `p_trend_val == 0.5`, lenh duoc phan loai vao `trending_list`
(theo quy tac da so >= 0.5). Khong bao gio con tinh trang mot lenh bi dem 2 lan.

---

### Bug Fix #3 — `CircuitBreaker.peak_equity` Khoi Tao Bang 0 Gay Vo Hieu Bao Ve Drawdown Sau Restart
**File tham chieu:** `src/aegis/risk/circuit_breaker.py`
**Dong da sua:** 25-36 (`__init__`)

> [!WARNING]
> BREAKING CHANGE: Chu ky `CircuitBreaker(...)` da duoc them tham so `initial_equity: float = 0.0`.
> Gia tri mac dinh = 0.0 nen tuong thich nguoc (backward compatible) nhung nen cap nhat chu dong.

**Van de da sua:**
Truoc day `peak_equity = 0.0` khong co cach nao cung cap von ban dau thuc te khi khoi tao.
Khi bot restart giua phien giao dich (crash/reboot) voi `current_equity` = 8,000 (dang lo tu dinh 10,000),
`peak_equity` se reset ve 8,000 thay vi giu nguyen 10,000 → toan bo co che drawdown protection
(3 tier thresholds 5%/10%/15%) **bi vo hieu hoa hoan toan** vi co so tinh drawdown bi sai.

**Hanh dong can lam cho Track B:**
- Khi khoi tao `CircuitBreaker` sau moi lan restart/reboot bot, **bat buoc truyen `initial_equity`
  bang gia tri equity hien tai doc tu database/state persistence**:
  ```python
  # TRUOC (sai - mat bao ve sau restart):
  cb = CircuitBreaker(tier1_threshold=0.05, ...)

  # SAU (dung - bao ve drawdown chinh xac sau restart):
  cb = CircuitBreaker(initial_equity=account.current_equity, tier1_threshold=0.05, ...)
  ```
- Neu Track B dang luu trang thai `CircuitBreaker` qua JSON/pickle, hay dam bao `peak_equity`
  cung duoc serialize va truyen lai vao `initial_equity` khi deserialize.

---

### Bug Fix #4 — Tick Gia NaN Lot Qua `detect_bad_tick_core` Nhu Good Tick (`outlier_detection.py`)
**File tham chieu:** `src/aegis/data/outlier_detection.py`
**Ham bi anh huong:** `detect_bad_tick_core`
**Dong da sua:** 83-92

> [!CAUTION]
> Day la loi causal pipeline nghiem trong. Tick NaN lot qua pipeline tin hieu nhu Good Tick se gay
> tinh toan chac chan sai lech tai tat ca cac module o ha luu: HMM, CUSUM, Kalman Filter.

**Van de da sua:**
Khi `prices[i]` hoac `prices[i-1]` la `NaN`:
- `diff_price = NaN - float = NaN`
- `NaN > 5.0 * sigma = False` (NumPy quy tac)
- Ket qua: `is_bad_tick[i]` = False — tick NaN LOT QUA nhu Good Tick

Hau qua ha luu: TickLevelKalmanReplacer van xu ly dung (vi `np.isnan` check trong Numba engine),
nhung `is_bad_tick` array tra ve cho cac module khac (HMM emission, signal bar builder) se bao cao
tick NaN la Good Tick, gay nhiem loai tin hieu tai tap hop du lieu OOS.

**Hanh vi moi (chinh xac):** Neu `prices[i]` hoac `prices[i-1]` la `NaN`,
`is_bad_tick[i]` = **True** ngay lap tuc ma khong can kiem tra Dieu Kien 1/2/3.

**Hanh dong can lam:** Khong co breaking change voi interface nao cua Track B.
Tuy nhien, ket qua `is_bad_tick` array tu `detect_bad_tick_core` tu nay se phan loai
tick NaN la `True` thay vi `False` nhu truoc. Neu Track B co bat ky su dung nao
gia dinh tick NaN la Good Tick, can cap nhat logic xu ly tuong ung.


