# Sổ Cân Bằng Hằng Số & Tham Số Chiến Lược Chuẩn Mực (`Aegis Canonical Parameter Registry — Single Source of Truth`)

Tài liệu này và tệp cấu hình song hành [`config/aegis_canonical_parameters.yaml`](file:///Users/hoangcongly/1907TacticalAI/aegis-trading-system/config/aegis_canonical_parameters.yaml) là **Chốt Chặn Duy Nhất (`Single Source of Truth`)** khóa chặt mọi thông số kiến trúc và tham số chiến lược cốt lõi của Hệ thống Giao dịch Thuật toán Định chế Aegis (`Aegis Trading System v11.9`).

Mọi thay đổi trong mã nguồn (`src/`), bộ kiểm thử (`tests/`) hoặc tài liệu báo cáo (`docs/`) **BẮT BUỘC** phải đối chiếu và tuân thủ tuyệt đối các giá trị được chốt tại đây. Cấm tự ý thay đổi tham số âm thầm trong quá trình refactor hoặc viết lại tài liệu.

---

## I. Bảng Đối Chiếu Tham Số Chiến Lược Cốt Lõi (`Strategic & Meta-Labeling Parameters`)

| Tham số | Giá trị chuẩn (`Canonical`) | Đơn vị / Kiểu | Nguồn gốc (`Origin`) | Ý nghĩa & Lý do Định chế (`Rationale`) |
| :--- | :---: | :---: | :--- | :--- |
| **`t_max_live_follow`** | `120` | `int` (bars) | `v11.7 C.2` / `trend_following_v1.yaml` | Thời gian sống tối đa cho vị thế theo xu hướng (`Follow Mode`). Cho phép gồng lời chạy dài (`let winners run`) tối đa 120 nến trước khi kích hoạt `TIME_STOP`. |
| **`t_max_live_fade`** | `40` | `int` (bars) | `v11.7 C.2` / `trend_following_v1.yaml` | Thời gian sống tối đa cho vị thế đánh chặn hồi quy (`Fade Mode`). Giới hạn trong 40 nến vì lệnh Fade đánh trong sideway/choppy; giữ quá lâu làm loãng giả thuyết ban đầu và tăng rủi ro bị cắt bởi biên fold CPCV. |
| **`embargo_bars`** | `24` | `int` (bars) | `v3/v4` / `v11.5` / `architecture.md` | Số nến cách ly cố định ngay sau mỗi `test fold` trong `PurgedKFold` nhằm triệt tiêu tự tương quan HMM ($24 \text{ bar} = 24 \text{ giờ}$). Có **Thứ tự ưu tiên cao nhất (`Override Priority`)**, không bị biến dạng bởi hàm `max()` hay kích thước mẫu $N$. |
| **`embargo_pct`** | `0.0` | `float` (ratio) | `v11.9 Canonical Alignment` | Tỷ lệ cách ly phụ trợ. Mặc định bằng `0.0`. Chỉ được sử dụng làm phương án dự phòng cuối cùng khi `embargo_bars is None` và `autocorrelation_lag_threshold = 0`. |

---

## II. Bảng Hằng Số Kiến Trúc (`Architectural Constants`)

| Hằng số | Giá trị chuẩn (`Canonical`) | Kiểu dữ liệu | Nguồn gốc (`Origin`) | Ý nghĩa & Lý do Định chế (`Rationale`) |
| :--- | :---: | :---: | :--- | :--- |
| **`n_states`** | `2` | `int` | `v11.6 Task B-1-2` | Số lượng trạng thái HMM cố định là 2 (`Trend` và `Chop`). Bọc thép xác định toán học $p_{\text{trend}} + p_{\text{chop}} = 1.0$, triệt tiêu hoàn toàn bẫy đoán mò mô hình (`State Guessing Trap`). |
| **`m_sl_follow`** | `2.0` | `float` | `v11.6` / `trend_following_v1.yaml` | Hệ số nhân ATR rào cản cắt lỗ (`Stop-loss multiplier`) cho chế độ `Follow`. |
| **`m_sl_fade`** | `1.5` | `float` | `v11.6` / `docs` | Hệ số nhân ATR rào cản cắt lỗ (`Stop-loss multiplier`) cho chế độ `Fade`. Rào cản hẹp hơn `Follow` để bảo vệ tài khoản khi đánh chặn đảo chiều. |

---

## III. Bảng Bảo Vệ Vi Cấu Trúc Sàn Giao Dịch (`Microstructure & Liquidation Guards`)

| Tham số | Giá trị chuẩn (`Canonical`) | Đơn vị / Kiểu | Nguồn gốc (`Origin`) | Ý nghĩa & Lý do Định chế (`Rationale`) |
| :--- | :---: | :---: | :--- | :--- |
| **`spoofing_discount`** | `0.70` | `float` (ratio) | `v11.6 Task B-2-2` | Hệ số chiết khấu thanh khoản ảo L2 (`Phantom Liquidity Discount`). Giả định $30\%$ tường lệnh L2 là ảo (`Spoofing`) sẽ bị rút trước khi giá quét tới. |
| **`safety_buffer_pct`** | `0.15` | `float` (ratio) | `v11.9 Liquidation Layer` | Khoảng cách đệm an toàn tối thiểu $15\%$ giữa giá Cắt Lỗ ban đầu (`sl_initial`) và giá Thanh Lý cưỡng chế (`p_liq`). |
| **`MARGIN_LEVERAGE_CAP`** | `20.0` | `float` | `v11.9 Liquidation Layer` | Trần đòn bẩy an toàn tối đa cho phép trong mọi phép tính toán của hệ thống. |

---

## IV. Cơ Chế Kiểm Định & Hải Quan Bọc Thép (`TDD Verification Protocol`)

Để bảo đảm các tham số này không bao giờ bị xê dịch hoặc trôi tự do giữa các đợt phát triển, hệ thống được nghiệm thu tự động thông qua bộ kiểm thử định chế [`tests/core/test_canonical_params.py`](file:///Users/hoangcongly/1907TacticalAI/aegis-trading-system/tests/core/test_canonical_params.py). Bài test này thực hiện:
1. Đọc trực tiếp tệp gốc `config/aegis_canonical_parameters.yaml`.
2. Kiểm tra chéo với tham số mặc định của `PurgedKFold` ([`purged_kfold.py`](file:///Users/hoangcongly/1907TacticalAI/aegis-trading-system/src/aegis/meta_labeling/purged_kfold.py)).
3. Kiểm tra chéo với cấu hình YAML chiến lược ([`trend_following_v1.yaml`](file:///Users/hoangcongly/1907TacticalAI/aegis-trading-system/config/strategies/trend_following_v1.yaml)).
4. Khẳng định 100% sự đồng nhất toán học trên toàn hệ thống trước khi xuất cấu hình sang Rust RTK hoặc chạy trực tiếp thực chiến.
