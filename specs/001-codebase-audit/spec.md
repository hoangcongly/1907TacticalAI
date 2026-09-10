# Feature Specification: Codebase vs Specification Audit (001)

**Feature Branch**: `001-codebase-audit`

**Created**: 2026-07-23

**Status**: Draft

**Input**: User description: "Sử dụng kỹ năng speckit-specify để phân tích lỗi hiện có và dò xem đặc tả kĩ thuật đang sai sót ở điểm nào so với code hiện tại"

## Quant Research & Mathematical Context *(mandatory)*

Việc phát hiện sai lệch giữa Tài liệu đặc tả (Constitution) và Mã nguồn (Source Code) trong hệ thống HFT (High-Frequency Trading) là cực kỳ quan trọng. 
Mục tiêu là quét toàn bộ repository để đảm bảo:
1. Không có sự vi phạm Causal Integrity (Zero-Leakage).
2. Không có sự sai lệch về mặt cấu trúc Dữ liệu (`@dataclass(frozen=True)`).
3. Đảm bảo toàn bộ logic Sizing/Liquidation tuân thủ đúng toán học (Kelly Criterion).

*Ghi chú: Bài test hiện tại đã pass 99/99, tuy nhiên chúng ta cần một công cụ/task để kiểm tra liên tục những 'Gaps' (lỗ hổng) về mặt Feature mà Code chưa implement nhưng Spec đã yêu cầu.*

## System Scenarios & Testing *(mandatory)*

### Scenario 1 - Quét lỗ hổng Zero-Leakage (Priority: P1)

Hệ thống phân tích sẽ đọc toàn bộ file `src/aegis/` và phát hiện bất kỳ hàm nào sử dụng `shift(-1)` hoặc truy cập `iloc[i+1]` mà không thông qua Pre-Slice Protocol.

**Acceptance Scenarios**:

1. **Given** một file code bị lỗi Leakage, **When** chạy script Audit, **Then** hệ thống ném ra cảnh báo CRITICAL.

### Scenario 2 - Đối chiếu Constitution và Implementation (Priority: P2)

Hệ thống sẽ rà soát `constitution.md` và kiểm tra xem có nguyên tắc nào chưa được bao phủ bởi Test Suite (ví dụ: Tier 3 Circuit Breaker đã code chưa?).

**Acceptance Scenarios**:

1. **Given** một rule trong Constitution chưa có code thực thi, **When** chạy Audit, **Then** xuất ra danh sách "Unimplemented Features".

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Hệ thống MUST quét được toàn bộ các file `.py` trong `src/aegis`.
- **FR-002**: Hệ thống MUST so sánh các hằng số trong code với `aegis_canonical_parameters.yaml`.
- **FR-003**: Hệ thống MUST tạo ra báo cáo `audit_report.md` chi tiết các sai lệch.

## Success Criteria *(mandatory)*

### Measurable Outcomes (Quantitative)

- **SC-001**: Thời gian quét và phân tích codebase < 10 giây.
- **SC-002**: Báo cáo đầu ra có độ chính xác 100% trong việc phát hiện `shift(-1)` (Zero-leakage violations).

## Assumptions & Constants

- Mặc định toàn bộ code hiện tại đang chạy ổn định (đã Pass 99/99 Tests).
- Mục tiêu chính là tìm ra "Thiếu sót tính năng" (Gaps) thay vì "Lỗi Runtime".
