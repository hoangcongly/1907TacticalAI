# Aegis System Audit: Backlog vs Codebase

Dựa trên cấu trúc file hiện tại trong `src/aegis/` và kết quả bộ Test (99 Passed), dưới đây là đối chiếu chi tiết giữa Backlog v11.8 và tình trạng Code thực tế.

## 🟢 Track A (Khang: Data & Signal Pipeline)

### Hoàn thành (Đã có code & Test pass)
- **[A-0-1] & [A-0-2]**: `ExperimentTracker` và `trial_classes.py` đã có trong `src/aegis/core/`.
- **[A-1-1] đến [A-1-3]**: Outlier/Bad tick detection đã được bạn vừa pull về tại `src/aegis/data/outlier_detection.py`.
- **[A-1-4] & [A-1-5]**: Kalman Replacer đã có tại `src/aegis/data/cleaning/tick_kalman_replacer.py`.
- **[A-2-1] đến [A-2-5]**: Các hàm về Dollar Volume và Thresholds đã có trong `src/aegis/data/bars/`.
- **[A-3-1] & [A-3-2]**: Local Linear Trend (`local_linear_trend.py`) và Gap handling đã có.
- **[A-4-1] đến [A-4-5]**: Fractional Differentiation đã có tại `features/fractional_diff.py`.
- **[A-5-1]**: Single Gaussian Port đã có trong `features/regime/hmm_causal.py`.
- **[A-5-4] & [A-5-5]**: Causal HMM Posterior và Hurst Exponent đã có.
- **[A-6-1]**: Rolling GHE đã có tại `features/regime/ghe.py`.
- **[A-7-1] đến [A-7-5]**: IMM Kalman đã có tại `features/kalman/imm_kalman.py`.
- **[A-8-1] & [A-8-2]**: Diagnostics và Signal Pipeline đã có.

### 🔴 Cần thực hiện (Stub/Placeholder)
- **[A-5-2] `fit_hmm_2state_loglik`**: Đang là Placeholder trả về `0.0`. Đây là **Blocker cực lớn** vì toàn bộ IMM Kalman và Regime Labeling phụ thuộc vào độ chính xác của HMM.
- **[A-5-3] `validate_two_regime_architecture_bootstrap`**: Phụ thuộc vào [A-5-2], chưa thể hoạt động thực tế dù có thể đã khai báo hàm.

---

## 🟢 Track B (Lý: Labeling & Decision Pipeline)

### Hoàn thành (Đã có code & Test pass)
- **[B-1-10]**: `TRADE_RECORD_SCHEMA` hoàn thiện tại `schemas.py`.
- **[B-1-1] đến [B-1-13]**: Toàn bộ luồng Kelly Sizing, Classify Trade Mode, Trailing Exit và Kelly Table đều đã code xong tại `meta_labeling/sizing/` và `labeling/`. 
- **[B-1-14]**: `PurgedKFold` hoàn thiện cực tốt tại `meta_labeling/purged_kfold.py`.
- **[B-2-1] & [B-2-2]**: CUSUM thresholds và Limit Queue Sim đã có.
- **[B-3-1]**: Risk Budget Correlation đã có tại `risk/covariance_shrinkage.py`.
- **[B-4-1] đến [B-4-5]**: CUSUM Filter và Triple Barrier động đã có tại `labeling/`.
- **[B-5-x] & [B-6-x]**: Consensus (MDI/MDA/SFI), Bootstrap Forest và Isotonic Calibration đã có toàn bộ tại `meta_labeling/feature_selection/` và thư mục gốc của nó.
- **[B-7-1] đến [B-7-5]**: Toàn bộ luồng Validation (CPCV, DSR, PBO, Flat Plateau) đã có tại `validation/`.
- **[B-11-x]**: Circuit Breaker và Drift Monitor đã hoàn thiện tại `risk/`.

### 🟡 Cần Rework / Đấu nối (Đã Stub)
- **[B-8-4] `compute_realized_pnl` (Bản thật)**: Đang dùng Stub PnL giả lập. Cần đấu nối (rewire) logic khớp lệnh thật vào B-1-9 theo như ghi chú của bạn.

---

## 🎯 Tổng kết: Nhiệm vụ tối mật tiếp theo

Dựa trên Backlog và tình trạng Codebase:
1. **Ưu tiên 1 (CRITICAL)**: Thực thi **Task A-5-2** (Hàm `fit_hmm_2state_loglik` qua thuật toán EM). Đây là móng vuốt của toàn bộ hệ thống Regime.
2. **Ưu tiên 2**: Thực thi **Task A-5-3** (Parametric Bootstrap LRT) ngay sau khi có HMM.
3. **Ưu tiên 3**: Rework **Task B-8-4** để đấu nối PnL thật (Tính toán Market Impact + Fee + Limit Queue) thay vì Stub.
