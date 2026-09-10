---
name: aegis-nav
description: Bản đồ điều hướng hệ thống Aegis trading. Dùng khi cần sửa lỗi, thêm tính năng, hoặc tìm code trong repo này — tra bảng để nhảy thẳng tới file:line thay vì grep dò dẫm. Kích hoạt với mọi câu hỏi về sizing, Kelly, PnL, funding, thanh lý, circuit breaker, CPCV, purge, labeling, HMM, feature, OMS, hoặc bất kỳ mã lỗi F1-F16 nào.
---

# Điều hướng hệ thống Aegis

Hệ thống giao dịch **perpetual futures USDⓈ-M** (KHÔNG phải margin spot).
Python, `src/aegis/`, ~9.5k dòng, 202 test.

## Quy tắc tiết kiệm token
1. **Tra bảng dưới trước — đừng grep dò dẫm.**
2. Chỉ nạp file reference thật sự cần (xem bảng "Nạp gì khi nào").
3. Mở code bằng `sed -n 'START,+40p' file` chứ đừng đọc cả file.
4. Sửa xong chạy `python scripts/gen_codemap.py` để cập nhật bản đồ.

## Nạp gì khi nào

| Tình huống | Nạp file |
|---|---|
| Sửa lỗi đã biết / hỏi "F5 là gì" | `references/defects.md` |
| Chạm vào tiền: PnL, phí, funding, sizing, thanh lý | `references/money-path.md` |
| Trước khi commit thay đổi logic giao dịch | `references/invariants.md` |
| Tìm hàm/class, không rõ ở đâu | `references/codemap.md` |
| Lập kế hoạch phần chưa làm | `references/plan.md` |

## Router: triệu chứng → đích

| Cần làm gì | Đi thẳng tới |
|---|---|
| Sizing / Kelly / đòn bẩy | `meta_labeling/sizing/kelly_empirical.py`, `execution/position_sizer.py` |
| PnL, phí, funding | `execution/pnl.py:12`, `governance/funding_accrual.py` |
| Giá thanh lý, MMR, đệm an toàn | `meta_labeling/sizing/liquidation_layer.py` |
| Cắt lỗ, trailing, thoát lệnh | `labeling/trailing_exit.py` |
| Gán nhãn, triple barrier, CUSUM | `labeling/triple_barrier.py`, `labeling/cusum_events.py` |
| Trọng số mẫu, độ trùng lặp | `labeling/sample_weights.py` |
| Cross-validation, purge, embargo | `meta_labeling/purged_kfold.py`, `validation/cpcv.py` |
| Sharpe / DSR / PBO | `validation/dsr.py`, `validation/pbo_cscv.py` |
| Chọn feature | `meta_labeling/feature_selection/` |
| Hiệu chỉnh xác suất | `meta_labeling/calibration.py` |
| Chế độ thị trường, HMM, Hurst | `features/regime/` |
| Kalman, IMM | `features/kalman/` |
| Sinh feature (nguồn gốc F3) | `features/signal_pipeline.py` |
| Ngắt mạch rủi ro, drift | `risk/circuit_breaker.py`, `risk/drift_monitor.py` |
| Đặt lệnh / gửi lệnh sàn | `oms/order_router.py` (post-only GTX, client id idempotent) |
| Vòng đời lệnh | `oms/state_machine.py` |
| Đối chiếu sổ sách | `oms/reconciliation.py` |
| Trọng số -> lệnh thật | `execution/portfolio_rebalancer.py` |
| **Vòng lặp vận hành** | `pipelines/xs_live_pipeline.py` + `scripts/run_daily.py` |
| Thuật toán khớp lệnh | `oms/order_router.py:execute_with_fallback` (maker rồi taker) |
| Trạng thái bền vững | `core/state_store.py` |
| Đo chi phí thực tế | `core/execution_log.py` (`run_daily.py --costs`) |
| Lọc universe | `data/universe.py` |
| Kỷ luật train/holdout | `research/holdout.py` |
| Nạp dữ liệu thật | `data/ingestion/binance_rest.py`, `binance_history.py` |
| **Cross-sectional / market-neutral** | `research/cross_sectional.py` ⭐ **nơi có edge** |
| **Panel đa tài sản** | `data/panel.py` |
| Kết nối sàn Binance | `data/ingestion/binance_rest.py`, `binance_history.py` |
| Khoá API | `core/credentials.py` (đọc `.env`) |
| Điều phối train | `pipelines/research_pipeline.py` |
| Điều phối backtest | `pipelines/cpcv_pipeline.py` |
| Vòng lặp giao dịch trực tiếp | `pipelines/live_pipeline.py:262` `on_bar()` |
| Tham số chiến lược | `config/aegis_canonical_parameters.yaml` + `core/config_loader.py` |
| Schema dữ liệu | `core/schemas.py` |

Tiền tố đường dẫn: `src/aegis/`

## 🆕 v3 — TRA BẢNG NÀY TRƯỚC (chi tiết: `docs/upgrade_v3_report.md`)

| Cần làm gì | Đi thẳng tới |
|---|---|
| Thêm/sửa tín hiệu (26 tín hiệu, 5 họ) | `research/signal_library.py` |
| Gộp tín hiệu, học dấu từ quá khứ | `research/adaptive_combiner.py` |
| Dựng trọng số, trung lập, chia đều rủi ro | `risk/portfolio.py` |
| Hiệp phương sai co (Ledoit-Wolf) | `risk/covariance_shrinkage.py` |
| Backtest có chi phí thật + trôi trọng số | `research/backtest_v2.py` |
| IC, suy giảm IC, **chênh lệch decile** | `research/ic_analysis.py` |
| Đòn bẩy, Kelly, xác suất đạt mục tiêu | `research/leverage.py` |
| Chiến lược đầu-cuối (research) | `research/strategy_v2.py` |
| Chiến lược đầu-cuối (live) | `pipelines/xs_live_pipeline.py:_compute_target_weights_v3` |
| Nạp panel + tổng hợp khung thời gian | `data/panel_v2.py` |

**Ba điều dễ sai nhất, đã trả giá để biết:**
1. Chi phí thật **4-5bp/chiều**, không phải 1bp.
2. **IC có thể ngược dấu với chênh lệch decile** — chọn tín hiệu bằng `decile_spread`.
3. **Số vị thế phải cố định** khi so sánh (`PortfolioSpec.n_positions`), nếu không ta
   chỉ đang đo tác dụng của việc nắm nhiều cặp hơn.

## ⭐ KẾT QUẢ NGHIÊN CỨU QUAN TRỌNG NHẤT
Chiến lược **directional một tài sản** (toàn bộ `pipelines/`) KHÔNG có edge:
t-stat 1.29 trên 6 năm BTC thật. Đừng tốn công tối ưu nó.

Edge nằm ở **cross-sectional market-neutral** (`research/cross_sectional.py`):
3 tín hiệu (funding carry, momentum 90, OFI flow) trên 61 cặp perp,
**Sharpe 1.71-2.36, t-stat 4.2-5.8, dương 7/7 năm**.
Cấu hình chốt: `artifacts/strategy_config.json`. Chi tiết: `plan.md`.

## Bối cảnh bắt buộc biết
- **24 file là stub rỗng** (chỉ docstring). Danh sách ở đầu `references/codemap.md`.
  Đừng đọc chúng để tìm logic — không có gì cả.
- **202 test xanh KHÔNG có nghĩa hệ thống chạy được.** Test chỉ phủ thư viện nghiên cứu;
  đường tiền tới sàn chưa tồn tại (F1).
- Lỗi **F1–F6 là mức T0** — vô hiệu hoá cả hệ thống, cần thiết kế lại chứ không vá vặt.
- Bar là **dollar-volume bar**, không phải nến thời gian. "120 nến" ≠ 120 phút.

## Lệnh hay dùng
```bash
python -m pytest -q                    # 202 test, ~60s
python scripts/gen_codemap.py          # sinh lại bản đồ
python scripts/check_docs.py           # kiểm tra docs chưa mục
make agent-sync                        # cả hai
grep -rn "\[FIX F" src/                # các lỗ hổng đã vá
```
