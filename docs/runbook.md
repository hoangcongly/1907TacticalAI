# Runbook — Quy Trình Xử Lý Sự Cố Khẩn Cấp

## 1. Khi Drawdown Circuit Breaker Kích Hoạt
- **Tier 1 (DD >= 5%)**: Hệ thống tự động giảm quy mô vị thế xuống 50%. Kiểm tra `logs/experiments.jsonl`.
- **Tier 2 (DD >= 10%)**: Đóng 100% vị thế hiện tại (Flatten), dừng mở lệnh mới 24h.
- **Tier 3 (DD >= 15%)**: Kill Switch tuyệt đối. Cần kiểm tra thủ công trước khi restart lại dịch vụ.

## 2. Khi CUSUM Brier Score Drift Alarm Báo Động
- Chạy `python scripts/retrain_walkforward.py` để refit lại cấu hình production và refresh lại `cusum_thresholds.json`.
