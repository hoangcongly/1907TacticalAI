# Aegis Trading System — Institutional Trend-Following Platform (v11.8)

## Quickstart
- Cài đặt package chế độ phát triển:
  ```bash
  pip install -e .
  ```
- Kiểm tra toàn bộ unit test và parity check:
  ```bash
  make test
  make parity-check
  ```
- Chạy hệ thống live/paper:
  ```bash
  python main.py --config config/environments/paper.yaml --strategy config/strategies/trend_following_v1.yaml
  ```
- Xem thiết kế kỹ thuật, lý thuyết toán học AFML và mã nguồn lõi tại `docs/architecture.md` (Master Blueprint v11.8).
- Xem hợp đồng dữ liệu giữa Track A (Signal) và Track B (Labeling) tại `CONSTRACT.md` (Data Contracts v11.9).
- Xem hướng dẫn xử lý sự cố khẩn cấp tại `docs/runbook.md`.
