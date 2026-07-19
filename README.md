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
Xem chi tiết tài liệu kỹ thuật tại `docs/architecture.md` và hướng dẫn xử lý sự cố tại `docs/runbook.md`.
