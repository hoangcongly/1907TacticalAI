# Research & Experimental Models Sandbox

## Cảnh Báo Ranh Giới Kiến Trúc
Thư mục `research/` chứa các mô hình thử nghiệm (TFT, XGBoost, RL PPO Sizing, Markov Scanners).
Mã nguồn tại đây **CHƯA qua kiểm định nghiêm ngặt CPCV 15-Fold / DSR / PBO**.
- **Tuyệt đối KHÔNG import từ `research/` ngược vào `src/aegis/`**.
- Khi một mô hình tại đây vượt qua kiểm định DSR >= 0.95 & PBO <= 0.40, phải thông qua Architecture Decision Record (ADR) để tái cấu trúc và đưa chính thức vào `src/aegis/`.
