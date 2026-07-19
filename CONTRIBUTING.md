# Contributing Guidelines — Aegis Trading System

## Quy ước Code & Kiến trúc (v11.8)
1. **src/aegis/ là ranh giới bất khả xâm phạm**: Chỉ chứa mã nguồn đã vượt qua kiểm định thống kê nghiêm ngặt (CPCV 15-Fold, DSR >= 0.95, PBO <= 0.40, Parity Check == 0 sign-flips).
2. **Tuyệt đối không import từ `research/` vào `src/aegis/`**.
3. **Mọi thay đổi siêu tham số chiến lược (`strategy_selection`)** phải được đăng ký log vào `ExperimentTracker` để tính đúng số lần thử nghiệm $N_{DSR}$.
