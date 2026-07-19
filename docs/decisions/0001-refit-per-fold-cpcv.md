# ADR 0001: Refit Độc Lập Cho Từng Fold Trong CPCV 15-Fold
- **Quyết định**: Mọi bước từ A.3 FFD, B.2 Kalman đến E.1 Meta-Labeler đều phải refit lại từ đầu trên mỗi fold huấn luyện của CPCV.
- **Lý do**: Ngăn chặn rò rỉ tham số và phân phối dữ liệu tương lai vào tập OOS.
