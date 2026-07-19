# ADR 0002: Thay Thế Công Thức Kelly Nhị Phân Bằng Empirical Kelly Log-Growth
- **Quyết định**: Sử dụng Brent's method tối đa hóa kỳ vọng E[log(1 + f*r)] trên phân phối lợi nhuận thực nghiệm OOS, kết hợp bảng tra cứu tách Follow/Fade qua classify_trade_mode.
- **Lý do**: Lệnh có Trailing-Exit có phân phối liên tục, bất đối xứng, không phù hợp với công thức nhị phân f* = p - (1-p)/b.
