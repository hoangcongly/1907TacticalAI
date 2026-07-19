"""Phân loại Trial Classes: model_fitting, strategy_selection (tính vào N_DSR), production_fit."""

from enum import Enum


class TrialClass(Enum):
    """
    Phân loại các lượt chạy thử nghiệm (trials) để tính toán DSR (Deflated Sharpe Ratio)
    và lưu vết mô hình.
    """

    MODEL_FITTING = "model_fitting"
    """Quá trình fit mô hình máy học/thống kê (không tính vào PBO/DSR penalty)."""

    STRATEGY_SELECTION = "strategy_selection"
    """Các vòng lặp tinh chỉnh tham số chọn chiến lược (được đếm để phạt PBO)."""

    PRODUCTION_FIT = "production_fit"
    """Quá trình huấn luyện mô hình cuối cùng trên toàn bộ tập dữ liệu để mang ra live."""
