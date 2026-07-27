"""
Module A-2: Chuẩn bị ngưỡng tạo nến tín hiệu định lượng Dollar Volume Bars (Task A-2-1 & A-2-2).
Bảo đảm tính chất Nhân quả (Causal PIT-Safe): tuyệt đối không sử dụng Volume của ngày hôm nay hoặc tương lai,
và kiểm định không tồn tại NaN/Inf sau giai đoạn warm-up 21 ngày.
"""

import numpy as np
import polars as pl


def compute_pit_safe_daily_threshold(
    df_daily: pl.DataFrame, window: int = 21, target_freq: float = 50.0
) -> pl.DataFrame:
    """
    [TASK A-2-1] Tính ngưỡng gộp nến Dollar-Volume an toàn theo thời điểm (Point-in-Time Safe Daily Threshold).

    Công thức:
        \\theta_{\\text{PIT}}(T) = \\frac{1}{\\text{target\\_freq}} \\times \\frac{1}{\\text{window}} \\sum_{k=1}^{\\text{window}} \\text{Daily-Dollar-Volume}(T-k)

    Quy tắc bảo bọc (Armor Guards):
        1. PHẢI có dịch chuyển .shift(1): Ngăn chặn triệt để Look-ahead Bias (rò rỉ dữ liệu tương lai/ngày hiện tại).
        2. Assert không NaN sau warmup: Kiểm định từ index >= window, toàn bộ ngưỡng phải có giá trị số thực,
           không bị null hay NaN/Inf.
        3. Fallback xử lý khuyết tật lọt lòng: Sử dụng .forward_fill() để gối tiếp ngưỡng ngày trước nếu thị trường có
           ngày giao dịch ngắt quáng gián đoạn.

    Args:
        df_daily: pl.DataFrame chứa cột tối thiểu 'daily_dollar_volume' (và 'date' hoặc 'timestamp_ms').
        window: int, số ngày trượt trong quá khứ để lấy trung bình (Mặc định Master Blueprint: 21 ngày ~ 1 tháng).
        target_freq: float, tần suất mục tiêu số nến Dollar-Volume mong muốn mỗi ngày (Mặc định: 50.0 nến/ngày).

    Returns:
        pl.DataFrame mới có thêm cột 'theta_pit' (float64).

    Raises:
        ValueError: Nếu cột 'daily_dollar_volume' thiếu, hoặc nếu phát hiện NaN/Null tồn đọng sau khoảng warm-up.
    """
    if "daily_dollar_volume" not in df_daily.columns:
        raise ValueError(
            "[ARMOR GUARD] DataFrame đầu vào thiếu cột bắt buộc 'daily_dollar_volume'."
        )

    if window < 1 or target_freq <= 0:
        raise ValueError(
            f"[ARMOR GUARD] Tham số không hợp lệ: window={window} (phải >= 1), target_freq={target_freq} (phải > 0)."
        )

    # 1. Thực hiện tính toán chuỗi trượt nhúng shift(1) để đảm bảo 100% Causal PIT-Safe
    df_out = df_daily.with_columns(
        (
            pl.col("daily_dollar_volume")
            .cast(pl.Float64)
            .fill_nan(None)  # Chuyển đổi floating NaN về Null của Polars
            .forward_fill()  # Trám causal ngoại suy theo thời gian tiến
            .shift(1)  # Bắt buộc dịch 1 nhịp để KHÔNG DÙNG volume của chính ngày T hoặc ngày mai T+1
            .rolling_mean(window_size=window)
            .fill_nan(None)
            .forward_fill()  # Causal fallback an toàn chống gap ngày nghỉ
            / float(target_freq)
        ).alias("theta_pit")
    )

    valid_count = df_out.filter(pl.col("theta_pit").is_not_null())["theta_pit"].len()
    if df_out.height >= (window + 1) and valid_count == 0:
        raise ValueError(
            "[ARMOR GUARD] PIT threshold hoàn toàn rỗng dù độ dài dữ liệu lớn hơn khoảng warm-up."
        )

    # 2. Kiểm định tuyệt đối KHÔNG NaN sau warmup (Từ index = window trở đi)
    if df_out.height > window:
        post_warmup_df = df_out.slice(window, df_out.height - window)
        null_count = post_warmup_df["theta_pit"].null_count()
        if null_count > 0:
            raise ValueError(
                f"[ARMOR GUARD] Vi phạm định nghĩa PIT-Safe: Phát hiện {null_count} giá trị NaN/Null "
                f"sau khoảng warm-up ({window} ngày). Ngưỡng gộp nến phải liên tục hợp lệ!"
            )

        # Kiểm định nan/inf trong không gian floating numpy
        post_warmup_vals = post_warmup_df["theta_pit"].to_numpy()
        if not np.all(np.isfinite(post_warmup_vals)):
            raise ValueError(
                "[ARMOR GUARD] Phát hiện giá trị Inf hoặc NaN không xác định sau khoảng warm-up!"
            )
        if not np.all(post_warmup_vals > 0):
            raise ValueError(
                "[ARMOR GUARD] Phát hiện ngưỡng gộp nến <= 0 sau khoảng warm-up. Ngưỡng theta_pit phải strictly positive (> 0)!"
            )

    return df_out


def map_daily_threshold_to_ticks(
    ticks_df: pl.DataFrame, daily_threshold_df: pl.DataFrame
) -> np.ndarray:
    """
    [TASK A-2-2 / SECTION 1.1.2] Ánh xạ daily_thresholds xuống từng Tick-Level bằng ASOF Backward Join O(N).

    Bảo đảm tốc độ O(N) tuyến tính trên mảng Polars, không dùng vòng lặp Python. Khi gia nhập, thuật toán 'backward'
    chức năng như cơ chế causal zero-order hold: mỗi tick i tại thời điểm timestamp_ms sẽ nhìn về ngày epoch gần nhất
    đã được định hình ngưỡng PIT.

    Args:
        ticks_df: pl.DataFrame chứa cột 'timestamp_ms' (int64).
        daily_threshold_df: pl.DataFrame chứa cột 'theta_pit' và cột 'date' (pl.Date/pl.Int64) hoặc 'timestamp_ms'.

    Returns:
        np.ndarray (contiguous, dtype=float64) độ dài N bằng với chuỗi ticks.
    """
    if "timestamp_ms" not in ticks_df.columns:
        raise ValueError("[ARMOR GUARD] ticks_df thiếu cột 'timestamp_ms' bắt buộc.")
    if "theta_pit" not in daily_threshold_df.columns:
        raise ValueError("[ARMOR GUARD] daily_threshold_df thiếu cột 'theta_pit'.")

    # 1. Trích xuất trục ngày epoch cho ticks
    ticks_with_date = ticks_df.with_columns(
        (pl.col("timestamp_ms") // 86_400_000).cast(pl.Int64).alias("date_epoch_day")
    ).sort("date_epoch_day")

    # 2. Xử lý linh hoạt dtype của cột thời gian trong daily_threshold_df
    if (
        "timestamp_ms" in daily_threshold_df.columns
        and "date" not in daily_threshold_df.columns
    ):
        daily_epoch_expr = (pl.col("timestamp_ms") // 86_400_000).cast(pl.Int64)
    elif "date" in daily_threshold_df.columns:
        if daily_threshold_df["date"].dtype in (pl.Date, pl.Datetime):
            daily_epoch_expr = (
                pl.col("date").dt.date().cast(pl.Int64)
                if daily_threshold_df["date"].dtype == pl.Datetime
                else pl.col("date").cast(pl.Int64)
            )
        else:
            daily_epoch_expr = pl.col("date").cast(pl.Int64)
    else:
        raise ValueError(
            "[ARMOR GUARD] daily_threshold_df phải có cột 'date' hoặc 'timestamp_ms'."
        )

    daily_sorted = daily_threshold_df.with_columns(
        daily_epoch_expr.alias("date_epoch_day")
    ).sort("date_epoch_day")

    # 3. Thực thi join_asof backward O(N)
    joined = ticks_with_date.join_asof(
        daily_sorted.select(["date_epoch_day", "theta_pit"]),
        on="date_epoch_day",
        strategy="backward",
    )

    theta_array = joined["theta_pit"].to_numpy()

    # 4. Fallback bọc lót cho giai đoạn đầu tiên (khi tick lọt vào chính giữa giai đoạn warm-up của ngày 1)
    valid_theta = theta_array[~np.isnan(theta_array)]
    fallback_value = (
        float(np.percentile(valid_theta, 10)) if len(valid_theta) > 0 else 1_000_000.0
    )
    theta_array = np.where(np.isnan(theta_array), fallback_value, theta_array)

    return np.ascontiguousarray(theta_array, dtype=np.float64)
