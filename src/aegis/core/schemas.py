"""TRADE_RECORD_SCHEMA chuẩn hóa (v11.8): phân định rõ exit_idx_relative vs exit_idx_absolute."""
import hashlib # thư viện dùng để tạo mã băm SHA-256->đóng dấu 'tem niêm phong số' cho mảng nến, ngăn chặn việc mang kq trade của bộ data này áp vào bộ data khác
import json #thư viện chuyển từ điển dict tham số thành chuỗi văn bản -> để băm cấu hình tham số cùng với dữ liệu nến thành một mã duy nhất
import pandas as pd #thư viện xử lý dữ liệu dạng bảng (như bảng tính excel)
import pandera.pandas as pa #thư viện kiểm tra dữ liệu dạng bảng (như bảng tính excel)
from pandera.pandas import Column, Check, DataFrameSchema #thư viện định nghĩa cấu trúc dữ liệu dạng bảng 
from typing import Dict, Any #thư viện định nghĩa kiểu dữ liệu (như dict, list, tuple, set, frozenset, bytes, bytearray)

# ============================================================================
# 0. BEHAVIORAL CONTRACT CHECK FUNCTIONS -> Các hàm kiểm tra logic dữ liệu
# ============================================================================
#Kiểm tra logic nến
def check_ohlc_logic(df: pd.DataFrame) -> pd.Series:
    return (df["high"] >= df["open"]) & (df["high"] >= df["close"]) & (df["high"] >= df["low"]) & \
           (df["low"] <= df["open"]) & (df["low"] <= df["close"])

#Kiểm tra điều kiện thiếu dữ liệu
def check_insufficient_history_nulls(df: pd.DataFrame) -> pd.Series:
    mask = df["insufficient_history"] == True
    if not mask.any():
        return pd.Series(True, index=df.index)
    invalid_rows = df[mask][["trend_score", "p_trend", "p_chop", "atr_14", "hurst_value"]].notna().any(axis=1)
    return ~invalid_rows

#Kiểm tra logic tuyệt đối của chỉ số exit
def check_absolute_index_logic(df: pd.DataFrame) -> pd.Series:
    return df["exit_idx_absolute"] == (df["entry_idx"] + 1 + df["exit_idx_relative"])

# ============================================================================
# 1. SIGNAL_BAR_SCHEMA (v1.0.0) -> Cấu trúc dữ liệu dạng bảng của các tín hiệu nến
# ============================================================================
SignalBarSchema = DataFrameSchema(
    columns={
        "bar_idx": Column(int, Check.ge(0), unique=True), # Chỉ số thứ tự của nến
        "symbol": Column(str), #Ký hiệu mã giao dịch
        "timestamp_ms": Column(int, Check.ge(0)), # Thời gian tính bằng mili giây
        "open": Column(float, Check.gt(0)), # Giá mở cửa
        "high": Column(float, Check.gt(0)), # Giá cao nhất
        "low": Column(float, Check.gt(0)), # Giá thấp nhất
        "close": Column(float, Check.gt(0)), # Giá đóng cửa
        "volume": Column(float, Check.ge(0)), # Khối lượng giao dịch
        "ofi": Column(float, Check.in_range(-1.0, 1.0)), # Chỉ số dòng tiền vào lệnh
        "tick_count": Column(int, Check.ge(0)), # Số lượng tick giao dịch
        "is_toxic_flag": Column(bool), # Cờ báo hiệu dữ liệu độc hại
        "is_tail_event": Column(bool), # Cờ báo hiệu sự kiện đuôi
        "insufficient_history": Column(bool), # Cờ báo hiệu dữ liệu không đủ
        
        "trend_score": Column(float, nullable=True), # Điểm số xu hướng
        "p_trend": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Xác suất xu hướng
        "p_chop": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Xác suất đi ngang
        "atr_14": Column(float, Check.ge(0), nullable=True), # Chỉ số biến động trung bình
        "hurst_value": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Chỉ số xu hướng dài hạn
        "d_star_used": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Chỉ số dòng tiền vào lệnh
    },
    checks=[
        Check(check_ohlc_logic, name="check_ohlc_logic"), # Kiểm tra logic nến
        Check(check_insufficient_history_nulls, name="check_insufficient_history_nulls")
    ],
    strict=True,
    name="SignalBarSchema_v1.0.0"
)

# ============================================================================
# 2. TRADE_RECORD_SCHEMA (v1.0.0) -> Cấu trúc dữ liệu dạng bảng của các giao dịch
# ============================================================================
TradeRecordSchema = DataFrameSchema(
    columns={
        "schema_version": Column(str, Check.eq("1.0.0")), # Phiên bản của cấu trúc dữ liệu
        "dataset_manifest_hash": Column(str), # Mã băm của dữ liệu nến
        "fold_id": Column(str, nullable=True), # ID của fold
        "symbol": Column(str), #Ký hiệu mã giao dịch
        "entry_idx": Column(int, Check.ge(0)), # Chỉ số thứ tự của nến tại thời điểm vào lệnh
        "entry_price": Column(float, Check.gt(0)), # Giá vào lệnh
        "p_i": Column(float, Check.in_range(0.0, 1.0)), # Xác suất vào lệnh
        "p_chop_i": Column(float, Check.in_range(0.0, 1.0)), # Xác suất cưa lệnh
        "mode": Column(str, Check.isin(["follow", "fade"])), # Chế độ vào lệnh
        "side": Column(int, Check.isin([-1, 1])), # Chiều của lệnh
        "sl_initial": Column(float, Check.gt(0)), # Giá cắt lỗ ban đầu
        "size_notional": Column(float, Check.gt(0)), # Khối lượng giao dịch
        "exit_idx_relative": Column(int, Check.ge(0)), # Chỉ số tương đối của nến tại thời điểm thoát lệnh
        "exit_idx_absolute": Column(int, Check.ge(0)), # Chỉ số tuyệt đối của nến tại thời điểm thoát lệnh
        "exit_reason": Column(str, Check.isin(["SL", "TRAIL", "REGIME_FLIP", "TIME_STOP"])), # Lý do thoát lệnh
        "fill_price_exit": Column(float, Check.gt(0)), # Giá thoát lệnh
        "boundary_truncated": Column(bool), # Cờ báo hiệu giao dịch bị cắt ngắn do hết dữ liệu
        "fee_entry": Column(float, Check.ge(0)), # Phí vào lệnh
        "fee_exit": Column(float, Check.ge(0)), # Phí thoát lệnh
        "funding_accrued": Column(float), # Tiền lãi/lỗ tích lũy do chênh lệch lãi suất qua đêm
        "gross_pnl": Column(float), # Lợi nhuận gộp
        "realized_return": Column(float), # Lợi nhuận thực tế
    },
    checks=[
        Check(check_absolute_index_logic, name="check_absolute_index_logic")
    ],
    strict=True,
    name="TradeRecordSchema_v1.0.0"
)


# ============================================================================
# 3. LINEAGE & VERSIONING UTILS -> Các hàm tiện ích về dòng dõi và phiên bản
# ============================================================================
# Hàm tính mã Hash bảo vệ cho mảng nến
def compute_dataset_manifest_hash(bar_df: pd.DataFrame, generation_params: Dict[str, Any]) -> str:
    """
    Tính mã Hash bảo vệ cho mảng nến.
    Bao gồm shape của mảng nến và các tham số sinh ra nó.
    """
    # Tạo một payload chứa các thông tin cần thiết để tính mã Hash
    payload = {
        "params": generation_params,
        "n_rows": len(bar_df),
        "start_time": int(bar_df["timestamp_ms"].min()),
        "end_time": int(bar_df["timestamp_ms"].max())
    }
    # Chuyển payload sang chuỗi văn bản đã được sắp xếp để tính mã Hash
    sorted_str = json.dumps(payload, sort_keys=True, default=str)
    # Tính mã Hash SHA-256
    return hashlib.sha256(sorted_str.encode('utf-8')).hexdigest()

# Hàm kiểm tra Trade Records có được sinh ra từ đúng phiên bản Bar Array hiện tại hay không
def assert_trade_records_match_bar_version(trade_df: pd.DataFrame, expected_manifest_hash: str):
    """
    Chặn đứng luồng chạy nếu Trade Records không được sinh ra từ đúng phiên bản Bar Array hiện tại.
    """
    # Lấy danh sách các mã hash duy nhất từ DataFrame Trade Records
    unique_hashes = trade_df["dataset_manifest_hash"].unique()
    # Kiểm tra xem có nhiều hơn 1 mã hash duy nhất không
    assert len(unique_hashes) == 1, "FATAL: Lẫn lộn nhiều phiên bản dataset trong cùng một Trade DataFrame."
    # Lấy mã hash thực tế
    actual_hash = unique_hashes[0]
    # Kiểm tra mã hash thực tế có khớp với mã hash kỳ vọng không
    assert actual_hash == expected_manifest_hash, \
        f"FATAL: DATA LINEAGE MISMATCH!\nExpected Hash: {expected_manifest_hash}\nActual Hash: {actual_hash}\n" \
        f"Trade records đã hết hạn so với Bar Array hiện tại. Yêu cầu chạy lại toàn bộ pipeline."