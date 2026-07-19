"""TRADE_RECORD_SCHEMA chuẩn hóa (v11.8): phân định rõ exit_idx_relative vs exit_idx_absolute."""
import hashlib # Thư viện dùng để tạo mã băm SHA-256 -> đóng dấu 'tem niêm phong số' cho mảng nến, ngăn chặn việc mang kq trade của bộ data này áp vào bộ data khác
import json # Thư viện chuyển từ điển dict tham số thành chuỗi văn bản -> để băm cấu hình tham số cùng với dữ liệu nến thành một mã duy nhất
import pandas as pd # Thư viện xử lý dữ liệu dạng bảng (như bảng tính excel)
import pandera.pandas as pa # Thư viện kiểm tra dữ liệu dạng bảng (như bảng tính excel, sử dụng namespace pandas chuẩn mới tránh cảnh báo tương lai)
from pandera.pandas import Column, Check, DataFrameSchema # Thư viện định nghĩa cấu trúc dữ liệu dạng bảng và các điều kiện ràng buộc
from typing import Dict, Any, TypedDict, Literal # Thư viện định nghĩa kiểu dữ liệu tĩnh (như dict, list, tuple, set, TypedDict, Literal)

# ============================================================================
# [TASK B-1-10] TYPED DICT CHO BẢN GHI GIAO DỊCH (Cho xử lý nội bộ dạng từ điển dict)
# ============================================================================
class TradeRecord(TypedDict):
    """
    Cấu trúc định nghĩa tĩnh cho một bản ghi giao dịch đơn lẻ (dạng từ điển dict).
    Giúp IDE tự động gợi ý code, kiểm tra lỗi gõ nhầm tên key trước khi gom thành DataFrame.
    """
    entry_idx: int # Chỉ số bar_idx tuyệt đối tại thời điểm vào lệnh
    p_i: float # Xác suất xu hướng tại thời điểm vào lệnh p_trend
    p_chop_i: float # Xác suất thị trường đi ngang tại thời điểm vào lệnh p_chop
    mode: Literal["follow", "fade", "none"] # Chế độ giao dịch: follow (theo xu hướng), fade (đánh đảo chiều), hoặc none
    side: int # Hướng lệnh giao dịch thực tế: +1 (Long/Mua) hoặc -1 (Short/Bán)
    sl_initial: float # Giá cắt lỗ ban đầu (Stop Loss) đã được đối xứng theo hướng lệnh
    exit_idx_relative: int # Khoảng cách số nến từ lúc vào lệnh đến lúc thoát lệnh (>= 0)
    exit_idx_absolute: int # Chỉ số bar_idx tuyệt đối tại thời điểm thoát lệnh (= entry_idx + 1 + exit_idx_relative)
    exit_reason: Literal["SL", "TRAIL", "REGIME_FLIP", "TIME_STOP"] # Lý do thoát lệnh theo chuẩn v11.8
    boundary_truncated: bool # Cờ báo hiệu giao dịch bị cắt ngắn do chạm biên fold CPCV hoặc hết dữ liệu test
    realized_return: float # Lợi nhuận thực tế (Realized Return) sau khi trừ phí và lãi qua đêm


# ============================================================================
# 0. BEHAVIORAL CONTRACT CHECK FUNCTIONS -> Các hàm kiểm tra logic dữ liệu
# ============================================================================
def check_ohlc_logic(df: pd.DataFrame) -> pd.Series:
    """
    Kiểm tra logic nến OHLC:
    - Giá cao nhất (high) bắt buộc phải lớn hơn hoặc bằng giá mở (open), đóng (close) và thấp nhất (low).
    - Giá thấp nhất (low) bắt buộc phải nhỏ hơn hoặc bằng giá mở (open) và đóng (close).
    """
    return (df["high"] >= df["open"]) & (df["high"] >= df["close"]) & (df["high"] >= df["low"]) & \
           (df["low"] <= df["open"]) & (df["low"] <= df["close"])

def check_insufficient_history_nulls(df: pd.DataFrame) -> pd.Series:
    """
    Kiểm tra điều kiện thiếu dữ liệu lịch sử:
    - Khi insufficient_history == True (giai đoạn khởi động W bar đầu tiên), các cột tín hiệu rolling
      bắt buộc phải là Null (rỗng), tuyệt đối không được điền số ảo làm hệ thống ngộ nhận tín hiệu.
    """
    mask = df["insufficient_history"] == True
    if not mask.any():
        return pd.Series(True, index=df.index)
    
    invalid_rows = df[mask][["trend_score", "p_trend", "p_chop", "atr_14", "hurst_value"]].notna().any(axis=1)
    return ~invalid_rows

def check_absolute_index_logic(df: pd.DataFrame) -> pd.Series:
    """
    Kiểm tra logic tuyệt đối của chỉ số exit (v11.8):
    - Chỉ số tuyệt đối exit_idx_absolute bắt buộc phải bằng entry_idx + 1 + exit_idx_relative.
    """
    return df["exit_idx_absolute"] == (df["entry_idx"] + 1 + df["exit_idx_relative"])


# ============================================================================
# 1. SIGNAL_BAR_SCHEMA (v1.0.0) -> Cấu trúc dữ liệu dạng bảng của các tín hiệu nến
# ============================================================================
SignalBarSchema = DataFrameSchema(
    columns={
        "bar_idx": Column(int, Check.ge(0), unique=True), # Chỉ số thứ tự tuyệt đối của nến (>= 0, duy nhất)
        "symbol": Column(str), # Ký hiệu mã giao dịch (ví dụ: BTCUSDT)
        "timestamp_ms": Column(int, Check.ge(0)), # Thời gian đóng nến tính bằng mili giây epoch (knowledge_time)
        "open": Column(float, Check.gt(0)), # Giá mở cửa (> 0)
        "high": Column(float, Check.gt(0)), # Giá cao nhất (> 0)
        "low": Column(float, Check.gt(0)), # Giá thấp nhất (> 0)
        "close": Column(float, Check.gt(0)), # Giá đóng cửa (> 0)
        "volume": Column(float, Check.ge(0)), # Khối lượng giao dịch của nến (>= 0)
        "ofi": Column(float, Check.in_range(-1.0, 1.0)), # Chỉ số mất cân bằng dòng lệnh chuẩn hóa Order Flow Imbalance [-1.0, 1.0]
        "tick_count": Column(int, Check.ge(0)), # Số lượng tick cấu thành nên cây nến Dollar-Volume (>= 0)
        "is_toxic_flag": Column(bool), # Cờ báo hiệu nến có thanh khoản độc hại (tick_count < 0.5 * median)
        "is_tail_event": Column(bool), # Cờ báo hiệu sự kiện đuôi thanh khoản đột biến (Black Swan / Tail Event)
        "insufficient_history": Column(bool), # Cờ báo hiệu nến thuộc giai đoạn khởi động chưa đủ dữ liệu lịch sử
        
        "trend_score": Column(float, nullable=True), # Điểm số xu hướng từ tín hiệu sơ cấp (cho phép Null khi khởi động)
        "p_trend": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Xác suất xu hướng HMM [0, 1] (cho phép Null)
        "p_chop": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Xác suất đi ngang HMM [0, 1] (cho phép Null)
        "atr_14": Column(float, Check.ge(0), nullable=True), # Chỉ số biến động trung bình ATR 14 chu kỳ (>= 0, cho phép Null)
        "hurst_value": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Chỉ số số mũ Hurst GHE [0, 1] (cho phép Null)
        "d_star_used": Column(float, Check.in_range(0.0, 1.0), nullable=True), # Bậc vi phân từng phần FFD tối ưu [0, 1] (cho phép Null)
    },
    checks=[
        Check(check_ohlc_logic, name="check_ohlc_logic"), # Gắn kiểm tra hợp đồng logic nến OHLC
        Check(check_insufficient_history_nulls, name="check_insufficient_history_nulls") # Gắn kiểm tra logic Null khi insufficient_history=True
    ],
    strict=True,
    name="SignalBarSchema_v1.0.0"
)


# ============================================================================
# 2. TRADE_RECORD_SCHEMA (v1.0.0) -> Cấu trúc dữ liệu dạng bảng của các giao dịch
# ============================================================================
TradeRecordSchema = DataFrameSchema(
    columns={
        "schema_version": Column(str, Check.eq("1.0.0")), # Phiên bản của cấu trúc dữ liệu giao dịch (khóa cứng "1.0.0")
        "dataset_manifest_hash": Column(str), # Mã băm SHA-256 của chuỗi nến sinh ra giao dịch này
        "fold_id": Column(str, nullable=True), # ID của fold kiểm định CPCV (ví dụ: "fold_0", cho phép Null khi chạy live)
        "symbol": Column(str), # Ký hiệu mã giao dịch
        "entry_idx": Column(int, Check.ge(0)), # Chỉ số thứ tự tuyệt đối của nến tại thời điểm vào lệnh (>= 0)
        "entry_price": Column(float, Check.gt(0)), # Giá thực tế vào lệnh (> 0)
        "p_i": Column(float, Check.in_range(0.0, 1.0)), # Xác suất xu hướng tại thời điểm vào lệnh [0, 1]
        "p_chop_i": Column(float, Check.in_range(0.0, 1.0)), # Xác suất đi ngang tại thời điểm vào lệnh [0, 1]
        "mode": Column(str, Check.isin(["follow", "fade"])), # Chế độ vào lệnh: follow (theo xu hướng) hoặc fade (đảo chiều)
        "side": Column(int, Check.isin([-1, 1])), # Chiều thực tế của lệnh: +1 (Long) hoặc -1 (Short)
        "sl_initial": Column(float, Check.gt(0)), # Giá cắt lỗ ban đầu (> 0)
        "size_notional": Column(float, Check.gt(0)), # Quy mô danh nghĩa của lệnh giao dịch (> 0)
        "exit_idx_relative": Column(int, Check.ge(0)), # Khoảng cách bar tương đối từ lúc vào lệnh đến lúc thoát lệnh (>= 0)
        "exit_idx_absolute": Column(int, Check.ge(0)), # Chỉ số bar tuyệt đối khi thoát lệnh (= entry_idx + 1 + exit_idx_relative)
        "exit_reason": Column(str, Check.isin(["SL", "TRAIL", "REGIME_FLIP", "TIME_STOP"])), # Lý do thoát lệnh chuẩn hóa
        "fill_price_exit": Column(float, Check.gt(0)), # Giá khớp lệnh thoát lệnh thực tế tra cứu tại exit_idx_absolute (> 0)
        "boundary_truncated": Column(bool), # Cờ báo hiệu giao dịch bị cắt ngắn do hết dữ liệu hoặc chạm biên fold
        "fee_entry": Column(float, Check.ge(0)), # Phí giao dịch vào lệnh (>= 0)
        "fee_exit": Column(float, Check.ge(0)), # Phí giao dịch thoát lệnh (>= 0)
        "funding_accrued": Column(float), # Tổng chi phí/lợi tức lãi qua đêm (funding rate) tích lũy
        "gross_pnl": Column(float), # Lợi nhuận gộp chưa trừ phí
        "realized_return": Column(float), # Lợi nhuận ròng thực tế (Realized Return) sau phí và funding
    },
    checks=[
        Check(check_absolute_index_logic, name="check_absolute_index_logic") # Gắn kiểm tra hợp đồng logic exit_idx_absolute
    ],
    strict=True,
    name="TradeRecordSchema_v1.0.0"
)


# ============================================================================
# 3. LINEAGE & VERSIONING UTILS -> Các hàm tiện ích về dòng dõi và phiên bản
# ============================================================================
def compute_dataset_manifest_hash(bar_df: pd.DataFrame, generation_params: Dict[str, Any]) -> str:
    """
    Tính mã Hash bảo vệ cho mảng nến.
    Bao gồm shape của mảng nến, khoảng thời gian và các tham số sinh ra nó.
    Giúp đóng dấu 'Tem niêm phong' SHA-256 duy nhất cho từng tập dữ liệu.
    """
    # Tạo một payload chứa các thông tin cốt lõi để tính mã Hash
    payload = {
        "params": generation_params,
        "n_rows": len(bar_df),
        "start_time": int(bar_df["timestamp_ms"].min()),
        "end_time": int(bar_df["timestamp_ms"].max())
    }
    # Chuyển payload sang chuỗi văn bản JSON đã được sắp xếp key chuẩn xác
    sorted_str = json.dumps(payload, sort_keys=True, default=str)
    # Tính và trả về mã băm SHA-256 (64 ký tự hex)
    return hashlib.sha256(sorted_str.encode('utf-8')).hexdigest()

def assert_trade_records_match_bar_version(trade_df: pd.DataFrame, expected_manifest_hash: str):
    """
    Chặn đứng luồng chạy (Gatekeeper check) nếu bảng Trade Records không được sinh ra
    từ đúng phiên bản Bar Array hiện tại, bảo vệ hệ thống khỏi việc ngộ nhận kết quả backtest ảo.
    """
    # Lấy danh sách các mã hash duy nhất từ cột dataset_manifest_hash của bảng giao dịch
    unique_hashes = trade_df["dataset_manifest_hash"].unique()
    # Kiểm tra xem có bị lẫn lộn nhiều phiên bản dataset khác nhau trong cùng một bảng không
    assert len(unique_hashes) == 1, "FATAL: Lẫn lộn nhiều phiên bản dataset trong cùng một Trade DataFrame."
    # Lấy mã hash thực tế của bảng giao dịch
    actual_hash = unique_hashes[0]
    # Kiểm tra mã hash thực tế có khớp tuyệt đối với mã hash kỳ vọng của mảng nến hiện tại không
    assert actual_hash == expected_manifest_hash, \
        f"FATAL: DATA LINEAGE MISMATCH!\nExpected Hash: {expected_manifest_hash}\nActual Hash: {actual_hash}\n" \
        f"Trade records đã hết hạn so với Bar Array hiện tại. Yêu cầu chạy lại toàn bộ pipeline."