# Aegis codemap (SINH TỰ ĐỘNG — đừng sửa tay)

Sinh lại: `python scripts/gen_codemap.py`

Cách dùng: tìm symbol -> đọc `file:line` -> mở thẳng bằng `sed -n 'START,+40p' file`.

---

## 1. CHƯA TRIỂN KHAI (18 file — chỉ có docstring/stub)

Không có logic. Đừng phí token đọc; đây là danh sách việc phải làm.

| file | dòng | ý định |
|---|---|---|
| `src/aegis/core/logging_setup.py` | 1 | Cấu hình JSON-lines logging cho hệ thống. |
| `src/aegis/core/pit.py` | 1 | Point-in-Time (PIT) Enforcer — Đảm bảo knowledge_time <= current_time tuyệt đố |
| `src/aegis/dashboard/app.py` | 1 | Web dashboard theo dõi trạng thái hệ thống theo thời gian thực. |
| `src/aegis/data/__init__.py` | 54 | Module A: Tien xu ly du lieu vi cau truc, Loc nhieu, The cho Bad Tick, va Nguo |
| `src/aegis/data/bars/__init__.py` | 19 | Module A.1 & A.2: Quản lý và Xây dựng Nến định lượng Dollar Volume Bars. |
| `src/aegis/data/cleaning/__init__.py` | 21 | Module Cleaning & Replacer — Loc nhieu 4 Dieu Kien, The cho Bad Tick bang Pred |
| `src/aegis/execution/latency_model.py` | 1 | sample_latency_regime_aware — mô phỏng độ trễ Lognormal co giãn theo toxicity. |
| `src/aegis/execution/market_impact.py` | 1 | simulate_market_fill — định luật tác động thị trường căn bậc hai (Square-Root  |
| `src/aegis/features/diagnostics.py` | 1 | Chẩn đoán đa cộng tuyến Spearman Rank Correlation |rho| > 0.80 & VIF > 5.0. |
| `src/aegis/features/regime/hurst_exponent.py` | 1 | Generalized Hurst Exponent (GHE window W=168, lags [2,4,8,16]). |
| `src/aegis/governance/dataset_manifest.py` | 1 | Module K.4 — Xuất SHA-256 Dataset Manifest cho toàn bộ dữ liệu và cấu hình. |
| `src/aegis/governance/l2_depth.py` | 1 | Module K.5 — L2 Order Book Depth Source & fallback tắt limit-order khi mất fee |
| `src/aegis/meta_labeling/__init__.py` | 4 | Meta-labeling module exports. |
| `src/aegis/meta_labeling/sizing/bet_overlap.py` | 1 | Điều chỉnh quy trình vào lệnh khi các cược chồng lấp nhau trên trục thời gian. |
| `src/aegis/sentiment_client/client.py` | 1 | Giao tiếp với aegis-sentiment-service, ép PIT trước khi đưa vào features. |
| `src/aegis/shadow/readiness_gate.py` | 1 | check_shadow_mode_readiness (Gate Kép: >= 30 sự kiện VÀ >= 2 tuần). |
| `src/aegis/validation/flat_plateau.py` | 1 | Kiểm chứng tính ổn định Flat Plateau >= 80% xung quanh 81 cấu hình lân cận. |
| `src/aegis/validation/structural_break_cusum.py` | 1 | CUSUM Brier Score với Reset & Refresh Thresholds (Module J.3). |

---

## 2. ĐÃ TRIỂN KHAI (75 file)

### `src/aegis/core/config_loader.py` (177 dòng)
_Tiện ích tải YAML configuration an toàn và quản lý cấu hình tập trung._

- `def get_config_dir` :10 — Tìm thư mục config/ chuẩn của dự án.
- `def load_canonical_parameters` :25 — Tải Sổ Cân Bằng Hằng Số (aegis_canonical_parameters.yaml).
- `def deep_merge` :35 — Hợp nhất đè 2 dictionary lồng nhau (deep merge).
- `def load_unified_config` :46 — Tải cấu hình hợp nhất 3 tầng:
- `def flatten_runtime_config` :125 — Chiếu cây cấu hình lồng nhau xuống bộ khóa phẳng mà các pipeline thực sự đọc.
- `def load_runtime_config` :163 — Tải cấu hình hợp nhất VÀ làm phẳng sang bộ khóa runtime (dùng cho pipeline).

### `src/aegis/core/credentials.py` (80 dòng)
_Nạp thông tin xác thực sàn từ biến môi trường / file .env._

- `def load_dotenv` :14 — Nạp file .env vào os.environ. Trả về dict các cặp đã đọc được.
- `class ExchangeCredentials` :37 — Khoá API của sàn. `__repr__` được che để khoá không lọt vào log/traceback.
- `def load_binance_credentials` :49 — Đọc khoá Binance Futures từ môi trường.

### `src/aegis/core/execution_log.py` (118 dòng)
_Nhật ký thực thi — đo chi phí THẬT thay vì tin vào giả định của backtest._

- `class ExecutionRecord` :26 — Một lượt tái cân bằng.
- `  . filled_notional` :42
- `  . maker_ratio` :46
- `  . realized_cost_usd` :51
- `  . realized_cost_bps` :55
- `  . fill_ratio` :60
- `class ExecutionLog` :66 — Ghi/đọc nhật ký thực thi dạng JSONL (append-only).
- `  . __init__` :69
- `  . append` :73
- `  . read_all` :77
- `  . summary` :93

### `src/aegis/core/experiment_tracker.py` (181 dòng)
_ExperimentTracker Singleton — Ghi nhận DSR trials và SHA-256 param hashes._

- `class ExperimentTracker` :31 — Singleton class để quản lý việc ghi nhận các lần chạy thử nghiệm (trials).
- `  . __init__` :48
- `  . hash_params` :69
- `  . reset_instance` :128
- `  . log_trial` :136
- `  . get_total_trials` :163

### `src/aegis/core/schemas.py` (272 dòng)
_TRADE_RECORD_SCHEMA chuẩn hóa (v11.8): phân định rõ exit_idx_relative vs exit__

- `class TradeRecord` :13 — Cấu trúc định nghĩa tĩnh cho một bản ghi giao dịch đơn lẻ (dạng từ điển dict).
- `class ImmutableTradeRecord` :48 — [KHẮC PHỤC LỖ HỔNG 9 - MUTABILITY TRAP]:
- `  . from_dict` :81
- `  . to_dict` :110
- `  . update` :114
- `def check_ohlc_logic` :122 — Kiểm tra logic nến OHLC:
- `def check_insufficient_history_nulls` :131 — Kiểm tra điều kiện thiếu dữ liệu lịch sử:
- `def check_absolute_index_logic` :145 — Kiểm tra logic tuyệt đối của chỉ số exit (v11.8):
- `def check_timestamp_logic` :152 — Kiểm tra logic thời gian (Vá BỌ SỐ 3):
- `def compute_dataset_manifest_hash` :237 — Tính mã Hash bảo vệ cho mảng nến.
- `def assert_trade_records_match_bar_version` :259 — Chặn đứng luồng chạy (Gatekeeper check) nếu bảng Trade Records không được sinh

### `src/aegis/core/state_store.py` (106 dòng)
_Trạng thái bền vững qua các lần khởi động lại._

- `class LiveState` :30 — Toàn bộ trạng thái phải sống sót qua restart.
- `  . drawdown` :49
- `class StateStore` :55 — Đọc/ghi `LiveState` ra đĩa một cách nguyên tử.
- `  . __init__` :58
- `  . load` :62
- `  . save` :82
- `  . update` :98

### `src/aegis/core/trial_classes.py` (19 dòng)
_Phân loại Trial Classes: model_fitting, strategy_selection (tính vào N_DSR), p_

- `class TrialClass` :6 — Phân loại các lượt chạy thử nghiệm (trials) để tính toán DSR (Deflated Sharpe 

### `src/aegis/data/bars/builder.py` (159 dòng)
_Module orchestrator (Entrypoint) cho Track A._

- `def build_clean_dollar_bars` :22 — [TASK A-2-5] End-to-End Orchestrator.

### `src/aegis/data/bars/dollar_volume_bars.py` (202 dòng)
_Module tạo nến Dollar Volume Bars (Task A-2-3 & A-2-4)._

- `def compute_median_ticks_to_fill_per_tick` :41 — [TASK A-2-3] Toán tử tính trung vị số lượng tick trên mỗi nến (PIT-Safe).
- `def generate_dollar_volume_bars_v11` :94 — [TASK A-2-4 & A-2-5] Thuật toán cốt lõi sinh Nến Dollar Volume theo chuẩn AFML

### `src/aegis/data/bars/pit_threshold.py` (160 dòng)
_Module A-2: Chuẩn bị ngưỡng tạo nến tín hiệu định lượng Dollar Volume Bars (Ta_

- `def compute_pit_safe_daily_threshold` :11 — [TASK A-2-1] Tính ngưỡng gộp nến Dollar-Volume an toàn theo thời điểm (Point-i
- `def map_daily_threshold_to_ticks` :93 — [TASK A-2-2 / SECTION 1.1.2] Ánh xạ daily_thresholds xuống từng Tick-Level bằn

### `src/aegis/data/bars/tick_rule_ofi.py` (118 dòng)
_Tick Rule Classification & chuẩn hóa Order Flow Imbalance (OFI_t)._

- `def classify_tick_rule` :43 — Phân loại chiều luồng lệnh theo thuật toán Tick Rule chuẩn AFML:
- `def compute_tick_rule_ofi` :86 — Tính Order Flow Imbalance (OFI) cho một bar nến từ luồng tick con:
- `def compute_rolling_ofi` :100 — Tính OFI cửa sổ trượt (rolling window) trên chuỗi nến hoặc chuỗi tick.

### `src/aegis/data/cleaning/gap_handling.py` (87 dòng)
_[TASK A-3-1] Gap Handling — Giao thức chiếu Kalman qua khoảng đứt gãy._

- `def kalman_predict_only_n_steps` :38 — [TASK A-3-1] Chiếu trạng thái hệ thống (State) và ma trận Hiệp phương sai (Cov

### `src/aegis/data/cleaning/outlier_filter.py` (229 dòng)
_[TASK A-1-5] Module Outlier Filter & Clean Tick Stream Pipeline._

- `class CleanedTickStreamResult` :23 — [TASK A-1-5] Output cau truc cua pipeline clean_tick_stream.
- `def filter_outliers_4_conditions` :41 — [TASK A-1-5] Kiem Dinh 4 Dieu Kien Dong Thoi (Outlier Filter):
- `def clean_tick_stream` :129 — [TASK A-1-5] Ghep Ngan Kien Truc clean_tick_stream Pipeline (Master Blueprint 

### `src/aegis/data/cleaning/tick_kalman_replacer.py` (265 dòng)
_[TASK A-1-4] TickLevelKalmanReplacer — Predict-Only Protocol khi phát hiện Bad_

- `def ensure_pd_matrix_2x2_numba` :14 — [ARMOR GUARD] Đảm bảo ma trận hiệp phương sai 2x2 là Positive Definite (PD) tr
- `def kalman_replacer_filter_series_numba` :49 — [TASK A-1-4 - OPTIMIZED NUMBA ENGINE] Thực thi Kalman Replacer trên toàn bộ mả
- `class TickLevelKalmanReplacer` :151 — [TASK A-1-4] TickLevelKalmanReplacer — Master Blueprint Compliant Implementati
- `  . __init__` :158
- `  . reset` :180
- `  . step` :185
- `  . filter_series` :232

### `src/aegis/data/ingestion/binance_history.py` (210 dòng)
_[FIX F2] Tải lịch sử nến + funding THẬT từ Binance và lưu parquet._

- `def interval_to_ms` :38
- `def klines_to_frame` :44 — Chuyển nến thô Binance thành DataFrame OHLCV chuẩn cho signal_pipeline.
- `def download_klines` :81 — Tải `days` ngày nến gần nhất, tự phân trang qua trần 1500 nến/request.
- `def download_funding_rates` :130 — Lịch sử funding rate THẬT -> dict {timestamp_ms: rate}.
- `def save_klines` :172 — Ghi/gộp nến vào parquet. Gộp theo `timestamp_ms` nên tải chồng lặp vẫn an toàn
- `def load_klines` :199

### `src/aegis/data/ingestion/binance_rest.py` (329 dòng)
_[FIX F1/F2] REST client Binance USDⓈ-M Futures._

- `class BinanceAPIError` :38 — Sàn trả về lỗi nghiệp vụ (có mã lỗi riêng của Binance).
- `  . __init__` :41
- `class BinanceFuturesREST` :47 — Client REST cho Binance USDⓈ-M Futures.
- `  . __init__` :50
- `  . public_mainnet` :76
- `  . sync_time` :167
- `  . ping` :177
- `  . exchange_info` :181
- `  . symbol_filters` :185
- `  . klines` :221
- `  . funding_rate_history` :242
- `  . book_ticker` :266
- `  . best_bid_ask` :275
- `  . premium_index` :279
- `  . account` :286
- `  . balance_usdt` :289
- `  . position_risk` :300
- `  . open_orders` :304
- `  . set_leverage` :307
- `  . set_margin_type` :313

### `src/aegis/data/ingestion/cross_venue_feed.py` (8 dòng)
_Tiếp nhận luồng dữ liệu sàn đối chứng (Cross-Venue Parity check)._

- `def create_cross_venue_feed` :4

### `src/aegis/data/ingestion/l2_orderbook_feed.py` (8 dòng)
_Tiếp nhận feed L2 Order Book Snapshot (Module K.5)._

- `def create_l2_orderbook_feed` :4

### `src/aegis/data/ingestion/tick_stream.py` (8 dòng)
_Module tiếp nhận luồng tick dữ liệu giá realtime._

- `def create_tick_stream` :4

### `src/aegis/data/outlier_detection.py` (178 dòng)
_[TASK A-1-1] Nền Tảng Toán Học Lọc Nhiễu Tick (MAD 5σ & Cross-Venue Parity)._

- `def compute_rolling_mad` :11 — [TASK A-1-1] Tính Robust Sigma dựa trên Median Absolute Deviation (MAD) cửa sổ
- `def detect_bad_tick_core` :55 — [TASK A-1-2] Phân loại Bad Tick và Tail Event dựa trên 3 điều kiện lõi.
- `def detect_bad_tick_cross_venue` :122 — [TASK A-1-3] Điều kiện 4: Kiểm tra Cross-Venue Parity.

### `src/aegis/data/panel.py` (84 dòng)
_Dựng panel đa tài sản (time x symbol) cho nghiên cứu cross-sectional._

- `def load_panel` :19 — Nạp nhiều cặp và căn theo lưới thời gian chung.
- `def load_funding_panel` :54 — Panel funding rate căn theo lưới thời gian của giá.
- `def forward_returns` :77 — Lợi suất TƯƠNG LAI qua `horizon` nến — biến mục tiêu.

### `src/aegis/data/panel_v2.py` (181 dòng)
_Nạp panel đa tài sản từ MỘT nguồn duy nhất (nến 1h) và tổng hợp lên khung bất _

- `def interval_hours` :41
- `def resample_bars` :47 — Tổng hợp nến từ khung `source` lên khung `target`.
- `def load_panel_v2` :94 — Nạp panel {trường: DataFrame(time x symbol)} ở khung `interval`.
- `def load_funding_panel_v2` :150 — Panel funding rate căn theo lưới giá.
- `def align_panel` :175 — Ép mọi trường về cùng bộ cột và cùng thứ tự — tránh lệch cột âm thầm.

### `src/aegis/data/universe.py` (125 dòng)
_Module K.3 — Dựng universe giao dịch được, chống thiên lệch sống sót._

- `class UniverseFilter` :36 — Tiêu chí lọc universe.
- `  . describe` :42
- `def tradeable_symbols` :48 — Tập perp USDT đang giao dịch được trên sàn mà client đang trỏ tới.
- `def liquid_symbols` :59 — Tập cặp có khối lượng 24h vượt ngưỡng.
- `def history_lengths` :70 — Số nến đã tải được cho từng cặp.
- `def build_universe` :85 — Lọc danh sách ứng viên xuống universe thực sự giao dịch được.

### `src/aegis/execution/limit_queue_sim.py` (297 dòng)
_limit_queue_sim.py — L2 Orderbook Queue Estimation & 3-Tier Progressive Fallba_

- `def estimate_queue_ahead` :25 — Ước tính khối lượng xếp hàng phía trước (Queue-Ahead Position) trên sổ lệnh L2
- `def resolve_execution_mode_with_l2_fallback` :130 — Hệ thống Suy thoái Đa tầng (3-Tier Progressive Fallback Protocol).
- `def simulate_limit_fill_with_queue` :236 — Mô phỏng khả năng khớp lệnh giới hạn qua nhiều bar kề tiếp theo.

### `src/aegis/execution/pnl.py` (128 dòng)

- `def compute_realized_pnl` :12 — [MODULE G] Động cơ duy nhất tính toán PnL trên toàn hệ thống.

### `src/aegis/execution/portfolio_rebalancer.py` (229 dòng)
_Cầu nối trọng số mục tiêu -> lệnh thật trên sàn (danh mục cross-sectional)._

- `class SymbolFilters` :26 — Ràng buộc giao dịch của một cặp (lấy từ /fapi/v1/exchangeInfo).
- `  . round_qty` :48
- `  . round_price` :56
- `  . format_qty` :63
- `  . format_price` :73
- `  . is_tradeable` :77
- `class TargetPosition` :84
- `class RebalanceOrder` :93
- `class RebalancePlan` :103
- `  . total_turnover` :110
- `def build_rebalance_plan` :114 — Dựng danh sách lệnh đưa danh mục hiện tại về trọng số mục tiêu.
- `def max_positions_for_capital` :215 — Số vị thế TỐI ĐA mà vốn cho phép, để mỗi lệnh vẫn vượt min_notional.

### `src/aegis/execution/position_sizer.py` (172 dòng)
_[PHÁT HIỆN O] Module G — Cầu Nối Kelly → Lệnh Thật._

- `def round_notional_down` :19 — Làm tròn xuống (round down / floor) quy mô danh nghĩa theo bước nhảy `lot_step
- `class AccountStateTracker` :37 — Phân định minh bạch 3 tầng số dư ví tài khoản Isolated Margin:
- `  . __init__` :48
- `  . margin_balance` :61
- `  . available_margin` :66
- `  . get_effective_equity_for_sizing` :73
- `def compute_position_size` :87 — [PHÁT HIỆN O + KHẮC PHỤC LỖ HỔNG 7 & BẪY 1] Biến f* thành size_notional cho lệ

### `src/aegis/features/feature_engine.py` (196 dòng)
_[FIX F3] Hai đường đi tính đặc trưng — batch (research) và online (live)._

- `class MissingFeatureError` :30 — Ném ra khi model yêu cầu một đặc trưng mà nến hiện tại không cung cấp được.
- `def batch_features` :49 — Tính đặc trưng phái sinh theo lô cho toàn bộ DataFrame nến.
- `class OnlineFeatureEngine` :81 — Bộ tính đặc trưng streaming cho live trading.
- `  . __init__` :89
- `  . reset` :97
- `  . update` :135
- `  . build_vector` :149

### `src/aegis/features/feature_spec.py` (189 dòng)
_[FIX F3] Đăng ký đặc trưng — NGUỒN SỰ THẬT DUY NHẤT cho mọi feature phái sinh._

- `class FeatureSpec` :40 — Định nghĩa một đặc trưng phái sinh.
- `def is_derived` :188

### `src/aegis/features/fractional_diff.py` (225 dòng)
_Module A.3 — Windowed FFD O(W*) Mặc Định & Sum-of-Exponentials O(M) nếu approv_

- `def compute_ffd_weights` :16 — [TASK A-4-1] Tính toán trọng số vi phân từng phần (Fractional Differentiation 
- `def apply_ffd` :57 — [TASK A-4-2] Áp dụng Fractional Differentiation lên chuỗi thời gian sử dụng cử
- `def find_optimal_d_star` :81 — [TASK A-4-2] Tìm bậc vi phân tối ưu d* nhỏ nhất làm cho chuỗi thời gian trở nê
- `def fit_sum_of_exponentials_v2` :158 — [TASK A-4-3] Fit Sum of Exponentials (Prony Approximation) cho trọng số FFD.
- `def select_ffd_production_engine` :204 — [TASK A-4-3] Windowed FFD (Phương án 1) là MẶC ĐỊNH production duy nhất. Sum-o

### `src/aegis/features/kalman/covariance_utils.py` (37 dòng)
_sanitize_covariance_matrix (Python & Rust closure) kẹp sàn eigenvalue_floor=1e_

- `def sanitize_covariance_matrix` :7 — [TASK A-4-5] Làm sạch ma trận hiệp phương sai (Covariance Matrix) để đảm bảo:

### `src/aegis/features/kalman/imm_kalman.py` (219 dòng)
_IMM Kalman 2D — 2 bộ lọc Trending & Choppy chạy song song kết hợp sanitize_cov_

- `class IMMKalman2D` :11 — Interacting Multiple Model (IMM) Kalman Filter 2D:
- `  . __init__` :20
- `  . reset` :63
- `  . step` :73
- `def filter_imm_kalman_series` :186 — Chạy bộ lọc IMM Kalman 2D qua toàn bộ chuỗi thời gian nến.

### `src/aegis/features/kalman/local_linear_trend.py` (111 dòng)
_Local Linear Trend Kalman Filter 2D [P_t, nu_t], xuất Trend_Score theo ATR._

- `class LocalLinearTrendKalman` :28
- `  . __init__` :29
- `  . reset` :51
- `  . step` :57
- `  . filter_series` :97

### `src/aegis/features/regime/bootstrap_lrt.py` (206 dòng)
_validate_two_regime_architecture_bootstrap (Parametric Bootstrap LRT N=1 vs N=_

- `def fit_single_gaussian_params` :8 — Khớp mô hình phân phối chuẩn 1 chiều (Single Gaussian) từ mảng quan sát.
- `def loglik_single_gaussian` :26 — Tính log-likelihood của O_array dựa trên params của Single Gaussian.
- `def simulate_from_single_gaussian` :54 — Sinh dữ liệu giả lập từ Single Gaussian.
- `def fit_hmm_2state_loglik` :79 — Fit HMM 2 trạng thái bằng thuật toán EM (Baum-Welch) với scaling an toàn.
- `def validate_two_regime_architecture_bootstrap` :177 — [TASK A-5-2] Thực hiện Parametric Bootstrap LRT kiểm định N=1 vs N=2.

### `src/aegis/features/regime/efficiency_regime.py` (115 dòng)
_[FIX F6] Phân loại chế độ thị trường nhân quả, KHÔNG có tham số phải fit._

- `class CausalEfficiencyRegime` :43 — Phân loại chế độ 2 trạng thái (Trend / Chop) bằng Efficiency Ratio làm mượt EW
- `  . __init__` :50
- `  . reset` :65
- `  . step` :69
- `  . filter_series` :106

### `src/aegis/features/regime/ghe.py` (67 dòng)
_Generalized Hurst Exponent (GHE) tính toán mức độ Persistent của chuỗi giá._

- `def compute_ghe` :9 — Tính Generalized Hurst Exponent bậc q.

### `src/aegis/features/regime/hmm_causal.py` (93 dòng)
_Causal-Only Hidden Markov Model (HMM) - Forward Alpha Pass._

- `class CausalHMM2State` :8
- `  . __init__` :9
- `  . step` :47
- `  . filter_series` :81

### `src/aegis/features/rolling.py` (52 dòng)
_[TASK A-3-2] Rolling Indicator Utilities_

- `def get_insufficient_history_mask` :12 — [TASK A-3-2] Tạo mask insufficient_history cho rolling indicator.

### `src/aegis/features/signal_pipeline.py` (188 dòng)
_Ghép nối Module A.3 + B thành một AegisSignalEngine chuẩn giao thức._

- `class AegisSignalEngine` :20 — Động cơ phát tín hiệu sơ cấp tổng hợp (Aegis Signal Engine).
- `  . __init__` :32
- `  . process_ohlcv_to_signal_bars` :55
- `def build_signal_bars` :180 — Hàm tiện ích cấp module để sinh SignalBarSchema DataFrame nhanh chóng.

### `src/aegis/governance/funding_accrual.py` (165 dòng)
_Module K.1 — Trừ chính xác chi phí funding (lãi qua đêm) của Perpetual Futures_

- `def funding_timestamps_between` :37 — Liệt kê các mốc quyết toán funding rơi vào khoảng (start_ts_ms, end_ts_ms].
- `def compute_funding_accrued_usd` :87 — Tổng chi phí funding (USD) mà vị thế phải gánh trong suốt thời gian nắm giữ.
- `def compute_funding_accrued_pct` :140 — Funding cộng dồn dưới dạng TỶ LỆ trên notional.

### `src/aegis/labeling/cusum_events.py` (273 dòng)
_cusum_events.py — Dynamic CUSUM Threshold Scaling & Spatial-Temporal Gating (T_

- `def compute_dynamic_cusum_thresholds` :24 — Tính toán ngưỡng CUSUM co giãn động h_t theo thời gian thực (Task B-2-1).
- `def filter_cusum_events_dynamic` :133 — Bộ lọc CUSUM biến động với Luật Cooldown & Gating Kép (Spatial-Temporal Cooldo

### `src/aegis/labeling/sample_weights.py` (120 dòng)
_Sample Weights & Average Uniqueness for Overlapping Labels._

- `def compute_num_concurrent_events` :14 — Tính số lượng sự kiện (trades) đang diễn ra (active) tại mỗi thời điểm t.
- `def compute_average_uniqueness` :60 — Tính Average Uniqueness cho mỗi sự kiện (trade).
- `def compute_sample_weights` :94 — Tính Sample Weights dựa trên Uniqueness và Absolute Returns.

### `src/aegis/labeling/trailing_exit.py` (728 dòng)

- `def round_sl_safe` :9 — Làm tròn giá Stop-Loss theo bước nhảy `tick_size` đảm bảo an toàn bất đối xứng
- `def compute_sl_initial` :32 — [TASK B-1-3] Tính toán mức Cắt lỗ gốc tĩnh (Initial Stop-Loss) hoàn toàn đối x
- `def resolve_regime_exit_threshold` :135 — [FIX F6b] Ngưỡng thoát Regime-Flip TỰ CHUẨN HOÁ theo phân phối p_trend thực tế
- `def compute_regime_aware_trailing_exit_v3_liquidation_aware` :208 — [v3 / v11.9] Nâng cấp từ v2: Tích hợp kiểm tra giá thanh lý (Liquidation Price
- `def simulate_trailing_exit_within_fold_bounds` :400 — [TASK B-1-5] SỬA LỖI: quay lại đúng nguyên tắc gốc — CẮT mảng future_* theo
- `def resolve_absolute_exit_idx` :502 — Chuyển offset TƯƠNG ĐỐI (k, trả về từ compute_regime_aware_trailing_exit_v3,
- `def run_trailing_exit_for_oos_event` :514 — [TASK B-1-8] Hàm glue nối 3 khâu: mô phỏng Trailing-Exit trong biên fold (B-1-
- `def finalize_trade_record` :620 — [TASK B-1-9] Bước hoàn thiện bản ghi cuối cùng: gắn realized_return vào bản gh

### `src/aegis/labeling/triple_barrier.py` (228 dòng)
_Dynamic HMM Triple-Barrier Labeling & compute_sl_initial đối xứng Long/Short._

- `def compute_dynamic_barriers` :10 — Tính rào cản Chốt lời (TP) và Cắt lỗ (SL) động theo xác suất regime HMM:
- `def apply_triple_barrier_single_event` :46 — Kiểm tra sự kiện chạm rào cản nào đầu tiên (TP, SL hay TIME stop):
- `def generate_meta_labels_triple_barrier` :164 — Sinh tập nhãn Meta-Labeling hoàn chỉnh cho danh sách sự kiện:

### `src/aegis/meta_labeling/calibration.py` (83 dòng)
_Isotonic Calibration using Purged K-Fold._

- `class PurgedKFoldAdapter` :15 — Adapter giúp PurgedKFold tương thích với sklearn CalibratedClassifierCV.
- `  . __init__` :23
- `  . split` :32
- `  . get_n_splits` :36
- `def build_calibrated_classifier` :41 — Tạo mô hình nắn chuẩn xác suất (Calibration) theo chuẩn AFML.

### `src/aegis/meta_labeling/feature_selection/clustering.py` (109 dòng)
_Hierarchical Clustering & Consensus Filter for Feature Selection._

- `def compute_distance_matrix` :12 — Tính ma trận khoảng cách dựa trên tương quan (Correlation-based Distance).
- `def cluster_features_hierarchical` :32 — Nhóm (Cluster) các feature bằng Hierarchical Clustering.
- `def apply_consensus_filter` :70 — Lọc Feature Trùng lặp (Consensus Filter).

### `src/aegis/meta_labeling/feature_selection/mdi_mda_sfi.py` (216 dòng)
_Triple Consensus Feature Selection (MDI, MDA, SFI)._

- `def compute_mdi` :18 — Tính Mean Decrease Impurity (MDI).
- `def compute_mda` :70 — Tính Mean Decrease Accuracy (MDA) bằng phương pháp OOS xáo trộn.
- `def compute_sfi` :140 — Tính Single Feature Importance (SFI).
- `def triple_consensus_ranker` :195 — Tổng hợp điểm MDI, MDA, SFI thành một bảng xếp hạng (Ranking).

### `src/aegis/meta_labeling/purged_kfold.py` (373 dòng)
_PurgedKFold Cross-Validation (Marcos Lopez de Prado - AFML Chapter 7)._

- `class PurgedKFold` :17 — [TASK B-1-14] PurgedKFold Cross-Validator (Marcos Lopez de Prado - AFML).
- `  . __init__` :25
- `  . split` :61
- `def assert_temporal_purging_invariant` :193 — [CANARY ASSERTION v11.9 — TEMPORAL PURGING & EMBARGO INVARIANT VERIFICATION]
- `def test_purged_kfold_no_overlap_simple` :226 — Kiểm tra trên chuỗi nến không chồng lấp t1 == t0 (mỗi nến chốt ngay tại bar đó
- `def test_purged_kfold_toy_overlap_mathematical_proof` :241 — [CRUCIAL MATHEMATICAL PROOF OF PURGING & EMBARGOING]:
- `def test_purged_kfold_override_priority` :300 — [TDD VERIFICATION - TASK B-1-14 OVERRIDE PRIORITY]:
- `def test_purged_kfold_input_validation_guards` :345 — Kiểm tra bẫy lỗi NaN / Inf và sai định dạng.

### `src/aegis/meta_labeling/sizing/kelly_empirical.py` (701 dòng)
_Empirical Kelly Sizing — solve E[log(1+f*r)], build_empirical_kelly_tables_v2._

- `class KellyConfidenceResult` :31 — Kết quả Bootstrap CI với đầy đủ thông tin chẩn đoán cho tầng giám sát.
- `def solve_empirical_kelly_fraction` :43 — [TASK B-1-1] Giải f* tối đa hóa kỳ vọng Log-growth E[log(1 + f*r)]
- `def solve_empirical_kelly_fraction_with_confidence` :104 — [PHÁT HIỆN M] Dùng Bootstrap để trích xuất phân vị bảo thủ + thông tin chẩn đo
- `def build_regime_returns_dict` :152 — [KHẮC PHỤC LỖ HỔNG - TRAIN/INFERENCE MISMATCH REGIME ASSIGNMENT]:
- `def compute_regime_weighted_bayesian_kelly` :202 — [PHÒNG THỦ KÉP & REGIME ALIGNMENT]:
- `def trade_records_to_kelly_table_inputs` :262 — [TASK B-1-11] Ánh xạ danh sách bản ghi giao dịch (TradeRecord dicts)
- `def build_empirical_kelly_table_v2` :352 — [TASK B-1-12] Dựng ma trận Kelly 2D (shape: num_bins x num_bins) từ đầu ra B-1
- `def compute_bi_directional_kelly_v14_unified` :416 — [TASK B-1-13] Hàm suy luận O(1) thống nhất cho Sizing (Inference Layer).
- `def test_f_max_is_leverage_search_bound` :473 — [QĐ #1] Xác nhận f_max = 20.0 là giới hạn TÌM KIẾM cho brentq,
- `def test_fractional_kelly_lambda_discount` :483 — [QĐ #2] Kiểm tra hệ số chiết khấu λ = 0.5 (Half Kelly)
- `def test_b_1_1_kelly_classical_coin_toss` :501 — UNIT TEST CHO B-1-1:
- `def test_kelly_canary_and_nan_safety` :529 — [FINDING C] Kiểm tra thứ tự: Canary assertion chạy SAU khi lọc NaN/Inf.
- `def test_kelly_dynamic_cap_with_liquidation` :551 — Kiểm chứng rằng khi mẫu chứa lệnh thanh lý (r = -1.0),
- `def test_regime_probability_blend_and_bayesian` :573 — [BAYESIAN-HMM] Kiểm tra chức năng phối trộn Kelly theo xác suất Regime
- `def test_b_1_11_trade_records_to_kelly_table_inputs` :600 — Kiểm tra Task B-1-11:
- `def test_b_1_12_build_empirical_kelly_table_v2` :621 — Kiểm tra Task B-1-12:
- `def test_b_1_13_compute_bi_directional_kelly_v14_unified` :652 — Kiểm tra Task B-1-13:

### `src/aegis/meta_labeling/sizing/liquidation_layer.py` (376 dòng)
_[v11.9] LIQUIDATION LAYER — XẤP XỈ GIÁ THANH LÝ & BẢO VỆ ĐÒN BẦY AN TOÀN PERPE_

- `def compute_liquidation_price` :21 — [v11.9] Xấp xỉ giá thanh lý cho Isolated Margin Perpetual Futures.
- `def validate_leverage_against_sl` :125 — [v11.9] Pre-Flight Check: Đảm bảo khoảng cách Cắt Lỗ (SL) đủ an toàn trước khi
- `def get_maintenance_margin_rate` :219 — Tra cứu tỷ lệ Ký quỹ duy trì (MMR) theo quy mô vị thế.
- `def resolve_max_safe_leverage` :235 — [v11.9 + Lỗ Hổng 6] Giải closed-form đòn bẩy tối đa cho phép có tính khấu hao 
- `def compute_liquidation_loss` :353 — [QĐ #7] Tính tổn thất vốn ký quỹ (Isolated Margin) khi bị thanh lý.

### `src/aegis/meta_labeling/sizing/trade_mode.py` (194 dòng)
_classify_trade_mode() — HÀM DUY NHẤT phân loại follow/fade/none (v11.6 Patch C_

- `def classify_trade_mode` :10 — Hàm DUY NHẤT phân loại Follow / Fade / None.
- `def resolve_trade_execution_params` :84 — [TASK B-1-6] Hàm glue NỐI 3 module đã có (B-1-2 classify_trade_mode,

### `src/aegis/meta_labeling/weighted_bootstrap_forest.py` (119 dòng)
_Weighted Bootstrap Forest Classifier._

- `class WeightedBootstrapForestClassifier` :17 — Random Forest tùy chỉnh chuẩn AFML:
- `  . __init__` :34
- `  . fit` :46
- `  . predict_proba` :88
- `  . predict` :116

### `src/aegis/oms/order_router.py` (442 dòng)
_Điều hướng lệnh sang Binance USDⓈ-M Futures._

- `class OrderRejected` :30 — Lệnh bị từ chối TRƯỚC khi gửi (vi phạm kiểm tra an toàn cục bộ).
- `def make_client_order_id` :34 — Sinh client order id TẤT ĐỊNH.
- `class BinanceOrderRouter` :46 — Gửi lệnh tới Binance Futures và theo dõi vòng đời qua OrderBook.
- `  . __init__` :49
- `  . configure_symbol` :67
- `  . submit` :107
- `  . poll_status` :201
- `  . submit_plan` :230
- `  . execute_with_fallback` :261
- `  . cancel_all` :400
- `  . kill_switch` :409

### `src/aegis/oms/reconciliation.py` (141 dòng)
_Đối chiếu trạng thái nội bộ với trạng thái THẬT trên sàn._

- `class Discrepancy` :27
- `  . delta` :34
- `class ReconciliationReport` :39
- `  . is_clean` :48
- `  . summary` :51
- `class ReconciliationError` :65 — Trạng thái lệch — TUYỆT ĐỐI không giao dịch tiếp cho tới khi giải quyết.
- `def reconcile` :69 — So trạng thái nội bộ với sàn.
- `def assert_clean_or_raise` :127 — Chốt chặn khởi động: lệch sổ sách thì DỪNG, không tự đoán.

### `src/aegis/oms/state_machine.py` (186 dòng)
_OMS State Machine — vòng đời lệnh và các chuyển trạng thái HỢP LỆ._

- `class OrderState` :18 — Trạng thái lệnh. Trạng thái CUỐI không bao giờ rời đi được.
- `class InvalidTransitionError` :68 — Chuyển trạng thái không hợp lệ — dấu hiệu logic sai hoặc sự kiện tới lệch thứ 
- `class ManagedOrder` :73 — Một lệnh cùng toàn bộ lịch sử vòng đời của nó.
- `  . is_terminal` :96
- `  . remaining_qty` :100
- `  . transition` :103
- `  . apply_exchange_status` :150
- `class OrderBook` :158 — Sổ theo dõi toàn bộ lệnh của phiên, tra cứu theo client_order_id.
- `  . __init__` :161
- `  . add` :164
- `  . get` :170
- `  . open_orders` :173
- `  . all_orders` :176
- `  . net_position` :179

### `src/aegis/pipelines/cpcv_pipeline.py` (519 dòng)
_Điều phối 15 folds CPCV theo đúng thứ tự 5 bước v11.8 (Mục 4.0)._

- `def filter_boundary_truncated_for_kelly_table` :63 — [BƯỚC 2.5 — QUY TRÌNH 5 BƯỚC v11.8]:
- `class CPCVPipeline` :98 — Điều phối Combinatorial Purged Cross-Validation (CPCV) chuẩn 5 bước v11.8:
- `  . __init__` :108
- `  . run` :120

### `src/aegis/pipelines/live_pipeline.py` (673 dòng)
_live_pipeline.py — Institutional Causal Streaming Live / Paper Trading Engine _

- `class ActivePosition` :51 — Đại diện cho vị thế đang mở trong phiên live/paper.
- `class AegisLivePipeline` :73 — Hệ thống vận hành trực tiếp (Live / Paper Trading Engine).
- `  . __init__` :78
- `  . from_artifacts` :135
- `  . on_bar` :268

### `src/aegis/pipelines/research_pipeline.py` (317 dòng)
_research_pipeline.py — Institutional Full-Fit Production Pipeline (Task 7)._

- `class ResearchPipeline` :53 — Điều phối quy trình huấn luyện toàn diện (Research Full-Fit Pipeline).
- `  . __init__` :58
- `  . run` :64

### `src/aegis/pipelines/xs_live_pipeline.py` (542 dòng)
_Pipeline live cho chiến lược cross-sectional market-neutral._

- `class LiveConfig` :58 — Cấu hình đã qua kiểm định holdout — xem artifacts/strategy_validated.json.
- `  . from_artifacts` :98
- `class CrossSectionalLivePipeline` :137 — Vòng lặp vận hành chiến lược cross-sectional.
- `  . __init__` :140
- `  . refresh_data` :160
- `  . load_filters` :173
- `  . resolve_universe` :187
- `  . compute_target_weights` :271
- `  . run_once` :370
- `  . kill` :535

### `src/aegis/research/adaptive_combiner.py` (227 dòng)
_Gộp tín hiệu THÍCH ỨNG theo cửa sổ trượt — quyết định trọng số bằng dữ liệu QU_

- `class CombinerSpec` :49 — Tham số của tầng gộp thích ứng.
- `def factor_returns` :62 — Chuỗi lợi suất của "danh mục nhân tố" ứng với MỘT tín hiệu.
- `def adaptive_weights` :129 — Trọng số từng họ theo thời gian, tính HOÀN TOÀN từ dữ liệu quá khứ.
- `def combine_adaptive` :183 — Gộp nhiều tín hiệu thành MỘT điểm số tổng hợp bằng trọng số thích ứng.

### `src/aegis/research/backtest_v2.py` (340 dòng)
_Engine backtest cross-sectional thế hệ 2 — nhận TRỌNG SỐ dựng sẵn._

- `class CostModel` :47 — Chi phí một chiều theo bp trên notional giao dịch.
- `  . base_bps` :68
- `  . bps_for` :74
- `def load_live_spreads` :87 — Spread THẬT đo từ sổ lệnh Binance (`/fapi/v1/ticker/bookTicker`), lưu sẵn ra f
- `def corwin_schultz_spread` :106 — Ước lượng spread Corwin-Schultz (2012) — CHỈ dùng khi không có spread thật.
- `def estimate_cost_bps` :136 — Chi phí TRƯỢT GIÁ một chiều theo từng cặp, tính bằng bp (chưa gồm phí sàn).
- `def drift_weights` :185 — Trọng số sau một chu kỳ, do giá dịch chuyển chứ không do giao dịch.
- `class BacktestV2Result` :204
- `  . stats` :216
- `def simulate` :259 — Mô phỏng danh mục từ chuỗi trọng số MỤC TIÊU.

### `src/aegis/research/cross_sectional.py` (414 dòng)
_Engine backtest cross-sectional (market-neutral) đa tài sản._

- `class BacktestResult` :24 — Kết quả backtest cross-sectional.
- `  . n_periods` :34
- `  . stats` :37
- `def rank_to_weights` :65 — Biến tín hiệu thành trọng số dollar-neutral.
- `def rank_to_weights_buffered` :97 — Xếp hạng có VÙNG ĐỆM — giảm turnover cho danh mục dựa trên thứ hạng.
- `def backtest_cross_sectional` :173 — Backtest danh mục cross-sectional có tính đủ chi phí.
- `def xs_zscore` :237 — Chuẩn hoá theo HÀNG (cross-sectional): khử mức chung, giữ thứ hạng tương đối.
- `def signal_momentum` :244 — Động lượng cross-sectional: lợi suất quá khứ, bỏ qua `skip` nến gần nhất.
- `def signal_reversal` :249 — Hồi quy ngắn hạn: NGƯỢC dấu lợi suất gần nhất.
- `def signal_funding_carry` :254 — Carry funding: BÁN cặp có funding cao, MUA cặp có funding thấp.
- `def signal_low_vol` :264 — Bất thường biến động thấp: mua tài sản ít biến động, bán tài sản nhiều.
- `def signal_ofi` :270 — Mất cân bằng dòng lệnh cross-sectional (vi cấu trúc).
- `def signal_risk_adjusted_momentum` :278 — Động lượng chia cho biến động ("Sharpe momentum").
- `def signal_long_term_reversal` :294 — Đảo chiều dài hạn: NGƯỢC dấu lợi suất rất dài hạn (bỏ qua đoạn gần đây).
- `def signal_negative_skew` :305 — Cầu xổ số (lottery demand): BÁN tài sản có độ lệch DƯƠNG cao.
- `def signal_funding_momentum` :317 — Đà thay đổi funding: funding đang TĂNG báo hiệu vị thế long đang chen chúc.
- `def signal_idiosyncratic_vol` :328 — Bất thường biến động riêng: MUA tài sản có biến động RIÊNG thấp.
- `def combine_inverse_vol` :341 — Gộp nhiều chiến lược theo trọng số NGHỊCH ĐẢO BIẾN ĐỘNG (risk parity đơn giản)
- `def volatility_target` :360 — Điều tiết vị thế theo mục tiêu biến động.
- `def combine_signals_zscore` :383 — Gộp nhiều tín hiệu cross-sectional thành MỘT điểm số tổng hợp.

### `src/aegis/research/ensemble.py` (139 dòng)
_Gộp DANH MỤC từ nhiều cấu hình — chống rủi ro chọn sai siêu tham số._

- `class EnsembleMember` :39 — Một thành phần: tín hiệu trên lưới riêng + cách dựng danh mục riêng.
- `def align_to_grid` :48 — Đưa trọng số của một thành phần về lưới CHUNG.
- `def ensemble_weights` :59 — Trọng số danh mục tổ hợp trên lưới chung `grid`.

### `src/aegis/research/holdout.py` (63 dòng)
_Kỷ luật train / holdout — hàng rào chống quá khớp khi tinh chỉnh chiến lược._

- `class TimeSplit` :26 — Ranh giới tách train/holdout theo thời gian.
- `  . train` :31
- `  . holdout` :34
- `  . describe` :37
- `def make_split` :45 — Tách theo thời gian tại phân vị `train_frac` của chỉ mục.
- `def split_panel` :54 — Cắt toàn bộ panel về một phía của ranh giới.

### `src/aegis/research/ic_analysis.py` (423 dòng)
_Phân tích hệ số thông tin (IC) — chẩn đoán tín hiệu TRƯỚC khi dựng danh mục._

- `class ICResult` :49 — IC theo từng nến cộng phần tóm tắt.
- `  . mean` :56
- `  . std` :60
- `  . ir` :64
- `  . t_stat` :69
- `  . hit_rate` :81
- `  . summary` :84
- `def information_coefficient` :122 — Tương quan theo mặt cắt ngang giữa tín hiệu tại t và lợi suất t -> t+horizon.
- `def ic_decay_curve` :160 — IC theo nhiều chân trời — công cụ chọn CHU KỲ TÁI CÂN BẰNG.
- `def ic_by_period` :185 — IC gộp theo năm/quý — phát hiện edge đang chết.
- `def signal_autocorrelation` :213 — Tự tương quan mặt cắt ngang của tín hiệu qua các độ trễ.
- `def effective_breadth` :239 — Độ rộng HIỆU DỤNG sau chiết khấu vì các cược không độc lập.
- `def implied_sharpe` :271 — Sharpe kỳ vọng theo định luật cơ bản, đã tính hệ số truyền tải.
- `def ic_correlation_matrix` :283 — Tương quan giữa các CHUỖI IC, không phải giữa các chuỗi lợi suất.
- `def quantile_returns` :309 — Lợi suất trung bình theo từng NHÓM PHÂN VỊ của tín hiệu.
- `def decile_spread` :366 — Chênh lệch lợi suất giữa nhóm đầu và nhóm cuối — ĐÚNG đại lượng danh mục kiếm 
- `def monotonicity` :409 — Mức độ đơn điệu: tương quan Spearman giữa số thứ tự nhóm và lợi suất nhóm.

### `src/aegis/research/leverage.py` (303 dòng)
_Đòn bẩy, mục tiêu biến động, và XÁC SUẤT ĐẠT MỤC TIÊU — trả lời bằng số, không_

- `class LeverageSpec` :51 — Tham số tầng đòn bẩy.
- `def kelly_leverage` :64 — Đòn bẩy tối ưu tăng trưởng (Kelly), nhân với `fraction`.
- `def volatility_target_series` :79 — Hệ số đòn bẩy theo thời gian để đưa biến động VỐN về mục tiêu.
- `def drawdown_throttle` :97 — Hệ số giảm đòn bẩy khi đang trong drawdown, giảm tuyến tính về 0.
- `def apply_leverage` :117 — Áp tầng đòn bẩy lên chuỗi lợi suất chưa đòn bẩy (gross = 1.0).
- `def simulate_paths` :165 — Block bootstrap phân phối kết quả sau `n_periods` kỳ ở đòn bẩy cố định.
- `def target_probability_table` :210 — Bảng đánh đổi: mỗi mức đòn bẩy -> xác suất đạt mục tiêu, xác suất cháy, trung 
- `def required_sharpe_for_target` :265 — Sharpe cần có để đạt mục tiêu với xác suất `confidence`, ở mức biến động cho t

### `src/aegis/research/signal_library.py` (425 dòng)
_Thư viện tín hiệu cross-sectional, tổ chức theo HỌ KINH TẾ._

- `def xs_zscore` :47 — Z-score theo HÀNG. Kẹp đuôi trước khi chuẩn hoá lại để một giá trị ngoại lai
- `def xs_rank` :57 — Thứ hạng theo hàng, ánh xạ về [-1, 1].
- `class SignalDef` :72 — Một tín hiệu: giả thuyết, họ, hàm tính, và số nến cần khởi động.
- `def build_signal` :385 — Tính MỘT tín hiệu và chuẩn hoá theo mặt cắt ngang.
- `def build_family` :396 — Gộp mọi biến thể trong một họ thành MỘT tín hiệu cấp họ.
- `def build_all_families` :420 — Dựng tín hiệu cấp họ cho mọi họ — đây là đầu vào của tầng gộp.

### `src/aegis/research/strategy_v2.py` (179 dòng)
_Chiến lược v2 — MỘT đường đi duy nhất từ dữ liệu tới trọng số vị thế._

- `class StrategyV2Config` :40 — Toàn bộ tham số chiến lược — khai báo MỘT chỗ.
- `  . bar_hours` :66
- `  . period_hours` :70
- `  . periods_per_year` :73
- `  . to_dict` :76
- `class StrategyV2` :83 — Chiến lược cross-sectional market-neutral thế hệ 2.
- `  . __init__` :86
- `  . score` :91
- `  . target_weights` :115
- `  . backtest` :127

### `src/aegis/risk/circuit_breaker.py` (79 dòng)
_3-Tier Drawdown Circuit Breakers (Module J)._

- `class CircuitBreakerTier` :11
- `class CircuitBreakerState` :18
- `class CircuitBreaker` :24
- `  . __init__` :25
- `  . update_equity` :45

### `src/aegis/risk/covariance_shrinkage.py` (200 dòng)
_Ước lượng ma trận hiệp phương sai bền vững cho danh mục đa tài sản._

- `def ledoit_wolf_constant_correlation` :43 — Ledoit-Wolf co về mục tiêu TƯƠNG QUAN HẰNG SỐ (Ledoit & Wolf 2004, "Honey,
- `def single_index_covariance` :107 — Hiệp phương sai theo mô hình MỘT NHÂN TỐ: sigma = beta beta' var_m + diag(var_
- `def ewma_covariance` :132 — Hiệp phương sai có trọng số mũ — phản ứng nhanh với đổi chế độ biến động.
- `def nearest_positive_definite` :157 — Kẹp trị riêng để ma trận chắc chắn xác định dương.
- `def shrink_covariance` :174 — Điểm vào duy nhất cho ước lượng hiệp phương sai.

### `src/aegis/risk/drift_monitor.py` (146 dòng)
_drift_monitor.py — CUSUM Brier drift monitoring & refresh_cusum_thresholds (Ta_

- `def refresh_cusum_thresholds` :25 — Tính toán và tái tạo cấu hình ngưỡng CUSUM sau production_fit (Task B-2-1).
- `def monitor_brier_score_cusum_drift` :83 — Theo dõi độ trôi sai số Brier Score bằng bộ lọc CUSUM (Page 1954).

### `src/aegis/risk/portfolio.py` (537 dòng)
_Dựng trọng số danh mục cross-sectional — tầng biến TÍN HIỆU thành VỊ THẾ._

- `class PortfolioSpec` :50 — Toàn bộ tham số dựng danh mục — khai báo một chỗ, dùng chung research và live.
- `  . side_count` :80
- `  . effective_min_names` :93
- `def realized_vol` :110 — Biến động thực hiện theo nến, tính tới t-1.
- `def rolling_beta` :122 — Beta của từng cặp so với "thị trường" (mặc định: rổ đều trọng số của universe)
- `def inverse_vol_weights` :148 — Chia tỷ lệ trọng số theo NGHỊCH ĐẢO biến động — chia đều RỦI RO thay vì VỐN.
- `def beta_neutralize` :161 — Trừ đi phần hình chiếu của danh mục lên nhân tố thị trường.
- `def project_neutral` :184 — Chiếu trọng số lên không gian thoả ĐỒNG THỜI: tổng trọng số = 0 và tổng
- `def apply_buffer` :288 — Mặt nạ vùng đệm: +1 giữ long, -1 giữ short, 0 ngoài danh mục.
- `def build_weights` :374 — Biến điểm số cross-sectional thành trọng số danh mục.

### `src/aegis/validation/cpcv.py` (162 dòng)
_[v11.9] Combinatorial Purged Cross-Validation (CPCV) — AFML Chapter 12._

- `class CombinatorialPurgedKFold` :16 — Combinatorial Purged Cross-Validation (CPCV) Engine.
- `  . __init__` :25
- `  . split` :49
- `  . generate_backtest_paths` :140

### `src/aegis/validation/dsr.py` (180 dòng)
_[v11.9] Deflated Sharpe Ratio (DSR >= 0.95) & Sensitivity Analysis Engine._

- `def euler_mascheroni_approx_max_sr` :12 — Xấp xỉ kỳ vọng giá trị lớn nhất của Sharpe Ratio dưới giả thuyết Null (SR_0^*)
- `def compute_probabilistic_sharpe_ratio` :40 — Tính Probabilistic Sharpe Ratio (PSR) — Xác suất SR ước lượng vượt qua SR_benc
- `def compute_deflated_sharpe_ratio` :66 — [ARMOR-PLATED GUARDS — HẢI QUAN BỌC THÉP]:
- `def compute_dsr_sensitivity` :149 — [MODULE E SENSITIVITY DIRECTIVE]:

### `src/aegis/validation/pbo_cscv.py` (120 dòng)
_[v11.9] Probability of Backtest Overfitting (PBO <= 0.40) & Combinatorial Symm_

- `def compute_pbo_cscv` :12 — Tính Probability of Backtest Overfitting (PBO) theo phương pháp Combinatorial 

