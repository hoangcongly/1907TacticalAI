"""
cusum_events.py — Dynamic CUSUM Threshold Scaling & Spatial-Temporal Gating (Task B-2-1).

Quy chuẩn B-2-1:
- Khắc phục Lỗ Hổng 1 (Price Scaling Division Hazard): Sử dụng Robust Smoothed Anchor Price
  (EWMA) thay vì giá tức thời P_t để làm mẫu số cho relative volatility. Điều này đảm bảo
  ngưỡng CUSUM không bị biến dạng khi xảy ra Flash Crash sập sốc hoặc khi chuyển đổi
  giữa các coin khác mệnh giá (BTC vs ETH vs SOL).
- Khắc phục rác dữ liệu: Armor-Plated Guards chặn đứng NaN/Inf, giá trị âm, hoặc mẫu số bằng 0.
- Spatial-Temporal Cooldown Gating: Yêu cầu đồng thời bar cách quãng >= cooldown_bars
  VÀ độ lệch giá >= spatial_delta_atr * ATR_t.
- v11.6 C.4 Metadata Convention: Mỗi sự kiện CUSUM hợp lệ bắt buộc lưu trade_mode và side.
"""

import math
from typing import Optional
import numpy as np
from aegis.meta_labeling.sizing.trade_mode import classify_trade_mode


# ============================================================================
# [TASK B-2-1] DYNAMIC CUSUM THRESHOLD SCALING WITH SMOOTHED ANCHOR PRICE
# ============================================================================
def compute_dynamic_cusum_thresholds(
    prices: np.ndarray,
    atr_series: np.ndarray,
    base_multiplier: float = 2.5,
    anchor_span: int = 50,
    min_rel_threshold: float = 1e-4,
    max_rel_threshold: float = 0.05,
    bars_per_atr_period: float = 1.0,
    use_ewma_anchor: bool = False,
    reset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Tính toán ngưỡng CUSUM co giãn động h_t theo thời gian thực (Task B-2-1).

    [KHẮC PHỤC LỖ HỔNG 1 - PRICE SCALING DIVISION HAZARD & THE EWMA GAP PROTOCOL]:
    - Trong không gian Log-Return phi thứ nguyên với Định luật Căn bậc hai Thời gian (Square-root of time):
        sigma_r,t = (ATR_t / P_anchor,t) * (1 / sqrt(bars_per_atr_period))
        h_log,t = clip(base_multiplier * sigma_r,t, min_rel_threshold, max_rel_threshold)
    - Tích hợp Robust Smoothed Anchor Price (EWMA P_bar_t, span=anchor_span) cùng Gap-Handling Protocol:
      Khi `use_ewma_anchor=True`, giá neo mẫu số được làm mượt bởi EWMA để chống biến dạng khi Flash Crash.
      Đặc biệt tuân thủ kỷ luật Giai đoạn 0 (Predict-only reset / insufficient_history):
      Khi gặp khoảng trống giá thực tế (gap) hoặc ranh giới khối CPCV (được đánh dấu qua `reset_mask[t] == True`),
      bộ nhớ EWMA buộc phải reset tự động về giá trị hiện tại P_bar_t = P_t để ngăn chặn rò rỉ dữ liệu hoặc méo ngưỡng.

    [ARMOR-PLATED GUARDS]:
    - Kiểm tra type và shape (1D numpy arrays, độ dài bằng nhau).
    - Chặn NaN / Inf, P_t <= 0, ATR_t < 0, bars_per_atr_period <= 0.
    """
    if not isinstance(prices, (list, tuple, np.ndarray)) or not isinstance(
        atr_series, (list, tuple, np.ndarray)
    ):
        raise ValueError("prices và atr_series phải là mảng numpy hoặc list/tuple.")

    prices_np = np.asarray(prices, dtype=np.float64)
    atr_np = np.asarray(atr_series, dtype=np.float64)

    if prices_np.ndim != 1 or atr_np.ndim != 1:
        raise ValueError("prices và atr_series phải là mảng 1D.")
    if len(prices_np) != len(atr_np):
        raise ValueError(
            f"Độ dài prices ({len(prices_np)}) không khớp với atr_series ({len(atr_np)})."
        )
    if len(prices_np) == 0:
        raise ValueError("Mảng đầu vào không được rỗng.")

    if not isinstance(base_multiplier, (int, float)) or base_multiplier <= 0 or math.isnan(base_multiplier) or math.isinf(base_multiplier):
        raise ValueError(f"base_multiplier phải là số thực dương, nhận {base_multiplier}")
    if not isinstance(anchor_span, int) or anchor_span <= 0:
        raise ValueError(f"anchor_span phải là số nguyên dương, nhận {anchor_span}")
    if not isinstance(bars_per_atr_period, (int, float)) or bars_per_atr_period <= 0 or math.isnan(bars_per_atr_period) or math.isinf(bars_per_atr_period):
        raise ValueError(f"bars_per_atr_period phải là số thực dương, nhận {bars_per_atr_period}")
    if (
        not isinstance(min_rel_threshold, (int, float))
        or not isinstance(max_rel_threshold, (int, float))
        or math.isnan(min_rel_threshold)
        or math.isnan(max_rel_threshold)
        or math.isinf(min_rel_threshold)
        or math.isinf(max_rel_threshold)
        or min_rel_threshold <= 0
        or max_rel_threshold <= min_rel_threshold
        or max_rel_threshold > 1.0
    ):
        raise ValueError(
            f"min_rel_threshold và max_rel_threshold không hợp lệ: ({min_rel_threshold}, {max_rel_threshold})"
        )

    # Kiểm tra NaN / Inf / non-positive prices và negative ATR
    if np.any(np.isnan(prices_np)) or np.any(np.isinf(prices_np)) or np.any(prices_np <= 0):
        raise ValueError("prices chứa NaN, Inf hoặc giá <= 0 phi lý.")
    if np.any(np.isnan(atr_np)) or np.any(np.isinf(atr_np)) or np.any(atr_np < 0):
        raise ValueError("atr_series chứa NaN, Inf hoặc ATR < 0 phi lý.")

    if reset_mask is not None:
        reset_mask_np = np.asarray(reset_mask, dtype=bool)
        if len(reset_mask_np) != len(prices_np):
            raise ValueError("reset_mask phải có độ dài bằng với prices.")
    else:
        reset_mask_np = np.zeros(len(prices_np), dtype=bool)

    # [KHẮC PHỤC LỖ HỔNG 2 - EWMA LAG TRAP]:
    # Tính toán biến động tương đối tức thời (instantaneous relative volatility) trong không gian Log-Return.
    # Loại bỏ hoàn toàn phụ thuộc vào mẫu số EWMA(P_t) bị trễ trong Flash Crash.
    inst_rel_vol = (atr_np / prices_np) * (1.0 / math.sqrt(float(bars_per_atr_period)))

    if use_ewma_anchor:
        alpha = 2.0 / (float(anchor_span) + 1.0)
        sigma_rel = np.zeros_like(inst_rel_vol)
        current_val = float(inst_rel_vol[0])
        sigma_rel[0] = current_val
        for t in range(1, len(inst_rel_vol)):
            # Gap-Handling Protocol (Predict-only reset): nếu gặp gap thật hoặc ranh giới CPCV -> reset về giá trị tức thời
            if reset_mask_np[t]:
                current_val = float(inst_rel_vol[t])
            else:
                current_val = alpha * float(inst_rel_vol[t]) + (1.0 - alpha) * current_val
            sigma_rel[t] = current_val
    else:
        sigma_rel = inst_rel_vol

    sigma_clamped = np.clip(
        base_multiplier * sigma_rel, min_rel_threshold, max_rel_threshold
    )

    return sigma_clamped


# ============================================================================
# [TASK B-2-1] CUSUM EVENT GENERATION WITH SPATIAL-TEMPORAL GATING
# ============================================================================
def filter_cusum_events_dynamic(
    prices: np.ndarray,
    dynamic_thresholds: np.ndarray,
    timestamps_ms: np.ndarray,
    atr_series: np.ndarray,
    cooldown_bars: int = 10,
    spatial_delta_atr: float = 1.0,
    trend_scores: Optional[np.ndarray] = None,
    p_trend: Optional[np.ndarray] = None,
    p_chop: Optional[np.ndarray] = None,
    fade_enabled: bool = False,
    fade_regime_gate_threshold: float = 0.60,
) -> list[dict]:
    """
    Bộ lọc CUSUM biến động với Luật Cooldown & Gating Kép (Spatial-Temporal Cooldown Gating).

    [COMPOSITION LOGIC 2 TẦNG RÕ RÀNG - AFML CH. 2 & MODULE C.1]:
    - Tầng 1 (Candidate Generation Gate): Phát hiện ứng viên sự kiện khi tích lũy Log-Return phá vỡ
      ngưỡng h_t trên HOẶC dưới: (S_t^+ > h_t) HOẶC (S_t^- < -h_t).
      Ngay tại thời điểm vọt ngưỡng này, bộ nhớ CUSUM lập tức được reset về 0 (s_plus = 0, s_minus = 0)
      để bắt đầu chu kỳ theo dõi mới độc lập.
    - Tầng 2 (Candidate Confirmation Gate - ĐỒNG THỜI AND): Một ứng viên từ Tầng 1 CHỈ ĐƯỢC XÁC NHẬN
      thành tín hiệu giao dịch hợp lệ nếu thỏa mãn ĐỒNG THỜI (AND) cả 2 điều kiện cách ly không-thời gian:
      1. Temporal Check: (i - last_event_idx) >= cooldown_bars
      2. Spatial Check: |P_i - P_last_event| > spatial_delta_atr * ATR_i
      Nếu vi phạm dù chỉ 1 trong 2 điều kiện (lệnh quá sát thời gian HOẶC giá chưa đi đủ xa),
      ứng viên sự kiện sẽ bị chặn lại và loại bỏ (trong khi accumulator đã được reset từ Tầng 1).

    [QUY ƯỚC v11.6 C.4 - METADATA]:
    Đính kèm `trade_mode` (thông qua `classify_trade_mode`) và `side` (`side_follow`, `side_fade`)
    vào mỗi bản ghi sự kiện, phục vụ tái tạo trong Module F.
    """
    prices_np = np.asarray(prices, dtype=np.float64)
    thresholds_np = np.asarray(dynamic_thresholds, dtype=np.float64)
    timestamps_np = np.asarray(timestamps_ms, dtype=np.int64)
    atr_np = np.asarray(atr_series, dtype=np.float64)

    n = len(prices_np)
    if not (len(thresholds_np) == len(timestamps_np) == len(atr_np) == n):
        raise ValueError("Tất cả mảng đầu vào (prices, thresholds, timestamps, atr) phải có cùng độ dài.")
    if n == 0:
        return []

    if not isinstance(cooldown_bars, int) or cooldown_bars < 0:
        raise ValueError(f"cooldown_bars phải là số nguyên >= 0, nhận {cooldown_bars}")
    if not isinstance(spatial_delta_atr, (int, float)) or spatial_delta_atr < 0 or math.isnan(spatial_delta_atr) or math.isinf(spatial_delta_atr):
        raise ValueError(f"spatial_delta_atr phải là số thực >= 0, nhận {spatial_delta_atr}")

    # Kiểm tra tính hợp lệ arrays
    if np.any(np.isnan(prices_np)) or np.any(np.isinf(prices_np)):
        raise ValueError("prices chứa NaN hoặc Inf.")
    if np.any(np.isnan(thresholds_np)) or np.any(np.isinf(thresholds_np)) or np.any(thresholds_np <= 0):
        raise ValueError("dynamic_thresholds chứa NaN, Inf hoặc <= 0 phi lý.")

    if trend_scores is not None:
        trend_scores_np = np.asarray(trend_scores, dtype=np.float64)
        if len(trend_scores_np) != n or np.any(np.isnan(trend_scores_np)):
            raise ValueError("trend_scores độ dài không khớp hoặc chứa NaN.")
    else:
        trend_scores_np = None

    if p_trend is not None and p_chop is not None:
        p_trend_np = np.asarray(p_trend, dtype=np.float64)
        p_chop_np = np.asarray(p_chop, dtype=np.float64)
        if len(p_trend_np) != n or len(p_chop_np) != n:
            raise ValueError("p_trend và p_chop độ dài không khớp.")
    else:
        p_trend_np = None
        p_chop_np = None

    events: list[dict] = []
    s_plus = 0.0
    s_minus = 0.0
    last_event_idx = -1
    last_event_price = -1.0

    for i in range(1, n):
        # Tích lũy Log-Return r_i = ln(P_i / P_{i-1})
        r_i = math.log(prices_np[i] / prices_np[i - 1])
        s_plus = max(0.0, s_plus + r_i)
        s_minus = min(0.0, s_minus + r_i)
        h_i = float(thresholds_np[i])

        triggered = False
        raw_dir = 0
        # Tầng 1 (Candidate Generation): vọt ngưỡng trên HOẶC dưới
        if s_plus > h_i:
            triggered = True
            raw_dir = 1
        elif s_minus < -h_i:
            triggered = True
            raw_dir = -1

        if triggered:
            # Reset CUSUM accumulators ngay khi có candidate (Page 1954)
            s_plus = 0.0
            s_minus = 0.0

            # Tầng 2 (Candidate Confirmation Gate): bắt buộc thỏa mãn ĐỒNG THỜI (AND) temporal VÀ spatial
            temporal_ok = (last_event_idx == -1) or ((i - last_event_idx) >= cooldown_bars)
            spatial_ok = (last_event_idx == -1) or (
                abs(prices_np[i] - last_event_price) > float(spatial_delta_atr) * float(atr_np[i])
            )

            # Luật AND đồng thời tuyệt đối
            if temporal_ok and spatial_ok:
                # Xác định trade_mode theo v11.6 C.4
                if p_trend_np is not None and p_chop_np is not None:
                    mode = classify_trade_mode(
                        float(p_trend_np[i]),
                        float(p_chop_np[i]),
                        fade_enabled=fade_enabled,
                        fade_regime_gate_threshold=fade_regime_gate_threshold,
                    )
                else:
                    mode = "none"

                # Xác định side
                if trend_scores_np is not None and mode != "none":
                    side_follow = 1 if trend_scores_np[i] >= 0 else -1
                    side = side_follow if mode == "follow" else -side_follow
                elif mode != "none":
                    side = raw_dir if mode == "follow" else -raw_dir
                else:
                    side = raw_dir

                event_record = {
                    "bar_idx": int(i),
                    "timestamp_ms": int(timestamps_np[i]),
                    "price": float(prices_np[i]),
                    "threshold": float(h_i),
                    "atr": float(atr_np[i]),
                    "trade_mode": mode,
                    "side": int(side),
                    "raw_dir": int(raw_dir),
                }
                events.append(event_record)
                last_event_idx = i
                last_event_price = float(prices_np[i])

    return events
