"""
Ghép nối Module A.3 + B thành một AegisSignalEngine chuẩn giao thức.
Xuất ra DataFrame đạt chuẩn 100% SignalBarSchema (19 cột).
"""
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from aegis.core.schemas import SignalBarSchema
from aegis.features.kalman.imm_kalman import IMMKalman2D
from aegis.features.regime.efficiency_regime import (
    DEFAULT_ER_WINDOW,
    DEFAULT_SMOOTHING,
    CausalEfficiencyRegime,
)
from aegis.features.regime.ghe import compute_ghe
from aegis.data.bars.tick_rule_ofi import compute_rolling_ofi


class AegisSignalEngine:
    """
    Động cơ phát tín hiệu sơ cấp tổng hợp (Aegis Signal Engine).
    Tích hợp:
    - Causal HMM 2 Trạng Thái (Forward-only)
    - IMM Kalman 2D Filter với sanitize_covariance_matrix
    - Generalized Hurst Exponent (GHE)
    - Average True Range (ATR 14)
    - Tick-Rule Order Flow Imbalance (OFI)
    - Cổng khóa khởi động Insufficient History Guard
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        warmup_window: int = 20,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.symbol = str(symbol)
        self.warmup_window = max(int(warmup_window), 5)
        self.config = config or {}

        # [FIX F6] Phân loại chế độ bằng Efficiency Ratio thay cho HMM tham số bịa.
        # HMM cũ hardcode means/stds và KHÔNG BAO GIỜ được fit (fit_hmm_2state_loglik
        # không tồn tại), nên p_trend thực chất chỉ là bộ dò "biến động cao" cố định,
        # không thích ứng với tài sản hay khung thời gian. Efficiency Ratio không có
        # tham số nào phải fit và tự chuẩn hoá theo mọi thang giá.
        self.regime = CausalEfficiencyRegime(
            window=int(self.config.get("er_window", DEFAULT_ER_WINDOW)),
            smoothing=float(self.config.get("er_smoothing", DEFAULT_SMOOTHING)),
        )

        # Khởi tạo IMM Kalman 2D
        self.imm = IMMKalman2D()

    def process_ohlcv_to_signal_bars(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Nạp DataFrame nến OHLCV và tính toán toàn bộ 19 trường theo SignalBarSchema.
        """
        df = ohlcv_df.copy()
        n = len(df)
        if n == 0:
            raise ValueError("ohlcv_df không được rỗng!")

        # 1. Các trường cơ sở
        bar_idx = np.arange(n, dtype=np.int64)
        symbol_col = np.full(n, self.symbol, dtype=object)

        if "timestamp_ms" in df.columns:
            ts_ms = df["timestamp_ms"].astype(np.int64).values
        else:
            ts_ms = np.arange(n, dtype=np.int64) * 60000

        opens = df["open"].astype(np.float64).values
        highs = df["high"].astype(np.float64).values
        lows = df["low"].astype(np.float64).values
        closes = df["close"].astype(np.float64).values
        volumes = df["volume"].astype(np.float64).values if "volume" in df.columns else np.ones(n, dtype=np.float64)
        tick_counts = df["tick_count"].astype(np.int64).values if "tick_count" in df.columns else np.full(n, 100, dtype=np.int64)

        # 2. OFI
        if "ofi" in df.columns:
            ofi_vals = np.clip(df["ofi"].astype(np.float64).values, -1.0, 1.0)
        else:
            ofi_vals = compute_rolling_ofi(closes, volumes, window=min(14, n))
            # Điền 0.0 cho các giá trị NaN ban đầu
            ofi_vals = np.nan_to_num(ofi_vals, nan=0.0)

        # 3. Cờ độc tính và sự kiện đuôi
        is_toxic_flag = df["is_toxic_flag"].astype(bool).values if "is_toxic_flag" in df.columns else np.zeros(n, dtype=bool)
        is_tail_event = df["is_tail_event"].astype(bool).values if "is_tail_event" in df.columns else np.zeros(n, dtype=bool)

        # 4. Tính True Range và ATR 14
        tr = np.empty(n, dtype=np.float64)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        
        atr_14 = np.empty(n, dtype=np.float64)
        atr_window = min(14, n)
        cumsum_tr = np.cumsum(tr)
        atr_14[:atr_window] = cumsum_tr[:atr_window] / np.arange(1, atr_window + 1)
        for i in range(atr_window, n):
            atr_14[i] = (cumsum_tr[i] - cumsum_tr[i - atr_window]) / atr_window

        # 5. [FIX F6] Posterior chế độ nhân quả từ Efficiency Ratio
        p_trend = np.empty(n, dtype=np.float64)
        p_chop = np.empty(n, dtype=np.float64)
        self.regime.reset()

        for i in range(n):
            pt, pc = self.regime.step(float(closes[i]))
            p_trend[i] = pt
            p_chop[i] = pc

        # 6. IMM Kalman 2D Filter
        trend_score = np.empty(n, dtype=np.float64)
        self.imm.reset(float(closes[0]))

        for i in range(n):
            _, _, ts = self.imm.step(
                observation=float(closes[i]),
                p_trend=float(p_trend[i]),
                p_chop=float(p_chop[i]),
                atr=float(atr_14[i]),
            )
            trend_score[i] = ts

        # 7. Hurst GHE & FFD
        hurst_value = np.full(n, 0.5, dtype=np.float64)
        ghe_window = min(50, n)
        for i in range(ghe_window, n):
            sub_prices = closes[i - ghe_window : i + 1]
            try:
                hurst_value[i] = float(compute_ghe(sub_prices, max_lag=min(10, len(sub_prices) // 3)))
            except Exception:
                hurst_value[i] = 0.5
        hurst_value = np.clip(hurst_value, 0.0, 1.0)

        d_star_used = np.full(n, 0.35, dtype=np.float64)

        # 8. Insufficient History Guard (Behavioral Contract)
        insufficient_history = np.zeros(n, dtype=bool)
        warmup = min(self.warmup_window, n)
        insufficient_history[:warmup] = True

        # Ép NaN cho các cột tín hiệu khi insufficient_history = True
        trend_score[:warmup] = np.nan
        p_trend[:warmup] = np.nan
        p_chop[:warmup] = np.nan
        atr_14[:warmup] = np.nan
        hurst_value[:warmup] = np.nan
        d_star_used[:warmup] = np.nan

        # 9. Đóng gói DataFrame theo đúng chuẩn SignalBarSchema
        result_df = pd.DataFrame({
            "bar_idx": bar_idx,
            "symbol": symbol_col,
            "timestamp_ms": ts_ms,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "ofi": ofi_vals,
            "tick_count": tick_counts,
            "is_toxic_flag": is_toxic_flag,
            "is_tail_event": is_tail_event,
            "insufficient_history": insufficient_history,
            "trend_score": trend_score,
            "p_trend": p_trend,
            "p_chop": p_chop,
            "atr_14": atr_14,
            "hurst_value": hurst_value,
            "d_star_used": d_star_used,
        })

        return SignalBarSchema.validate(result_df)


def build_signal_bars(
    ohlcv_df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    warmup_window: int = 20,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Hàm tiện ích cấp module để sinh SignalBarSchema DataFrame nhanh chóng."""
    engine = AegisSignalEngine(symbol=symbol, warmup_window=warmup_window, config=config)
    return engine.process_ohlcv_to_signal_bars(ohlcv_df)
