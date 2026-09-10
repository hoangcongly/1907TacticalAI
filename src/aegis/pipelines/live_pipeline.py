"""
live_pipeline.py — Institutional Causal Streaming Live / Paper Trading Engine (Task 8).

Điều phối luồng streaming từng nến (causal event-driven):
1. Quản lý trạng thái tài khoản Isolated Margin (AccountStateTracker).
2. Trailing Exit & Stop-Loss động theo từng nến.
3. Multi-Tier Circuit Breaker (5% -> giảm 50%, 10% -> đóng băng 24h, 15% -> dừng khẩn cấp).
4. Sizing động theo Half-Kelly + Volatility-Targeting.
5. Ghi log thử nghiệm & nhật ký giao dịch thread-safe qua ExperimentTracker.
"""

import os
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

from aegis.core.config_loader import load_canonical_config
from aegis.core.experiment_tracker import ExperimentTracker
from aegis.core.trial_classes import TrialClass
from aegis.execution.position_sizer import (
    AccountStateTracker,
    compute_position_size,
    round_notional_down,
)
from aegis.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerTier,
    CircuitBreakerState,
)
from aegis.labeling.trailing_exit import compute_sl_initial
from aegis.meta_labeling.sizing.kelly_empirical import compute_bi_directional_kelly_v14_unified
from aegis.meta_labeling.sizing.liquidation_layer import (
    compute_liquidation_price,
    get_maintenance_margin_rate,
    validate_leverage_against_sl,
)
from aegis.governance.funding_accrual import (
    DEFAULT_FUNDING_RATE,
    compute_funding_accrued_usd,
)
from aegis.execution.pnl import compute_realized_pnl
from aegis.features.feature_engine import MissingFeatureError, OnlineFeatureEngine


@dataclass
class ActivePosition:
    """Đại diện cho vị thế đang mở trong phiên live/paper."""
    symbol: str
    entry_idx: int
    entry_timestamp_ms: int
    entry_price: float
    side: int  # +1: Long, -1: Short
    mode: str  # 'follow' hoặc 'fade'
    sl_initial: float
    sl_current: float
    p_i: float
    p_chop_i: float
    size_notional: float
    high_watermark: float
    bars_held: int = 0
    # --- Trường riêng của PERPETUAL FUTURES ---
    leverage: float = 1.0          # Đòn bẩy thực tế đã dùng cho vị thế
    liquidation_price: float = 0.0 # Giá bị sàn cưỡng chế đóng lệnh
    funding_accrued_usd: float = 0.0  # Funding cộng dồn (>0 = ta trả)
    last_funding_ts_ms: int = 0    # Mốc đã tính funding tới


class AegisLivePipeline:
    """
    Hệ thống vận hành trực tiếp (Live / Paper Trading Engine).
    """

    def __init__(
        self,
        model: Any,
        selected_features: List[str],
        kelly_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        initial_capital: float = 10000.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        base_config = load_canonical_config()
        if config:
            base_config.update(config)
        self.config = base_config

        self.model = model
        self.selected_features = selected_features
        self.kelly_dict = kelly_dict
        self.metadata = metadata

        # Chuẩn bị ma trận Kelly Follow/Fade
        self.kelly_follow_table = np.array(kelly_dict["kelly_follow"]["grid"], dtype=np.float64)
        self.follow_p_edges = np.array(kelly_dict["kelly_follow"]["p_edges"], dtype=np.float64)
        self.follow_chop_edges = [
            np.array(e, dtype=np.float64) for e in kelly_dict["kelly_follow"]["chop_edges"]
        ]

        self.kelly_fade_table = np.array(kelly_dict["kelly_fade"]["grid"], dtype=np.float64)
        self.fade_p_edges = np.array(kelly_dict["kelly_fade"]["p_edges"], dtype=np.float64)
        self.fade_chop_edges = [
            np.array(e, dtype=np.float64) for e in kelly_dict["kelly_fade"]["chop_edges"]
        ]

        # Khởi tạo Quản lý vốn & Ngắt mạch rủi ro
        self.account_tracker = AccountStateTracker(
            wallet_balance=float(initial_capital),
            unrealized_pnl=0.0,
            used_initial_margin=0.0,
        )
        self.circuit_breaker = CircuitBreaker(initial_equity=float(initial_capital))
        self.experiment_tracker = ExperimentTracker()

        self.current_position: Optional[ActivePosition] = None
        self.closed_trades: List[Dict[str, Any]] = []

        # [FIX F3] Bộ tính đặc trưng streaming — DÙNG CHUNG code với research.
        # Trước đây live đọc `bar.get(col, 0.0)` nên 4/5 đặc trưng của model bằng 0.0
        # suốt phiên mà không có cảnh báo nào.
        self.feature_engine = OnlineFeatureEngine()

        # CUSUM streaming accumulators
        self.s_plus: float = 0.0
        self.s_minus: float = 0.0
        self.last_price: Optional[float] = None
        self.last_event_idx: int = -1
        self.last_event_price: float = 0.0

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str = "artifacts/",
        initial_capital: float = 10000.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> "AegisLivePipeline":
        """Khởi tạo Live Pipeline từ thư mục artifacts."""
        model_path = os.path.join(artifacts_dir, "model.pkl")
        features_path = os.path.join(artifacts_dir, "selected_features.json")
        kelly_path = os.path.join(artifacts_dir, "kelly_tables.json")
        meta_path = os.path.join(artifacts_dir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model artifact: {model_path}")

        model = joblib.load(model_path)
        with open(features_path, "r") as f:
            features = json.load(f)
        with open(kelly_path, "r") as f:
            kelly_dict = json.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        return cls(
            model=model,
            selected_features=features,
            kelly_dict=kelly_dict,
            metadata=metadata,
            initial_capital=initial_capital,
            config=config,
        )

    def _exit_slippage_pct(self) -> float:
        """
        [FIX F12] Trượt giá ước tính khi thoát lệnh bằng MARKET/STOP (tỷ lệ trên giá).

        Lệnh dừng luôn khớp ĐÚNG vào nhịp giá đang chạy ngược vị thế, nên chi phí
        thực tế = nửa spread + phần trượt do quét sổ lệnh. Lấy tham số từ Canonical
        Registry (spread_pct, slippage_penalty_factor).
        """
        spread = float(self.config.get("spread_pct", 0.0002))
        penalty = float(self.config.get("slippage_penalty_factor", 0.1))
        return max(0.0, 0.5 * spread + penalty * spread)

    def _lookup_kelly(self, p_i: float, p_chop_i: float) -> Dict[str, Any]:
        """
        [FIX F13] Tra bảng Kelly ĐÚNG theo chế độ giao dịch.

        Bản cũ luôn truyền kelly_follow_table kể cả khi chế độ được phân loại là
        'fade' — tức lấy f* của Follow áp cho lệnh Fade. Ở đây phân loại chế độ
        trước, rồi tra đúng bảng tương ứng.
        """
        from aegis.meta_labeling.sizing.trade_mode import classify_trade_mode

        fade_enabled = bool(self.config.get("fade_enabled", False))
        gate = float(self.config.get("fade_regime_gate_threshold", 0.60))

        mode = classify_trade_mode(
            p_i=float(p_i),
            p_chop_i=float(p_chop_i),
            fade_enabled=fade_enabled,
            fade_regime_gate_threshold=gate,
        )
        if mode == "none":
            return {"f_target": 0.0, "mode": "none"}

        if mode == "fade":
            table, p_edges, chop_edges = (
                self.kelly_fade_table, self.fade_p_edges, self.fade_chop_edges,
            )
        else:
            table, p_edges, chop_edges = (
                self.kelly_follow_table, self.follow_p_edges, self.follow_chop_edges,
            )

        return compute_bi_directional_kelly_v14_unified(
            p_i=float(p_i),
            p_chop_i=float(p_chop_i),
            kelly_table=table,
            p_edges=p_edges,
            chop_edges_list=chop_edges,
            fade_enabled=fade_enabled,
            fade_regime_gate_threshold=gate,
        )

    def _check_cusum_trigger(
        self,
        bar_idx: int,
        price: float,
        atr: float,
        cooldown_bars: int = 5,
        spatial_delta_atr: float = 0.5,
    ) -> int:
        """
        Cập nhật tích lũy CUSUM trực tiếp từng nến.
        Trả về: +1 (Breakout Up), -1 (Breakdown Down), hoặc 0 (Không kích hoạt).
        """
        if self.last_price is None:
            self.last_price = price
            return 0

        r_t = math.log(max(price, 1e-6) / max(self.last_price, 1e-6))
        self.last_price = price

        self.s_plus = max(0.0, self.s_plus + r_t)
        self.s_minus = min(0.0, self.s_minus + r_t)

        # Ngưỡng động
        h_t = max(2.0 * (atr / max(price, 1e-6)), 1e-4)

        triggered = 0
        if self.s_plus > h_t:
            triggered = 1
            self.s_plus = 0.0
            self.s_minus = 0.0
        elif self.s_minus < -h_t:
            triggered = -1
            self.s_plus = 0.0
            self.s_minus = 0.0

        if triggered != 0:
            # Kiểm tra Spatial-Temporal Cooldown
            temporal_ok = (self.last_event_idx == -1) or ((bar_idx - self.last_event_idx) >= cooldown_bars)
            spatial_ok = (self.last_event_idx == -1) or (
                abs(price - self.last_event_price) >= spatial_delta_atr * atr
            )
            if temporal_ok and spatial_ok:
                self.last_event_idx = bar_idx
                self.last_event_price = price
                return triggered

        return 0

    def on_bar(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý từng nến mới đẩy vào theo thời gian thực (Causal Streaming).
        """
        bar_idx = int(bar.get("bar_idx", 0))
        timestamp_ms = int(bar.get("timestamp_ms", int(time.time() * 1000)))
        symbol = str(bar.get("symbol", "BTCUSDT"))
        open_p = float(bar.get("open", 100.0))
        high_p = float(bar.get("high", open_p))
        low_p = float(bar.get("low", open_p))
        close_p = float(bar.get("close", open_p))
        atr = float(bar.get("atr_14", 1.0)) if not np.isnan(bar.get("atr_14", 1.0)) else 1.0
        p_trend = float(bar.get("p_trend", 0.5)) if not np.isnan(bar.get("p_trend", 0.5)) else 0.5
        p_chop = float(bar.get("p_chop", 0.5)) if not np.isnan(bar.get("p_chop", 0.5)) else 0.5
        trend_score = float(bar.get("trend_score", 0.0)) if not np.isnan(bar.get("trend_score", 0.0)) else 0.0
        is_toxic = bool(bar.get("is_toxic_flag", False))
        insufficient_history = bool(bar.get("insufficient_history", False))

        c_trade = float(self.config.get("c_trade", 0.0004))
        t_max_live = int(self.config.get("t_max_live", 20))
        leverage = float(self.config.get("max_safe_leverage", 3.0))
        # [FIX F14] Hệ số ATR cho trailing, lấy từ Canonical Registry (m_sl_follow = 2.0).
        m_trail = float(self.config.get("m_trail", self.config.get("m_sl", 2.0)))

        # [FIX F3] Nạp nến vào bộ đặc trưng ở MỌI nến để cửa sổ lịch sử luôn liên tục,
        # kể cả những nến ta không giao dịch. Bỏ sót nến nào là lệch parity với research.
        try:
            self.feature_engine.update(bar)
        except MissingFeatureError as exc:
            # Feed thiếu cột thô = sự cố toàn vẹn dữ liệu. Dừng giao dịch nhưng
            # không làm sập tiến trình — nêu rõ lý do để vận hành viên thấy ngay.
            return {
                "action": "HOLD",
                "reason": "BAR_DATA_INCOMPLETE",
                "detail": str(exc),
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        # ====================================================================
        # BƯỚC 0 [FIX F7]: MARK-TO-MARKET VỊ THẾ MỞ TRƯỚC MỌI QUYẾT ĐỊNH RỦI RO
        # ====================================================================
        # Trước đây unrealized_pnl không bao giờ được cập nhật và Circuit Breaker
        # chỉ nhìn wallet_balance (tiền mặt đã chốt). Với đòn bẩy, một vị thế đang
        # mở có thể bay sạch tài khoản mà Circuit Breaker vẫn báo NORMAL.
        if self.current_position is not None:
            _pos = self.current_position

            # [FIX FUTURES-1] Cộng dồn FUNDING của hợp đồng vĩnh cửu.
            # Mỗi chu kỳ 8h (00:00/08:00/16:00 UTC) vị thế phải trả (hoặc nhận) funding.
            # Bỏ qua khoản này là bỏ qua chi phí ĐẶC TRƯNG của perpetual futures —
            # với lệnh giữ 120 nến, nó ăn mòn trực tiếp vào ký quỹ.
            _new_funding = compute_funding_accrued_usd(
                size_notional=_pos.size_notional,
                side=_pos.side,
                entry_ts_ms=_pos.last_funding_ts_ms,
                exit_ts_ms=timestamp_ms,
                funding_rate=float(bar.get("funding_rate", self.config.get("funding_rate", DEFAULT_FUNDING_RATE))),
                interval_hours=int(self.config.get("funding_interval_hours", 8)),
            )
            _pos.funding_accrued_usd += _new_funding
            _pos.last_funding_ts_ms = timestamp_ms

            _gross_ret_mtm = (
                (close_p - _pos.entry_price) / _pos.entry_price
                if _pos.side == 1
                else (_pos.entry_price - close_p) / _pos.entry_price
            )
            # uPnL của futures = PnL giá - funding đã cộng dồn (funding rời ký quỹ ngay).
            self.account_tracker.unrealized_pnl = float(
                _gross_ret_mtm * _pos.size_notional - _pos.funding_accrued_usd
            )
        else:
            self.account_tracker.unrealized_pnl = 0.0

        # Circuit Breaker nay chấm trên equity mark-to-market (margin_balance).
        cb_state = self.circuit_breaker.update_equity(
            self.account_tracker.margin_balance, current_time_ms=timestamp_ms
        )
        force_flatten = (
            self.current_position is not None
            and (cb_state.is_frozen or cb_state.tier.value >= CircuitBreakerTier.TIER_2_FLATTEN.value)
        )

        # ====================================================================
        # BƯỚC 1: QUẢN LÝ VỊ THẾ HIỆN TẠI (TRAILING EXIT & STOP-LOSS)
        # ====================================================================
        if self.current_position is not None:
            pos = self.current_position
            pos.bars_held += 1

            exit_triggered = False
            exit_reason = ""
            fill_price_exit = close_p

            # [FIX F8 / FUTURES-2] KIỂM TRA THANH LÝ CƯỠNG CHẾ — ƯU TIÊN TUYỆT ĐỐI.
            # Trên futures, sàn đóng lệnh của ta ở giá thanh lý TRƯỚC khi ta kịp
            # chạm SL nếu SL nằm ngoài vùng an toàn. Bỏ qua nhánh này là mô hình
            # hoá sai kịch bản mất tiền tệ nhất của hợp đồng vĩnh cửu.
            liq = pos.liquidation_price
            liq_hit = liq > 0.0 and (
                (pos.side == 1 and low_p <= liq) or (pos.side == -1 and high_p >= liq)
            )
            if liq_hit:
                exit_triggered = True
                exit_reason = "LIQUIDATION"
                fill_price_exit = liq

            if not exit_triggered and pos.side == 1:
                # Long: chạm SL nếu low <= sl_current
                if low_p <= pos.sl_current:
                    exit_triggered = True
                    # [FIX F12] Lệnh dừng khớp bằng MARKET ngay trên nhịp giá bất lợi:
                    # phải trừ spread + slippage, không được khớp đúng bằng sl_current.
                    fill_price_exit = min(pos.sl_current, open_p) * (1.0 - self._exit_slippage_pct())
                    fill_price_exit = max(fill_price_exit, low_p)
                    exit_reason = "SL" if pos.sl_current == pos.sl_initial else "TRAIL"
                else:
                    # [FIX F14] Chandelier exit neo vào HIGH-WATERMARK, không neo vào close.
                    # Bản cũ tính high_watermark rồi vứt và trail theo `close - k*ATR`,
                    # lệch khỏi labeling/trailing_exit.py — chính là module đã dựng ra
                    # bảng Kelly. Lệch nhau nghĩa là calibration và inference không khớp.
                    pos.high_watermark = max(pos.high_watermark, high_p)
                    candidate_sl = pos.high_watermark - m_trail * atr
                    if candidate_sl > pos.sl_current:
                        pos.sl_current = candidate_sl
            elif not exit_triggered:
                # Short: chạm SL nếu high >= sl_current
                if high_p >= pos.sl_current:
                    exit_triggered = True
                    # [FIX F12] Đối xứng cho Short: trượt giá đẩy giá khớp lên cao hơn.
                    fill_price_exit = max(pos.sl_current, open_p) * (1.0 + self._exit_slippage_pct())
                    fill_price_exit = min(fill_price_exit, high_p)
                    exit_reason = "SL" if pos.sl_current == pos.sl_initial else "TRAIL"
                else:
                    # [FIX F14] Đối xứng cho Short: high_watermark giữ giá THẤP NHẤT đã chạm.
                    pos.high_watermark = min(pos.high_watermark, low_p)
                    candidate_sl = pos.high_watermark + m_trail * atr
                    if candidate_sl < pos.sl_current:
                        pos.sl_current = candidate_sl

            # Time stop
            if not exit_triggered and pos.bars_held >= t_max_live:
                exit_triggered = True
                exit_reason = "TIME_STOP"
                fill_price_exit = close_p

            # [FIX F7] Circuit Breaker Tier>=2 hoặc đang đóng băng: buộc thoát vị thế.
            if not exit_triggered and force_flatten:
                exit_triggered = True
                exit_reason = "CIRCUIT_BREAKER_FLATTEN"
                slip = self._exit_slippage_pct()
                fill_price_exit = close_p * (1.0 - slip) if pos.side == 1 else close_p * (1.0 + slip)

            if exit_triggered:
                # [FIX F10] Quyết toán qua ĐÚNG MỘT engine PnL dùng chung với backtest
                # (aegis.execution.pnl.compute_realized_pnl). Bản cũ tự viết công thức
                # riêng tại đây: không funding, không nhánh thanh lý, không phân biệt
                # maker/taker — khiến backtest và live tính tiền khác nhau.
                pnl_res = compute_realized_pnl(
                    entry_price=pos.entry_price,
                    exit_price=fill_price_exit,
                    side=pos.side,
                    size_notional=pos.size_notional,
                    leverage=pos.leverage,
                    exit_reason=exit_reason if exit_reason in (
                        "SL", "TRAIL", "REGIME_FLIP", "TIME_STOP",
                        "LIQUIDATION", "BOUNDARY_TRUNCATED",
                    ) else "TIME_STOP",
                    entry_fill_type="taker",
                    exit_fill_type="taker",
                    maker_fee_rate=float(self.config.get("maker_fee_rate", 0.0001)),
                    taker_fee_rate=float(self.config.get("taker_fee_rate", c_trade)),
                    funding_accrued_usd=pos.funding_accrued_usd,
                )
                gross_pnl = pnl_res["gross_pnl"]
                net_pnl = pnl_res["net_pnl"]
                gross_ret = pnl_res["realized_return"]

                # Ký quỹ cô lập: lỗ không bao giờ vượt quá margin đã cọc.
                net_pnl = max(net_pnl, -pos.size_notional / max(pos.leverage, 1.0))

                # Cập nhật số dư tiền mặt
                self.account_tracker.wallet_balance = max(
                    0.0, self.account_tracker.wallet_balance + net_pnl
                )
                self.account_tracker.used_initial_margin = 0.0
                self.account_tracker.unrealized_pnl = 0.0

                trade_record = {
                    "symbol": pos.symbol,
                    "entry_idx": pos.entry_idx,
                    "exit_idx": bar_idx,
                    "side": pos.side,
                    "mode": pos.mode,
                    "entry_price": pos.entry_price,
                    "fill_price_exit": fill_price_exit,
                    "exit_reason": exit_reason,
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "realized_return": float(gross_ret),
                    "leverage": float(pos.leverage),
                    "liquidation_price": float(pos.liquidation_price),
                    "funding_accrued": float(pos.funding_accrued_usd),
                    "fee_paid": float(pnl_res["fee_paid"]),
                    "wallet_balance": float(self.account_tracker.wallet_balance),
                }
                self.closed_trades.append(trade_record)
                self.current_position = None

                return {
                    "action": "CLOSE",
                    "trade_record": trade_record,
                    "wallet_balance": self.account_tracker.wallet_balance,
                }

        # ====================================================================
        # BƯỚC 2: TÌM KIẾM CƠ HỘI MỞ LỆNH MỚI
        # ====================================================================
        # 1. Circuit Breaker đã được chấm trên equity mark-to-market ở BƯỚC 0.
        if cb_state.is_frozen or cb_state.tier == CircuitBreakerTier.TIER_3_KILL:
            return {
                "action": "HOLD",
                "reason": "CIRCUIT_BREAKER_BLOCKED",
                "circuit_breaker": cb_state.tier.name,
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        # 2. Kiểm tra rào cản nến (Microstructure Guards)
        if insufficient_history or is_toxic:
            return {
                "action": "HOLD",
                "reason": "DATA_GUARD_BLOCKED",
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        # 3. Kiểm tra CUSUM Event Trigger
        cusum_trig = self._check_cusum_trigger(
            bar_idx=bar_idx,
            price=close_p,
            atr=atr,
            cooldown_bars=self.config.get("cooldown_bars", 5),
            spatial_delta_atr=self.config.get("spatial_delta_atr", 0.5),
        )

        if cusum_trig == 0:
            return {
                "action": "HOLD",
                "reason": "NO_CUSUM_TRIGGER",
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        # 4. Dự báo Xác suất Thắng P(y=1) từ Meta-Labeling Model
        # [FIX F3] Vector dựng bằng CHÍNH module research dùng. Thiếu đặc trưng ->
        # NÉM LỖI và không giao dịch, thay vì im lặng điền 0.0 rồi cược bằng model mù.
        try:
            X_input = self.feature_engine.build_vector(bar, self.selected_features)
        except MissingFeatureError as exc:
            return {
                "action": "HOLD",
                "reason": "FEATURE_UNAVAILABLE",
                "detail": str(exc),
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        proba = self.model.predict_proba(X_input)
        p_i = float(proba[0, 1])

        # 5. Tra cứu ma trận Kelly 2D & Sizing
        kelly_res = self._lookup_kelly(p_i=p_i, p_chop_i=p_chop)

        f_star = float(kelly_res.get("f_target", 0.0))
        mode = str(kelly_res.get("mode", "follow"))

        if f_star <= 0.0 or mode == "none":
            return {
                "action": "HOLD",
                "reason": "KELLY_ZERO",
                "p_i": p_i,
                "mode": mode,
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        # Sizing với Half-Kelly + Volatility-Targeting
        size_notional = compute_position_size(
            f_star=f_star,
            current_equity=self.account_tracker,
            max_safe_leverage=leverage,
            lambda_kelly=0.5,
            lot_step_size=self.config.get("lot_step_size", 10.0),
        )

        # Áp dụng giảm tỷ trọng nếu Circuit Breaker Tier 1
        size_notional *= cb_state.max_position_multiplier

        if size_notional <= 0.0:
            return {
                "action": "HOLD",
                "reason": "SIZE_ZERO",
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        # Xác định Side: nếu mode == 'fade' thì ngược hướng trend
        side_follow = 1 if trend_score >= 0 else -1
        side = side_follow if mode == "follow" else -side_follow

        # 6. Tính SL ban đầu đối xứng
        sl_initial = compute_sl_initial(
            entry_price=close_p,
            side=side,
            m_sl=self.config.get("m_sl", 2.0),
            sigma=atr / max(close_p, 1e-6),
            c_trade_adj=c_trade,
        )

        # ====================================================================
        # [FIX F8 / FUTURES-3] PRE-FLIGHT CHECK THANH LÝ TRƯỚC KHI VÀO LỆNH
        # ====================================================================
        # Trên perpetual futures, đòn bẩy quyết định khoảng cách tới giá thanh lý.
        # Nếu SL nằm NGOÀI vùng an toàn (quá sát giá thanh lý), sàn sẽ cưỡng chế
        # đóng lệnh trước khi SL kịp chạy — biến một khoản lỗ có kiểm soát thành
        # mất trắng toàn bộ ký quỹ. Bắt buộc chặn trước khi đặt lệnh.
        mmr = get_maintenance_margin_rate(
            size_notional,
            margin_tier_table=self.config.get("margin_tier_table"),
        )
        # Funding dự kiến phải trả trong thời gian gồng lệnh cũng đẩy giá thanh lý
        # lại gần entry hơn, nên phải tính vào ngay từ đầu.
        expected_funding_pct = abs(
            float(self.config.get("funding_rate", DEFAULT_FUNDING_RATE))
        ) * max(1.0, float(t_max_live) / 8.0)

        try:
            liq_check = validate_leverage_against_sl(
                entry_price=close_p,
                side=side,
                sl_initial=sl_initial,
                leverage=leverage,
                maintenance_margin_rate=mmr,
                safety_buffer_pct=float(self.config.get("safety_buffer_pct", 0.15)),
                fee_rate=c_trade,
                liquidation_fee_rate=float(self.config.get("liquidation_fee_rate", 0.005)),
                funding_accrued_pct=expected_funding_pct,
            )
        except ValueError as exc:
            return {
                "action": "HOLD",
                "reason": f"LIQUIDATION_GUARD_ERROR: {exc}",
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        if not liq_check["is_safe"]:
            return {
                "action": "HOLD",
                "reason": "LIQUIDATION_UNSAFE",
                "detail": liq_check["reason"],
                "liquidation_price": liq_check["liq_price"],
                "sl_initial": sl_initial,
                "wallet_balance": self.account_tracker.wallet_balance,
            }

        liquidation_price = float(liq_check["liq_price"])

        # Cập nhật số dư cọc ký quỹ
        margin_required = size_notional / max(leverage, 1.0)
        if margin_required > self.account_tracker.available_margin + 1e-9:
            return {
                "action": "HOLD",
                "reason": "INSUFFICIENT_MARGIN",
                "margin_required": float(margin_required),
                "available_margin": float(self.account_tracker.available_margin),
                "wallet_balance": self.account_tracker.wallet_balance,
            }
        self.account_tracker.used_initial_margin = margin_required

        self.current_position = ActivePosition(
            symbol=symbol,
            entry_idx=bar_idx,
            entry_timestamp_ms=timestamp_ms,
            entry_price=close_p,
            side=side,
            mode=mode,
            sl_initial=sl_initial,
            sl_current=sl_initial,
            p_i=p_i,
            p_chop_i=p_chop,
            size_notional=size_notional,
            high_watermark=close_p,
            bars_held=0,
            leverage=leverage,
            liquidation_price=liquidation_price,
            funding_accrued_usd=0.0,
            last_funding_ts_ms=timestamp_ms,
        )

        return {
            "action": "OPEN",
            "leverage": leverage,
            "liquidation_price": liquidation_price,
            "side": side,
            "mode": mode,
            "size_notional": size_notional,
            "entry_price": close_p,
            "sl_initial": sl_initial,
            "p_i": p_i,
            "wallet_balance": self.account_tracker.wallet_balance,
        }
