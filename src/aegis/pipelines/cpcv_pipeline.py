"""
Điều phối 15 folds CPCV theo đúng thứ tự 5 bước v11.8 (Mục 4.0).
Bảo chứng Zero-Leakage & Tra cứu chỉ số tuyệt đối resolve_absolute_exit_idx.
"""
from typing import Dict, Any, List, Tuple, Optional
import warnings
import numpy as np
import pandas as pd

from aegis.core.schemas import (
    TradeRecordSchema,
    compute_dataset_manifest_hash,
)
from aegis.validation.cpcv import CombinatorialPurgedKFold
from aegis.labeling.cusum_events import (
    compute_dynamic_cusum_thresholds,
    filter_cusum_events_dynamic,
)
from aegis.labeling.trailing_exit import (
    run_trailing_exit_for_oos_event,
    finalize_trade_record,
    resolve_regime_exit_threshold,
)
from aegis.meta_labeling.sizing.kelly_empirical import (
    trade_records_to_kelly_table_inputs,
    build_empirical_kelly_table_v2,
)
from aegis.validation.dsr import compute_deflated_sharpe_ratio
from aegis.features.feature_engine import batch_features
from aegis.features.feature_spec import CANDIDATE_FEATURES
from aegis.labeling.triple_barrier import generate_meta_labels_triple_barrier
from aegis.labeling.sample_weights import (
    compute_num_concurrent_events,
    compute_sample_weights,
)
from aegis.meta_labeling.weighted_bootstrap_forest import WeightedBootstrapForestClassifier
from aegis.governance.funding_accrual import (
    DEFAULT_FUNDING_RATE,
    compute_funding_accrued_usd,
)


def _safe_skew(x: np.ndarray) -> float:
    """Độ lệch (skewness) mẫu, trả 0.0 khi mẫu suy biến — DSR cần tham số này."""
    if len(x) < 3:
        return 0.0
    sd = float(np.std(x, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(((x - np.mean(x)) / sd) ** 3))


def _safe_kurtosis(x: np.ndarray) -> float:
    """Độ nhọn (kurtosis) mẫu KHÔNG trừ 3 — DSR kỳ vọng chuẩn Gaussian = 3.0."""
    if len(x) < 4:
        return 3.0
    sd = float(np.std(x, ddof=1))
    if sd <= 1e-12:
        return 3.0
    return float(np.mean(((x - np.mean(x)) / sd) ** 4))


def filter_boundary_truncated_for_kelly_table(
    records: List[Dict[str, Any]],
    truncation_warning_threshold: float = 0.15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    [BƯỚC 2.5 — QUY TRÌNH 5 BƯỚC v11.8]:
    Phân loại danh sách trade_records thành 2 tập:
    1. clean_records: boundary_truncated == False (dùng để giải bảng Kelly).
    2. diagnostic_records: boundary_truncated == True (bị cắt biên fold CPCV).
    
    Cảnh báo nếu tỷ lệ cắt biên vượt quá 15%.
    """
    clean_records = []
    diagnostic_records = []

    for r in records:
        if r.get("boundary_truncated", False):
            diagnostic_records.append(r)
        else:
            clean_records.append(r)

    n_total = len(records)
    if n_total > 0:
        trunc_ratio = len(diagnostic_records) / n_total
        if trunc_ratio > truncation_warning_threshold:
            warnings.warn(
                f"[CPCV WARNING] Tỷ lệ lệnh bị cắt biên fold CPCV là {trunc_ratio:.1%} "
                f"(vượt ngưỡng an toàn {truncation_warning_threshold:.1%}). "
                f"Hãy cân nhắc giảm t_max_live hoặc tăng kích thước sub-block CPCV!",
                UserWarning,
            )

    return clean_records, diagnostic_records


class CPCVPipeline:
    """
    Điều phối Combinatorial Purged Cross-Validation (CPCV) chuẩn 5 bước v11.8:
    1. Chia splits CPCV với embargo_bars.
    2. Refit trên từng fold và tạo bản ghi giao dịch OOS hoàn chỉnh.
    3. Tách lọc lệnh cắt biên (boundary_truncated).
    4. Xây dựng bảng Kelly 2D (Follow / Fade).
    5. Đánh giá kiểm định chéo OOS: Sharpe, DSR, PBO.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_bars: int = 24,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.embargo_bars = int(embargo_bars)
        self.config = config or {}

    def run(self, signal_bars: pd.DataFrame, num_bins: int = 10) -> Dict[str, Any]:
        """
        Chạy toàn bộ quy trình CPCV trên DataFrame nến SignalBarSchema.
        """
        n_bars = len(signal_bars)
        if n_bars < self.n_groups * 5:
            raise ValueError(f"Số lượng bars ({n_bars}) quá nhỏ để chia {self.n_groups} nhóm CPCV!")

        # Tem niêm phong SHA-256 của dataset nến
        manifest_hash = compute_dataset_manifest_hash(signal_bars, self.config)
        symbol = str(signal_bars["symbol"].iloc[0]) if "symbol" in signal_bars.columns else "BTCUSDT"

        closes = signal_bars["close"].astype(np.float64).values
        highs = signal_bars["high"].astype(np.float64).values
        lows = signal_bars["low"].astype(np.float64).values
        timestamps_ms = signal_bars["timestamp_ms"].astype(np.int64).values
        
        p_trend = signal_bars["p_trend"].values
        p_chop = signal_bars["p_chop"].values
        atr_14 = signal_bars["atr_14"].values
        trend_score = signal_bars["trend_score"].values
        is_toxic = signal_bars["is_toxic_flag"].values
        insufficient_history = signal_bars["insufficient_history"].values

        # 1. Phát hiện sự kiện CUSUM (Bỏ qua nến insufficient_history)
        valid_start = int(np.where(~insufficient_history)[0][0]) if np.any(~insufficient_history) else 0
        safe_atr = np.nan_to_num(atr_14, nan=1.0)
        safe_atr = np.maximum(safe_atr, 1e-6)

        dynamic_thresholds = compute_dynamic_cusum_thresholds(
            prices=closes,
            atr_series=safe_atr,
            base_multiplier=self.config.get("cusum_multiplier", 2.0),
            anchor_span=50,
            use_ewma_anchor=True,
            reset_mask=insufficient_history,
        )

        safe_trend_score = np.nan_to_num(trend_score, nan=0.0)
        safe_p_trend = np.nan_to_num(p_trend, nan=0.5)
        safe_p_chop = np.nan_to_num(p_chop, nan=0.5)

        cusum_records = filter_cusum_events_dynamic(
            prices=closes,
            dynamic_thresholds=dynamic_thresholds,
            timestamps_ms=timestamps_ms,
            atr_series=safe_atr,
            cooldown_bars=self.config.get("cooldown_bars", 5),
            spatial_delta_atr=self.config.get("spatial_delta_atr", 0.5),
            trend_scores=safe_trend_score,
            p_trend=safe_p_trend,
            p_chop=safe_p_chop,
        )

        event_indices = np.array([e["bar_idx"] for e in cusum_records if e["bar_idx"] >= valid_start], dtype=np.int64)
        if len(event_indices) < self.n_groups * 2:
            # Fallback đều đặn nếu CUSUM kích hoạt quá ít sự kiện
            step_size = max(1, (n_bars - valid_start) // max(self.n_groups * 3, 10))
            event_indices = np.arange(valid_start, n_bars - 5, step=step_size, dtype=np.int64)

        n_events = len(event_indices)
        t0 = event_indices

        # [FIX F16] Chân trời purge PHẢI phủ toàn bộ thời gian nắm giữ tối đa.
        # Bản cũ hardcode `t0 + 10` trong khi lệnh follow chạy tới t_max_live_follow
        # (canonical = 120 nến) -> purge chỉ phủ 10/120, rò rỉ thông tin từ test sang
        # train, đồng thời cắt biên hàng loạt (truncation 17-37%).
        self.t_max_live_follow = int(self.config.get("t_max_live_follow", 120))
        self.t_max_live_fade = int(self.config.get("t_max_live_fade", 40))
        purge_horizon = max(self.t_max_live_follow, self.t_max_live_fade)
        t1 = np.clip(t0 + purge_horizon, 0, n_bars - 1)

        # ====================================================================
        # [FIX F5] GÁN NHÃN + ĐẶC TRƯNG TẠI CÁC SỰ KIỆN
        # ====================================================================
        # Bản cũ không hề gán nhãn và không hề huấn luyện: vòng lặp fold bỏ trống
        # `train_idx`, nên "OOS" chỉ là một luật trend-following cố định chạy trên
        # TOÀN BỘ dữ liệu. Muốn có số OOS thật thì CPCV phải tự fit được model.
        # [EDGE-HUNT] Chiều lệnh gốc: +1 = đu theo xu hướng (follow),
        # -1 = đánh ngược / hồi quy về trung bình (mean-reversion).
        # BTC 1h chỉ có xu hướng ~6% thời gian, nên giả thuyết đánh ngược đáng thử.
        primary_direction = int(self.config.get("primary_direction", 1))
        if primary_direction not in (1, -1):
            raise ValueError(f"primary_direction phải là +1 hoặc -1, nhận {primary_direction}")
        self.primary_direction = primary_direction

        sides_primary = (
            np.where(safe_trend_score[t0] >= 0.0, 1, -1) * primary_direction
        ).astype(np.int64)

        meta_labels_df = generate_meta_labels_triple_barrier(
            events_idx=t0,
            sides=sides_primary,
            closes=closes,
            highs=highs,
            lows=lows,
            p_trend=safe_p_trend,
            p_chop=safe_p_chop,
            sigmas=safe_atr / np.maximum(closes, 1e-6),
            is_toxic=is_toxic,
            t_window=int(self.config.get("t_window", 20)),
            c_trade=float(self.config.get("c_trade", 0.0004)),
        )

        feature_bars = batch_features(signal_bars)
        feature_cols = [c for c in CANDIDATE_FEATURES if c in feature_bars.columns]

        labelled_t0 = meta_labels_df["t0"].to_numpy().astype(np.int64)
        y_all = meta_labels_df["label"].to_numpy().astype(np.int64)
        X_all = feature_bars[feature_cols].iloc[labelled_t0].fillna(0.0).to_numpy(dtype=np.float64)

        # Trọng số mẫu theo độ trùng lặp thời gian (AFML) — dùng khi fit từng fold.
        try:
            t1_series = pd.Series(data=meta_labels_df["t1"].values, index=labelled_t0)
            c_t = compute_num_concurrent_events(
                t1=t1_series, bar_index=pd.Series(np.arange(n_bars))
            )
            weights_all = compute_sample_weights(
                t1=t1_series,
                c_t=c_t,
                returns=pd.Series(meta_labels_df["realized_return"].values, index=labelled_t0),
            ).to_numpy(dtype=np.float64)
        except Exception:
            weights_all = np.ones(len(y_all), dtype=np.float64)

        # Ánh xạ bar_idx -> vị trí trong tập sự kiện đã gán nhãn.
        pos_of_event = {int(b): i for i, b in enumerate(labelled_t0)}

        effective_n_groups = min(self.n_groups, max(2, n_events // 2))
        effective_n_test = min(self.n_test_groups, max(1, effective_n_groups // 2))

        # 2. Khởi tạo CPCV
        cpcv = CombinatorialPurgedKFold(
            n_groups=effective_n_groups,
            n_test_groups=effective_n_test,
            embargo_bars=self.embargo_bars,
        )

        all_trade_records: List[Dict[str, Any]] = []

        # Tách splits trên tập sự kiện
        splits = cpcv.split(X=np.zeros((n_events, 1)), pred_times=t0, eval_times=t1)

        # [FIX F18] Lịch funding: ưu tiên dữ liệu THẬT từ sàn ({ts_ms: rate}),
        # không có thì rơi về hằng số danh nghĩa 0.01%/8h.
        funding_schedule = self.config.get("funding_rates")
        if not isinstance(funding_schedule, dict) or not funding_schedule:
            funding_schedule = float(self.config.get("funding_rate", DEFAULT_FUNDING_RATE))

        self.n_fits = 0  # đếm số lần fit thật — test refit-per-fold kiểm tra biến này
        self.n_folds_starved = 0  # fold không đủ dữ liệu train sau purge/embargo

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_id = f"fold_{fold_idx}"
            if len(test_idx) == 0:
                continue

            test_events = t0[test_idx]
            max_test_bar = int(np.max(t1[test_idx]))

            # ================================================================
            # [FIX F5] HUẤN LUYỆN THẬT TRÊN TRAIN FOLD, DỰ BÁO TRÊN TEST FOLD
            # ================================================================
            # `train_idx` giờ mới thực sự được dùng. Xác suất thu được là OOS thật:
            # model chưa từng nhìn thấy các sự kiện trong test fold.
            train_pos = [pos_of_event[int(b)] for b in t0[train_idx] if int(b) in pos_of_event]
            fold_model = None
            if len(train_pos) >= 10 and len(np.unique(y_all[train_pos])) >= 2:
                try:
                    fold_model = WeightedBootstrapForestClassifier(
                        n_estimators=int(self.config.get("n_estimators", 50)),
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=42,
                    )
                    fold_model.fit(
                        X_all[train_pos],
                        y_all[train_pos],
                        sample_weight=weights_all[train_pos],
                    )
                    self.n_fits += 1
                except Exception:
                    fold_model = None
            if fold_model is None:
                self.n_folds_starved += 1

            # [FIX F6b] Ngưỡng Regime-Flip lấy theo PHÂN VỊ của p_trend TRONG TRAIN FOLD.
            # Chỉ dùng dữ liệu train -> không rò rỉ sang test.
            train_bars = t0[train_idx]
            regime_q = float(self.config.get("regime_exit_quantile", 0.10))
            thr_follow = resolve_regime_exit_threshold(
                safe_p_trend[train_bars], quantile=regime_q, mode="follow"
            )
            thr_fade = resolve_regime_exit_threshold(
                safe_p_trend[train_bars], quantile=regime_q, mode="fade"
            )

            for ev_idx in test_events:
                if ev_idx >= n_bars - 2:
                    continue

                entry_p = float(closes[ev_idx])
                pt = float(p_trend[ev_idx]) if not np.isnan(p_trend[ev_idx]) else 0.5
                pc = float(p_chop[ev_idx]) if not np.isnan(p_chop[ev_idx]) else 0.5
                ts = float(trend_score[ev_idx]) if not np.isnan(trend_score[ev_idx]) else 0.0
                at = float(atr_14[ev_idx]) if not np.isnan(atr_14[ev_idx]) else 1.0

                side_primary = (1 if ts >= 0.0 else -1) * self.primary_direction
                sigma = at / max(entry_p, 1e-6)

                # ============================================================
                # [FIX F4] p_i PHẢI LÀ ĐẠI LƯỢNG DÙNG LÚC INFERENCE
                # ============================================================
                # Bản cũ truyền p_i = p_trend (posterior HMM) khi dựng bảng Kelly,
                # nhưng live lại tra bảng bằng model.predict_proba() — hai đại lượng
                # khác phân phối hoàn toàn, khiến toàn bộ sizing là lỗi phạm trù.
                # Nay dùng chính xác suất OOS của model, đúng thứ live sẽ dùng.
                pos_ev = pos_of_event.get(int(ev_idx))
                if fold_model is not None and pos_ev is not None:
                    p_model = float(
                        fold_model.predict_proba(X_all[pos_ev].reshape(1, -1))[0, 1]
                    )
                else:
                    # Không fit được fold này -> bỏ qua, KHÔNG thay thế bằng p_trend.
                    # Trộn hai đại lượng khác nhau vào cùng một bảng Kelly chính là F4.
                    continue

                # Chạy trailing exit OOS
                partial = run_trailing_exit_for_oos_event(
                    entry_idx=int(ev_idx),
                    entry_price=entry_p,
                    test_window_end_idx=max_test_bar,
                    p_i=p_model,
                    p_chop_i=pc,
                    side_primary=side_primary,
                    m_sl=float(self.config.get("m_sl", 2.0)),
                    sigma=sigma,
                    c_trade_adj=float(self.config.get("c_trade", 0.0004)),
                    fade_enabled=bool(self.config.get("fade_enabled", True)),
                    fade_regime_gate_threshold=0.60,
                    full_highs=highs,
                    full_lows=lows,
                    full_atr=atr_14,
                    full_p_trend=p_trend,
                    t_max_live_follow=self.t_max_live_follow,
                    t_max_live_fade=self.t_max_live_fade,
                    p_trend_exit_threshold_follow=thr_follow,
                    p_trend_exit_threshold_fade=thr_fade,
                    full_timestamps=timestamps_ms,
                )

                if partial is None:
                    continue

                partial["dataset_manifest_hash"] = manifest_hash
                partial["fold_id"] = fold_id
                partial["p_source"] = "model_oos"  # [FIX F4] nguồn gốc của p_i
                partial["symbol"] = symbol

                # Hoàn thiện bản ghi giao dịch
                # ============================================================
                # [FIX F18] TRỪ FUNDING CỦA HỢP ĐỒNG VĨNH CỬU
                # ============================================================
                # Bản cũ gọi finalize_trade_record KHÔNG truyền funding, nên mọi
                # backtest bỏ qua chi phí ĐẶC TRƯNG nhất của perp. Với BTC, funding
                # trung bình +0.0101%/8h; lệnh giữ 5 ngày = 15 chu kỳ = 0.152%
                # notional — đủ để xoá sạch một "edge" +0.209%/lệnh.
                size_notional = float(self.config.get("backtest_size_notional", 10000.0))
                entry_ts = int(timestamps_ms[int(ev_idx)])
                exit_ts = int(timestamps_ms[min(int(partial["exit_idx_absolute"]), n_bars - 1)])
                funding_usd = compute_funding_accrued_usd(
                    size_notional=size_notional,
                    side=int(partial["side"]),
                    entry_ts_ms=entry_ts,
                    exit_ts_ms=exit_ts,
                    funding_rate=funding_schedule,
                    interval_hours=int(self.config.get("funding_interval_hours", 8)),
                )

                full_rec = finalize_trade_record(
                    partial_record=partial,
                    full_closes=closes,
                    size_notional=size_notional,
                    full_timestamps=timestamps_ms,
                    funding_accrued=funding_usd,
                )
                all_trade_records.append(full_rec)

        # ====================================================================
        # [FIX F5] CẢNH BÁO ĐÓI DỮ LIỆU — KHÔNG ĐƯỢC IM LẶNG XUẤT BẢNG KELLY RÁC
        # ====================================================================
        # Chân trời purge đúng (= t_max_live) ăn rất nhiều mẫu train. Trước đây bản
        # cũ purge 10 nến nên "có vẻ chạy được" — thực chất là do rò rỉ. Khi purge
        # đúng, dữ liệu ngắn sẽ lộ ra ngay: phần lớn fold không đủ mẫu để fit.
        n_folds_total = self.n_fits + self.n_folds_starved
        if n_folds_total > 0 and self.n_fits < n_folds_total:
            warnings.warn(
                f"[CPCV ĐÓI DỮ LIỆU] Chỉ fit được {self.n_fits}/{n_folds_total} fold. "
                f"{self.n_folds_starved} fold không đủ mẫu train sau purge "
                f"({purge_horizon} nến) + embargo ({self.embargo_bars} nến). "
                f"Bảng Kelly dựng từ đây KHÔNG đáng tin. "
                f"Cần nhiều dữ liệu hơn: khuyến nghị tối thiểu "
                f"~{purge_horizon * 50} nến và ~{self.n_groups * 20} sự kiện "
                f"(hiện có {n_bars} nến, {n_events} sự kiện).",
                UserWarning,
            )
        if self.n_fits == 0:
            warnings.warn(
                "[CPCV NGHIÊM TRỌNG] Không fit được fold nào — mọi chỉ số OOS và "
                "bảng Kelly trả về đều VÔ NGHĨA. Tuyệt đối không đem đi giao dịch.",
                UserWarning,
            )

        # 3. Bước 2.5: Tách boundary truncated
        clean_records, diagnostic_records = filter_boundary_truncated_for_kelly_table(all_trade_records)

        # 4. Bước 3: Dựng bảng Kelly 2D cho Follow và Fade
        records_follow = [r for r in clean_records if r.get("mode") == "follow"]
        records_fade = [r for r in clean_records if r.get("mode") == "fade"]

        kelly_follow = build_empirical_kelly_table_v2(records_follow, num_bins=num_bins)
        kelly_fade = build_empirical_kelly_table_v2(records_fade, num_bins=num_bins)

        # 5. Bước 4 & 5: Đánh giá OOS Sharpe, DSR, PBO trên TOÀN BỘ trade records
        all_returns = np.array([r["realized_return"] for r in all_trade_records], dtype=np.float64)
        n_trades = len(all_returns)

        # [FIX F15] Sharpe phải quy năm bằng TẦN SUẤT GIAO DỊCH THẬT.
        # Bản cũ nhân cứng sqrt(252) vào lợi suất MỖI LỆNH — hằng số lịch đó chỉ đúng
        # nếu hệ thống vào đúng 1 lệnh mỗi ngày giao dịch. Với dollar-volume bar, tần
        # suất thật có thể lệch hàng chục lần, khiến con số Sharpe báo cáo vô nghĩa.
        span_ms = float(timestamps_ms[-1] - timestamps_ms[0]) if n_bars > 1 else 0.0
        years = span_ms / (365.25 * 24 * 3600 * 1000.0)
        trades_per_year = (n_trades / years) if years > 1e-9 else 0.0

        if n_trades > 1 and trades_per_year > 0.0:
            mean_ret = float(np.mean(all_returns))
            std_ret = float(np.std(all_returns, ddof=1)) + 1e-12
            sharpe_per_trade = mean_ret / std_ret
            sharpe_oos = sharpe_per_trade * float(np.sqrt(trades_per_year))

            # [FIX F15] DSR cần PHƯƠNG SAI CỦA CÁC SHARPE QUA CÁC TRIAL, không phải
            # phương sai của lợi suất. Mỗi fold CPCV là một trial -> tính Sharpe từng
            # fold rồi lấy phương sai giữa chúng.
            fold_sharpes = []
            for fold_id in {r.get("fold_id") for r in all_trade_records}:
                fold_rets = np.array(
                    [r["realized_return"] for r in all_trade_records if r.get("fold_id") == fold_id],
                    dtype=np.float64,
                )
                if len(fold_rets) > 1:
                    fold_std = float(np.std(fold_rets, ddof=1))
                    if fold_std > 1e-12:
                        fold_sharpes.append(
                            float(np.mean(fold_rets)) / fold_std * float(np.sqrt(trades_per_year))
                        )

            variance_of_srs = (
                float(np.var(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0
            )

            try:
                dsr_res = compute_deflated_sharpe_ratio(
                    sr_estimated=sharpe_oos,
                    sample_length=n_trades,
                    num_trials=max(1, len(fold_sharpes)),
                    variance_of_srs=max(variance_of_srs, 1e-12),
                    skewness=float(_safe_skew(all_returns)),
                    kurtosis=float(_safe_kurtosis(all_returns)),
                )
                dsr_val = dsr_res.get("dsr", 0.0)
            except Exception:
                dsr_val = 0.0
        else:
            sharpe_oos = 0.0
            dsr_val = 0.0
            trades_per_year = 0.0

        metrics = {
            "n_total_trades": n_trades,
            "n_clean_trades": len(clean_records),
            "n_truncated_trades": len(diagnostic_records),
            "truncation_rate": float(len(diagnostic_records) / max(n_trades, 1)),
            "sharpe_oos": float(sharpe_oos),
            "trades_per_year": float(trades_per_year),
            "n_fits": int(self.n_fits),  # [FIX F5] số model đã fit thật (0 = chưa fit gì)
            "n_folds_starved": int(self.n_folds_starved),
            "purge_horizon": int(purge_horizon),
            "dsr": float(dsr_val),
            "dataset_manifest_hash": manifest_hash,
        }

        return {
            "trade_records": all_trade_records,
            "clean_records": clean_records,
            "diagnostic_records": diagnostic_records,
            "kelly_table_follow": kelly_follow,
            "kelly_table_fade": kelly_fade,
            "metrics": metrics,
        }
