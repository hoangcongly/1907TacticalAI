"""
Pipeline live cho chiến lược cross-sectional market-neutral.

`live_pipeline.py` cũ điều khiển MỘT vị thế theo hướng dự đoán — chiến lược đó có
t-stat 1.29 và không dùng nữa. Module này chạy chiến lược ĐÃ QUA HOLDOUT: danh mục
~12 vị thế long/short, tái cân bằng hằng ngày, khớp maker.

Thứ tự thực hiện mỗi lượt (không được đảo):
  1. Nạp trạng thái bền vững      -> khôi phục đỉnh vốn, kill switch
  2. ĐỐI CHIẾU với sàn            -> lệch thì DỪNG, không đoán
  3. Kiểm tra ngắt mạch rủi ro    -> drawdown vượt ngưỡng thì đóng băng
  4. Kiểm tra độ tươi dữ liệu     -> dữ liệu cũ thì không giao dịch
  5. Tính tín hiệu -> trọng số
  6. Dựng kế hoạch -> gửi lệnh
  7. Lưu trạng thái

Bước 2 và 4 là hai chốt chặn quan trọng nhất: giao dịch trên trạng thái sai hoặc
dữ liệu cũ nguy hiểm hơn nhiều so với việc bỏ lỡ một lượt tái cân bằng.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aegis.core.execution_log import ExecutionLog, ExecutionRecord
from aegis.core.state_store import LiveState, StateStore
from aegis.data.ingestion.binance_history import download_klines, save_klines
from aegis.data.ingestion.binance_rest import BinanceFuturesREST
from aegis.data.panel import load_funding_panel, load_panel
from aegis.execution.portfolio_rebalancer import (
    RebalancePlan, SymbolFilters, build_rebalance_plan, max_positions_for_capital,
)
from aegis.oms.order_router import BinanceOrderRouter
from aegis.oms.reconciliation import ReconciliationError, assert_clean_or_raise, reconcile
from aegis.data.universe import UniverseFilter, build_universe
from aegis.research.cross_sectional import (
    combine_signals_zscore, rank_to_weights, rank_to_weights_buffered,
    signal_funding_carry, signal_funding_momentum, signal_momentum, signal_ofi,
)
from aegis.data.panel_v2 import load_funding_panel_v2, load_panel_v2
from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights

logger = logging.getLogger(__name__)

# Ngưỡng ngắt mạch trên equity danh mục.
TIER1_DD, TIER2_DD, TIER3_DD = 0.10, 0.20, 0.30
FREEZE_MS = 24 * 60 * 60 * 1000


@dataclass
class LiveConfig:
    """Cấu hình đã qua kiểm định holdout — xem artifacts/strategy_validated.json."""
    universe: List[str]
    interval: str = "4h"
    top_frac: float = 0.10
    leverage: float = 2.0
    rebalance_hours: float = 24.0
    max_data_age_hours: float = 8.0
    post_only: bool = True
    min_notional_safety: float = 1.2
    passive_offset_ticks: int = 1     # đệm khỏi BBO để lệnh post-only không bị -5022
    exit_frac: float = 0.15           # vùng đệm thứ hạng: giữ tới khi rơi khỏi top này
    passive_wait_s: float = 300.0     # chờ khớp maker trước khi cắn giá
    allow_taker_fallback: bool = True # đảm bảo về đúng trạng thái trung lập
    requote: bool = True              # bám theo BBO để lệnh maker khớp được
    max_chase_bps: float = 15.0       # trần đuổi giá, chống rượt theo cú chạy
    min_signal_coverage: int = 3      # đa số tín hiệu phải có dữ liệu
    min_history_bars: int = 120       # > lookback dài nhất (momentum_90)
    min_quote_volume_24h: float = 5e6
    signals: List[str] = field(default_factory=lambda: [
        "funding_carry", "momentum_90", "ofi_flow", "funding_mom",
    ])

    # --- v3: thư viện tín hiệu rộng + gộp thích ứng ---
    # engine="v1" giữ nguyên hành vi cũ (4 tín hiệu, xếp hạng nhị phân).
    # engine="v3" dùng ĐÚNG đường tính của research (`research/strategy_v2.py`) —
    # đây là điều kiện để không tái lập lỗi F3 (research và live lệch công thức).
    engine: str = "v1"
    n_positions: int = 12
    rebalance_bars: int = 18          # số nến giữa hai lần tái cân bằng (v3)
    source_interval: str = "1h"       # khung nguồn để tổng hợp (v3)
    weight_mode: str = "zscore_riskparity"
    max_weight: float = 0.20
    combiner_lookback: int = 500
    combiner_min_periods: int = 120
    combiner_t_threshold: float = 2.0
    combiner_max_abs_weight: float = 0.20
    combiner_max_step: float = 0.05

    @classmethod
    def from_artifacts(cls, path: str = "artifacts/strategy_validated.json",
                       universe_path: str = "artifacts/universe70.json") -> "LiveConfig":
        cfg = json.load(open(path, encoding="utf-8"))
        uni_file = cfg.get("universe_file", universe_path)
        universe = json.load(open(uni_file, encoding="utf-8"))
        # File v3 gói cấu hình trong khoá "config"; file v1 để phẳng ở gốc.
        inner = cfg.get("config", cfg)
        uni_file = inner.get("universe_file", uni_file)
        universe = json.load(open(uni_file, encoding="utf-8"))
        engine = "v3" if "config" in cfg else "v1"

        kw: Dict[str, Any] = dict(
            universe=universe,
            interval=inner.get("interval", "4h"),
            top_frac=cfg.get("top_frac", 0.10),
            signals=cfg.get("signals", []),
            min_signal_coverage=int(cfg.get("min_coverage", 3)),
            engine=engine,
        )
        if engine == "v3":
            comb = inner.get("combiner", {})
            port = inner.get("portfolio", {})
            kw.update(
                n_positions=int(inner.get("n_positions", 12)),
                rebalance_bars=int(inner.get("rebalance_every", 18)),
                rebalance_hours=float(inner.get("period_hours", 72.0)),
                weight_mode=port.get("mode", "zscore_riskparity"),
                max_weight=float(port.get("max_weight", 0.20)),
                combiner_lookback=int(comb.get("lookback", 500)),
                combiner_min_periods=int(comb.get("min_periods", 120)),
                combiner_t_threshold=float(comb.get("t_threshold", 2.0)),
                combiner_max_abs_weight=float(comb.get("max_abs_weight", 0.20)),
                combiner_max_step=float(comb.get("max_step", 0.05)),
                # Thư viện v3 cần lịch sử dài nhất là mom_52w_high (720 nến 4h).
                min_history_bars=760,
            )
        return cls(**kw)


class CrossSectionalLivePipeline:
    """Vòng lặp vận hành chiến lược cross-sectional."""

    def __init__(
        self,
        config: LiveConfig,
        client: Optional[BinanceFuturesREST] = None,
        router: Optional[BinanceOrderRouter] = None,
        state_store: Optional[StateStore] = None,
        dry_run: bool = True,
        execution_log: Optional[ExecutionLog] = None,
    ):
        self.config = config
        self.client = client or BinanceFuturesREST()
        self.dry_run = bool(dry_run)
        self.router = router or BinanceOrderRouter(client=self.client, dry_run=dry_run)
        self.store = state_store or StateStore()
        self.exec_log = execution_log or ExecutionLog()
        # Dữ liệu thị trường LUÔN lấy từ mainnet: sổ lệnh testnet là thanh khoản giả.
        self.data_client = BinanceFuturesREST.public_mainnet()
        self._filters: Dict[str, SymbolFilters] = {}

    # ------------------------------------------------------------------ dữ liệu
    def refresh_data(self, days: float = 30.0) -> int:
        """Tải nến mới nhất cho universe (gộp vào parquet đã có)."""
        updated = 0
        for symbol in self.config.universe:
            try:
                bars = download_klines(symbol, self.config.interval, days=days,
                                       client=self.data_client)
                save_klines(bars, symbol, self.config.interval)
                updated += 1
            except Exception as exc:
                logger.warning("Không cập nhật được %s: %s", symbol, exc)
        return updated

    def load_filters(self) -> Dict[str, SymbolFilters]:
        """Bộ lọc sàn cho từng cặp (cache trong phiên)."""
        if self._filters:
            return self._filters
        for symbol in self.config.universe:
            try:
                f = self.client.symbol_filters(symbol)
                self._filters[symbol] = SymbolFilters(
                    symbol, f["tick_size"], f["step_size"], f["min_qty"], f["min_notional"])
            except Exception as exc:
                logger.warning("Không lấy được bộ lọc %s: %s", symbol, exc)
        return self._filters

    # ------------------------------------------------------------------ tín hiệu
    def resolve_universe(self) -> List[str]:
        """
        Lọc universe xuống các cặp THỰC SỰ giao dịch được trên sàn thực thi.

        Phải làm TRƯỚC khi tính trọng số. Nếu để tới lúc gửi lệnh mới phát hiện cặp
        không tồn tại, lệnh bị bỏ âm thầm và danh mục mất cân bằng long/short —
        biến chiến lược market-neutral thành một cược có hướng ngoài ý muốn.
        """
        keep, rejected = build_universe(
            candidates=self.config.universe,
            interval=self.config.interval,
            filt=UniverseFilter(
                min_history_bars=self.config.min_history_bars,
                min_quote_volume_24h=self.config.min_quote_volume_24h,
            ),
            execution_client=self.client,
            data_client=self.data_client,
        )
        if rejected:
            logger.info("Universe loại %d cặp: %s", len(rejected), list(rejected)[:5])
        if len(keep) < 10:
            raise RuntimeError(f"Universe sau lọc chỉ còn {len(keep)} cặp — quá ít để market-neutral")
        return keep

    def _compute_target_weights_v3(self) -> tuple[Dict[str, float], int]:
        """
        Trọng số mục tiêu theo chiến lược v3 — GỌI ĐÚNG các hàm research dùng.

        Đây là điểm mấu chốt về kiến trúc. Lỗi F3 của hệ thống cũ không phải lỗi
        công thức mà là lỗi CÓ HAI ĐƯỜNG: research tính feature một kiểu, live tính
        một kiểu, và chúng trôi xa nhau theo thời gian. Ở đây live gọi nguyên si
        `build_signal` / `combine_adaptive` / `build_weights` rồi lấy HÀNG CUỐI —
        không có công thức nào được viết lại lần thứ hai.

        Hệ quả bắt buộc chấp nhận: live phải nạp đủ lịch sử để tính được đúng những
        gì research tính (tín hiệu dài nhất cần 720 nến, tầng gộp cần thêm
        `combiner_lookback` mốc tái cân bằng). Nạp thiếu thì tín hiệu KHÁC, và cách
        duy nhất an toàn là ném lỗi chứ không chạy tiếp với dữ liệu ngắn.
        """
        universe = self.resolve_universe()
        panel = load_panel_v2(universe, self.config.interval,
                              source_interval=self.config.source_interval,
                              min_coverage=0.15)
        close = panel["close"]
        funding = load_funding_panel_v2(universe, close.index).reindex(columns=close.columns)

        need = self.config.min_history_bars + self.config.rebalance_bars * 8
        if len(close) < need:
            raise RuntimeError(
                f"Chỉ có {len(close)} nến {self.config.interval}, cần >= {need} để tín hiệu v3 "
                f"khớp với research. Chạy scripts/download_wide_universe.py trước.")

        sigs = {n: build_signal(n, panel, funding) for n in SIGNAL_REGISTRY}

        marks = close.index[::self.config.rebalance_bars]
        close_m = close.reindex(marks)
        combined = combine_adaptive(
            {k: v.reindex(marks) for k, v in sigs.items()}, close_m,
            CombinerSpec(lookback=self.config.combiner_lookback,
                         min_periods=self.config.combiner_min_periods,
                         t_threshold=self.config.combiner_t_threshold,
                         max_abs_weight=self.config.combiner_max_abs_weight,
                         max_step=self.config.combiner_max_step),
            top_frac=self.config.top_frac)

        spec = PortfolioSpec(
            mode=self.config.weight_mode,
            n_positions=self.config.n_positions,
            top_frac=self.config.top_frac,
            max_weight=1.0 if self.config.weight_mode == "rank_binary" else self.config.max_weight,
            beta_neutral=False,
        )
        # Chỉ dựng trọng số cho đuôi chuỗi: `build_weights` lặp theo hàng, mà live
        # chỉ cần hàng cuối. Vẫn phải chừa đủ cửa sổ ước lượng biến động.
        tail = marks[-(spec.vol_window + 5):]
        W = build_weights(combined.reindex(tail), close.reindex(tail), spec)

        row = W.iloc[-1]
        weights = {s: float(w) for s, w in row.items() if abs(w) > 1e-12}
        if not weights:
            raise RuntimeError("Chiến lược v3 không mở vị thế nào ở nến mới nhất "
                               "(thiếu độ phủ tín hiệu?)")
        return weights, int(close.index[-1])

    def compute_target_weights(
        self, current_positions: Optional[Dict[str, float]] = None
    ) -> tuple[Dict[str, float], int]:
        """
        Tính 4 tín hiệu đã kiểm định, gộp đều, trả về trọng số mục tiêu.

        Trả kèm timestamp của nến mới nhất để bước sau kiểm tra độ tươi dữ liệu.
        """
        if self.config.engine == "v3":
            return self._compute_target_weights_v3()

        universe = self.resolve_universe()
        panel = load_panel(universe, self.config.interval, min_coverage=0.15)
        close, ofi = panel["close"], panel["ofi"]
        funding = load_funding_panel(universe, close.index).reindex(columns=close.columns)

        available = {
            "funding_carry": lambda: signal_funding_carry(funding, 6),
            "momentum_90": lambda: signal_momentum(close, 90),
            "ofi_flow": lambda: signal_ofi(ofi, 6),
            "funding_mom": lambda: signal_funding_momentum(funding, 42),
        }
        chosen = [s for s in self.config.signals if s in available]
        if not chosen:
            raise RuntimeError(f"Không tín hiệu nào hợp lệ trong {self.config.signals}")

        # Gộp Z-SCORE rồi xếp hạng MỘT LẦN, GIỮ NaN (không fillna(0)).
        # fillna(0) coi "không có dữ liệu" như "tín hiệu trung tính" -> cặp thiếu
        # dữ liệu vẫn lọt vào danh mục dựa trên thông tin không tồn tại.
        combined = combine_signals_zscore(
            {s: available[s]() for s in chosen},
            min_coverage=self.config.min_signal_coverage,
        )

        latest_ts = int(close.index[-1])
        last_row = combined.iloc[[-1]]

        # [VÙNG ĐỆM] Áp trực tiếp lên vị thế ĐANG NẮM: cặp vẫn nằm trong vùng giữ
        # thì không bán, dù đã rơi khỏi top vào. Giảm turnover ~19% mà không mất
        # Sharpe (đo trên train), đồng thời hạ drawdown 26.2% -> 21.3%.
        if current_positions and self.config.exit_frac > self.config.top_frac:
            row = last_row.iloc[-1].dropna()
            n = len(row)
            if n >= 4:
                k_in = max(1, int(round(n * self.config.top_frac)))
                k_out = max(k_in, int(round(n * self.config.exit_frac)))
                ordered = row.sort_values()
                keep_short = set(ordered.index[:k_out])
                keep_long = set(ordered.index[-k_out:])
                enter_short = set(ordered.index[:k_in])
                enter_long = set(ordered.index[-k_in:])

                held_long = {s for s, q in current_positions.items() if q > 0}
                held_short = {s for s, q in current_positions.items() if q < 0}
                surv_long = (held_long & keep_long) - held_short
                surv_short = (held_short & keep_short) - held_long

                cand_long = [x for x in reversed(ordered.index)
                             if x in enter_long and x not in surv_long and x not in surv_short]
                cand_short = [x for x in ordered.index
                              if x in enter_short and x not in surv_short and x not in surv_long]

                final_long = list(surv_long) + cand_long[:max(0, k_in - len(surv_long))]
                final_short = list(surv_short) + cand_short[:max(0, k_in - len(surv_short))]
                final_long = sorted(final_long, key=lambda x: -row[x])[:k_in]
                final_short = sorted(final_short, key=lambda x: row[x])[:k_in]

                if final_long and final_short:
                    out = {s: 0.5 / len(final_long) for s in final_long}
                    out.update({s: -0.5 / len(final_short) for s in final_short})
                    return out, latest_ts

        weights = rank_to_weights(last_row, top_frac=self.config.top_frac, neutral=True)
        return {s: float(w) for s, w in weights.iloc[-1].items() if abs(w) > 1e-12}, latest_ts

    # ------------------------------------------------------------------ rủi ro
    def _check_circuit_breaker(self, state: LiveState, equity: float, now_ms: int) -> Optional[str]:
        """Trả về lý do chặn giao dịch, hoặc None nếu được phép."""
        if state.is_dead:
            return "KILL_SWITCH đã kích hoạt vĩnh viễn — cần con người can thiệp"
        if now_ms < state.frozen_until_ms:
            remaining = (state.frozen_until_ms - now_ms) / 3_600_000
            return f"ĐANG ĐÓNG BĂNG còn {remaining:.1f} giờ"

        dd = state.drawdown(equity)
        if dd >= TIER3_DD:
            state.is_dead = True
            return f"TIER 3: drawdown {dd:.1%} >= {TIER3_DD:.0%} — KILL SWITCH"
        if dd >= TIER2_DD:
            state.frozen_until_ms = now_ms + FREEZE_MS
            return f"TIER 2: drawdown {dd:.1%} >= {TIER2_DD:.0%} — đóng băng 24h"
        if dd >= TIER1_DD:
            logger.warning("TIER 1: drawdown %.1f%% — giảm nửa vị thế", dd * 100)
        return None

    def _position_multiplier(self, state: LiveState, equity: float) -> float:
        return 0.5 if state.drawdown(equity) >= TIER1_DD else 1.0

    # ------------------------------------------------------------------ vòng lặp
    def run_once(self, skip_data_refresh: bool = False) -> Dict[str, Any]:
        """Chạy một lượt tái cân bằng đầy đủ."""
        now_ms = int(time.time() * 1000)
        state = self.store.load()
        result: Dict[str, Any] = {"timestamp_ms": now_ms, "dry_run": self.dry_run}

        # --- 1. Số dư & đối chiếu ---
        balance = self.client.balance_usdt()
        equity = balance["wallet_balance"] + balance["unrealized_pnl"]
        result["equity"] = equity

        exchange_pos = {
            p["symbol"]: float(p["positionAmt"])
            for p in self.client.position_risk()
            if abs(float(p.get("positionAmt", 0))) > 0
        }
        try:
            report = reconcile(self.client, expected_positions=state.positions)
            assert_clean_or_raise(report)
        except ReconciliationError as exc:
            # Lần chạy đầu chưa có trạng thái -> nhận vị thế hiện có làm mốc.
            if state.rebalance_count == 0 and not state.positions:
                logger.warning("Lần chạy đầu: nhận %d vị thế sẵn có làm mốc", len(exchange_pos))
                state.positions = dict(exchange_pos)
            else:
                state.last_error = str(exc)
                self.store.save(state)
                result.update(action="HALT", reason="RECONCILIATION_FAILED", detail=str(exc))
                return result

        # --- 2. Cập nhật đỉnh vốn rồi kiểm ngắt mạch ---
        state.peak_equity = max(state.peak_equity, equity)
        state.last_equity = equity
        blocked = self._check_circuit_breaker(state, equity, now_ms)
        if blocked:
            self.store.save(state)
            result.update(action="HALT", reason="CIRCUIT_BREAKER", detail=blocked,
                          drawdown=state.drawdown(equity))
            return result

        # --- 3. Dữ liệu ---
        if not skip_data_refresh:
            result["symbols_updated"] = self.refresh_data(days=30.0)

        weights, latest_ts = self.compute_target_weights(current_positions=exchange_pos)
        age_hours = (now_ms - latest_ts) / 3_600_000
        result["data_age_hours"] = round(age_hours, 2)
        if age_hours > self.config.max_data_age_hours:
            result.update(action="HALT", reason="STALE_DATA",
                          detail=f"nến mới nhất đã {age_hours:.1f}h tuổi "
                                 f"(trần {self.config.max_data_age_hours}h)")
            self.store.save(state)
            return result

        # --- 4. Sức chứa vốn ---
        cap = max_positions_for_capital(equity, self.config.leverage,
                                        min_notional=5.0, safety=self.config.min_notional_safety)
        if len(weights) > cap:
            # Giữ lại các vị thế có tín hiệu MẠNH NHẤT ở cả hai chiều.
            ranked = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
            weights = dict(ranked[:cap])
            logger.warning("Vốn chỉ đủ %d vị thế — cắt từ %d xuống", cap, len(ranked))
        result["n_targets"] = len(weights)
        result["capacity"] = cap

        # --- 5. Kế hoạch & gửi lệnh ---
        # [FIX PASSIVE-PRICING] Lấy BBO của SÀN THỰC THI (không phải mainnet):
        # lệnh sẽ khớp trên sổ của sàn này, nên phải neo vào chính sổ đó.
        # Dùng giá giữa (mid) để dựng kế hoạch, rồi neo lại theo bid/ask khi đặt lệnh.
        prices, book = {}, {}
        for symbol in set(weights) | set(exchange_pos):
            try:
                bid, ask = self.client.best_bid_ask(symbol)
                if bid <= 0 or ask <= 0:
                    continue
                book[symbol] = (bid, ask)
                prices[symbol] = (bid + ask) / 2.0
            except Exception as exc:
                logger.warning("Không lấy được BBO %s: %s", symbol, exc)

        multiplier = self._position_multiplier(state, equity)
        plan: RebalancePlan = build_rebalance_plan(
            target_weights=weights,
            current_qty=exchange_pos,
            prices=prices,
            filters=self.load_filters(),
            equity=equity,
            leverage=self.config.leverage * multiplier,
        )
        result.update(n_orders=len(plan.orders), turnover=plan.total_turnover,
                      gross_notional=plan.gross_notional, skipped=len(plan.skipped),
                      position_multiplier=multiplier)

        if self.dry_run:
            result.update(action="DRY_RUN", orders=[
                {"symbol": o.symbol, "side": o.side, "qty": o.qty,
                 "price": o.price, "notional": round(o.notional, 2), "reason": o.reason}
                for o in plan.orders])
            return result

        # [FIX STALE-ORDERS] Huỷ lệnh treo từ lượt trước. Lệnh maker không khớp hết
        # sẽ nằm lại trên sổ; để nguyên thì lượt sau chồng thêm lệnh mới lên trên
        # cùng ý định và vị thế bị nhân đôi khi cả hai cùng khớp.
        for symbol in {o.symbol for o in plan.orders} | set(exchange_pos):
            self.router.cancel_all(symbol)
        time.sleep(0.5)

        # [FIX PASSIVE-PRICING] Neo giá về phía THỤ ĐỘNG của sổ lệnh:
        # BUY đặt ở BID, SELL đặt ở ASK. Đặt ở giá vượt qua sổ sẽ khiến lệnh
        # post-only bị từ chối với -5022 (đã xảy ra thật khi chạy testnet).
        filters = self.load_filters()
        for o in plan.orders:
            bb = book.get(o.symbol)
            if bb is None:
                continue
            bid, ask = bb
            f = filters.get(o.symbol)
            tick = f.tick_size if f else 0.0
            # Lùi thêm `passive_offset_ticks` khỏi BBO làm đệm: sổ lệnh dịch chuyển
            # trong vài chục ms giữa lúc đọc giá và lúc lệnh tới sàn.
            off = tick * self.config.passive_offset_ticks
            passive = (bid - off) if o.side == "BUY" else (ask + off)
            o.price = f.round_price(passive) if f else passive

        exec_report = self.router.execute_with_fallback(
            plan.orders, filters,
            passive_wait_s=self.config.passive_wait_s,
            allow_taker_fallback=self.config.allow_taker_fallback,
            requote=self.config.requote,
            max_chase_bps=self.config.max_chase_bps,
        )
        result["execution"] = exec_report
        result["submitted"] = exec_report["passive_submitted"]
        result["taker_fallback"] = exec_report["taker_orders"]
        result["unfilled"] = len(exec_report["unfilled"])

        # --- 6. Lưu trạng thái ---
        time.sleep(1.5)
        state.positions = {
            p["symbol"]: float(p["positionAmt"])
            for p in self.client.position_risk()
            if abs(float(p.get("positionAmt", 0))) > 0
        }
        state.last_rebalance_ms = now_ms
        state.rebalance_count += 1
        state.last_error = None
        self.store.save(state)

        # Ghi nhật ký chi phí THẬT để về sau đối chiếu với giả định backtest.
        gross_live = sum(abs(q) * prices.get(sym, 0.0) for sym, q in state.positions.items())
        net_live = sum(q * prices.get(sym, 0.0) for sym, q in state.positions.items())
        self.exec_log.append(ExecutionRecord(
            timestamp_ms=now_ms, equity=equity, n_targets=len(weights),
            n_orders=len(plan.orders), gross_notional=gross_live,
            planned_turnover=plan.total_turnover,
            maker_notional=float(exec_report["passive_filled_notional"]),
            taker_notional=float(exec_report["taker_notional"]),
            unfilled_count=len(exec_report["unfilled"]),
            net_exposure=(net_live / gross_live) if gross_live > 0 else 0.0,
            testnet=bool(getattr(self.client.credentials, "testnet", True)),
        ))

        result["action"] = "REBALANCED"
        return result

    def kill(self) -> Dict[str, Any]:
        """Dừng khẩn cấp: đóng sạch vị thế và đánh dấu kill switch vĩnh viễn."""
        report = self.router.kill_switch()
        state = self.store.load()
        state.is_dead = True
        state.positions = {}
        self.store.save(state)
        return report
