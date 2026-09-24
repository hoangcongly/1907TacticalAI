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
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aegis.core.execution_log import ExecutionLog, ExecutionRecord
from aegis.core.state_store import LiveState, StateStore
from aegis.data.ingestion.binance_history import (
    download_funding_rates, download_klines, save_funding_rates, save_klines,
)
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
from aegis.data.panel_v2 import (
    funding_last_ms, interval_hours, load_funding_panel_v2, load_panel_v2,
)
from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.research.strategy_v3 import V3_SIGNALS
from aegis.risk.goal_overlay import GoalOverlay, GoalOverlayConfig
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
    #: [FIX F52] Trần tuổi funding (trung vị qua universe). Funding công bố mỗi 8h (vài
    #: cặp 4h) — 24h là đã lỡ ít nhất 3 mốc liên tiếp, không còn là trễ ngẫu nhiên.
    max_funding_age_hours: float = 24.0
    post_only: bool = True
    min_notional_safety: float = 1.2
    passive_offset_ticks: int = 1     # đệm khỏi BBO để lệnh post-only không bị -5022
    exit_frac: float = 0.15           # vùng đệm thứ hạng: giữ tới khi rơi khỏi top này
    #: Thời gian chờ khớp maker trước khi cắn giá.
    #:
    #: 300s -> 900s (14/09/2026). Con số 300 cũ là một cách chặn rủi ro GIÁN TIẾP:
    #: vòng chờ hoàn toàn mù về trung lập, nên phải để ngắn. Nay `max_fill_imbalance`
    #: chặn trực tiếp đúng rủi ro đó, nên chờ lâu hơn AN TOÀN HƠN bản cũ chứ không
    #: kém an toàn hơn — và tỷ lệ maker đo được mới chỉ 0,378-0,49, tức chi phí đang
    #: cao gấp đôi mức có thể đạt.
    #:
    #: Trong chu kỳ 72h, 15 phút là 0,3% thời gian. Đổi lại: maker 0,378 -> 0,85 đáng
    #: +0,06 Sharpe và +7,7% lợi nhuận (đo ở `scripts/risk_overlay_study.py` và bảng
    #: nhạy chi phí). Chỉ có hiệu lực sau khi NẠP LẠI daemon.
    passive_wait_s: float = 900.0
    allow_taker_fallback: bool = True # đảm bảo về đúng trạng thái trung lập
    requote: bool = True              # bám theo BBO để lệnh maker khớp được
    max_chase_bps: float = 15.0       # trần đuổi giá SÀN, dùng khi không đo được biến động
    #: [FIX F44] Bội số biến động cho trần đuổi giá theo từng cặp.
    #:
    #: Trần CỐ ĐỊNH 15bp sai về bản chất: rất rộng với BTC, cực hẹp với một altcoin
    #: σ=5%/giờ. Đo trên 60 cặp thật, dịch giá trung vị trong cửa sổ chờ 900s là
    #: 64,4bp -> 100% số cặp vượt trần, và trần bị chạm sau ~50 giây của 900 giây.
    #: Lệnh đóng băng ngoài thị trường 94% thời gian chờ rồi rơi xuống taker.
    #:
    #: Trần mới = max(`max_chase_bps`, `chase_sigma_mult` x σ của cặp trong cửa sổ chờ).
    #: 1,5σ nghĩa là "đuổi theo nhiễu bình thường, dừng khi là cú chạy thật" — cùng
    #: một ý nghĩa ở mọi cặp, điều mà một con số duy nhất không làm được.
    chase_sigma_mult: float = 1.5
    #: [FIX F45] Trần lệch trung lập trong lúc khớp thụ động, tính theo GROSS KẾ HOẠCH.
    #:
    #: Vượt trần là thoát chờ và cắn giá cho cân ngay. Đây là thứ cho phép
    #: `passive_wait_s` dài ra một cách AN TOÀN: rủi ro thật (khớp lệch một chiều) bị
    #: chặn trực tiếp thay vì bị chặn gián tiếp bằng một cái đồng hồ ngắn.
    #:
    #: THAY THẾ `neutrality_tolerance` (0,10 trên gross ĐÃ KHỚP) — thước đo cũ SAI
    #: MẪU SỐ. Chia cho phần đã khớp khiến phép đo nhiễu 26% ở tiến độ 25%, tức nhiễu
    #: lớn gấp 2,6 lần chính ngưỡng. Hệ quả đo được ở lượt 19/09/2026 (50 lệnh):
    #: van bắn ở tiến độ ~2%, `early_exit=neutrality_drift`, `requotes=0` — vòng báo
    #: giá lại và trần đuổi co giãn của F44 trở thành CODE CHẾT, maker tụt còn 39%.
    #:
    #: Đổi tên chứ không giữ tên cũ là CỐ Ý: ý nghĩa của con số đã đổi (mẫu số khác),
    #: nên mọi chỗ gọi phải nổ ra bằng TypeError thay vì âm thầm chạy sai đơn vị.
    #: Đây đúng khe hở F38 — công thức thì khoá, cấu hình thì không.
    #:
    #: 0,25 chọn bằng mô phỏng 8.000 lượt/kịch bản: báo nhầm 0,3%, bắt đúng 100%.
    #: Bảng đầy đủ ở `order_router.fill_imbalance` và `_passive_wait_loop`.
    max_fill_imbalance: float = 0.25
    min_signal_coverage: int = 3      # đa số tín hiệu phải có dữ liệu
    min_history_bars: int = 120       # > lookback dài nhất (momentum_90)
    min_quote_volume_24h: float = 5e6
    max_net_exposure: float = 0.02     # trần |net|/gross sau khi thực thi [FIX F25]
    max_gross_error: float = 0.15      # sai lệch đòn bẩy gộp cho phép [FIX F29]
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
    #: Danh sách tín hiệu v3. `None` = đúng bộ 26 đã kiểm định (`V3_SIGNALS`).
    #:
    #: [FIX F38] Bản trước duyệt thẳng `SIGNAL_REGISTRY`. Registry là nơi CHỨA mọi tín
    #: hiệu từng viết, kể cả tín hiệu đang thử nghiệm — nó không phải danh sách "những
    #: gì đang giao dịch". Thêm một họ mới vào registry (hoàn toàn hợp lệ khi nghiên
    #: cứu) sẽ âm thầm đổi thứ LIVE đặt lệnh, và `adaptive_weights` chuẩn hoá theo số
    #: họ nên mọi trọng số dịch đi. Không crash, không cảnh báo.
    #:
    #: Đây đúng là cơ chế của F3 và F37: research và live đọc hai nguồn sự thật khác
    #: nhau cho cùng một quyết định. Nay cả hai đọc `V3_SIGNALS`.
    v3_signals: Optional[List[str]] = None

    # --- Tầng phủ đòn bẩy theo mục tiêu (research/goal_dp.py) ---
    # None = tắt hoàn toàn, hành vi y hệt trước khi có tầng này. Bật nó KHÔNG đổi
    # thành phần danh mục, chỉ co giãn gross — nhờ vậy bất biến parity research/live
    # (`tests/pipelines/test_research_live_parity_v3.py`) vẫn nguyên vẹn.
    goal_overlay: Optional[GoalOverlayConfig] = None

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


def _pct(vals, q: float) -> float:
    """Phân vị `q` của một danh sách có thể rỗng/None — rỗng thì 0.0. [FIX F49]"""
    if not vals:
        return 0.0
    import numpy as _np
    return float(_np.percentile(_np.asarray(vals, dtype=float), q))


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
        self._goal_overlay: Optional[GoalOverlay] = None   # nạp lười ở lần dùng đầu
        # Dữ liệu thị trường LUÔN lấy từ mainnet: sổ lệnh testnet là thanh khoản giả.
        self.data_client = BinanceFuturesREST.public_mainnet()
        self._filters: Dict[str, SymbolFilters] = {}

    # ------------------------------------------------------------------ dữ liệu
    def data_interval(self) -> str:
        """
        Khung thời gian PHẢI được làm mới — chính là khung engine đọc từ đĩa.

        [FIX F26] Engine v3 nạp panel bằng `load_panel_v2(..., source_interval="1h")`
        rồi tự tổng hợp lên 4h. Nhưng `refresh_data` cũ tải `config.interval` ("4h").
        Hệ quả: file 1h KHÔNG BAO GIỜ được cập nhật, nên mọi lượt chạy v3 đều thấy
        dữ liệu cũ và dừng ở chốt STALE_DATA — hệ thống tự khoá vĩnh viễn dù không
        có gì hỏng. Lỗi này chỉ lộ ra khi chạy thật, vì backtest đọc file có sẵn.
        """
        return self.config.source_interval if self.config.engine == "v3" else self.config.interval

    def refresh_data(self, days: float = 30.0) -> int:
        """Tải nến VÀ funding mới nhất cho universe (gộp vào parquet đã có)."""
        interval = self.data_interval()
        updated, failed, f_failed = 0, [], []
        for symbol in self.config.universe:
            try:
                bars = download_klines(symbol, interval, days=days, client=self.data_client)
                save_klines(bars, symbol, interval)
                updated += 1
            except Exception as exc:
                logger.warning("Không cập nhật được %s (%s): %s", symbol, interval, exc)
                failed.append(symbol)
            # [FIX F52] Funding là ĐẦU VÀO của 5/26 tín hiệu (họ carry). Trước đây hàm này
            # chỉ tải nến, nên funding đứng yên từ lần tải tay cuối (10/09) mà không ai
            # biết: `load_funding_panel_v2` ffill giá trị cuối tới vô hạn. Tách try riêng:
            # funding hỏng không được kéo nến hỏng theo, và ngược lại.
            try:
                rates = download_funding_rates(symbol, days=days, client=self.data_client)
                if rates:
                    save_funding_rates(rates, symbol)
            except Exception as exc:
                logger.warning("Không cập nhật được funding %s: %s", symbol, exc)
                f_failed.append(symbol)
        if failed:
            logger.warning("[F26] %d/%d cặp không cập nhật được dữ liệu %s",
                           len(failed), len(self.config.universe), interval)
        if f_failed:
            logger.warning("[F52] %d/%d cặp không cập nhật được funding",
                           len(f_failed), len(self.config.universe))
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
    def resolve_universe(self, equity: Optional[float] = None) -> List[str]:
        """
        Lọc universe xuống các cặp THỰC SỰ giao dịch được VÀ vốn mua nổi.

        Phải làm TRƯỚC khi tính trọng số. Nếu để tới lúc gửi lệnh mới phát hiện cặp
        không tồn tại, lệnh bị bỏ âm thầm và danh mục mất cân bằng long/short —
        biến chiến lược market-neutral thành một cược có hướng ngoài ý muốn.

        [FIX F40] Cùng lập luận đó áp cho MIN NOTIONAL, và trước đây thì không. Min
        notional không đồng nhất: mainnet có 122/128 cặp ở $5 nhưng ETH/LTC/LINK/ETC/BCH
        ở $20 và BTC ở $50. Vốn $38 ở 2x cho $6,34 mỗi vị thế — sáu cặp đó không mua
        nổi, và nếu một trong chúng lọt vào top thì lệnh bị bỏ, danh mục mất một chân.

        `equity = None` thì bỏ qua bộ lọc vốn (dùng khi chỉ cần danh sách, không đặt lệnh).
        """
        max_notional = None
        if equity and equity > 0:
            # Dùng số vị thế ĐÃ ĐIỀU CHỈNH THEO VỐN, không dùng `n_positions` danh nghĩa.
            #
            # Nếu chia cho `n_positions` cứng, ngưỡng tụt theo vốn và có thể rơi XUỐNG
            # DƯỚI $5 — mức mà gần như mọi cặp đều cần. Khi đó bộ lọc quét sạch universe
            # và `resolve_universe` ném lỗi "còn 0 cặp". Một cú sụt 8% vốn sẽ làm chết
            # đường chạy: bộ lọc an toàn tự biến thành nguyên nhân sự cố.
            #
            # `max_positions_for_capital` đã trả lời đúng câu hỏi "vốn này nuôi nổi mấy
            # vị thế". Dùng nó thì ngưỡng LUÔN >= $5 theo đại số:
            #   n_eff = floor(E*L / (5*safety))  =>  E*L/n_eff >= 5*safety
            #   ngưỡng = (E*L/n_eff)/safety >= 5
            # Tức cặp $5 không bao giờ bị loại nhầm, dù vốn nhỏ tới đâu.
            n_eff = min(self.config.n_positions,
                        max_positions_for_capital(
                            equity, self.config.leverage, min_notional=5.0,
                            safety=self.config.min_notional_safety))
            if n_eff >= 1:
                max_notional = (equity * self.config.leverage / n_eff
                                / max(1.0, self.config.min_notional_safety))

        keep, rejected = build_universe(
            candidates=self.config.universe,
            interval=self.config.interval,
            filt=UniverseFilter(
                min_history_bars=self.config.min_history_bars,
                min_quote_volume_24h=self.config.min_quote_volume_24h,
                max_min_notional=max_notional,
            ),
            execution_client=self.client,
            data_client=self.data_client,
        )
        if rejected:
            logger.info("Universe loại %d cặp: %s", len(rejected), list(rejected)[:5])
        if len(keep) < 10:
            raise RuntimeError(f"Universe sau lọc chỉ còn {len(keep)} cặp — quá ít để market-neutral")
        return keep

    def _compute_target_weights_v3(self, equity: Optional[float] = None
                                   ) -> tuple[Dict[str, float], int]:
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
        universe = self.resolve_universe(equity)
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

        # [FIX F38] Bộ tín hiệu ĐÃ KIỂM ĐỊNH, không phải "mọi thứ có trong registry".
        names = tuple(self.config.v3_signals) if self.config.v3_signals else V3_SIGNALS
        missing = [n for n in names if n not in SIGNAL_REGISTRY]
        if missing:
            raise RuntimeError(
                f"Cấu hình yêu cầu tín hiệu không có trong registry: {missing}. "
                f"Không chạy tiếp với bộ tín hiệu khác bộ đã kiểm định.")
        sigs = {n: build_signal(n, panel, funding) for n in names}

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
        self, current_positions: Optional[Dict[str, float]] = None,
        equity: Optional[float] = None,
    ) -> tuple[Dict[str, float], int]:
        """
        Tính 4 tín hiệu đã kiểm định, gộp đều, trả về trọng số mục tiêu.

        Trả kèm timestamp của nến mới nhất để bước sau kiểm tra độ tươi dữ liệu.
        """
        if self.config.engine == "v3":
            return self._compute_target_weights_v3(equity)

        universe = self.resolve_universe(equity)
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
    def notifier_enabled(self) -> bool:
        n = getattr(self.router, "notifier", None)
        return bool(n and getattr(n, "is_configured", False) and not self.dry_run)

    def _chase_caps(self, orders) -> Dict[str, float]:
        """
        [FIX F44] Trần đuổi giá RIÊNG cho từng cặp, co giãn theo biến động của chính nó.

        Vì sao cần: trần cố định 15bp bị 100% số cặp vượt qua trong cửa sổ chờ 900s
        (dịch giá trung vị 64,4bp, cặp mạnh nhất 261bp). Lệnh bị đóng băng ngoài thị
        trường rồi rơi xuống taker — đó là nút thắt thật của tỷ lệ maker 54%, KHÔNG
        phải thời gian chờ.

        σ lấy từ nến 1h đã có sẵn trên đĩa, quy về cửa sổ chờ theo căn bậc hai thời
        gian. Cặp nào không đo được thì dùng trần sàn — thiếu dữ liệu phải ngả về
        phía THẬN TRỌNG, không phải phía nới rộng.
        """
        caps: Dict[str, float] = {}
        scale = math.sqrt(max(1.0, self.config.passive_wait_s) / 3600.0)
        for o in orders:
            try:
                kl = self.client.klines(o.symbol, "1h", limit=200)
                closes = np.array([float(k[4]) for k in kl], dtype=float)
                if len(closes) < 50:
                    continue
                sigma_1h = float(np.std(np.diff(np.log(closes))))
                if not np.isfinite(sigma_1h) or sigma_1h <= 0:
                    continue
                caps[o.symbol] = max(self.config.max_chase_bps,
                                     self.config.chase_sigma_mult * sigma_1h * scale * 1e4)
            except Exception as exc:
                logger.warning("[F44] Không đo được biến động %s (%s) — dùng trần sàn %.1fbp",
                               o.symbol, exc, self.config.max_chase_bps)
        return caps

    def _settle_positions(self, tries: int = 6, delay: float = 1.2,
                          min_reads: int = 3) -> Dict[str, float]:
        """
        Đọc vị thế từ sàn tới khi ỔN ĐỊNH, thay vì chụp một lần rồi tin.

        [FIX F24] Bản cũ `sleep(1.5)` rồi gọi `position_risk()` đúng một lần.
        `/fapi/v2/positionRisk` của Binance nhất quán theo kiểu eventual — lệnh khớp
        tại thời điểm T có thể chưa hiện ở T+1.5s. Chụp một lần là canh bạc; nếu
        trượt, sổ nội bộ thiếu vị thế và lượt sau sẽ tự khoá vì lệch sổ sách.

        `min_reads` là chi tiết KHÔNG được bỏ, và một test hồi quy đã chứng minh
        điều đó: chỉ đòi "hai lần đọc liên tiếp giống nhau" thì khi sàn trễ đều đặn
        vài nhịp, hai lần đọc đầu cùng thiếu một vị thế sẽ trùng nhau và hàm trả về
        SỚM với kết quả thiếu — tái tạo lại đúng lỗi định sửa. Vì vậy phải đọc đủ
        `min_reads` lần, trải qua ít nhất `min_reads * delay` giây, RỒI mới xét ổn định.

        Đây vẫn là biện pháp giảm xác suất, không phải bảo đảm. Lá chắn cuối cùng
        là bước đối chiếu đầu lượt sau (`reconcile`), và nó phải được giữ nguyên.
        """
        prev: Optional[Dict[str, float]] = None
        reads = 0
        for attempt in range(max(tries, min_reads)):
            time.sleep(delay)
            try:
                cur = {
                    p["symbol"]: float(p["positionAmt"])
                    for p in self.client.position_risk()
                    if abs(float(p.get("positionAmt", 0))) > 0
                }
            except Exception as exc:
                logger.warning("[F24] Đọc vị thế lỗi (lần %d): %s", attempt + 1, exc)
                continue
            reads += 1
            if reads >= min_reads and prev is not None and cur == prev:
                return cur
            prev = cur
        if prev is None:
            logger.error("[F24] KHÔNG đọc được vị thế từ sàn sau %d lần", tries)
            return {}
        logger.warning("[F24] Vị thế chưa ổn định sau %d lần đọc — dùng ảnh chụp cuối", reads)
        return prev

    def check_neutrality(self, positions: Dict[str, float], prices: Dict[str, float],
                         tolerance: float = 0.02, equity: Optional[float] = None,
                         target_leverage: Optional[float] = None,
                         gross_tolerance: float = 0.15) -> Dict[str, Any]:
        """
        Cổng chặn LỆCH HƯỚNG — bất biến quan trọng nhất của chiến lược này.

        [FIX F25] Ngày 11/09/2026 đo được sổ đang long ròng **+34.95% gross** trong
        khi thiết kế là trung lập. Hệ thống chạy 27 giờ mà không có gì báo động, vì
        không tồn tại phép kiểm nào trên đại lượng này sau khi thực thi.

        Vì sao đây là lỗi giết chiến lược chứ không phải sai số nhỏ: toàn bộ cơ sở
        của cross-sectional market-neutral là KHỬ beta thị trường, nhờ đó tín hiệu
        yếu vẫn cho Sharpe cao. Sổ lệch 35% không còn là chiến lược đã kiểm định —
        nó là một cược có hướng mà không ai chủ ý đặt, và mọi con số backtest mất
        hiệu lực. Đo thực tế: trong 27h thị trường giảm 6.09%, riêng phần lệch hướng
        mất 2.61% equity, trong khi phần chọn cặp lãi.
        """
        val = {s: q * prices.get(s, 0.0) for s, q in positions.items()}
        gross = sum(abs(v) for v in val.values())
        net = sum(val.values())
        ratio = (net / gross) if gross > 1e-9 else 0.0

        out = {
            "gross": gross, "net": net, "net_ratio": ratio,
            "tolerance": tolerance,
            "net_ok": abs(ratio) <= tolerance,
            "long_notional": sum(v for v in val.values() if v > 0),
            "short_notional": sum(v for v in val.values() if v < 0),
            "gross_ok": True, "leverage": None, "target_leverage": target_leverage,
        }

        # [FIX F29] KIỂM CẢ ĐÒN BẨY GỘP, không chỉ độ lệch hướng.
        # Sự cố 11/09/2026: lỗi báo giá lại (F27) làm mọi vị thế nhân đôi, danh mục
        # chạy 3,69x thay vì 2,0x. Cổng chặn lúc đó CHỈ kiểm net nên bắt được lệch
        # +6,4% mà hoàn toàn không thấy rủi ro đã gần gấp đôi. Một sổ nhân đôi vẫn
        # có thể trung lập hoàn hảo — hai đại lượng này độc lập nhau và phải kiểm
        # riêng.
        if equity and equity > 0 and target_leverage:
            lev = gross / equity
            out["leverage"] = lev
            out["gross_ok"] = abs(lev - target_leverage) <= gross_tolerance * target_leverage

        out["ok"] = out["net_ok"] and out["gross_ok"]
        return out

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

    def _position_multiplier(self, state: LiveState, equity: float,
                             now_ms: Optional[int] = None) -> tuple[float, Dict[str, Any]]:
        """
        Hệ số nhân áp lên đòn bẩy gộp, kèm lý do đầy đủ để kiểm toán sau sự cố.

        Hai van chạy song song và ta lấy van CHẶT HƠN:

          * van drawdown (cũ)  — cắt nửa vị thế khi sụt quá TIER1_DD. Đây là van an
            toàn vận hành: nó không biết gì về mục tiêu, chỉ biết hệ thống đang xấu.
          * tầng phủ mục tiêu  — tra chính sách DP đã giải sẵn (`goal_overlay.py`).

        LẤY MIN CHỨ KHÔNG NHÂN, và đó là lựa chọn có giá phải trả. DP đã tính ngưỡng
        cháy vào bài toán rồi, nên chồng thêm van drawdown là phạt hai lần và làm chính
        sách kém tối ưu hơn con số đo trong nghiên cứu. Đổi lại ta có một bảo đảm cứng:
        bật tầng phủ KHÔNG BAO GIỜ làm hệ thống liều hơn hành vi hiện tại ở cùng mức
        drawdown. Với đường ống mới đạt 1/3 lượt sạch, bảo đảm đó đáng giá hơn phần tối
        ưu bị mất.
        """
        dd_mult = 0.5 if state.drawdown(equity) >= TIER1_DD else 1.0
        info: Dict[str, Any] = {"dd_multiplier": dd_mult, "goal_overlay": None}

        cfg = self.config.goal_overlay
        if cfg is None or not cfg.enabled:
            return dd_mult, info

        if self._goal_overlay is None:
            try:
                self._goal_overlay = GoalOverlay.from_config(cfg)
            except FileNotFoundError as exc:
                # Thiếu artifact chính sách là lỗi CẤU HÌNH, không phải lý do để im lặng
                # chạy tiếp với hành vi khác điều người vận hành nghĩ mình đã bật.
                logger.error("Tầng phủ mục tiêu bật nhưng thiếu chính sách: %s", exc)
                info["goal_overlay"] = {"error": str(exc)}
                return dd_mult, info

        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        g = self._goal_overlay.multiplier(
            equity=equity, now_ms=now_ms, base_leverage=self.config.leverage,
            current_leverage=self.config.leverage * dd_mult,
            period_hours=interval_hours(self.config.interval))
        info["goal_overlay"] = g

        mult = min(dd_mult, float(g["multiplier"]))
        logger.info("Hệ số vị thế %.2f (drawdown %.2f, mục tiêu %.2f) — %s",
                    mult, dd_mult, g["multiplier"], g["reason"])
        return mult, info

    # ------------------------------------------------------------------ vòng lặp
    def run_once(self, skip_data_refresh: bool = False, force: bool = False) -> Dict[str, Any]:
        """
        Chạy một lượt. Nếu CHƯA đến hạn tái cân bằng thì chỉ giám sát, không giao dịch.

        `force=True` bỏ qua chốt nhịp — chỉ dùng khi con người chủ động can thiệp.
        """
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

        # --- 2b. CHỐT NHỊP TÁI CÂN BẰNG [FIX F30] ---
        # Chốt này trước đây CHỈ tồn tại trong chế độ daemon (`--loop`), không có ở
        # đường chạy một-lượt mà `crontab` đang dùng. Hệ quả: cron chạy 07:00 hằng
        # ngày sẽ tái cân bằng mỗi 24h, trong khi chiến lược được kiểm định ở chu kỳ
        # 72h. Đó không phải "chạy dày hơn một chút" — tín hiệu được tính trên lưới
        # 18 nến còn vị thế lại bị đặt lại mỗi ngày, nên nhịp tín hiệu và nhịp thực
        # thi lệch nhau, và cấu hình đang chạy KHÔNG phải cấu hình nào đã kiểm định.
        hours_since = (now_ms - state.last_rebalance_ms) / 3_600_000.0
        due = force or state.rebalance_count == 0 or hours_since >= self.config.rebalance_hours

        # --- Cò GIẢM RỦI RO NGOÀI CHU KỲ (tầng phủ mục tiêu) -------------------
        # Chính sách DP chỉ có giá trị nếu hành động kịp lúc. Luật "đạt đích thì đóng
        # sạch" mà phải chờ tới mốc 72h kế tiếp thì có thể muộn gần ba ngày — và trong
        # ba ngày đó khoản lãi vừa chạm đích hoàn toàn có thể bốc hơi. Đo được: biến
        # động 4h của chiến lược là 0,90%, tức 72h là ~3,7% — lớn hơn cả mục tiêu 5%
        # theo bất kỳ nghĩa nào đáng quan tâm.
        #
        # Vì vậy khi tầng phủ đòi hệ số 0 (đạt đích hoặc hết hạn) mà sổ vẫn còn vị thế,
        # ta MỞ chốt nhịp ngay trong lượt giám sát này.
        #
        # BẤT ĐỐI XỨNG LÀ CÓ CHỦ Ý: cò này chỉ kích hoạt theo hướng ĐÓNG vị thế, không
        # bao giờ theo hướng mở thêm. Tái cân bằng ngoài nhịp để GIẢM rủi ro thì tệ
        # nhất cũng chỉ tốn phí đóng; tái cân bằng ngoài nhịp để TĂNG rủi ro sẽ làm
        # nhịp tín hiệu (18 nến) lệch khỏi nhịp thực thi — đúng thứ mà ghi chú ngay
        # phía trên cảnh báo, và đúng thứ biến "cấu hình đang chạy" thành một cấu hình
        # chưa từng được kiểm định.
        derisk_now = False
        if not due and state.positions and self.config.goal_overlay is not None:
            mult_probe, probe_info = self._position_multiplier(state, equity, now_ms)
            result["multiplier_probe"] = probe_info
            if mult_probe <= 1e-9:
                derisk_now, due = True, True
                g = (probe_info.get("goal_overlay") or {})
                logger.warning("GIẢM RỦI RO NGOÀI CHU KỲ sau %.1fh (chu kỳ %.0fh) — %s",
                               hours_since, self.config.rebalance_hours,
                               g.get("reason", "tầng phủ mục tiêu yêu cầu"))
                result["derisk_out_of_cycle"] = True

        if not due:
            prices_hb = {}
            for symbol in state.positions:
                try:
                    bid, ask = self.client.best_bid_ask(symbol)
                    if bid > 0 and ask > 0:
                        prices_hb[symbol] = (bid + ask) / 2.0
                except Exception as exc:
                    logger.warning("Không lấy được BBO %s: %s", symbol, exc)

            nt = self.check_neutrality(
                state.positions, prices_hb, tolerance=self.config.max_net_exposure,
                equity=equity, target_leverage=self.config.leverage,
                gross_tolerance=self.config.max_gross_error)
            self.store.save(state)
            result.update(action="HEARTBEAT", reason="CHUA_DEN_HAN",
                          hours_since_rebalance=round(hours_since, 2),
                          rebalance_hours=self.config.rebalance_hours,
                          hours_remaining=round(self.config.rebalance_hours - hours_since, 2),
                          n_positions=len(state.positions), neutrality=nt,
                          drawdown=state.drawdown(equity))
            if not nt["ok"]:
                logger.error("[F25/F29] Giám sát phát hiện sổ lệch thiết kế: net "
                             "%+.2f%% gross, đòn bẩy %s", nt["net_ratio"] * 100,
                             f"{nt['leverage']:.2f}x" if nt["leverage"] else "n/a")
            return result

        # --- 3. Dữ liệu ---
        # ĐƯỜNG GIẢM RỦI RO KHÔNG ĐI QUA TẦNG DỮ LIỆU.
        #
        # Khi tầng phủ mục tiêu đòi hệ số 0, trọng số đích đã biết trước và bằng RỖNG —
        # đóng sạch. Không có tín hiệu nào cần tính, không có universe nào cần lọc,
        # không có panel nào cần nạp.
        #
        # Đây không phải tối ưu tốc độ mà là tối ưu ĐỘ TIN CẬY. Chốt lời phải là thao
        # tác đáng tin cậy nhất trong hệ thống, vì nó là thao tác duy nhất biến lãi
        # trên giấy thành lãi đã thực hiện. Bắt nó đi qua `compute_target_weights` là
        # gắn nó vào dữ liệu mới, vào bộ lọc universe, vào toàn bộ những thứ có thể
        # hỏng — và chúng có thể hỏng, đã hỏng: universe sau lọc còn 0 cặp thì hàm đó
        # NÉM LỖI chứ không trả về trạng thái dừng. Lúc đó vị thế vẫn mở nguyên.
        if derisk_now:
            weights, latest_ts = {}, now_ms
            result["data_age_hours"] = None
            logger.info("Đường giảm rủi ro: bỏ qua tầng dữ liệu, trọng số đích = rỗng")
        else:
            if not skip_data_refresh:
                result["symbols_updated"] = self.refresh_data(days=30.0)

            weights, latest_ts = self.compute_target_weights(
                current_positions=exchange_pos, equity=equity)
            age_hours = (now_ms - latest_ts) / 3_600_000
            result["data_age_hours"] = round(age_hours, 2)
            if age_hours > self.config.max_data_age_hours:
                result.update(action="HALT", reason="STALE_DATA",
                              detail=f"nến mới nhất đã {age_hours:.1f}h tuổi "
                                     f"(trần {self.config.max_data_age_hours}h)")
                self.store.save(state)
                return result

            # [FIX F52] Nến tươi KHÔNG có nghĩa funding tươi — hai nguồn, hai đường tải.
            # Cổng STALE_DATA ở trên chưa từng nhìn funding, nên 14 ngày funding đứng yên
            # lọt qua mọi lượt. Đo ở NGUỒN (mốc cuối trong file), không đo trên panel đã
            # ffill. Chỉ áp cho v3: đó là engine daemon chạy và là nơi họ carry quyết định.
            if self.config.engine == "v3":
                last = funding_last_ms(list(self.config.universe))
                f_age = (now_ms - last) / 3_600_000
                med = float(f_age.median()) if f_age.notna().any() else float("inf")
                result["funding_age_hours"] = round(med, 2) if np.isfinite(med) else None
                if not med <= self.config.max_funding_age_hours:
                    n_old = int((f_age > self.config.max_funding_age_hours).sum()
                                + f_age.isna().sum())
                    result.update(action="HALT", reason="STALE_FUNDING",
                                  detail=f"funding trung vị đã {med:.1f}h tuổi "
                                         f"(trần {self.config.max_funding_age_hours}h), "
                                         f"{n_old}/{len(f_age)} cặp quá hạn hoặc thiếu")
                    self.store.save(state)
                    return result

        # --- 4. Sức chứa vốn ---
        cap = max_positions_for_capital(equity, self.config.leverage,
                                        min_notional=5.0, safety=self.config.min_notional_safety)
        if len(weights) > cap:
            # [FIX F43] Cắt phải CÂN HAI CHÂN. Bản cũ lấy top-N theo |trọng số| bất kể
            # dấu — chú thích ghi "ở cả hai chiều" nhưng code không hề làm thế.
            #
            # Vì sao chưa từng cắn: testnet có $6.226 nên `cap` luôn >> 12. Nó sẽ cắn
            # đúng lúc chạm tài khoản thật. Với vốn $38 ở 2x, `cap` = 12 — vừa khít.
            # Sụt 8% xuống $35 là `cap` = 11, và cắt lẻ thì hai chân không thể bằng nhau.
            #
            # Mô phỏng 20.000 lượt với độ mạnh tín hiệu ngẫu nhiên:
            #   vốn $35 (cap 11): 100% lượt vi phạm trần, net 9,1%
            #   vốn $28 (cap  9): 100% lượt vi phạm, xấu nhất 33,3%
            #   vốn $20 (cap  6):  57% lượt vi phạm, xấu nhất **100%**
            # Net 100% nghĩa là danh mục MỘT CHIỀU hoàn toàn ở đòn bẩy 2x — đúng thứ
            # chiến lược này được thiết kế để không bao giờ làm. DD kỳ vọng 24,7%/năm
            # nên vốn $38 chạm $35 là chuyện gần như chắc chắn, không phải biên hiếm.
            longs = sorted(((s_, w_) for s_, w_ in weights.items() if w_ > 0),
                           key=lambda kv: -kv[1])
            shorts = sorted(((s_, w_) for s_, w_ in weights.items() if w_ < 0),
                            key=lambda kv: kv[1])
            per_side = min(cap // 2, len(longs), len(shorts))
            if per_side <= 0:
                # Không đủ vốn cho dù MỘT cặp mỗi chân. Giao dịch một chiều còn tệ hơn
                # không giao dịch: nó biến quỹ market-neutral thành cược hướng có đòn bẩy.
                result.update(action="HALT", reason="INSUFFICIENT_CAPITAL",
                              detail=f"vốn ${equity:,.2f} ở {self.config.leverage}x chỉ đủ "
                                     f"{cap} vị thế — không đủ 1 cặp mỗi chân, "
                                     f"không thể giữ trung lập")
                logger.error("[F43] %s", result["detail"])
                self.store.save(state)
                return result
            n_before = len(weights)
            kept_l = longs[:per_side]
            kept_s = shorts[:per_side]

            # Cân SỐ CẶP thôi chưa đủ: hai chân bằng số cặp vẫn có thể khác tổng
            # notional, vì trọng số theo hạng có độ lớn khác nhau. Co giãn mỗi chân
            # về đúng nửa gross -> net = 0 CHÍNH XÁC.
            #
            # Cố tình KHÔNG dùng `project_neutral`: phép chiếu trực giao trừ đi trung
            # bình nên có thể ĐẢO DẤU một trọng số nhỏ, biến một cặp long thành short
            # ngược với tín hiệu sinh ra nó. Co giãn theo chân giữ nguyên mọi dấu và
            # mọi thứ hạng trong chân — can thiệp nhỏ nhất đạt đúng mục tiêu.
            sum_l = sum(w_ for _, w_ in kept_l)
            sum_s = -sum(w_ for _, w_ in kept_s)
            if sum_l <= 0 or sum_s <= 0:
                result.update(action="HALT", reason="DEGENERATE_WEIGHTS",
                              detail=f"sau khi cắt: tổng long {sum_l:.6f}, tổng short "
                                     f"{sum_s:.6f} — không dựng được sổ trung lập")
                logger.error("[F43] %s", result["detail"])
                self.store.save(state)
                return result
            half = (sum_l + sum_s) / 2.0
            weights = {s_: w_ * (half / sum_l) for s_, w_ in kept_l}
            weights.update({s_: w_ * (half / sum_s) for s_, w_ in kept_s})

            net_after = sum(weights.values())
            gross_after = sum(abs(w_) for w_ in weights.values())
            logger.warning("[F43] Vốn chỉ đủ %d vị thế — cắt %d xuống %d "
                           "(%d long / %d short) | net sau cắt %+.2e gross",
                           cap, n_before, len(weights), per_side, per_side,
                           net_after / gross_after if gross_after else 0.0)
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

        multiplier, mult_info = self._position_multiplier(state, equity, now_ms)
        result["multiplier_detail"] = mult_info

        # `build_rebalance_plan` từ chối leverage <= 0 — đúng, vì gross notional của
        # một danh mục CÓ vị thế mà bằng 0 thì vô nghĩa. Nhưng hệ số 0 ở đây nghĩa là
        # "đóng sạch", và lúc đó trọng số đích RỖNG nên chẳng có gì để định cỡ: mọi
        # giá trị leverage dương đều sinh ra đúng một kế hoạch — đóng hết.
        #
        # Bắt lỗi này bằng test tích hợp chứ không phải khi vận hành là điều may: nếu
        # để nguyên, lệnh chốt lời sẽ NÉM LỖI đúng vào lúc nó cần chạy nhất, và vị thế
        # ở lại nguyên trên sàn.
        plan_leverage = self.config.leverage * multiplier
        if plan_leverage <= 1e-9:
            if weights:
                raise RuntimeError(
                    f"Hệ số vị thế {multiplier} nhưng vẫn có {len(weights)} trọng số đích — "
                    f"mâu thuẫn, không đoán ý định.")
            plan_leverage = self.config.leverage      # danh nghĩa; không có gì để định cỡ

        plan: RebalancePlan = build_rebalance_plan(
            target_weights=weights,
            current_qty=exchange_pos,
            prices=prices,
            filters=self.load_filters(),
            equity=equity,
            leverage=plan_leverage,
            # [FIX F42] Trần trung lập phải là CÙNG MỘT con số mà cổng F25 dùng để
            # chấm sau khi thực thi. Trước đây kế hoạch hoàn toàn mù về trung lập:
            # dải không giao dịch phá net exposure, và mãi tới bước 6b — khi lệnh ĐÃ
            # nằm trên sàn — mới có thứ đo nó. Chặn trước rẻ hơn hét sau.
            neutrality_tolerance=self.config.max_net_exposure,
        )
        result.update(n_orders=len(plan.orders), turnover=plan.total_turnover,
                      gross_notional=plan.gross_notional, skipped=len(plan.skipped),
                      position_multiplier=multiplier)

        orders_list = [
            {"symbol": o.symbol, "side": o.side, "qty": o.qty,
             "price": o.price, "notional": round(o.notional, 2), "reason": o.reason}
            for o in plan.orders
        ]
        result["orders"] = orders_list
        result["equity"] = equity

        if self.dry_run:
            result.update(action="DRY_RUN")
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

        # [FIX F24] ĐỒNG BỘ SỔ TỪ SÀN PHẢI LUÔN CHẠY, kể cả khi thực thi ném lỗi.
        # Sự cố 10/09/2026: một ngoại lệ giữa chừng khiến bước lưu trạng thái không
        # bao giờ tới được, dù lệnh ĐÃ khớp trên sàn. Kết quả là sổ nội bộ đứng yên
        # ở lượt trước trong khi sàn đã đổi — và hệ thống tự khoá 27 giờ sau đó.
        # Sổ nội bộ không bao giờ được phép "cũ hơn" sàn.
        exec_report: Dict[str, Any] = {}
        exec_error: Optional[str] = None
        try:
            exec_report = self.router.execute_with_fallback(
                plan.orders, filters,
                passive_wait_s=self.config.passive_wait_s,
                allow_taker_fallback=self.config.allow_taker_fallback,
                requote=self.config.requote,
                max_chase_bps=self.config.max_chase_bps,
                chase_caps=self._chase_caps(plan.orders),   # [FIX F44]
                # [FIX F45] Phải truyền: bỏ trống là van dùng mặc định của router
                # thay vì cấu hình đã kiểm định. Đo trên gross KẾ HOẠCH.
                max_fill_imbalance=self.config.max_fill_imbalance,
            )
        except Exception as exc:
            logger.exception("[F24] Thực thi lỗi — vẫn đồng bộ sổ từ sàn")
            exec_error = f"{type(exc).__name__}: {exc}"
        finally:
            state.positions = self._settle_positions()
            state.last_rebalance_ms = now_ms
            state.rebalance_count += 1
            state.last_error = exec_error
            self.store.save(state)

        exec_report.setdefault("passive_submitted", 0)
        exec_report.setdefault("passive_filled_notional", 0.0)
        exec_report.setdefault("taker_orders", 0)
        exec_report.setdefault("taker_notional", 0.0)
        exec_report.setdefault("unfilled", [])
        exec_report.setdefault("plan_failures", [])
        exec_report.setdefault("uncancelled", [])

        result["execution"] = exec_report
        result["submitted"] = exec_report["passive_submitted"]
        result["taker_fallback"] = exec_report["taker_orders"]
        result["unfilled"] = len(exec_report["unfilled"])
        result["plan_failures"] = exec_report["plan_failures"]
        result["uncancelled"] = exec_report["uncancelled"]
        if exec_error:
            result["exec_error"] = exec_error

        # --- 6b. CỔNG CHẶN LỆCH HƯỚNG [FIX F25] ---
        # Chạy SAU khi đã đồng bộ sổ, trên vị thế THẬT của sàn. Không sửa được ở
        # lượt này thì ít nhất phải hét lên — im lặng là chế độ hỏng tệ nhất.
        neutrality = self.check_neutrality(
            state.positions, prices, tolerance=self.config.max_net_exposure,
            equity=equity, target_leverage=self.config.leverage * multiplier,
            gross_tolerance=self.config.max_gross_error)
        result["neutrality"] = neutrality
        if not neutrality["ok"]:
            parts = []
            if not neutrality["net_ok"]:
                parts.append(
                    f"LỆCH HƯỚNG {neutrality['net_ratio']*100:+.1f}% gross "
                    f"(trần ±{neutrality['tolerance']*100:.0f}%) — "
                    f"long ${neutrality['long_notional']:,.0f} vs "
                    f"short ${abs(neutrality['short_notional']):,.0f}")
            if not neutrality["gross_ok"]:
                parts.append(
                    f"ĐÒN BẨY SAI {neutrality['leverage']:.2f}x so với mục tiêu "
                    f"{neutrality['target_leverage']:.2f}x — rủi ro không đúng dự kiến")
            msg = " | ".join(parts) + ". Danh mục KHÔNG đúng thiết kế; mọi con số kiểm định mất hiệu lực."
            logger.error("[F25] %s", msg)
            state.last_error = (state.last_error + " | " if state.last_error else "") + msg
            self.store.save(state)
            result["action"] = "NEUTRALITY_BREACH"
            result["neutrality_error"] = msg
            if self.notifier_enabled():
                self.router.notifier.send_anomaly_alert(
                    title="Danh mục lệch khỏi trung lập", message=msg, level="ERROR")

        if exec_report["plan_failures"]:
            logger.error("[F22] %d lệnh không đặt được: %s", len(exec_report["plan_failures"]),
                         ", ".join(f["symbol"] for f in exec_report["plan_failures"]))
        if exec_report["uncancelled"]:
            logger.error("[F21] Còn lệnh sống không xác nhận huỷ được: %s",
                         ", ".join(exec_report["uncancelled"]))

        # Ghi nhật ký chi phí THẬT để về sau đối chiếu với giả định backtest.
        gross_live = sum(abs(q) * prices.get(sym, 0.0) for sym, q in state.positions.items())
        net_live = sum(q * prices.get(sym, 0.0) for sym, q in state.positions.items())
        self.exec_log.append(ExecutionRecord(
            timestamp_ms=now_ms, equity=equity, n_targets=len(weights),
            n_orders=len(plan.orders), gross_notional=gross_live,
            planned_turnover=plan.total_turnover,
            # [FIX F46] Chụp lại sổ kế hoạch cạnh sổ thực tế. Hai con số này cạnh nhau
            # là thứ biến "danh mục lệch" từ một bí ẩn thành một phép trừ.
            plan_net_exposure=(plan.net_notional / plan.gross_notional
                               if plan.gross_notional > 0 else 0.0),
            maker_notional=float(exec_report["passive_filled_notional"]),
            taker_notional=float(exec_report["taker_notional"]),
            unfilled_count=len(exec_report["unfilled"]),
            net_exposure=(net_live / gross_live) if gross_live > 0 else 0.0,
            testnet=bool(getattr(self.client.credentials, "testnet", True)),
            plan_failures=len(exec_report["plan_failures"]),
            uncancelled=len(exec_report["uncancelled"]),
            leverage=(gross_live / equity) if equity > 0 else None,
            target_leverage=self.config.leverage * multiplier,
            net_ok=bool(neutrality["net_ok"]),
            gross_ok=bool(neutrality["gross_ok"]),
            exec_error=exec_error,
            # Chẩn đoán tỷ lệ maker — xem ghi chú trong `core/execution_log.py`.
            passive_timed_out=int(exec_report.get("passive_timed_out", 0)),
            requote_blocked_by_chase_cap=int(
                exec_report.get("requote_blocked_by_chase_cap", 0)),
            requote_skipped_no_move=int(exec_report.get("requote_skipped_no_move", 0)),
            requotes=int(exec_report.get("requotes", 0)),
            max_drift_bps_seen=float(exec_report.get("max_drift_bps_seen", 0.0)),
            # [FIX F49] Phân vị của mức VƯỢT TRẦN, để lần chỉnh trần tới là phép đo
            # chứ không phải phỏng đoán.
            chase_block_p50_ratio=_pct(exec_report.get("chase_block_ratios"), 50),
            chase_block_p90_ratio=_pct(exec_report.get("chase_block_ratios"), 90),
            fill_drift=float((exec_report.get("fill_imbalance") or {}).get("drift", 0.0)),
            # [FIX F45] Hai trường này là thứ đã THIẾU khi chẩn đoán lượt 19/09: bản
            # ghi có `fill_drift` nhưng không có tiến độ lúc van bắn, nên không thể
            # phân biệt "lệch thật" với "mới khớp được 2% nên tỷ lệ nhiễu".
            fill_drift_plan=float(
                (exec_report.get("fill_imbalance") or {}).get("drift_plan", 0.0)),
            fill_progress=float(
                (exec_report.get("fill_imbalance") or {}).get("progress", 0.0)),
            early_exit=exec_report.get("early_exit"),
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
