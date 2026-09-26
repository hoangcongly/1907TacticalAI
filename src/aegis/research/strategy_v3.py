"""
Chiến lược v3 — CẤU HÌNH CHỐT + đường chạy nghiên cứu dùng lại được.

VÌ SAO MODULE NÀY TỒN TẠI: cấu hình v3 đã chốt nằm rải trong `scripts/validate_v3.py`
dưới dạng hằng số module. Kết quả là mọi phân tích downstream (đòn bẩy, mục tiêu,
rủi ro) phải CHÉP LẠI cấu hình đó — và chép lại là cách lỗi F3 tái sinh: hai bản sao
của cùng một sự thật thì sớm muộn cũng lệch nhau.

Ở đây cấu hình được khai báo MỘT LẦN (`V3`) và đường chạy được viết MỘT LẦN
(`run_v3`). `scripts/validate_v3.py` giữ nguyên không đụng tới — nó là script chỉ
được chạy một lần trên holdout, và `tests/research/test_strategy_v3_parity.py` khoá
bất biến rằng module này tái tạo ĐÚNG các con số nó đã ghi ra.

⚠️ BẪY ĐO LƯỜNG ĐÃ TRẢ GIÁ MỘT LẦN (Sharpe 0.71 sai so với 1.28 đúng):
tầng gộp thích ứng (`adaptive_combiner.adaptive_weights`) phụ thuộc ĐƯỜNG ĐI — nó
đi tới từ chỉ số 0 với `prev = 0` và trần `max_step` mỗi kỳ. Cắt lưới về holdout
TRƯỚC khi tính sẽ khởi động lại tầng này từ con số 0: 120 kỳ đầu chạy trọng số đều
và cửa sổ học không bao giờ đầy. Live KHÔNG gặp handicap đó — live luôn có toàn bộ
lịch sử trong tay.

Quy tắc là: TÍNH TRÊN TOÀN DÒNG THỜI GIAN, CẮT SAU. `run_v3` cưỡng chế thứ tự đó;
không có tham số nào đảo ngược được nó.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from aegis.research.adaptive_combiner import CombinerSpec, combine_adaptive
from aegis.research.backtest_v2 import (
    BacktestV2Result, CostModel, estimate_cost_bps, simulate,
    simulate_marked_to_market,
)
from aegis.research.signal_library import SIGNAL_REGISTRY, build_signal
from aegis.risk.portfolio import PortfolioSpec, build_weights

__all__ = ["StrategyV3Config", "V3", "V3_SIGNALS", "V3_VINTAGE_MS", "V3Data", "load_v3_data",
           "combined_signal", "run_v3", "run_v3_fine", "split_train_holdout",
           "WIDE_CONFIG_FILE", "config_from_json", "run_v3_tranched"]


# ---------------------------------------------------------------------------
# Cấu hình chốt
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyV3Config:
    """
    Cấu hình v3 ĐÃ CHỐT trên tập train qua `scripts/stability.py` (chọn theo ĐỘ ỔN
    ĐỊNH qua 5 giai đoạn con, không theo Sharpe cao nhất).

    `frozen=True` là cố ý: mọi thay đổi tham số ở đây sau khi holdout đã chạy đều
    biến holdout thành train. Muốn thử cấu hình khác thì tạo instance mới và nói rõ
    đó là thí nghiệm trên train, đừng sửa `V3`.
    """

    #: Danh sách tín hiệu. `None` = ĐÚNG 26 tín hiệu của bản đã kiểm định (`V3_SIGNALS`).
    #:
    #: VÌ SAO PHẢI GHIM CHỨ KHÔNG DÙNG CẢ REGISTRY: `adaptive_weights` chuẩn hoá trọng
    #: số theo SỐ HỌ đang có (`equal = 1/n`) và chia `raw / total`. Thêm một họ mới —
    #: kể cả họ toàn NaN không bao giờ được trọng số — vẫn đổi `n`, và vì thế đổi trọng
    #: số của giai đoạn warm-up. Kết quả v3 sẽ dịch đi mà không ai hiểu vì sao.
    #:
    #: Vậy nên thêm tín hiệu vào registry KHÔNG được phép âm thầm đổi V3. Muốn thử bộ
    #: tín hiệu rộng hơn thì tạo cấu hình MỚI và so sánh tường minh.
    signals: Optional[tuple] = None
    universe_file: str = "artifacts/universe_wide.json"
    interval: str = "4h"
    source_interval: str = "1h"
    rebalance_every: int = 18        # 18 nến 4h = 72h giữa hai lần tái cân bằng
    n_positions: int = 12            # 6 long + 6 short — RÀNG BUỘC CỨNG, không phải top_frac
    top_frac: float = 0.10           # dùng cho tầng gộp (lợi suất nhân tố), không phải sizing
    maker_ratio: float = 0.50        # giả định BI QUAN; testnet đo 0.378, mục tiêu 0.85
    min_coverage: float = 0.15
    capital_usd: float = 38.0        # 1.000.000 VND
    min_notional_usd: float = 5.0
    gross_leverage: float = 2.0      # đòn bẩy gộp đang chạy trên testnet
    #: Đòn bẩy dùng khi ƯỚC LƯỢNG chi phí trượt giá. KHÁC `gross_leverage` một cách
    #: có chủ ý: `validate_v3.py` ước lượng ở 3x ($9.50/lệnh) trong khi hệ thống chạy
    #: 2x ($6.33/lệnh). Giữ 3.0 ở đây để tái lập được con số đã ghi; chênh lệch thật
    #: chỉ nằm ở số hạng tác động giá (~0.01-0.1bp) nên không đổi kết luận nào, nhưng
    #: nếu im lặng sửa thành 2.0 thì parity gãy mà không ai biết vì sao.
    cost_notional_leverage: float = 3.0
    num_trials_declared: int = 250
    #: Số lô tái cân bằng lệch pha (`research/tranching.py`). 1 = một sổ đổi toàn bộ mỗi
    #: `rebalance_every` nến — mọi con số đã kiểm định đều ở chế độ này.
    n_tranches: int = 1

    combiner: CombinerSpec = field(default_factory=lambda: CombinerSpec(
        lookback=500, min_periods=120, t_threshold=2.0,
        max_abs_weight=0.20, max_step=0.05))
    portfolio: PortfolioSpec = field(default_factory=lambda: PortfolioSpec(
        mode="zscore_riskparity", n_positions=12, max_weight=0.20,
        beta_neutral=False, vol_window=60))

    # -- đại lượng dẫn xuất, không phải tham số tự do ------------------------
    @property
    def bar_hours(self) -> float:
        from aegis.data.panel_v2 import interval_hours
        return interval_hours(self.interval)

    @property
    def period_hours(self) -> float:
        return self.bar_hours * self.rebalance_every

    @property
    def periods_per_year(self) -> float:
        return 24 * 365.0 / self.period_hours

    @property
    def periods_per_week(self) -> float:
        return 7 * 24.0 / self.period_hours

    def signal_names(self) -> tuple:
        """Danh sách tín hiệu thực dùng. `None` -> bộ 26 đã kiểm định."""
        return tuple(self.signals) if self.signals is not None else V3_SIGNALS

    def notional_per_order(self, leverage: Optional[float] = None) -> float:
        """Notional mỗi lệnh ở đòn bẩy cho trước — cái quyết định có vượt min notional không."""
        lev = self.gross_leverage if leverage is None else leverage
        return self.capital_usd * lev / self.n_positions

    def max_positions_affordable(self, leverage: Optional[float] = None) -> int:
        """
        Số vị thế TỐI ĐA mua được ở đòn bẩy cho trước.

        Đây là ràng buộc VỐN chứ không phải ràng buộc thuật toán, và nó là lý do
        đòn bẩy ở tài khoản nhỏ mua được ĐỘ RỘNG: $38 ở 2x chỉ nuôi nổi 15 vị thế
        $5, ở 4x nuôi được 30. Độ rộng hạ biến động danh mục mà không đụng tới
        chất lượng tín hiệu.
        """
        lev = self.gross_leverage if leverage is None else leverage
        return int(self.capital_usd * lev // self.min_notional_usd)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.update(period_hours=self.period_hours,
                 periods_per_year=self.periods_per_year,
                 periods_per_week=self.periods_per_week)
        return d


#: 26 tín hiệu của bản đã kiểm định holdout. ĐỪNG THÊM BỚT — xem `StrategyV3Config.signals`.
V3_SIGNALS: tuple = (
    "carry_level", "carry_level_slow", "carry_zscore", "carry_momentum", "carry_vol",
    "mom_fast", "mom_mid", "mom_slow", "mom_vlong", "mom_risk_adj", "mom_consistency",
    "mom_52w_high",
    "ofi_fast", "ofi_mid", "ofi_slow", "ofi_persistence", "volume_shock", "amihud_illiq",
    "low_idio_vol", "neg_skew", "low_max_ret", "vol_of_vol", "downside_beta",
    "close_location", "range_compression", "trade_intensity",
)

#: Cấu hình đang chạy. ĐỪNG SỬA — xem docstring của `StrategyV3Config`.
V3 = StrategyV3Config()

#: File cấu hình daemon đang chạy (`run_daily.py --config ...`), 50 vị thế.
WIDE_CONFIG_FILE = "artifacts/strategy_v3_wide.json"


def config_from_json(path: str = WIDE_CONFIG_FILE) -> StrategyV3Config:
    """
    Dựng cấu hình NGHIÊN CỨU từ ĐÚNG file JSON mà live nạp, với ĐÚNG phép ánh xạ của
    `LiveConfig.from_artifacts` — hai phía đọc cùng một nguồn sự thật.

    [FIX F51] Các nghiên cứu về cấu hình wide (`breadth_beyond_50`, `sizing_mode_study`,
    `maker_ratio_study`, `leverage_study`...) tự dựng `PortfolioSpec` cho n=50 nhưng
    GIỮ tầng gộp của `V3` (`max_step=0,05`). Trong khi đó `strategy_v3_wide.json` —
    file daemon chạy — ghi `max_step=0,025`. Mọi Sharpe 2,23 báo cho cấu hình wide vì
    thế đo một cấu hình KHÁC thứ đang đặt lệnh. Test parity cấu hình cũ chỉ kiểm
    `strategy_v3.json` (n=12), file daemon không còn chạy, nên khe hở này lọt qua —
    cùng họ F38: công thức thì khoá, cấu hình thì không.
    """
    import json
    from dataclasses import replace

    inner = json.load(open(path, encoding="utf-8"))
    inner = inner.get("config", inner)
    comb, port = inner.get("combiner", {}), inner.get("portfolio", {})
    n = int(inner.get("n_positions", V3.n_positions))
    return replace(
        V3,
        universe_file=inner.get("universe_file", V3.universe_file),
        interval=inner.get("interval", V3.interval),
        rebalance_every=int(inner.get("rebalance_every", V3.rebalance_every)),
        n_positions=n,
        n_tranches=int(inner.get("n_tranches", 1)),
        maker_ratio=float(inner.get("maker_ratio_assumed", V3.maker_ratio)),
        combiner=replace(
            V3.combiner,
            lookback=int(comb.get("lookback", 500)),
            min_periods=int(comb.get("min_periods", 120)),
            t_threshold=float(comb.get("t_threshold", 2.0)),
            max_abs_weight=float(comb.get("max_abs_weight", 0.20)),
            max_step=float(comb.get("max_step", 0.05))),
        # Y HỆT `xs_live_pipeline._compute_target_weights_v3`: live dựng PortfolioSpec
        # với `beta_neutral=False` cố định và `vol_window` mặc định, BỎ QUA hai trường
        # đó trong JSON; `rank_binary` thì trần 1,0. Đọc JSON "đúng hơn" live ở đây là
        # đo một cấu hình live không chạy.
        portfolio=PortfolioSpec(
            mode=port.get("mode", "zscore_riskparity"),
            n_positions=n,
            max_weight=(1.0 if port.get("mode") == "rank_binary"
                        else float(port.get("max_weight", 0.20))),
            beta_neutral=False),
    )

#: MỐC DỮ LIỆU mà `artifacts/strategy_v3.json` được sinh ra (nến 4h cuối cùng nó thấy).
#:
#: VÌ SAO PHẢI GHIM: `load_panel_v2` lọc universe bằng `min_coverage` tính TRÊN CHÍNH
#: panel được nạp. Panel dài thêm -> tỷ lệ phủ của mọi cặp đổi -> một cặp mới có thể
#: vượt ngưỡng và vào rổ. Đã xảy ra thật: rổ là 127 cặp tới 12/09 20:00, thành 128 cặp
#: ngay sau đó. Một cặp thêm vào làm đổi THỨ HẠNG của toàn mặt cắt ngang, nên mọi con
#: số — kể cả đoạn train năm 2021 — đều dịch đi.
#:
#: Hệ quả nếu không ghim: chạy lại `validate_v3.py` hôm nay ra Sharpe khác hôm qua, và
#: KHÔNG phân biệt được "code hỏng" với "dữ liệu dài ra". Đó là mất khả năng tái lập.
#:
#: Hệ quả thứ hai, tinh vi hơn: rổ được lọc bằng dữ liệu tới HÔM NAY rồi áp ngược cho
#: quá khứ. Một cặp có mặt trong backtest 2021 vì hôm nay nó đủ phủ — đó là một dạng
#: thiên vị sống sót nhẹ. Ghim mốc không xoá được nó, nhưng làm nó ĐỨNG YÊN và đo được.
V3_VINTAGE_MS = 1789012800000   # 2026-09-10 04:00:00 UTC — mốc cho holdout đúng 219 kỳ


# ---------------------------------------------------------------------------
# Nạp dữ liệu (đắt: ~170 cặp x nến 1h) — tách riêng để tái dùng qua nhiều phân tích
# ---------------------------------------------------------------------------
@dataclass
class V3Data:
    """Panel + funding + tín hiệu đã dựng, đủ để chạy bất kỳ biến thể nào của v3."""

    panel: Dict[str, pd.DataFrame]
    funding: pd.DataFrame
    signals: Dict[str, pd.DataFrame]
    per_symbol_bps: pd.Series
    split_ts: int
    #: Nến cuối cùng thực sự có trong panel — ghi vào mọi artifact để tái lập được.
    data_end_ms: int = 0

    @property
    def close(self) -> pd.DataFrame:
        return self.panel["close"]

    @property
    def index(self) -> pd.Index:
        return self.panel["close"].index


def load_v3_data(cfg: StrategyV3Config = V3, verbose: bool = True,
                 end_ms: Optional[int] = None, with_metrics: bool = False) -> V3Data:
    """
    Nạp panel, funding và dựng toàn bộ tín hiệu TRÊN TOÀN CHUỖI.

    Tín hiệu dựng trên toàn chuỗi là hợp lệ và bắt buộc: mọi tín hiệu trong
    `signal_library` đều nhân quả (chỉ nhìn `<= t`), nên giá trị tại t không đổi
    khi ta thêm hay bớt dữ liệu SAU t. Dựng trước rồi cắt sau giúp không mất vài
    trăm nến warm-up ở đầu mỗi đoạn cần đo.

    `end_ms` cắt panel TRƯỚC khi lọc `min_coverage`, nên nó ghim luôn cả THÀNH PHẦN
    rổ — đó là điều kiện đủ để tái lập một kết quả cũ (xem `V3_VINTAGE_MS`).
    """
    import json

    from aegis.data.panel_v2 import align_panel, load_funding_panel_v2, load_panel_v2

    syms = json.load(open(cfg.universe_file))
    if verbose:
        print(f"nạp panel: {len(syms)} cặp, {cfg.interval} (từ {cfg.source_interval})...", flush=True)
    panel = load_panel_v2(syms, cfg.interval, source_interval=cfg.source_interval,
                          min_coverage=cfg.min_coverage, end_ms=end_ms)
    funding = load_funding_panel_v2(syms, panel["close"].index)
    panel, funding = align_panel(panel, funding)

    if with_metrics:
        # Trường VỊ THẾ (open interest, tỷ lệ long/short) — nguồn riêng, xem
        # `scripts/download_metrics.py`. Thêm vào panel chứ không thay gì cả, nên
        # mọi tín hiệu cũ tính ra y hệt.
        from aegis.data.panel_v2 import load_metrics_panel_v2
        met = load_metrics_panel_v2(list(panel["close"].columns), panel["close"].index,
                                    interval=cfg.interval)
        n_cov = int(met["sum_open_interest_value"].notna().any().sum()) if met else 0
        panel.update(met)
        if verbose:
            print(f"nạp dữ liệu vị thế: {n_cov}/{panel['close'].shape[1]} cặp có metrics",
                  flush=True)

    names = cfg.signal_names()
    if verbose:
        print(f"dựng {len(names)} tín hiệu trên {panel['close'].shape[1]} cặp...", flush=True)
    signals = {n: build_signal(n, panel, funding) for n in names}

    per_sym = estimate_cost_bps(
        panel, notional_usd=cfg.notional_per_order(cfg.cost_notional_leverage),
        bars_per_day=6)

    split_ts = json.load(open("artifacts/holdout_split.json"))["split_ts"]
    return V3Data(panel=panel, funding=funding, signals=signals,
                  per_symbol_bps=per_sym, split_ts=int(split_ts),
                  data_end_ms=int(panel["close"].index[-1]))


# ---------------------------------------------------------------------------
# Đường chạy — TÍNH TOÀN DÒNG THỜI GIAN, CẮT SAU
# ---------------------------------------------------------------------------
def combined_signal(data: V3Data, cfg: StrategyV3Config = V3) -> pd.DataFrame:
    """
    Điểm số tổng hợp trên lưới tái cân bằng — tách riêng vì nó ĐẮT và TÁI DÙNG ĐƯỢC.

    `combine_adaptive` chạy 26 tín hiệu qua tầng gộp thích ứng trên toàn lịch sử: đây
    là ~90% chi phí của một lần chạy v3. Nó KHÔNG phụ thuộc `n_positions`, `max_weight`
    hay bất cứ tham số danh mục nào — những thứ đó chỉ vào ở `build_weights`.

    Tách ra để một nghiên cứu quét 10 cấu hình danh mục trả giá tầng gộp ĐÚNG MỘT LẦN
    thay vì 10 lần. Đo được: quét 7 giá trị `n_positions` giảm từ 112s xuống ~20s.
    """
    marks = data.close.index[::cfg.rebalance_every]

    # [FIX F39] LỌC theo cấu hình, KHÔNG dùng cả `data.signals`.
    #
    # `V3Data` được nạp một lần rồi tái dùng cho nhiều cấu hình — đó là cả điểm mạnh
    # lẫn cái bẫy. Nếu ở đây lấy nguyên `data.signals`, thì so sánh "26 tín hiệu so với
    # 26+10" sẽ chạy 36 tín hiệu ở CẢ HAI phía và cho ra hai con số y hệt nhau. Không
    # crash, không cảnh báo — chỉ là một thí nghiệm không đo cái nó tưởng đang đo.
    #
    # Cùng họ lỗi với F38, chỉ khác là ở tầng nghiên cứu thay vì tầng live.
    names = cfg.signal_names()
    missing = [n for n in names if n not in data.signals]
    if missing:
        raise KeyError(
            f"V3Data thiếu {len(missing)} tín hiệu mà cấu hình yêu cầu: {missing[:5]}. "
            f"Nạp lại bằng `load_v3_data(cfg)` với ĐÚNG cấu hình này.")
    sigs = {n: data.signals[n].reindex(marks) for n in names}
    return combine_adaptive(sigs, data.close.reindex(marks), cfg.combiner,
                            top_frac=cfg.top_frac)


def run_v3(
    data: V3Data,
    cfg: StrategyV3Config = V3,
    mask: Optional[np.ndarray] = None,
    maker_ratio: Optional[float] = None,
    n_positions: Optional[int] = None,
    portfolio: Optional[PortfolioSpec] = None,
    sig: Optional[pd.DataFrame] = None,
    cost_bps: Optional[float] = None,
) -> BacktestV2Result:
    """
    Chạy v3 rồi CẮT lợi suất về đoạn `mask` yêu cầu.

    `mask` là mảng boolean trên `data.index` (lưới nến gốc), không phải trên lưới
    tái cân bằng — giống chữ ký mà `validate_v3.py` dùng.

    THỨ TỰ LÀ BẤT BIẾN: gộp thích ứng và dựng trọng số luôn chạy trên TOÀN lưới tái
    cân bằng; `mask` chỉ áp vào lúc cuối để chọn lợi suất nào được tính vào thống kê.
    Đảo thứ tự này là tái lập lỗi đo lường đã mô tả ở đầu file.
    """
    if cfg.n_tranches > 1:
        # Lưới 72h của hàm này chỉ mô phỏng được MỘT lô. Chạy im lặng với cấu hình chia
        # lô là đo một chiến lược khác thứ daemon chạy — đúng họ lỗi F51.
        raise ValueError(
            f"run_v3 chỉ mô phỏng một lô, cấu hình có n_tranches={cfg.n_tranches}. "
            f"Dùng run_v3_fine (tự chia lô), hoặc replace(cfg, n_tranches=1) nếu CỐ Ý "
            f"đo một lô.")
    maker_ratio = cfg.maker_ratio if maker_ratio is None else maker_ratio
    close_full = data.close

    marks = close_full.index[::cfg.rebalance_every]
    close = close_full.reindex(marks)

    # `sig` truyền vào để tái dùng giữa nhiều cấu hình danh mục — xem `combined_signal`.
    if sig is None:
        sig = combined_signal(data, cfg)

    spec = portfolio if portfolio is not None else cfg.portfolio
    if n_positions is not None and n_positions != spec.n_positions:
        from dataclasses import replace
        spec = replace(spec, n_positions=n_positions)
    W = build_weights(sig, close, spec)

    # `cost_bps` = chi phí một chiều ĐO THẬT, thay cả mô hình (xem `CostModel.flat_bps`).
    cost = CostModel(maker_ratio=maker_ratio, half_spread_bps=0.0,
                     per_symbol_bps=data.per_symbol_bps * (1 - maker_ratio), min_bps=0.5,
                     flat_bps=cost_bps)
    res = simulate(W, close, data.funding.reindex(marks), cost,
                   bar_hours=cfg.bar_hours * cfg.rebalance_every, rebalance_every=1)

    if mask is None:
        return res

    keep = np.isin(res.returns.index, close_full.index[mask])
    for fld in ("returns", "gross_returns", "cost_drag", "funding_pnl",
                "turnover", "n_positions", "net_exposure", "gross_exposure"):
        setattr(res, fld, getattr(res, fld)[keep])
    return res


def split_train_holdout(data: V3Data, cfg: StrategyV3Config = V3, **kw):
    """Trả về `(res_train, res_holdout)` theo mốc chia đã chốt trong `holdout_split.json`."""
    idx = data.index
    return (run_v3(data, cfg, mask=idx < data.split_ts, **kw),
            run_v3(data, cfg, mask=idx >= data.split_ts, **kw))


def _live_orderable(W: pd.DataFrame, cfg: StrategyV3Config,
                    capital_usd: Optional[float], leverage: Optional[float]) -> pd.DataFrame:
    """[FIX F54] Bỏ vị thế dưới min notional như live, khi đã cho vốn + đòn bẩy."""
    if capital_usd is None and leverage is None:
        return W
    if capital_usd is None or leverage is None:
        raise ValueError("capital_usd và leverage phải đi CÙNG nhau — thiếu một là lọc sai")
    from aegis.research.small_capital import drop_below_min_notional
    return drop_below_min_notional(W, capital_usd, leverage, cfg.min_notional_usd)


def run_v3_fine(
    data: V3Data,
    cfg: StrategyV3Config = V3,
    mask: Optional[np.ndarray] = None,
    maker_ratio: Optional[float] = None,
    cost_bps: Optional[float] = None,
    sig: Optional[pd.DataFrame] = None,
    capital_usd: Optional[float] = None,
    leverage: Optional[float] = None,
    no_trade_band: float = 0.0,
    smooth_halflife: Optional[float] = None,
) -> BacktestV2Result:
    """
    ĐÚNG chiến lược v3 (tái cân bằng 72h), nhưng đường vốn đo trên nến 4h.

    `no_trade_band` > 0: giao dịch như live (`research/trade_band.py`) — bỏ điều chỉnh
    nhỏ, bỏ lệnh mở/tăng dưới min notional ở `capital_usd` × `leverage`, cân lại trung
    lập. Live đang chạy dải 0,20; mặc định 0 giữ nguyên mọi con số cũ.
    `smooth_halflife`: làm mượt điểm tổng hợp (chỉ nghiên cứu, live chưa có).

    `sig` = tín hiệu gộp tính sẵn bằng `combined_signal(data, cfg)` — tái dùng khi quét
    nhiều cấu hình DANH MỤC trên cùng tầng gộp (tầng gộp không phụ thuộc danh mục).

    `capital_usd` + `leverage` [FIX F54]: bỏ mọi vị thế có notional < min notional ở
    đúng vốn và đòn bẩy đó, như live làm (`research/small_capital.py`). Bỏ trống = giả
    định mọi vị thế đặt được, chỉ đúng ở vốn đủ lớn.

    Dùng để (a) đo sụt giảm TRONG KỲ mà lưới 72h không nhìn thấy, và (b) có đủ điểm
    quyết định cho bài toán điều khiển đòn bẩy — xem `research/goal_dp.py`.

    Trọng số mục tiêu vẫn được dựng y hệt `run_v3`; khác biệt duy nhất nằm ở engine
    mô phỏng (`simulate_marked_to_market`), thứ GIỮ vị thế và để trọng số trôi giữa
    hai mốc tái cân bằng thay vì kéo về mục tiêu mỗi nến.
    """
    if cfg.n_tranches > 1:
        if no_trade_band or smooth_halflife:
            raise ValueError("chia lô chưa hỗ trợ dải không giao dịch / làm mượt")
        return run_v3_tranched(data, cfg, mask=mask, maker_ratio=maker_ratio,
                               cost_bps=cost_bps, capital_usd=capital_usd, leverage=leverage)
    maker_ratio = cfg.maker_ratio if maker_ratio is None else maker_ratio
    close_full = data.close

    marks = close_full.index[::cfg.rebalance_every]
    # [FIX F39] Lọc tín hiệu theo CẤU HÌNH như `run_v3`. Bản cũ gộp CẢ `data.signals`,
    # nên nạp dữ liệu một lần cho nhiều cấu hình là âm thầm chạy sai bộ tín hiệu.
    if sig is None:
        sig = combined_signal(data, cfg)
    if smooth_halflife:
        from aegis.research.trade_band import smooth_signal
        sig = smooth_signal(sig, smooth_halflife)
    W = _live_orderable(build_weights(sig, close_full.reindex(marks), cfg.portfolio),
                        cfg, capital_usd, leverage)
    min_trade = (cfg.min_notional_usd / (capital_usd * leverage)
                 if capital_usd and leverage else 0.0)

    # `cost_bps` = chi phí một chiều ĐO THẬT, thay cả mô hình (xem `CostModel.flat_bps`).
    cost = CostModel(maker_ratio=maker_ratio, half_spread_bps=0.0,
                     per_symbol_bps=data.per_symbol_bps * (1 - maker_ratio), min_bps=0.5,
                     flat_bps=cost_bps)
    res = simulate_marked_to_market(W, close_full, data.funding, cost,
                                    bar_hours=cfg.bar_hours, no_trade_band=no_trade_band,
                                    min_trade=min_trade if no_trade_band else 0.0)

    if mask is None:
        return res
    keep = np.isin(res.returns.index, close_full.index[mask])
    for fld in ("returns", "gross_returns", "cost_drag", "funding_pnl",
                "turnover", "n_positions", "net_exposure", "gross_exposure"):
        setattr(res, fld, getattr(res, fld)[keep])
    return res


def run_v3_tranched(
    data: V3Data,
    cfg: StrategyV3Config = V3,
    n_tranches: Optional[int] = None,
    phase: int = 0,
    mask: Optional[np.ndarray] = None,
    maker_ratio: Optional[float] = None,
    cost_bps: Optional[float] = None,
    cache: Optional[dict] = None,
    capital_usd: Optional[float] = None,
    leverage: Optional[float] = None,
) -> BacktestV2Result:
    """
    Chiến lược v3 CHIA LÔ, đường vốn trên nến 4h — CÙNG hàm trọng số với live
    (`research/tranching.py`).

    `n_tranches=1, phase=p` là chiến lược một lô bắt đầu ở pha p (p nến 4h sau mốc
    gốc) — quét p = 0..rebalance_every-1 là đo timing luck bằng backtest.

    Mô phỏng đặt lại CẢ SỔ về đích gộp mỗi khi có một lô tái cân bằng, kể cả phần trôi
    giá của các lô không tới lượt. Live có dải không giao dịch 20% nên bỏ qua phần lớn
    điều chỉnh vụn đó; ở đây thì trả phí cho chúng — tức ước lượng chi phí THẬN TRỌNG.
    """
    from aegis.research.tranching import tranched_weight_panel

    k = cfg.n_tranches if n_tranches is None else n_tranches
    maker_ratio = cfg.maker_ratio if maker_ratio is None else maker_ratio
    names = cfg.signal_names()
    missing = [n for n in names if n not in data.signals]
    if missing:
        raise KeyError(f"V3Data thiếu {len(missing)} tín hiệu cấu hình yêu cầu: {missing[:5]}")
    close_full = data.close
    W = tranched_weight_panel({n: data.signals[n] for n in names}, close_full,
                              cfg.rebalance_every, k, cfg.combiner, cfg.portfolio,
                              cfg.top_frac, phase=phase, cache=cache)
    W = _live_orderable(W, cfg, capital_usd, leverage)
    cost = CostModel(maker_ratio=maker_ratio, half_spread_bps=0.0,
                     per_symbol_bps=data.per_symbol_bps * (1 - maker_ratio), min_bps=0.5,
                     flat_bps=cost_bps)
    res = simulate_marked_to_market(W, close_full, data.funding, cost,
                                    bar_hours=cfg.bar_hours)
    if mask is None:
        return res
    keep = np.isin(res.returns.index, close_full.index[mask])
    for fld in ("returns", "gross_returns", "cost_drag", "funding_pnl",
                "turnover", "n_positions", "net_exposure", "gross_exposure"):
        setattr(res, fld, getattr(res, fld)[keep])
    return res

