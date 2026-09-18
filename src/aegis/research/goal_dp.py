"""
ĐẠT MỤC TIÊU TRƯỚC HẠN CHÓT — quy hoạch động trên xác suất, không phải trên Sharpe.

Câu hỏi "vốn 1 triệu, lời 50 nghìn trong 7 ngày" KHÔNG phải câu hỏi tối đa hoá Sharpe,
cũng không phải tối đa hoá tăng trưởng dài hạn (Kelly). Nó là một bài toán khác hẳn:

    tối đa hoá  P( W_T >= (1+g) * W_0 )   với T cố định

và lời giải của nó khác lời giải Kelly theo cách không trực giác được:

* KELLY tối đa hoá tốc độ tăng trưởng log **dài hạn**. Nó không quan tâm hạn chót.
* MỤC TIÊU CÓ HẠN CHÓT thưởng cho việc CHẠM ĐÍCH, và không thưởng thêm một xu nào cho
  việc vượt đích. Vượt đích 20% cũng chỉ được tính là 1, y như vừa đủ 0,1%.

Hệ quả trực tiếp: khi đã ở trên đích, mọi rủi ro tiếp theo chỉ có thể LÀM HỎNG, không
thể làm tốt hơn — nên đòn bẩy tối ưu ở đó là 0. Và khi đang ở dưới đích mà sắp hết
giờ, giữ đòn bẩy thấp gần như bảo đảm thua, nên đòn bẩy tối ưu TĂNG. Chính sách tối ưu
vì vậy là hình chữ U ngược theo thời gian còn lại và giảm dần theo mức vốn hiện có —
"nhát khi đã ăn, liều khi sắp hết giờ". Không mức đòn bẩy CỐ ĐỊNH nào tái tạo được
hành vi đó.

═══════════════════════════════════════════════════════════════════════════════
BA MỐC CHẶN TRÊN, XẾP THEO ĐỘ CHẶT — biết trước khi tính là biết mình đang ở đâu
═══════════════════════════════════════════════════════════════════════════════

(1) TRẦN CỦA ĐÒN BẨY CỐ ĐỊNH — dạng đóng, không cần mô phỏng.
    Với chuyển động Brown hình học, Sharpe năm S, chân trời T năm, mục tiêu g:

        log-vốn ~ N( L*mu*T - L^2*sigma^2*T/2 ,  L^2*sigma^2*T )

        P(L) = Phi( S*sqrt(T) - (sigma*sqrt(T)/2)*L - ln(1+g)/(sigma*sqrt(T)*L) )

    Đạo hàm theo L rồi cho bằng 0:  L* = sqrt( 2*ln(1+g) / (sigma^2 * T) )
    Thay ngược lại, hai số hạng cuối bằng nhau và bằng sqrt(ln(1+g)/2), nên:

        P_max_const = Phi( S*sqrt(T) - sqrt( 2*ln(1+g) ) )          <-- DẠNG ĐÓNG

    Đọc công thức này kỹ, vì nó nói ba điều mà bảng đòn bẩy không nói:
      - Số hạng thứ hai KHÔNG chứa S. Cái giá của mục tiêu là một hằng số trừ thẳng
        vào z-score, bất kể chiến lược tốt tới đâu.
      - TĂNG ĐÒN BẨY KHÔNG BAO GIỜ VƯỢT ĐƯỢC MỐC NÀY. Đòn bẩy nhân cả mu lẫn sigma,
        và phần lỗ do biến động (`L^2*sigma^2/2`) lớn dần nhanh hơn phần lãi.
      - Muốn P cao hơn thì chỉ còn hai đường: tăng S, hoặc kéo dài T.

(2) TRẦN CỦA ĐIỀU KHIỂN ĐỘNG KHÔNG RÀNG BUỘC — Neyman-Pearson / quyền chọn nhị phân.
    Nếu đòn bẩy không bị chặn và giao dịch liên tục, bài toán trở thành "mua rẻ nhất
    một quyền chọn nhị phân trả (1+g) tại T". Bổ đề Neyman-Pearson cho miền tối ưu và:

        P_max_dyn = Phi( Phi^-1( 1/(1+g) ) + S*sqrt(T) )

    Mốc này CAO HƠN (1) rất nhiều, và khoảng cách giữa hai mốc chính là giá trị của
    việc điều khiển động. Nhưng nó cần đòn bẩy không giới hạn — nên nó là mốc KHÔNG
    ĐẠT ĐƯỢC, chỉ dùng để biết còn bao nhiêu dư địa.

(3) SỰ THẬT — quy hoạch động dưới ràng buộc THẬT: trần đòn bẩy, số điểm quyết định
    hữu hạn, chi phí đổi đòn bẩy, phân phối lợi suất có đuôi dày, ngưỡng cháy hấp thụ.
    Đó là phần còn lại của module này. Nó luôn nằm giữa (1) và (2).

═══════════════════════════════════════════════════════════════════════════════
KỶ LUẬT DỮ LIỆU
═══════════════════════════════════════════════════════════════════════════════
Chính sách được GIẢI trên tập train và ĐÁNH GIÁ trên holdout. Giải trên holdout rồi
khoe kết quả holdout là tự chấm bài mình. `solve_goal_dp` và `evaluate_policy` tách
đôi cố ý để không thể vô tình làm điều đó.

Phân phối chuyển trạng thái trong DP là i.i.d. rút từ lợi suất lịch sử — một xấp xỉ
Markov, vì nó bỏ qua hiện tượng biến động theo cụm. Bù lại, phần ĐÁNH GIÁ dùng block
bootstrap giữ nguyên cụm. Chia như vậy là có chủ ý: xấp xỉ ở chỗ rẻ (tìm chính sách),
trung thực ở chỗ đắt (chấm điểm chính sách).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "GoalSpec",
    "GoalPolicy",
    "constant_leverage_ceiling",
    "dynamic_leverage_ceiling",
    "solve_goal_dp",
    "evaluate_policy",
    "policy_frontier",
]


# ---------------------------------------------------------------------------
# Mốc chặn dạng đóng
# ---------------------------------------------------------------------------
def constant_leverage_ceiling(sharpe_ann: float, ann_vol: float,
                              horizon_years: float, target_return: float) -> Dict[str, float]:
    """
    Mốc (1): xác suất TỐT NHẤT đạt được bằng một mức đòn bẩy CỐ ĐỊNH, và mức đó là bao nhiêu.

    Đây là con số cần biết trước khi bàn bất cứ điều gì về đòn bẩy: nếu nó đã thấp thì
    không có cách siết đòn bẩy nào cứu được, và câu trả lời đúng là đổi mục tiêu hoặc
    đổi chân trời, không phải đổi số nhân.
    """
    from scipy.stats import norm

    b = float(np.log1p(target_return))
    sT = float(np.sqrt(max(horizon_years, 1e-12)))
    sig_T = float(ann_vol * sT)                      # độ lệch chuẩn trên cả chân trời

    z = sharpe_ann * sT - np.sqrt(2.0 * b)
    lev_star = np.sqrt(2.0 * b) / sig_T if sig_T > 1e-12 else np.inf
    return {
        "p_max_constant": float(norm.cdf(z)),
        "leverage_star": float(lev_star),
        "z_score": float(z),
        "edge_term": float(sharpe_ann * sT),          # phần do chiến lược đóng góp
        "goal_cost_term": float(np.sqrt(2.0 * b)),    # cái giá của mục tiêu, không phụ thuộc S
        "horizon_vol": sig_T,
    }


def dynamic_leverage_ceiling(sharpe_ann: float, horizon_years: float,
                             target_return: float) -> Dict[str, float]:
    """
    Mốc (2): trần lý thuyết khi đòn bẩy KHÔNG bị chặn và giao dịch liên tục.

    Không đạt được trong thực tế. Giá trị của nó nằm ở KHOẢNG CÁCH tới mốc (1): khoảng
    cách đó là toàn bộ phần thưởng có thể có của việc điều khiển động. Nếu khoảng cách
    hẹp thì đừng mất công viết bộ điều khiển; nếu rộng thì đáng.
    """
    from scipy.stats import norm

    sT = float(np.sqrt(max(horizon_years, 1e-12)))
    z = float(norm.ppf(1.0 / (1.0 + target_return)) + sharpe_ann * sT)
    return {"p_max_dynamic": float(norm.cdf(z)), "z_score": z}


# ---------------------------------------------------------------------------
# Đặc tả bài toán
# ---------------------------------------------------------------------------
@dataclass
class GoalSpec:
    """
    Đặc tả bài toán "đạt mục tiêu trước hạn chót".

    `ruin_floor` là ngưỡng HẤP THỤ, không phải một mức khó chịu: chạm nó là hết, không
    có đường về. Đặt nó ở đâu là quyết định của người bỏ vốn chứ không phải của thuật
    toán — vì vậy nó là tham số, và `policy_frontier` quét nó để bày ra đánh đổi.

    `cost_per_leverage_turn` là chi phí đổi gross đi 1.0x, tính theo tỷ lệ vốn. Đổi đòn
    bẩy nghĩa là co/giãn toàn bộ sổ, tức giao dịch |dL| lần vốn. Ở 4-5bp/chiều thì
    con số này vào khoảng 0.0005. Nhỏ, nhưng nếu bỏ qua thì DP sẽ đề xuất một chính
    sách nhảy nhót mỗi nến — và đó là chính sách chỉ tồn tại được trên giấy.
    """

    target_return: float = 0.05          # +5% = 50.000 VND trên vốn 1.000.000
    horizon_periods: int = 42            # 42 nến 4h = 7 ngày
    ruin_floor: float = 0.70             # còn 70% vốn là dừng cuộc chơi
    max_leverage: float = 5.0
    n_leverage: int = 21                 # số mức trong khoảng [min_executable, max]
    #: Đòn bẩy gộp THẤP NHẤT còn đặt được lệnh. Không phải sở thích rủi ro — là số học:
    #: chiến lược cần `n_positions` vị thế để trung lập, mỗi vị thế phải vượt min notional
    #: của sàn. Vốn $38, 12 vị thế, min $5 => gross tối thiểu 12*5/38 = 1.58x.
    #:
    #: Vì vậy lưới hành động là {0} HỢP [min, max], KHÔNG phải [0, max] liên tục:
    #: "đóng sạch, đứng ngoài" là đặt được, còn "0.25x" thì sàn từ chối từng lệnh một.
    #: Bỏ qua ràng buộc này thì DP trả về một chính sách tối ưu KHÔNG THỰC HIỆN ĐƯỢC —
    #: đã xảy ra: nó chọn trung bình 1.27x, tương đương $4.01/vị thế, dưới min $5.
    min_executable_leverage: float = 0.0
    cost_per_leverage_turn: float = 0.0005
    n_wealth: int = 361                  # số nút lưới log-vốn
    n_quantiles: int = 64                # số nút rời rạc hoá phân phối lợi suất
    wealth_headroom: float = 0.06        # nới trần lưới trên mức mục tiêu

    def leverage_grid(self) -> np.ndarray:
        """{0} hợp [min_executable, max_leverage]. Xem `min_executable_leverage`."""
        lo = max(0.0, float(self.min_executable_leverage))
        if lo <= 1e-9:
            return np.linspace(0.0, self.max_leverage, self.n_leverage)
        if lo >= self.max_leverage:
            return np.array([0.0, self.max_leverage])
        return np.concatenate([[0.0], np.linspace(lo, self.max_leverage, self.n_leverage)])

    @staticmethod
    def executable_floor(capital_usd: float, n_positions: int,
                         min_notional_usd: float) -> float:
        """Gross tối thiểu để mọi vị thế vượt min notional của sàn."""
        return float(n_positions * min_notional_usd / max(capital_usd, 1e-9))

    def log_target(self) -> float:
        return float(np.log1p(self.target_return))

    def log_floor(self) -> float:
        return float(np.log(self.ruin_floor))

    def wealth_grid(self) -> np.ndarray:
        """Lưới log-vốn, từ ngưỡng cháy tới trên mục tiêu một khoảng."""
        return np.linspace(self.log_floor(), self.log_target() + self.wealth_headroom,
                           self.n_wealth)


@dataclass
class GoalPolicy:
    """
    Chính sách tối ưu + hàm giá trị.

    QUY ƯỚC TRỤC THỜI GIAN — đọc kỹ, đây là chỗ dễ lật ngược nhất:

        value[k]  : xác suất đạt đích khi CÒN `k` kỳ.   k = 0..T  (k=0 là hết giờ)
        policy[k] : đòn bẩy nên đặt khi CÒN `k+1` kỳ.   k = 0..T-1

    Hai trục lệch nhau MỘT bậc vì hành động cuối cùng được ra khi còn 1 kỳ chứ không
    phải khi còn 0 kỳ. Để hai mảng cùng chiều rồi tự nhớ độ lệch là cách chắc chắn sẽ
    nhầm — đã nhầm một lần: bảng chính sách in ra ngược hoàn toàn trục thời gian, và
    nó vẫn TRÔNG hợp lý (liều khi sắp hết giờ) nên suýt lọt. Vì vậy độ lệch được viết
    thẳng vào đây và lặp lại ở mọi hàm truy cập.
    """

    spec: GoalSpec
    policy: np.ndarray          # (T, n_wealth, n_leverage_prev) -> chỉ số hành động
    value: np.ndarray           # (T+1, n_wealth, n_leverage_prev) -> xác suất đạt đích
    wealth_grid: np.ndarray
    leverage_grid: np.ndarray
    meta: Dict = field(default_factory=dict)

    def p_success(self, wealth_ratio: float = 1.0, periods_left: Optional[int] = None,
                  current_leverage: float = 0.0) -> float:
        """Xác suất đạt đích từ trạng thái hiện tại — con số duy nhất đáng nhìn lúc vận hành."""
        k = self.spec.horizon_periods if periods_left is None else periods_left
        k = int(np.clip(k, 0, self.spec.horizon_periods))
        i = float(np.interp(np.log(max(wealth_ratio, 1e-12)),
                            self.wealth_grid, np.arange(len(self.wealth_grid))))
        j = int(np.argmin(np.abs(self.leverage_grid - current_leverage)))
        lo, hi = int(np.floor(i)), min(int(np.ceil(i)), len(self.wealth_grid) - 1)
        f = i - lo
        v = self.value[k]                       # value[k] <-> còn k kỳ
        return float(v[lo, j] * (1 - f) + v[hi, j] * f)

    def leverage_for(self, wealth_ratio: float, periods_left: int,
                     current_leverage: float = 0.0) -> float:
        """Đòn bẩy nên đặt ngay bây giờ. Đây là toàn bộ đầu ra dùng được lúc vận hành."""
        k = int(np.clip(periods_left, 1, self.spec.horizon_periods))
        row = k - 1                             # hành động khi còn k kỳ nằm ở policy[k-1]
        i = int(np.clip(np.searchsorted(self.wealth_grid, np.log(max(wealth_ratio, 1e-12))),
                        0, len(self.wealth_grid) - 1))
        j = int(np.argmin(np.abs(self.leverage_grid - current_leverage)))
        return float(self.leverage_grid[self.policy[row, i, j]])


# ---------------------------------------------------------------------------
# Rời rạc hoá phân phối lợi suất
# ---------------------------------------------------------------------------
def _quantise(returns: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nén chuỗi lợi suất thành `n` nút có trọng số bằng nhau, GIỮ NGUYÊN ĐUÔI.

    Dùng phân vị chứ không dùng lưới đều: lưới đều dồn hầu hết nút vào giữa phân phối
    (nơi chẳng có gì xảy ra) và để đuôi — nơi quyết định sống chết — cho hai ba nút.
    Phân vị cho mỗi nút cùng xác suất, nên mật độ nút tự động dày ở chỗ dày dữ liệu và
    vẫn đặt nút tận cùng đúng vào quan sát cực đoan nhất từng xảy ra.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < n:
        return r.copy(), np.full(len(r), 1.0 / len(r))
    qs = (np.arange(n) + 0.5) / n
    nodes = np.quantile(r, qs)
    return nodes, np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Quy hoạch động
# ---------------------------------------------------------------------------
def solve_goal_dp(returns: pd.Series | np.ndarray, spec: Optional[GoalSpec] = None,
                  verbose: bool = False) -> GoalPolicy:
    """
    Giải bài toán bằng quy nạp lùi từ hạn chót về hiện tại.

    Trạng thái là (log-vốn, đòn bẩy đang giữ). Đòn bẩy đang giữ PHẢI nằm trong trạng
    thái vì chi phí phụ thuộc |dL| — bỏ nó đi thì DP không thấy cái giá của việc nhảy
    nhót và sẽ đề xuất một chính sách không ai chạy được.

    Điều kiện biên:
      * t = T (hết giờ):  V = 1 nếu log-vốn >= log(1+g), ngược lại 0.
      * chạm ngưỡng cháy: V = 0 vĩnh viễn (hấp thụ) — cài bằng cách ghim nút đáy = 0.
      * trên đỉnh lưới :  V = 1, và DP tự tìm ra rằng giữ nó bằng cách chọn L = 0.
        Quy tắc "đã đạt thì ngừng mạo hiểm" KHÔNG được viết tay ở đâu cả — nó là hệ
        quả mà quy nạp lùi tự suy ra. Đó là kiểm chứng tốt nhất rằng DP chạy đúng.
    """
    spec = spec or GoalSpec()
    r = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
    nodes, probs = _quantise(r, spec.n_quantiles)

    x = spec.wealth_grid()                      # lưới log-vốn
    L = spec.leverage_grid()
    nx, nL, nq = len(x), len(L), len(nodes)
    x_target = spec.log_target()

    # Hệ số tăng trưởng log cho mọi bộ ba (đòn bẩy trước, hành động, nút lợi suất).
    # Kẹp sàn ở -0.999: tài khoản không âm được, và chạm sàn là kết thúc.
    dcost = spec.cost_per_leverage_turn * np.abs(L[:, None] - L[None, :])   # (nL_prev, nL_act)
    step = 1.0 + L[None, :, None] * nodes[None, None, :] - dcost[:, :, None]
    G = np.log(np.maximum(step, 1e-3))                                     # (nL_prev, nL_act, nq)

    V = np.zeros((spec.horizon_periods + 1, nx, nL))
    V[0] = (x >= x_target).astype(np.float64)[:, None]        # V[0] = hết giờ
    policy = np.zeros((spec.horizon_periods, nx, nL), dtype=np.int16)

    for t in range(1, spec.horizon_periods + 1):
        Vn = V[t - 1]
        best = np.full((nx, nL), -1.0)
        arg = np.zeros((nx, nL), dtype=np.int16)
        for a in range(nL):                                   # hành động: đặt đòn bẩy L[a]
            col = Vn[:, a]                                    # trạng thái kế: đòn bẩy giữ = L[a]
            for pj in range(nL):                              # đòn bẩy đang giữ
                xn = x[:, None] + G[pj, a, :][None, :]        # (nx, nq)
                ev = np.interp(xn.ravel(), x, col).reshape(nx, nq) @ probs
                upd = ev > best[:, pj]
                best[upd, pj] = ev[upd]
                arg[upd, pj] = a
        # Ngưỡng cháy hấp thụ: nút đáy không bao giờ hồi phục.
        best[0, :] = 0.0
        V[t], policy[t - 1] = best, arg
        if verbose and t % 10 == 0:
            print(f"  t={t:>3}/{spec.horizon_periods}  V(vốn=1)={best[np.argmin(np.abs(x)), 0]:.4f}")

    return GoalPolicy(spec=spec, policy=policy, value=V,
                      wealth_grid=x, leverage_grid=L,
                      meta={"n_obs": int(len(r)), "n_quantiles": nq,
                            "mean_per_period": float(np.mean(r)),
                            "std_per_period": float(np.std(r, ddof=1))})


# ---------------------------------------------------------------------------
# Đánh giá TRUNG THỰC: block bootstrap ngoài mẫu
# ---------------------------------------------------------------------------
def _bootstrap_paths(r: np.ndarray, n_periods: int, n_paths: int,
                     block: int, seed: int) -> np.ndarray:
    """Ma trận (n_paths, n_periods) lợi suất chưa đòn bẩy, lấy mẫu theo KHỐI."""
    rng = np.random.default_rng(seed)
    n = len(r)
    if n < block * 3:
        raise ValueError(f"Cần ít nhất {block*3} quan sát, nhận {n}")
    n_blocks = int(np.ceil(n_periods / block))
    starts = rng.integers(0, n - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)
    return r[idx[:, :n_periods]]


def evaluate_policy(
    returns_oos: pd.Series | np.ndarray,
    policy: Optional[GoalPolicy] = None,
    spec: Optional[GoalSpec] = None,
    constant_leverages: Optional[List[float]] = None,
    n_paths: int = 40000,
    block: int = 6,
    seed: int = 11,
) -> pd.DataFrame:
    """
    Chấm điểm chính sách trên lợi suất NGOÀI MẪU, bằng đường lấy mẫu theo khối.

    Mọi chính sách được chấm trên CÙNG một bộ đường (cùng seed) — nếu không thì chênh
    lệch giữa chúng lẫn với nhiễu lấy mẫu, và với xác suất quanh 40% thì nhiễu Monte
    Carlo dễ dàng lớn hơn hiệu ứng cần đo.

    Lấy mẫu theo KHỐI chứ không theo điểm: lợi suất chiến lược có cụm biến động, và
    chính chuỗi lỗ liên tiếp — thứ bị xoá sạch khi lấy mẫu từng điểm — mới là thứ đẩy
    tài khoản xuống ngưỡng cháy.
    """
    spec = spec or (policy.spec if policy is not None else GoalSpec())
    r = returns_oos.to_numpy() if isinstance(returns_oos, pd.Series) else np.asarray(returns_oos)
    r = r[np.isfinite(r)]

    paths = _bootstrap_paths(r, spec.horizon_periods, n_paths, block, seed)
    x_target, x_floor = spec.log_target(), spec.log_floor()
    rows = []

    def _measure(label, final_log, ruined, lev_used, turns, lev_paths=None):
        final = np.exp(final_log)
        # Đòn bẩy TRUNG BÌNH trộn lẫn lúc đứng ngoài (0x) với lúc vào lệnh, nên nó
        # thường rơi vào một giá trị KHÔNG ĐẶT ĐƯỢC — ví dụ 1.32x khi lưới hành động
        # chỉ có {0} và [1.58, 5.0]. Báo cáo thêm hai con số có nghĩa vật lý:
        #   frac_flat        : tỷ lệ thời gian đứng ngoài thị trường
        #   avg_leverage_on  : đòn bẩy trung bình KHI ĐANG có vị thế
        frac_flat, lev_on = 0.0, float(np.mean(lev_used))
        if lev_paths is not None:
            active = lev_paths > 1e-9
            frac_flat = float(1.0 - active.mean())
            lev_on = float(lev_paths[active].mean()) if active.any() else 0.0
        return {
            "policy": label,
            "p_target": float((final_log >= x_target - 1e-12).mean()),
            "p_ruin": float(ruined.mean()),
            "p_loss": float((final < 1.0).mean()),
            "median": float(np.median(final) - 1.0),
            "mean": float(final.mean() - 1.0),
            "p05": float(np.percentile(final, 5) - 1.0),
            "p95": float(np.percentile(final, 95) - 1.0),
            "avg_leverage": float(np.mean(lev_used)),
            "avg_leverage_on": lev_on,
            "frac_flat": frac_flat,
            "max_leverage": float(np.max(lev_used)),
            "leverage_turns": float(np.mean(turns)),
        }

    # -- các mức đòn bẩy CỐ ĐỊNH (mốc so sánh) ------------------------------
    # `is None` chứ KHÔNG phải `or`: danh sách RỖNG là falsy trong Python, nên `or` sẽ
    # âm thầm rơi về mặc định đúng lúc người gọi nói "đừng chấm mức cố định nào cả".
    # Đã dính: bảng đánh đổi in ra 27,6% ở mọi dòng vì .iloc[0] lấy nhầm mức 1.0x.
    if constant_leverages is None:
        constant_leverages = [1.0, 2.0, 3.0, 4.0, 5.0]
    for Lc in constant_leverages:
        step = np.log(np.maximum(1.0 + Lc * paths, 1e-3))
        cum = np.cumsum(step, axis=1)
        ruined = (cum.min(axis=1) <= x_floor)
        final_log = np.where(ruined, x_floor, cum[:, -1])
        rows.append(_measure(f"cố định {Lc:.2f}x", final_log, ruined,
                             np.full(n_paths, Lc), np.zeros(n_paths),
                             lev_paths=np.full((n_paths, spec.horizon_periods), Lc)))

    # -- chính sách quy hoạch động -------------------------------------------
    if policy is not None:
        xg, Lg = policy.wealth_grid, policy.leverage_grid
        cur_x = np.zeros(n_paths)
        cur_j = np.zeros(n_paths, dtype=np.int64)        # chỉ số đòn bẩy đang giữ
        dead = np.zeros(n_paths, dtype=bool)
        lev_sum = np.zeros(n_paths); turns = np.zeros(n_paths)
        lev_hist = np.zeros((n_paths, spec.horizon_periods))

        for t in range(spec.horizon_periods):
            left = spec.horizon_periods - t      # số kỳ CÒN LẠI trước khi hành động
            row = policy.policy[left - 1]
            i = np.clip(np.searchsorted(xg, cur_x), 0, len(xg) - 1)
            a = row[i, cur_j]
            Lt = Lg[a]
            Lt = np.where(dead, 0.0, Lt)

            dcost = spec.cost_per_leverage_turn * np.abs(Lt - Lg[cur_j])
            gross = 1.0 + Lt * paths[:, t] - dcost
            cur_x = np.where(dead, cur_x, cur_x + np.log(np.maximum(gross, 1e-3)))
            newly = (~dead) & (cur_x <= x_floor)
            cur_x = np.where(newly, x_floor, cur_x)
            dead |= newly

            turns += (a != cur_j) & (~dead)
            lev_sum += Lt
            lev_hist[:, t] = Lt
            cur_j = a

        rows.append(_measure("DP mục tiêu", cur_x, dead,
                             lev_sum / spec.horizon_periods, turns,
                             lev_paths=lev_hist))

    return pd.DataFrame(rows)


def policy_frontier(returns_train, returns_oos, base_spec: Optional[GoalSpec] = None,
                    ruin_floors: Optional[List[float]] = None,
                    max_leverages: Optional[List[float]] = None,
                    **eval_kw) -> pd.DataFrame:
    """
    Quét đánh đổi: mỗi (ngưỡng cháy, trần đòn bẩy) -> xác suất đạt đích ngoài mẫu.

    Đây là bảng để NGƯỜI chọn, không phải để máy chọn. Máy không biết mất 30% vốn của
    người dùng nặng bằng bao nhiêu lần việc lỡ mục tiêu — và giả vờ biết bằng cách nhét
    một hàm thoả dụng vào code là cách tệ nhất để giấu một giả định quan trọng.
    """
    from dataclasses import replace

    base = base_spec or GoalSpec()
    out = []
    for floor in (ruin_floors or [0.70]):
        for mx in (max_leverages or [base.max_leverage]):
            spec = replace(base, ruin_floor=floor, max_leverage=mx)
            pol = solve_goal_dp(returns_train, spec)
            df = evaluate_policy(returns_oos, pol, spec, constant_leverages=[], **eval_kw)
            # constant_leverages=[] => CHỈ chấm DP (xem ghi chú `is None` trong evaluate_policy)
            row = df.iloc[-1].to_dict()
            row.update(ruin_floor=floor, max_leverage=mx)
            out.append(row)
    return pd.DataFrame(out)
