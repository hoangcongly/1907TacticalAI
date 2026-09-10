"""
Ước lượng ma trận hiệp phương sai bền vững cho danh mục đa tài sản.

VÌ SAO CẦN: hiệp phương sai mẫu với N tài sản cần N(N+1)/2 tham số. Với 100 cặp
và 500 quan sát, ta ước lượng 5.050 tham số từ 50.000 số — ma trận gần suy biến,
và mọi bài toán tối ưu sẽ "ăn" đúng vào phần nhiễu ước lượng đó (error maximization).

Ledoit-Wolf co ma trận mẫu về một MỤC TIÊU có cấu trúc. Cường độ co được chọn
theo công thức đóng tối thiểu hoá sai số bình phương kỳ vọng, không phải tinh chỉnh
bằng tay.

Ba mục tiêu co được cài ở đây, theo thứ tự phù hợp với crypto:
  - `constant_correlation`: mọi cặp có cùng tương quan trung bình, giữ nguyên
    biến động riêng. Đây là mục tiêu Ledoit-Wolf khuyến nghị cho tài sản tài chính
    và là MẶC ĐỊNH ở đây — crypto có một nhân tố chung rất mạnh (beta BTC).
  - `single_index`: mô hình một nhân tố (thị trường) + nhiễu riêng.
  - `identity`: co về ma trận đơn vị đã chuẩn hoá — thô nhất, dùng khi mẫu cực ngắn.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "shrink_covariance",
    "ledoit_wolf_constant_correlation",
    "single_index_covariance",
    "nearest_positive_definite",
    "ewma_covariance",
]


def _to_matrix(returns) -> Tuple[np.ndarray, Optional[pd.Index]]:
    """Chuẩn hoá đầu vào về ndarray (T x N) đã bỏ hàng thiếu dữ liệu."""
    if isinstance(returns, pd.DataFrame):
        clean = returns.dropna(how="any")
        return clean.to_numpy(dtype=np.float64), returns.columns
    arr = np.asarray(returns, dtype=np.float64)
    return arr[~np.isnan(arr).any(axis=1)], None


def ledoit_wolf_constant_correlation(returns) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf co về mục tiêu TƯƠNG QUAN HẰNG SỐ (Ledoit & Wolf 2004, "Honey,
    I Shrunk the Sample Covariance Matrix").

    Mục tiêu F giữ nguyên phương sai từng tài sản nhưng thay mọi tương quan bằng
    tương quan trung bình toàn cục. Cường độ co delta* = max(0, min(1, kappa/T)).

    Trả về `(sigma_shrunk, delta)`. `delta` gần 1 nghĩa là mẫu quá nhiễu, ước lượng
    gần như hoàn toàn dựa vào cấu trúc áp đặt — dấu hiệu cần cửa sổ dài hơn.
    """
    X, _ = _to_matrix(returns)
    T, N = X.shape
    if T < 3 or N < 2:
        raise ValueError(f"Cần T>=3 và N>=2, nhận T={T}, N={N}")

    Xc = X - X.mean(axis=0)
    # Hiệp phương sai mẫu chuẩn hoá theo T (ML estimator, khớp với dẫn xuất gốc).
    S = (Xc.T @ Xc) / T

    var = np.diag(S).copy()
    var[var <= 0] = 1e-18
    sd = np.sqrt(var)
    outer_sd = np.outer(sd, sd)

    corr = S / outer_sd
    np.fill_diagonal(corr, 1.0)
    off = ~np.eye(N, dtype=bool)
    rbar = float(corr[off].mean())

    # Mục tiêu co F: phương sai giữ nguyên, tương quan = rbar.
    F = rbar * outer_sd
    np.fill_diagonal(F, var)

    # pi = tổng phương sai tiệm cận của các phần tử hiệp phương sai mẫu.
    X2 = Xc ** 2
    pi_mat = (X2.T @ X2) / T - S ** 2
    pi = float(pi_mat.sum())

    # rho = hiệp phương sai giữa sai số ước lượng của S và của mục tiêu F.
    # theta_ii_ij = E[(x_i^2 - S_ii)(x_i x_j - S_ij)] = (1/T) sum x_i^3 x_j - S_ii S_ij
    m3 = ((Xc ** 3).T @ Xc) / T
    theta_ii_ij = m3 - var[:, None] * S          # [i, j]
    theta_jj_ij = m3.T - var[None, :] * S        # [i, j]

    ratio = np.outer(1.0 / sd, sd)               # [i, j] = sd_j / sd_i
    rho_off = rbar * (ratio * theta_ii_ij + (1.0 / ratio) * theta_jj_ij) / 2.0
    rho = float(np.diag(pi_mat).sum()) + float(rho_off[off].sum())

    # gamma = khoảng cách bình phương giữa mẫu và mục tiêu.
    gamma = float(((F - S) ** 2).sum())

    if gamma <= 1e-18:
        delta = 0.0
    else:
        kappa = (pi - rho) / gamma
        delta = float(np.clip(kappa / T, 0.0, 1.0))

    sigma = delta * F + (1.0 - delta) * S
    # Chuyển về ước lượng không chệch theo T-1 để khớp quy ước phần còn lại của repo.
    sigma *= T / max(T - 1, 1)
    return nearest_positive_definite(sigma), delta


def single_index_covariance(returns, market: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Hiệp phương sai theo mô hình MỘT NHÂN TỐ: sigma = beta beta' var_m + diag(var_eps).

    Với crypto, nhân tố chung (rổ đều trọng số ~ "thị trường") giải thích phần lớn
    hiệp phương sai. Mô hình này chỉ cần 2N+1 tham số thay vì N(N+1)/2.
    """
    X, _ = _to_matrix(returns)
    T, N = X.shape
    m = X.mean(axis=1) if market is None else np.asarray(market, dtype=np.float64)[-T:]

    mc = m - m.mean()
    var_m = float(mc @ mc) / max(T - 1, 1)
    if var_m <= 0:
        return nearest_positive_definite(np.cov(X, rowvar=False))

    Xc = X - X.mean(axis=0)
    beta = (Xc.T @ mc) / (mc @ mc)
    resid = Xc - np.outer(mc, beta)
    var_eps = (resid ** 2).sum(axis=0) / max(T - 2, 1)

    sigma = np.outer(beta, beta) * var_m + np.diag(var_eps)
    return nearest_positive_definite(sigma)


def ewma_covariance(returns, halflife: float = 60.0, min_periods: int = 30) -> np.ndarray:
    """
    Hiệp phương sai có trọng số mũ — phản ứng nhanh với đổi chế độ biến động.

    Crypto đổi chế độ rất nhanh; cửa sổ đều trọng số 250 nến vẫn "nhớ" một giai đoạn
    đã không còn liên quan. Nửa đời `halflife` nến cho ước lượng bám sát hiện tại
    mà vẫn đủ mẫu hiệu dụng.
    """
    X, _ = _to_matrix(returns)
    T, N = X.shape
    if T < min_periods:
        raise ValueError(f"Cần ít nhất {min_periods} quan sát, nhận {T}")

    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()

    mu = (w[:, None] * X).sum(axis=0)
    Xc = X - mu
    sigma = (Xc * w[:, None]).T @ Xc
    # Hiệu chỉnh chệch cho trọng số (tương đương 1/(1 - sum w^2)).
    sigma /= max(1.0 - float((w ** 2).sum()), 1e-12)
    return nearest_positive_definite(sigma)


def nearest_positive_definite(sigma: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """
    Kẹp trị riêng để ma trận chắc chắn xác định dương.

    Sau khi co, sai số số học vẫn có thể để lại trị riêng âm nhỏ. Bất kỳ bài toán
    tối ưu nào gặp trị riêng âm sẽ trả về đòn bẩy vô hạn theo hướng đó — kẹp trước
    rẻ hơn nhiều so với gỡ một danh mục vô nghĩa.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)
    vals, vecs = np.linalg.eigh(sigma)
    if vals.min() >= min_eig:
        return sigma
    floor = max(min_eig, float(vals.max()) * 1e-10)
    return vecs @ np.diag(np.maximum(vals, floor)) @ vecs.T


def shrink_covariance(
    returns,
    method: str = "constant_correlation",
    halflife: Optional[float] = None,
) -> np.ndarray:
    """
    Điểm vào duy nhất cho ước lượng hiệp phương sai.

    `method`: "constant_correlation" (mặc định), "single_index", "ewma", "sample".
    `halflife`: nếu đặt, tính hiệp phương sai EWMA trước rồi mới co — kết hợp
    phản ứng nhanh với ổn định cấu trúc.
    """
    if method == "sample":
        X, _ = _to_matrix(returns)
        return nearest_positive_definite(np.cov(X, rowvar=False))
    if method == "single_index":
        return single_index_covariance(returns)
    if method == "ewma":
        return ewma_covariance(returns, halflife=halflife or 60.0)
    if method == "constant_correlation":
        sigma, _ = ledoit_wolf_constant_correlation(returns)
        if halflife is None:
            return sigma
        # Trộn EWMA (bám hiện tại) với LW (ổn định) — lấy trung bình hình học biến động.
        ew = ewma_covariance(returns, halflife=halflife)
        return nearest_positive_definite(0.5 * sigma + 0.5 * ew)
    raise ValueError(f"method không hợp lệ: {method}")
