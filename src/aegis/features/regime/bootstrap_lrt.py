"""
validate_two_regime_architecture_bootstrap (Parametric Bootstrap LRT N=1 vs N=2).
"""
import numpy as np
from typing import Dict, Any
from aegis.features.kalman.covariance_utils import sanitize_covariance_matrix

def fit_single_gaussian_params(O_array: np.ndarray) -> Dict[str, Any]:
    """
    Khớp mô hình phân phối chuẩn 1 chiều (Single Gaussian) từ mảng quan sát.
    Sử dụng sanitize_covariance_matrix để đảm bảo Positive Definite (PD).
    """
    if O_array.ndim != 2:
        O_array = O_array.reshape(-1, 1)

    mu = np.mean(O_array, axis=0)
    cov = np.cov(O_array, rowvar=False)
    
    # Ép ma trận sang 2D nếu input là 1D
    if cov.ndim == 0:
        cov = np.array([[cov]])
        
    cov = sanitize_covariance_matrix(cov)
    return {"mu": mu, "cov": cov}

def loglik_single_gaussian(O_array: np.ndarray, params: Dict[str, Any]) -> float:
    """
    Tính log-likelihood của O_array dựa trên params của Single Gaussian.
    """
    if O_array.ndim != 2:
        O_array = O_array.reshape(-1, 1)
        
    mu, cov = params["mu"], params["cov"]
    n_dim = O_array.shape[1]
    
    # Kiểm tra PD qua cholesky (an toàn kép)
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = sanitize_covariance_matrix(cov)
        
    det_cov = max(np.linalg.det(cov), 1e-12)
    inv_cov = np.linalg.inv(cov)
    diff = O_array - mu
    
    # Toán tử bậc 2: (x - mu)^T * inv_cov * (x - mu)
    quad = np.sum((diff @ inv_cov) * diff, axis=1)
    
    # Log pdf cho mỗi mẫu
    log_probs = -0.5 * (n_dim * np.log(2.0 * np.pi) + np.log(det_cov) + quad)
    
    return float(np.sum(log_probs))

def simulate_from_single_gaussian(params: Dict[str, Any], n_steps: int) -> np.ndarray:
    """Sinh dữ liệu giả lập từ Single Gaussian."""
    return np.random.multivariate_normal(params["mu"], params["cov"], size=n_steps)

def _gaussian_pdf_multivariate(O_array: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Tính xác suất phát xạ Gaussian (Vectorized) cho từng mẫu."""
    if O_array.ndim != 2:
        O_array = O_array.reshape(-1, 1)
        
    n_dim = O_array.shape[1]
    
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = sanitize_covariance_matrix(cov)
        
    det_cov = max(np.linalg.det(cov), 1e-12)
    inv_cov = np.linalg.inv(cov)
    diff = O_array - mu
    
    quad = np.sum((diff @ inv_cov) * diff, axis=1)
    probs = (1.0 / np.sqrt(((2.0 * np.pi) ** n_dim) * det_cov)) * np.exp(-0.5 * quad)
    
    return np.clip(probs, 1e-15, None)  # Prevent zero probability

def fit_hmm_2state_loglik(
    O_array: np.ndarray, 
    n_restarts: int = 3, 
    max_iter: int = 50, 
    tol: float = 1e-4
) -> float:
    """
    Fit HMM 2 trạng thái bằng thuật toán EM (Baum-Welch) với scaling an toàn.
    Sử dụng multi-restart để tìm global optimum, trả về log-likelihood lớn nhất.
    """
    if O_array.ndim != 2:
        O_array = O_array.reshape(-1, 1)
    
    T, n_dim = O_array.shape
    best_loglik = -np.inf
    
    for restart in range(n_restarts):
        # 1. Khởi tạo ngẫu nhiên
        pi = np.random.dirichlet([1.0, 1.0])
        A = np.random.dirichlet([1.0, 1.0], size=2)
        
        # Chọn 2 điểm ngẫu nhiên làm mean
        idx = np.random.choice(T, 2, replace=False)
        mu = O_array[idx].copy()
        
        # Covariance khởi tạo từ toàn bộ data
        global_cov = np.cov(O_array, rowvar=False)
        if global_cov.ndim == 0:
            global_cov = np.array([[global_cov]])
        global_cov = sanitize_covariance_matrix(global_cov)
        cov = np.array([global_cov.copy(), global_cov.copy()])
        
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # --- E-STEP ---
            B = np.zeros((T, 2))
            B[:, 0] = _gaussian_pdf_multivariate(O_array, mu[0], cov[0])
            B[:, 1] = _gaussian_pdf_multivariate(O_array, mu[1], cov[1])
            
            # Forward pass (với scaling)
            alpha = np.zeros((T, 2))
            c = np.zeros(T)
            
            alpha[0] = pi * B[0]
            c[0] = max(np.sum(alpha[0]), 1e-15)
            alpha[0] /= c[0]
            
            for t in range(1, T):
                alpha[t] = (alpha[t-1] @ A) * B[t]
                c[t] = max(np.sum(alpha[t]), 1e-15)
                alpha[t] /= c[t]
                
            current_ll = np.sum(np.log(c))
            
            # Backward pass (với scaling)
            beta = np.zeros((T, 2))
            beta[-1] = 1.0
            
            for t in range(T-2, -1, -1):
                beta[t] = (A @ (B[t+1] * beta[t+1])) / c[t+1]
                
            # Tính gamma và xi
            gamma = alpha * beta
            gamma_sum = np.sum(gamma, axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1e-15
            gamma /= gamma_sum
            
            xi = np.zeros((T-1, 2, 2))
            for t in range(T-1):
                temp = np.outer(alpha[t], B[t+1] * beta[t+1]) * A
                xi[t] = temp / max(np.sum(temp), 1e-15)
                
            # --- M-STEP ---
            gamma_sum_T = np.sum(gamma, axis=0)
            gamma_sum_T_minus_1 = np.sum(gamma[:-1], axis=0)
            
            pi = gamma[0]
            
            for i in range(2):
                if gamma_sum_T_minus_1[i] > 1e-12:
                    A[i] = np.sum(xi[:, i, :], axis=0) / gamma_sum_T_minus_1[i]
                
                if gamma_sum_T[i] > 1e-12:
                    mu[i] = np.sum(gamma[:, i:i+1] * O_array, axis=0) / gamma_sum_T[i]
                    diff = O_array - mu[i]
                    cov_i = (diff.T @ (gamma[:, i:i+1] * diff)) / gamma_sum_T[i]
                    cov[i] = sanitize_covariance_matrix(cov_i)
                    
            if current_ll - prev_ll < tol and iteration > 0:
                break
            prev_ll = current_ll
            
        if prev_ll > best_loglik:
            best_loglik = prev_ll
            
    return float(best_loglik)

def validate_two_regime_architecture_bootstrap(
    O_full: np.ndarray, 
    n_bootstrap: int = 500, 
    p_value_threshold: float = 0.01
) -> bool:
    """
    [TASK A-5-2] Thực hiện Parametric Bootstrap LRT kiểm định N=1 vs N=2.
    Trả về True nếu mô hình 2 trạng thái thực sự vượt trội có ý nghĩa thống kê.
    """
    null_params = fit_single_gaussian_params(O_full)
    ll_n1_real = loglik_single_gaussian(O_full, null_params)
    ll_n2_real = fit_hmm_2state_loglik(O_full)
    lr_stat_real = 2.0 * (ll_n2_real - ll_n1_real)
    
    boot_stats = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Sinh dữ liệu từ H0 (Single Gaussian)
        O_sim = simulate_from_single_gaussian(null_params, n_steps=len(O_full))
        
        # Fit lại cả N=1 và N=2 trên dữ liệu mô phỏng
        null_params_sim = fit_single_gaussian_params(O_sim)
        ll_n1_sim = loglik_single_gaussian(O_sim, null_params_sim)
        # Chỉ chạy 1 restart cho bootstrap để giảm thiểu thời gian
        ll_n2_sim = fit_hmm_2state_loglik(O_sim, n_restarts=1)
        
        boot_stats[b] = 2.0 * (ll_n2_sim - ll_n1_sim)
        
    p_value = np.mean(boot_stats >= lr_stat_real)
    return bool(p_value < p_value_threshold)
