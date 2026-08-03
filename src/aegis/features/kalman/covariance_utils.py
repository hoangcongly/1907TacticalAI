"""
sanitize_covariance_matrix (Python & Rust closure) kẹp sàn eigenvalue_floor=1e-10.
"""

import numpy as np

def sanitize_covariance_matrix(cov: np.ndarray, eigenvalue_floor: float = 1e-10) -> np.ndarray:
    """
    [TASK A-4-5] Làm sạch ma trận hiệp phương sai (Covariance Matrix) để đảm bảo:
    1. Tính đối xứng (Symmetric)
    2. Xác định dương (Positive Definite - PD) thông qua việc kẹp sàn (clamping) các eigenvalues.
    
    Args:
        cov (np.ndarray): Ma trận hiệp phương sai N x N đầu vào.
        eigenvalue_floor (float): Giá trị eigenvalue tối thiểu cho phép. Mặc định là 1e-10.
        
    Returns:
        np.ndarray: Ma trận hiệp phương sai đã được làm sạch (Symmetric & Positive Definite).
    """
    # 1. Ép đối xứng (Symmetrize)
    cov_sym = (cov + cov.T) / 2.0
    
    # 2. Phân rã giá trị riêng (Eigen decomposition)
    # Dùng eigh cho ma trận đối xứng, nhanh và ổn định hơn eig
    eigenvalues, eigenvectors = np.linalg.eigh(cov_sym)
    
    # 3. Kẹp sàn (Clamp) các eigenvalues nhỏ hơn mức cho phép
    eigenvalues_clamped = np.maximum(eigenvalues, eigenvalue_floor)
    
    # 4. Tái tạo ma trận (Reconstruct matrix)
    # Công thức: V * D_clamped * V^T
    cov_sanitized = (eigenvectors * eigenvalues_clamped) @ eigenvectors.T
    
    # 5. Ép đối xứng lần cuối để triệt tiêu nhiễu từ phép nhân dấu phẩy động
    cov_sanitized = (cov_sanitized + cov_sanitized.T) / 2.0
    
    return cov_sanitized
