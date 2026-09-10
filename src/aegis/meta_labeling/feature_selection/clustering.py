"""
Hierarchical Clustering & Consensus Filter for Feature Selection.
Module B-5-4 (AFML Chapter 8).
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import List

def compute_distance_matrix(X: pd.DataFrame) -> pd.DataFrame:
    r"""
    Tính ma trận khoảng cách dựa trên tương quan (Correlation-based Distance).
    D_{i,j} = \sqrt{0.5 * (1 - \rho_{i,j})}
    Tuy nhiên, để chặn tuyệt đối độ lớn của tương quan (gộp cả âm và dương),
    ta dùng D_{i,j} = 1 - |\rho_{i,j}|.
    """
    # Ma trận tương quan
    corr = X.corr(method='spearman')
    
    # Khoảng cách dựa trên độ lớn tương quan tuyệt đối
    # Nếu |\rho| = 1 -> D = 0 (Giống nhau hoàn toàn)
    # Nếu |\rho| = 0 -> D = 1 (Khác biệt hoàn toàn)
    distance = 1 - corr.abs()
    
    # Ép sai số float
    distance = distance.clip(lower=0.0, upper=1.0)
    
    return distance

def cluster_features_hierarchical(distance_matrix: pd.DataFrame, corr_threshold: float = 0.70) -> dict:
    """
    Nhóm (Cluster) các feature bằng Hierarchical Clustering.
    Cắt cây tại khoảng cách tương ứng với corr_threshold.
    
    Args:
        distance_matrix: Ma trận khoảng cách D.
        corr_threshold: Ngưỡng tương quan (|\rho|). Các feature có |\rho| > threshold sẽ chung 1 cụm.
        
    Returns:
        dict: Mapping từ Cluster_ID -> List[Feature_Names].
    """
    # FIX #8: Copy trước khi mutate để không phá dữ liệu gốc của caller
    arr = distance_matrix.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    condensed_dist = squareform(arr)
    
    # Xây dựng cây phân cấp (dùng phương pháp 'complete' hoặc 'single')
    Z = linkage(condensed_dist, method='complete')
    
    # Tính threshold khoảng cách
    # Vì D = 1 - |\rho|, nên nếu yêu cầu |\rho| > 0.70 thì D < 0.30
    dist_threshold = 1.0 - corr_threshold
    
    # Cắt cây
    # fcluster trả về ID cụm cho mỗi feature
    cluster_labels = fcluster(Z, t=dist_threshold, criterion='distance')
    
    # Tạo dictionary
    clusters = {}
    features = distance_matrix.columns
    for feat, cid in zip(features, cluster_labels):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(feat)
        
    return clusters

def apply_consensus_filter(X: pd.DataFrame, consensus_ranks: pd.DataFrame, corr_threshold: float = 0.70) -> List[str]:
    """
    Lọc Feature Trùng lặp (Consensus Filter).
    
    Quy trình:
    1. Gom cụm các feature có |\rho| > corr_threshold.
    2. Trong mỗi cụm, giữ lại ĐÚNG 1 feature có thứ hạng (Final_Rank) TỐT NHẤT từ Triple Consensus.
    3. Loại bỏ các feature bị thay thế (Substitution Effect).
    
    Args:
        X: DataFrame chứa giá trị các features.
        consensus_ranks: Bảng xếp hạng từ `triple_consensus_ranker` (chứa cột 'Final_Rank').
        corr_threshold: Ngưỡng tương quan.
        
    Returns:
        Danh sách các feature_name được giữ lại (Sạch, không cộng tuyến).
    """
    dist_matrix = compute_distance_matrix(X)
    clusters = cluster_features_hierarchical(dist_matrix, corr_threshold)
    
    selected_features = []
    
    for cid, feats in clusters.items():
        if len(feats) == 1:
            selected_features.append(feats[0])
        else:
            # Có nhiều feature trong cụm -> cạnh tranh
            # Lấy hạng từ bảng consensus (Số nhỏ = Hạng cao)
            # Lọc những feature tồn tại trong bảng consensus
            valid_feats = [f for f in feats if f in consensus_ranks.index]
            if not valid_feats:
                # Nếu không có thông tin rank, lấy bừa thằng đầu tiên
                selected_features.append(feats[0])
                continue
                
            # Tìm feature có Final_Rank nhỏ nhất (tốt nhất)
            best_feat = min(valid_feats, key=lambda f: consensus_ranks.loc[f, 'Final_Rank'])
            selected_features.append(best_feat)
            
    return selected_features
