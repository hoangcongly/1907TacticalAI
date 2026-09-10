"""
Tick Rule Classification & chuẩn hóa Order Flow Imbalance (OFI_t).
Module A.2.3 — Chuẩn hóa OFI trong phạm vi [-1.0, 1.0].
"""
import numpy as np
from numba import njit


@njit(fastmath=True)
def _classify_tick_rule_numba(prices: np.ndarray, initial_sign: float = 0.0) -> np.ndarray:
    n = len(prices)
    signs = np.empty(n, dtype=np.float64)
    if n == 0:
        return signs
    
    current_sign = initial_sign
    if current_sign == 0.0:
        # Tìm bước nhảy giá khác 0 đầu tiên để xác định hướng ban đầu
        for i in range(1, n):
            diff = prices[i] - prices[i - 1]
            if diff > 0.0:
                current_sign = 1.0
                break
            elif diff < 0.0:
                current_sign = -1.0
                break
        if current_sign == 0.0:
            current_sign = 1.0
            
    signs[0] = current_sign
    
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0.0:
            current_sign = 1.0
        elif diff < 0.0:
            current_sign = -1.0
        signs[i] = current_sign
        
    return signs


def classify_tick_rule(prices: np.ndarray, initial_sign: float = 0.0) -> np.ndarray:
    """
    Phân loại chiều luồng lệnh theo thuật toán Tick Rule chuẩn AFML:
    b_t = +1 nếu Delta P_t > 0
    b_t = -1 nếu Delta P_t < 0
    b_t = b_{t-1} nếu Delta P_t == 0 (zero-tick kế thừa chiều trước đó).
    """
    if len(prices) == 0:
        return np.array([], dtype=np.float64)
    prices_arr = np.asarray(prices, dtype=np.float64)
    return _classify_tick_rule_numba(prices_arr, float(initial_sign))


@njit(fastmath=True)
def _compute_tick_rule_ofi_numba(prices: np.ndarray, volumes: np.ndarray, initial_sign: float = 0.0) -> float:
    n = len(prices)
    if n == 0:
        return 0.0
    
    signs = _classify_tick_rule_numba(prices, initial_sign)
    buy_vol = 0.0
    sell_vol = 0.0
    
    for i in range(n):
        v = volumes[i]
        if v > 0.0:
            if signs[i] > 0.0:
                buy_vol += v
            else:
                sell_vol += v
                
    total_vol = buy_vol + sell_vol
    if total_vol <= 1e-12:
        return 0.0
        
    ofi = (buy_vol - sell_vol) / total_vol
    if ofi > 1.0:
        return 1.0
    elif ofi < -1.0:
        return -1.0
    return ofi


def compute_tick_rule_ofi(prices: np.ndarray, volumes: np.ndarray, initial_sign: float = 0.0) -> float:
    """
    Tính Order Flow Imbalance (OFI) cho một bar nến từ luồng tick con:
    OFI = (V_buy - V_sell) / (V_buy + V_sell) in [-1.0, 1.0].
    """
    if len(prices) == 0 or len(volumes) == 0:
        return 0.0
    return float(_compute_tick_rule_ofi_numba(
        np.asarray(prices, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
        float(initial_sign)
    ))


def compute_rolling_ofi(prices: np.ndarray, volumes: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Tính OFI cửa sổ trượt (rolling window) trên chuỗi nến hoặc chuỗi tick.
    Các vị trí i < window - 1 được gán NaN để tránh look-ahead và insufficient data.
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window or window <= 0:
        return result
        
    prices_arr = np.asarray(prices, dtype=np.float64)
    volumes_arr = np.asarray(volumes, dtype=np.float64)
    
    for i in range(window - 1, n):
        w_prices = prices_arr[i - window + 1 : i + 1]
        w_vols = volumes_arr[i - window + 1 : i + 1]
        result[i] = _compute_tick_rule_ofi_numba(w_prices, w_vols, 0.0)
        
    return result
