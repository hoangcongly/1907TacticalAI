import pytest
import numpy as np
from aegis.execution.limit_queue_sim import simulate_limit_fill_with_queue

def test_simulate_limit_fill_price_aware():
    """
    F5 / G1: Limit order mô phỏng phải kiểm tra giá.
    Nếu giá không quét tới limit_price, filled phải là False bất kể volume lớn cỡ nào.
    """
    # Lệnh Mua ở giá 100.0
    limit_price = 100.0
    side = 1
    
    # Giá Low của các nến đều lớn hơn 100.0 (ví dụ 101.0, 102.0)
    # Nghĩa là giá chưa bao giờ quét xuống 100.0 để khớp lệnh Limit Mua.
    subsequent_lows = np.array([101.0, 102.0, 101.5, 105.0])
    subsequent_highs = np.array([102.0, 103.0, 102.5, 106.0])
    
    # Volume giao dịch thị trường cực lớn (1e12)
    subsequent_volumes = np.array([1e12, 1e12, 1e12, 1e12])
    
    queue_effective = 50.0
    order_size = 100.0
    
    result = simulate_limit_fill_with_queue(
        queue_effective=queue_effective,
        order_size=order_size,
        subsequent_volumes=subsequent_volumes,
        limit_price=limit_price,
        side=side,
        subsequent_highs=subsequent_highs,
        subsequent_lows=subsequent_lows,
        timeout_bars=4
    )
    
    assert result["filled"] is False, "Lệnh Limit Buy khớp mù quáng mặc dù giá không quét tới limit_price!"
    assert result["cumulative_volume_processed"] == 0.0, "Tích lũy volume mặc dù giá không chạm!"
