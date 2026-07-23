import pytest
from aegis.risk.circuit_breaker import CircuitBreaker, CircuitBreakerTier

def test_circuit_breaker_tiers():
    """
    Kiểm chứng 3 mốc Circuit Breaker hoạt động chính xác.
    Tier 1 (5% DD): Giảm 50% vị thế.
    Tier 2 (10% DD): Flatten toàn bộ, đóng băng 24h.
    Tier 3 (15% DD): Kill Switch hệ thống.
    """
    cb = CircuitBreaker(tier1_threshold=0.05, tier2_threshold=0.10, tier3_threshold=0.15)
    
    # Bắt đầu với equity 100,000
    t = 0
    state = cb.update_equity(100_000, t)
    assert state.tier == CircuitBreakerTier.NORMAL
    assert state.max_position_multiplier == 1.0
    
    # 1. Rớt 6% -> Tier 1 (Giảm 50%)
    state = cb.update_equity(94_000, t)
    assert state.tier == CircuitBreakerTier.TIER_1_REDUCE
    assert state.max_position_multiplier == 0.5
    
    # 2. Phục hồi một phần -> Vẫn ở Tier 1 hoặc Normal nếu DD < 5%
    # Lên lại 96,000 -> DD = 4% < 5% -> NORMAL
    state = cb.update_equity(96_000, t)
    assert state.tier == CircuitBreakerTier.NORMAL
    
    # 3. Rớt 11% -> Tier 2 (Flatten, Freeze)
    state = cb.update_equity(89_000, t)
    assert state.tier == CircuitBreakerTier.TIER_2_FLATTEN
    assert state.max_position_multiplier == 0.0
    assert state.is_frozen == True
    
    # 4. Phục hồi ngay lập tức lên 99,000 -> Đang bị Freeze nên KHÔNG được giao dịch!
    t += 1000 # 1 giây sau
    state = cb.update_equity(99_000, t)
    assert state.is_frozen == True
    assert state.max_position_multiplier == 0.0 # Bị đóng băng
    
    # 5. Rớt 16% -> Tier 3 (Kill Switch)
    state = cb.update_equity(84_000, t)
    assert state.tier == CircuitBreakerTier.TIER_3_KILL
    assert state.max_position_multiplier == 0.0
    
    # 6. Có bơm tiền lên 200,000 thì hệ thống đã CHẾT (Dead)
    state = cb.update_equity(200_000, t)
    assert state.tier == CircuitBreakerTier.TIER_3_KILL
    assert state.max_position_multiplier == 0.0
