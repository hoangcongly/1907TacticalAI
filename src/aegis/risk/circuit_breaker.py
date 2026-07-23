"""
3-Tier Drawdown Circuit Breakers (Module J).
Tier 1 (5% DD): Giảm 50% vị thế.
Tier 2 (10% DD): Flatten toàn bộ, đóng băng 24h.
Tier 3 (15% DD): Kill Switch hệ thống.
"""
import time
from enum import Enum
from dataclasses import dataclass

class CircuitBreakerTier(Enum):
    NORMAL = 0
    TIER_1_REDUCE = 1   # > 5% DD
    TIER_2_FLATTEN = 2  # > 10% DD
    TIER_3_KILL = 3     # > 15% DD

@dataclass
class CircuitBreakerState:
    tier: CircuitBreakerTier
    max_position_multiplier: float
    is_frozen: bool
    freeze_until_ms: int

class CircuitBreaker:
    def __init__(self, 
                 tier1_threshold: float = 0.05, 
                 tier2_threshold: float = 0.10, 
                 tier3_threshold: float = 0.15,
                 freeze_duration_ms: int = 24 * 60 * 60 * 1000): # 24h
        self.t1 = tier1_threshold
        self.t2 = tier2_threshold
        self.t3 = tier3_threshold
        self.freeze_duration_ms = freeze_duration_ms
        
        self.peak_equity = 0.0
        self.is_dead = False # Bị Kill Switch vĩnh viễn
        self.frozen_until_ms = 0
        
    def update_equity(self, current_equity: float, current_time_ms: int) -> CircuitBreakerState:
        """
        Cập nhật equity hiện tại, trả về trạng thái Circuit Breaker.
        """
        if self.is_dead:
            return CircuitBreakerState(CircuitBreakerTier.TIER_3_KILL, 0.0, True, 2**63 - 1)
            
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        if self.peak_equity <= 0:
            return CircuitBreakerState(CircuitBreakerTier.NORMAL, 1.0, False, 0)
            
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check freeze
        is_frozen = current_time_ms < self.frozen_until_ms
        
        if drawdown >= self.t3:
            self.is_dead = True
            return CircuitBreakerState(CircuitBreakerTier.TIER_3_KILL, 0.0, True, 2**63 - 1)
            
        if drawdown >= self.t2:
            self.frozen_until_ms = max(self.frozen_until_ms, current_time_ms + self.freeze_duration_ms)
            return CircuitBreakerState(CircuitBreakerTier.TIER_2_FLATTEN, 0.0, True, self.frozen_until_ms)
            
        if drawdown >= self.t1:
            return CircuitBreakerState(CircuitBreakerTier.TIER_1_REDUCE, 0.5, is_frozen, self.frozen_until_ms)
            
        return CircuitBreakerState(CircuitBreakerTier.NORMAL, 1.0 if not is_frozen else 0.0, is_frozen, self.frozen_until_ms)
