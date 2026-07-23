import numpy as np
from aegis.features.regime.hmm_causal import CausalHMM2State

def test_hmm_causal_no_lookahead():
    """
    Kiểm chứng tính Causal-only (không có look-ahead bias):
    Thêm dữ liệu tương lai t+1 không được phép làm thay đổi 
    xác suất posterior đã tính ở thời điểm t.
    """
    A = np.array([[0.9, 0.1], 
                  [0.2, 0.8]])
    # State 0: mean=0, std=1 (Choppy)
    # State 1: mean=5, std=1 (Trending)
    means = np.array([0.0, 5.0])
    stds = np.array([1.0, 1.0])
    
    hmm = CausalHMM2State(A, means, stds)
    
    obs_t1 = 0.5
    obs_t2 = 4.8
    obs_future = -0.5
    
    # 1. Chạy với chuỗi ban đầu [t1, t2]
    hmm.alpha = np.array([0.5, 0.5])
    p1_a = hmm.step(obs_t1).copy()
    p2_a = hmm.step(obs_t2).copy()
    
    # 2. Chạy với chuỗi có thêm tương lai [t1, t2, t3]
    hmm.alpha = np.array([0.5, 0.5])
    p1_b = hmm.step(obs_t1).copy()
    p2_b = hmm.step(obs_t2).copy()
    p3_b = hmm.step(obs_future).copy()
    
    # Xác suất tại t1 và t2 phải hoàn toàn giống nhau bất kể có tương lai hay không
    np.testing.assert_allclose(p1_a, p1_b, err_msg="Look-ahead bias tại t=1!")
    np.testing.assert_allclose(p2_a, p2_b, err_msg="Look-ahead bias tại t=2!")

def test_hmm_causal_state_inference():
    """
    Kiểm chứng HMM nhận dạng đúng trạng thái khi quan sát rõ ràng.
    """
    A = np.array([[0.9, 0.1], 
                  [0.1, 0.9]])
    means = np.array([0.0, 5.0])
    stds = np.array([1.0, 1.0])
    
    hmm = CausalHMM2State(A, means, stds)
    
    # Đưa vào 10 quan sát xoay quanh 5.0 (State 1)
    obs = np.random.normal(5.0, 0.5, 10)
    posteriors = hmm.filter_series(obs)
    
    # Ở cuối chuỗi, xác suất State 1 phải rất cao
    assert posteriors[-1, 1] > 0.9
