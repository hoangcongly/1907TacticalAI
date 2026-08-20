import numpy as np
import pytest
from aegis.features.regime.bootstrap_lrt import (
    fit_single_gaussian_params, loglik_single_gaussian,
    fit_hmm_2state_loglik, validate_two_regime_architecture_bootstrap,
    simulate_from_single_gaussian
)

def test_fit_single_gaussian_params():
    np.random.seed(42)
    data = np.random.normal(loc=1.0, scale=2.0, size=(1000, 2))
    
    params = fit_single_gaussian_params(data)
    
    assert "mu" in params
    assert "cov" in params
    assert params["mu"].shape == (2,)
    assert params["cov"].shape == (2, 2)
    
    # Check if PD
    np.linalg.cholesky(params["cov"]) # Should not raise

def test_loglik_single_gaussian():
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
    
    params = fit_single_gaussian_params(data)
    ll = loglik_single_gaussian(data, params)
    
    assert isinstance(ll, float)
    assert not np.isnan(ll)
    assert ll < 0  # Log-likelihood is negative for this PDF

def test_edge_case_degenerate_cov():
    # Constant data gives 0 variance
    data = np.ones((50, 2))
    params = fit_single_gaussian_params(data)
    
    # Should be clamped by sanitize_covariance_matrix, so det is > 0 and Cholesky passes
    assert np.linalg.det(params["cov"]) > 0
    np.linalg.cholesky(params["cov"])
    
    ll = loglik_single_gaussian(data, params)
    assert not np.isnan(ll)

def test_hmm_2state_loglik_better_than_single():
    np.random.seed(42)
    # Generate 2-regime synthetic data
    data = np.zeros((200, 2))
    state = 0
    for i in range(200):
        if np.random.rand() < 0.1: # 10% chance to switch
            state = 1 - state
        if state == 0:
            data[i] = np.random.normal(loc=0.0, scale=1.0, size=2)
        else:
            data[i] = np.random.normal(loc=5.0, scale=2.0, size=2)
            
    null_params = fit_single_gaussian_params(data)
    ll_n1 = loglik_single_gaussian(data, null_params)
    
    ll_n2 = fit_hmm_2state_loglik(data, n_restarts=2, max_iter=20)
    
    # 2-state HMM should fit 2-regime data much better than single Gaussian
    assert ll_n2 > ll_n1

def test_validate_two_regime_architecture_bootstrap_rejection():
    np.random.seed(42)
    # 1. Test data that is strongly 2-regime
    data_2regime = np.zeros((100, 2))
    state = 0
    for i in range(100):
        if np.random.rand() < 0.2:
            state = 1 - state
        if state == 0:
            data_2regime[i] = np.random.normal(loc=0.0, scale=0.5, size=2)
        else:
            data_2regime[i] = np.random.normal(loc=10.0, scale=0.5, size=2)
            
    # Use small bootstrap for speed
    is_2_regime = validate_two_regime_architecture_bootstrap(data_2regime, n_bootstrap=10, p_value_threshold=0.05)
    assert is_2_regime == True
    
    # 2. Test data that is strongly 1-regime (Random Walk / Gaussian Noise)
    data_1regime = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
    is_2_regime_false = validate_two_regime_architecture_bootstrap(data_1regime, n_bootstrap=10, p_value_threshold=0.05)
    assert is_2_regime_false == False

def test_validate_two_regime_architecture_bootstrap_type_i_error():
    """
    [TASK A-5-3] Type I error rate test.
    Trên dữ liệu 1-regime thuần túy, p-value > 0.01 trong >= 95% lần lặp.
    (Không over-reject H0)
    """
    np.random.seed(42)
    n_trials = 40
    n_bootstrap = 50
    
    non_rejections = 0
    for _ in range(n_trials):
        # Dữ liệu 1-regime thuần túy (Noise)
        data_1regime = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
        
        # p_value_threshold = 0.01
        is_2_regime = validate_two_regime_architecture_bootstrap(
            data_1regime, 
            n_bootstrap=n_bootstrap, 
            p_value_threshold=0.01
        )
        
        if not is_2_regime:
            non_rejections += 1
            
    non_rejection_rate = non_rejections / n_trials
    assert non_rejection_rate >= 0.95, f"Over-rejected: non_rejection_rate = {non_rejection_rate}"
