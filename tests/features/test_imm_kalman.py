"""Unit tests for IMM Kalman 2D Filter (Module B.2)."""
import numpy as np
import pytest
from aegis.features.kalman.imm_kalman import IMMKalman2D, filter_imm_kalman_series


def test_imm_kalman_initialization():
    imm = IMMKalman2D()
    assert imm.x1.shape == (2, 1)
    assert imm.x2.shape == (2, 1)
    assert imm.P1.shape == (2, 2)
    assert imm.P2.shape == (2, 2)
    # Check PD via Cholesky
    np.linalg.cholesky(imm.P1)
    np.linalg.cholesky(imm.P2)


def test_imm_kalman_step_positive_definite():
    imm = IMMKalman2D()
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, 100))
    
    for i, p in enumerate(prices):
        p_hat, nu_hat, trend_score = imm.step(p, p_trend=0.8, p_chop=0.2, atr=1.0)
        assert np.isfinite(p_hat)
        assert np.isfinite(nu_hat)
        assert np.isfinite(trend_score)
        # Covariance matrices must remain strictly PD at all times
        np.linalg.cholesky(imm.P1)
        np.linalg.cholesky(imm.P2)


def test_imm_kalman_trend_score_upward():
    imm = IMMKalman2D()
    # Strong upward trend: price increases by +1 every step
    prices = [100.0 + i * 1.5 for i in range(40)]
    
    for p in prices:
        p_hat, nu_hat, trend_score = imm.step(p, p_trend=0.9, p_chop=0.1, atr=1.5)
        
    # Velocity nu_hat and trend_score must be positive and strong
    assert nu_hat > 0.5
    assert trend_score > 0.3


def test_imm_kalman_trend_score_downward():
    imm = IMMKalman2D()
    # Strong downward trend: price decreases by -1.5 every step
    prices = [100.0 - i * 1.5 for i in range(40)]
    
    for p in prices:
        p_hat, nu_hat, trend_score = imm.step(p, p_trend=0.9, p_chop=0.1, atr=1.5)
        
    assert nu_hat < -0.5
    assert trend_score < -0.3


def test_filter_imm_kalman_series():
    n = 50
    prices = np.linspace(100, 150, n)
    p_trends = np.full(n, 0.8)
    p_chops = np.full(n, 0.2)
    atrs = np.full(n, 1.0)
    
    df_res = filter_imm_kalman_series(prices, p_trends, p_chops, atrs)
    assert len(df_res) == n
    assert "p_hat" in df_res.columns
    assert "nu_hat" in df_res.columns
    assert "trend_score" in df_res.columns
    assert df_res["trend_score"].iloc[-1] > 0.0
