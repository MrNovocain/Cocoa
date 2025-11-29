import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.experiments.break_detection import estimate_break_mohr_ll


@pytest.fixture
def dgp_with_break():
    """
    Generates a simple DGP with a structural break in the mean.
    y_t = alpha_t + epsilon_t, where alpha_t changes at T1.
    """
    T = 100
    T1 = 50
    alpha1 = 0.0
    alpha2 = 5.0
    sigma = 1.0
    
    np.random.seed(42)
    
    y = np.zeros(T)
    y[:T1] = alpha1 + sigma * np.random.randn(T1)
    y[T1:] = alpha2 + sigma * np.random.randn(T - T1)
    
    # Generate a dummy regressor, as it's required by the function
    X = np.random.randn(T, 1)
    
    return y, X, T1

def test_estimate_break_mohr_ll_finds_break(dgp_with_break):
    """
    Tests that estimate_break_mohr_ll can find a simple break in the mean.
    """
    y, X, T1_true = dgp_with_break
    
    # For this simple DGP, a pilot estimate of the overall mean is sufficient.
    m_hat = np.mean(y) * np.ones_like(y)
    
    # Estimate the break
    T1_hat = estimate_break_mohr_ll(
        y=y,
        X=X,
        m_hat=m_hat,
        trim_frac=0.05  # A reasonable trim fraction
    )
    
    # Assert that the estimated break is close to the true break
    # We allow for a small deviation
    print(f"True break: {T1_true}, Estimated break: {T1_hat}")
    assert abs(T1_hat - T1_true) <= 5
