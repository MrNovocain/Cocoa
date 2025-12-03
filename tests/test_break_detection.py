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


