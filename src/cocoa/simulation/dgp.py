
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict

class BaseDGP(ABC):
    """
    Abstract Base Class for Data Generating Processes.
    """
    @abstractmethod
    def generate(self) -> Dict[str, np.ndarray]:
        """
        Generates one realization of the data.
        Returns a dictionary containing:
        - 'X_pre', 'y_pre'
        - 'X_post', 'y_post'
        - 'X_oos', 'y_oos'
        - 'true_break_size' (optional metadata)
        """
        pass

class LargeBreakDGP(BaseDGP):
    """
    Generates a simple mean-shift process with a configurable break size.
    Model: y = mu + epsilon
    
    Pre-break: mu = 0
    Post-break: mu = drift_size
    """
    def __init__(self, T_pre: int = 1000, T_post: int = 20, drift_size: float = 10.0, noise_std: float = 1.0):
        self.T_pre = T_pre
        self.T_post = T_post
        self.drift_size = drift_size
        self.noise_std = noise_std
        
    def generate(self) -> Dict[str, np.ndarray]:
        # Pre-break data (Mean = 0)
        y_pre = np.random.normal(0, self.noise_std, self.T_pre)
        X_pre = np.zeros((self.T_pre, 1)) # Dummy feature 0
        
        # Post-break data (Mean = drift_size)
        y_post = np.random.normal(self.drift_size, self.noise_std, self.T_post)
        X_post = np.zeros((self.T_post, 1)) # Dummy feature 0
        
        # OOS data (Next step, Mean = drift_size)
        y_oos = np.random.normal(self.drift_size, self.noise_std, 1)
        X_oos = np.zeros((1, 1))
        

class HeavyTailDGP(BaseDGP):
    """
    Generates data with Heavy-Tailed noise (Student-t).
    Drift might be small, but outliers are frequent.
    """
    def __init__(self, T_pre: int = 1000, T_post: int = 20, drift_size: float = 0.0, df: float = 1.5):
        self.T_pre = T_pre
        self.T_post = T_post
        self.drift_size = drift_size
        self.df = df # Degrees of freedom (lower = heavier tails)
        
    def generate(self) -> Dict[str, np.ndarray]:
        # Helper for t-distribution
        def noise(n):
            return np.random.standard_t(self.df, size=n)
            
        y_pre = noise(self.T_pre)
        X_pre = np.zeros((self.T_pre, 1))
        
        y_post = noise(self.T_post) + self.drift_size
        X_post = np.zeros((self.T_post, 1))
        
        y_oos = noise(1) + self.drift_size
        X_oos = np.zeros((1, 1))
        

class RareRegionDGP(BaseDGP):
    """
    Generates data with Covariate Shift and a Localized Huge Break.
    X in [0, 1]. Rare region approx [0.95, 1.0].
    
    Pre-Break: High density in Rare Region. Mean = 0.
    Post-Break: Low density in Rare Region. Mean = BreakSize * I(x > 0.95).
    """
    def __init__(self, T_pre: int = 1000, T_post: int = 50, break_size: float = 100.0, noise_std: float = 1.0):
        self.T_pre = T_pre
        self.T_post = T_post
        self.break_size = break_size
        self.noise_std = noise_std
        self.threshold = 0.95
        
    def generate(self) -> Dict[str, np.ndarray]:
        
        # Helper distributions
        # Pre-break X: Skewed towards 1 (High density in rare region)
        # Beta(5, 1) mean is 5/6 ~ 0.83.
        # P(X > 0.95) = (1)^5 - (0.95)^5 = 1 - 0.77 = 0.23.
        # T=1000 -> ~230 points. Variance is tiny.
        X_pre = np.random.beta(5, 1, self.T_pre).reshape(-1, 1)
        y_pre = np.zeros(self.T_pre) + np.random.normal(0, self.noise_std, self.T_pre)
        
        # Post-break X: Sparse in rare region.
        # Uniform [0, 1] means P(X > 0.95) = 0.05.
        # T=50 -> Expect 2.5 points.
        # Variance will be high (few neighbors), but Estimator exists (unbiased-ish).
        X_post = np.random.uniform(0, 1, self.T_post).reshape(-1, 1)
        
        # Post-break Y: Mean is 0 except in rare region where it is break_size
        mask_post = (X_post > self.threshold).flatten()
        y_post = np.random.normal(0, self.noise_std, self.T_post)
        y_post[mask_post] += self.break_size
        
        # OOS Point: Explicitly IN the rare region
        X_oos = np.array([[0.98]]) # Inside [0.95, 1.0]
        y_oos = np.random.normal(self.break_size, self.noise_std, 1)
        

class MisleadingDGP(BaseDGP):
    """
    Validation period has SMALL BREAK (Mean 5).
    OOS period has MASSIVE BREAK (Mean 100).
    
    This tricks MFV into picking High Gamma on the small break.
    Then we test what happens if we carry that Gamma to the massive break 
    (assuming Post-Model adapts to the new mean, but Gamma is sticky).
    """
    def __init__(self, T_pre: int = 1000, T_post_val: int = 20, break_size_val: float = 8.0, break_size_oos: float = 100.0, noise_std: float = 5.0):
        self.T_pre = T_pre
        self.T_post_val = T_post_val
        self.break_size_val = break_size_val
        self.break_size_oos = break_size_oos
        self.noise_std = noise_std
        
    def generate(self) -> Dict[str, np.ndarray]:
        # Pre-break: Mean 0
        y_pre = np.random.normal(0, self.noise_std, self.T_pre)
        X_pre = np.zeros((self.T_pre, 1))
        
        # Post-Validation (Small Break): Mean 8
        y_post_val = np.random.normal(self.break_size_val, self.noise_std, self.T_post_val)
        X_post_val = np.zeros((self.T_post_val, 1))
        
        # OOS (High Break): Mean 100
        y_oos = np.random.normal(self.break_size_oos, self.noise_std, 1)
        X_oos = np.zeros((1, 1))
        
        return {
            "X_pre": X_pre, "y_pre": y_pre,
            "X_post": X_post_val, "y_post": y_post_val,
            "X_oos": X_oos, "y_oos": y_oos,
            "oracle_mean_val": self.break_size_val,
            "oracle_mean_oos": self.break_size_oos
        }
