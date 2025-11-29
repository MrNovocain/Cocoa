import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.models.base_model import BaseModel
from cocoa.models.mfv_CV import MFVConvexComboValidator


class SimpleOLSModel(BaseModel):
    """Tiny OLS model for testing MFV convex-combo gamma selection."""

    def __init__(self):
        super().__init__()
        self.coef_ = None

    def fit(self, X, y, sample_weight=None):
        x_arr = np.asarray(X.iloc[:, 0], dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_design = np.column_stack([np.ones_like(x_arr), x_arr])
        coef, _, _, _ = np.linalg.lstsq(X_design, y_arr, rcond=None)
        self.coef_ = coef
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        x_arr = np.asarray(X.iloc[:, 0], dtype=float)
        return self.coef_[0] + self.coef_[1] * x_arr


def _make_linear_break_sample(
    *,
    T_pre: int,
    T_post: int,
    pre_slope: float,
    post_slope: float,
    noise: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    x_pre = rng.normal(scale=1.0, size=T_pre)
    x_post = rng.normal(scale=1.0, size=T_post)

    y_pre = pre_slope * x_pre + rng.normal(scale=noise, size=T_pre)
    y_post = post_slope * x_post + rng.normal(scale=noise, size=T_post)

    X = pd.DataFrame({"x": np.concatenate([x_pre, x_post])})
    y = pd.Series(np.concatenate([y_pre, y_post]))

    return X, y, T_pre


def test_gamma_prefers_post_model_when_break_is_large():
    X, y, break_index = _make_linear_break_sample(
        T_pre=120,
        T_post=80,
        pre_slope=0.5,
        post_slope=2.0,
        noise=0.1,
        seed=10,
    )

    validator = MFVConvexComboValidator(Q=3)
    best_gamma, _ = validator.tune_gamma(
        SimpleOLSModel,
        {},
        SimpleOLSModel,
        {},
        X,
        y,
        break_index,
        verbose=False,
    )

    assert best_gamma < 0.3


def test_gamma_stays_high_when_no_break_and_few_post_points():
    X, y, break_index = _make_linear_break_sample(
        T_pre=160,
        T_post=30,
        pre_slope=1.0,
        post_slope=1.0,
        noise=0.05,
        seed=22,
    )

    validator = MFVConvexComboValidator(Q=3)
    best_gamma, _ = validator.tune_gamma(
        SimpleOLSModel,
        {},
        SimpleOLSModel,
        {},
        X,
        y,
        break_index,
        verbose=False,
    )

    assert best_gamma > 0.5
