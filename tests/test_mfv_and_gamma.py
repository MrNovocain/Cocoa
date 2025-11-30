import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.models import MFVValidator, MFVConvexComboValidator, BaseModel


class BiasModel(BaseModel):
    """Simple model that predicts a constant bias."""

    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias
        self.hyperparams = {"bias": bias}

    def fit(self, X, y, sample_weight=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        return np.full(len(X), self.bias)


def _make_series(length=60, shift=0.0):
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, length)
    y = x + shift + rng.normal(scale=0.01, size=length)
    return pd.DataFrame({"feature": x}), pd.Series(y)


def test_mfv_validator_requires_block_size():
    validator = MFVValidator(Q=3)
    X, y = _make_series()
    with pytest.raises(ValueError):
        validator.score(BiasModel, X, y, {"bias": 0.0})


def test_mfv_validator_prefers_correct_bias():
    validator = MFVValidator(Q=3)
    X = pd.DataFrame({"feature": np.zeros(60)})
    y = pd.Series(np.zeros(60))
    validator._set_block_size(len(X))

    param_grid = [{"bias": 0.0}, {"bias": 1.0}]
    best_params, best_score, _ = validator.grid_search(
        BiasModel,
        X,
        y,
        param_grid,
        verbose=False,
    )

    assert best_params["bias"] == pytest.approx(0.0)
    assert np.isfinite(best_score)


def test_mfv_convex_combo_validator_requires_post_data():
    X, y = _make_series(20)
    validator = MFVConvexComboValidator(Q=3)
    break_index = len(X) - 1  # leaves only one post-break point

    with pytest.raises(ValueError):
        validator.tune_gamma(
            BiasModel,
            {"bias": 0.0},
            BiasModel,
            {"bias": 1.0},
            X,
            y,
            break_index,
            gamma_grid=[0.0, 0.5, 1.0],
            verbose=False,
        )

