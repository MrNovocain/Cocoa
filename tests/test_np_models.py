import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.models.np_engines import LocalPolynomialEngine
from cocoa.models.np_kernels import EpanechnikovKernel
from cocoa.models.np_regime import NPRegimeModel
from cocoa.models.np_combo import NPConvexCombinationModel
from cocoa.models.np_base import BaseKernel, BaseLocalEngine


class DummyKernel(BaseKernel):
    def weight(self, u):  # type: ignore[override]
        return np.ones_like(u)


class DummyEngine(BaseLocalEngine):
    def fit(self, X_train, y_train, X_eval, h, kernel):  # type: ignore[override]
        return np.full(len(X_eval), y_train.mean())


def test_local_polynomial_engine_recovers_linear_function():
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_train = pd.Series(2 * X_train["x"] + 1)
    X_eval = pd.DataFrame({"x": [0.5, 1.5]})

    engine = LocalPolynomialEngine(order=1, use_gpu=False)
    kernel = EpanechnikovKernel()

    preds = engine.fit(X_train, y_train, X_eval, h=5.0, kernel=kernel)
    expected = 2 * X_eval["x"].to_numpy() + 1

    np.testing.assert_allclose(preds, expected, rtol=1e-4, atol=1e-4)


def test_local_polynomial_engine_uses_nearest_neighbor_fallback():
    X_train = pd.DataFrame({"x": [0.0, 10.0]})
    y_train = pd.Series([1.0, 5.0])
    X_eval = pd.DataFrame({"x": [5.0]})

    engine = LocalPolynomialEngine(order=0, use_gpu=False)
    kernel = EpanechnikovKernel()

    preds = engine.fit(X_train, y_train, X_eval, h=0.01, kernel=kernel)

    assert preds.shape == (1,)
    assert preds[0] == pytest.approx(1.0)


def test_np_regime_model_delegates_to_local_engine():
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])
    X_eval = pd.DataFrame({"x": [4.0, 5.0]})

    model = NPRegimeModel(kernel=DummyKernel(), local_engine=DummyEngine(), bandwidth=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_eval)

    assert np.all(preds == np.full(len(X_eval), y_train.mean()))


def test_np_convex_combination_model_combines_regimes():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 1.0, 3.0, 5.0])
    kernel = DummyKernel()
    engine = DummyEngine()

    combo = NPConvexCombinationModel(
        kernel=kernel,
        local_engine=engine,
        pre_bandwidth=1.0,
        post_bandwidth=1.0,
        break_index=2,
        gamma=0.25,
    )

    combo.fit(X, y)

    preds = combo.predict(pd.DataFrame({"x": [10.0, 11.0]}))

    assert combo.post_model_is_active is True
    np.testing.assert_allclose(preds, np.full(len(preds), 3.25))
