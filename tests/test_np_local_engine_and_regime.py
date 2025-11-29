import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.models.np_engines import LocalPolynomialEngine
from cocoa.models.np_kernels import GaussianKernel
from cocoa.models.np_regime import NPRegimeModel
from cocoa.models.np_base import BaseLocalEngine


class StubLocalEngine(BaseLocalEngine):
    """Local engine that simply records the last call and returns preset values."""

    def __init__(self, return_values: np.ndarray):
        self.return_values = return_values
        self.calls = []

    def fit(self, X_train, y_train, X_eval, h, kernel):  # pragma: no cover - interface defined in BaseLocalEngine
        self.calls.append(
            {
                "X_train": X_train.copy(),
                "y_train": y_train.copy(),
                "X_eval": X_eval.copy(),
                "h": h,
                "kernel": kernel,
            }
        )
        return self.return_values


def test_local_polynomial_engine_recovers_linear_function():
    rng = np.random.default_rng(123)
    X_train = pd.DataFrame({"x": rng.uniform(-3, 3, size=200)})
    y_train = 3.0 * X_train["x"] + 1.5 + rng.normal(scale=0.05, size=len(X_train))

    X_eval = pd.DataFrame({"x": np.linspace(-2.5, 2.5, 10)})

    engine = LocalPolynomialEngine(order=1, use_gpu=False)
    kernel = GaussianKernel()
    preds = engine.fit(X_train, y_train, X_eval, h=1.0, kernel=kernel)

    expected = 3.0 * X_eval["x"] + 1.5
    mae = np.mean(np.abs(preds - expected))
    assert mae < 0.1


def test_local_polynomial_engine_falls_back_to_nearest_neighbor():
    X_train = pd.DataFrame({"x": [0.0, 10.0]})
    y_train = pd.Series([1.0, 5.0])
    X_eval = pd.DataFrame({"x": [0.0, 4.0, 10.0]})

    engine = LocalPolynomialEngine(order=1, use_gpu=False)
    kernel = GaussianKernel()
    preds = engine.fit(X_train, y_train, X_eval, h=1e-9, kernel=kernel)

    # With an extremely small bandwidth, weights are ~0 and the engine falls back to NN.
    np.testing.assert_allclose(preds, np.array([1.0, 1.0, 5.0]), atol=1e-6)


def test_np_regime_model_uses_local_engine_for_predictions():
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])
    X_eval = pd.DataFrame({"x": [0.5, 1.5]})

    stub_preds = np.array([10.0, 20.0])
    stub_engine = StubLocalEngine(return_values=stub_preds)
    model = NPRegimeModel(kernel=GaussianKernel(), local_engine=stub_engine, bandwidth=0.5)

    model.fit(X_train, y_train)
    preds = model.predict(X_eval)

    assert np.array_equal(preds, stub_preds)
    assert len(stub_engine.calls) == 1
    call = stub_engine.calls[0]
    pd.testing.assert_frame_equal(call["X_train"], X_train)
    pd.testing.assert_series_equal(call["y_train"], y_train)
    pd.testing.assert_frame_equal(call["X_eval"], X_eval)
    assert call["h"] == 0.5
    assert isinstance(call["kernel"], GaussianKernel)


def test_np_regime_model_handles_empty_training_data():
    model = NPRegimeModel(kernel=GaussianKernel(), local_engine=StubLocalEngine(np.array([])), bandwidth=0.3)
    model.fit(pd.DataFrame({"x": []}), pd.Series(dtype=float))
    preds = model.predict(pd.DataFrame({"x": [1.0, 2.0]}))
    assert np.array_equal(preds, np.zeros(2))
