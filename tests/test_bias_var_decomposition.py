import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.utils.bias_var_decomposition import bias_variance_decomposition
from cocoa.models.base_model import BaseModel
from cocoa.models.combo_base import BaseConvexCombinationModel
from cocoa.models.ml_combo import MLConvexCombinationModel
from cocoa.models.np_regime import NPRegimeModel
from cocoa.models.np_combo import NPConvexCombinationModel
from cocoa.models.np_base import BaseKernel, BaseLocalEngine


class ConstantModel(BaseModel):
    """A simple model that always predicts a constant value."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.value = value
        self.hyperparams = {"value": value}

    def fit(self, X, y, sample_weight=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        return np.full(len(X), self.value)


class ConstantComboModel(BaseConvexCombinationModel):
    """A convex combination model using ConstantModel pre/post components."""

    def __init__(self, pre_value: float, post_value: float, break_index: int, gamma: float):
        super().__init__(break_index=break_index, gamma=gamma)
        self.pre_value = pre_value
        self.post_value = post_value
        self.hyperparams = {
            "pre_value": pre_value,
            "post_value": post_value,
            "break_index": break_index,
            "gamma": gamma,
        }

    def _initialize_sub_models(self) -> None:
        self.model_pre = ConstantModel(self.pre_value)
        self.model_post = ConstantModel(self.post_value)


class DummyKernel(BaseKernel):
    def weight(self, u: float) -> float:
        return 1.0


class DummyLocalEngine(BaseLocalEngine):
    def fit(self, X_train, y_train, X_eval, h, kernel):
        # Return the mean of the training labels for any evaluation point
        return np.full(len(X_eval), y_train.mean())


@pytest.fixture
def toy_dataset():
    X = pd.DataFrame({"x": np.linspace(0, 1, 10)})
    y = pd.Series(np.linspace(0, 1, 10))
    return X.iloc[:6], y.iloc[:6], X.iloc[6:], y.iloc[6:]


def test_bvd_with_partial_model(toy_dataset):
    X_train, y_train, X_test, y_test = toy_dataset
    partial_model = partial(ConstantModel, value=0.5)

    mse, bias_sq, var = bias_variance_decomposition(
        model_class=partial_model,
        hyperparams={"value": 0.5},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_bootstrap_rounds=3,
        random_seed=0,
    )

    assert mse >= 0
    assert bias_sq >= 0
    assert var >= 0


def test_bvd_with_convex_combo_model(toy_dataset):
    X_train, y_train, X_test, y_test = toy_dataset

    final_model = ConstantComboModel(pre_value=0.2, post_value=0.8, break_index=len(X_train) // 2, gamma=0.6)
    final_model.fit(X_train, y_train)

    mse, bias_sq, var = bias_variance_decomposition(
        model_class=ConstantComboModel,
        hyperparams={
            "pre_value": 0.2,
            "post_value": 0.8,
            "break_index": len(X_train) // 2,
            "gamma": 0.6,
            "final_model_instance": final_model,
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_bootstrap_rounds=3,
        random_seed=0,
    )

    assert mse >= 0
    assert bias_sq >= 0
    assert var >= 0


def test_bvd_with_ml_combo_model_and_model_class(toy_dataset):
    X_train, y_train, X_test, y_test = toy_dataset

    final_model = MLConvexCombinationModel(
        model_class=ConstantModel,
        params_pre={"value": 0.1},
        params_post={"value": 0.9},
        break_index=len(X_train) // 2,
        gamma=0.5,
    )
    final_model.fit(X_train, y_train)

    mse, bias_sq, var = bias_variance_decomposition(
        model_class=MLConvexCombinationModel,
        hyperparams={
            "model_class": ConstantModel,
            "params_pre": {"value": 0.1},
            "params_post": {"value": 0.9},
            "break_index": len(X_train) // 2,
            "gamma": 0.5,
            "final_model_instance": final_model,
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_bootstrap_rounds=3,
        random_seed=0,
    )

    assert mse >= 0
    assert bias_sq >= 0
    assert var >= 0


def test_bvd_with_np_regime_model(toy_dataset):
    X_train, y_train, X_test, y_test = toy_dataset
    kernel = DummyKernel()
    engine = DummyLocalEngine()

    mse, bias_sq, var = bias_variance_decomposition(
        model_class=NPRegimeModel,
        hyperparams={
            "kernel": kernel,
            "local_engine": engine,
            "bandwidth": 0.5,
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_bootstrap_rounds=3,
        random_seed=0,
    )

    assert mse >= 0
    assert bias_sq >= 0
    assert var >= 0


def test_bvd_with_np_combo_model(toy_dataset):
    X_train, y_train, X_test, y_test = toy_dataset
    kernel = DummyKernel()
    engine = DummyLocalEngine()

    final_model = NPConvexCombinationModel(
        kernel=kernel,
        local_engine=engine,
        pre_bandwidth=0.3,
        post_bandwidth=0.7,
        break_index=len(X_train) // 2,
        gamma=0.4,
    )
    final_model.fit(X_train, y_train)

    mse, bias_sq, var = bias_variance_decomposition(
        model_class=NPConvexCombinationModel,
        hyperparams={
            "kernel": kernel,
            "local_engine": engine,
            "pre_bandwidth": 0.3,
            "post_bandwidth": 0.7,
            "break_index": len(X_train) // 2,
            "gamma": 0.4,
            "final_model_instance": final_model,
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_bootstrap_rounds=3,
        random_seed=0,
    )

    assert mse >= 0
    assert bias_sq >= 0
    assert var >= 0
