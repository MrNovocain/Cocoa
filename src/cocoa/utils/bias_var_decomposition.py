import numpy as np
import pandas as pd
from typing import Type, Any, Dict, Tuple
from abc import ABC, abstractmethod
from functools import partial
from tqdm import tqdm

from ..models.base_model import BaseModel
from ..models.combo_base import BaseConvexCombinationModel


class ModelInstantiator(ABC):
    """
    Abstract base class for a strategy that knows how to instantiate a model
    for a bias-variance decomposition bootstrap round.
    """
    @abstractmethod
    def create_model(self, model_class: Type[BaseModel], hyperparams: Dict[str, Any]) -> BaseModel:
        """Creates and returns an unfitted model instance."""
        pass


class DefaultModelInstantiator(ModelInstantiator):
    """Instantiates simple models (e.g., RFModel, NPRegimeModel) that accept hyperparameters directly."""
    def create_model(self, model_class: Type[BaseModel], hyperparams: Dict[str, Any]) -> BaseModel:
        # Create a clean copy of params, excluding keys not meant for the constructor.
        params_for_constructor = hyperparams.copy()
        params_for_constructor.pop("final_model_instance", None)
        return model_class(**params_for_constructor)


class ComboModelInstantiator(ModelInstantiator):
    """Instantiates convex combination models (NP or ML)."""
    def create_model(self, model_class: Type[BaseModel], hyperparams: Dict[str, Any]) -> BaseModel:
        # The runner provides the final fitted model instance in the hyperparams dict.
        # This instance contains all the necessary configuration to reconstruct a new model.
        final_model_instance = hyperparams.get("final_model_instance")
        if not final_model_instance:
            raise ValueError("Hyperparameter dictionary for combo models must contain 'final_model_instance' for BVD.")

        # The model's own `hyperparams` attribute holds the arguments needed for its constructor.
        constructor_params = final_model_instance.hyperparams.copy()

        # For NP models, we also need to pass the kernel and engine objects, which are stored on the instance.
        if hasattr(final_model_instance, 'kernel'):
            constructor_params['kernel'] = final_model_instance.kernel
            constructor_params['local_engine'] = final_model_instance.local_engine

        # For ML models, we need to pass the sub-model class.
        if hasattr(final_model_instance, 'model_class'):
            constructor_params['model_class'] = final_model_instance.model_class

        return model_class(**constructor_params)


def _resolve_model_class(model_class: Type[BaseModel] | partial[BaseModel]) -> Type[BaseModel]:
    """Return the underlying model class, unwrapping functools.partial if needed."""

    if isinstance(model_class, partial):
        return model_class.func  # type: ignore[return-value]
    return model_class


class BiasVarianceDecomposer:
    """
    Performs bias-variance decomposition for a given model using bootstrapping.
    """
    def __init__(
        self,
        model_class: Type[BaseModel],
        hyperparams: Dict[str, Any],
        instantiator: ModelInstantiator,
        n_bootstrap_rounds: int = 50,
        random_seed: int = 42,
    ):
        self.model_class = model_class
        self.hyperparams = hyperparams
        self.instantiator = instantiator
        self.n_bootstrap_rounds = n_bootstrap_rounds
        self.rng = np.random.default_rng(random_seed)

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[float, float, float]:
        print(f"Starting bias-variance decomposition with {self.n_bootstrap_rounds} rounds...")
        n_test = len(X_test)
        n_train = len(X_train)
        all_preds = np.zeros((self.n_bootstrap_rounds, n_test))

        for i in tqdm(range(self.n_bootstrap_rounds), desc="BVD Bootstrap Rounds"):
            bootstrap_indices = self.rng.choice(n_train, size=n_train, replace=True)
            X_boot = X_train.iloc[bootstrap_indices]
            y_boot = y_train.iloc[bootstrap_indices]

            model = self.instantiator.create_model(self.model_class, self.hyperparams)
            model.fit(X_boot, y_boot)
            all_preds[i, :] = np.asarray(model.predict(X_test))

        y_test_np = y_test.values
        avg_predictions = np.mean(all_preds, axis=0)
        avg_variance = np.mean(np.var(all_preds, axis=0))
        avg_bias_sq = np.mean((avg_predictions - y_test_np) ** 2)
        avg_mse = np.mean((all_preds - y_test_np.reshape(1, -1)) ** 2)

        print("Decomposition complete.")
        return avg_mse, avg_bias_sq, avg_variance


# This function is now a compatibility wrapper around the new class structure.
def bias_variance_decomposition(
    model_class: Type[BaseModel],
    hyperparams: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bootstrap_rounds: int = 50,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Performs bias-variance decomposition for a given model using bootstrapping.
    This function now delegates to the BiasVarianceDecomposer class.
    """
    # Determine which instantiator to use based on model name convention
    resolved_model_class = _resolve_model_class(model_class)

    # Determine which instantiator to use based on model inheritance
    if issubclass(resolved_model_class, BaseConvexCombinationModel):
        instantiator = ComboModelInstantiator()
    else:
        instantiator = DefaultModelInstantiator()

    decomposer = BiasVarianceDecomposer(
        model_class=model_class,
        hyperparams=hyperparams,
        instantiator=instantiator,
        n_bootstrap_rounds=n_bootstrap_rounds,
        random_seed=random_seed,
    )
    return decomposer.run(X_train, y_train, X_test, y_test)

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