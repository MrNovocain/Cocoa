import numpy as np
import pandas as pd
from typing import Type, Any, Dict, Tuple, List
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..models.base_model import BaseModel


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
            all_preds[i, :] = model.predict(X_test)

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
    if "Combo" in model_class.__name__:
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