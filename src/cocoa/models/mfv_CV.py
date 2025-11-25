import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterable, Tuple, Type

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import product

from .base_model import BaseModel



class MFVValidator:
    """
    MFV-style (multi-fold forward) CV for time series, as in your original function.

    - X_train, y_train must be already in time order.
    - We use the last Q * block_size points as the CV region.
    - Each fold q:
        * validation block = q-th segment of that region (consecutive, length block_size)
        * training block   = all observations strictly before that validation block
          (expanding window).
    """

    def __init__(self, Q: int = 5, block_size: int = 200):
        self.Q = Q
        self.block_size = block_size

    def score(
        self,
        model_class: Type[BaseModel],
        X_train,
        y_train,
        params: Dict[str, Any],
    ) -> float:
        """
        Compute MFV MSE for a given model class and hyperparameter dictionary.

        model_class: a subclass of BaseModel (e.g. RFModel)
        X_train, y_train: pandas objects or numpy arrays, in time order
        params: dict of hyperparameters passed to model_class(**params)

        Returns: MFV average MSE over all folds and points (float).
        """
        # Assume pandas DataFrame/Series with iloc; if numpy, convert to pandas-like indices.
        if not hasattr(X_train, "iloc"):
            # If numpy arrays: wrap with DataFrame/Series just for indexing
            X_train = pd.DataFrame(X_train)
        if not hasattr(y_train, "iloc"):
            y_train = pd.Series(y_train)

        T = len(X_train)
        total_block_length = self.Q * self.block_size

        if total_block_length >= T:
            raise ValueError("Q * block_size must be smaller than training length.")

        cv_start = T - total_block_length  # index where CV region starts

        sq_errors: List[float] = []

        for q in range(self.Q):
            # q-th validation block within CV region
            val_start = cv_start + q * self.block_size
            val_end = val_start + self.block_size  # exclusive

            # Training block: all observations strictly before validation block
            train_end = val_start  # exclusive
            X_tr = X_train.iloc[:train_end]
            y_tr = y_train.iloc[:train_end]

            X_val = X_train.iloc[val_start:val_end]
            y_val = y_train.iloc[val_start:val_end]

            # Instantiate and fit the model for this fold
            model = model_class(**params)
            model.fit(X_tr, y_tr)

            # Predict and accumulate squared errors
            y_pred = model.predict(X_val)
            sq_errors.extend((y_val.values - y_pred) ** 2)

        return float(np.mean(sq_errors))

    def grid_search(
        self,
        model_class: Type[BaseModel],
        X_train,
        y_train,
        param_grid: Iterable[Dict[str, Any]],
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Run MFV over a grid of hyperparameters.

        Returns:
            best_params: dict
            best_score: float
            all_results: list of {"params": ..., "mfv_mse": ...}
        """
        best_params = None
        best_score = np.inf
        all_results: List[Dict[str, Any]] = []

        for params in param_grid:
            mfv_mse = self.score(model_class, X_train, y_train, params)
            all_results.append({"params": params, "mfv_mse": mfv_mse})

            if verbose:
                print("Params:", params, "MFV MSE:", mfv_mse)

            if mfv_mse < best_score:
                best_score = mfv_mse
                best_params = params

        if best_params is None:
            raise RuntimeError("Grid search found no valid parameter set.")

        return best_params, best_score, all_results
