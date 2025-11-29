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

    def __init__(self, Q: int = 5):
        self.Q = Q
        self.block_size = None

    def _set_block_size(self, T: int):
        self.block_size = T // (self.Q + 1)
        print(f"Set block size to {self.block_size} for training length {T} and Q={self.Q}.")   
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
        if self.block_size is None:
            raise ValueError("block_size must be set before calling score().")
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
        T = len(X_train)
        self._set_block_size(T)

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


class MFVConvexComboValidator(MFVValidator):
    """
    A validator for tuning the gamma parameter of a convex combination of two models.
    It uses MFV on the post-break data to find the optimal gamma.
    """

    def tune_gamma(
        self,
        model_class_pre: Type[BaseModel],
        params_pre: Dict[str, Any],
        model_class_post: Type[BaseModel],
        params_post: Dict[str, Any],
        X_train_full: pd.DataFrame,
        y_train_full: pd.Series,
        break_index: int,
        gamma_grid: Iterable[float] = np.linspace(0, 1, 21),
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """
        Finds the best gamma for combining a pre-break and a post-break model.

        The MFV process happens only on the post-break data. For each fold:
        - The validation set is a block of post-break data.
        - The training set is all data before that validation block.
        - The pre-model is trained on pre-break data from the training set.
        - The post-model is trained on post-break data from the training set.

        Returns:
            best_gamma (float): The gamma value that minimized MSE.
            best_score (float): The minimum MSE achieved.
        """
        X_train_pre_full = X_train_full.iloc[:break_index]
        y_train_pre_full = y_train_full.iloc[:break_index]
        X_train_post_full = X_train_full.iloc[break_index:]
        y_train_post_full = y_train_full.iloc[break_index:]

        T_post = len(X_train_post_full)
        self._set_block_size(T_post)
        if self.block_size == 0:
            raise ValueError("Post-break training data is too short for the number of MFV folds.")

        cv_start_post = T_post - self.Q * self.block_size

        y_val_all, y_pre_all, y_post_all = [], [], []

        for q in range(self.Q):
            val_start_idx_post = cv_start_post + q * self.block_size
            val_end_idx_post = val_start_idx_post + self.block_size

            X_val = X_train_post_full.iloc[val_start_idx_post:val_end_idx_post]
            y_val = y_train_post_full.iloc[val_start_idx_post:val_end_idx_post]

            # Training set for this fold (relative to post-break data)
            train_end_idx_post = val_start_idx_post
            X_tr_post = X_train_post_full.iloc[:train_end_idx_post]
            y_tr_post = y_train_post_full.iloc[:train_end_idx_post]

            # Pre-model is always trained on all available pre-break data
            model_pre = model_class_pre(**params_pre)
            model_pre.fit(X_train_pre_full, y_train_pre_full)
            y_pre_pred = model_pre.predict(X_val)

            # Post-model is trained on its part of the expanding window
            model_post = model_class_post(**params_post)
            if not X_tr_post.empty:
                model_post.fit(X_tr_post, y_tr_post)
                y_post_pred = model_post.predict(X_val)
            else:
                y_post_pred = np.full(len(X_val), np.nan)

            y_val_all.extend(y_val.to_list())
            y_pre_all.extend(y_pre_pred.tolist())
            y_post_all.extend(y_post_pred.tolist())

        y_val_arr = np.array(y_val_all)
        y_pre_arr = np.array(y_pre_all)
        y_post_arr = np.array(y_post_all)

        valid_mask = ~np.isnan(y_post_arr)
        if not np.any(valid_mask):
            raise ValueError("Could not make any valid post-break predictions during MFV for gamma tuning.")

        y_val_filt, y_pre_filt, y_post_filt = y_val_arr[valid_mask], y_pre_arr[valid_mask], y_post_arr[valid_mask]

        gamma_losses = []
        for gamma in gamma_grid:
            y_hat = gamma * y_pre_filt + (1.0 - gamma) * y_post_filt
            loss = np.mean((y_val_filt - y_hat)**2)
            gamma_losses.append(loss)
            if verbose:
                print(f"  Gamma: {gamma:.2f}, MFV MSE: {loss:.6f}")

        best_idx = np.argmin(gamma_losses)
        best_gamma = list(gamma_grid)[best_idx]
        best_score = gamma_losses[best_idx]

        return best_gamma, best_score
