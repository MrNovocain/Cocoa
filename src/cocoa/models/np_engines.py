import pandas as pd
import numpy as np
from typing import Any
from tqdm import tqdm

from .np_base import BaseLocalEngine, BaseKernel


class LocalPolynomialEngine(BaseLocalEngine):
    """
    Local polynomial regression engine.

    Handles Nadaraya-Watson (order 0), Local Linear (order 1),
    and Local Quadratic (order 2) regression.
    """

    def __init__(self, order: int):
        if order not in [0, 1, 2]:
            raise ValueError("Order must be 0, 1, or 2.")
        self.order = order

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        h: float,
        kernel: BaseKernel,
    ) -> Any:
        """
        Fits the local model at each evaluation point in X_eval.

        For each point x_eval in X_eval, this method solves a weighted
        least squares problem using training data points close to x_eval.
        """
        X_train_np = X_train.values
        y_train_np = y_train.values
        X_eval_np = X_eval.values
        n_eval = X_eval_np.shape[0]
        predictions = np.zeros(n_eval)

        # Use tqdm for a progress bar, as this can be slow.
        for i in tqdm(range(n_eval), desc="Local Polynomial Fit", leave=False):
            x_eval_point = X_eval_np[i, :]

            # 1. Calculate distances and kernel weights
            # Using Euclidean distance for multivariate X
            distances = np.linalg.norm(X_train_np - x_eval_point, axis=1)
            u = distances / h
            w = kernel.weight(u)
            W = np.diag(w)

            # 2. Construct the design matrix R for the polynomial
            diffs = X_train_np - x_eval_point
            R = np.ones((len(X_train_np), 1))

            if self.order >= 1:
                R = np.hstack([R, diffs])
            if self.order >= 2:
                # Add quadratic and interaction terms
                quad_terms = []
                for j in range(diffs.shape[1]):
                    for k in range(j, diffs.shape[1]):
                        quad_terms.append((diffs[:, j] * diffs[:, k])[:, np.newaxis])
                R = np.hstack([R] + quad_terms)

            # 3. Solve the weighted least squares problem: min (y - R*beta)' W (y - R*beta)
            # The solution is beta_hat = (R'WR)^-1 R'Wy
            try:
                R_T_W = R.T @ W
                inv_term = np.linalg.inv(R_T_W @ R)
                beta_hat = inv_term @ R_T_W @ y_train_np
                predictions[i] = beta_hat[0]  # The intercept is the prediction
            except np.linalg.LinAlgError:
                # If matrix is singular (e.g., not enough points in window), predict with mean
                predictions[i] = np.mean(y_train_np) if len(y_train_np) > 0 else 0

        return predictions