import pandas as pd
import numpy as np
from typing import Any
from tqdm import tqdm

import torch
from .np_base import BaseLocalEngine, BaseKernel


class LocalPolynomialEngine(BaseLocalEngine):
    """
    Local polynomial regression engine.

    Handles Nadaraya-Watson (order 0), Local Linear (order 1),
    and Local Quadratic (order 2) regression.
    """

    def __init__(self, order: int):
    def __init__(self, order: int, use_gpu: bool = True):
        if order not in [0, 1, 2]:
            raise ValueError("Order must be 0, 1, or 2.")
        self.order = order

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        if use_gpu and not torch.cuda.is_available():
            print("Warning: GPU not available, falling back to CPU.")
        print(f"LocalPolynomialEngine will use device: {self.device}")
 
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
        # 1. Convert pandas to torch tensors and move to the selected device
        X_train_t = torch.from_numpy(X_train.values).float().to(self.device)
        y_train_t = torch.from_numpy(y_train.values).float().to(self.device)
        X_eval_t = torch.from_numpy(X_eval.values).float().to(self.device)
        
        n_train, d = X_train_t.shape
        n_eval = X_eval_t.shape[0]
        
        # Expand dimensions for broadcasting
        # X_train_t: (n_train, d) -> (1, n_train, d)
        # X_eval_t: (n_eval, d) -> (n_eval, 1, d)
        X_train_exp = X_train_t.unsqueeze(0)
        X_eval_exp = X_eval_t.unsqueeze(1)
        
        # 2. Batched calculations for all evaluation points
        # This avoids the Python loop and leverages GPU parallelism.
        
        # Calculate differences and distances
        # diffs will have shape (n_eval, n_train, d)
        diffs = X_train_exp - X_eval_exp
        # distances will have shape (n_eval, n_train)
        distances = torch.linalg.norm(diffs, ord=2, dim=2)
        
        # Calculate kernel weights
        u = distances / h
        # The kernel function needs to handle torch tensors now.
        # Assuming kernel.weight can take a numpy array, we move u to CPU.
        # For full GPU acceleration, the kernel itself should be implemented in torch.
        w_np = kernel.weight(u.cpu().numpy())
        W = torch.from_numpy(w_np).float().to(self.device) # W has shape (n_eval, n_train)
        
        # Construct the design matrix R for each evaluation point
        # R will have shape (n_eval, n_train, n_poly_terms)
        R_list = [torch.ones(n_eval, n_train, 1, device=self.device)]
        if self.order >= 1:
            R_list.append(diffs)
        if self.order >= 2:
            # Add quadratic and interaction terms
            for j in range(d):
                for k in range(j, d):
                    quad_term = (diffs[:, :, j] * diffs[:, :, k]).unsqueeze(2)
                    R_list.append(quad_term)
        R = torch.cat(R_list, dim=2)
        
        # 3. Solve the batched weighted least squares problem
        # (R'WR)^-1 R'Wy for each evaluation point
        R_T = R.transpose(1, 2)
        R_T_W = R_T * W.unsqueeze(1) # Element-wise multiplication, shape (n_eval, n_poly, n_train)
        R_T_W_R = R_T_W @ R # (n_eval, n_poly, n_poly)
        R_T_W_y = R_T_W @ y_train_t.unsqueeze(1) # (n_eval, n_poly, 1)
        
        # Solve the linear system for beta_hat for all eval points at once
        try:
            beta_hat = torch.linalg.solve(R_T_W_R, R_T_W_y) # (n_eval, n_poly, 1)
            # The prediction is the first component (intercept) of beta_hat
            predictions = beta_hat[:, 0, 0]
        except torch.linalg.LinAlgError:
            print("Warning: Singular matrix encountered. Using global mean as fallback.")
            # Fallback for all predictions if batch solve fails
            mean_pred = y_train_t.mean()
            predictions = torch.full((n_eval,), mean_pred, device=self.device)
            
        # 4. Return predictions as a numpy array
        return predictions.cpu().numpy()