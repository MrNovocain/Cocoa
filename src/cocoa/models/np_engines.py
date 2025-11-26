import pandas as pd
import numpy as np
from typing import Any

import torch
from .np_base import BaseLocalEngine, BaseKernel


class LocalPolynomialEngine(BaseLocalEngine):
    """
    Local polynomial regression engine.

    Handles Nadaraya-Watson (order 0), Local Linear (order 1),
    and Local Quadratic (order 2) regression.
    """

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

        # CHANGE 1: Regularization (Jitter) - This is the key fix.
        # This is a standard technique (Ridge Regression) to ensure the matrix is
        # invertible, preventing most failures without significantly biasing the result.
        jitter = 1e-8
        R_T_W_R += torch.eye(R_T_W_R.shape[-1], device=self.device) * jitter

        # Solve the linear system for beta_hat for all eval points at once.
        beta_hat = torch.linalg.solve(R_T_W_R, R_T_W_y) # (n_eval, n_poly, 1)
        predictions = beta_hat[:, 0, 0]

        # CHANGE 2: A Better, Localized Fallback
        # Instead of a global mean, we now check for the rare remaining failures
        # and use a much more sensible local estimate: the nearest neighbor's value.
        failed_solves = ~torch.isfinite(predictions)
        if torch.any(failed_solves):
            failed_indices = torch.where(failed_solves)[0]
            nearest_neighbor_indices = torch.argmin(distances[failed_indices], dim=1)
            predictions[failed_solves] = y_train_t[nearest_neighbor_indices]

        # 4. Return predictions as a numpy array
        return predictions.cpu().numpy()