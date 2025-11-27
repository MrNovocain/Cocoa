import pandas as pd
import numpy as np
from typing import Type

from .base_model import BaseModel
from .np_regime import NPRegimeModel


class WLLModel(BaseModel):
    """
    Weighted Local Linear (WLL) Estimator.

    This model implements the CGS (2025) weighted estimator by creating a convex
    combination of two sub-models: one fitted on pre-break data and one on
    post-break data.

    y_hat = gamma * y_hat_pre + (1 - gamma) * y_hat_post

    The weight `gamma` is a hyperparameter, intended to be tuned via MFV
    cross-validation to balance the bias-variance trade-off.
    """

    def __init__(
        self,
        pre_model_class: Type[NPRegimeModel],
        post_model_class: Type[NPRegimeModel],
        pre_model_params: dict,
        post_model_params: dict,
        break_idx: int,
        gamma: float,
    ):
        super().__init__()
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be between 0 and 1.")

        # Store hyperparameters needed for cloning
        self.hyperparams = {
            "pre_model_class": pre_model_class,
            "post_model_class": post_model_class,
            "pre_model_params": pre_model_params,
            "post_model_params": post_model_params,
            "break_idx": break_idx,
            "gamma": gamma,
        }

        self.pre_model = pre_model_class(**pre_model_params)
        self.post_model = post_model_class(**post_model_params)
        self.break_idx = break_idx
        self.gamma = gamma

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "WLLModel":
        """
        Fits the pre- and post-break sub-models on their respective data regimes.
        """
        if self.break_idx >= len(X):
            raise ValueError(f"break_idx {self.break_idx} is out of bounds for training data of length {len(X)}.")

        # 1. Split training data into pre- and post-break regimes
        X_pre, y_pre = X.iloc[:self.break_idx], y.iloc[:self.break_idx]
        X_post, y_post = X.iloc[self.break_idx:], y.iloc[self.break_idx:]

        # 2. Fit the sub-models
        if not X_pre.empty:
            self.pre_model.fit(X_pre, y_pre)
        
        if not X_post.empty:
            self.post_model.fit(X_post, y_post)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("WLLModel must be fitted before calling predict().")

        pred_pre = self.pre_model.predict(X) if self.pre_model.is_fitted else np.zeros(len(X), dtype=np.float32)
        pred_post = self.post_model.predict(X) if self.post_model.is_fitted else np.zeros(len(X), dtype=np.float32)

        # Ensure predictions are numpy arrays on the CPU before combining
        if hasattr(pred_pre, "cpu"):  # Check if it's a PyTorch tensor
            pred_pre = pred_pre.cpu().numpy()
        if hasattr(pred_post, "cpu"):
            pred_post = pred_post.cpu().numpy()

        # Return the convex combination of predictions
        return self.gamma * pred_pre + (1.0 - self.gamma) * pred_post