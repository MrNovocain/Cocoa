
import pandas as pd
import numpy as np

from .base_model import BaseModel
from .np_regime import NPRegimeModel


class NPConvexCombinationModel(BaseModel):
    """
    A convex combination of two non-parametric models.

    This model combines the predictions of two sub-models:
    - model_pre: trained on data before `post_start_index`.
    - model_post: trained on data from `post_start_index` onwards.

    The final prediction is a weighted average:
    y_pred = gamma * pred_pre + (1 - gamma) * pred_post
    """

    def __init__(
        self,
        model_pre: NPRegimeModel,
        model_post: NPRegimeModel,
        post_start_index: int,
        gamma: float,
    ):
        super().__init__()
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")

        # Clone unfitted models to ensure this model is self-contained
        self.model_pre = model_pre.clone()
        self.model_post = model_post.clone()
        self.post_start_index = post_start_index
        self.gamma = gamma
        self.hyperparams = {
            'gamma': gamma,
            'post_start_index': post_start_index,
            'model_pre': model_pre,
            'model_post': model_post
        }
        self.pre_model_is_active = False
        self.post_model_is_active = False


    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "NPConvexCombinationModel":
        """
        Fits the pre-break and post-break sub-models on their respective data portions.
        """
        # Fit the "pre" model on data before the break
        if self.post_start_index > 0:
            self.pre_model_is_active = True
            X_pre = X.iloc[:self.post_start_index]
            y_pre = y.iloc[:self.post_start_index]
            self.model_pre.fit(X_pre, y_pre)

        # Fit the "post" model on data after the break
        if self.post_start_index < len(X):
            self.post_model_is_active = True
            X_post = X.iloc[self.post_start_index:]
            y_post = y.iloc[self.post_start_index:]
            self.model_post.fit(X_post, y_post)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts by taking a convex combination of the two sub-models.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        # Handle edge cases where one of the models was not fitted
        if not self.pre_model_is_active and not self.post_model_is_active:
            raise RuntimeError("Neither pre nor post models were fitted.")
        
        if not self.pre_model_is_active:
            return self.model_post.predict(X) # Only post-break data was available
        
        if not self.post_model_is_active:
            return self.model_pre.predict(X) # Only pre-break data was available

        # If both are active, compute the convex combination
        pred_pre = self.model_pre.predict(X)
        pred_post = self.model_post.predict(X)

        return self.gamma * pred_pre + (1 - self.gamma) * pred_post

    @staticmethod
    def predict_from_sub_models(
        gamma: float,
        pred_pre: np.ndarray,
        pred_post: np.ndarray
    ) -> np.ndarray:
        """
        A static method to compute the convex combination from pre-computed predictions.
        This is useful for efficiently tuning gamma without re-fitting models.
        """
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")
        
        return gamma * pred_pre + (1 - gamma) * pred_post
