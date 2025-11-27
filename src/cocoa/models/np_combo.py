
import pandas as pd
import numpy as np

from .base_model import BaseModel
from .np_regime import NPRegimeModel


class NPConvexCombinationModel(BaseModel):
    """
    A convex combination of two non-parametric models.

    This model combines the predictions of two sub-models:
    - model_full: trained on the entire dataset.
    - model_post: trained on data from `post_start_index` onwards.

    The final prediction is a weighted average:
    y_pred = gamma * pred_full + (1 - gamma) * pred_post
    """

    def __init__(
        self,
        model_full: NPRegimeModel,
        model_post: NPRegimeModel,
        post_start_index: int,
        gamma: float,
    ):
        super().__init__()
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")

        # Clone unfitted models to ensure this model is self-contained
        self.model_full = model_full.clone()
        self.model_post = model_post.clone()
        self.post_start_index = post_start_index
        self.gamma = gamma
        self.hyperparams = {
            'gamma': gamma,
            'post_start_index': post_start_index,
            'model_full': model_full,
            'model_post': model_post
        }
        self.post_model_is_active = False


    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "NPConvexCombinationModel":
        """
        Fits the two sub-models on their respective data portions.
        """
        # Fit the "full" model on all data
        self.model_full.fit(X, y)

        if self.post_start_index < len(X):
            # Fit the "post" model on the tail of the data
            self.post_model_is_active = True
            X_post = X.iloc[self.post_start_index:]
            y_post = y.iloc[self.post_start_index:]
            self.model_post.fit(X_post, y_post)
        else:
            self.post_model_is_active = False

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts by taking a convex combination of the two sub-models.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        pred_full = self.model_full.predict(X)

        if not self.post_model_is_active:
            return pred_full

        pred_post = self.model_post.predict(X)

        return self.gamma * pred_full + (1 - self.gamma) * pred_post
