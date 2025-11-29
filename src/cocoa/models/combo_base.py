import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Type

from .base_model import BaseModel


class BaseConvexCombinationModel(BaseModel):
    """
    An abstract base class for models that are a convex combination of two
    sub-models, one for a "pre-break" regime and one for a "post-break" regime.

    The final prediction is a weighted average:
    y_pred = gamma * pred_pre + (1 - gamma) * pred_post
    """

    def __init__(self, break_index: int, gamma: float):
        super().__init__()
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")

        self.break_index = break_index
        self.gamma = gamma
        self.post_model_is_active = False

        # Concrete classes must define these
        self.model_pre: BaseModel
        self.model_post: BaseModel

    @abstractmethod
    def _initialize_sub_models(self) -> None:
        """
        A method to be implemented by concrete classes to instantiate
        self.model_pre and self.model_post with their specific hyperparameters.
        """
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "BaseConvexCombinationModel":
        """
        Fits the two sub-models on their respective data portions.
        """
        self._initialize_sub_models()

        X_pre = X.iloc[:self.break_index]
        y_pre = y.iloc[:self.break_index]
        self.model_pre.fit(X_pre, y_pre)

        if self.break_index < len(X):
            self.post_model_is_active = True
            X_post = X.iloc[self.break_index:]
            y_post = y.iloc[self.break_index:]
            self.model_post.fit(X_post, y_post)
        else:
            self.post_model_is_active = False

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        pred_pre = self.model_pre.predict(X)
        pred_post = self.model_post.predict(X) if self.post_model_is_active else np.zeros_like(pred_pre)

        return self.gamma * pred_pre + (1 - self.gamma) * pred_post