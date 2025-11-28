
import pandas as pd
import numpy as np

from .base_model import BaseModel
from .np_regime import NPRegimeModel


from .np_base import BaseKernel, BaseLocalEngine

class NPConvexCombinationModel(BaseModel):
    """
    A convex combination of two non-parametric models.

    This model combines the predictions of two sub-models:
    - model_full: trained on the entire dataset.
    - model_pre: trained on data before `break_index`.
    - model_post: trained on data from `break_index` onwards.

    The final prediction is a weighted average:
    y_pred = gamma * pred_pre + (1 - gamma) * pred_post
    """

    def __init__(
        self,
        kernel: BaseKernel,
        local_engine: BaseLocalEngine,
        pre_bandwidth: float,
        post_bandwidth: float,
        break_index: int,
        gamma: float,
    ):
        super().__init__()
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be between 0 and 1.")

        self.kernel = kernel
        self.local_engine = local_engine
        self.pre_bandwidth = pre_bandwidth
        self.post_bandwidth = post_bandwidth

        self.model_pre = NPRegimeModel(kernel=self.kernel, local_engine=self.local_engine, bandwidth=self.pre_bandwidth)
        self.model_post = NPRegimeModel(kernel=self.kernel, local_engine=self.local_engine, bandwidth=self.post_bandwidth)
        self.break_index = break_index
        self.gamma = gamma
        self.hyperparams = {
            'gamma': gamma,
            'break_index': break_index,
            'pre_bandwidth': pre_bandwidth,
            'post_bandwidth': post_bandwidth,
            'kernel': kernel,
            'local_engine': local_engine
        }
        self.post_model_is_active = False


    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "NPConvexCombinationModel":
        """
        Fits the two sub-models on their respective data portions.
        """
        # Fit the "pre" model on data before the break
        X_pre = X.iloc[:self.break_index]
        y_pre = y.iloc[:self.break_index]
        self.model_pre.fit(X_pre, y_pre)

        if self.break_index < len(X):
            # Fit the "post" model on the tail of the data
            self.post_model_is_active = True
            X_post = X.iloc[self.break_index:]
            y_post = y.iloc[self.break_index:]
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

        pred_pre = self.model_pre.predict(X)

        if not self.post_model_is_active:
            return pred_pre

        pred_post = self.model_post.predict(X)

        return self.gamma * pred_pre + (1 - self.gamma) * pred_post
