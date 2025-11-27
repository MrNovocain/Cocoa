

import pandas as pd
import numpy as np

from .base_model import BaseModel
from .np_base import BaseKernel, BaseLocalEngine


class NPRegimeModel(BaseModel):
    """
    A non-parametric regressor for a specific data regime.

    This model uses a kernel-based local approximation method. It is "fitted"
    by storing the training data of its regime. Predictions for new points
    are generated on-the-fly using a specified local engine (e.g., local linear).
    """

    def __init__(
        self,
        kernel: BaseKernel,
        local_engine: BaseLocalEngine,
        bandwidth: float | None = None,
    ):
        super().__init__()
        if bandwidth is None:
            raise ValueError("Must provide a fixed 'bandwidth'.")

        self.kernel = kernel
        self.local_engine = local_engine
        self.h = bandwidth  # Bandwidth
        self.hyperparams = {
            "kernel": self.kernel,
            "local_engine": self.local_engine,
            "bandwidth": self.h,
        }

        # Training data for the regime will be stored here
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "NPRegimeModel":
        """
        "Fits" the model by storing the regime's training data and selecting bandwidth.
        """
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self._X_train is None or self._y_train is None or self.h is None:
            raise RuntimeError("NPRegimeModel must be fitted before calling predict().")

        # Use the local engine to compute predictions for the new data points X
        predictions = self.local_engine.fit(self._X_train, self._y_train, X, self.h, self.kernel)
        return predictions


        