import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterable, Tuple, Type, Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import product
from .base_model import BaseModel

class RFModel(BaseModel):
    """
    Random Forest regression model implementing the BaseModel interface.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | float | Literal["sqrt", "log2"] | None = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state= random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y, sample_weight=None) -> "RFModel":
        self._rf.fit(X, y, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("RFModel must be fitted before calling predict().")
        return self._rf.predict(X)