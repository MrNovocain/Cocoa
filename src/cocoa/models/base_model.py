import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Iterable, Tuple, Type

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import product


# ============================================================
# Abstract base model
# ============================================================

class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    The interface is intentionally minimal: fit + predict.
    """

    def __init__(self) -> None:
        self.hyperparams: Dict[str, Any] = {}
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X, y, sample_weight=None) -> "BaseModel":
        """Fit the model on a given training sample."""
        ...

    @abstractmethod
    def predict(self, X):
        """Predict target values for new covariates X."""
        ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def clone(self) -> "BaseModel":
        """Return a fresh, unfitted copy with the same hyperparameters."""
        return self.__class__(**self.hyperparams)