"""Defines the abstract base class for all models."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Abstract Base Class for all forecasting models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model to the training data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions on new data."""
        raise NotImplementedError