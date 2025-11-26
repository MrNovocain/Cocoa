from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseKernel(ABC):
    """Base class for univariate kernels."""

    @abstractmethod
    def weight(self, u: float) -> float:
        """Computes the kernel weight for a given scaled distance u."""
        ...


class BaseLocalEngine(ABC):
    """Base class for local polynomial regression engines."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, h: float, kernel: BaseKernel) -> Any:
        """Fits the local model at evaluation points X_eval and returns predictions."""
        ...