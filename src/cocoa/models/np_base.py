from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseKernel(ABC):
    """Base class for univariate kernels."""

    @abstractmethod
    def weight(self, u: float) -> float:
        """Computes the kernel weight for a given scaled distance u."""
        ...


class BaseBandwidthSelector(ABC):
    """Base class for bandwidth selection methods."""

    @abstractmethod
    def select_bandwidth(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Selects the optimal bandwidth."""
        ...


class BaseLocalEngine(ABC):
    """Base class for local polynomial regression engines."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame, h: float, kernel: BaseKernel) -> Any:
        """Fits the local model at evaluation points X_eval and returns predictions."""
        ...