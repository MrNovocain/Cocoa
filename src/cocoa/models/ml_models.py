import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from .assets import RF_PARAM_GRID, XGB_PARAM_GRID
from .base_model import BaseModel


class BaseSklearnModel(BaseModel):
    """
    An abstract base class for models that wrap a scikit-learn compatible model.
    This class provides a template for classic machine learning models.
    """

    def __init__(self, model_class: Any, **hyperparams: Any):
        super().__init__()
        self.model = model_class(**hyperparams)
        self.hyperparams = hyperparams

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> "BaseSklearnModel":
        """
        Fits the underlying scikit-learn model.
        """
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the fitted scikit-learn model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Returns the hyperparameters of the model."""
        return self.hyperparams


class RFModel(BaseSklearnModel):
    """A wrapper for the scikit-learn RandomForestRegressor."""

    def __init__(
        self,
        n_estimators: int = RF_PARAM_GRID["n_estimators"][0],
        max_features: float | str = RF_PARAM_GRID["max_features"][0],
        min_samples_leaf: int = RF_PARAM_GRID["min_samples_leaf"][0],
        random_state: int = 42,
        **kwargs: Any
    ):
        hyperparams = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            **kwargs
        }
        super().__init__(model_class=RandomForestRegressor, **hyperparams)


class XGBModel(BaseSklearnModel):
    """A wrapper for the XGBoost XGBRegressor."""

    def __init__(
        self,
        n_estimators: int = XGB_PARAM_GRID["n_estimators"][0],
        max_depth: int = XGB_PARAM_GRID["max_depth"][0],
        learning_rate: float = XGB_PARAM_GRID["learning_rate"][0],
        subsample: float = XGB_PARAM_GRID["subsample"][0],
        colsample_bytree: float = XGB_PARAM_GRID["colsample_bytree"][0],
        random_state: int = 42,
        **kwargs: Any
    ):
        hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": 'reg:squarederror', # Common default
            **kwargs
        }
        super().__init__(model_class=xgb.XGBRegressor, **hyperparams)