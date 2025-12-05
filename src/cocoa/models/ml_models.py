import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
import xgboost as xgb

from .assets import RF_PARAM_GRID, XGB_PARAM_GRID
from .base_model import BaseModel

try:
    from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor
    _HAS_CUML = True
except ImportError:  # GPU fallback to CPU implementation
    CuMLRandomForestRegressor = None
    _HAS_CUML = False


def _xgb_gpu_available() -> bool:
    """Returns True when the installed XGBoost build has CUDA support."""
    try:
        # Try to use device="cuda" on a dummy booster
        # This is more reliable than checking internal flags in newer XGBoost versions
        from xgboost import XGBRegressor
        model = XGBRegressor(device="cuda", n_estimators=1, max_depth=1)
        # We don't need to fit, just checking if the parameter is accepted might be enough?
        # Actually, let's try to fit on tiny data to be sure.
        X = np.array([[0.0]])
        y = np.array([0.0])
        model.fit(X, y)
        return True
    except Exception:
        return False


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
        return np.asarray(self.model.predict(X))

    def get_params(self) -> Dict[str, Any]:
        """Returns the hyperparameters of the model."""
        return self.hyperparams


class RFModel(BaseSklearnModel):
    """Random Forest model that prefers a GPU implementation when available."""

    def __init__(
        self,
        n_estimators: int = RF_PARAM_GRID["n_estimators"][0],
        max_features: float | str = RF_PARAM_GRID["max_features"][0],
        min_samples_leaf: int = RF_PARAM_GRID["min_samples_leaf"][0],
        random_state: int = 42,
        use_gpu: bool = True,
        **kwargs: Any
    ):
        hyperparams = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            **kwargs
        }
        use_cuml = use_gpu and _HAS_CUML
        model_class = CuMLRandomForestRegressor if use_cuml else SklearnRandomForestRegressor

        # scikit-learn CPU fallback should use all cores
        if not use_cuml:
            hyperparams.setdefault("n_jobs", -1)
        else:
            # cuML requires an integer for max_depth, cannot be None
            if hyperparams.get("max_depth") is None:
                # print("Warning: max_depth=None is not supported by cuML RF. Setting max_depth=16.")
                hyperparams["max_depth"] = 16

        self.using_gpu = use_cuml
        super().__init__(model_class=model_class, **hyperparams)


class XGBModel(BaseSklearnModel):
    """XGBoost regressor configured for GPU acceleration when available."""

    def __init__(
        self,
        n_estimators: int = XGB_PARAM_GRID["n_estimators"][0],
        max_depth: int = XGB_PARAM_GRID["max_depth"][0],
        learning_rate: float = XGB_PARAM_GRID["learning_rate"][0],
        subsample: float = XGB_PARAM_GRID["subsample"][0],
        colsample_bytree: float = XGB_PARAM_GRID["colsample_bytree"][0],
        random_state: int = 42,
        use_gpu: bool = True,
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
        gpu_enabled = use_gpu and _xgb_gpu_available()
        if gpu_enabled:
            # XGBoost 2.0+ uses device="cuda"
            hyperparams.update({"device": "cuda", "tree_method": "hist"})
        else:
            hyperparams.setdefault("tree_method", "hist")

        self.using_gpu = gpu_enabled
        super().__init__(model_class=xgb.XGBRegressor, **hyperparams)
