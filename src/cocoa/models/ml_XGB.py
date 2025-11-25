from typing import Dict, Any, Literal

from xgboost import XGBRegressor
from .base_model import BaseModel


class XGBModel(BaseModel):
    """
    XGBoost regression model implementing the BaseModel interface.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

        self._xgb = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y, sample_weight=None) -> "XGBModel":
        self._xgb.fit(X, y, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("XGBModel must be fitted before calling predict().")
        return self._xgb.predict(X)