"""Placeholders for ML models (RF, XGBoost, etc.) for comparison with nonparametrics."""

from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel


class RandomForestModel(BaseModel):
    """A wrapper for the scikit-learn RandomForestRegressor to conform to the BaseModel interface."""

    def __init__(self, **kwargs: Any):
        """
        Initializes the RandomForestRegressor.
        Passes any keyword arguments (e.g., n_estimators, random_state) to the regressor.
        """
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fits the Random Forest model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Makes predictions using the fitted Random Forest model."""
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Gets parameters for this estimator."""
        return self.model.get_params(deep=deep)
