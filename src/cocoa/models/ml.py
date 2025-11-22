"""Placeholders for ML models (RF, XGBoost, etc.) for comparison with nonparametrics."""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def fit_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Simple Random Forest baseline for Cocoa."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict_model(model: Any, X_new: pd.DataFrame) -> pd.Series:
    """Wrapper predicting from any sklearn-like model."""
    return pd.Series(model.predict(X_new), index=X_new.index)
