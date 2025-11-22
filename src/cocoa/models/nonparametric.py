"""Placeholders for nonparametric estimators (e.g., local linear, WLL)."""

from typing import Any

import pandas as pd


def fit_local_linear(
    X: pd.DataFrame,
    y: pd.Series,
    bandwidth: float,
    **kwargs: Any,
) -> Any:
    """Skeleton for local-linear regression fitter."""
    raise NotImplementedError("Implement local-linear regression here.")


def predict_local_linear(model: Any, X_new: pd.DataFrame) -> pd.Series:
    """Predict using a fitted local-linear model."""
    raise NotImplementedError("Implement local-linear prediction here.")
