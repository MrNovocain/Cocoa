import numpy as np
import pandas as pd


def mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    return float(np.mean((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2))


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    return float(np.mean(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1])))


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
