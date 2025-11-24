import pandas as pd
import sklearn


def naive_forecast(y: pd.Series, horizon: int = 1) -> pd.Series:
    """Simple random-walk / last-value baseline."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return y.shift(horizon)
