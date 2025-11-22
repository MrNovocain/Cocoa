import numpy as np
import pandas as pd

from ..config import settings


def preprocess_cocoa(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing for Cocoa data."""
    df = df.sort_values(settings.date_column).reset_index(drop=True)

    if settings.target_column not in df.columns:
        if "price" not in df.columns:
            raise KeyError(
                f"Expected either {settings.target_column} or 'price' in columns: {df.columns}"
            )
        df[settings.target_column] = np.log(df["price"])

    df = df.dropna(subset=[settings.date_column, settings.target_column])
    return df
