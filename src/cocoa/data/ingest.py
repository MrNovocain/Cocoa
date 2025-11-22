from pathlib import Path
import pandas as pd

from ..paths import PATHS
from ..config import settings


def load_cocoa_raw(filename: str = "cocoa_raw.csv") -> pd.DataFrame:
    """Load raw cocoa data from data/raw."""
    path = PATHS.data_raw / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    df[settings.date_column] = pd.to_datetime(df[settings.date_column])
    df = df.sort_values(settings.date_column).reset_index(drop=True)
    return df
