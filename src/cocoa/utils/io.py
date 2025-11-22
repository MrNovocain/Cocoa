from pathlib import Path

import pandas as pd

from ..paths import PATHS


def save_dataframe(df: pd.DataFrame, filename: str, subdir: str = "processed") -> Path:
    """Save a DataFrame under data/<subdir>/<filename>."""
    if subdir == "processed":
        base = PATHS.data_processed
    elif subdir == "interim":
        base = PATHS.data_interim
    elif subdir == "raw":
        base = PATHS.data_raw
    else:
        raise ValueError(f"Unknown subdir: {subdir}")

    path = base / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
