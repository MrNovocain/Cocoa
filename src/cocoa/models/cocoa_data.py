import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .data_types import TrainTestSplit as BaseTrainTestSplit


@dataclass
class TrainTestSplit(BaseTrainTestSplit):
    T_train: int
    T_test: int
    # This inherits X_train, y_train, X_test, y_test from the base class
    # and adds the time period lengths.


class CocoaDataset:
    """Convenience wrapper around the cocoa+Ghana weather panel.

    Responsibilities:
    - Load and clean the CSV.
    - Expose X, y, dates.
    - Provide standard train/OOS split by date.
    - Provide simple time-window slicing utilities.
    """

    def __init__(
        self,
        csv_path: str,
        feature_cols: List[str],
        target_col: str,
        start_date: Optional[str | pd.Timestamp] = None,
    ) -> None:
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df["date"] >= start_date]

        df = df.reset_index(drop=True)

        self.df = df
        self.dates = df["date"].copy()
        self.X = df[feature_cols].copy()
        self.y = df[target_col].copy()

    def trim_data_by_start_date(self, start_date: Optional[str | pd.Timestamp]=None):
        """Discard observations before a given start date."""
        if start_date is None:
            print("no break assumed")
            pass # No trimming needed
        else:
            start_date = pd.to_datetime(start_date)
            mask = self.df["date"] >= start_date
            self.df = self.df[mask].reset_index(drop=True)
            self.dates = self.df["date"].copy()
            self.X = self.df[self.feature_cols].copy()
            self.y = self.df[self.target_col].copy()
            print(f"Data trimmed to start from {start_date.date()}. New length: {len(self.df)}")

    def get_window(self, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X_window, y_window) for start_date <= date <= end_date.
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        mask = (self.dates >= start_date) & (self.dates <= end_date)
        X_w = self.X.loc[mask].reset_index(drop=True)
        y_w = self.y.loc[mask].reset_index(drop=True)
        return X_w, y_w

    def get_date_from_1_based_index(self, index_1_based: int) -> pd.Timestamp:
        """
        Returns the date corresponding to a 1-based index in the dataset.
        """
        if not (1 <= index_1_based <= len(self.dates)):
            raise ValueError(f"1-based index {index_1_based} is out of bounds for dataset of length {len(self.dates)}.")
        return self.dates.iloc[index_1_based - 1]

    def get_1_based_index_from_date(self, date: str | pd.Timestamp) -> int:
        """
        Returns the 1-based index corresponding to a given date.
        Finds the first occurrence if dates are not unique.
        """
        date = pd.to_datetime(date)
        matches = self.dates[self.dates.dt.date == date.date()]
        if matches.empty:
            raise ValueError(f"Date {date.date()} not found in the dataset.")
        return matches.index[0] + 1
