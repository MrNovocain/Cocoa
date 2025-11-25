import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
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
    ) -> None:
        self.csv_path = csv_path
        self.feature_cols = feature_cols
        self.target_col = target_col

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        self.df = df
        self.dates = df["date"].copy()
        self.X = df[feature_cols].copy()
        self.y = df[target_col].copy()

    def split_oos_by_date(self, oos_start_date: str | pd.Timestamp) -> TrainTestSplit:
        """Split into (train+CV) and final OOS test window by calendar date.

        oos_start_date: first date included in the OOS test region.
        """
        oos_start_date = pd.to_datetime(oos_start_date)
        mask_test = self.df["date"] >= oos_start_date

        if not mask_test.any():
            raise ValueError("No observations on/after the chosen OOS start date.")

        test_start_idx = self.df.index[mask_test][0]

        X_train = self.X.iloc[:test_start_idx].reset_index(drop=True)
        y_train = self.y.iloc[:test_start_idx].reset_index(drop=True)

        X_test = self.X.iloc[test_start_idx:].reset_index(drop=True)
        y_test = self.y.iloc[test_start_idx:].reset_index(drop=True)

        T_train = len(X_train)
        T_test = len(X_test)

        return TrainTestSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            T_train=T_train,
            T_test=T_test,
        )

    def get_window(self, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X_window, y_window) for start_date <= date <= end_date.

        Useful if you later want a more CGS-style regime-specific estimation.
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        mask = (self.dates >= start_date) & (self.dates <= end_date)
        X_w = self.X.loc[mask].reset_index(drop=True)
        y_w = self.y.loc[mask].reset_index(drop=True)
        return X_w, y_w
