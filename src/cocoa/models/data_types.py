import pandas as pd
from dataclasses import dataclass


@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series