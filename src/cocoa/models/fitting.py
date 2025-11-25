from itertools import product
import numpy as np
import pandas as pd

from cocoa.models import CocoaDataset, RFModel, MFVValidator, plot_forecast
from .assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    RF_FEATURE_COLS,
    RF_TARGET_COL,
    RF_PARAM_GRID,
)

def expand_grid(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

if __name__ == "__main__":
    # 1. Construct dataset
    dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=RF_FEATURE_COLS,
        target_col=RF_TARGET_COL,
    )

    # 2. Train / OOS split
    split = dataset.split_oos_by_date(OOS_START_DATE)

    X_train_cv = split.X_train
    y_train_cv = split.y_train
    X_test = split.X_test
    y_test = split.y_test
    T_train = split.T_train
    T_test = split.T_test

    print(T_train, T_test)

    # 3. MFV RF tuning
    param_list = list(expand_grid(RF_PARAM_GRID))

    mfv = MFVValidator(Q=5, block_size=200)
    best_params, best_mfv, _ = mfv.grid_search(
        model_class=RFModel,
        X_train=X_train_cv,
        y_train=y_train_cv,
        param_grid=param_list,
    )

    # 4. Fit final RF and evaluate on OOS
    rf_final = RFModel(**best_params)
    rf_final.fit(X_train_cv, y_train_cv)
    y_train_pred = rf_final.predict(X_train_cv)
    y_test_pred = rf_final.predict(X_test)
    y_full_pred = np.concatenate([y_train_pred, y_test_pred])

    # 5. Plot OOS forecast
    plot_forecast(
        df=dataset.df,
        target_col=RF_TARGET_COL,
        y_pred=y_full_pred,
        model_label="Random Forest",
        output_path="data/processed/rf_oos_forecast.png",
    )
