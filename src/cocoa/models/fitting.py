from cocoa.models import CocoaDataset, RFModel, MFVValidator, plot_forecast
from itertools import product
import numpy as np
import pandas as pd

def expand_grid(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

if __name__ == "__main__":
    # 1. Construct dataset
    feature_cols = [
        "PRCP_anom_mean",
        "TAVG_anom_mean",
        "PRCP_anom_std",
        "TAVG_anom_std",
        "N_stations",
    ]
    target_col = "log_price"

    dataset = CocoaDataset(
        csv_path="w:/Research/NP/Cocoa/data/processed/cocoa_ghana_full.csv",
        feature_cols=feature_cols,
        target_col=target_col,
    )

    # 2. Train / OOS split
    split = dataset.split_oos_by_date("2024-11-29")

    X_train_cv = split.X_train
    y_train_cv = split.y_train
    X_test = split.X_test
    y_test = split.y_test
    T_train = split.T_train
    T_test = split.T_test

    print(T_train, T_test)

    # 3. MFV RF tuning
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", "log2"],
    }

    param_list = list(expand_grid(param_grid))

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
        target_col=target_col,
        y_pred=y_full_pred,
        model_label="Random Forest",
        output_path="data/processed/rf_oos_forecast.png",
    )
