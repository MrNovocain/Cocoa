from cocoa.models import CocoaDataset, XGBModel, MFVValidator, plot_forecast
from itertools import product
import numpy as np


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

    print(f"Train/CV set size: {T_train}, OOS test set size: {T_test}")

    # 3. MFV XGBoost tuning
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
    }

    param_list = list(expand_grid(param_grid))

    mfv = MFVValidator(Q=5, block_size=200)
    best_params, best_mfv, _ = mfv.grid_search(
        model_class=XGBModel,
        X_train=X_train_cv,
        y_train=y_train_cv,
        param_grid=param_list,
    )

    # 4. Fit final XGB and evaluate on OOS
    xgb_final = XGBModel(**best_params)
    xgb_final.fit(X_train_cv, y_train_cv)
    y_train_pred = xgb_final.predict(X_train_cv)
    y_test_pred = xgb_final.predict(X_test)
    y_full_pred = np.concatenate([y_train_pred, y_test_pred])

    # 5. Plot OOS forecast
    plot_forecast(
        df=dataset.df,
        target_col=target_col,
        y_pred=y_full_pred,
        model_label="XGBoost",
        output_path="data/processed/xgb_oos_forecast.png",
    )