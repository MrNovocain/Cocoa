"""
Project-level constants for the cocoa forecasting project.

This includes feature sets, target variables, and hyperparameter grids
for different models.
"""

# ============================================================
# General Experiment Setup
# ============================================================
BREAK_ID_ONE_BASED = 6130 #3594 previous last break date 
# BREAK_DATE = "2022-07-05"
OOS_START_DATE = "2025-01-02"
PROCESSED_DATA_PATH = "w:/Research/NP/Cocoa/data/processed/cocoa_ghana_full.csv"

DEFAULT_FEATURE_COLS = [
    "PRCP_anom_mean",
    # "TAVG_anom_mean",
    # "PRCP_anom_std",
    # "TAVG_anom_std",
    "log_price_lagt",
    # "log_price_lag2",
    # "N_stations",
]
# DEFAULT_TARGET_COL = "log_return"
DEFAULT_TARGET_COL = "log_return_forecast_target"


Q_VALUE = 4  # Number of block for MFV validation


# ============================================================
# Random Forest (RF) Model Configuration
# ============================================================

RF_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [5, None],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt"],
}

# ============================================================
# XGBoost (XGB) Model Configuration
# ============================================================

XGB_FEATURE_COLS = [
    "PRCP_anom_mean",
    "TAVG_anom_mean",
    "PRCP_anom_std",
    "N_stations",
    "log_price_lagt",
]
XGB_TARGET_COL = "log_return"

XGB_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}

# NP_BANDWIDTH_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]



### Grid search for least bias break date is"2019-07-15"
BREAK_DATE_LB_DATE = "2019-07-15"