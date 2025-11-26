"""
Project-level constants for the cocoa forecasting project.

This includes feature sets, target variables, and hyperparameter grids
for different models.
"""

# ============================================================
# General Experiment Setup
# ============================================================
Break_ID_ONE_BASED = 6441
# BREAK_DATE = "2022-07-05"
OOS_START_DATE = "2024-11-29"
PROCESSED_DATA_PATH = "w:/Research/NP/Cocoa/data/processed/cocoa_ghana_full.csv"

DEFAULT_FEATURE_COLS = [
    "PRCP_anom_mean",
    # "TAVG_anom_mean",
    # "PRCP_anom_std",
    # "TAVG_anom_std",
    "log_price_lagt",
    # "N_stations",
]
DEFAULT_TARGET_COL = "log_return"

# ============================================================
# Random Forest (RF) Model Configuration
# ============================================================

RF_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt", "log2"],
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
}

NP_BANDWIDTH_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]