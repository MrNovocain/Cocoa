"""
Project-level constants for the cocoa forecasting project.

This includes feature sets, target variables, and hyperparameter grids
for different models.
"""

# ============================================================
# General Experiment Setup
# ============================================================

OOS_START_DATE = "2024-11-29"
PROCESSED_DATA_PATH = "w:/Research/NP/Cocoa/data/processed/cocoa_ghana_full.csv"

# ============================================================
# Random Forest (RF) Model Configuration
# ============================================================

RF_FEATURE_COLS = [
    "PRCP_anom_mean",
    "TAVG_anom_mean",
    "PRCP_anom_std",
    "N_stations",
    "log_price_lagt",
]
RF_TARGET_COL = "log_return"

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