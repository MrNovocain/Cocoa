import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from itertools import product

# 1. Load and sort data
df = pd.read_csv("data/processed/cocoa_ghana_full.csv")

# Ensure date is datetime and sorted
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 2. Define features and target
feature_cols = [
    "PRCP_anom_mean",
    "TAVG_anom_mean",
    "PRCP_anom_std",
    "TAVG_anom_std",
    "N_stations",
    # "log_price_lagt",   # lagged price: allowed as predictor
]
target_col = "log_price"

X = df[feature_cols].copy()
y = df[target_col].copy()

# 3. Choose a final OOS test window
#    Define the start date for the OOS test window
oos_test_start_date = pd.to_datetime("2024-11-29") 

# Determine the index where the OOS test period begins
test_start_idx = df[df["date"] >= oos_test_start_date].index[0]
T_test = len(df) - test_start_idx # Calculate T_test based on the start date

X_train_cv = X.iloc[:test_start_idx].reset_index(drop=True)
y_train_cv = y.iloc[:test_start_idx].reset_index(drop=True)

X_test = X.iloc[-T_test:].reset_index(drop=True)
y_test = y.iloc[-T_test:].reset_index(drop=True)

T_train = len(X_train_cv)


# 4. MFV-style forward CV for RF

def mfv_score_rf(X_train, y_train, params, Q=5, block_size=200):
    """
    Compute MFV-style validation MSE for RF with given hyperparams.
    
    X_train, y_train: training+CV region (already in time order).
    Q: number of folds (blocks).
    block_size: length of each validation block (m in CGS notation).

    Returns: MFV MSE (float).
    """
    T = len(X_train)
    total_block_length = Q * block_size
    if total_block_length >= T:
        raise ValueError("Q * block_size must be smaller than training length.")

    # The last Q*block_size points form the CV region
    # We'll cut that region into Q consecutive blocks.
    cv_start = T - total_block_length  # index where CV region starts

    sq_errors = []
    
    for q in range(Q):
        # q-th block within the CV region
        val_start = cv_start + q * block_size
        val_end = val_start + block_size  # exclusive

        # Training indices: everything strictly before the validation block
        # (this is an expanding-window scheme)
        train_end = val_start  # exclusive
        X_tr = X_train.iloc[:train_end]
        y_tr = y_train.iloc[:train_end]
        
        X_val = X_train.iloc[val_start:val_end]
        y_val = y_train.iloc[val_start:val_end]

        # Fit RF with current hyperparameters
        rf = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)

        # Forecast and collect squared errors
        y_pred = rf.predict(X_val)
        sq_errors.extend((y_val.values - y_pred) ** 2)

    # MFV objective = average squared forecast error over all folds & points
    return float(np.mean(sq_errors))


# 5. Hyperparameter grid for RF

param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 5, 20],
    "max_features": ["sqrt", "log2"],
}

# Turn dict-of-lists into a list of param dicts
def expand_grid(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

# 6. Search for MFV-optimal RF hyperparameters

best_params = None
best_mfv = np.inf

Q = 5          # number of folds
block_size = 200  # validation block length m

for params in expand_grid(param_grid):
    mfv = mfv_score_rf(X_train_cv, y_train_cv, params, Q=Q, block_size=block_size)
    print("Params:", params, "MFV MSE:", mfv)
    if mfv < best_mfv:
        best_mfv = mfv
        best_params = params

print("Best params:", best_params)
print("Best MFV MSE:", best_mfv)

# 7. Fit final RF on the full training+CV region with best_params
rf_final = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X_train_cv, y_train_cv)

# 8. Evaluate true OOS MSFE on the final test window
y_test_pred = rf_final.predict(X_test)
msfe_oos = mean_squared_error(y_test, y_test_pred)
print("Final OOS MSFE (RF):", msfe_oos)

# 9. Plot point forecasting result against real data for the test set
all_dates = df["date"]
all_y = df[target_col]
test_dates = df["date"].iloc[-T_test:]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 7))

# Plot the entire time series of actual data
ax.plot(all_dates, all_y, label="Actual Log Return (Full History)", linestyle='-', color='black', linewidth=1.0)

# Overlay the point forecast for the last 30 days
ax.plot(test_dates, y_test_pred, label="RF Point Forecast (Last 30 days)", linestyle='--', color='firebrick', marker='x')

ax.set_title("Cocoa Log Return: Full History and 30-Day Forecast", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Log Return", fontsize=12)
ax.legend(fontsize=10)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
