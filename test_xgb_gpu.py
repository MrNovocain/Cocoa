
import xgboost as xgb
import numpy as np

print(f"XGBoost version: {xgb.__version__}")

# Create dummy data
X = np.random.rand(100, 10)
y = np.random.rand(100)

try:
    model = xgb.XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor")
    model.fit(X, y)
    print("XGBoost GPU training successful!")
except Exception as e:
    print(f"XGBoost GPU training failed: {e}")
