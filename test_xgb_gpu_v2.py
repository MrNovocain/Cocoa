
import xgboost as xgb
import numpy as np

print(f"XGBoost version: {xgb.__version__}")

# Create dummy data
X = np.random.rand(100, 10)
y = np.random.rand(100)

try:
    # New API for XGBoost 2.0+
    model = xgb.XGBRegressor(device="cuda", tree_method="hist")
    model.fit(X, y)
    print("XGBoost GPU training (device='cuda') successful!")
except Exception as e:
    print(f"XGBoost GPU training (device='cuda') failed: {e}")
