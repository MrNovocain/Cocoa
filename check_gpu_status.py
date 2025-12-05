
import sys
import xgboost as xgb
import torch
import numpy as np

print(f"Python version: {sys.version}")

print("\n--- XGBoost ---")
try:
    # New API for XGBoost 2.0+
    model = xgb.XGBRegressor(device="cuda", n_estimators=1, max_depth=1)
    X = np.array([[0.0]])
    y = np.array([0.0])
    model.fit(X, y)
    print("XGBoost GPU support: AVAILABLE (via device='cuda')")
except Exception as e:
    print(f"XGBoost GPU support: NOT AVAILABLE ({e})")

print("\n--- cuML (RAPIDS) ---")
try:
    import cuml
    print("cuML: INSTALLED")
except ImportError:
    print("cuML: NOT INSTALLED (Random Forest will run on CPU)")

print("\n--- PyTorch ---")
if torch.cuda.is_available():
    print(f"PyTorch GPU: AVAILABLE ({torch.cuda.get_device_name(0)})")
else:
    print("PyTorch GPU: NOT AVAILABLE")
