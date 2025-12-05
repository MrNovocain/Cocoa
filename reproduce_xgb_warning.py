import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from cocoa.models.ml_models import XGBModel
except ImportError as e:
    print(f"Error importing XGBModel: {e}")
    sys.exit(1)

print("Checking libraries...")
try:
    import cupy
    print(f"cupy version: {cupy.__version__}")
except ImportError:
    print("cupy not found")

try:
    import cudf
    print(f"cudf version: {cudf.__version__}")
except ImportError:
    print("cudf not found")

try:
    import cuml
    print(f"cuml version: {cuml.__version__}")
except ImportError:
    print("cuml not found")

print(f"XGBoost version: {xgb.__version__}")

# Create dummy data
X = pd.DataFrame(np.random.rand(100, 10), columns=[f"col_{i}" for i in range(10)])
y = pd.Series(np.random.rand(100))

print("\nInitializing XGBModel...")
model = XGBModel(use_gpu=True, n_estimators=10, max_depth=3)
print(f"Model using GPU: {model.using_gpu}")

print("\nFitting model...")
model.fit(X, y)

print("\nPredicting...")
try:
    preds = model.predict(X)
    print("Prediction successful")
except Exception as e:
    print(f"Prediction failed: {e}")
