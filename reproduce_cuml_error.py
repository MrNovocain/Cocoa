
import numpy as np
print("Starting import...")
try:
    from cuml.ensemble import RandomForestRegressor
except ImportError:
    print("cuml not installed")
    exit(1)

print("Testing CuML RandomForestRegressor with max_depth=None")

try:
    # Reproducing the error with max_depth=None
    model = RandomForestRegressor(n_estimators=10, max_depth=None)
    print("Initialization successful")
    
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)
    
    model.fit(X, y)
    print("Fit successful")
except Exception as e:
    print(f"Caught expected error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting CuML RandomForestRegressor with max_depth=16 (default)")
try:
    model = RandomForestRegressor(n_estimators=10, max_depth=16)
    print("Initialization successful")
    model.fit(X, y)
    print("Fit successful")
except Exception as e:
    print(f"Caught error: {e}")
