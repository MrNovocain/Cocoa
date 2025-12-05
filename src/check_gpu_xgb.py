import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

def check_gpu_xgb():
    print("Checking XGBoost GPU support...")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Parameters for GPU training
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',  # Use GPU
        'eval_metric': 'logloss'
    }
    
    try:
        start_time = time.time()
        bst = xgb.train(params, dtrain, num_boost_round=100)
        end_time = time.time()
        
        print(f"Training completed in {end_time - start_time:.4f} seconds.")
        
        # Predict
        preds = bst.predict(dtest)
        print(f"Prediction shape: {preds.shape}")
        print("SUCCESS: XGBoost trained on GPU.")
        
    except Exception as e:
        print(f"FAILURE: XGBoost GPU training failed. Error: {e}")

if __name__ == "__main__":
    check_gpu_xgb()
