import xgboost as xgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

def check_gpu_rf():
    print("Checking Random Forest (via XGBoost) GPU support...")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Parameters for Random Forest using XGBoost
    # To simulate RF:
    # 1. subsample < 1.0 (bagging)
    # 2. colsample_bynode < 1.0 (feature subset per node)
    # 3. num_parallel_tree > 1 (forest)
    # 4. learning_rate = 1.0 (no boosting step shrinking)
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cuda',  # Use GPU
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'learning_rate': 1.0,
        'num_parallel_tree': 100, # 100 trees in the forest
        'eval_metric': 'logloss'
    }
    
    # Note: num_boost_round should be 1 for RF if using num_parallel_tree for the forest size
    # But often we just set num_boost_round=1 and let num_parallel_tree do the work.
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    try:
        start_time = time.time()
        # Train 1 round of boosting, which builds 'num_parallel_tree' trees
        bst = xgb.train(params, dtrain, num_boost_round=1) 
        end_time = time.time()
        
        print(f"RF Training completed in {end_time - start_time:.4f} seconds.")
        
        # Predict
        preds = bst.predict(dtest)
        print(f"Prediction shape: {preds.shape}")
        print("SUCCESS: Random Forest (XGBoost) trained on GPU.")
        
    except Exception as e:
        print(f"FAILURE: Random Forest GPU training failed. Error: {e}")

if __name__ == "__main__":
    check_gpu_rf()
