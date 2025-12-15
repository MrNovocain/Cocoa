
import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from joblib import Parallel, delayed
from cocoa.simulation.dgp import BaseDGP
from cocoa.models.generalized_combo import GeneralizedNonLinearComboModel
from cocoa.models.shrinkage import get_shrinkage_function

# Mock Model to act as Pre/Post sub-models (Since DGP is simple mean)
from sklearn.neighbors import KNeighborsRegressor

# Mock Model to act as Pre/Post sub-models (Local Estimator)
# We use KNN to simulate "Local" Constant/Linear estimation.
# Logic: average of nearest neighbors.

# Mock Model to act as Pre/Post sub-models (Since DGP is simple mean)
class MockMeanModel:
    def __init__(self, fixed_value=None):
        self.fixed_value = fixed_value
        self.fitted_mean = None

    def fit(self, X, y):
        # If fixed_value is set (oracle), use it. Else estimate mean.
        if self.fixed_value is not None:
            self.fitted_mean = self.fixed_value
        else:
            self.fitted_mean = np.mean(y)
    
    def predict(self, X):
        return np.full(X.shape[0], self.fitted_mean)

class SimulationEngine:
    def __init__(self, dgp: BaseDGP, n_trials: int = 100):
        self.dgp = dgp
        self.n_trials = n_trials

    def _run_single_trial(self, trial_id: int):
        data = self.dgp.generate()
        
        # Check for Oracle Means (Parameter Instability Sim)
        if "oracle_mean_val" in data:
            # Oracle Mode: We assume independent ideal estimators for each regime
            model_pre = MockMeanModel(fixed_value=0.0)
            model_pre.fit(None, None) # Init fitted_mean
            
            # Tuning Phase: Use Validation Mean
            model_post_tune = MockMeanModel(fixed_value=data['oracle_mean_val'])
            model_post_tune.fit(data['X_post'], data['y_post'])
            
            # Eval Phase: Use OOS Mean
            model_post_eval = MockMeanModel(fixed_value=data['oracle_mean_oos'])
            model_post_eval.fit(data['X_oos'], data['y_oos']) # Just to init, value is fixed
            
            # For compatibility with legacy code below, assign to generic vars
            # But wait, logic below splits tune/eval.
            # We need to inject these models into the logic.
            
            # Pre-calc IS (Tuning)
            pred_pre_is = model_pre.predict(data['X_post'])
            pred_post_is = model_post_tune.predict(data['X_post'])
            
            # Pre-calc OOS (Eval)
            pred_pre_oos = model_pre.predict(data['X_oos'])[0]
            pred_post_oos = model_post_eval.predict(data['X_oos'])[0]
            
            grid = np.linspace(0, 1, 101)
            
            def get_tuned_weight(shrinkage_type: str):
                best_mse = float('inf')
                best_weight = 0.0
                diff_is = pred_pre_is - pred_post_is
                shrink_func = get_shrinkage_function(shrinkage_type)
                g_diff_is = shrink_func(diff_is)
                for w in grid:
                    y_hat = pred_post_is + w * g_diff_is
                    mse = np.mean((y_hat - data['y_post'])**2)
                    if mse < best_mse:
                        best_mse = mse
                        best_weight = w
                return best_weight
                
            w_linear_tuned = get_tuned_weight("linear")
            w_log_tuned = get_tuned_weight("log")

            def get_eval_mse(shrinkage_type: str, w: float):
                shrink_func = get_shrinkage_function(shrinkage_type)
                g_diff = shrink_func(np.array([pred_pre_oos - pred_post_oos]))[0]
                y_pred = pred_post_oos + w * g_diff
                return (y_pred - data['y_oos'][0])**2
            
            # Return immediately for Oracle Mode
            return {
                "trial_id": trial_id,
                "mse_linear_tuned": get_eval_mse("linear", w_linear_tuned),
                "w_linear_tuned": w_linear_tuned,
                "mse_log_tuned": get_eval_mse("log", w_log_tuned),
                "w_log_tuned": w_log_tuned,
                "mse_post_only": (pred_post_oos - data['y_oos'][0])**2,
                "info": "oracle"
            }

        # Legacy / Standard Logic (Local KNN Estimators)
        # Use small k to mimic local bandwidth.
        # Rare region logic:
        # Pre: Dense data -> k neighbors are close -> Low Bias/Var.
        # Post: Sparse data -> k neighbors are far (maybe outside rare region w/ diff mean) -> Bias!
        # wait, if k neighbors are outside rare region in Post, they will predict 0 (Majority Mean).
        # But True Mean is BreakSize.
        # So Post-Only will be BIASED towards 0 because of smoothing over low density?
        # User said: "Post-break only... very high variance there".
        # If we use k=1 or small bandwidth, and there are NO points, we can't predict.
        # If there are FEW points, variance is high.
        # Let's use k=5.
        
        model_pre = MockKNNModel(k=20) # Pre has lots of data
        model_pre.fit(data['X_pre'], data['y_pre'])
        
        model_post = MockKNNModel(k=5) # Post is sparse.
        try:
            model_post.fit(data['X_post'], data['y_post'])
        except ValueError:
            # Handle empty post case if it ever happens
            pass
        
        # 2. Define Combinations to Test
        # Standard WLL (Linear) vs Log WLL
        
        
        # Grid of weights to search
        grid = np.linspace(0, 1, 101)
        
        # Pre-calculate sub-model OOS predictions once (optimization)
        pred_pre_oos = model_pre.predict(data['X_oos'])[0]
        pred_post_oos = model_post.predict(data['X_oos'])[0]
        target = data['y_oos'][0]
        diff = pred_pre_oos - pred_post_oos
        
        def get_best_mse(shrinkage_type: str):
            best_mse = float('inf')
            best_weight = 0.0

            
            shrink_func = get_shrinkage_function(shrinkage_type)
            # Array-ify for vectorized calc if possible, but shrink_func might expect array
            # g(u)
            g_diff = shrink_func(np.array([diff]))[0]
            
            # y_hat = y_post + beta * g_diff
            # MSE = (y_post + beta * g_diff - target)^2
            
            # Vectorized search
            betas = grid
            y_preds = pred_post_oos + betas * g_diff
            mses = (y_preds - target)**2
            
            min_idx = np.argmin(mses)
            return mses[min_idx], betas[min_idx]

        linear_mse, linear_opt_w = get_best_mse("linear")
        log_mse, log_opt_w = get_best_mse("log")
        post_mse, _ = get_best_mse("linear") # Should match beta=0 check, but safe to calc
        # Actually post_only is just beta=0
        
        # Explicit Post-Only (Gamma=0)
        # linear_mse search includes 0, so it should be <= post_mse
        
        # Post Only Calc
        pred_post_oos = model_post.predict(data['X_oos'])[0]
        mse_post_only = (pred_post_oos - data['y_oos'][0])**2
        
        
        # Calculate Fixed Weight (0.5) Performance for Robustness Check
        def get_fixed_mse(shrinkage_type: str, w: float):
            shrink_func = get_shrinkage_function(shrinkage_type)
            g_diff = shrink_func(np.array([pred_pre_oos - pred_post_oos]))[0]
            # Prediction = Post + w * g(diff)
            y_pred = pred_post_oos + w * g_diff
            return (y_pred - data['y_oos'][0])**2



        # Calculate "Tuned" Performance (Simulating MFV)
        # We tune on the provided data (X_post, y_post) which for MisleadingDGP has NO Break.
        # We find the weight that minimizes MSE on y_post.
        
        # Pre-calc IS predictions
        pred_pre_is = model_pre.predict(data['X_post']) # Predict on Post Validation set using Pre model
        pred_post_is = model_post.predict(data['X_post']) # Predict on Post Val using Post model
        
        def get_tuned_weight(shrinkage_type: str):
            best_mse = float('inf')
            best_weight = 0.0
            
            diff_is = pred_pre_is - pred_post_is
            shrink_func = get_shrinkage_function(shrinkage_type)
            g_diff_is = shrink_func(diff_is)
            
            # Simple grid search 0 to 1
            for w in grid:
                y_hat = pred_post_is + w * g_diff_is
                mse = np.mean((y_hat - data['y_post'])**2)
                if mse < best_mse:
                    best_mse = mse
                    best_weight = w
            return best_weight
        
        w_linear_tuned = get_tuned_weight("linear")
        w_log_tuned = get_tuned_weight("log")

        def get_eval_mse(shrinkage_type: str, w: float):
            shrink_func = get_shrinkage_function(shrinkage_type)
            g_diff = shrink_func(np.array([pred_pre_oos - pred_post_oos]))[0]
            y_pred = pred_post_oos + w * g_diff
            return (y_pred - data['y_oos'][0])**2

        res = {
            "trial_id": trial_id,
            "mse_linear_opt": linear_mse,
            "w_linear_opt": linear_opt_w,
            "mse_log_opt": log_mse,
            "w_log_opt": log_opt_w,
            "mse_post_only": mse_post_only,
            "mse_linear_0.5": get_fixed_mse("linear", 0.5),
            "mse_log_0.5": get_fixed_mse("log", 0.5),
            "mse_linear_tuned": get_eval_mse("linear", w_linear_tuned),
            "w_linear_tuned": w_linear_tuned,
            "mse_log_tuned": get_eval_mse("log", w_log_tuned),
            "w_log_tuned": w_log_tuned
        }
        return res

    def run(self):
        results = Parallel(n_jobs=-1)(
            delayed(self._run_single_trial)(i) for i in range(self.n_trials)
        )
        return pd.DataFrame(results)
