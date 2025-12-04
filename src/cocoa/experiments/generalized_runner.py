
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Type
from functools import partial

from cocoa.models.assets import RF_PARAM_GRID, Q_VALUE
from cocoa.models import (
    CocoaDataset,
    RFModel,
    NPRegimeModel,
    GaussianKernel,
    LocalPolynomialEngine,
)
from cocoa.models.mfv_CV import MFVValidator, MFVGeneralizedComboValidator
from cocoa.models.generalized_combo import GeneralizedNonLinearComboModel
from cocoa.models.shrinkage import get_shrinkage_function
from cocoa.models.bandwidth import create_precentered_grid
from cocoa.experiments.runner import ExperimentRunner, expand_grid

class GeneralizedComboExperimentRunner(ExperimentRunner):
    """
    Runner for the Generalized Non-linear Combination Estimator.
    
    Supports both NP and ML sub-models, combined via:
    y_hat = y_post + beta * g(y_pre - y_post)
    """

    def __init__(self, combo_type: str, shrinkage_type: str = "log", shrinkage_params: Dict = None, **kwargs):
        self.combo_type = combo_type.upper()
        if self.combo_type not in ['NP', 'ML']:
            raise ValueError("combo_type must be either 'NP' or 'ML'.")
        
        # Setup shrinkage function
        self.shrinkage_type = shrinkage_type
        self.shrinkage_params = shrinkage_params or {}
        self.shrinkage_func = get_shrinkage_function(self.shrinkage_type, **self.shrinkage_params)
        
        # Replicate ExperimentRunner setup (similar to ConvexComboExperimentRunner)
        self.model_name = kwargs.get("model_name", f"{self.combo_type}_Generalized_{self.shrinkage_type}")
        self.model_class = GeneralizedNonLinearComboModel
        self.feature_cols = kwargs['feature_cols']
        self.target_col = kwargs['target_col']
        self.data_path = kwargs['data_path']
        self.break_index = kwargs.get('break_index')
        self.oos_start_date = kwargs['oos_start_date']
        self.poly_order = kwargs.get('poly_order')
        self.n_bootstrap_rounds = kwargs.get('n_bootstrap_rounds', 50)
        self.save_results = kwargs.get('save_results', True)
        self.run_bvd = kwargs.get('run_bvd', False)
        self.param_grid = None
        self.beta = None
        self.train_start_index = self.break_index
        self.split = None
        self.Q = Q_VALUE
        
        if self.break_index is None:
            raise ValueError("GeneralizedComboExperimentRunner requires a 'break_index'.")

        # Setup sub-models
        if self.combo_type == 'NP':
            self.kernel_name = kwargs.get('kernel_name', 'GaussianKernel')
            self.kernel = GaussianKernel()
            self.engine = LocalPolynomialEngine(order=self.poly_order if self.poly_order is not None else 1)
        elif self.combo_type == 'ML':
            self.sub_model_class = kwargs.get('sub_model_class', RFModel)
            self.sub_model_param_grid = kwargs.get('sub_model_param_grid', RF_PARAM_GRID)
            self.kernel_name = None

        # Output directory
        output_base_dir = kwargs.get("output_base_dir", str(Path(__file__).resolve().parents[3] / "output" / "cocoa_forecast"))
        if self.save_results:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_base_dir, f"{run_timestamp}_{self.model_name}")
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Instantiated runner for {self.model_name}. Results will be in:\n{self.output_dir}")
        else:
            self.output_dir = None
            print(f"Instantiated runner for {self.model_name}. Results will not be saved.")

        # Load data
        self.data_set = CocoaDataset(self.data_path, self.feature_cols, self.target_col)
        self.start_date = self.data_set.get_date_from_1_based_index(self.break_index)
        print(f"Structural break date identified: {pd.to_datetime(self.start_date).date()}")
        
        self.split = self._split_data(self.data_set.df, self.oos_start_date)
        print(f"Full train/CV size: {self.split.T_train}, OOS test size: {self.split.T_test}")

    def _fit_model(self):
        if self.split is None or self.start_date is None:
            raise RuntimeError("Data has not been split or break date is not set.")

        X_train_full = self.split.X_train
        y_train_full = self.split.y_train

        # Configuration
        validator = MFVValidator(Q=self.Q)
        
        # Identify post-break start index in training set
        train_dates = self.data_set.df['date'].iloc[:self.split.T_train]
        post_start_mask = train_dates >= pd.to_datetime(self.start_date)
        if not post_start_mask.any():
            post_start_index = len(X_train_full)
        else:
            label_index = train_dates[post_start_mask].index[0]
            post_start_index = X_train_full.index.get_loc(label_index)

        X_train_pre = X_train_full.iloc[:post_start_index]
        y_train_pre = y_train_full.iloc[:post_start_index]
        X_train_post = X_train_full.iloc[post_start_index:]
        y_train_post = y_train_full.iloc[post_start_index:]

        # 1. Tune sub-models
        if self.combo_type == 'NP':
            best_params_pre, best_params_post = self._tune_submodel_params_np(validator, X_train_pre, y_train_pre, X_train_post, y_train_post)
            sub_model_class_partial = partial(NPRegimeModel, kernel=self.kernel, local_engine=self.engine)
        elif self.combo_type == 'ML':
            best_params_pre, best_params_post = self._tune_submodel_params_ml(validator, X_train_pre, y_train_pre, X_train_post, y_train_post)
            sub_model_class_partial = self.sub_model_class

        # 2. Tune beta
        print(f"\n--- Tuning beta for Generalized Combination ({self.shrinkage_type}) ---")
        combo_validator = MFVGeneralizedComboValidator(Q=self.Q)
        beta_values = np.linspace(0, 1, 21) # Can expand range if needed, e.g. > 1

        best_beta, best_score_beta = combo_validator.tune_beta(
            model_pre=None, # Not used
            model_post=None, # Not used
            model_class_pre=sub_model_class_partial,
            params_pre=best_params_pre,
            model_class_post=sub_model_class_partial,
            params_post=best_params_post,
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            break_index=post_start_index,
            shrinkage_func=self.shrinkage_func,
            beta_grid=beta_values,
            verbose=True,
        )
        self.beta = best_beta
        print(f"Best beta: {best_beta:.2f} (MFV MSE: {best_score_beta:.6f})")

        # 3. Final Model Fitting
        # Instantiate sub-models with best params
        model_pre_final = sub_model_class_partial(**best_params_pre)
        model_post_final = sub_model_class_partial(**best_params_post)
        
        # Fit sub-models on full available data (pre on all pre, post on all post)
        model_pre_final.fit(X_train_pre, y_train_pre) # Pre model uses only pre data
        if not X_train_post.empty:
            model_post_final.fit(X_train_post, y_train_post) # Post model uses only post data
        
        # Construct Generalized Model
        final_model = GeneralizedNonLinearComboModel(
            model_pre=model_pre_final,
            model_post=model_post_final,
            break_index=post_start_index,
            beta=best_beta,
            shrinkage_func=self.shrinkage_func
        )
        # Note: GeneralizedNonLinearComboModel doesn't strictly need .fit() if sub-models are fitted,
        # but for consistency with API we can call it (it just sets flags).
        final_model.fit(X_train_full, y_train_full)

        # 4. Predictions
        pred_train = final_model.predict(self.split.X_train)
        pred_test = final_model.predict(self.split.X_test)
        
        y_full_pred = np.concatenate([np.asarray(pred_train), np.asarray(pred_test)])

        # Params for logging
        best_params_combined = {
            "beta": best_beta,
            "break_index": post_start_index,
            "shrinkage": self.shrinkage_type,
            "shrinkage_params": self.shrinkage_params,
            "params_pre": best_params_pre,
            "params_post": best_params_post,
        }

        return best_params_combined, best_score_beta, final_model, y_full_pred

    # Reuse tuning methods from ConvexComboExperimentRunner logic
    # We can copy them or inherit. Inheriting from ConvexComboExperimentRunner might be cleaner 
    # but it has specific __init__ logic. Let's just copy the helper methods for safety/independence.
    
    def _tune_submodel_params_np(self, validator, X_pre, y_pre, X_post, y_post):
        print("\n--- (1/3) Tuning bandwidth for PRE-break model ---")
        NPModelPartial = partial(NPRegimeModel, kernel=self.kernel, local_engine=self.engine)
        T_pre, d_pre = X_pre.shape
        bw_grid_pre = [{"bandwidth": h} for h in create_precentered_grid(T=T_pre, d=d_pre)]
        best_params_pre, _, _ = validator.grid_search(NPModelPartial, X_pre, y_pre, bw_grid_pre, verbose=False)
        print(f"Best bandwidth for PRE-break model: {best_params_pre['bandwidth']:.4f}")

        print("\n--- (2/3) Tuning bandwidth for POST-break model ---")
        T_post, d_post = X_post.shape
        if T_post <= 0:
            raise ValueError("Post-break training set is empty.")
        bw_grid_post = [{"bandwidth": h} for h in create_precentered_grid(T=T_post, d=d_post)]
        best_params_post, _, _ = validator.grid_search(NPModelPartial, X_post, y_post, bw_grid_post, verbose=False)
        print(f"Best bandwidth for POST-break model: {best_params_post['bandwidth']:.4f}")

        return best_params_pre, best_params_post

    def _tune_submodel_params_ml(self, validator, X_pre, y_pre, X_post, y_post):
        print("\n--- (1/3) Tuning params for PRE-break model ---")
        param_list_pre = expand_grid(self.sub_model_param_grid)
        best_params_pre, _, _ = validator.grid_search(
            self.sub_model_class, X_pre, y_pre, param_list_pre, verbose=False
        )
        print(f"Best params for PRE-break model: {best_params_pre}")

        print("\n--- (2/3) Tuning params for POST-break model ---")
        if len(X_post) == 0:
            raise ValueError("Post-break training set is empty.")
        param_list_post = expand_grid(self.sub_model_param_grid)
        best_params_post, _, _ = validator.grid_search(
            self.sub_model_class, X_post, y_post, param_list_post, verbose=False
        )
        print(f"Best params for POST-break model: {best_params_post}")

        return best_params_pre, best_params_post
