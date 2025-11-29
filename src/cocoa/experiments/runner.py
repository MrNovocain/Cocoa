import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from typing import List, Dict, Any, Type, Tuple
from functools import partial
import sys
from contextlib import contextmanager

from cocoa.models.bandwidth import create_precentered_grid
from cocoa.models.assets import RF_PARAM_GRID, XGB_PARAM_GRID, BREAK_ID_ONE_BASED, Q_VALUE
from ..models import (
    BaseModel,
    CocoaDataset,
    TrainTestSplit,
    MFVValidator,
    MFVConvexComboValidator,
    evaluate_forecast,
    plot_forecast,
    MLConvexCombinationModel,
    RFModel,
    XGBModel,
    NPRegimeModel,
    NPConvexCombinationModel,
    GaussianKernel,
    LocalPolynomialEngine,
)
from ..utils.bias_var_decomposition import (
    bias_variance_decomposition,
    BiasVarianceDecomposer,
    DefaultModelInstantiator,
    ComboModelInstantiator
)


class Tee:
    """A helper class to redirect stdout to both console and a file."""
    def __init__(self, original_stdout, file):
        self.original_stdout = original_stdout
        self.file = file

    def write(self, text):
        self.original_stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.original_stdout.flush()
        self.file.flush()

@contextmanager
def redirect_stdout_to_log_file(filepath):
    """A context manager to redirect stdout to a log file and the console."""
    original_stdout = sys.stdout
    try:
        with open(filepath, 'w', encoding='utf-8') as log_file:
            tee = Tee(original_stdout, log_file)
            sys.stdout = tee
            yield
    finally:
        sys.stdout = original_stdout


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def expand_grid(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Helper to create a list of parameter dictionaries from a grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    from itertools import product

    return [dict(zip(keys, combo)) for combo in product(*values)]


class ExperimentRunner:
    """
    A reusable class to run, evaluate, and save a model fitting experiment.
    """

    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseModel],
        feature_cols: List[str],
        target_col: str,
        data_path: str,
        oos_start_date: str,
        sample_start_index: int | None = None,
        kernel_name: str | None = None,
        poly_order: int | None = None,
        n_bootstrap_rounds: int = 50,
        save_results: bool = True,
        run_bvd: bool = False,
        output_base_dir: str = "w:/Research/NP/Cocoa/output/cocoa_forecast",
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.data_path = data_path
        self.sample_start_index = sample_start_index
        self.oos_start_date = oos_start_date
        self.kernel_name = kernel_name
        self.poly_order = poly_order
        self.start_date = None
        self.run_bvd = run_bvd
        if self.run_bvd:
            print("Bias-Variance Decomposition will be run for this experiment.")
            self.n_bootstrap_rounds = n_bootstrap_rounds
        else:
            print("Bias-Variance Decomposition will NOT be run for this experiment.")
            self.n_bootstrap_rounds = 0
        self.param_grid = None
        self.gamma = None
        self.save_results = save_results
        self.output_dir = None
        self.data_set = None
        self.split = None
        self.Q = Q_VALUE

        
        if self.save_results:
            # --- Create a unique directory for this experiment run ---
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_base_dir, f"{run_timestamp}_{self.model_name}")
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Instantiated runner for {self.model_name}. Results will be in:\n{self.output_dir}")
        else:
            print(f"Instantiated runner for {self.model_name}. Results will not be saved.")

        # --- Load and prepare data ---
        self.data_set = CocoaDataset(self.data_path, self.feature_cols, self.target_col)
        print(f"indexing from {self.sample_start_index} ")
        if self.sample_start_index is not None:
            self.start_date = self.data_set.get_date_from_1_based_index(self.sample_start_index)
        else:
            self.start_date = None

        self.data_set.trim_data_by_start_date(self.start_date)
        print(f"New start date after trimming: {self.data_set.dates.iloc[0].date()}")

        # --- Split data and store it ---
        self.split = self._split_data(self.data_set.df, self.oos_start_date)
        print(f"Train/CV size: {self.split.T_train}, OOS test size: {self.split.T_test}")
        
        print(f"Feature columns: {self.feature_cols}")

    def _split_data(self, df: pd.DataFrame, oos_start_date: str | pd.Timestamp) -> TrainTestSplit:
        """Splits the dataframe into train and test sets based on a date."""
        oos_start_date = pd.to_datetime(oos_start_date)
        mask_test = df["date"] >= oos_start_date

        if not mask_test.any():
            raise ValueError("No observations on/after the chosen OOS start date.")

        test_start_idx = df.index[mask_test][0]

        X_train = df.loc[:test_start_idx-1, self.feature_cols].reset_index(drop=True)
        y_train = df.loc[:test_start_idx-1, self.target_col].reset_index(drop=True)

        X_test = df.loc[test_start_idx:, self.feature_cols].reset_index(drop=True)
        y_test = df.loc[test_start_idx:, self.target_col].reset_index(drop=True)

        return TrainTestSplit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            T_train=len(X_train),
            T_test=len(X_test),
        )

    def run(self) -> Dict[str, Any]:
        """Executes the full experiment pipeline."""
        if self.split is None:
            raise RuntimeError("Data has not been split. Check runner initialization.")

        def _run_logic():
            # 1. Fit model using the pre-split data
            best_params, best_mfv, final_model, y_full_pred = self._fit_model()

            # For NPComboExperimentRunner, gamma is in best_params
            if 'gamma' in best_params:
                self.gamma = best_params['gamma']
            
            # Add the final fitted model to the params dict for BVD
            best_params["final_model_instance"] = final_model

            # 2. Perform Bias-Variance Decomposition
            avg_mse, avg_bias_sq, avg_variance = None, None, None
            if self.run_bvd:
                print("\n--- Starting Bias-Variance Decomposition ---")
                avg_mse, avg_bias_sq, avg_variance = bias_variance_decomposition(
                    model_class=self.model_class,
                    hyperparams=best_params,
                    X_train=self.split.X_train,
                    y_train=self.split.y_train,
                    X_test=self.split.X_test,
                    y_test=self.split.y_test,
                    n_bootstrap_rounds=self.n_bootstrap_rounds,
                )
            else:
                print("\n--- Skipping Bias-Variance Decomposition ---")

            # 4. Evaluate and save all artifacts
            if self.save_results:
                self._save_artifacts(self.split.y_test, y_full_pred, final_model, best_params, best_mfv, avg_mse, avg_bias_sq, avg_variance)
                print(f"Successfully completed run for {self.model_name}.")
                oos_metrics = evaluate_forecast(self.split.y_test, pd.Series(y_full_pred[-len(self.split.y_test):], index=self.split.y_test.index))
            else:
                print(f"Successfully completed run for {self.model_name} (without saving artifacts).")
                oos_metrics = {} # Ensure oos_metrics exists

            return {
                "best_params": best_params,
                "best_mfv": best_mfv,
                "avg_mse": avg_mse,
                "avg_bias_sq": avg_bias_sq,
                "avg_variance": avg_variance,
                "oos_mse": oos_metrics.get("mse") if self.save_results else None,
                "in_sample_cv_mse": best_mfv, # Expose the in-sample CV score
            }

        results = {}
        if self.save_results and self.output_dir:
            log_path = os.path.join(self.output_dir, "run_log.txt")
            with redirect_stdout_to_log_file(log_path):
                results = _run_logic()
                # The log file is automatically closed here
        else:
            results = _run_logic()
        return results

    def _fit_model(self):
        Q = Q_VALUE

        # Determine the base model class, handling functools.partial
        model_class_to_check = self.model_class
        if isinstance(model_class_to_check, partial):
            model_class_to_check = model_class_to_check.func

        # 2. Prepare parameter grid
        if model_class_to_check == RFModel:
            self.param_grid = RF_PARAM_GRID
        elif model_class_to_check == XGBModel:
            self.param_grid = XGB_PARAM_GRID
        elif model_class_to_check == NPRegimeModel:
            T_train, d_train = self.split.X_train.shape
            self.param_grid = {
                "bandwidth": create_precentered_grid(T=T_train, d=d_train),
            }



        # 3. Tune hyperparameters with MFV
        mfv = MFVValidator(Q=Q)
        if self.param_grid is None:
            raise ValueError("Parameter grid has not been set.")
        else:
            param_list = expand_grid(self.param_grid) # pyright: ignore[reportArgumentType]
        best_params, best_mfv, _ = mfv.grid_search(
            model_class=self.model_class,
            X_train=self.split.X_train,
            y_train=self.split.y_train,
            param_grid=param_list,
        )
        print(f"Best params for {self.model_name}: {best_params} (MFV MSE: {best_mfv:.6f})")

        # 4. Fit final model and get predictions
        final_model = self.model_class(**best_params)
        final_model.fit(self.split.X_train, self.split.y_train)

        # Call predict for train/test, ensure results are not None and are arrays before concatenating
        pred_train = final_model.predict(self.split.X_train)
        pred_test = final_model.predict(self.split.X_test)

        if pred_train is None or pred_test is None:
            raise ValueError("Model.predict returned None for train or test predictions; ensure the model's predict method returns an array-like object.")

        pred_train = np.asarray(pred_train)
        pred_test = np.asarray(pred_test)

        y_full_pred = np.concatenate([pred_train, pred_test])
        return best_params, best_mfv, final_model, y_full_pred

    def _save_artifacts(self, y_test, y_full_pred, model, best_params, best_mfv, bv_mse, bv_bias_sq, bv_variance):
        """Saves all model outputs to the unique run directory."""
        if not self.output_dir:
            raise ValueError("Output directory is not set. Cannot save artifacts.")

        # 1. Evaluate OOS performance
        oos_metrics_original = evaluate_forecast(y_test, pd.Series(y_full_pred[-len(y_test):], index=y_test.index))

        # 2. Convert to MSFE and calculate RMSE for reporting
        msfe = oos_metrics_original.get("mse", 0)
        rmse = np.sqrt(msfe)
        oos_metrics = {
            "MSFE": msfe,
            "RMSE": rmse,
            "MAE": oos_metrics_original.get("mae", 0)
        }
        print(f"OOS Metrics: {oos_metrics}")
        # Save config
        run_config = {
            "model_name": self.model_name,
            "run_timestamp": os.path.basename(self.output_dir).split('_')[0],
            "test_set_start_date": self.oos_start_date,
            "structural_break_index": int(self.sample_start_index) if self.sample_start_index is not None else "not applied",
            "structural_break_date": str(pd.to_datetime(self.start_date).date()) if self.start_date is not None else "not applied",
            "regressors": self.feature_cols,
            "target": self.target_col,
            "hyperparameter_grid": self.param_grid,
            "mfv_best_score": best_mfv,
            "bias_variance_rounds": self.n_bootstrap_rounds,
        }

        # The original `best_params` contains the model instance for BVD,
        # but we must remove it before saving the JSON config to avoid serialization errors.
        serializable_params = best_params.copy()
        serializable_params.pop("final_model_instance", None)
        run_config["optimal_hyperparameters"] = serializable_params

        if self.kernel_name:
            run_config["kernel"] = self.kernel_name
        if self.poly_order is not None:
            run_config["polynomial_order"] = int(self.poly_order)
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=4, cls=NpEncoder)

        # Save model object, predictions, metrics, and plot
        dump(model, os.path.join(self.output_dir, "model.joblib"))
        pd.DataFrame({"date": self.data_set.dates, "y_pred": y_full_pred}).to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)
        with open(os.path.join(self.output_dir, "oos_metrics.json"), "w") as f:
            json.dump(oos_metrics, f, indent=4, cls=NpEncoder)

        # Save bias-variance decomposition results
        decomposition_results = {
            "oos_mse_from_bvd": bv_mse,
            "bias_squared": bv_bias_sq,
            "variance": bv_variance,
            # The "plain bias" is the average error, sqrt(bias_squared) is its magnitude.
            # We can't recover the sign, but we can show the non-squared bias magnitude.
            "bias_plain": np.sqrt(bv_bias_sq) if bv_bias_sq is not None else None,
        }
        with open(os.path.join(self.output_dir, "bias_variance_decomposition.json"), "w") as f:
            json.dump(decomposition_results, f, indent=4, cls=NpEncoder)
        
        print("\n--- Bias-Variance Decomposition Results ---")
        print(f"  MSE (from BVD): {bv_mse:.6f}" if bv_mse is not None else "  MSE (from BVD): N/A")
        print(f"  Bias^2:         {bv_bias_sq:.6f}" if bv_bias_sq is not None else "  Bias^2:         N/A")
        print(f"  Variance:       {bv_variance:.6f}" if bv_variance is not None else "  Variance:       N/A")
        # The sum of Bias^2 and Variance should be close to the MSE.
        print(f"  Bias^2 + Var:   {(bv_bias_sq + bv_variance):.6f}" if bv_bias_sq is not None and bv_variance is not None else "  Bias^2 + Var:   N/A")

        plot_forecast(
            df=self.data_set.df, target_col=self.target_col, y_pred=y_full_pred,
            model_label=self.model_name, output_path=os.path.join(self.output_dir, "oos_forecast.png"),
            oos_start_date=self.oos_start_date
        )


    def run_BVD_only(self) -> Tuple[float, float, float, pd.Timestamp]:
        """Run fitted NP post model and return BVD results only."""
        if self.split is None:
            raise RuntimeError("Data has not been split. Check runner initialization.")

        # 1. fit model
        best_params, _, _, _ = self._fit_model()

        avg_mse, avg_bias_sq, avg_variance = bias_variance_decomposition(
            model_class=self.model_class,
            hyperparams=best_params,
            X_train=self.split.X_train,
            y_train=self.split.y_train,
            X_test=self.split.X_test,
            y_test=self.split.y_test,
            n_bootstrap_rounds=self.n_bootstrap_rounds,
        )
        if self.start_date is None:
            raise ValueError("start_date is None; cannot return it.")
        return (avg_mse, avg_bias_sq, avg_variance, self.start_date)
    
    def get_train_size(self) -> int:
        """Returns the size of the training set after trimming."""
        if self.split is None:
            raise ValueError("Data not split. Cannot determine training set size.")
        return self.split.T_train


class ConvexComboExperimentRunner(ExperimentRunner):
    """
    An ExperimentRunner for convex combination models (NP or ML).

    It performs a three-stage cross-validation procedure to tune sub-model
    hyperparameters and the combination weight (gamma).
    """

    def __init__(self, combo_type: str, *args, **kwargs):
        # This __init__ is "standalone". It replicates the necessary setup from
        # ExperimentRunner.__init__ without calling super().__init__() because
        # the parent's data trimming logic is not suitable for the combo model.
        self.combo_type = combo_type.upper()
        if self.combo_type not in ['NP', 'ML']:
            raise ValueError("combo_type must be either 'NP' or 'ML'.")
        
        # --- 1. Replicate property setup from parent ---
        self.model_name = kwargs.get("model_name", "NP_LL_Combo")
        # Allow child classes (like MLComboExperimentRunner) to specify the model class.
        # Default to NPConvexCombinationModel for backward compatibility.
        self.model_class = kwargs.get("model_class", NPConvexCombinationModel)
        self.feature_cols = kwargs['feature_cols']
        self.target_col = kwargs['target_col']
        self.data_path = kwargs['data_path']
        self.sample_start_index = kwargs.get('sample_start_index')
        self.oos_start_date = kwargs['oos_start_date']
        self.poly_order = kwargs.get('poly_order')
        self.n_bootstrap_rounds = kwargs.get('n_bootstrap_rounds', 50)
        self.save_results = kwargs.get('save_results', True)
        self.run_bvd = kwargs.get('run_bvd', False)
        self.param_grid = None
        self.gamma = None
        self.split = None
        self.Q = Q_VALUE

        if self.combo_type == 'NP':
            self.model_class = NPConvexCombinationModel
            self.kernel_name = kwargs.get('kernel_name', 'GaussianKernel')
            self.kernel = GaussianKernel()
            self.engine = LocalPolynomialEngine(order=self.poly_order if self.poly_order is not None else 1)
        elif self.combo_type == 'ML':
            self.model_class = MLConvexCombinationModel
            # The specific ML model to use within the combination, e.g., RFModel
            self.sub_model_class = kwargs.get('sub_model_class', RFModel)
            self.sub_model_param_grid = kwargs.get('sub_model_param_grid', RF_PARAM_GRID)
            self.kernel_name = None # Not applicable
        
        if self.sample_start_index is None:
            raise ValueError("ConvexComboExperimentRunner requires a 'sample_start_index' to define the post-break period.")

        # --- 2. Replicate output directory setup from parent ---
        output_base_dir = kwargs.get("output_base_dir", "w:/Research/NP/Cocoa/output/cocoa_forecast")
        if self.save_results:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_base_dir, f"{run_timestamp}_{self.model_name}")
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Instantiated runner for {self.model_name}. Results will be in:\n{self.output_dir}")
        else:
            self.output_dir = None
            print(f"Instantiated runner for {self.model_name}. Results will not be saved.")

        # --- 3. Custom data handling for combo model ---
        # Load data WITHOUT trimming the start, so "full" model gets all data.
        self.data_set = CocoaDataset(self.data_path, self.feature_cols, self.target_col)
        
        # Define the break date from the full, untrimmed dataset.
        self.start_date = self.data_set.get_date_from_1_based_index(self.sample_start_index)
        print(f"Structural break date identified: {pd.to_datetime(self.start_date).date()}")

        # Split data into train/test based on OOS date. This uses the _split_data method
        # from the parent class, which is fine.
        self.split = self._split_data(self.data_set.df, self.oos_start_date)
        print(f"Full train/CV size: {self.split.T_train}, OOS test size: {self.split.T_test}")

    def _fit_model(self):
        """
        Overrides the base _fit_model to perform the three-stage CV required
        for the ConvexCombinationModel.
        """
        if self.split is None or self.start_date is None:
            raise RuntimeError("Data has not been split or break date is not set. Check runner initialization.")

        X_train_full = self.split.X_train
        y_train_full = self.split.y_train

        # --- Configuration ---
        validator = MFVValidator(Q=self.Q)
        
        train_dates = self.data_set.df['date'].iloc[:self.split.T_train]

        # Find the 0-based index in the training set where the post-break period begins
        post_start_mask = train_dates >= pd.to_datetime(self.start_date)
        if not post_start_mask.any():
            # This can happen if the break date is after the training period ends.
            post_start_index = len(X_train_full)
        else:
            # Get the label-based index of the first True value
            label_index = train_dates[post_start_mask].index[0]
            # Convert this label-based index into a position-based one for iloc.
            # This is crucial because X_train_full has a reset index (0, 1, 2...).
            post_start_index = X_train_full.index.get_loc(label_index)

        X_train_pre = X_train_full.iloc[:post_start_index]
        y_train_pre = y_train_full.iloc[:post_start_index]
        X_train_post = X_train_full.iloc[post_start_index:]
        y_train_post = y_train_full.iloc[post_start_index:]

        if self.combo_type == 'NP':
            best_params_pre, best_params_post = self._tune_submodel_params_np(validator, X_train_pre, y_train_pre, X_train_post, y_train_post)
            sub_model_class_partial = partial(NPRegimeModel, kernel=self.kernel, local_engine=self.engine)
        elif self.combo_type == 'ML':
            best_params_pre, best_params_post = self._tune_submodel_params_ml(validator, X_train_pre, y_train_pre, X_train_post, y_train_post)
            sub_model_class_partial = self.sub_model_class

        # 3. Tune gamma for the Convex Combination model
        best_gamma, best_score_gamma = self._tune_gamma(
            sub_model_class_partial, best_params_pre, best_params_post,
            X_train_full, y_train_full, post_start_index
        )

        # --- Final Model Fitting ---
        if self.combo_type == 'NP':
            final_model = NPConvexCombinationModel(
                kernel=self.kernel, local_engine=self.engine,
                pre_bandwidth=best_params_pre['bandwidth'],
                post_bandwidth=best_params_post['bandwidth'],
                break_index=post_start_index,
                gamma=best_gamma,
            )
        else: # ML
            final_model = MLConvexCombinationModel(
                model_class=self.sub_model_class,
                params_pre=best_params_pre,
                params_post=best_params_post,
                break_index=post_start_index,
                gamma=best_gamma,
            )

        final_model.fit(X_train_full, y_train_full)

        # --- Generate predictions ---
        pred_train = final_model.predict(self.split.X_train)
        pred_test = final_model.predict(self.split.X_test)

        if pred_train is None or pred_test is None:
            raise ValueError("Model.predict returned None.")

        y_full_pred = np.concatenate([np.asarray(pred_train), np.asarray(pred_test)])
        
        if self.combo_type == 'NP':
            best_params_combined = {
                "gamma": best_gamma, "break_index": post_start_index,
                "bandwidth_pre": best_params_pre['bandwidth'],
                "bandwidth_post": best_params_post['bandwidth'],
                "poly_order": self.poly_order if self.poly_order is not None else 1,
            }
        else: # ML
            best_params_combined = {
                "gamma": best_gamma, "break_index": post_start_index,
                "params_pre": best_params_pre,
                "params_post": best_params_post,
            }

        return best_params_combined, best_score_gamma, final_model, y_full_pred

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

    def _tune_gamma(self, sub_model_class, params_pre, params_post, X_full, y_full, break_idx):
        print("\n--- (3/3) Tuning gamma for Convex Combination model ---")
        combo_validator = MFVConvexComboValidator(Q=self.Q)
        gamma_values = np.linspace(0, 1, 21)

        best_gamma, best_score_gamma = combo_validator.tune_gamma(
            model_class_pre=sub_model_class,
            params_pre=params_pre,
            model_class_post=sub_model_class,
            params_post=params_post,
            X_train_full=X_full,
            y_train_full=y_full,
            break_index=break_idx,
            gamma_grid=gamma_values,
            verbose=True,
        )
        self.gamma = best_gamma
        print(f"Best gamma: {best_gamma:.2f} (MFV MSE: {best_score_gamma:.6f})")

        # Store param grid for logging purposes
        if self.combo_type == 'NP':
             self.param_grid = {
                "gamma": gamma_values.tolist(),
                "bandwidth_pre": params_pre['bandwidth'],
                "bandwidth_post": params_post['bandwidth'],
            }
        else:
             self.param_grid = {
                "gamma": gamma_values.tolist(),
                "params_pre": self.sub_model_param_grid,
                "params_post": self.sub_model_param_grid,
            }

        return best_gamma, best_score_gamma