import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from typing import List, Dict, Any, Type, Tuple

from functools import partial
from cocoa.models.bandwidth import create_precentered_grid
from cocoa.models.assets import RF_PARAM_GRID, XGB_PARAM_GRID
from ..models import RFModel, XGBModel, NPRegimeModel
from ..models import (
    BaseModel,
    CocoaDataset,
    TrainTestSplit,
    MFVValidator,
    evaluate_forecast,
    plot_forecast,
)
from ..utils.bias_var_decomposition import bias_variance_decomposition


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
        self.n_bootstrap_rounds = n_bootstrap_rounds
        self.param_grid = None
        self.save_results = save_results
        self.output_dir = None
        self.data_set = None
        self.split = None


        
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

    def run(self) -> None:
        """Executes the full experiment pipeline."""
        if self.split is None:
            raise RuntimeError("Data has not been split. Check runner initialization.")

        # 1. Fit model using the pre-split data
        best_params, best_mfv, final_model, y_full_pred = self._fit_model()

        # 4. Perform Bias-Variance Decomposition
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

        # 4. Evaluate and save all artifacts
        if self.save_results:
            self._save_artifacts(self.split.y_test, y_full_pred, final_model, best_params, best_mfv, avg_mse, avg_bias_sq, avg_variance)
            print(f"Successfully completed run for {self.model_name}.")
        else:
            print(f"Successfully completed run for {self.model_name} (without saving artifacts).")

    def _fit_model(self):
        Q = 4

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
            "structural_break_time": int(self.sample_start_index) if self.sample_start_index is not None else "not applied",
            "regressors": self.feature_cols,
            "target": self.target_col,
            "best_hyperparameters": best_params,
            "mfv_best_score": best_mfv,
            "bias_variance_rounds": self.n_bootstrap_rounds,
        }
        if self.kernel_name:
            run_config["kernel"] = self.kernel_name
        if self.poly_order is not None:
            run_config["polynomial_order"] = int(self.poly_order)
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=4)

        # Save model object, predictions, metrics, and plot
        dump(model, os.path.join(self.output_dir, "model.joblib"))
        pd.DataFrame({"date": self.data_set.dates, "y_pred": y_full_pred}).to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)
        with open(os.path.join(self.output_dir, "oos_metrics.json"), "w") as f:
            json.dump(oos_metrics, f, indent=4)

        # Save bias-variance decomposition results
        decomposition_results = {
            "oos_mse_from_bvd": bv_mse,
            "bias_squared": bv_bias_sq,
            "variance": bv_variance,
            # The "plain bias" is the average error, sqrt(bias_squared) is its magnitude.
            # We can't recover the sign, but we can show the non-squared bias magnitude.
            "bias_plain": np.sqrt(bv_bias_sq),
        }
        with open(os.path.join(self.output_dir, "bias_variance_decomposition.json"), "w") as f:
            json.dump(decomposition_results, f, indent=4)
        
        print("\n--- Bias-Variance Decomposition Results ---")
        print(f"  MSE (from BVD): {bv_mse:.6f}")
        print(f"  Bias^2:         {bv_bias_sq:.6f}")
        print(f"  Variance:       {bv_variance:.6f}")
        # The sum of Bias^2 and Variance should be close to the MSE.
        print(f"  Bias^2 + Var:   {bv_bias_sq + bv_variance:.6f}")

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