import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from typing import List, Dict, Any, Type

from ..models import (
    BaseModel,
    CocoaDataset,
    MFVValidator,
    evaluate_forecast,
    plot_forecast,
)


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
        param_grid: Dict[str, List],
        data_path: str,
        oos_start_date: str,
        sample_start_index: int | None = None,
        kernel_name: str | None = None,
        poly_order: int | None = None,
        output_base_dir: str = "w:/Research/NP/Cocoa/output/cocoa_forecast",
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.param_grid = param_grid
        self.data_path = data_path
        self.sample_start_index = sample_start_index
        self.oos_start_date = oos_start_date
        self.kernel_name = kernel_name
        self.poly_order = poly_order
        self.start_date = None

        # --- Create a unique directory for this experiment run ---
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base_dir, f"{run_timestamp}_{self.model_name}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Instantiated runner for {self.model_name}. Results will be in:\n{self.output_dir}")

    def run(self) -> None:
        """Executes the full experiment pipeline."""
        # 1. Load and split data
        dataset = CocoaDataset(self.data_path, self.feature_cols, self.target_col)
        
        if self.sample_start_index is not None:
            start_date = dataset.get_date_from_1_based_index(self.sample_start_index)
        else:
            start_date = None

        dataset.trim_data_by_start_date(start_date)
        
        # Pass the (potentially trimmed) dataframe to the splitting method
        # The trimmed dataframe is in the .df attribute
        split = dataset.split_oos_by_date(self.oos_start_date, df=dataset.df)
        print(f"Train/CV size: {split.T_train}, OOS test size: {split.T_test}")
        Q = 4
        # 2. Tune hyperparameters with MFV
        mfv = MFVValidator(Q=Q)
        param_list = expand_grid(self.param_grid)
        best_params, best_mfv, _ = mfv.grid_search(
            model_class=self.model_class,
            X_train=split.X_train,
            y_train=split.y_train,
            param_grid=param_list,
        )
        print(f"Best params for {self.model_name}: {best_params} (MFV MSE: {best_mfv:.6f})")

        # 3. Fit final model and get predictions
        final_model = self.model_class(**best_params)
        final_model.fit(split.X_train, split.y_train)

        # Call predict for train/test, ensure results are not None and are arrays before concatenating
        pred_train = final_model.predict(split.X_train)
        pred_test = final_model.predict(split.X_test)

        if pred_train is None or pred_test is None:
            raise ValueError("Model.predict returned None for train or test predictions; ensure the model's predict method returns an array-like object.")

        pred_train = np.asarray(pred_train)
        pred_test = np.asarray(pred_test)

        y_full_pred = np.concatenate([pred_train, pred_test])

        # 4. Evaluate and save all artifacts
        self._save_artifacts(dataset, split.y_test, y_full_pred, final_model, best_params, best_mfv)
        print(f"Successfully completed run for {self.model_name}.")

    def _save_artifacts(self, dataset, y_test, y_full_pred, model, best_params, best_mfv):
        """Saves all model outputs to the unique run directory."""
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
            "structural_break_time": self.sample_start_index if self.sample_start_index else "not applied",
            "regressors": self.feature_cols,
            "target": self.target_col,
            "best_hyperparameters": best_params,
            "mfv_best_score": best_mfv,
        }
        if self.kernel_name:
            run_config["kernel"] = self.kernel_name
        if self.poly_order is not None:
            run_config["polynomial_order"] = self.poly_order
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(run_config, f, indent=4)

        # Save model object, predictions, metrics, and plot
        dump(model, os.path.join(self.output_dir, "model.joblib"))
        pd.DataFrame({"date": dataset.dates, "y_pred": y_full_pred}).to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)
        with open(os.path.join(self.output_dir, "oos_metrics.json"), "w") as f:
            json.dump(oos_metrics, f, indent=4)

        plot_forecast(
            df=dataset.df, target_col=self.target_col, y_pred=y_full_pred,
            model_label=self.model_name, output_path=os.path.join(self.output_dir, "oos_forecast.png"),
            oos_start_date=self.oos_start_date
        )