
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from cocoa.experiments.generalized_runner import GeneralizedComboExperimentRunner
from cocoa.experiments.runner import ConvexComboExperimentRunner, ExperimentRunner
from cocoa.experiments.break_detection import MohrRunner
from cocoa.models.cocoa_data import CocoaDataset
from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
)
from cocoa.models.np_regime import NPRegimeModel

def run_head_to_head():
    print("--- Starting Head-to-Head Comparison ---")
    
    # Selected OOS Date to target the 2022 Break (Small T_post)
    OOS_DATE = "2024-04-01"
    
    print(f"Configuration:\n  OOS Date: {OOS_DATE}\n  Target: {DEFAULT_TARGET_COL}\n")

    # 1. Detect Break
    print("\n[Step 1] Running Mohr Break Detection...")
    mohr_runner = MohrRunner(oos_start_date=OOS_DATE)
    detected_break_index = mohr_runner.run_mohr_break_detection()
    
    ds = CocoaDataset()
    detected_break_date = ds.get_date_from_1_based_index(detected_break_index)
    print(f"  -> Detected Break: {detected_break_date.date()} (Index: {detected_break_index})")

    results = {}

    # Helper to clean MSFE
    def get_mse(res: Dict[str, Any]):
        return res.get("oos_mse", 9999.0)

    # 2. Run Log WLL
    print("\n[Step 2] Running Log WLL...")
    log_runner = GeneralizedComboExperimentRunner(
        combo_type="NP",
        model_name=f"Log_WLL_{OOS_DATE}",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_DATE,
        break_index=detected_break_index,
        shrinkage_type="log",
        poly_order=1,
        save_results=True,
    )
    res_log = log_runner.run()
    results["Log WLL"] = {
        "mse": get_mse(res_log),
        "param": res_log['best_params'].get('beta')
    }

    # 3. Run Std WLL (Convex)
    print("\n[Step 3] Running Std WLL (Convex)...")
    std_runner = ConvexComboExperimentRunner(
        combo_type="NP",
        model_name=f"Std_WLL_{OOS_DATE}",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_DATE,
        break_index=detected_break_index,
        poly_order=1,
        save_results=True,
    )
    res_std = std_runner.run()
    results["Std WLL"] = {
        "mse": get_mse(res_std),
        "param": res_std['best_params'].get('gamma')
    }

    # 4. Run Post-Break NP (Pure Post)
    # Using ExperimentRunner with train_start_index = detected_break_index
    print(f"\n[Step 4] Running Post-Break NP (Start Index: {detected_break_index})...")
    post_runner = ExperimentRunner(
        model_name=f"Post_NP_{OOS_DATE}",
        model_class=None, # Will default incorrectly if not passed? 
        # ExperimentRunner.__init__ expects model_class. 
        # Let's import NPRegimeModel.
        # But wait, ExperimentRunner usually instantiates model_class. 
        # For NP, we usually use specific setup in creating `self.engine` in the runner?
        # Actually ExperimentRunner is generic. 
        # The `ConvexCombo` setup instantiates `self.engine`. 
        # Let's look at `ExperimentRunner._fit_model`:
        # "elif model_class_to_check == NPRegimeModel: ... self.param_grid = ..."
        # So we just pass NPRegimeModel.
        # BUT `NPRegimeModel` needs `kernel` and `local_engine` in `__init__`.
        # ExperimentRunner `_fit_model` does `final_model = self.model_class(**best_params)`.
        # This implies `model_class` must accept `bandwidth` (from grid) plus whatever else?
        # `NPRegimeModel` takes `kernel, local_engine, bandwidth`.
        # `ExperimentRunner` doesn't seem to have valid logic to inject `kernel` and `engine` into `NPRegimeModel` 
        # automatically unless we use `functools.partial`.
        
    )
    # Let's import and configure partials
    from cocoa.models.np_regime import NPRegimeModel
    from cocoa.models.np_kernels import GaussianKernel
    from cocoa.models.np_engines import LocalPolynomialEngine
    from functools import partial
    
    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=1)
    NPModelPartial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)
    
    # Now instantiate runner with this Partial
    post_runner = ExperimentRunner(
        model_name=f"Post_NP_{OOS_DATE}",
        model_class=NPModelPartial,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_DATE,
        train_start_index=detected_break_index, # Trim to post-break
        poly_order=1, # Used for logging, not passed to model logic here (since we pre-baked engine)
        save_results=True,
    )
    res_post = post_runner.run()
    results["Post NP"] = {
        "mse": get_mse(res_post),
        "param": res_post['best_params'].get('bandwidth')
    }


    # 5. Run Full-Sample NP (Ignore Break)
    print("\n[Step 5] Running Full-Sample NP...")
    full_runner = ExperimentRunner(
        model_name=f"Full_NP_{OOS_DATE}",
        model_class=NPModelPartial,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_DATE,
        train_start_index=None, # Full history
        poly_order=1,
        save_results=True,
    )
    res_full = full_runner.run()
    results["Full NP"] = {
        "mse": get_mse(res_full),
        "param": res_full['best_params'].get('bandwidth')
    }

    # Print Summary
    print("\n" + "="*80)
    print(f"FINAL HEAD-TO-HEAD REPORT (OOS: {OOS_DATE}, Break: {detected_break_date.date()})")
    print("="*80)
    
    # Sort by MSE
    sorted_res = dict(sorted(results.items(), key=lambda item: item[1]['mse']))
    
    print(f"{'Model':<15} | {'MSFE':<15} | {'Main Param':<15}")
    print("-" * 55)
    for model, data in sorted_res.items():
        param_str = f"{data['param']:.4f}" if isinstance(data['param'], (int, float)) else str(data['param'])
        print(f"{model:<15} | {data['mse']:.9f}   | {param_str:<15}")
    print("-" * 55)
    
    # Explicit comparison
    log_mse = results["Log WLL"]["mse"]
    std_mse = results["Std WLL"]["mse"]
    diff = std_mse - log_mse
    
    if diff > 0:
        print(f"\n>> WINNER: Log WLL is better by {diff:.9f} ({(diff/std_mse)*100:.2f}%)")
    elif diff < 0:
        print(f"\n>> WINNER: Std WLL is better by {-diff:.9f} ({( -diff/std_mse)*100:.2f}%)")
    else:
        print(f"\n>> TIE: Both models performed identically.")

if __name__ == "__main__":
    run_head_to_head()
