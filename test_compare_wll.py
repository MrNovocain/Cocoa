
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from cocoa.experiments.generalized_runner import GeneralizedComboExperimentRunner
from cocoa.experiments.runner import ConvexComboExperimentRunner
from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
    DEFAULT_OOS_START_DATE,
    BREAK_ID_ONE_BASED,
)

def run_comparison_test():
    print("--- Starting Comparison: Log Order vs Standard WLL ---")
    
    break_date_index = BREAK_ID_ONE_BASED if BREAK_ID_ONE_BASED else 6117
    print(f"Comparison Settings:\n  Break Index: {break_date_index}\n  Target: {DEFAULT_TARGET_COL}\n")

    # 1. Run Log Order WLL
    print("\n[1/2] Running Log Order (Generalized) WLL...")
    log_runner = GeneralizedComboExperimentRunner(
        combo_type="NP",
        model_name="NP_Gen_Log_Test",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=DEFAULT_OOS_START_DATE,
        break_index=break_date_index,
        shrinkage_type="log",
        poly_order=1,
        save_results=True,
        output_base_dir="output/test_runs_compare"
    )
    log_results = log_runner.run()

    # 2. Run Standard WLL (Convex Combo)
    print("\n[2/2] Running Standard WLL (Linear Convex Combo)...")
    std_runner = ConvexComboExperimentRunner(
        combo_type="NP",
        model_name="NP_Standard_Test",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=DEFAULT_OOS_START_DATE,
        break_index=break_date_index,
        poly_order=1,
        save_results=True,
        output_base_dir="output/test_runs_compare" # Passed but might not be used by base class init logic identically
    )
    std_results = std_runner.run()
    
    # 3. Compile Results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Log Model Data
    log_beta = log_results.get("best_params", {}).get("beta", "N/A")
    log_mse = log_results.get("oos_mse", "N/A")
    
    # Standard Model Data
    # ConvexComboRunner stores 'gamma' in the runner instance or result
    # It might be in best_params['gamma'] or runner.gamma
    std_gamma = std_results.get("best_params", {}).get("gamma", getattr(std_runner, 'gamma', "N/A"))
    std_mse = std_results.get("oos_mse", "N/A")
    
    results_df = pd.DataFrame({
        "Model": ["Log Order WLL", "Standard WLL"],
        "Shrinkage": ["Logarithmic", "Linear (Identity)"],
        "Tuning Param": ["Beta", "Gamma"],
        "Optimal Value": [log_beta, std_gamma],
        "OOS MSE": [log_mse, std_mse]
    })
    
    print(results_df.to_string(index=False))
    
    diff = std_mse - log_mse
    print("-" * 60)
    if diff > 0:
        print(f"Log Order WLL is BETTER by {diff:.8f} (Lower MSE)")
    elif diff < 0:
        print(f"Standard WLL is BETTER by {abs(diff):.8f} (Lower MSE)")
    else:
        print("Models have IDENTICAL performance.")
    print("="*60)

if __name__ == "__main__":
    run_comparison_test()
