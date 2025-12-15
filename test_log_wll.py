
import os
import sys
from pathlib import Path

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from cocoa.experiments.generalized_runner import GeneralizedComboExperimentRunner
from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
    DEFAULT_OOS_START_DATE,
    BREAK_ID_ONE_BASED,
)

def run_log_wll_test():
    print("--- Starting Single Test for Log Order WLL (Generalized Combo) ---")
    
    # Use default break index if available, else hardcode the one from run_np_combo_cv.py
    break_date_index = BREAK_ID_ONE_BASED if BREAK_ID_ONE_BASED else 6117
    
    runner = GeneralizedComboExperimentRunner(
        combo_type="NP",
        model_name="NP_Gen_Log_Test",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=DEFAULT_OOS_START_DATE,
        break_index=break_date_index,
        shrinkage_type="log",    # <--- The key requested feature
        poly_order=1,
        save_results=False,       # Don't clutter with full artifacts, just want console output
        output_base_dir="output/test_runs"
    )
    
    print(f"Runner initialized. Break Index: {break_date_index}")
    
    results = runner.run()
    
    print("\n" + "="*40)
    print("TEST RESULTS")
    print("="*40)
    
    best_params = results.get("best_params", {})
    beta = best_params.get("beta")
    oos_mse = results.get("oos_mse")
    
    print(f"Optimal Beta: {beta}")
    print(f"OOS MSE:      {oos_mse}")
    print(f"Full Params:  {best_params}")
    print("="*40)

if __name__ == "__main__":
    run_log_wll_test()
