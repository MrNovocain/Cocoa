
import os
import sys
import pandas as pd
from pathlib import Path

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2]))

from cocoa.experiments.generalized_runner import GeneralizedComboExperimentRunner
from cocoa.experiments.runner import ConvexComboExperimentRunner
from cocoa.experiments.break_detection import MohrRunner
from cocoa.models.cocoa_data import CocoaDataset
from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
    BREAK_ID_ONE_BASED, # We might override this
)

def run_rolling_mimic_test():
    print("--- Starting Rolling WLL Mimic Test ---")
    
    # Target OOS Date from user request
    OOS_DATE = "2024-04-01"
    FORCED_BREAK_INDEX = 2085 # 2005-11-11 (Old Regime Reference)
    
    print(f"Configuration:\n  OOS Date: {OOS_DATE}\n  Target: {DEFAULT_TARGET_COL}\n")

    # 1. Dynamic Break Detection (Mimicking rolling_wll)
    print("\n[Step 1] Running Mohr Break Detection...")
    # NOTE: MohrRunner typically takes the OOS start date to define the training window end
    mohr_runner = MohrRunner(oos_start_date=OOS_DATE)
    detected_break_index = mohr_runner.run_mohr_break_detection()
    
    dataset = CocoaDataset()
    detected_break_date = dataset.get_date_from_1_based_index(detected_break_index)
    print(f"  -> Detected Break Index: {detected_break_index} (Date: {detected_break_date.date()})")
    print(f"  -> User Forced Index:    {FORCED_BREAK_INDEX}")

    # Helper to run comparison for a given break index
    def run_comparison(name_suffix, break_idx):
        print(f"\n  >> Running Comparison for {name_suffix} (Break Index: {break_idx})...")
        
        # Log WLL
        log_runner = GeneralizedComboExperimentRunner(
            combo_type="NP",
            model_name=f"NP_Gen_Log_{name_suffix}",
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_DATE,
            break_index=break_idx,
            shrinkage_type="log",
            poly_order=1,
            save_results=True,
        )
        log_res = log_runner.run()
        
        # Standard WLL
        std_runner = ConvexComboExperimentRunner(
            combo_type="NP",
            model_name=f"NP_Std_{name_suffix}",
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_DATE,
            break_index=break_idx,
            poly_order=1,
            save_results=True,
        )
        std_res = std_runner.run()
        
        return log_res, std_res

    # 2. Run Comparison with DETECTED Break
    log_res_det, std_res_det = run_comparison("Detected", detected_break_index)
    
    # 3. Run Comparison with FORCED Break (if different)
    if detected_break_index != FORCED_BREAK_INDEX:
        log_res_forced, std_res_forced = run_comparison("Forced", FORCED_BREAK_INDEX)
    else:
        log_res_forced, std_res_forced = log_res_det, std_res_det
        print("\n  >> Skipping Forced Break run (same as detected).")

    # 4. Compile and Print Report
    print("\n" + "="*80)
    print(f"FINAL REPORT (OOS Date: {OOS_DATE})")
    print("="*80)
    
    def print_row(label, log_r, std_r):
        log_beta = log_r.get("best_params", {}).get("beta", "N/A")
        log_mse = log_r.get("oos_mse", "N/A")
        std_gamma = std_r.get("best_params", {}).get("gamma", "N/A")
        std_mse = std_r.get("oos_mse", "N/A")
        
        diff = std_mse - log_mse if (isinstance(std_mse, float) and isinstance(log_mse, float)) else "N/A"
        better = "Log" if diff > 0 else "Std" if diff < 0 else "Same"
        
        print(f"--- {label} ---")
        print(f"  Log WLL (Beta={log_beta}): MSE = {log_mse}")
        print(f"  Std WLL (Gamma={std_gamma}): MSE = {std_mse}")
        print(f"  Diff (Std - Log): {diff:.9f} ({better} is better)")
        print("")

    print_row(f"Using DETECTED Break ({detected_break_index})", log_res_det, std_res_det)
    
    if detected_break_index != FORCED_BREAK_INDEX:
        print_row(f"Using FORCED Break ({FORCED_BREAK_INDEX})", log_res_forced, std_res_forced)

if __name__ == "__main__":
    run_rolling_mimic_test()
