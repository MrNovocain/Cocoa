
import sys
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cocoa.experiments.break_detection import MohrRunner
from cocoa.models.generalized_combo import GeneralizedNonLinearComboModel
from cocoa.models.shrinkage import get_shrinkage_function
from cocoa.models.cocoa_data import CocoaDataset
from cocoa.models.assets import DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, PROCESSED_DATA_PATH
from cocoa.models.mfv_CV import MFVGeneralizedComboValidator, MFVConvexComboValidator
from cocoa.models.fitting_np_full import NPRegimeModel

def run_grid_search():
    print("--- Searching for Log WLL Dominance (2024-01 to 2024-05) ---")
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-05-01")
    
    current_date = start_date
    results = []
    
    dataset = CocoaDataset()
    # Need to load data once
    try:
        full_df = pd.read_csv(PROCESSED_DATA_PATH)
        full_df['date'] = pd.to_datetime(full_df['date'])
        full_df = full_df.sort_values('date')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    while current_date <= end_date:
        oos_date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n>> Checking Date: {oos_date_str}")
        
        try:
            # 1. Detect Break
            mohr = MohrRunner(oos_start_date=oos_date_str)
            break_idx = mohr.run_mohr_break_detection()
            break_date = dataset.get_date_from_1_based_index(break_idx)
            
            # 2. Setup Data
            # We need to create the pre/post models just like experiment runner
            # This is heavy to do fully inside loop if we re-fit every time.
            # But NP models are lazy (fit is cheap, predict is cost).
            # Let's rely on the Runners to handle the heavy lifting? 
            # Or assume we can mock it? No, need real results.
            # I will instantiate the ExperimentRunners but suppress output.
            
            from cocoa.experiments.generalized_runner import GeneralizedComboExperimentRunner
            from cocoa.experiments.runner import ConvexComboExperimentRunner
            
            # Log WLL (Run Only)
            log_runner = GeneralizedComboExperimentRunner(
                combo_type="NP", model_name="GridSeq", feature_cols=DEFAULT_FEATURE_COLS, target_col=DEFAULT_TARGET_COL,
                data_path=PROCESSED_DATA_PATH, oos_start_date=oos_date_str, break_index=break_idx,
                shrinkage_type="log", poly_order=1, save_results=False
            )
            # The runner.run() does everything: fit, tune, predict OOS
            log_res = log_runner.run() # Returns dict
            
            # Std WLL
            std_runner = ConvexComboExperimentRunner(
                combo_type="NP", model_name="GridSeq_Std", feature_cols=DEFAULT_FEATURE_COLS, target_col=DEFAULT_TARGET_COL,
                data_path=PROCESSED_DATA_PATH, oos_start_date=oos_date_str, break_index=break_idx,
                poly_order=1, save_results=False
            )
            std_res = std_runner.run()
            
            log_beta = log_res.get("best_params", {}).get("beta", 0.0)
            log_mse = log_res.get("oos_mse", 999.9)
            
            std_gamma = std_res.get("best_params", {}).get("gamma", 0.0)
            std_mse = std_res.get("oos_mse", 999.9)
            
            # Post-Only Logic (Gamma=0)
            # Both models converge to Post-Only if weight=0.
            # So if weight=0, that is the Post-Only MSE.
            
            diff = std_mse - log_mse
            
            print(f"   Break: {break_date.date()} (Idx {break_idx})")
            print(f"   Log Beta: {log_beta:.2f} | Std Gamma: {std_gamma:.2f}")
            if diff > 1e-9:
                print(f"   *** LOG WINS by {diff:.9f} ***")
            elif diff < -1e-9:
                 print(f"   std wins by {-diff:.9f}")
            else:
                 print("   Tie (Usually Post-Only)")
                 
            results.append({
                "date": oos_date_str, 
                "break_date": break_date.date(),
                "log_beta": log_beta,
                "log_mse": log_mse,
                "std_mse": std_mse,
                "diff": diff
            })
            
        except Exception as e:
            print(f"ERROR for {oos_date_str}: {e}")
        
        # Advance by 1 week
        current_date += pd.Timedelta(weeks=1)

    # Summary
    df_res = pd.DataFrame(results)
    winners = df_res[df_res['diff'] > 1e-9]
    if not winners.empty:
        print("\nFOUND WINNING DATES:")
        print(winners)
    else:
        print("\nNo dates found where Log WLL strictly beats Linear WLL.")

if __name__ == "__main__":
    run_grid_search()
