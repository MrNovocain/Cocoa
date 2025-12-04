
import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
)
from cocoa.models import CocoaDataset
from cocoa.experiments.specific_runners import LogComboExperimentRunner
from cocoa.experiments.break_detection import MohrRunner

def test_single_trial():
    print("Setting up test trial...")
    
    # 1. Load Dataset to get dates
    dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
    )
    
    # 2. Pick a recent date for OOS start (e.g., 20 days from end)
    last_date = dataset.get_last_date()
    last_idx = dataset.get_1_based_index_from_date(last_date)
    oos_idx = last_idx - 20
    oos_start_date = dataset.get_date_from_1_based_index(oos_idx)
    print(f"OOS Start Date: {oos_start_date}")

    # 3. Detect Break (Pilot Mohr)
    print("Running Pilot Mohr detection...")
    mohr_runner = MohrRunner(
        oos_start_date=last_date, # Use full sample for pilot
        dataset=dataset,
    )
    break_index = mohr_runner.run_mohr_break_detection()
    print(f"Detected Break Index: {break_index}")

    # 4. Run LogComboExperimentRunner
    print("\nRunning LogComboExperimentRunner...")
    runner = LogComboExperimentRunner(
        combo_type='NP',
        model_name="Test_LogCombo_NP",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=oos_start_date,
        break_index=break_index,
        poly_order=1,
        save_results=False, # Don't clutter output dir for this test
    )
    
    results = runner.run()
    
    print("\n--- Test Results ---")
    print(f"Best Beta: {results['best_params']['beta']}")
    print(f"OOS MSE: {results['oos_mse']}")
    print("Test completed successfully.")

if __name__ == "__main__":
    test_single_trial()
