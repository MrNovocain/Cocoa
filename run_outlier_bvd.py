
from cocoa.experiments.runner import ConvexComboExperimentRunner
from cocoa.models import CocoaDataset
from cocoa.models.assets import DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, PROCESSED_DATA_PATH
import pandas as pd

def run_outlier_experiment():
    print("Initializing Outlier Experiment with BVD...")
    
    # 1. Setup Data
    ds = CocoaDataset()
    
    # Outlier parameters identified from analysis
    oos_start_date = "2024-02-06"
    break_date_str = "2005-11-11"
    
    print(f"Target OOS Date: {oos_start_date}")
    print(f"Break Date: {break_date_str}")
    
    # Get break index
    break_date = pd.Timestamp(break_date_str)
    try:
        break_index = ds.get_1_based_index_from_date(break_date)
        print(f"Break Index: {break_index}")
    except ValueError:
        print(f"Break date {break_date} not found exactly. Finding closest...")
        closest_date = ds.dates[ds.dates <= break_date].max()
        break_index = ds.get_1_based_index_from_date(closest_date)
        print(f"Closest Break Index: {break_index} ({closest_date})")

    # 2. Configure Runner
    # Using 'NP' combo (WLL) as it had the high MSFE
    runner = ConvexComboExperimentRunner(
        combo_type='NP',
        model_name="WLL_Outlier_Debug",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=oos_start_date,
        break_index=break_index,
        poly_order=1,
        save_results=True,
        run_bvd=True,          # ENABLE BVD
        n_bootstrap_rounds=50 # Standard number
    )
    
    # 3. Run
    print("Starting run... (This may take a moment for BVD)")
    results = runner.run()
    
    print("\n--- Experiment Complete ---")
    print(f"OOS MSE: {results.get('oos_mse')}")
    print(f"BVD MSE: {results.get('avg_mse')}")
    print(f"BVD Bias^2: {results.get('avg_bias_sq')}")
    print(f"BVD Variance: {results.get('avg_variance')}")
    print(f"Output Directory: {runner.output_dir}")

if __name__ == "__main__":
    run_outlier_experiment()
