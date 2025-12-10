
from cocoa.experiments.runner import ConvexComboExperimentRunner
from cocoa.models import CocoaDataset
from cocoa.models.assets import DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL, PROCESSED_DATA_PATH
import pandas as pd
import numpy as np

class FixedGammaRunner(ConvexComboExperimentRunner):
    """
    Experimental runner that forces Gamma to a fixed value (1.0).
    """
    def _tune_gamma(self, sub_model_class, params_pre, params_post, X_full, y_full, break_idx):
        print("\n--- (3/3) Forced Gamma (Diagnostic) ---")
        forced_gamma = 1.0
        print(f"Forcing Gamma to {forced_gamma} (Full Pre-Break Model)")
        
        self.gamma = forced_gamma
        # Return dummy score as we are skipping tuning
        return forced_gamma, 0.0

def run_gamma1_experiment():
    print("Initializing Outlier Experiment with Gamma=1...")
    
    # 1. Setup Data
    ds = CocoaDataset()
    oos_start_date = "2024-02-06"
    break_date_str = "2005-11-11"
    
    break_date = pd.Timestamp(break_date_str)
    try:
        break_index = ds.get_1_based_index_from_date(break_date)
    except ValueError:
        closest_date = ds.dates[ds.dates <= break_date].max()
        break_index = ds.get_1_based_index_from_date(closest_date)

    # 2. Configure Runner
    runner = FixedGammaRunner(
        combo_type='NP',
        model_name="WLL_Outlier_Gamma1",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=oos_start_date,
        break_index=break_index,
        poly_order=1,
        save_results=True,
        run_bvd=True,  # Keep BVD to compare decomposition
        n_bootstrap_rounds=50
    )
    
    print("Starting run with Gamma=1...")
    results = runner.run()
    
    print("\n--- Gamma=1 Experiment Complete ---")
    print(f"OOS MSE: {results.get('oos_mse')}")
    print(f"BVD MSE: {results.get('avg_mse')}")
    print(f"BVD Bias^2: {results.get('avg_bias_sq')}")
    print(f"BVD Variance: {results.get('avg_variance')}")

if __name__ == "__main__":
    run_gamma1_experiment()
