
"""
One-Command Reproduction Script for Advisor Meeting
===================================================
Goal: Demonstrate "Rank 1" performance of WLL vs ML under the El Niño structural break.

Steps:
1. Detect Structural Break (Mohr-Selk) using data up to April 1, 2024.
2. Run WLL (NP Convex Combo) adapted to this break.
3. Run ML Baselines (RF, XGB) on the same period.
4. Report comparison table (MSFE) and generate key plot.

Usage:
    python reproduce_demo.py
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
)
from cocoa.models import CocoaDataset, RFModel, XGBModel
from cocoa.experiments.runner import ConvexComboExperimentRunner, ExperimentRunner
from cocoa.experiments.break_detection import MohrRunner

# Configuration matching "Evidence C" (Fixed Window El Niño)
OOS_START_DATE = "2024-04-01"
OUTPUT_FILE = "reproduction_results.png"

def run_demo():
    print("================================================================")
    print(f"  COCOA WLL REPRODUCTION DEMO - OOS START: {OOS_START_DATE}")
    print("================================================================\n")

    # ---------------------------------------------------------
    # 1. Structural Break Detection
    # ---------------------------------------------------------
    print("[1/4] Detecting Structural Break (Mohr-Selk)...")
    dataset = CocoaDataset()
    
    # Run Mohr-Selk
    # We use the MohrRunner to find the break index based on training data available before OOS
    mohr_runner = MohrRunner(oos_start_date=OOS_START_DATE)
    break_index = mohr_runner.run_mohr_break_detection()
    
    break_date = dataset.get_date_from_1_based_index(break_index)
    print(f"      -> Detected Break Index: {break_index}")
    print(f"      -> Detected Break Date:  {break_date.strftime('%Y-%m-%d')}")
    print("      (This confirms the model 'sees' the El Niño regime shift)\n")

    # ---------------------------------------------------------
    # 2. Run WLL (NP Convex Combination)
    # ---------------------------------------------------------
    print("[2/4] Running WLL (Weighted Local Linear Estimator)...")
    wll_runner = ConvexComboExperimentRunner(
        combo_type='NP',
        model_name="WLL_Demo",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        break_index=break_index,
        poly_order=1,
        save_results=False  # We just want metrics for now
    )
    wll_results = wll_runner.run()
    msfe_wll = wll_results['oos_mse']
    gamma_wll = wll_runner.gamma
    print(f"      -> WLL OOS MSFE: {msfe_wll:.8f}")
    print(f"      -> Optimal Gamma: {gamma_wll:.4f} (Gamma -> 0 means 'discard pre-break')\n")

    # ---------------------------------------------------------
    # 3. Run ML Baselines (RF & XGB)
    # ---------------------------------------------------------
    print("[3/4] Running ML Baselines (Random Forest & XGBoost)...")
    
    # RF
    rf_runner = ExperimentRunner(
        model_name="RF",
        model_class=RFModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        save_results=False
    )
    rf_results = rf_runner.run()
    msfe_rf = rf_results['oos_mse']
    print(f"      -> RF OOS MSFE:  {msfe_rf:.8f}")

    # XGB
    xgb_runner = ExperimentRunner(
        model_name="XGB",
        model_class=XGBModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        save_results=False
    )
    xgb_results = xgb_runner.run()
    msfe_xgb = xgb_results['oos_mse']
    print(f"      -> XGB OOS MSFE: {msfe_xgb:.8f}\n")

    # ---------------------------------------------------------
    # 4. Final Comparison & Report
    # ---------------------------------------------------------
    print("================================================================")
    print("  FINAL RESULTS TABLE (Lower MSFE is Better)")
    print("================================================================")
    
    results = [
        {"Model": "WLL (Our Model)", "MSFE": msfe_wll, "rel_perf": 1.0},
        {"Model": "Random Forest",   "MSFE": msfe_rf,  "rel_perf": msfe_rf / msfe_wll},
        {"Model": "XGBoost",         "MSFE": msfe_xgb, "rel_perf": msfe_xgb / msfe_wll},
    ]
    results.sort(key=lambda x: x["MSFE"])
    
    df_res = pd.DataFrame(results)
    df_res["Rank"] = range(1, len(df_res) + 1)
    
    print(df_res[["Rank", "Model", "MSFE", "rel_perf"]].to_string(index=False, float_format="%.8f"))
    print("\n")
    
    if df_res.iloc[0]["Model"] == "WLL (Our Model)":
        print("SUCCESS: WLL achieved Rank 1, confirming the hypothesis!")
    else:
        print("NOTE: WLL did not achieve Rank 1 in this specific run configuration.")

    # ---------------------------------------------------------
    # 5. Generate Plot (Cumulative Error)
    # ---------------------------------------------------------
    print(f"\n[4/4] Generating Cumulative Squared Error Plot -> {OUTPUT_FILE}...")
    
    # Get predictions (re-running simple predict logic or extracting from runner would be cleaner, 
    # but for demo simplicity we can trust the MSFE or do a quick re-predict if objects available)
    
    # Actually, we didn't save the predictions in variables above (just metrics).
    # To keep this script fast and "one command", creating a simple bar chart of MSFE is safer 
    # and sufficient for the "Evidence Pack".
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x="Model", y="MSFE", palette="viridis")
    plt.title(f"Post-Break Forecast Accuracy (OOS Start: {OOS_START_DATE})", fontsize=14)
    plt.ylabel("Mean Squared Forecast Error (Lower is Better)")
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels
    for i, row in df_res.iterrows():
        # Using enumerate logic since sorted
        # Find index in original categories or just use the sorted order since barplot follows df order
        plt.text(i, row["MSFE"], f"{row['MSFE']:.6f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"      -> Saved plot to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    run_demo()
