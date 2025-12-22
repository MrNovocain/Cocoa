
"""
Script to Reproduce "Plot 4" (Early Post-Break Adaptation Speed) with Clean UI
==============================================================================

Goal: Generate a clean, single-panel plot of Cumulative Squared Error (SE) 
      in the immediate post-break period (first 60 days).

Comparison:
    - WLL (Rank 1 Model)
    - Random Forest
    - XGBoost

"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Configuration
OOS_START_DATE = "2024-04-01"  # The El Niño Fixed Window
OOS_HORIZON = 60               # First X days to show "Immediate" adaptation
OUTPUT_FILE = "reports/advisor_meeting_visuals/4_speed_cumulative_error_clean.png"

def run_speed_plot():
    print(f"--- Generating Clean Speed Plot (OOS: {OOS_START_DATE}, Horizon: {OOS_HORIZON}) ---")
    
    # 1. Detect Break (Re-using logic)
    print("1. Detecting Break...")
    mohr_runner = MohrRunner(oos_start_date=OOS_START_DATE)
    break_index = mohr_runner.run_mohr_break_detection()
    
    # 2. Run Models & Capture Predictions
    print("2. Running Models...")
    
    # Grid for storing cumulative errors
    # We need predictions day-by-day. 
    # The Runners return 'oos_metrics' but we need the actual SERIES.
    # The Runners SAVE predictions to CSV if save_results=True.
    # Or we can modify them?
    # ExperimentRunner._save_artifacts saves predictions.csv.
    # But for a clean script, let's just use the fitted model to predict.
    
    ds = CocoaDataset()
    split = ds.split_oos_by_date(OOS_START_DATE)
    X_test = split.X_test.head(OOS_HORIZON)
    y_test = split.y_test.head(OOS_HORIZON)
    dates = ds.dates[split.X_test.index].head(OOS_HORIZON)
    
    # --- WLL ---
    print("   -> Fitting WLL...")
    wll_runner = ConvexComboExperimentRunner(
        combo_type='NP',
        model_name="WLL_Speed",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        break_index=break_index,
        poly_order=1,
        save_results=False
    )
    _, _, wll_model, _ = wll_runner._fit_model() # Access internal fit to get model object
    pred_wll = wll_model.predict(X_test)
    
    # --- RF ---
    print("   -> Fitting RF...")
    rf_runner = ExperimentRunner(
        model_name="RF",
        model_class=RFModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        save_results=False
    )
    _, _, rf_model, _ = rf_runner._fit_model()
    pred_rf = rf_model.predict(X_test)
    
    # --- XGB ---
    print("   -> Fitting XGB...")
    xgb_runner = ExperimentRunner(
        model_name="XGB",
        model_class=XGBModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        save_results=False
    )
    _, _, xgb_model, _ = xgb_runner._fit_model()
    pred_xgb = xgb_model.predict(X_test)
    
    # 3. Calculate Cumulative Squared Error
    def get_cse(y_true, y_pred):
        se = (y_true - y_pred) ** 2
        return np.cumsum(se)
    
    cse_wll = get_cse(y_test, pred_wll)
    cse_rf = get_cse(y_test, pred_rf)
    cse_xgb = get_cse(y_test, pred_xgb)
    
    # 4. Plot
    print("3. Plotting...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    x_axis = range(1, len(y_test) + 1)
    
    plt.plot(x_axis, cse_wll, label="WLL (Ours)", color="#2ECC71", linewidth=3)
    plt.plot(x_axis, cse_rf, label="Random Forest", color="#9B59B6", linewidth=2, linestyle="--")
    plt.plot(x_axis, cse_xgb, label="XGBoost", color="#F39C12", linewidth=2, linestyle="--")
    
    plt.title(f"Immediate Post-Break Adaptation Speed (First {OOS_HORIZON} Days)", fontsize=14, fontweight='bold')
    plt.xlabel("Trading Days Since Break Config (April 1, 2024)", fontsize=12)
    plt.ylabel("Cumulative Squared Forecast Error", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    
    # Annotation
    final_wll = cse_wll.iloc[-1]
    final_ml = min(cse_rf.iloc[-1], cse_xgb.iloc[-1])
    imp = (1 - final_wll/final_ml) * 100
    
    plt.annotate(f"WLL leads by {imp:.1f}%", 
                 xy=(OOS_HORIZON, final_wll), 
                 xytext=(OOS_HORIZON - 20, final_wll * 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Ensure dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    print(f"Success! Clean plot saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_speed_plot()
