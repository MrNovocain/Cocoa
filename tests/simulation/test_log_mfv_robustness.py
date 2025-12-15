
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cocoa.simulation.dgp import MisleadingDGP
from cocoa.simulation.engine import SimulationEngine

def run_mfv_robustness_test():
    print("--- Running MFV Robustness Simulation (Misleading History) ---")
    print("Goal: Show that if MFV tunes on quiet data, Log WLL survives a future Massive Break better than Linear WLL.")
    
    # Configuration
    # Validation Period: No Break (Mean 0 -> Mean 0).
    # OOS Period: Massive Break (Mean 100).
    # MFV Strategy: Tuning on Validation should select High Gamma (mixing reduces var).
    
    DRIFT_OOS = 100.0
    DRIFT_VAL = 8.0
    TRIALS = 2000
    
    print(f"Parameters:\n  OOS Drift: {DRIFT_OOS}\n  Validation Drift: {DRIFT_VAL}\n  Trials: {TRIALS}\n")
    
    dgp = MisleadingDGP(T_pre=1000, T_post_val=20, break_size_val=DRIFT_VAL, break_size_oos=DRIFT_OOS, noise_std=5.0)
    engine = SimulationEngine(dgp, n_trials=TRIALS)
    
    results = engine.run()
    
    # Results
    avg_linear = results["mse_linear_tuned"].mean()
    avg_log = results["mse_log_tuned"].mean()
    avg_post = results["mse_post_only"].mean()
    
    w_lin = results["w_linear_tuned"].mean()
    w_log = results["w_log_tuned"].mean()
    
    print("Results (Average Tuned MSE):")
    print(f"  Post-Only (Benchmark):   {avg_post:.4f} (Unbiased, Variance=1.0)")
    print(f"  Linear WLL (Tuned):      {avg_linear:.4f} (Gamma={w_lin:.2f})")
    print(f"  Log WLL (Tuned):         {avg_log:.4f} (Beta={w_log:.2f})")
    
    print("-" * 40)
    
    # Linear Bias ~ (Gamma * 100)^2
    # Log Bias ~ (Beta * log(100))^2 ~ (Beta * 4.6)^2
    
    if avg_log < avg_linear:
         print("SUCCESS: Log WLL Beats Linear WLL!")
         print(f"  Win Margin: {avg_linear - avg_log:.4f}")
    
    if avg_log < 200 and avg_linear > 1000:
        print("  Log WLL Effectively 'Survived' the break, Linear WLL Exploded.")

if __name__ == "__main__":
    run_mfv_robustness_test()
