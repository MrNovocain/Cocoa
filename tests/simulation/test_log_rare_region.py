
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cocoa.simulation.dgp import RareRegionDGP
from cocoa.simulation.engine import SimulationEngine

def run_rare_region_test():
    print("--- Running Rare Region Simulation (CGS Special Case) ---")
    print("Goal: Show Log WLL wins in 'Rare Region + Huge Break' Scenario")
    
    # Configuration
    # Break 100: Massive.
    # Rare Region [0.95, 1.0].
    # Post-Break has very few points there.
    # Pre-Break has many.
    
    DRIFT = 100.0
    TRIALS = 3000
    
    print(f"Parameters:\n  Drift: {DRIFT}\n  Trials: {TRIALS}\n")
    
    # We use RareRegionDGP
    dgp = RareRegionDGP(T_pre=1000, T_post=50, break_size=DRIFT, noise_std=1.0)
    engine = SimulationEngine(dgp, n_trials=TRIALS)
    
    results = engine.run()
    
    # We look at Oracle Optimized results
    avg_linear = results["mse_linear_opt"].mean()
    avg_log = results["mse_log_opt"].mean()
    avg_post = results["mse_post_only"].mean()
    
    print("Results (Average Oracle-Optimized MSE):")
    print(f"  Post-Only (Pure NP):     {avg_post:.4f} (Suffers from sparsity/variance)")
    print(f"  Linear WLL (Std):        {avg_linear:.4f} (Must avoid bias)")
    print(f"  Log WLL:                 {avg_log:.4f} (Can mix!)")
    
    print("-" * 40)
    print("Avg Weights Selected:")
    print(f"  Linear Gamma: {results['w_linear_opt'].mean():.4f}")
    print(f"  Log Beta:     {results['w_log_opt'].mean():.4f}")
    
    print("-" * 40)
    
    # Fixed Weight Robustness
    fix_lin = results["mse_linear_0.5"].mean()
    fix_log = results["mse_log_0.5"].mean()
    print("Fixed Weight 0.5 Results:")
    print(f"  Linear (0.5): {fix_lin:.4f}")
    print(f"  Log (0.5):    {fix_log:.4f}")
    
    if avg_log < avg_post and avg_log < avg_linear:
         print("SUCCESS: Log WLL Dominates!")
    elif fix_log < fix_lin and fix_log < avg_post:
         print("SUCCESS: Log WLL Dominates in Robust/Fixed mode!")
         
    if fix_lin > 1000:
        print("Linear WLL exploded (Correct behavior for large bias).")

if __name__ == "__main__":
    run_rare_region_test()
