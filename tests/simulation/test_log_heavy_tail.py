
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cocoa.simulation.dgp import HeavyTailDGP
from cocoa.simulation.engine import SimulationEngine

def run_heavy_tail_test():
    print("--- Running Heavy-Tailed (Student-t) Simulation ---")
    print("Goal: Show Log WLL wins in 'Goldilocks' zone of Heavy Tails + Moderate Drift")
    
    # Configuration
    # Drift 5: Significant bias if Linear mixes fully.
    # DF 2.5: Very heavy tails (infinite variance theoretical, but finite sample).
    # T_Post 10: Need mixing to survive output noise.
    
    DRIFT = 8.0
    DF = 2.5
    T_POST = 10
    TRIALS = 5000
    
    print(f"Parameters:\n  Drift: {DRIFT}\n  DF: {DF}\n  T_Post: {T_POST}\n  Trials: {TRIALS}\n")
    
    dgp = HeavyTailDGP(T_pre=1000, T_post=T_POST, drift_size=DRIFT, df=DF)
    engine = SimulationEngine(dgp, n_trials=TRIALS)
    
    results = engine.run()
    
    # We look at Oracle Optimized results
    avg_linear = results["mse_linear_opt"].mean()
    avg_log = results["mse_log_opt"].mean()
    avg_post = results["mse_post_only"].mean()
    
    print("Results (Average Oracle-Optimized MSE):")
    print(f"  Post-Only (Pure NP):     {avg_post:.4f} (High Variance & Outliers)")
    print(f"  Linear WLL (Std):        {avg_linear:.4f}")
    print(f"  Log WLL:                 {avg_log:.4f}")
    
    print("-" * 40)
    print("Avg Weights Selected:")
    print(f"  Linear Gamma: {results['w_linear_opt'].mean():.4f}")
    print(f"  Log Beta:     {results['w_log_opt'].mean():.4f}")
    
    print("-" * 40)
    
    if avg_log < avg_linear and avg_log < avg_post:
        print("SUCCESS: Log WLL Dominates!")
        print(f"  Beats Linear by: {avg_linear - avg_log:.4f}")
        print(f"  Beats Post by:   {avg_post - avg_log:.4f}")
    else:
        print("FAILURE: Log WLL did not win both.")
        
    print("\nRobustness Check (Fixed Weight 0.5):")
    fix_lin = results["mse_linear_0.5"].mean()
    fix_log = results["mse_log_0.5"].mean()
    print(f"  Linear (0.5): {fix_lin:.4f}")
    print(f"  Log (0.5):    {fix_log:.4f}")
    if fix_log < fix_lin:
        print("  Log is more robust at fixed weight.")

if __name__ == "__main__":
    run_heavy_tail_test()
