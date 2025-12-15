
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cocoa.simulation.dgp import LargeBreakDGP
from cocoa.simulation.engine import SimulationEngine

def run_dominance_test():
    print("--- Running Log WLL Dominance Simulation (Fixed Weight 0.5) ---")
    print("Goal: Show Log WLL (0.5) < Linear WLL (0.5) AND Log WLL (0.5) < Post-Only")
    
    # "The Sweet Spot" for Robustness
    # Drift 15, Noise 10, T=5.
    
    DRIFT = 15.0
    NOISE = 10.0
    T_POST = 5
    TRIALS = 5000
    
    print(f"Parameters:\n  Drift: {DRIFT}\n  Noise: {NOISE}\n  T_Post: {T_POST}\n  Trials: {TRIALS}\n")
    
    dgp = LargeBreakDGP(T_pre=1000, T_post=T_POST, drift_size=DRIFT, noise_std=NOISE)
    engine = SimulationEngine(dgp, n_trials=TRIALS)
    
    results = engine.run()
    
    # Calculate Averages
    avg_linear = results["mse_linear_0.5"].mean()
    avg_log = results["mse_log_0.5"].mean()
    avg_post = results["mse_post_only"].mean()
    
    print("Results (Average MSE with Fixed Weight 0.5):")
    print(f"  Post-Only (Pure NP):     {avg_post:.4f}")
    print(f"  Linear WLL (Gamma=0.5):  {avg_linear:.4f}")
    print(f"  Log WLL (Beta=0.5):      {avg_log:.4f}")
    
    print("-" * 40)
    
    success_linear = avg_log < avg_linear
    success_post = avg_log < avg_post
    
    if success_linear:
        print(f"SUCCESS: Log WLL beats Linear WLL by {avg_linear - avg_log:.4f}")
    else:
        print(f"FAILURE: Linear beat Log.")

    if success_post:
        print(f"SUCCESS: Log WLL beats Post-Only by {avg_post - avg_log:.4f}")
    else:
        print(f"FAILURE: Post-Only beat Log.")

    if success_linear and success_post:
        print("\n*** DOUBLE WIN: LOG WLL DOMINATES ***")
    else:
        print("\n*** NO DOMINANCE ***")

if __name__ == "__main__":
    run_dominance_test()
