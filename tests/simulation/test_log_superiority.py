
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from cocoa.simulation.dgp import LargeBreakDGP
from cocoa.simulation.engine import SimulationEngine

def run_log_superiority_demo():
    print("--- Running Log WLL Superiority Simulation ---")
    
    # SCENARIO: Massive Break (20 sigma), Small Post-Break Sample (N=10)
    # Why Log Wins?
    # Linear WLL with Gamma=1 adds (Pre - Post) ~ (0 - 20) = -20 correction. 
    # If Pre is 0, Post is 20. Target is 20.
    # Prediction = Post + 1.0 * (Pre - Post) = 20 + (0 - 20) = 0. 
    # Linear WLL wipes out the structural break adaptation if weight is 1.0!
    # Log WLL with Beta=1 adds sgn(diff)*log(1+|diff|).
    # Diff = -20. Log correction = -1 * log(21) ~ -3.
    # Prediction = Post + Correction = 20 - 3 = 17.
    # Target = 20.
    # Linear Error = (0 - 20)^2 = 400.
    # Log Error = (17 - 20)^2 = 9.
    # Log WLL wins massively because it "refuses" to fully revert to the pre-break mean.
    
    DRIFT = 20.0
    T_POST = 10
    TRIALS = 1000
    
    print(f"Parameters:\n  Drift: {DRIFT}\n  T_Post: {T_POST}\n  Trials: {TRIALS}\n")
    
    # Run Simulation
    dgp = LargeBreakDGP(T_pre=1000, T_post=T_POST, drift_size=DRIFT, noise_std=1.0)
    engine = SimulationEngine(dgp, n_trials=TRIALS)
    
    results = engine.run()
    
    avg_mse_linear = results["mse_linear_opt"].mean()
    avg_w_linear = results["w_linear_opt"].mean()
    
    avg_mse_log = results["mse_log_opt"].mean()
    avg_w_log = results["w_log_opt"].mean()
    
    avg_mse_post = results["mse_post_only"].mean()
    
    print("Results (Average Oracle-Optimized MSE):")
    print(f"  Standard WLL (Avg Opt Gamma={avg_w_linear:.2f}): {avg_mse_linear:.4f}")
    print(f"  Log WLL (Avg Opt Beta={avg_w_log:.2f}):          {avg_mse_log:.4f}")
    print(f"  Post-Only (Gamma=0.0):                           {avg_mse_post:.4f}")
    
    print("-" * 40)
    print("Interpretation:")
    print("With Oracle Tuning, both Linear and Log models should correctly identify\nthat the Post-Break model is superior (Weight ~ 0).")
    print(f"Post-Only MSE: {avg_mse_post:.4f}")
    
    diff_linear = abs(avg_mse_linear - avg_mse_post)
    diff_log = abs(avg_mse_log - avg_mse_post)
    
    if diff_linear < 0.1 and diff_log < 0.1:
        print("SUCCESS: Both models successfully recovered the Post-Only performance (Weights -> 0).")
        print("This confirms the implementation logic is correct.")
    else:
        print("WARNING: One or both models failed to match Post-Only performance.")
        
    print("-" * 40)
    print("Safety Check (Comparison of 'Bad' Weights):")
    print("If tuning fails (e.g. Gamma=1), Log WLL is much safer (as shown in previous fixed-weight test).")


if __name__ == "__main__":
    run_log_superiority_demo()
