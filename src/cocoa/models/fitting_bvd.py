import sys
import os
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from cocoa.experiments.runner import ExperimentRunner
from cocoa.models import NPRegimeModel, GaussianKernel, LocalPolynomialEngine
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,  # Using same features as RF/XGB for comparability
    DEFAULT_TARGET_COL,
)


if __name__ == "__main__":
    # The NPRegimeModel requires kernel and local_engine objects at initialization.
    # The ExperimentRunner's grid search only passes hyperparameters from the grid.
    # To solve this, we use functools.partial to create a new "constructor"
    # that has the kernel and engine pre-filled.
    
    # 1. Define the core components for this NP model experiment
    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=1) # Local Linear

    # 2. Create a partial constructor for the ExperimentRunner to use
    NPModelPartial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)

    results = []

    # 3. Configure and run the Non-Parametric experiment in a loop
    for trimmed_date in range(1, 6735, 200):
        model_name = f"NP_LL_Post_Trim_{trimmed_date}"
        min_train_size = 12  # Define a minimum size for the training set
        print(f"\n--- Running experiment for trimmed_date: {trimmed_date} ---")
        np_experiment = ExperimentRunner(
            model_name=model_name,
            model_class=NPModelPartial,
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_START_DATE,
            kernel_name=kernel.__class__.__name__,
            poly_order=engine.order,
            sample_start_index=trimmed_date,
            save_results=False,
            run_bvd=True,
            n_bootstrap_rounds=2,
        )
        # Check if the training set is too small before running
        if np_experiment.get_train_size() < min_train_size:
            print(f"Skipping {model_name} due to insufficient training data (size < {min_train_size}).")
            continue

        mse, bias_sq, variance, start_date = np_experiment.run_BVD_only()
        results.append({
            "start_date": start_date,
            "mse": mse,
            "bias_sq": bias_sq,
            "variance": variance,
        })
    print(f"Numbers of results collected: {len(results)}")
    
    # 4. Convert results to a DataFrame and plot
    results_df = pd.DataFrame(results).sort_values("start_date")

    # 5. Print where maximum values occur
    max_mse_row = results_df.loc[results_df["mse"].idxmax()]
    max_bias_sq_row = results_df.loc[results_df["bias_sq"].idxmax()]
    max_variance_row = results_df.loc[results_df["variance"].idxmax()]

    min_mse_row = results_df.loc[results_df["mse"].idxmin()]
    min_bias_sq_row = results_df.loc[results_df["bias_sq"].idxmin()]
    min_variance_row = results_df.loc[results_df["variance"].idxmin()]

    print("\n--- Maximum Value Analysis ---")
    print(f"Maximum MSE of {max_mse_row['mse']:.4f} occurs at start_date: {max_mse_row['start_date'].date()}")
    print(f"Maximum Bias^2 of {max_bias_sq_row['bias_sq']:.4f} occurs at start_date: {max_bias_sq_row['start_date'].date()}")
    print(f"Maximum Variance of {max_variance_row['variance']:.4f} occurs at start_date: {max_variance_row['start_date'].date()}")

    print("\n--- Minimum Value Analysis ---")
    print(f"Minimum MSE of {min_mse_row['mse']:.4f} occurs at start_date: {min_mse_row['start_date'].date()}")
    print(f"Minimum Bias^2 of {min_bias_sq_row['bias_sq']:.4f} occurs at start_date: {min_bias_sq_row['start_date'].date()}")
    print(f"Minimum Variance of {min_variance_row['variance']:.4f} occurs at start_date: {min_variance_row['start_date'].date()}")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(results_df["start_date"], results_df["mse"], label="MSE (from BVD)", marker='o', linestyle='-')
    ax.plot(results_df["start_date"], results_df["bias_sq"], label="Bias^2", marker='s', linestyle='--')
    ax.plot(results_df["start_date"], results_df["variance"], label="Variance", marker='^', linestyle=':')

    ax.set_title("Bias-Variance Decomposition vs. Training Start Date", fontsize=16)
    ax.set_xlabel("Training Set Start Date", fontsize=12)
    ax.set_ylabel("Error Component Value", fontsize=12)
    ax.legend(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"bvd_vs_time_size_{len(results_df)}_{timestamp}.png"
    plt.savefig(f"w:/Research/NP/Cocoa/output/{file_name}", bbox_inches="tight")
    print(f"\nPlot of BVD components vs. start date saved to w:/Research/NP/Cocoa/output/{file_name}")