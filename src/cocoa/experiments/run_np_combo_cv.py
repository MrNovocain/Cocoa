from cocoa.experiments.runner import NPComboExperimentRunner
import matplotlib.pyplot as plt
import pandas as pd

from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    BREAK_ID_ONE_BASED,
)
from cocoa.models.cocoa_data import CocoaDataset
### Last detection 5334
dataset = CocoaDataset(
    csv_path=PROCESSED_DATA_PATH,
    feature_cols=DEFAULT_FEATURE_COLS,
    target_col=DEFAULT_TARGET_COL,
)
# sample_start_index = dataset.get_1_based_index_from_date("2019-10-01")
sample_start_index = 3594
def run_np_combo_cv_for_gamma_analysis(start_index, end_index,jump_size=100):
    """
    Loops over a range of sample_start_index values, runs the NPConvexCombinationModel
    cross-validation, and records the chosen gamma for each index.

    :param start_index: The starting index for the loop.
    :param end_index: The ending index for the loop.
    """
    print(f"--- Starting NP Convex Combination Model CV for gamma analysis from index {start_index} to {end_index} ---")
    results_per_index = {}


    for i in range(start_index, end_index + 1, jump_size):
        print(f"\n--- Running for sample_start_index: {i} ---")
        runner = NPComboExperimentRunner(
            model_name="NP_LL_Combo",
            feature_cols=DEFAULT_FEATURE_COLS,
            break_date=dataset.get_date_from_1_based_index(i),
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_START_DATE,
            sample_start_index=i,  # Structural break, required for Combo model
            poly_order=1,
            save_results=True,  # Must be True to get OOS MSE
        )

        run_results = runner.run()
        # For hyperparameter tuning of the break date, use the in-sample CV score
        in_sample_cv_mse = run_results.get('in_sample_cv_mse')
        gamma = getattr(runner, 'gamma', 'N/A')

        results_per_index[i] = {'gamma': gamma, 'in_sample_cv_mse': in_sample_cv_mse}
        print(f"--- Finished for sample_start_index: {i}, Chosen gamma: {gamma}, In-Sample CV MSE: {in_sample_cv_mse} ---")

    print("\n--- NP Convex Combination Model CV for gamma analysis finished ---")
    print("\nIn-Sample CV Results per sample_start_index:")
    print(results_per_index)

    # --- Find the best structural break date based on the lowest in-sample CV MSE ---
    valid_results = {idx: res for idx, res in results_per_index.items() if res.get('in_sample_cv_mse') is not None}
    if valid_results:
        best_break_index = min(valid_results, key=lambda k: valid_results[k]['in_sample_cv_mse'])
        best_in_sample_mse = valid_results[best_break_index]['in_sample_cv_mse']
        best_break_date = dataset.get_date_from_1_based_index(best_break_index)
        gamma_for_best_model = valid_results[best_break_index]['gamma']

        print("\n--- Optimal Hyperparameters Found ---")
        print(f"Best structural break index: {best_break_index} (Date: {best_break_date.date()})")
        print(f"This was chosen based on the lowest In-Sample CV MSE: {best_in_sample_mse:.6f}")
        print(f"The optimal gamma for this break date was: {gamma_for_best_model:.4f}")

        # --- Now, run the model one last time with the optimal break date to get the final OOS performance ---
        print("\n--- Evaluating final model with optimal hyperparameters on the OOS test set ---")
        final_runner = NPComboExperimentRunner(
            model_name="NP_LL_Combo_Final",
            feature_cols=DEFAULT_FEATURE_COLS,
            break_date=best_break_date,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_START_DATE,
            sample_start_index=best_break_index,
            poly_order=1,
            save_results=True,
        )
        final_run_results = final_runner.run()
        final_oos_mse = final_run_results.get('oos_mse')

        print("\n--- Final Model Performance ---")
        print(f"Final Out-of-Sample MSE: {final_oos_mse:.6f}")

    # --- Plotting the results ---
    plot_data = {
        "index": [],
        "date": [],
        "gamma": [],
        "in_sample_cv_mse": [],
    }
    for index, data in results_per_index.items():
        if data['gamma'] != 'N/A' and data['gamma'] is not None:
            plot_data["index"].append(index)
            plot_data["date"].append(dataset.get_date_from_1_based_index(index))
            plot_data["gamma"].append(data['gamma'])
            if data['in_sample_cv_mse'] is not None:
                plot_data["in_sample_cv_mse"].append(data['in_sample_cv_mse'])

    if plot_data["date"]:
        plt.figure(figsize=(14, 7))
        plt.plot(plot_data["date"], plot_data["gamma"], marker='o', linestyle='-')
        plt.title('Optimal Gamma vs. Structural Break Start Date')
        plt.xlabel('Date of Structural Break')
        plt.ylabel('Optimal Gamma (Weight on Pre-Break Model)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        file_name_gamma = f"gamma_vs_break_date_{start_index}_{end_index}_{jump_size}.png"
        plt.savefig(f"w:/Research/NP/Cocoa/output/{file_name_gamma}", bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    if plot_data["in_sample_cv_mse"]:
        plt.figure(figsize=(14, 7))
        plt.plot(plot_data["date"], plot_data["in_sample_cv_mse"], marker='x', linestyle='--', color='r')
        plt.title('In-Sample CV MSE vs. Structural Break Start Date')
        plt.xlabel('Date of Structural Break')
        plt.ylabel('In-Sample Cross-Validation MSE')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        file_name_mse = f"mse_vs_break_date_{start_index}_{end_index}_{jump_size}.png"
        plt.savefig(f"w:/Research/NP/Cocoa/output/{file_name_mse}", bbox_inches="tight")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Define the index range you want to analyze
    run_np_combo_cv_for_gamma_analysis(start_index=200, end_index=6700, jump_size=200)
    # runner = NPComboExperimentRunner(
    #         model_name="NP_LL_Combo",
    #         feature_cols=DEFAULT_FEATURE_COLS,
    #         break_date=sample_start_index,
    #         target_col=DEFAULT_TARGET_COL,
    #         data_path=PROCESSED_DATA_PATH,
    #         oos_start_date=OOS_START_DATE,
    #         sample_start_index= sample_start_index,  # Structural break, required for Combo model
    #         poly_order=1,
    #         save_results=True,  # Must be True to get OOS MSE
    #     )

    # run_results = runner.run()