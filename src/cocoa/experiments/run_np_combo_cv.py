from cocoa.experiments.runner import NPComboExperimentRunner

from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from cocoa.models.cocoa_data import CocoaDataset
### Last detection 5334
dataset = CocoaDataset(
    csv_path=PROCESSED_DATA_PATH,
    feature_cols=DEFAULT_FEATURE_COLS,
    target_col=DEFAULT_TARGET_COL,
)
# sample_start_index = dataset.get_1_based_index_from_date("2019-07-15")
sample_start_index = 5336
def run_np_combo_cv():
    """
    Runs the full cross-validation pipeline for the NPConvexCombinationModel
    using the dedicated ExperimentRunner.
    """
    print("--- Starting NP Convex Combination Model CV using ExperimentRunner ---")

    runner = NPComboExperimentRunner(
        model_name="NP_LL_Combo",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        sample_start_index=sample_start_index,  # Structural break, required for Combo model
        poly_order=1,  # As it was in the original script
        save_results=True,
    )

    runner.run()
    print("--- NP Convex Combination Model CV finished ---")


if __name__ == "__main__":
    run_np_combo_cv()