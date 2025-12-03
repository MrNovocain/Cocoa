import os
import sys

# Add src directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cocoa.experiments.runner import ExperimentRunner
from cocoa.models import RFModel
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    DEFAULT_OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)

if __name__ == "__main__":
    # Configure and run the Random Forest experiment on the FULL dataset
    rf_experiment = ExperimentRunner(
        model_name="RF_Full",
        model_class=RFModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=DEFAULT_OOS_START_DATE,
        # No sample_start_index is provided, so it runs on the full history
    )
    rf_experiment.run()
