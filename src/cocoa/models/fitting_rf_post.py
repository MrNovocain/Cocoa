import os
import sys

# Add src directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cocoa.experiments.runner import ExperimentRunner
from cocoa.models import RFModel
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    BREAK_ID_ONE_BASED,
)

if __name__ == "__main__":
    # Configure and run the Random Forest experiment on POST-BREAK data
    rf_post_experiment = ExperimentRunner(
        model_name="RF_Post",
        model_class=RFModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        sample_start_index=BREAK_ID_ONE_BASED,  # This is the key change
    )
    rf_post_experiment.run()