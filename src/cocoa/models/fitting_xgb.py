from ..experiments.runner import ExperimentRunner
from ..models import XGBModel
from ..models.assets import (
    PROCESSED_DATA_PATH,
    DEFAULT_OOS_START_DATE,
    XGB_FEATURE_COLS,
    XGB_TARGET_COL,
    XGB_PARAM_GRID,
)
 
def main():
    """
    Configures and runs the XGBoost experiment.
    This function is intended to be called when this script is executed
    as part of the 'cocoa' package.
    """

    # Configure and run the XGBoost experiment
    xgb_experiment = ExperimentRunner(
        model_name="XGB",
        model_class=XGBModel,
        feature_cols=XGB_FEATURE_COLS,
        target_col=XGB_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=DEFAULT_OOS_START_DATE,
    )
    xgb_experiment.run()

if __name__ == "__main__":
    # To run this script directly, you must execute it as a module to ensure
    # that relative imports work correctly. From the project root directory
    # ('W:\Research\NP\Cocoa'), use the following command:
    # python -m cocoa.models.fitting_xgb
    main()