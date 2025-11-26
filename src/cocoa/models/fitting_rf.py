from ..experiments.runner import ExperimentRunner
from . import RFModel
from .assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    RF_PARAM_GRID,
)
 
if __name__ == "__main__":
    # Configure and run the Random Forest experiment
    rf_experiment = ExperimentRunner(
        model_name="RF",
        model_class=RFModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        param_grid=RF_PARAM_GRID,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
    )
    rf_experiment.run()
