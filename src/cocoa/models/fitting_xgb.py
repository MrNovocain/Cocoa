from ..experiments.runner import ExperimentRunner
from ..models import XGBModel
from ..models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    XGB_FEATURE_COLS,
    XGB_TARGET_COL,
    XGB_PARAM_GRID,
    BREAK_DATE
)
 
if __name__ == "__main__":
    # Configure and run the XGBoost experiment
    xgb_experiment = ExperimentRunner(
        model_name="XGB",
        model_class=XGBModel,
        feature_cols=XGB_FEATURE_COLS,
        target_col=XGB_TARGET_COL,
        param_grid=XGB_PARAM_GRID,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        sample_start_date= BREAK_DATE
    )
    xgb_experiment.run()