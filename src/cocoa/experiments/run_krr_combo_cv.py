from cocoa.experiments.runner import ConvexComboExperimentRunner
from cocoa.models import (
    KRRModel,
)
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    KRR_PARAM_GRID,
    BREAK_ID_ONE_BASED,
)


if __name__ == "__main__":
    runner = ConvexComboExperimentRunner(
        combo_type='ML',
        model_name="ML_KRR_Combo",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        sample_start_index=BREAK_ID_ONE_BASED,
        sub_model_class=KRRModel,
        sub_model_param_grid=KRR_PARAM_GRID,
        save_results=True,
    )
    run_results = runner.run()