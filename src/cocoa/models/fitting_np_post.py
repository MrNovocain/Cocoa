from functools import partial

from cocoa.experiments.runner import ExperimentRunner
from cocoa.models import NPRegimeModel, GaussianKernel, LocalPolynomialEngine, CocoaDataset
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,  # Using same features as RF/XGB for comparability
    DEFAULT_TARGET_COL,
    BREAK_ID_ONE_BASED,
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
    dataset= CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
    )
    index = dataset.get_1_based_index_from_date("2019-10-01") #From grid search
    
    # 3. Configure and run the Non-Parametric experiment
    np_experiment = ExperimentRunner(
        model_name="NP_LL_Post",
        model_class=NPModelPartial,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        kernel_name=kernel.__class__.__name__,
        poly_order=engine.order,
        sample_start_index=index,
        save_results=True,
    )
    np_experiment.run()
    