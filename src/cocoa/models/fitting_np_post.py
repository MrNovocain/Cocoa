from functools import partial

from ..experiments.runner import ExperimentRunner
from . import NPRegimeModel, GaussianKernel, LocalPolynomialEngine
from .assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,  # Using same features as RF/XGB for comparability
    DEFAULT_TARGET_COL,
    # BREAK_DATE,
    Break_ID_ONE_BASED,
)

# Define a parameter grid for the bandwidth 'h'.
# These values are just a starting point; a wider or finer grid may be needed.
NP_PARAM_GRID = {
    "bandwidth": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
}

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

    # 3. Configure and run the Non-Parametric experiment
    np_experiment = ExperimentRunner(
        model_name="NP_LL_Post",
        model_class=NPModelPartial,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        param_grid=NP_PARAM_GRID,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        kernel_name=kernel.__class__.__name__,
        poly_order=engine.order,
        sample_start_index=Break_ID_ONE_BASED,
    )
    np_experiment.run()
    