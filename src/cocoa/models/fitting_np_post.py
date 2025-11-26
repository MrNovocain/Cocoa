from functools import partial

from ..experiments.runner import ExperimentRunner
from . import NPRegimeModel, GaussianKernel, LocalPolynomialEngine, CocoaDataset
from .assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,  # Using same features as RF/XGB for comparability
    DEFAULT_TARGET_COL,
    Break_ID_ONE_BASED,
)
from .bandwidth import create_precentered_grid

# To create the pre-centered bandwidth grid, we need the dimensions of the
# training data (T, d). We load the dataset here to get those values before
# the ExperimentRunner is instantiated. While this means loading data twice,
# it's a simple approach that avoids modifying the core runner logic.
dataset = CocoaDataset(
    csv_path=PROCESSED_DATA_PATH,
    feature_cols=DEFAULT_FEATURE_COLS,
    target_col=DEFAULT_TARGET_COL,
)
split = dataset.split_oos_by_date(OOS_START_DATE)
dataset.trim_data_by_start_date(Break_ID_ONE_BASED)
T_train, d_train = split.X_train.shape

# Define a parameter grid for the bandwidth 'h'.
NP_PARAM_GRID = {
    "bandwidth": create_precentered_grid(T=T_train, d=d_train),
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
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        kernel_name=kernel.__class__.__name__,
        poly_order=engine.order,
        sample_start_index=Break_ID_ONE_BASED,
    )
    np_experiment.run()
    