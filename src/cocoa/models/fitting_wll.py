from functools import partial
import numpy as np

from ..experiments.runner import ExperimentRunner
from . import (
    NPRegimeModel,
    WLLModel,
    GaussianKernel,
    LocalPolynomialEngine,
    CocoaDataset,
    MFVValidator,
)
from .assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    Break_ID_ONE_BASED,
)
from .bandwidth import create_precentered_grid


if __name__ == "__main__":
    # --- 1. Setup Dataset and Identify Break Index ---
    # Load the full dataset to find the relative index of the structural break
    # This is crucial for splitting the training data correctly inside the WLLModel
    full_dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
    )
    # We need the 0-based index relative to the start of the *training set*
    break_date = full_dataset.get_date_from_1_based_index(Break_ID_ONE_BASED)
    
    # The runner will trim data to start at index 0, so the break index is just its original position
    # minus one (for 0-based indexing).
    break_idx_in_train = Break_ID_ONE_BASED - 1

    # --- 2. Define Core NP Components ---
    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=1)  # Local Linear
    NPModelPartial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)

    # --- 3. Tune Bandwidths for Pre and Post Models Separately ---
    # This is a simplification. A full implementation might nest this inside the
    # main gamma search, but that is computationally very expensive.
    
    # Create a temporary runner to access its split data for tuning
    temp_runner = ExperimentRunner(
        model_name="TEMP", model_class=NPModelPartial, feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL, data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE, save_results=False
    )
    X_train_full, y_train_full = temp_runner.split.X_train, temp_runner.split.y_train

    # Tune bandwidth for POST-break model
    X_post, y_post = X_train_full.iloc[break_idx_in_train:], y_train_full.iloc[break_idx_in_train:]
    post_grid = {"bandwidth": create_precentered_grid(T=len(X_post), d=X_post.shape[1])}
    mfv = MFVValidator(Q=4)
    best_post_params, _, _ = mfv.grid_search(NPModelPartial, X_post, y_post, [post_grid])
    print(f"Found best bandwidth for Post-Break Model: {best_post_params['bandwidth']:.4f}")

    # Tune bandwidth for PRE-break model
    X_pre, y_pre = X_train_full.iloc[:break_idx_in_train], y_train_full.iloc[:break_idx_in_train]
    pre_grid = {"bandwidth": create_precentered_grid(T=len(X_pre), d=X_pre.shape[1])}
    best_pre_params, _, _ = mfv.grid_search(NPModelPartial, X_pre, y_pre, [pre_grid])
    print(f"Found best bandwidth for Pre-Break Model: {best_pre_params['bandwidth']:.4f}")

    # --- 4. Create a WLL Partial Constructor for the Main Experiment ---
    # The ExperimentRunner will tune over the 'gamma' hyperparameter.
    WLLModelPartial = partial(
        WLLModel,
        pre_model_class=NPModelPartial,
        post_model_class=NPModelPartial,
        pre_model_params=best_pre_params,
        post_model_params=best_post_params,
        break_idx=break_idx_in_train,
    )

    # --- 5. Manually override the runner's param_grid logic ---
    # The ExperimentRunner is designed to create the grid itself. For WLL, we
    # need to provide a grid of 'gamma' values. We will create a custom runner
    # class or modify the existing one to accept a pre-made grid.
    # For now, let's assume we can set it directly.
    
    # --- 6. Configure and Run the WLL Experiment ---
    wll_experiment = ExperimentRunner(
        model_name="WLL_NP",
        model_class=WLLModelPartial, # This partial now includes everything but gamma
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        kernel_name=kernel.__class__.__name__,
        poly_order=engine.order,
        save_results=True,
    )
    
    # Manually set the parameter grid for the runner to be for 'gamma'
    wll_experiment.param_grid = {"gamma": np.linspace(0, 1, 11)} # e.g., [0.0, 0.1, ..., 1.0]

    wll_experiment.run()

    print("\n\nCompleted WLL experiment.")