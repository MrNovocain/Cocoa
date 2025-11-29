"""
Runs non-parametric model experiments for various combinations of features.

This script systematically tests subsets of features, ensuring that 'log_price_lagt'
is always included. For each combination, it finds the optimal bandwidth using
MFV cross-validation and runs the full experiment, saving the results.
"""
import itertools
from functools import partial
import pandas as pd

from .runner import ExperimentRunner
from ..models import NPRegimeModel, GaussianKernel, LocalPolynomialEngine, CocoaDataset
from ..models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    XGB_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from ..models.bandwidth import create_precentered_grid


def get_feature_combinations():
    """
    Generates all combinations of features from XGB_FEATURE_COLS that include
    'log_price_lagt'.
    """
    base_feature = "log_price_lagt"
    other_features = [feat for feat in XGB_FEATURE_COLS if feat != base_feature]
    
    all_combinations = []
    for i in range(len(other_features) + 1):
        for combo in itertools.combinations(other_features, i):
            all_combinations.append([base_feature] + list(combo))
            
    return all_combinations


if __name__ == "__main__":
    feature_sets = get_feature_combinations()
    results = []

    for i, features in enumerate(feature_sets):
        print("-" * 80)
        print(f"Running experiment {i+1}/{len(feature_sets)} with {len(features)} features:")
        print(f"Features: {features}")
        print("-" * 80)

        # 1. Load data and get dimensions for bandwidth grid for the current feature set
        dataset = CocoaDataset(
            csv_path=PROCESSED_DATA_PATH,
            feature_cols=features,
            target_col=DEFAULT_TARGET_COL,
        )
        split = dataset.split_oos_by_date(OOS_START_DATE)
        T_train, d_train = split.X_train.shape

        # 2. Define parameter grid for the current experiment
        param_grid = {
            "bandwidth": create_precentered_grid(T=T_train, d=d_train),
        }

        # 3. Define model components and create partial constructor
        kernel = GaussianKernel()
        engine = LocalPolynomialEngine(order=1)  # Local Linear
        NPModelPartial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)

        # 4. Configure and run the experiment
        model_name = f"NP_LL_Combo_{i+1}_d{d_train}"
        np_experiment = ExperimentRunner(
            model_name=model_name,
            model_class=NPModelPartial,
            feature_cols=features,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_START_DATE,
            kernel_name=kernel.__class__.__name__,
            poly_order=engine.order,
        )
        np_experiment.run()

    print("\n\nAll combinatorial experiments completed.")