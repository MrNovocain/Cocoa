
import pandas as pd
import numpy as np
from functools import partial

# Model and data imports
from cocoa.models import (
    NPRegimeModel,
    NPConvexCombinationModel,
    GaussianKernel,
    LocalPolynomialEngine,
    CocoaDataset,
    MFVValidator,
    TrainTestSplit,
)
from cocoa.models.bandwidth import create_precentered_grid

# Asset and config imports
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    Break_ID_ONE_BASED,
)


def run_np_combo_cv():
    """
    Runs the full cross-validation pipeline for the NPConvexCombinationModel.
    """
    print("--- Starting NP Convex Combination Model CV ---")

    # --- 1. Configuration & Data Loading ---
    Q_folds = 5
    
    dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
    )
    
    post_start_index = Break_ID_ONE_BASED - 1  # Convert to 0-based index
    
    # --- 2. Data Splitting (Train/Test) ---
    oos_start_date = pd.to_datetime(OOS_START_DATE)
    mask_test = dataset.df["date"] >= oos_start_date
    test_start_idx = dataset.df.index[mask_test][0]

    X_train_full = dataset.df.loc[:test_start_idx - 1, DEFAULT_FEATURE_COLS].reset_index(drop=True)
    y_train_full = dataset.df.loc[:test_start_idx - 1, DEFAULT_TARGET_COL].reset_index(drop=True)

    X_test = dataset.df.loc[test_start_idx:, DEFAULT_FEATURE_COLS].reset_index(drop=True)
    y_test = dataset.df.loc[test_start_idx:, DEFAULT_TARGET_COL].reset_index(drop=True)

    print(f"Full training size: {len(X_train_full)}, Test size: {len(X_test)}")
    
    # --- 3. Bandwidth Selection for Sub-Models ---
    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=1)
    validator = MFVValidator(Q=Q_folds)
    
    # Partial constructor for the base NP model
    NPModelPartial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)

    # 3a. Find best bandwidth for the "full" model
    print("\n--- Tuning bandwidth for FULL model ---")
    T_full, d_full = X_train_full.shape
    bw_grid_full = [{"bandwidth": h} for h in create_precentered_grid(T=T_full, d=d_full)]
    best_params_full, _, _ = validator.grid_search(
        model_class=NPModelPartial,
        X_train=X_train_full,
        y_train=y_train_full,
        param_grid=bw_grid_full,
        verbose=False
    )
    h_full = best_params_full['bandwidth']
    print(f"Best bandwidth for FULL model: {h_full:.4f}")

    # 3b. Find best bandwidth for the "post" model
    print("\n--- Tuning bandwidth for POST model ---")
    X_train_post = X_train_full.iloc[post_start_index:]
    y_train_post = y_train_full.iloc[post_start_index:]
    T_post, d_post = X_train_post.shape
    
    bw_grid_post = [{"bandwidth": h} for h in create_precentered_grid(T=T_post, d=d_post)]
    best_params_post, _, _ = validator.grid_search(
        model_class=NPModelPartial,
        X_train=X_train_post,
        y_train=y_train_post,
        param_grid=bw_grid_post,
        verbose=False
    )
    h_post = best_params_post['bandwidth']
    print(f"Best bandwidth for POST model: {h_post:.4f}")

    # --- 4. Gamma Selection for Convex Combination Model ---
    print("\n--- Tuning gamma for Convex Combination model ---")
    
    # Create the two sub-models with their optimal bandwidths
    model_full = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=h_full)
    model_post = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=h_post)
    
    # Create partial constructor for the combination model
    ComboModelPartial = partial(
        NPConvexCombinationModel,
        model_full=model_full,
        model_post=model_post,
        post_start_index=post_start_index,
    )
    
    # Create gamma grid and run CV
    gamma_grid = [{"gamma": g} for g in np.linspace(0, 1, 20)]
    best_params_gamma, best_score_gamma, _ = validator.grid_search(
        model_class=ComboModelPartial,
        X_train=X_train_full,
        y_train=y_train_full,
        param_grid=gamma_grid,
    )
    best_gamma = best_params_gamma['gamma']
    print(f"Best gamma: {best_gamma:.2f} (MFV MSE: {best_score_gamma:.6f})")

    # --- 5. Final Evaluation ---
    print("\n--- Evaluating final model on test set ---")
    final_model = NPConvexCombinationModel(
        model_full=model_full,
        model_post=model_post,
        post_start_index=post_start_index,
        gamma=best_gamma,
    )
    final_model.fit(X_train_full, y_train_full)
    y_pred_test = final_model.predict(X_test)
    
    mse_test = np.mean((y_test - y_pred_test) ** 2)
    print(f"Final Test MSE: {mse_test:.6f}")


if __name__ == "__main__":
    run_np_combo_cv()
