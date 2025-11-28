"""Utilities to run MFV gamma sweeps for a fixed break date (tau)."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, List

from cocoa.models import CocoaDataset, NPRegimeModel, GaussianKernel, LocalPolynomialEngine
from cocoa.models.assets import (
    BREAK_ID_ONE_BASED,
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    Q_VALUE,
)


@dataclass
class TauSliceResult:
    """Container for the MFV gamma sweep at a fixed break date."""

    tau: pd.Timestamp
    gamma_grid: List[float]
    gamma_losses: List[float]
    best_gamma: float
    best_loss: float
    pre_bandwidth: float
    post_bandwidth: float
    block_size: int


def run_mfv_gamma_slice(
    tau: str | pd.Timestamp,
    *,
    gamma_grid: Iterable[float] | None = None,
    pre_bandwidth: float,
    post_bandwidth: float,
    feature_cols: List[str] | None = None,
    target_col: str = DEFAULT_TARGET_COL,
    data_path: str = PROCESSED_DATA_PATH,
    oos_start_date: str | pd.Timestamp = OOS_START_DATE,
    Q: int = Q_VALUE,
    poly_order: int = 1,
) -> TauSliceResult:
    """
    Run a one-τ MFV slice test for the convex-combination NP model.

    For a fixed structural break date ``tau`` and fixed pre/post bandwidths,
    this function sweeps ``gamma`` values without re-fitting the underlying
    non-parametric models for every grid point.

    The MFV for gamma selection is performed ONLY on post-break data, to align
    with the logic in the NPComboExperimentRunner.

    Args:
        tau: Break date used to define pre/post regimes.
        gamma_grid: Sequence of gamma values to evaluate. Defaults to 21 points
            in [0, 1].
        pre_bandwidth: Bandwidth for the pre-break NP model.
        post_bandwidth: Bandwidth for the post-break NP model.
        feature_cols: Feature columns to use. Defaults to ``DEFAULT_FEATURE_COLS``.
        target_col: Target column name.
        data_path: Path to the processed cocoa dataset.
        oos_start_date: Date that splits train/OOS sets.
        Q: Number of MFV folds.
        poly_order: Local polynomial order (0=Nadaraya-Watson, 1=LL, 2=LQ).

    Returns:
        TauSliceResult with the gamma grid, losses, and the minimizing gamma.
    """

    gamma_values = list(gamma_grid) if gamma_grid is not None else np.linspace(0.0, 1.0, 21).tolist()
    feature_cols = feature_cols if feature_cols is not None else DEFAULT_FEATURE_COLS
    tau_ts = pd.to_datetime(tau)

    dataset = CocoaDataset(
        csv_path=data_path,
        feature_cols=feature_cols,
        target_col=target_col,
    )
    split = dataset.split_oos_by_date(oos_start_date)

    X_train = split.X_train
    y_train = split.y_train
    train_dates = dataset.df["date"].iloc[: split.T_train]

    # --- Isolate post-break data for MFV validation ---
    post_break_mask = train_dates > tau_ts
    X_train_post = X_train[post_break_mask].reset_index(drop=True)
    y_train_post = y_train[post_break_mask].reset_index(drop=True)
    post_train_dates = train_dates[post_break_mask].reset_index(drop=True)
    
    T_post = len(X_train_post)
    if T_post == 0:
        raise ValueError(f"No training data available after the specified break date {tau_ts.date()}.")

    block_size = T_post // (Q + 1)
    if block_size == 0:
        raise ValueError("Post-break training data too short for the requested number of MFV folds.")
    cv_start_post = T_post - block_size * Q

    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=poly_order)

    y_val_all: list[float] = []
    y_pre_all: list[float] = []
    y_post_all: list[float] = []

    for q in range(Q):
        # --- Validation set is a slice of the POST-BREAK data ---
        val_start_idx_post = cv_start_post + q * block_size
        val_end_idx_post = val_start_idx_post + block_size

        X_val = X_train_post.iloc[val_start_idx_post:val_end_idx_post]
        y_val = y_train_post.iloc[val_start_idx_post:val_end_idx_post]

        # --- Training set is ALL data prior to the validation fold start ---
        val_start_date = post_train_dates.iloc[val_start_idx_post]
        train_fold_mask = train_dates < val_start_date
        
        X_train_fold = X_train[train_fold_mask]
        y_train_fold = y_train[train_fold_mask]
        fold_dates = train_dates[train_fold_mask]

        pre_mask = fold_dates <= tau_ts
        post_mask = fold_dates > tau_ts

        # Fit pre-model on its portion of the fold's training data
        pre_model = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=pre_bandwidth)
        pre_model.fit(X_train_fold.loc[pre_mask], y_train_fold.loc[pre_mask])
        y_pre_pred = pre_model.predict(X_val)

        # Fit post-model on its portion, if available
        post_model = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=post_bandwidth)
        if post_mask.any():
            post_model.fit(X_train_fold.loc[post_mask], y_train_fold.loc[post_mask])
            y_post_pred = post_model.predict(X_val)
        else:
            # If no post-data in this training fold, predict NaNs
            y_post_pred = np.full(len(X_val), np.nan)

        y_val_all.extend(y_val.to_list())
        y_pre_all.extend(y_pre_pred.tolist())
        y_post_all.extend(y_post_pred.tolist())

    y_val_arr = np.asarray(y_val_all)
    y_pre_arr = np.asarray(y_pre_all)
    y_post_arr = np.asarray(y_post_all)

    # Filter out points where post-model could not make a prediction
    valid_mask = ~np.isnan(y_post_arr)
    y_val_filt = y_val_arr[valid_mask]
    y_pre_filt = y_pre_arr[valid_mask]
    y_post_filt = y_post_arr[valid_mask]

    if len(y_val_filt) == 0:
        raise ValueError("Could not make any valid post-break predictions during MFV. Check data and break date.")

    gamma_losses: List[float] = []
    for gamma in gamma_values:
        y_hat = gamma * y_pre_filt + (1.0 - gamma) * y_post_filt
        loss = float(np.mean((y_val_filt - y_hat) ** 2))
        gamma_losses.append(loss)

    best_idx = int(np.argmin(gamma_losses))
    best_gamma = float(gamma_values[best_idx])
    best_loss = float(gamma_losses[best_idx])

    return TauSliceResult(
        tau=tau_ts,
        gamma_grid=gamma_values,
        gamma_losses=gamma_losses,
        best_gamma=best_gamma,
        best_loss=best_loss,
        pre_bandwidth=pre_bandwidth,
        post_bandwidth=post_bandwidth,
        block_size=block_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MFV gamma slice test for a given break date.")
    parser.add_argument(
        "--tau",
        type=str,
        default=None,
        help="Break date in YYYY-MM-DD format. If not provided, a default value is used.",
    )
    args = parser.parse_args()

    if args.tau:
        tau_to_test = pd.to_datetime(args.tau)
        print(f"Using provided break date tau: {tau_to_test.strftime('%Y-%m-%d')}")
    else:
        default_dataset = CocoaDataset(
            csv_path=PROCESSED_DATA_PATH,
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
        )
        tau_to_test = default_dataset.get_date_from_1_based_index(BREAK_ID_ONE_BASED)

    result = run_mfv_gamma_slice(
        tau=tau_to_test,
        pre_bandwidth=1.0,
        post_bandwidth=1.0,
    )

    # Check for convexity using second-order differences.
    # A small tolerance is added to account for floating point inaccuracies.
    second_diff = np.diff(result.gamma_losses, n=2)
    is_convex = np.all(second_diff >= -1e-9)
    print(result)
    print(
        f"Test for tau={result.tau.strftime('%Y-%m-%d')}: "
        f"Convexity check: {'Pass' if is_convex else 'Fail'} ({is_convex})"
    )
