"""Utilities to run MFV gamma sweeps for a fixed break date (tau)."""

from __future__ import annotations

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

    block_size = split.T_train // (Q + 1)
    if block_size == 0:
        raise ValueError("Training data too short for the requested number of MFV folds.")
    cv_start = split.T_train - block_size * Q

    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=poly_order)

    y_val_all: list[float] = []
    y_pre_all: list[float] = []
    y_post_all: list[float] = []

    for q in range(Q):
        val_start = cv_start + q * block_size
        val_end = val_start + block_size

        train_end = val_start
        X_train_fold = X_train.iloc[:train_end]
        y_train_fold = y_train.iloc[:train_end]
        fold_dates = train_dates.iloc[:train_end]

        X_val = X_train.iloc[val_start:val_end]
        y_val = y_train.iloc[val_start:val_end]

        pre_mask = fold_dates <= tau_ts
        post_mask = fold_dates > tau_ts

        pre_model = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=pre_bandwidth)
        pre_model.fit(X_train_fold.loc[pre_mask], y_train_fold.loc[pre_mask])
        y_pre_pred = pre_model.predict(X_val)

        post_model = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=post_bandwidth)
        post_model.fit(X_train_fold.loc[post_mask], y_train_fold.loc[post_mask])
        y_post_pred = post_model.predict(X_val)

        y_val_all.extend(y_val.to_list())
        y_pre_all.extend(y_pre_pred.tolist())
        y_post_all.extend(y_post_pred.tolist())

    y_val_arr = np.asarray(y_val_all)
    y_pre_arr = np.asarray(y_pre_all)
    y_post_arr = np.asarray(y_post_all)

    gamma_losses: List[float] = []
    for gamma in gamma_values:
        y_hat = gamma * y_pre_arr + (1.0 - gamma) * y_post_arr
        loss = float(np.mean((y_val_arr - y_hat) ** 2))
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
    default_dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
    )
    tau_default = default_dataset.get_date_from_1_based_index(BREAK_ID_ONE_BASED)

    result = run_mfv_gamma_slice(
        tau=tau_default,
        pre_bandwidth=1.0,
        post_bandwidth=1.0,
    )
    print(result)
