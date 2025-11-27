"""
Structural break detection methods for time series regression.
"""
import numpy as np
import pandas as pd
from typing import Protocol, Optional, Union


class Kernel(Protocol):
    """Protocol for kernel functions."""
    def __call__(self, u: np.ndarray) -> np.ndarray:
        ...
    

def estimate_break_mohr_ll(
    y: np.ndarray,
    X: np.ndarray,
    m_hat: np.ndarray,
    center_X: bool = True,
    standardize_X: bool = True,
    trim_frac: Optional[float] = 0.05,
) -> int:
    """
    Estimate single break index T1_hat using Mohr–Selk (2020) logic,
    using pre-computed pilot estimates for the conditional mean.

    Args:
        y (np.ndarray): Target variable, shape (T,).
        X (np.ndarray): Regressors, shape (T, d), time-ordered.
        m_hat (np.ndarray): Pre-computed pilot estimates of E[y|X], shape (T,).
        center_X (bool): Whether to center the columns of X.
        standardize_X (bool): Whether to standardize the columns of X.
        trim_frac (Optional[float]): Fraction to trim from each end of the
            sample when searching for the break date. If None, search over
            the full sample. Defaults to 0.05.

    Returns:
        int: Estimated 1-based break index T1_hat in {1, ..., T-1}.
    """
    T, d = X.shape

    # 2. Preprocessing X
    X_proc = X.copy()
    if center_X or standardize_X:
        mean = X_proc.mean(axis=0)
        std = X_proc.std(axis=0, ddof=1)
        if center_X:
            X_proc -= mean
        if standardize_X:
            # Avoid division by zero for constant features
            std[std == 0] = 1.0
            X_proc /= std

    # 3. Step 1 – Calculate residuals from pre-computed pilot
    U_hat = y - m_hat

    # 4. Step 2 – Truncation weights
    R_T = np.log(T) ** (1.0 / (d + 1.0))
    in_box = np.all((X_proc >= -R_T) & (X_proc <= R_T), axis=1)
    omega = in_box.astype(float)

    # 5. Step 3 – Define the discrete process
    contrib = (U_hat * omega) / T

    # Indicator matrix I[t, j] = 1{X_t <= X_j}
    # Using broadcasting for efficiency instead of a loop
    # X_proc[:, np.newaxis, :] -> (T, 1, d)
    # X_proc[np.newaxis, :, :] -> (1, T, d)
    # Comparison gives a (T, T, d) boolean array, then .all(axis=2) gives (T, T)
    I = np.all(X_proc[:, np.newaxis, :] <= X_proc[np.newaxis, :, :], axis=2)

    # 6. Step 4 – Sequential update of the process and KS functional
    S = np.zeros(T)  # Stores S_j(k) at current k
    D = np.zeros(T)  # D[k] = D_T(s_{k+1})

    for k in range(T):
        # Add contribution of time t=k (0-indexed)
        # I[k, :] is the row for X_k, indicating which X_j are >= X_k
        S += contrib[k] * I[k, :]
        D[k] = np.max(np.abs(S))

    # 7. Step 5 – Extract the break date
    if trim_frac is not None and trim_frac > 0:
        lo = int(np.floor(T * trim_frac))
        # The search space for k is up to T-2 for a break at T-1
        hi = int(np.ceil(T * (1 - trim_frac)))
        if hi >= T:
            hi = T - 1

        search_slice = slice(lo, hi)
        # If lo >= hi, search_slice will be empty, and argmax will raise an error.
        # We handle this by checking if the slice has any elements.
        if search_slice.start < search_slice.stop:
            k_star_local = int(np.argmax(D[search_slice]))
            k_star = lo + k_star_local
        else:
            # Fallback to full search if trim interval is invalid or empty
            k_star = int(np.argmax(D[:-1])) if T > 1 else 0
    else:
        # Search over the whole sample, excluding the last point
        k_star = int(np.argmax(D[:-1])) if T > 1 else 0

    # Convert 0-based index k_star to 1-based time index T1_hat
    T1_hat = k_star + 1

    return T1_hat


"""
Example script to run the Mohr-Selk style break date estimation.

This script loads the cocoa dataset, defines a wrapper for the local linear
regressor to make it compatible with the break detection function, and then
estimates the single structural break date on the training portion of the data.
"""

import numpy as np
import pandas as pd

from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
)
from cocoa.models.cocoa_data import CocoaDataset
from cocoa.models.np_kernels import GaussianKernel
from cocoa.models.np_engines import LocalPolynomialEngine
from cocoa.models.np_regime import NPRegimeModel
from cocoa.models.mfv_CV import MFVValidator
from cocoa.models.bandwidth import create_precentered_grid


def mfv_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def _create_mfv_splits(T: int, Q: int):
    """
    Create indices for Modified Forward-looking Validation (MFV).

    Splits data of length T into Q blocks. For each block q in {1,...,Q},
    the validation set is block q and the training set is all preceding
    blocks {1,...,q-1}. The first block is used for validation only, with
    an empty training set initially, which is often handled by the model logic.
    """
    if not 1 <= Q <= T:
        raise ValueError("Q must be between 1 and T.")

    block_size = T // Q
    indices = np.arange(T)
    splits = []

    for q in range(1, Q + 1):
        val_start = (q - 1) * block_size
        val_end = q * block_size if q < Q else T
        train_indices = indices[:val_start]
        val_indices = indices[val_start:val_end]
        splits.append((train_indices, val_indices))
    return splits

if __name__ == "__main__":
    # 1. Load the dataset and get the training split
    print("Loading and splitting data...")
    dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
    )
    split = dataset.split_oos_by_date(OOS_START_DATE)
    X_train_np = split.X_train.values
    y_train_np = split.y_train.values.flatten()
    
    T, d = X_train_np.shape

    # 2. Find the optimal pilot bandwidth `h` using MFV cross-validation.
    print("\nFinding optimal pilot bandwidth using MFV cross-validation...")
    kernel = GaussianKernel()
    ll_engine = LocalPolynomialEngine(order=1)
    
    # Define a grid of bandwidths to search.
    bandwidth_grid = create_precentered_grid(T=T, d=d)

    Q = 5  # Number of MFV blocks
    mfv_splits = _create_mfv_splits(T, Q)
    
    scores = []
    for h in bandwidth_grid:
        print(f"  Testing h = {h:.4f}...")
        preds = np.zeros_like(y_train_np)
        for train_indices, val_indices in mfv_splits:
            X_mfv_train, y_mfv_train = X_train_np[train_indices], y_train_np[train_indices]
            if len(X_mfv_train) == 0:
                # Skip first fold where training set is empty
                continue
            X_mfv_val = X_train_np[val_indices]

            # The engine expects pandas inputs
            X_mfv_train_df = pd.DataFrame(X_mfv_train)
            y_mfv_train_s = pd.Series(y_mfv_train)
            X_mfv_val_df = pd.DataFrame(X_mfv_val)

            preds[val_indices] = ll_engine.fit(X_mfv_train_df, y_mfv_train_s, X_mfv_val_df, h, kernel)
        
        score = mfv_mse(y_train_np, preds)
        scores.append(score)

    best_idx = np.argmin(scores)
    pilot_bandwidth = bandwidth_grid[best_idx]
    best_score = scores[best_idx]
    print(f"\nFound best pilot bandwidth: h = {pilot_bandwidth:.4f} (MFV MSE: {best_score:.4f})")

    # 3. Get pilot estimates `m_hat` using the full training data and best bandwidth
    print(f"\nCalculating pilot estimates with optimal bandwidth h={pilot_bandwidth:.4f}...")
    m_hat = ll_engine.fit(split.X_train, split.y_train, split.X_train, pilot_bandwidth, kernel)

    # 4. Run the break date estimation with the pilot estimates
    print(f"\nEstimating break date with optimal pilot bandwidth h={pilot_bandwidth:.4f}...")
    T1_hat = estimate_break_mohr_ll(
        y=y_train_np,
        X=X_train_np,
        m_hat=m_hat,
    )

    print(f"\nEstimated break date T1_hat (1-based index): {T1_hat}")
# ============================================================