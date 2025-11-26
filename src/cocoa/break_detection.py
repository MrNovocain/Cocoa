"""
Structural break detection methods for time series regression.
"""
from functools import partial
import numpy as np
import pandas as pd
from typing import Protocol, Optional


class Kernel(Protocol):
    """Protocol for kernel functions."""
    def __call__(self, u: np.ndarray) -> np.ndarray:
        ...
    

class LocalLinearRegressor(Protocol):
    """
    Protocol for a Local Linear Regressor, defining the expected interface.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocalLinearRegressor":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def estimate_break_mohr_ll(
    y: np.ndarray,
    X: np.ndarray,
    pilot_estimator: LocalLinearRegressor,
    center_X: bool = True,
    standardize_X: bool = True,
    trim_frac: Optional[float] = 0.05,
) -> int:
    """
    Estimate single break index T1_hat using Mohr–Selk (2020) logic,
    but with a global Local Linear estimator as pilot.

    Args:
        y (np.ndarray): Target variable, shape (T,).
        X (np.ndarray): Regressors, shape (T, d), time-ordered.
        pilot_estimator (LocalLinearRegressor): An instantiated local linear
            regressor to be used as the pilot estimator.
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

    # 3. Step 1 – Fit global local linear and residuals
    pilot_estimator.fit(X_proc, y)
    m_hat = pilot_estimator.predict(X_proc)
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
from cocoa.models.base_model import BaseModel
from cocoa.models.bandwidth import create_precentered_grid


class LocalLinearWrapper(BaseModel, LocalLinearRegressor):
    """
    A wrapper to make LocalPolynomialEngine conform to the LocalLinearRegressor
    protocol and be compatible with MFVValidator. It holds the engine,
    kernel, and bandwidth.
    """
    def __init__(self, kernel: Kernel, bandwidth: float):
        self.engine = LocalPolynomialEngine(order=1)
        self.kernel = kernel
        self.bandwidth = bandwidth
        self._X_train = None
        self._y_train = None

    def fit(self, X, y, X_val=None) -> "LocalLinearWrapper":
        # MFV validator passes pandas objects, so convert to numpy before storing.
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs X and y must be numpy arrays or pandas objects.")

        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Model must be fitted before prediction.")
        
        # The MFV validator might pass a pandas object for X.
        if hasattr(X, 'values'):
            X = X.values

        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array or pandas object.")

        # The underlying engine expects pandas, so we convert here.
        X_train_df = pd.DataFrame(self._X_train)
        y_train_s = pd.Series(self._y_train)
        X_eval = pd.DataFrame(X)
        return self.engine.fit(X_train_df, y_train_s, X_eval, self.bandwidth, self.kernel)


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
    y_train_np = split.y_train.values
    
    T, d = X_train_np.shape

    # 2. Find the optimal pilot bandwidth `h` using MFV cross-validation.
    print("\nFinding optimal pilot bandwidth using MFV cross-validation...")
    kernel = GaussianKernel()
    
    # Define a grid of bandwidths to search.
    bandwidth_grid = create_precentered_grid(T=T, d=d)
    param_grid = [
        {"kernel": kernel, "bandwidth": h} for h in bandwidth_grid
    ]

    validator = MFVValidator(Q=5)
    
    best_params, best_score, _ = validator.grid_search(
        model_class=LocalLinearWrapper,
        X_train=X_train_np,
        y_train=y_train_np,
        param_grid=param_grid,
        verbose=True
    )

    pilot_bandwidth = best_params["bandwidth"]
    print(f"\nFound best pilot bandwidth: h = {pilot_bandwidth:.4f} (MFV MSE: {best_score:.4f})")

    # 3. Instantiate the pilot estimator with the best bandwidth
    pilot_ll = LocalLinearWrapper(kernel=kernel, bandwidth=pilot_bandwidth)

    # 4. Run the break date estimation
    print(f"\nEstimating break date with optimal pilot bandwidth h={pilot_bandwidth:.4f}...")
    T1_hat = estimate_break_mohr_ll(
        y=y_train_np,
        X=X_train_np,
        pilot_estimator=pilot_ll,
    )

    print(f"\nEstimated break date T1_hat (1-based index): {T1_hat}")
###Estimating break date with pilot bandwidth h=1.5...
                                                                                                                                                                                   
###Estimated break date T1_hat (1-based index): 4914