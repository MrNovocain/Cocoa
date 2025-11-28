"""
Bandwidth selection methodologies for non-parametric models.
"""
from typing import List, Optional, Union
import numpy as np


def create_precentered_grid(
    T: int,
    d: int,
    C: float = 1.0,
    multipliers: Optional[Union[List[float], np.ndarray]] = None,
) -> np.ndarray:
    """
    Generates a pre-centered bandwidth grid for non-parametric regression.

    The method follows a rule-of-thumb baseline for the bandwidth, adjusted by
    a multiplicative grid. The baseline bandwidth is calculated as:
    h_0 = C * T^(-1 / (d + 4))

    This baseline is then multiplied by a set of factors to create the final
    grid of candidates for cross-validation.

    Args:
        T (int): The sample size (number of observations).
        d (int): The effective dimension (number of continuous regressors).
        C (float): A constant of order 1 for the baseline calculation.
                   Defaults to 1.0.
        multipliers (Optional[Union[List[float], np.ndarray]]):
            A list or array of dimensionless adjustment factors. If None, a
            default grid ([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]) is used.

    Returns:
        np.ndarray: An array of bandwidth candidates.
    """
    if T <= 0:
        raise ValueError("Sample size T must be positive.")
    if d < 0:
        raise ValueError("Dimension d cannot be negative.")
    print(f"Creating bandwidth grid with T={T}, d={d}")
    # 1. Calculate the theoretical / rule-of-thumb baseline bandwidth (h_0)
    # For local constant/linear regression, the classical optimal rate is T^(-1/(d+4)).
    h0 = C * (T ** (-1.0 / (d + 4.0)))

    # 2. Define the multiplicative grid for the adjustment factor (m)
    if multipliers is None:
        logs = np.linspace(-1.0, 1.0, num=10)
        mults = np.power(10.0, logs)
    else:
        mults = np.asarray(multipliers)
        if not np.all(mults > 0):
            raise ValueError("Multipliers must be positive.")

    # 3. Create the final bandwidth grid
    bandwidth_grid = h0 * mults
    print(f"Generated bandwidth grid: {bandwidth_grid}")
    return bandwidth_grid
