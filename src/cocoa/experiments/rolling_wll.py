"""
Skeleton for the rolling WLL vs ML experiment.

This is a non-functional placeholder that mirrors the design in
docs/rolling_wll_experiment_plan.md. Implementations should wire in the
existing runners and Mohr detector without changing signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
    Q_VALUE,
)
from cocoa.models.bandwidth import create_precentered_grid
from cocoa.models.np_engines import LocalPolynomialEngine
from cocoa.models.np_kernels import GaussianKernel
from cocoa.models.np_regime import NPRegimeModel
from cocoa.models.mfv_CV import MFVValidator
from cocoa.experiments.runner import ExperimentRunner, ConvexComboExperimentRunner
from cocoa.models import CocoaDataset, RFModel, XGBModel
from cocoa.experiments.break_detection import estimate_break_mohr_ll, MohrRunner


@dataclass
class RollingResultRow:
    origin_idx: int
    origin_date: pd.Timestamp
    detected_break_idx: int
    detected_break_date: pd.Timestamp
    baseline: str
    msfe_wll: float
    msfe_base: float
    pi: float
    hit: bool


def build_descending_origins(start_idx: int, end_idx: int, step: int) -> List[int]:
    """
    Return descending origin indices (recent -> past), stepping by `step`.
    Should include a final earliest-admissible origin when the remaining span < step.
    Rolling backward matters: by starting from the most recent origin and moving
    earlier, we avoid accidentally trimming away the true convergent break date
    that might have been detected in a later (more recent) iteration.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    if start_idx < end_idx:
        raise ValueError("start_idx should be >= end_idx for descending traversal.")

    origins: List[int] = []
    current = start_idx
    while current >= end_idx:
        origins.append(current)
        next_val = current - step
        if next_val < end_idx:
            # include the earliest admissible origin once when the remaining span < step
            if origins[-1] != end_idx:
                origins.append(end_idx)
            break
        current = next_val

    return origins


def build_candidate_grid_excluding(
    prior_breaks: set[int],
    dataset: CocoaDataset,
    origin_idx: int,
) -> pd.DataFrame:
    """
    Build the Mohr candidate grid for a given origin, excluding any indices
    in `prior_breaks`. If exclusion leaves no admissible candidates, return
    an empty DataFrame so the caller can halt.
    """
    raise NotImplementedError("build_candidate_grid_excluding is a placeholder.")


def run_mohr_break(
    dataset: CocoaDataset,
    feature_cols: Sequence[str],
    target_col: str,
    trim_frac: float | None = 0.05,
    q_folds: int = Q_VALUE,
) -> int:
    """
    Run Mohr–Selk break detection with an MFV-tuned pilot local linear fit.
    Returns the detected break index (0-based); pilot bandwidth is kept internal.
    """
    df = dataset.df
    X_np = df[feature_cols].to_numpy()
    y_np = df[target_col].to_numpy().flatten()

    T, d = X_np.shape
    bw_grid = create_precentered_grid(T=T, d=d)
    kernel = GaussianKernel()
    ll_engine = LocalPolynomialEngine(order=1)
    np_partial = partial(NPRegimeModel, kernel=kernel, local_engine=ll_engine)

    validator = MFVValidator(Q=q_folds)
    scores: list[float] = []
    for h in bw_grid:
        params = {"bandwidth": h}
        score = validator.score(
            model_class=np_partial,
            X_train=df[feature_cols],
            y_train=df[target_col],
            params=params,
        )
        scores.append(score)

    best_idx = int(np.argmin(scores))
    pilot_bw = bw_grid[best_idx]

    m_hat = ll_engine.fit(df[feature_cols], df[target_col], df[feature_cols], pilot_bw, kernel)
    T1_hat_1based = estimate_break_mohr_ll(
        y=y_np,
        X=X_np,
        m_hat=m_hat,
        trim_frac=trim_frac,
    )
    return int(T1_hat_1based - 1)




def plot_break_vs_origin(*, df: pd.DataFrame, outfile: Path, reverse_time: bool = True) -> None:
    """Placeholder for break path plotting with optional reverse-time axis and variance band."""
    raise NotImplementedError("plot_break_vs_origin is a placeholder.")


def plot_pi_vs_break(*, df: pd.DataFrame, outfile: Path) -> None:
    """Placeholder for PI vs break plot with shading where PI > 0."""
    raise NotImplementedError("plot_pi_vs_break is a placeholder.")


def plot_hit_rate(*, df: pd.DataFrame, outfile: Path) -> None:
    """Placeholder for hit-rate scalar/curve plotting."""
    raise NotImplementedError("plot_hit_rate is a placeholder.")


def plot_pi_distribution(*, df: pd.DataFrame, outfile: Path) -> None:
    """Placeholder for PI distribution plot (box/violin/hist)."""
    raise NotImplementedError("plot_pi_distribution is a placeholder.")


def run_rolling_wll(
    start_idx: int,
    end_idx: Optional[int] = None,
    step: int = 1,
    baseline_mode: str = "auto_best_ml",  # or "fixed_rf"
    output_dir: Optional[Path] = None,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    target_col: str = DEFAULT_TARGET_COL,
    data_path: str = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """
    Rolling break-aware WLL vs ML experiment (skeleton, not implemented).

    Key behaviors to implement:
    - Loop origins from recent -> past via build_descending_origins.
    - For each origin, run Mohr with a candidate grid that excludes prior breaks;
      if exclusion empties the grid, halt to avoid trimming away the potential
      convergent break point.
    - Fit WLL via ConvexComboExperimentRunner (NP combo) with sample_start_index
      set from the detected break (1-based).
    - Fit RF/XGB baselines via ExperimentRunner; choose baseline per origin
      (auto-best ML or fixed RF).
    - Compute PI and hit flag; collect rows.
    - Persist results and produce plots with shaded hit regions.
    """
    # Run a pilot Mohr test on full dataset to get the initial break date 
    # as the only final candiate for logic A.




if __name__ == "__main__":
    break_candidate = []
    dataset = CocoaDataset(
        csv_path=PROCESSED_DATA_PATH,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL)
    
    last_date = dataset.get_last_date()
    break_detector =MohrRunner(last_date)

    break_candidate.append(break_detector.run_mohr_break_detection())

    print("Initial break candidate from full data Mohr test:", break_candidate)
