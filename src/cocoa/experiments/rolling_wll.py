"""
Skeleton for the rolling WLL vs ML experiment.

This is a non-functional placeholder that mirrors the design in
docs/rolling_wll_experiment_plan.md. Implementations should wire in the
existing runners and Mohr detector without changing signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
)
from cocoa.experiments.runner import ExperimentRunner, ConvexComboExperimentRunner
from cocoa.models import CocoaDataset, RFModel, XGBModel
from cocoa.experiments.break_detection import MohrRunner
from cocoa.models.cocoa_data import BaseDataset
from math import inf
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


class RollingWLLExperiment:
    """
    Class-based skeleton to manage rolling WLL runs and break candidates.
    Pilot Mohr produces a single T_*; loop stops if trimming would exclude it.
    """

    def __init__(
        self,
        start_index:int,
        end_index:int ,
        step: int,
        feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
        target_col: str = DEFAULT_TARGET_COL,
        data_path: str = PROCESSED_DATA_PATH,
        dataset: BaseDataset = CocoaDataset(
            csv_path=PROCESSED_DATA_PATH,
            feature_cols= DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
        ),
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.data_path = data_path
        self.start_index = start_index
        self.end_index = end_index
        self.break_candidates: List[int] = []
        self.trim_window: Optional[Tuple[int, int]] = None
        self.dataset = dataset
        self.bestML = (None, inf)
        self.step = step

    def run_single_trial(
        self,
        cur: int,
        baseline_mode: str = "auto_best_ml",
    ) -> RollingResultRow:
        """
        Run a single rolling trial at a given origin using the pilot break T_*.
        Mimics the notebook: WLL via NP combo, RF/XGB baselines, MSFE + PI.
        """
        if not self.break_candidates:
            raise ValueError("Pilot Mohr break not set. Call run_pilot_mohr first.")
        break_index = self.break_candidates[0]

        # If the pilot break would be trimmed by the right-tail window, signal stop.
        if self.trim_window is not None:
            trim_start, trim_end = self.trim_window
            if self.is_candidate_trimmed(trim_start, trim_end, self.break_candidates):
                raise RuntimeError("Pilot break would be trimmed; stop rolling loop.")

        oos_start_date = self.dataset.dates.iloc[cur]


        # WLL (NP combo) using ConvexComboExperimentRunner
        wll_runner = ConvexComboExperimentRunner(
            combo_type='NP',
            model_name="NP_LL_Combo",
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=oos_start_date,
            break_index=break_index,  # Structural break, required for Combo model
            poly_order=1,
            save_results=True,  # Must be True to get OOS MSE
        )
            



        wll_out = wll_runner.run()
        msfe_wll = wll_out.get("oos_mse")

        # RF baseline
        rf_runner = ExperimentRunner(
            model_name="RF",
            model_class=RFModel,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            data_path=self.data_path,
            oos_start_date=oos_start_date,
            save_results=False,
        )
        rf_out = rf_runner.run()
        msfe_rf = rf_out.get("oos_mse")

        # XGB baseline
        xgb_runner = ExperimentRunner(
            model_name="XGB",
            model_class=XGBModel,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            data_path=self.data_path,
            oos_start_date=oos_start_date,
            save_results=False,
        )
        xgb_out = xgb_runner.run()
        msfe_xgb = xgb_out.get("oos_mse")

        best_ml_msfe ,best_type = min(msfe_rf, msfe_xgb), "RF" if msfe_rf < msfe_xgb else "XGB"

        if best_ml_msfe < self.bestML[1]:
            self.bestML = (f"{best_type}with break index {break_index}", best_ml_msfe)
        msfe_base = self.bestML[1]


        pi = 1 - msfe_wll / msfe_base if msfe_base not in (None, 0) else None
        hit = bool(pi is not None and pi > 0)

        return RollingResultRow(
            origin_idx=cur,
            origin_date=oos_start_date,
            detected_break_idx=break_index,
            detected_break_date=self.dataset.dates.iloc[break_index],
            baseline=self.bestML[0],
            msfe_wll=msfe_wll,
            msfe_base=msfe_base,
            pi=pi if pi is not None else float("nan"),
            hit=hit,
        )











    def run_pilot_mohr(self) -> int:
        """
        Run Mohr on full data once to define T_* and the right-tail trimming window.
        Store break candidate as 0-based index and keep trim window for stop checks.
        """
        last_date = self.dataset.get_last_date()
        print(f"Last date in training set is {last_date}")
        mohr_runner = MohrRunner(
            oos_start_date= last_date,
            dataset=self.dataset,
        )
        break_index = mohr_runner.run_mohr_break_detection()
        self.break_candidates.append(break_index)
        trim_start, trim_end = mohr_runner.get_trimmed_indexies()
        self.trim_window = (trim_start, trim_end)
        return break_index

    @staticmethod
    def is_candidate_trimmed(trim_start: int, trim_end: int, candidates: List[int]) -> bool:
        """Return True if any candidate lies within the trimming window [trim_start, trim_end]."""
        return any(trim_start <= c <= trim_end for c in candidates)





    def plot_break_vs_origin(self, reverse_time: bool = True) -> None:
        """Placeholder for break path plotting with optional reverse-time axis and variance band."""
        raise NotImplementedError("plot_break_vs_origin is a placeholder.")


    def plot_pi_vs_break(self) -> None:
        """Placeholder for PI vs break plot with shading where PI > 0."""
        raise NotImplementedError("plot_pi_vs_break is a placeholder.")


    def plot_hit_rate(self) -> None:
        """Placeholder for hit-rate scalar/curve plotting."""
        raise NotImplementedError("plot_hit_rate is a placeholder.")


    def plot_pi_distribution(self) -> None:
        """Placeholder for PI distribution plot (box/violin/hist)."""
        raise NotImplementedError("plot_pi_distribution is a placeholder.")


    def run_rolling_wll(self) -> pd.DataFrame:
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
        raise NotImplementedError("run_rolling_wll is a placeholder.")


if __name__ == "__main__":
    dataset = CocoaDataset()
    index = dataset.get_1_based_index_from_date("2025-01-02")

    exp = RollingWLLExperiment(index,index,1)
    exp.run_pilot_mohr()
    print(exp.break_candidates)
    result = exp.run_single_trial(index)
    print(result)






























    # def build_candidate_grid_excluding(self, prior_breaks: set[int], origin_idx: int) -> pd.DataFrame:
    #     """
    #     Placeholder for building Mohr candidate grids per origin, excluding prior breaks.
    #     """
    #     raise NotImplementedError("build_candidate_grid_excluding is a placeholder.")