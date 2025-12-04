"""
Skeleton for the rolling WLL vs ML experiment.

This is a non-functional placeholder that mirrors the design in
docs/rolling_wll_experiment_plan.md. Implementations should wire in the
existing runners and Mohr detector without changing signatures.
"""

from __future__ import annotations
import matplotlib
matplotlib.use('Agg')


from dataclasses import dataclass
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
@dataclass
class RollingResultRow:
    origin_idx: int
    origin_date: pd.Timestamp
    detected_break_idx: int
    detected_break_date: pd.Timestamp
    baseline: str
    msfe_wll: float
    msfe_XGb: float
    msfe_RF: float
    pi: float
    hit: bool
    trimmed_window: Optional[Tuple[int, int]]
    trimmed_flag: bool



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
        self.break_pilot: Optional[int] = None
        self.trim_window: Optional[Tuple[int, int]] = None
        self.dataset = dataset
        self.step = step
        self.last_date = self.dataset.get_last_date()
        self.records: List[RollingResultRow] = []

    def run_single_trial(
        self,
        cur: int,
        baseline_mode: str = "auto_best_ml",
    ) -> RollingResultRow:
        """
        Run a single rolling trial at a given origin using the pilot break T_*.
        Mimics the notebook: WLL via NP combo, RF/XGB baselines, MSFE + PI.
        """
        trimmed_flag = False

        if self.break_pilot is None:
            raise ValueError("Pilot Mohr break not set. Call run_pilot_mohr first.")

        oos_start_date = self.dataset.dates.iloc[cur -1]
        cur_date = self.dataset.get_date_from_1_based_index(cur)
        if not self.last_date == cur_date:
            mohr_test = MohrRunner(cur_date.strftime("%Y-%m-%d"))
            break_index = mohr_test.run_mohr_break_detection()
            self.trim_window = mohr_test.get_trimmed_indexies()
        else:
            break_index = self.break_pilot
            # trim_window comes from the pilot Mohr

        if self.trim_window is not None and break_index is not None:
            trim_start, trim_end = self.trim_window
            # trim bounds are 0-based; break_index is 1-based
            trimmed_flag = trim_start <= (break_index - 1) <= trim_end


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
        msfe_base = best_ml_msfe

        pi = 1 - msfe_wll / msfe_base if msfe_base not in (None, 0) else None
        hit = bool(pi is not None and pi > 0)

        return RollingResultRow(
            origin_idx=cur,
            origin_date=oos_start_date,
            detected_break_idx=break_index,
            detected_break_date=self.dataset.dates.iloc[break_index - 1],
            baseline=best_type,
            msfe_wll=msfe_wll,
            msfe_XGb=msfe_xgb,
            msfe_RF=msfe_rf,
            pi=pi if pi is not None else float("nan"),
            hit=hit,
            trimmed_window=self.trim_window,
            trimmed_flag=trimmed_flag,
        )











    def run_pilot_mohr(self) -> int:
        """
        Run Mohr on full data once to define T_* and the right-tail trimming window.
        Store break candidate as 0-based index and keep trim window for stop checks.
        """
        last_date = self.last_date
        print(f"Last date in training set is {last_date}")
        mohr_runner = MohrRunner(
            oos_start_date= last_date,
            dataset=self.dataset,
        )
        break_index = mohr_runner.run_mohr_break_detection()
        self.break_candidates.append(break_index)
        trim_start, trim_end = mohr_runner.get_trimmed_indexies()
        self.trim_window = (trim_start, trim_end)
        self.break_pilot = break_index
        return break_index

    def run(self):
        """
        Rolling break-aware WLL vs ML experiment in a loop.

        Key behaviors to implement:
        - Loop origins from recent -> past via a descending range.
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
        last_index = self.dataset.get_1_based_index_from_date(self.last_date)
        if self.end_index == last_index:
            raise ValueError("There is no testing sample to run, set end_index earlier than the last observation")
        self.run_pilot_mohr()
        if self.step % (self.end_index - self.start_index) == 0:
            end_point_trial = False
        else:
            end_point_trial = True
        for i in range(self.end_index, self.start_index, -self.step):
            self.records.append(self.run_single_trial(i))
        if end_point_trial == True:
            self.records.append(self.run_single_trial(self.start_index))







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



if __name__ == "__main__":
    dataset = CocoaDataset()
    date = dataset.get_last_date()
    index = dataset.get_1_based_index_from_date(date)
    exp = RollingWLLExperiment(index-4,index-1,1)
    exp.run()
    print(len(exp.records))






























    # def build_candidate_grid_excluding(self, prior_breaks: set[int], origin_idx: int) -> pd.DataFrame:
    #     """
    #     Placeholder for building Mohr candidate grids per origin, excluding prior breaks.
    #     """
    #     raise NotImplementedError("build_candidate_grid_excluding is a placeholder.")
