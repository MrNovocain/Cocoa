"""
Skeleton for the rolling WLL vs ML experiment.

This is a non-functional placeholder that mirrors the design in
docs/rolling_wll_experiment_plan.md. Implementations should wire in the
existing runners and Mohr detector without changing signatures.
"""


from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from cocoa.models.assets import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    PROCESSED_DATA_PATH,
)
from cocoa.experiments.runner import ExperimentRunner, ConvexComboExperimentRunner, MockExperimentRunner
from cocoa.models import CocoaDataset, RFModel, XGBModel
from cocoa.experiments.break_detection import MohrRunner
from cocoa.models.cocoa_data import BaseDataset

# Plotting configuration matching the notebook
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Color scheme
COLORS = {
    'pre_break': '#E74C3C',
    'post_break': '#3498DB',
    'wll': '#2ECC71',
    'rf': '#9B59B6',
    'xgb': '#F39C12',
    'rf_combo': '#1ABC9C',
    'xgb_combo': '#E67E22',
    'hit': '#2ECC71',
    'miss': '#E74C3C'
}
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
        self.dataset = dataset
        
        # Enforce reverse chronological order (recent -> past)
        if start_index > end_index:
            self.step = -abs(step)
        else:
            raise ValueError(f"Start index ({start_index}) must be greater than end index ({end_index}) for reverse rolling (recent to past).")
            
        self.step = self.step # Assign the (potentially modified) step
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


        if baseline_mode == "mock":
             # Use MockExperimentRunner for testing plotting logic
            wll_runner = MockExperimentRunner(model_name="NP_LL_Combo")
            wll_out = wll_runner.run()
            msfe_wll = wll_out.get("oos_mse")

            rf_runner = MockExperimentRunner(model_name="RF")
            rf_out = rf_runner.run()
            msfe_rf = rf_out.get("oos_mse")

            xgb_runner = MockExperimentRunner(model_name="XGB")
            xgb_out = xgb_runner.run()
            msfe_xgb = xgb_out.get("oos_mse")

        else:
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

    def run(self, n_jobs: int = -1):
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
        from joblib import Parallel, delayed

        last_index = self.dataset.get_1_based_index_from_date(self.last_date)
        # Check if we are trying to predict the future (impossible) or just OOS
        # This check might need adjustment based on exact definitions, but let's keep it simple
        
        self.run_pilot_mohr()
        
        # Iterate from start_index (recent) down to end_index (past)
        # self.step is negative, so we subtract 1 from end_index to include it
        indices = range(self.start_index, self.end_index - 1, self.step)
        
        def safe_run_trial(i):
            try:
                return self.run_single_trial(i)
            except Exception as e:
                print(f"Error in trial {i}: {e}")
                return None

        results = Parallel(n_jobs=n_jobs)(
            delayed(safe_run_trial)(i) for i in indices
        )
        
        # Filter out failed trials
        self.records = [r for r in results if r is not None]







    def _get_results_df(self) -> pd.DataFrame:
        """Convert records to a DataFrame."""
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self.records])

    def _save_plot(self, filename: str) -> None:
        """Save the current figure to the output directory."""
        output_dir = os.path.join("output", "rolling_experiments")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved plot to {path}")

    def plot_break_vs_origin(self, reverse_time: bool = True) -> None:
        """Plot detected break date vs origin date."""
        df = self._get_results_df()
        if df.empty:
            print("No records to plot.")
            return

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="origin_date", y="detected_break_date", marker="o", color=COLORS['post_break'])
        
        if reverse_time:
            plt.gca().invert_xaxis()
            
        plt.title("Detected Break Date vs Rolling Origin", fontsize=14, fontweight='bold')
        plt.xlabel("Rolling Origin Date", fontsize=12)
        plt.ylabel("Detected Break Date", fontsize=12)
        plt.grid(True, alpha=0.3)
        self._save_plot("break_vs_origin.png")

    def plot_pi_vs_origin(self, reverse_time: bool = True) -> None:
        """Plot PI vs rolling origin date."""
        df = self._get_results_df()
        if df.empty:
            print("No records to plot.")
            return

        plt.figure(figsize=(12, 6))
        
        # Plot PI points
        sns.scatterplot(
            data=df, 
            x="origin_date", 
            y="pi", 
            hue="hit", 
            palette={True: COLORS['hit'], False: COLORS['miss']},
            style="hit",
            s=100
        )
        
        # Add horizontal line at 0
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        
        if reverse_time:
            plt.gca().invert_xaxis()

        plt.title("Performance Improvement (PI) vs Rolling Origin", fontsize=14, fontweight='bold')
        plt.xlabel("Rolling Origin Date", fontsize=12)
        plt.ylabel("PI (1 - MSFE_WLL / MSFE_Base)", fontsize=12)
        plt.legend(title="WLL Outperforms", loc='upper right')
        plt.grid(True, alpha=0.3)
        self._save_plot("pi_vs_origin.png")

    def plot_msfe_comparison(self, reverse_time: bool = True) -> None:
        """Plot MSFE of WLL vs Best ML Baseline over time."""
        df = self._get_results_df()
        if df.empty:
            print("No records to plot.")
            return

        # Calculate msfe_base based on the chosen baseline
        df['msfe_base'] = df.apply(lambda row: row['msfe_RF'] if row['baseline'] == 'RF' else row['msfe_XGb'], axis=1)

        plt.figure(figsize=(12, 6))
        
        # Plot WLL MSFE
        sns.lineplot(data=df, x="origin_date", y="msfe_wll", marker="o", label="WLL (NP Combo)", color=COLORS['wll'])
        
        # Plot Baseline MSFE
        sns.lineplot(data=df, x="origin_date", y="msfe_base", marker="x", label="Best ML Baseline", color=COLORS['rf'])
        
        if reverse_time:
            plt.gca().invert_xaxis()
            
        plt.title("MSFE Comparison: WLL vs Baseline", fontsize=14, fontweight='bold')
        plt.xlabel("Rolling Origin Date", fontsize=12)
        plt.ylabel("MSFE", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_plot("msfe_comparison.png")

    def plot_hit_rate(self) -> None:
        """Plot cumulative hit rate over time."""
        df = self._get_results_df()
        if df.empty:
            print("No records to plot.")
            return

        # Sort by origin date to calculate cumulative hit rate correctly
        df_sorted = df.sort_values("origin_date")
        df_sorted["cumulative_hit_rate"] = df_sorted["hit"].expanding().mean()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_sorted, x="origin_date", y="cumulative_hit_rate", marker="o", color=COLORS['wll'], linewidth=2)
        
        overall_rate = df["hit"].mean()
        plt.axhline(overall_rate, color="black", linestyle="--", label=f"Overall: {overall_rate:.2%}")
        
        plt.title("Cumulative Hit Rate over Rolling Origins", fontsize=14, fontweight='bold')
        plt.xlabel("Rolling Origin Date", fontsize=12)
        plt.ylabel("Cumulative Hit Rate", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_plot("hit_rate.png")

    def plot_pi_distribution(self) -> None:
        """Plot PI distribution (box/violin/hist)."""
        df = self._get_results_df()
        if df.empty:
            print("No records to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot
        sns.boxplot(data=df, y="pi", ax=axes[0], color=COLORS['post_break'])
        axes[0].set_title("PI Distribution (Boxplot)", fontsize=12, fontweight='bold')
        axes[0].axhline(0, color="black", linestyle="--")
        axes[0].set_ylabel("PI", fontsize=10)
        
        # Histogram/KDE
        sns.histplot(data=df, x="pi", kde=True, ax=axes[1], color=COLORS['post_break'])
        axes[1].set_title("PI Distribution (Histogram)", fontsize=12, fontweight='bold')
        axes[1].axvline(0, color="black", linestyle="--")
        axes[1].set_xlabel("PI", fontsize=10)
        
        plt.tight_layout()
        self._save_plot("pi_distribution.png")



if __name__ == "__main__":
    print("Starting rolling_wll.py...")
    dataset = CocoaDataset()
    print("Dataset loaded.")
    date = dataset.get_last_date()
    index = dataset.get_1_based_index_from_date(date)
    # Start from recent (index-1) and roll back to past (index-20)
    exp = RollingWLLExperiment(index-1, index-20, 1)
    # Use mock mode for testing
    # exp.run_single_trial = lambda cur: RollingWLLExperiment.run_single_trial(exp, cur, baseline_mode="mock")
    exp.run()
    print(f"Ran {len(exp.records)} trials.")
    
    # Generate plots
    exp.plot_break_vs_origin()
    exp.plot_pi_vs_origin()
    exp.plot_msfe_comparison()
    exp.plot_hit_rate()
    exp.plot_pi_distribution()






























    # def build_candidate_grid_excluding(self, prior_breaks: set[int], origin_idx: int) -> pd.DataFrame:
    #     """
    #     Placeholder for building Mohr candidate grids per origin, excluding prior breaks.
    #     """
    #     raise NotImplementedError("build_candidate_grid_excluding is a placeholder.")
