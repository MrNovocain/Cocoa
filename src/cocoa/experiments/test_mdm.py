import io
import os
import sys
import pandas as pd
from datetime import datetime
from functools import partial
from pathlib import Path
from contextlib import contextmanager

# Add src directory to Python path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from cocoa.experiments.runner import ExperimentRunner, ConvexComboExperimentRunner
from cocoa.models import RFModel, NPRegimeModel, GaussianKernel, LocalPolynomialEngine
from cocoa.models.assets import (
    PROCESSED_DATA_PATH,
    OOS_START_DATE,
    DEFAULT_FEATURE_COLS,
    DEFAULT_TARGET_COL,
    BREAK_ID_ONE_BASED,
    Q_VALUE
)
from cocoa.experiments.MDM import MDM


class MDMTestRunner:
    """
    A class to run a comparative test between two models using the MDM test.
    """

    def __init__(self, runner_benchmark: ExperimentRunner, runner_candidate: ExperimentRunner, h: int = 1):
        """
        Initializes the MDMTestRunner.

        Args:
            runner_benchmark (ExperimentRunner): The initialized runner for the benchmark model.
            runner_candidate (ExperimentRunner): The initialized runner for the candidate model.
            h (int): Forecast horizon.
        """
        if not isinstance(runner_benchmark, ExperimentRunner) or not isinstance(runner_candidate, ExperimentRunner):
            raise TypeError("runner_benchmark and runner_candidate must be instances of ExperimentRunner.")

        self.runner_benchmark = runner_benchmark
        
        self.runner_candidate = runner_candidate
        self.h = h
        self.y_true_test = None
        self.f_i_test = None
        self.f_j_test = None
        self.mdm_results = None
        self._stdout_buffer = None

    @contextmanager
    def _capture_stdout(self):
        """Context manager to capture stdout while still echoing to the console."""
        buffer = io.StringIO()
        original_stdout = sys.stdout

        class _Recorder:
            def __init__(self, buf, original):
                self.buffer = buf
                self.original = original

            def write(self, text):
                self.buffer.write(text)
                self.original.write(text)

            def flush(self):
                self.buffer.flush()
                self.original.flush()

        recorder = _Recorder(buffer, original_stdout)

        try:
            sys.stdout = recorder
            yield buffer
        finally:
            sys.stdout = original_stdout

    def run_experiments(self):
        """
        Runs the experiments for both models to generate forecasts if they haven't been run.
        """
        print(f"--- Running Experiment for Benchmark Model ({self.runner_benchmark.model_name}) ---")
        self.runner_benchmark.run()
        print(f"\n--- Running Experiment for Candidate Model ({self.runner_candidate.model_name}) ---")
        self.runner_candidate.run()

    def load_forecasts(self):
        """
        Loads the forecasts from the experiment output directories.
        """
        if not self.runner_benchmark.output_dir or not self.runner_candidate.output_dir:
            raise ValueError("Output directory not found for one or both runners. Ensure 'save_results=True'.")

        # y_true_test is the same for both runners
        self.y_true_test = self.runner_benchmark.split.y_test.values

        # Load benchmark model forecasts for the test set
        preds_i_df = pd.read_csv(os.path.join(self.runner_benchmark.output_dir, "predictions.csv"))
        self.f_i_test = preds_i_df['y_pred'].values[-len(self.y_true_test):]

        # Load candidate model forecasts for the test set
        preds_j_df = pd.read_csv(os.path.join(self.runner_candidate.output_dir, "predictions.csv"))
        self.f_j_test = preds_j_df['y_pred'].values[-len(self.y_true_test):]

        print("\n--- Forecasts Loaded for MDM Test ---")

    def perform_test(self):
        """
        Performs the MDM test and prints the summary.
        """
        mdm_test = MDM(
            y_true=self.y_true_test,
            f_i=self.f_i_test,
            f_j=self.f_j_test,
            h=self.h,
            model_i_name=self.runner_benchmark.model_name,
            model_j_name=self.runner_candidate.model_name
        )
        self.mdm_results = mdm_test.compute()
        mdm_test.summary()
        return mdm_test

    def _save_record(self, error: Exception | None = None):
        """Saves captured terminal output and statistics to an MDM record file."""
        if self._stdout_buffer is None:
            return

        base_dir = Path(__file__).resolve().parents[3] / "output" / "MDM_test"
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_part = f"{self.runner_benchmark.model_name}_vs_{self.runner_candidate.model_name}"
        sanitized_model_part = model_part.replace(" ", "_")
        filename = f"{timestamp}_{sanitized_model_part}.txt"
        filepath = base_dir / filename

        lines = [
            "MDM Test Record",
            f"Timestamp: {timestamp}",
            f"Benchmark Model: {self.runner_benchmark.model_name} ({self.runner_benchmark.model_class.__name__})",
            f"Candidate Model: {self.runner_candidate.model_name} ({self.runner_candidate.model_class.__name__})",
            f"Benchmark Output Directory: {self.runner_benchmark.output_dir}",
            f"Candidate Output Directory: {self.runner_candidate.output_dir}",
            f"Forecast Horizon: {self.h}",
        ]

        if self.mdm_results:
            lines.extend([
                "",
                "MDM Statistics:",
                f"- Observations (T): {self.mdm_results.get('df', -1) + 1}",
                f"- MSFE {self.runner_benchmark.model_name}: {self.mdm_results.get('msfe_i')}",
                f"- MSFE {self.runner_candidate.model_name}: {self.mdm_results.get('msfe_j')}",
                f"- Loss Differential (d_bar): {self.mdm_results.get('d_bar')}",
                f"- Omega Hat Squared: {self.mdm_results.get('omega_hat_sq')}",
                f"- MDM Statistic: {self.mdm_results.get('mdm_stat')}",
                f"- P-value (one-sided): {self.mdm_results.get('p_value')}",
            ])

        lines.extend([
            "",
            "Model Specifications:",
            f"- Benchmark Features: {self.runner_benchmark.feature_cols}",
            f"- Candidate Features: {self.runner_candidate.feature_cols}",
            f"- Target Column: {self.runner_benchmark.target_col}",
            f"- Data Path: {self.runner_benchmark.data_path}",
            f"- OOS Start Date: {self.runner_benchmark.oos_start_date}",
        ])

        if error is not None:
            lines.extend([
                "",
                "Error:",
                f"- {error.__class__.__name__}: {error}",
            ])

        lines.extend([
            "",
            "Terminal Output:",
            self._stdout_buffer.getvalue(),
        ])

        filepath.write_text("\n".join(lines), encoding="utf-8")
        print(f"MDM test record saved to {filepath}")

    def run(self):
        """
        Runs the full MDM test pipeline: experiments, forecast loading, and test execution.
        """
        error: Exception | None = None
        with self._capture_stdout() as buffer:
            self._stdout_buffer = buffer
            try:
                self.run_experiments()
                self.load_forecasts()
                self.perform_test()
            except Exception as exc:  # noqa: BLE001 - propagate while still saving logs
                error = exc
                raise
            finally:
                self._save_record(error)


if __name__ == '__main__':
    # --- 1. Configure the benchmark model runner (e.g., NP model) ---
    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=1)
    NPModelPartial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)

    runner_candidate_np_full = ExperimentRunner(
        model_name="np_full",
        model_class=NPModelPartial,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        save_results=True,
        run_bvd=False,
        kernel_name=kernel.__class__.__name__,
        poly_order=engine.order,
    )

    # --- 2. Configure the candidate model runner (e.g., RF model) ---
    runner_benchmark_rf_full = ExperimentRunner(
        model_name="ml_rf_full",
        model_class=RFModel,
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        save_results=True,
        run_bvd=False
    )
    runner_candidiate_rf_combo = ConvexComboExperimentRunner(
        combo_type='ML',
        model_name="RF_Combo",
        feature_cols=DEFAULT_FEATURE_COLS,
        target_col=DEFAULT_TARGET_COL,
        data_path=PROCESSED_DATA_PATH,
        oos_start_date=OOS_START_DATE,
        sample_start_index= BREAK_ID_ONE_BASED,  # Structural break, required for Combo
        save_results=True,  # Must be True to get OOS MSE
    )





    runner_benchmark_np_combo = ConvexComboExperimentRunner(
            combo_type='NP',
            model_name="NP_LL_Combo",
            feature_cols=DEFAULT_FEATURE_COLS,
            target_col=DEFAULT_TARGET_COL,
            data_path=PROCESSED_DATA_PATH,
            oos_start_date=OOS_START_DATE,
            sample_start_index= BREAK_ID_ONE_BASED,  # Structural break, required for Combo model
            poly_order=1,
            save_results=True,  # Must be True to get OOS MSE
        )
















    # --- 3. Initialize and run the MDM test ---
    mdm_runner = MDMTestRunner(runner_benchmark=runner_benchmark_rf_full, runner_candidate=runner_candidiate_rf_combo, h=1)
    mdm_runner.run()
    