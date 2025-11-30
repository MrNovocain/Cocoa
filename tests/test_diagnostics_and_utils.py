import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.experiments.MDM import MDM
from cocoa.experiments.test_mdm import MDMTestRunner
from cocoa.experiments.tau_slice_mfv import run_mfv_gamma_slice
from cocoa.experiments.gamma_tuning_simulation import GammaMFVSimulator
from cocoa.experiments.runner import ExperimentRunner
from cocoa.models.plot import plot_forecast
from cocoa.utils.plotting import plot_series
from cocoa.utils.seed import set_global_seed


class StubExperimentRunner(ExperimentRunner):
    """Minimal ExperimentRunner subclass for testing MDMTestRunner."""

    class DummyModel:
        __name__ = "DummyModel"

    def __init__(self, name: str, predictions: np.ndarray, y_true: np.ndarray, tmp_path: Path):
        self.model_name = name
        self.model_class = StubExperimentRunner.DummyModel
        self.feature_cols = ["x"]
        self.target_col = "y"
        self.data_path = "dummy.csv"
        self.oos_start_date = "2025-01-01"
        self.output_dir = tmp_path / name
        self.output_dir.mkdir()
        self.split = type("Split", (), {"y_test": pd.Series(y_true)})()
        self._predictions = predictions

    def run(self):
        df = pd.DataFrame({"y_pred": self._predictions})
        df.to_csv(self.output_dir / "predictions.csv", index=False)
        return {}


def test_plot_forecast_creates_full_and_oos_files(tmp_path):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    log_price = np.linspace(4, 5, len(dates))
    log_price_lagt = pd.Series(log_price).shift(1).bfill().to_numpy()
    df = pd.DataFrame(
        {
            "date": dates,
            "log_price": log_price,
            "log_price_lagt": log_price_lagt,
            "log_return": np.linspace(0.01, 0.05, len(dates)),
        }
    )
    y_pred = np.linspace(0.01, 0.05, len(dates))
    output_path = tmp_path / "plots" / "oos_forecast.png"
    plot_forecast(df, "log_return", y_pred, oos_start_date=dates[-3], output_path=str(output_path))
    assert output_path.exists()
    full_path = output_path.parent / "full_forecast.png"
    assert full_path.exists()


def test_plot_series_runs_without_errors():
    series = pd.Series(np.arange(5), index=pd.date_range("2020-01-01", periods=5, freq="D"))
    plot_series(series, title="Test Series")
    plt.close("all")


def test_mdm_computation_detects_better_model():
    y_true = np.array([1.0, 1.2, 0.9, 1.1])
    f_benchmark = np.array([0.5, 0.5, 0.5, 0.5])
    f_candidate = np.array([1.0, 1.2, 0.9, 1.1])
    mdm = MDM(y_true, f_benchmark, f_candidate, h=1, model_i_name="bench", model_j_name="cand")
    results = mdm.compute()
    assert results["msfe_j"] < results["msfe_i"]


def test_mdm_test_runner_smoke(monkeypatch, tmp_path):
    y_true = np.array([1.0, 1.1, 1.2])
    preds_b = np.array([0.8, 0.9, 1.0])
    preds_c = np.array([1.0, 1.1, 1.2])

    runner_b = StubExperimentRunner("bench", preds_b, y_true, tmp_path)
    runner_c = StubExperimentRunner("cand", preds_c, y_true, tmp_path)

    monkeypatch.setattr(MDMTestRunner, "_save_record", lambda self, error: None, raising=False)

    test_runner = MDMTestRunner(runner_benchmark=runner_b, runner_candidate=runner_c, h=1)
    runner_b.run()
    runner_c.run()
    test_runner.load_forecasts()
    mdm_obj = test_runner.perform_test()
    assert mdm_obj.results["msfe_j"] <= mdm_obj.results["msfe_i"]


def test_gamma_mfv_simulator_returns_losses():
    sim = GammaMFVSimulator(Q=3)
    result = sim.run_simulation(T=120, tau=60, random_seed=0)
    assert 0.0 <= result.tuned_gamma <= 1.0
    assert len(result.losses) == len(result.gamma_grid)


def test_run_mfv_gamma_slice_returns_valid_gamma(tmp_path):
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    x = np.linspace(0, 1, len(dates))
    y = 2 * x + np.random.default_rng(0).normal(scale=0.05, size=len(dates))
    df = pd.DataFrame({"date": dates, "x": x, "y": y})
    csv_path = tmp_path / "slice.csv"
    df.to_csv(csv_path, index=False)

    tau = dates[30]
    result = run_mfv_gamma_slice(
        tau=tau,
        gamma_grid=np.linspace(0.0, 1.0, 5),
        pre_bandwidth=0.5,
        post_bandwidth=0.5,
        feature_cols=["x"],
        target_col="y",
        data_path=str(csv_path),
        oos_start_date=dates[-10],
    )
    assert 0.0 <= result.best_gamma <= 1.0
    assert len(result.gamma_losses) == len(result.gamma_grid)


def test_set_global_seed_synchronizes_rng():
    seed = set_global_seed(123)
    from random import random as py_random

    py_val = py_random()
    np_val = np.random.rand()
    assert seed == 123
    # Subsequent calls after resetting should repeat
    seed = set_global_seed(123)
    from random import random as py_random2

    assert py_val == py_random2()
    assert np.isclose(np_val, np.random.rand())

