import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.experiments import runner as runner_module
from cocoa.models import RFModel


def _write_tiny_dataset(tmp_path: Path, n_samples: int = 80):
    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D")
    feature = np.linspace(0.0, 1.0, n_samples)
    target = 0.5 * feature + np.sin(feature * np.pi) * 0.1
    df = pd.DataFrame({"date": dates, "feature": feature, "target": target})
    csv_path = tmp_path / "tiny.csv"
    df.to_csv(csv_path, index=False)
    oos_start = dates[-20].strftime("%Y-%m-%d")
    return csv_path, oos_start


@pytest.fixture
def small_rf_grid(monkeypatch):
    grid = {
        "n_estimators": [20],
        "max_depth": [3],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"],
    }
    monkeypatch.setattr(runner_module, "RF_PARAM_GRID", grid)
    yield


def test_experiment_runner_runs_bvd(monkeypatch, tmp_path, small_rf_grid):
    data_path, oos_start = _write_tiny_dataset(tmp_path)

    runner = runner_module.ExperimentRunner(
        model_name="RF_Test_BVD",
        model_class=RFModel,
        feature_cols=["feature"],
        target_col="target",
        data_path=str(data_path),
        oos_start_date=oos_start,
        save_results=False,
        run_bvd=True,
        n_bootstrap_rounds=2,
    )

    results = runner.run()
    assert results["avg_mse"] is not None
    assert results["avg_bias_sq"] is not None
    assert results["avg_variance"] is not None


def test_experiment_runner_skips_bvd_when_disabled(tmp_path, small_rf_grid):
    data_path, oos_start = _write_tiny_dataset(tmp_path)

    runner = runner_module.ExperimentRunner(
        model_name="RF_Test_NoBVD",
        model_class=RFModel,
        feature_cols=["feature"],
        target_col="target",
        data_path=str(data_path),
        oos_start_date=oos_start,
        save_results=False,
        run_bvd=False,
    )

    results = runner.run()
    assert results["avg_mse"] is None
    assert results["avg_bias_sq"] is None
    assert results["avg_variance"] is None

