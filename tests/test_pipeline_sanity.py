import os
import sys
import inspect
from functools import partial

import numpy as np
import pandas as pd

# Ensure the src package is importable when running tests from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cocoa.experiments.runner import ExperimentRunner
from cocoa.models import GaussianKernel, LocalPolynomialEngine, NPRegimeModel, MFVConvexComboValidator


class LinearOLSModel(NPRegimeModel):
    """A lightweight OLS-style model using the NPRegimeModel API."""

    def __init__(self):
        super().__init__(kernel=GaussianKernel(), local_engine=LocalPolynomialEngine(order=1), bandwidth=1.0)

    def fit(self, X, y, sample_weight=None):
        x_arr = np.asarray(X.iloc[:, 0], dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_design = np.column_stack([np.ones_like(x_arr), x_arr])
        coef, _, _, _ = np.linalg.lstsq(X_design, y_arr, rcond=None)
        self.coef_ = coef
        self._is_fitted = True
        return self

    def predict(self, X):
        if not getattr(self, "_is_fitted", False):
            raise ValueError("Model must be fitted before prediction.")
        x_arr = np.asarray(X.iloc[:, 0], dtype=float)
        return self.coef_[0] + self.coef_[1] * x_arr


def _make_signal_df(n_samples: int = 260, noise: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    feature = rng.normal(size=n_samples)
    seasonal = np.sin(np.linspace(0, 4 * np.pi, n_samples))
    target = 1.5 * feature + 0.3 * seasonal + rng.normal(scale=noise, size=n_samples)
    log_price = np.cumsum(target) + 10.0
    log_price_lagt = pd.Series(log_price).shift(1).bfill().to_numpy()
    return pd.DataFrame(
        {
            "date": dates,
            "feature": feature,
            "log_return": target,
            "log_price": log_price,
            "log_price_lagt": log_price_lagt,
        }
    )


def _run_full_np_pipeline_and_get_msfe(df: pd.DataFrame, tmp_path_factory) -> float:
    csv_path = tmp_path_factory.mktemp("cocoa_data") / "cocoa.csv"
    df.to_csv(csv_path, index=False)

    kernel = GaussianKernel()
    engine = LocalPolynomialEngine(order=1)
    np_model_partial = partial(NPRegimeModel, kernel=kernel, local_engine=engine)

    # Reserve the last 60 observations for OOS evaluation
    oos_start_date = df.sort_values("date")["date"].iloc[-60].strftime("%Y-%m-%d")

    runner = ExperimentRunner(
        model_name="NP_LL_Full_Test",
        model_class=np_model_partial,
        feature_cols=["feature"],
        target_col="log_return",
        data_path=str(csv_path),
        oos_start_date=oos_start_date,
        save_results=True,
        output_base_dir=str(tmp_path_factory.mktemp("np_outputs")),
    )

    results = runner.run()
    if results["oos_mse"] is None:
        raise AssertionError("Expected an OOS MSFE value from the pipeline run.")
    return float(results["oos_mse"])


def test_pipeline_detects_when_signal_is_destroyed(tmp_path_factory):
    df = _make_signal_df()
    df = df.sort_values("date")

    msfe_real = _run_full_np_pipeline_and_get_msfe(df, tmp_path_factory)

    rng = np.random.default_rng(123)
    df_perm = df.copy()
    df_perm["log_return"] = rng.permutation(df_perm["log_return"].values)

    msfe_perm = _run_full_np_pipeline_and_get_msfe(df_perm, tmp_path_factory)

    assert msfe_perm > msfe_real * 1.2


def test_gamma_grid_includes_pure_pre_and_post():
    signature = inspect.signature(MFVConvexComboValidator.tune_gamma)
    default_grid = signature.parameters["gamma_grid"].default

    assert np.isclose(default_grid[0], 0.0)
    assert np.isclose(default_grid[-1], 1.0)


def test_gamma_search_uses_boundary_candidates():
    rng = np.random.default_rng(321)
    T_pre, T_post = 90, 90

    x_pre = rng.normal(size=T_pre)
    x_post = rng.normal(size=T_post)

    y_pre = rng.normal(scale=0.5, size=T_pre)
    y_post = 2.5 * x_post + rng.normal(scale=0.1, size=T_post)

    X = pd.DataFrame({"feature": np.concatenate([x_pre, x_post])})
    y = pd.Series(np.concatenate([y_pre, y_post]))
    break_index = T_pre

    validator = MFVConvexComboValidator(Q=3)
    gamma_grid = np.array([0.0, 1.0])

    best_gamma, best_score = validator.tune_gamma(
        LinearOLSModel,
        {},
        LinearOLSModel,
        {},
        X,
        y,
        break_index,
        gamma_grid=gamma_grid,
        verbose=False,
    )

    assert best_gamma == 0.0
    assert np.isfinite(best_score)
