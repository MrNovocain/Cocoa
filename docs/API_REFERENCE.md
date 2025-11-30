# Cocoa API Reference

The Cocoa project exposes a small but flexible API surface for ingesting data,
building features, training structural-break-aware models, and running forecast
experiments. This document groups every public class, function, and component by
layer and includes sample usage snippets you can paste into a notebook or
script.

> **Tip:** All import paths in the examples assume you are working from the
> repository root with `PYTHONPATH` pointing to `src`.

---

## Core Configuration & Paths

### `cocoa.config.Settings` and `settings`

Typed dataclass that loads environment variables (via `.env`) to drive default
column names, filesystem locations, and experiment seeds. Use the module-level
`settings` instance whenever you need a consistent view of the runtime
configuration.

```python
from cocoa.config import Settings, settings

custom = Settings(environment="prod", random_seed=7)
assert settings.data_dir == "data"          # From .env or default
```

### `cocoa.paths.ProjectPaths` and constants

`ProjectPaths` captures canonical directories (`root`, `src`, `data_*`,
`models_dir`, etc.) and eagerly ensures they exist through
`ProjectPaths.ensure_directories()`. Import `PATHS`, `ROOT`, or `SRC` for quick
path building:

```python
from cocoa.paths import PATHS, ROOT

raw_csv = PATHS.data_raw / "Daily Prices_ICCO.csv"
print("Repository root:", ROOT)
```

### `cocoa.logging_config.setup_logging()`

Configures the root logger with the level coming from `settings.log_level`.
Call this once at process start if you rely on the built-in logging helpers.

```python
from cocoa.logging_config import setup_logging

setup_logging()
```

---

## Data Access & Feature Engineering

### `cocoa.data.ingest.load_cocoa_raw(filename="cocoa_raw.csv")`

Loads a raw CSV from `data/raw`, enforces datetime parsing for the configured
date column, and returns a time-sorted `DataFrame`. Raises `FileNotFoundError`
if the file is missing.

```python
from cocoa.data.ingest import load_cocoa_raw

raw = load_cocoa_raw("Ghana_data.csv")
```

### `cocoa.data.preprocess.preprocess_cocoa(df)`

Sorts by the configured date column, constructs a log-price target if needed,
and drops rows missing either the date or target. Returns a clean `DataFrame`.

```python
from cocoa.data.preprocess import preprocess_cocoa

clean = preprocess_cocoa(raw)
```

### `cocoa.data.features.build_features(raw_dir, processed_dir, reading_path, file_name)`

End-to-end feature builder that:

- reads Ghana weather and price panels,
- computes expanding-window climatology anomalies,
- merges with lagged log prices,
- derives forward-looking return targets,
- and persists the processed dataset.

```python
from pathlib import Path
from cocoa.data.features import build_features

processed = build_features(
    raw_data_dir=Path("data/raw"),
    processed_data_dir=Path("data/processed"),
    reading_path="Ghana_data_full.csv",
    file_name="cocoa_ghana_full.csv",
)
```

---

## General Utilities

### `cocoa.utils.seed.set_global_seed(seed: int | None)`

Synchronizes Python’s `random` module and NumPy with either the provided seed or
`RANDOM_SEED` from the environment. Returns the integer seed that was applied.

```python
from cocoa.utils.seed import set_global_seed

effective_seed = set_global_seed(123)
```

### `cocoa.utils.io.save_dataframe(df, filename, subdir="processed")`

Writes a `DataFrame` to `data/<subdir>/<filename>` (creating directories when
needed) and returns the final `Path`. Supported subdirectories: `processed`,
`interim`, `raw`.

```python
from cocoa.utils.io import save_dataframe

save_path = save_dataframe(df, "features.csv", subdir="interim")
```

### `cocoa.utils.plotting.plot_series(y, title="Cocoa price series")`

Quick Matplotlib helper for inspecting a single time series.

```python
from cocoa.utils.plotting import plot_series

plot_series(clean["log_price"])
```

### Bias–Variance Decomposition Helpers (`cocoa.utils.bias_var_decomposition`)

- `ModelInstantiator` (abstract) plus `DefaultModelInstantiator` and
  `ComboModelInstantiator` provide strategies for constructing unfitted models
  inside bootstrap loops.
- `BiasVarianceDecomposer` encapsulates the bootstrap workflow and exposes
  `.run(X_train, y_train, X_test, y_test) -> (mse, bias_sq, variance)`.
- `bias_variance_decomposition(...)` is a convenience wrapper that chooses the
  right instantiator (simple vs. convex combination) and returns the same tuple.

```python
from cocoa.utils.bias_var_decomposition import bias_variance_decomposition
from cocoa.models import RFModel

mse, bias_sq, var = bias_variance_decomposition(
    model_class=RFModel,
    hyperparams={"n_estimators": 500},
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_bootstrap_rounds=10,
)
```

---

## Modeling Building Blocks

### Core abstractions

- `cocoa.models.base_model.BaseModel`: abstract base with `fit`, `predict`,
  `is_fitted`, and `clone()`. Subclasses stash their constructor kwargs inside
  `self.hyperparams` so the infrastructure can recreate them.
- `cocoa.models.combo_base.BaseConvexCombinationModel`: handles splitting the
  training sample at `break_index`, fitting `model_pre`/`model_post`, and
  blending predictions via `gamma`.

### Data containers

- `cocoa.models.data_types.TrainTestSplit`: simple dataclass returned by every
  split routine (`X_train`, `y_train`, `X_test`, `y_test`).
- `cocoa.models.cocoa_data.CocoaDataset`: wraps the processed CSV and exposes:
  - attributes: `.df`, `.dates`, `.X`, `.y`
  - `trim_data_by_start_date(start_date)`
  - `get_window(start_date, end_date)`
  - `get_date_from_1_based_index(idx)`
  - `get_1_based_index_from_date(date)`
  - `split_oos_by_date(oos_start_date)` → `TrainTestSplit`.

Example:

```python
from cocoa.models.cocoa_data import CocoaDataset

dataset = CocoaDataset(
    csv_path="data/processed/cocoa_ghana_full.csv",
    feature_cols=["log_price_lagt"],
    target_col="log_return_forecast_target",
)
split = dataset.split_oos_by_date("2025-01-02")
```

### Experiment constants (`cocoa.models.assets`)

Convenience constants for reproducible experiments:

- `BREAK_ID_ONE_BASED`, `OOS_START_DATE`, `PROCESSED_DATA_PATH`
- Feature/target defaults (`DEFAULT_FEATURE_COLS`, `DEFAULT_TARGET_COL`,
  `XGB_FEATURE_COLS`, etc.)
- Hyperparameter grids (`RF_PARAM_GRID`, `XGB_PARAM_GRID`, `KRR_PARAM_GRID`)
- `Q_VALUE` for MFV folds and `BREAK_DATE_LB_DATE` for least-bias analyses.

Import the module and read the symbols directly:

```python
from cocoa.models import assets

print(assets.DEFAULT_FEATURE_COLS)
```

### Bandwidth utilities

`cocoa.models.bandwidth.create_precentered_grid(T, d, C=1.0, multipliers=None)`
constructs a log-spaced bandwidth grid around the rule-of-thumb value
`C * T^{-1/(d+4)}`. Designed for MFV tuning of NP models.

```python
from cocoa.models.bandwidth import create_precentered_grid

grid = create_precentered_grid(T=len(split.X_train), d=split.X_train.shape[1])
```

### Kernel & local engine interfaces

- `cocoa.models.np_base.BaseKernel` and `BaseLocalEngine` define the shape of
  kernel functions and local polynomial engines.
- Implementations:
  - `cocoa.models.np_kernels.GaussianKernel`
  - `cocoa.models.np_kernels.EpanechnikovKernel`
  - `cocoa.models.np_engines.LocalPolynomialEngine(order, use_gpu=True)`
    supports orders 0–2 (NW, local linear, local quadratic) and transparently
    uses CUDA when available.

```python
from cocoa.models import GaussianKernel, LocalPolynomialEngine

kernel = GaussianKernel()
engine = LocalPolynomialEngine(order=1)
```

### Non-parametric models

- `cocoa.models.np_regime.NPRegimeModel(kernel, local_engine, bandwidth)`
  stores its training data and uses the engine to produce predictions on demand.
- `cocoa.models.np_combo.NPConvexCombinationModel(...)` wraps two
  `NPRegimeModel` instances (pre/post break) and inherits the convex-combo
  behavior from `BaseConvexCombinationModel`.

```python
from cocoa.models import NPRegimeModel, NPConvexCombinationModel, GaussianKernel, LocalPolynomialEngine

engine = LocalPolynomialEngine(order=1)
kernel = GaussianKernel()
np_model = NPRegimeModel(kernel=kernel, local_engine=engine, bandwidth=0.5)
np_model.fit(split.X_train, split.y_train)
```

### Machine-learning models

- `cocoa.models.ml_models.BaseSklearnModel` is the shared adapter for sklearn
  estimators.
- Ready-made subclasses:
  - `RFModel` (RandomForestRegressor)
  - `XGBModel` (XGBoost regressor with squared-error objective)
  - `KRRModel` (Kernel Ridge Regression)
- `cocoa.models.ml_combo.MLConvexCombinationModel(model_class, params_pre,
  params_post, break_index, gamma)` blends any `BaseModel` subclass across a
  break date.

```python
from cocoa.models import RFModel

rf = RFModel(n_estimators=300, max_depth=5)
rf.fit(split.X_train, split.y_train)
preds = rf.predict(split.X_test)
```

### Evaluation utilities (`cocoa.models.evaluation`)

- `mean_squared_error(y_true, y_pred)`
- `mean_absolute_error(y_true, y_pred)`
- `evaluate_forecast(y_true, y_pred)` → `{"mse": ..., "mae": ...}`

Every function aligns the inputs by index before computing metrics.

### Plotting forecasts (`cocoa.models.plot.plot_forecast`) 

Reconstructs price levels from predicted returns (`log_return_hat`) and
generates full-history plus OOS-only PNGs.

```python
from cocoa.models.plot import plot_forecast

plot_forecast(
    df=dataset.df,
    target_col="log_return_forecast_target",
    y_pred=y_full_pred,
    oos_start_date="2025-01-02",
    model_label="NP_LL_Combo",
    output_path="output/np_combo/oos_forecast.png",
)
```

---

## Cross-Validation & Gamma Tuning

### `cocoa.models.mfv_CV.MFVValidator`

Implements Modified Forward-looking Validation (MFV). Key methods:

- `_set_block_size(T)` (called automatically by `grid_search`)
- `score(model_class, X_train, y_train, params)` → fold-average MSE
- `grid_search(model_class, X_train, y_train, param_grid)` → `(best_params,
  best_score, all_results)`

Usage:

```python
from cocoa.models import MFVValidator, RFModel
from cocoa.models.assets import RF_PARAM_GRID
from cocoa.experiments.runner import expand_grid

validator = MFVValidator(Q=4)
validator._set_block_size(len(split.X_train))
best_params, best_score, _ = validator.grid_search(
    RFModel, split.X_train, split.y_train, expand_grid(RF_PARAM_GRID)
)
```

### `MFVConvexComboValidator.tune_gamma(...)`

Specialized MFV routine for convex combinations. Accepts pre/post model classes
and parameter dictionaries plus the `break_index`, returning `(best_gamma,
best_score)`. Automatically restricts folds to post-break data.

---

## Experiment Framework

### `cocoa.experiments.runner.expand_grid(grid_dict)`

Utility that transforms a param grid dict into a list of dictionaries (Cartesian
product). Used by the runner and available for custom sweeps.

### `cocoa.experiments.runner.ExperimentRunner`

Single-model experiment orchestrator. Major constructor arguments:

- `model_name`, `model_class`
- `feature_cols`, `target_col`, `data_path`
- `oos_start_date`, optional `sample_start_index`
- `kernel_name`, `poly_order` (logging only)
- `save_results`, `run_bvd`, `n_bootstrap_rounds`

Public methods:

- `run() -> dict`: fits the model (with MFV hyperparameter tuning), optionally
  runs bias–variance decomposition, saves artifacts under
  `output/cocoa_forecast/<timestamp>_<model_name>`, and returns key metrics.
- `run_BVD_only() -> (mse, bias_sq, var, start_date)`
- `get_train_size() -> int`

Example:

```python
from cocoa.experiments.runner import ExperimentRunner
from cocoa.models import RFModel
from cocoa.models.assets import (
    PROCESSED_DATA_PATH, OOS_START_DATE, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL
)

runner = ExperimentRunner(
    model_name="RF_Full",
    model_class=RFModel,
    feature_cols=DEFAULT_FEATURE_COLS,
    target_col=DEFAULT_TARGET_COL,
    data_path=PROCESSED_DATA_PATH,
    oos_start_date=OOS_START_DATE,
    save_results=True,
    run_bvd=False,
)
runner.run()
```

### `cocoa.experiments.runner.ConvexComboExperimentRunner`

Tailored runner for convex combination models. Key parameters:

- `combo_type`: `"NP"` or `"ML"`
- `sample_start_index`: 1-based structural break position (required)
- `sub_model_class`/`sub_model_param_grid` (ML combos)
- `poly_order` (NP combos)

It performs three CV stages (pre-bandwidth, post-bandwidth, gamma) before
training the final combo model and exposes the same `run()` API as the base
runner. After `run()`, the chosen `gamma` is available via `runner.gamma`.

### CLI entry points

Several scripts simply instantiate the runners and call `.run()`; use them as
ready-made pipelines or templates:

- `python -m cocoa.models.fitting_rf` (Random Forest baseline)
- `python -m cocoa.models.fitting_xgb`
- `python -m cocoa.models.fitting_np_full`
- `python -m cocoa.experiments.run_ml_combo_cv`
- `python -m cocoa.experiments.run_np_combo_cv`
- `python -m cocoa.experiments.run_krr_combo_cv`

### `run_np_combo_cv_for_gamma_analysis(start_index, end_index, jump_size=100)`

Function inside `cocoa.experiments.run_np_combo_cv` that sweeps the break date
(`sample_start_index`) and records the MFV-chosen gamma plus CV scores across
runs. It also plots `gamma` and MFV MSE vs break date into `output/`.

```python
from cocoa.experiments.run_np_combo_cv import run_np_combo_cv_for_gamma_analysis

run_np_combo_cv_for_gamma_analysis(5000, 6200, jump_size=50)
```

### `cocoa.experiments.testing_regressors.get_feature_combinations()`

Enumerates every subset of `XGB_FEATURE_COLS` that retains
`"log_price_lagt"`. Used by the combinatorial NP experiments but callable for
custom sweeps.

```python
from cocoa.experiments.testing_regressors import get_feature_combinations

for combo in get_feature_combinations():
    print(combo)
```

---

## Structural Break Detection & Analysis

### `cocoa.experiments.break_detection.Kernel` Protocol

Typing protocol describing callables that accept an array of scaled distances
and return weights. Useful when authoring alternative kernels for the break
estimator.

### `estimate_break_mohr_ll(y, X, m_hat, center_X=True, standardize_X=True, trim_frac=0.05)`

Implements the Mohr–Selk (2020) single-break estimator using pilot residuals
`U_hat = y - m_hat`. Returns the 1-based break index. Handles trimming, feature
centering/standardization, and the Kolmogorov–Smirnov maximization routine.

```python
from cocoa.experiments.break_detection import estimate_break_mohr_ll
import numpy as np

T = 200
X = np.random.randn(T, 1)
y = np.sin(np.linspace(0, 3, T)) + np.random.randn(T) * 0.1
pilot = np.mean(y) * np.ones_like(y)
break_idx = estimate_break_mohr_ll(y, X, m_hat=pilot)
```

### `mfv_mse(y_true, y_pred)`

Convenience function for computing mean squared error in the break-detection
workflow.

---

## Gamma Tuning & Simulation Utilities

### `cocoa.experiments.gamma_tuning_simulation.GammaTuningResult`

Dataclass storing `tuned_gamma`, `oracle_gamma`, the `gamma_grid`, and MFV
losses from a simulation.

### `GammaMFVSimulator`

Recreates the MFV gamma-selection logic on synthetic linear DGPs with
structural breaks. Primary methods:

- `run_simulation(T=400, tau=200, phi=0.5, ..., random_seed=None)`
- `run_many(n_sim, random_seed=None, **dgp_kwargs)`

Each call returns a `GammaTuningResult`.

```python
from cocoa.experiments.gamma_tuning_simulation import GammaMFVSimulator

sim = GammaMFVSimulator(Q=4)
result = sim.run_simulation(T=300, tau=150, random_seed=42)
print(result.tuned_gamma, result.oracle_gamma)
```

### `cocoa.experiments.tau_slice_mfv.TauSliceResult`

Captures the losses from sweeping gamma at a fixed break date: `tau`, the grid,
losses, best gamma/loss, the bandwidths that were assumed, and the MFV block
size.

### `run_mfv_gamma_slice(tau, gamma_grid=None, pre_bandwidth, post_bandwidth, ...)`

Performs the slice test without re-fitting the underlying NP models for every
gamma. Ideal for scenario analyses when the optimal bandwidths are known.

```python
from cocoa.experiments.tau_slice_mfv import run_mfv_gamma_slice

slice_result = run_mfv_gamma_slice(
    tau="2019-07-15",
    pre_bandwidth=0.8,
    post_bandwidth=1.2,
)
```

---

## Forecast Comparison & Diagnostics

### `cocoa.experiments.MDM.MDM`

Implements the Modified Diebold–Mariano forecast comparison test following CGS
(2020). Instantiate with aligned numpy arrays of true values (`y_true`) and the
two competing forecast sequences (`f_i` benchmark, `f_j` candidate) plus the
forecast horizon `h`.

- `compute()` → dict containing the statistic, p-value, MSFEs, etc.
- `summary()` pretty-prints the result with significance stars.
- `get_significance_star(p_value)` helper for rendering.

```python
from cocoa.experiments.MDM import MDM

mdm = MDM(y_true=y_test, f_i=rf_preds, f_j=np_preds, h=1,
          model_i_name="RF", model_j_name="NP combo")
mdm.compute()
mdm.summary()
```

### `cocoa.experiments.test_mdm.MDMTestRunner`

High-level orchestrator that accepts two fully-configured `ExperimentRunner`
instances, re-runs them if necessary, loads their stored predictions, executes
the MDM test, and saves a timestamped record under `output/MDM_test/`.

Public workflow:

1. `run_experiments()`
2. `load_forecasts()`
3. `perform_test()`
4. `run()` combines all steps and captures logs automatically.

```python
from cocoa.experiments.runner import ExperimentRunner
from cocoa.experiments.test_mdm import MDMTestRunner
from cocoa.models import RFModel

runner_rf = ExperimentRunner(...)
runner_np = ExperimentRunner(...)

mdm_runner = MDMTestRunner(runner_benchmark=runner_rf, runner_candidate=runner_np, h=1)
mdm_runner.run()
```

---

## Visualization & Reporting Outputs

Many experiment scripts (for example `fitting_bvd.py`, `run_np_combo_cv.py`,
and `tau_slice_mfv.py`) emit diagnostic PNGs under `output/`, such as:

- Bias–variance curves vs. training start (`bvd_vs_time_*.png`)
- Gamma/MSE vs. structural break start (`gamma_vs_break_date_*.png`,
  `mse_vs_break_date_*.png`)
- Forecast overlays (`cocoa_forecast/.../oos_forecast.png`)

Use these artifacts alongside the documented APIs to interpret experiment
results quickly.

---

## Putting Everything Together

Below is a minimal end-to-end snippet that combines the major APIs:

```python
from cocoa.utils.seed import set_global_seed
from cocoa.data.features import build_features
from cocoa.models import RFModel
from cocoa.models.assets import (
    PROCESSED_DATA_PATH, OOS_START_DATE, DEFAULT_FEATURE_COLS, DEFAULT_TARGET_COL
)
from cocoa.experiments.runner import ExperimentRunner

set_global_seed()
build_features(Path("data/raw"), Path("data/processed"))

runner = ExperimentRunner(
    model_name="RF_Full",
    model_class=RFModel,
    feature_cols=DEFAULT_FEATURE_COLS,
    target_col=DEFAULT_TARGET_COL,
    data_path=PROCESSED_DATA_PATH,
    oos_start_date=OOS_START_DATE,
    save_results=True,
    run_bvd=True,
    n_bootstrap_rounds=5,
)
results = runner.run()
print(results["oos_mse"])
```

Use this reference as a checklist whenever you add new functionality—if a class
or function is part of the public surface, document it here with a short
description and runnable example.

